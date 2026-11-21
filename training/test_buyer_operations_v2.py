"""CPU tests of paired decisions, action quotas, DDP sharding and weighted gradients."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from training.buyer_operations_v2_data import make_pair,FAMILIES,encode_record
from training.buyer_operations_sampling import paired_batches,PairedActionSampler,batch_manifest,QUOTAS
from training.buyer_diagnostics import evaluate


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[r for g in (20261119,20262128) for p in range(3) for f in FAMILIES
                     for r in make_pair(g,f,p,.4)]
        cls.metadata=[{k:r[k] for k in ('id','pair_id','group','task','family','positive','label_action','history_mode','persona_index')}
                      for r in cls.records]

    def test_pairs_engine_legality_surplus_and_context_identity(self):
        for i in range(0,len(self.records),2):
            a,b=self.records[i:i+2]
            self.assertTrue(a['positive']);self.assertFalse(b['positive'])
            self.assertEqual(a['pair_id'],b['pair_id'])
            self.assertEqual(a['messages'][:-1],b['messages'][:-1])
            self.assertGreater(a['realized_surplus_cents'],0)
            self.assertEqual(b['label_action'],'wait')
            negative=b['negative_trade_check']
            if negative['result']['ok']:self.assertLess(negative['realized_surplus_cents'],0)
            if b['family'].endswith('value'):self.assertTrue(negative['result']['ok'])
            for r in (a,b):
                fn=r['answer']['tool_calls'][0]['function']
                self.assertTrue(r['engine_result']['ok'])
                result=evaluate({**r,'name':r['id'],'state_mode':'operations'},fn['name'],fn['arguments'])
                self.assertEqual(result['errors'],[],r['id'])
                body=json.loads(r['messages'][-1]['content'])
                self.assertNotIn('positive',body);self.assertNotIn('label_action',body)
                self.assertNotIn('negative_trade_check',body)
                self.assertLess(len(r['situation']['your_id']),100)
                if r['history_mode']=='recorded':
                    actions=[m['tool_calls'][0]['function']['name'] for m in r['messages'] if m['role']=='assistant']
                    self.assertEqual(Counter(actions),Counter({'buy_listed':1,'accept_offer':1,'wait':1}))
                else:self.assertEqual(len(r['messages']),2)
            if a['family'].endswith('cash'):
                x=copy.deepcopy(a['observation']);x['you']['cash']=b['observation']['you']['cash']
                self.assertEqual(x,b['observation'])
        self.assertEqual({r['persona_index'] for r in self.records},{1,2,3})
        self.assertTrue({r['group'] for r in self.records}.isdisjoint({20261118+1009*i for i in range(80)}))
        for drop,expected in [(0,'recorded'),(1,'fresh')]:
            self.assertEqual(make_pair(20261119,'board_cash',0,drop)[0]['history_mode'],expected)
        last_actions={next(m for m in reversed(make_pair(20261119,'board_cash',p,0)[0]['messages'])
                           if m['role']=='assistant')['tool_calls'][0]['function']['name'] for p in range(3)}
        self.assertEqual(last_actions,{'buy_listed','accept_offer','wait'})
        # This group previously made price>value also exceed cash, hiding the valuation condition.
        for family in ('board_value','offer_value'):
            negative=make_pair(20268182,family,2,0)[1]
            self.assertTrue(negative['negative_trade_check']['result']['ok'])
            self.assertLess(negative['negative_trade_check']['realized_surplus_cents'],0)
        zero=copy.deepcopy(self.records[0]);zero['observation']['you']['cash']=0
        from training.buyer_operations_v2_data import choose
        tool,args=choose(zero['situation'],zero['observation'])
        self.assertEqual(tool,'wait')
        self.assertEqual(evaluate({**zero,'name':zero['id'],'state_mode':'operations'},tool,args)['errors'],[])

    def test_global_quota_pairs_reproducibility_and_two_rank_reconstruction(self):
        batches=paired_batches(self.metadata,16,7)
        self.assertEqual(batches,paired_batches(self.metadata,16,7))
        self.assertNotEqual(batches,paired_batches(self.metadata,16,8))
        total=Counter()
        for batch in batches:
            self.assertEqual(len(set(batch)),8)
            counts=Counter(self.metadata[i]['label_action'] for i in batch)
            self.assertEqual(counts,Counter(QUOTAS));total.update(counts)
            self.assertEqual(sorted(Counter(self.metadata[i]['persona_index'] for i in batch).values()),[2,3,3])
            pairs=Counter(self.metadata[i]['pair_id'] for i in batch)
            self.assertGreaterEqual(sum(n==2 for n in pairs.values()),2)
        self.assertEqual(total,Counter({'buy_listed':64,'accept_offer':32,'wait':32}))
        self.assertEqual(Counter(self.metadata[i]['persona_index'] for b in batches for i in b),Counter({1:43,2:43,3:42}))
        ranks=[list(PairedActionSampler(self.metadata,16,7,r,2)) for r in (0,1)]
        self.assertEqual([len(r) for r in ranks],[64,64])
        from torch.utils.data import DataLoader
        for rank in (0,1):
            loaded=list(DataLoader(self.metadata,batch_size=None,
                                   sampler=PairedActionSampler(self.metadata,16,7,rank,2),num_workers=0))
            self.assertEqual([r['id'] for r in loaded],[self.metadata[i]['id'] for i in ranks[rank]])
        for step,batch in enumerate(batches):
            reconstructed=[ranks[r][step*4+micro] for micro in range(4) for r in (0,1)]
            self.assertEqual(reconstructed,batch)
        manifest=batch_manifest(self.metadata,16,7,2)
        self.assertEqual(len(manifest['steps']),16)
        for rank in (0,1):
            planned=[r['id'] for step in manifest['steps'] for r in step['ranks'][rank]]
            self.assertEqual(planned,[self.metadata[i]['id'] for i in ranks[rank]])
        with self.assertRaisesRegex(ValueError,'Incomplete'):paired_batches(self.metadata[:-1],1,7)

    def test_official_tool_mask_and_paired_validation(self):
        from transformers import AutoTokenizer
        from training.buyer_operations_data import BASE,REVISION
        from training.buyer_operations_train import TokenRows,validation_indices
        tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
        # Entire fixture subset exercises all families/personas without duplicating token data in memory.
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_v2_tokens_cpu_') as temporary:
            path=Path(temporary)/'rows.jsonl'
            encoded=[encode_record(tokenizer,r) for r in self.records[:16]]
            from training.buyer_hf_check import target_encoding
            for r,new in zip(self.records[:2],encoded[:2]):
                fn=r['answer']['tool_calls'][0]['function']
                old=target_encoding(tokenizer,r['messages'],r['tools'],fn['name'],fn['arguments'])
                self.assertEqual(old['input_ids'],new['input_ids'])
                self.assertEqual([i for i,s in enumerate(old['regions']) if s=='tool_name'],new['tool_name_positions'])
            path.write_text(''.join(json.dumps(r)+'\n' for r in encoded))
            dataset=TokenRows(path,12288,require_tool_spans=True)
            indices=validation_indices(dataset,16,7)
            self.assertEqual(len(indices),16)
            self.assertEqual(sum(dataset.metadata[i]['positive'] for i in indices),8)
            for row in encoded:
                p=row['prompt_length'];mask=row['tool_name_positions']
                text=tokenizer.decode([row['input_ids'][p+i] for i in mask])
                self.assertIn(row['label_action'],text)
                self.assertNotIn('private_state',text)
                self.assertTrue(all(0<=i<row['target_length'] for i in mask))
            broken=copy.deepcopy(encoded[0]);broken['tool_name_positions']=[broken['target_length']]
            path.write_text(json.dumps(broken)+'\n')
            with self.assertRaisesRegex(ValueError,'Invalid tool-name'):TokenRows(path,12288,True)

    def test_weighted_completion_loss_and_gradient_match_manual_reference(self):
        import torch
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
        from training.buyer_operations_train import adapter_model,completion_losses
        torch.manual_seed(7)
        config=Qwen3_5TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,
            linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],tie_word_embeddings=True)
        base=Qwen3_5ForCausalLM(config)
        model=adapter_model(base,{'rank':2,'alpha':4,'dropout':0.,'target_modules':['q_proj']})
        model.eval();row={'input_ids':[4,5,6,7,8,9,10,11,12],'prompt_length':4,'tool_name_positions':[0,1]}
        parts=completion_losses(model,row,'cpu',1.)
        parts['total'].backward()
        actual={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        self.assertTrue(actual)
        self.assertTrue(all('lora_' in n for n in actual))
        model.zero_grad(set_to_none=True)
        ids=torch.tensor([row['input_ids']]);logits=model(input_ids=ids,use_cache=False).logits[:,3:-1]
        nll=torch.nn.functional.cross_entropy(logits.reshape(-1,128),ids[:,4:].reshape(-1),reduction='none')
        reference=nll.mean()+nll[:2].mean()
        self.assertTrue(torch.allclose(parts['total'],reference,atol=1e-6))
        reference.backward()
        for n,p in model.named_parameters():
            if n in actual:self.assertTrue(torch.allclose(actual[n],p.grad,atol=1e-6),n)
        unweighted=completion_losses(model,row,'cpu',0.)
        self.assertTrue(torch.allclose(unweighted['total'],unweighted['completion']))
        with self.assertRaisesRegex(ValueError,'requires tool-name'):
            completion_losses(model,{**row,'tool_name_positions':[]},'cpu',1.)

    def test_training_wrapper_plans_actual_two_worker_ddp_and_preserves_quota(self):
        from training.train_buyer_operations import main
        import yaml
        root=Path(__file__).resolve().parents[1]
        cfg=yaml.safe_load((root/'training/configs/buyer_operations_v2.yaml').read_text())
        class MetadataOnly:
            metadata=self.metadata
            def __len__(self):return len(self.metadata)
        data=MetadataOnly()
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_v2_wrapper_cpu_') as temporary:
            directory=Path(temporary);(directory/'manifest.json').write_text('{}')
            cfg['data_dir']=str(directory)
            path=directory/'config.yaml';path.write_text(yaml.safe_dump(cfg))
            (directory/'config.json').write_text('{}')
            (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'tiny.safetensors'}}))
            (directory/'tiny.safetensors').write_bytes(b'fixture')
            launched=[]
            def launch(command,**kwargs):
                launched.append((command,kwargs['env']))
                return SimpleNamespace(stdout=io.BytesIO(b'CPU fake worker; no model training\n'),wait=lambda:0)
            cards=[{'index':str(i),'fraction':.85} for i in range(2)]
            with patch('sys.argv',['train_buyer_operations.py','--config',str(path),'--gpus','0,1','--output-dir',temporary]), \
                 patch('training.buyer_operations_train.check_data',return_value=(data,data)), \
                 patch('huggingface_hub.try_to_load_from_cache',side_effect=lambda model,name,**kw:str(directory/name)), \
                 patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{'selection':'0,1'})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)), \
                 patch('training.train_buyer_operations.subprocess.Popen',side_effect=launch), redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(len(launched),1)
            command,env=launched[0]
            self.assertIn('--nproc_per_node=2',command)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
            self.assertEqual(env['HF_HUB_OFFLINE'],'1')
            session=next(directory.glob('*_buyer_operations_*'))
            saved=json.loads((session/'train_config.json').read_text())
            self.assertEqual(saved['world_size'],2)
            self.assertEqual(saved['decision_weight'],1.)
            self.assertIn('training/buyer_operations_sampling.py',saved['training_sources'])
            plan=json.loads((session/'batch_plan.json').read_text())
            self.assertEqual(plan['world_size'],2)
            self.assertTrue(all(s['actions']==QUOTAS for s in plan['steps']))


if __name__=='__main__':unittest.main()
