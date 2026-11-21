"""Official data provenance, group isolation, native targets and full-epoch recipe checks."""
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data/buyer_official_v1'


class OfficialBuyerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from training.train_official_buyer import prepare
        cls.cfg,cls.batches,cls.cases,cls.heldout,cls.train_cases,cls.manifest=prepare(yaml.safe_load((ROOT/'training/configs/buyer_official_v1.yaml').read_text()))
        cls.raw={s:list(map(json.loads,(DATA/f'{s}.jsonl').read_text().splitlines())) for s in ('train','validation','test')}

    def test_memory_preflight_activates_real_decoder_checkpointing_after_hf_eval(self):
        import torch
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
        from training.buyer_operations_train import adapter_model,completion_losses
        from training.official_buyer_train import preflight_training_mode
        torch.manual_seed(7)
        cfg=Qwen3_5TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=2,
            linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],tie_word_embeddings=True)
        base=Qwen3_5ForCausalLM(cfg).eval()  # from_pretrained's default mode
        model=adapter_model(base,{**self.cfg,'rank':2,'alpha':4})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        layers=list(base.model.layers)
        self.assertFalse(any(layer.training for layer in layers))
        row={'input_ids':[4,5,6,7,8,9,10,11],'prompt_length':5}
        with torch.no_grad():reference=completion_losses(model,row,'cpu')['completion']
        before=base.model.embed_tokens.weight.detach().clone()
        self.assertEqual(preflight_training_mode(model),2)
        with patch.object(layers[0],'_gradient_checkpointing_func',wraps=layers[0]._gradient_checkpointing_func) as first, \
             patch.object(layers[1],'_gradient_checkpointing_func',wraps=layers[1]._gradient_checkpointing_func) as second:
            loss=completion_losses(model,row,'cpu')['completion'];loss.backward()
            self.assertGreater(first.call_count,0);self.assertGreater(second.call_count,0)
        self.assertTrue(torch.allclose(loss.detach(),reference,atol=1e-6))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None),0)
        self.assertTrue(torch.equal(before,base.model.embed_tokens.weight))
        with self.assertRaisesRegex(RuntimeError,'active gradient checkpointing'):
            preflight_training_mode(torch.nn.Linear(2,2))

    def test_native_generation_stops_at_chat_eos_not_only_model_eos(self):
        import torch
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM,LogitsProcessor
        from training.official_buyer_train import generation_stop_ids
        cfg=Qwen3_5TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            layer_types=['full_attention'],eos_token_id=6,pad_token_id=0,tie_word_embeddings=True)
        model=Qwen3_5ForCausalLM(cfg).eval();tokenizer=SimpleNamespace(eos_token_id=7)
        self.assertEqual(generation_stop_ids(model,tokenizer),[6,7])
        class ForceChatEnd(LogitsProcessor):
            def __call__(self,input_ids,scores):
                scores.fill_(-float('inf'));scores[:,7]=0;return scores
        with torch.inference_mode():
            output=model.generate(torch.tensor([[4,5]]),attention_mask=torch.ones(1,2,dtype=torch.long),
                max_new_tokens=3,do_sample=False,logits_processor=[ForceChatEnd()],eos_token_id=generation_stop_ids(model,tokenizer))
        self.assertEqual(output.tolist(),[[4,5,7]])

    def test_recheck_uses_existing_weights_and_original_cases_without_gpu_launch(self):
        from training.recheck_official_buyer import prepare_recheck,main
        run=ROOT/'training/runs/20261119-213016_buyer_official_u3csnqyh'
        cfg,hashes=prepare_recheck(run,[84,112])
        self.assertTrue(cfg['evaluation_only']);self.assertEqual([p['step'] for p in cfg['evaluation_plan']],[0,84,112])
        self.assertNotIn('resume_checkpoint',cfg);self.assertEqual(len(hashes),3)
        with patch('sys.argv',['recheck',str(run),'--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.recheck_official_buyer.subprocess.Popen') as launch,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();launch.assert_not_called()

    def test_pinned_source_inventory_and_conservative_market_holdouts(self):
        from training.official_buyer_data import economic_group,read_official
        self.assertEqual(self.manifest['raw_roles'],{'buyer':1091,'seller':1208})
        self.assertEqual(self.manifest['source']['revision'],'fa0f94cff3eeed5bc54da597ad5c2a45ef491e45')
        groups={s:set(self.manifest['groups'][s]) for s in self.raw};trajectories={s:{r['trajectory_id'] for r in rows} for s,rows in self.raw.items()}
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            self.assertFalse(groups[a]&groups[b]);self.assertFalse(trajectories[a]&trajectories[b])
        sit=self.raw['train'][0]['situation'];other=copy.deepcopy(sit)
        other['your_utility_per_unit']={p:[1234] for p in sit['your_utility_per_unit']};other['your_id']='another_buyer';other['sellers'].reverse()
        self.assertEqual(economic_group(sit),economic_group(other))
        # The entire original dataset is readable after removing null union-schema padding.
        count=0
        for row in read_official(ROOT/'training/data/official_raw'):
            for tool in row['tools']:
                self.assertTrue(all(v is not None for v in tool['function']['parameters']['properties'].values()))
            count+=1
        self.assertEqual(count,2299)

    def test_curated_targets_preserve_official_answers_and_successful_source(self):
        from training.official_buyer_data import normalize,read_official,schema_error,target_error
        selected={r['id']:r for rows in self.raw.values() for r in rows};wanted=defaultdict(list)
        for r in selected.values():wanted[r['trajectory_id']].append(r)
        matched=0
        for episode in read_official(ROOT/'training/data/official_raw'):
            if episode['id'] not in wanted:continue
            for row in wanted[episode['id']]:
                answer=normalize(episode['messages'][row['source_message_index']]);self.assertEqual(row['answer'],answer)
                self.assertEqual(row['messages'][0],episode['messages'][0]);self.assertEqual(json.loads(row['messages'][-1]['content']),row['observation'])
                self.assertEqual(row['original_tool_result']['ok'],True);self.assertTrue(answer['reasoning_content'])
                fn=answer['tool_calls'][0]['function'];self.assertIsNone(schema_error(fn['name'],fn['arguments'],row['tools']))
                self.assertLessEqual(row['observation']['round'],5)
                if fn['name']=='buy_listed':
                    self.assertEqual(fn['arguments']['qty'],1);self.assertEqual(row['observation']['you']['units_bought'].get(fn['arguments']['product_id'],0),0)
                self.assertLessEqual(sum(m['role']=='user' for m in row['messages']),7)
                self.assertTrue(all('reasoning_content' not in m for m in row['messages'][1:]))
                matched+=1
        self.assertEqual(matched,len(selected))

    def test_native_token_prefix_masks_and_qwen_parser(self):
        from transformers import AutoTokenizer
        from training.buyer_hf_check import parse_raw
        t=AutoTokenizer.from_pretrained(self.cfg['base_model'],revision=self.cfg['revision'],local_files_only=True)
        encoded={r['id']:r for r in map(json.loads,(DATA/'train.tokens.jsonl').read_text().splitlines())}
        # One real example per action verifies optional private_state and all native tools.
        examples={}
        for row in self.raw['train']:examples.setdefault(row['label_action'],row)
        for action,row in examples.items():
            token=encoded[row['id']];prompt=t.apply_chat_template(row['messages'],tools=row['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            self.assertTrue(prompt.endswith('<think>\n'));self.assertEqual(t.encode(prompt,add_special_tokens=False),token['input_ids'][:token['prompt_length']])
            self.assertLessEqual(token['reasoning_tokens'],512);self.assertLessEqual(token['target_length'],1024)
            tail=t.decode(token['input_ids'][token['prompt_length']:],skip_special_tokens=False)
            self.assertIn('</think>',tail)
            parsed=parse_raw(t,tail,row['tools']);self.assertEqual(parsed['parsed_calls'],1);self.assertEqual(parsed['tool'],action)
            self.assertEqual(parsed['arguments'],row['answer']['tool_calls'][0]['function']['arguments'])

    def test_epoch_coverage_warmup_and_quarter_checkpoints(self):
        from training.official_buyer_train import batch_indices,catastrophic,probe_summary
        n=len(self.raw['train']);counts=Counter(i for b in self.batches for i in b)
        self.assertEqual(set(counts),set(range(n)));self.assertLessEqual(max(counts.values()),2)
        self.assertEqual(len(self.batches),math.ceil(n/8));self.assertEqual(self.cfg['warmup_steps'],math.ceil(self.cfg['max_steps']*.05))
        self.assertLess(self.cfg['warmup_steps'],self.cfg['max_steps']);self.assertEqual(self.cfg['checkpoint_steps'][-1],self.cfg['max_steps'])
        self.assertEqual(self.cfg['padding_examples'],(-n)%8);self.assertEqual(self.batches,batch_indices(n,8,self.cfg['seed']))
        rank_rows=[[i for b in self.batches for i in b[r::2]] for r in (0,1)]
        self.assertEqual(len(rank_rows[0]),len(rank_rows[1]))
        self.assertNotIn('initial_adapter',self.cfg);self.assertEqual(self.cfg['learning_rate'],5e-5)
        self.assertTrue(catastrophic({'positive_captured':7,'valid_actions':16},{'positive_captured':3,'valid_actions':16}))
        self.assertFalse(catastrophic({'positive_captured':7,'valid_actions':16},{'positive_captured':6,'valid_actions':16}))

    def test_target_rules_cash_quantity_quality_and_wait(self):
        from training.official_buyer_data import target_error,low_multiplier
        obs={'round':1,'turns_left_this_round':6,'board':[{'seller':'s','product_id':'P','price':5,'in_stock':1}],'offers_to_you':[],
             'you':{'cash':5,'units_bought':{},'utility_of_next_unit':{'P':10}}}
        sit={'market_overview':[{'product_id':'P','quality_varies':False}],'low_grade_utility':'a low-grade unit is worth 0.4 x the utility above'}
        args={'seller_id':'s','product_id':'P','qty':1};success={'ok':True,'data':{'trade_id':'t'}}
        self.assertEqual(low_multiplier(sit['low_grade_utility']),.4)
        self.assertIsNone(target_error('buy_listed',args,obs,sit,success,None))
        poor=copy.deepcopy(obs);poor['you']['cash']=4.99;self.assertEqual(target_error('buy_listed',args,poor,sit,success,None),'unaffordable_trade')
        self.assertEqual(target_error('wait',{},obs,sit,{'ok':True},None),'wait_with_affordable_fixed_positive')
        self.assertEqual(target_error('buy_listed',{**args,'qty':2},obs,sit,success,None),'multiunit_trade')
        after={'your_unreviewed_trades':[{'trade_id':'t','grade':'low'}]};self.assertEqual(target_error('buy_listed',args,obs,sit,success,after),'observed_low_nonpositive_trade')

    def test_official_native_checks_allow_alternative_profitable_actions(self):
        from training.official_buyer_train import official_snapshot_review
        self.assertTrue(self.heldout)
        first_ids={self.raw['train'][i]['id'] for b in self.batches[:self.cfg['checkpoint_steps'][0]] for i in b}
        self.assertEqual(len(self.train_cases),6)
        ranks=defaultdict(set)
        for c in self.cases:ranks[c['origin']['pair_id']].add(c['evaluation_rank'])
        self.assertTrue(all(len(r)==1 for r in ranks.values()))
        self.assertEqual(Counter(c['evaluation_rank'] for c in self.cases),{0:8,1:8})
        for c in self.train_cases:self.assertIn(c['source_sample_id'],first_ids)
        val_ids={r['id'] for r in self.raw['validation']};train_ids={r['id'] for r in self.raw['train']}
        for c in self.heldout:
            source=next(r for r in self.raw['validation'] if r['id']==c['source_sample_id']);fn=source['answer']['tool_calls'][0]['function']
            self.assertTrue(official_snapshot_review(c,fn['name'],fn['arguments'])['valid_snapshot'])
            self.assertIn(c['name'].removeprefix('official_'),val_ids);self.assertNotIn(c['name'].removeprefix('official_'),train_ids)
        c={'tools':self.raw['train'][0]['tools'],'situation':{'sellers':['s'],'buyers':['b']},'expected_tool':'wait',
            'observation':{'round':1,'board':[{'seller':'s','product_id':'P','price':5,'in_stock':1}],'offers_to_you':[],
                'you':{'cash':10,'units_bought':{},'utility_of_next_unit':{'P':12}}}}
        r=official_snapshot_review(c,'buy_listed',{'seller_id':'s','product_id':'P','qty':1})
        self.assertTrue(r['valid_snapshot']);self.assertTrue(r['positive_trade'])

    def test_cpu_entry_and_mock_launcher_exact_resume_recipe_checks(self):
        from training.train_official_buyer import main,resume_info
        from training.decision_ablation import fingerprint
        with patch('sys.argv',['official','--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu,patch('training.train_official_buyer.subprocess.Popen') as launch,redirect_stdout(io.StringIO()):
            config=main();gpu.assert_not_called();launch.assert_not_called()
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_official_entry_') as tmp:
            def fake(cmd,**kwargs):
                folder=Path(cmd[-1]);self.assertIn('--nproc_per_node=2',cmd)
                (folder/'finished.json').write_text(json.dumps({'status':'CPU_mock','development_candidate':None}))
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock: no model execution\n'),wait=lambda:0)
            cards=[{'index':str(i),'fraction':.85} for i in (0,1)]
            with patch('sys.argv',['official','--output-dir',tmp,'--gpus','0,1']),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})),patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.train_official_buyer.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):folder=main()
            cfg=json.loads((folder/'train_config.json').read_text());self.assertEqual(cfg['objective'],self.cfg['objective'])
            self.assertEqual(len(json.loads((folder/'native_cases.json').read_text())),16)
            self.assertEqual(len(json.loads((folder/'official_native_cases.json').read_text())),len(self.heldout))
            cp=folder/'checkpoints/step_0001';cp.mkdir(parents=True)
            for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(ROOT/'training/runs/20261119-134251_buyer_operations_50lhkufb/checkpoints/step_0032'/name,cp/name)
            for name in ('optimizer.pt','rng_rank0.pt','rng_rank1.pt'):(cp/name).write_bytes(b'CPU existence fixture, not optimizer tensors')
            (cp/'checkpoint_complete.json').write_text(json.dumps({'step':1,'baseline_behavior':{'positive_captured':7},'adapter_fingerprints':fingerprint(cp)}))
            got=resume_info(self.cfg,cp,self.batches,self.cases);self.assertEqual(got['start_step'],1)
            with self.assertRaisesRegex(ValueError,'recipe/data/source changed'):resume_info({**self.cfg,'learning_rate':1e-3},cp,self.batches,self.cases)


if __name__=='__main__':unittest.main()
