"""CPU-only native-template, coverage, exact-resume and ON inference/gate checks."""
from collections import Counter
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data/buyer_reasoning_v1'


class ReasoningDataTests(unittest.TestCase):
    def test_native_reasoning_precedes_decision_and_old_input_facts_are_identical(self):
        from transformers import AutoTokenizer
        from training.buyer_reasoning_data import BASE,REVISION,transform,encode_row,personas
        from training.buyer_operations_train import TokenRows,check_data
        import yaml
        tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
        original=[]
        with (ROOT/'training/data/buyer_operations_v2_1/validation.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if len(original)<32:original.append(row)
        for old in original:
            row=transform(old);encoded=encode_row(tokenizer,row)
            self.assertEqual(old['observation'],row['observation'])
            self.assertEqual(old['messages'][1:],row['messages'][1:])
            completion=tokenizer.decode(encoded['input_ids'][encoded['prompt_length']:])
            self.assertIn(row['answer']['reasoning_content'],completion)
            self.assertLess(completion.index('</think>'),completion.index('<function='))
            self.assertEqual(completion.count('<tool_call>'),1)
            name=tokenizer.decode([encoded['input_ids'][encoded['prompt_length']+i] for i in encoded['tool_name_positions']])
            self.assertEqual(name,row['label_action'])
            self.assertLess(max(encoded['reasoning_positions']),min(encoded['tool_name_positions']))
        self.assertTrue(all(len(p)<=8000 for p in personas()))
        cfg=yaml.safe_load((ROOT/'training/configs/buyer_reasoning_v1.yaml').read_text())
        train,val=check_data(cfg,ROOT)
        self.assertEqual((len(train),len(val)),(3072,384))
        with self.assertRaisesRegex(ValueError,'thinking mode mismatch'):
            check_data({**cfg,'thinking':False},ROOT)

    def test_epoch_coverage_and_pilot_prefix_match_two_rank_continuation(self):
        from training.buyer_operations_train import TokenRows
        from training.buyer_reasoning_sampling import paired_batches,PairedActionSampler
        rows=TokenRows(DATA/'train.tokens.jsonl',12288,True)
        all_batches=paired_batches(rows.metadata,768,20261119)
        self.assertEqual(all_batches[:32],paired_batches(rows.metadata,32,20261119))
        for epoch in range(2):
            counts=Counter(i for b in all_batches[epoch*384:(epoch+1)*384] for i in b)
            self.assertEqual(counts,Counter(range(3072)))
        self.assertNotEqual(all_batches[:384],all_batches[384:])
        rank_rows=[]
        for rank in (0,1):
            sampler=PairedActionSampler(rows.metadata,384,20261119,rank,2);sampler.start_step=32
            rank_rows.append(list(sampler))
        actual=[rank_rows[r][micro] for start in range(0,len(rank_rows[0]),4)
                for micro in range(start,start+4) for r in (0,1)]
        self.assertEqual(actual,[i for b in all_batches[32:384] for i in b])
        prefix=[rows.metadata[i] for b in all_batches[:32] for i in b]
        self.assertEqual(Counter(r['label_action'] for r in prefix),{'buy_listed':64,'accept_offer':64,'wait':128})
        self.assertEqual(len({r['family'] for r in prefix}),8)
        self.assertEqual({r['cash_variant'] for r in prefix if r['family'].endswith('cash')},
                         {'equal','above_one','wide','zero'})

    def test_long_run_cannot_launch_without_passed_pilot(self):
        from training.train_buyer_operations import main
        with patch('sys.argv',['train_buyer_operations.py','--config',str(ROOT/'training/configs/buyer_reasoning_v1.yaml'),
                               '--epochs','1','--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_operations.subprocess.Popen') as launch,redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'requires a passed pilot gate'):
                main()
            gpu.assert_not_called();launch.assert_not_called()

    def test_user_wrapper_sets_on_and_evaluator_prepare_only_never_starts_gpu(self):
        # Import schema before mocking global subprocess.Popen: vLLM's first import
        # asks platform.uname() about CPU/OS through subprocess.check_output.
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from training.train_buyer_operations import main
        from training.evaluate_buyer_reasoning import main as evaluate
        import shutil
        cards=[{'index':str(i),'fraction':.8} for i in (0,1)]
        launched=[]
        def launch(command,**kw):
            launched.append(command)
            return SimpleNamespace(stdout=io.BytesIO(b'CPU fake worker\n'),wait=lambda:0)
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_reasoning_entry_cpu_') as temp:
            with patch('sys.argv',['train_buyer_operations.py','--config',str(ROOT/'training/configs/buyer_reasoning_v1.yaml'),
                                   '--gpus','0,1','--output-dir',temp]), \
                 patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)), \
                 patch('training.train_buyer_operations.subprocess.Popen',side_effect=launch),redirect_stdout(io.StringIO()):main()
            run=next(Path(temp).glob('*_buyer_operations_*'))
            cfg=json.loads((run/'train_config.json').read_text())
            self.assertTrue(cfg['thinking']);self.assertEqual(cfg['world_size'],2)
            self.assertEqual((run/'agent.yaml').read_text(),'reasoning: true\n')
            self.assertIn('--nproc_per_node=2',launched[0])
            cp=run/'checkpoints/step_0032';cp.mkdir(parents=True)
            # Tiny starter adapter is a schema fixture, not a trained reasoning checkpoint.
            for name in ('adapter_config.json','adapter_model.safetensors'):
                shutil.copy2(ROOT/'agents/example_student/buyer/checkpoint'/name,cp/name)
            (cp/'training_step.json').write_text(json.dumps({'step':32,'thinking':True}))
            output=run/'cpu_reports';output.mkdir()
            with patch('sys.argv',['evaluate_buyer_reasoning.py',str(run),'--prepare-only']), \
                 patch('training.local_resources.output_parent',return_value=(output,None)), \
                 patch('training.local_resources.query_gpus') as gpu,patch('training.buyer_diagnostics.subprocess.Popen') as processes, \
                 redirect_stdout(io.StringIO()):report=evaluate()
            gpu.assert_not_called();processes.assert_not_called()
            plan=json.loads((report/'plan.json').read_text())
            self.assertTrue(plan['thinking']);self.assertEqual(plan['calls'],768)
            self.assertEqual(plan['adapters']['base_on'],None)
            self.assertFalse((report/'evaluation_status.json').exists())


class ResumeTests(unittest.TestCase):
    def test_optimizer_scheduler_and_rng_resume_matches_uninterrupted_cpu_training(self):
        import torch
        from training.reasoning_checkpoint import save_state,load_optimizer,restore_rng
        torch.manual_seed(8);random.seed(9)
        model=torch.nn.Linear(4,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(8-s)/8)
        def update(m,o,s):
            o.zero_grad();x=torch.randn(2,4)*random.random()
            loss=m(x).square().mean();loss.backward();o.step();s.step()
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_reasoning_resume_cpu_') as temp:
            folder=Path(temp)
            for _ in range(3):update(model,optimizer,scheduler)
            torch.save(model.state_dict(),folder/'weights.pt')
            with patch('torch.cuda.get_rng_state',side_effect=lambda d:torch.get_rng_state()):
                save_state(folder,model,optimizer,scheduler,3,0,'cpu')
            for _ in range(5):update(model,optimizer,scheduler)
            expected={k:v.clone() for k,v in model.state_dict().items()}
            expected_lr=optimizer.param_groups[0]['lr']
            resumed=torch.nn.Linear(4,1);resumed.load_state_dict(torch.load(folder/'weights.pt',weights_only=True))
            opt=torch.optim.AdamW(resumed.parameters(),lr=.01)
            sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(8-s)/8)
            load_optimizer(opt,sch,folder,'cpu',3)
            with patch('torch.cuda.set_rng_state',side_effect=lambda value,d:torch.set_rng_state(value)):
                restore_rng(folder,0,'cpu',3)
            for _ in range(5):update(resumed,opt,sch)
            for k,v in resumed.state_dict().items():self.assertTrue(torch.equal(expected[k],v),k)
            self.assertEqual(expected_lr,opt.param_groups[0]['lr'])

    def test_resume_adapter_roundtrip_and_recipe_drift(self):
        import torch
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
        from training.buyer_operations_train import adapter_model
        from training.reasoning_checkpoint import load_weights,validate_resume
        config=Qwen3_5TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,layer_types=['full_attention'])
        model=adapter_model(Qwen3_5ForCausalLM(config),{'rank':2,'alpha':4,'dropout':0.,'target_modules':['q_proj']})
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_' in name:p.fill_(.12)
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_reasoning_adapter_cpu_') as temp:
            run=Path(temp);cp=run/'checkpoints/step_0003';cp.mkdir(parents=True)
            model.save_pretrained(cp,safe_serialization=True,save_embedding_layers=False)
            other=adapter_model(Qwen3_5ForCausalLM(config),{'rank':2,'alpha':4,'dropout':0.,'target_modules':['q_proj']})
            load_weights(other,cp)
            for n,p in other.named_parameters():
                if 'lora_' in n:self.assertTrue(torch.equal(p,dict(model.named_parameters())[n]))
            cfg={'world_size':2,'max_steps':8,'thinking':True}
            (run/'train_config.json').write_text(json.dumps(cfg))
            (cp/'training_step.json').write_text(json.dumps({'step':3,'thinking':True}))
            plan={'steps':[{'step':i} for i in range(1,9)]}
            (run/'batch_plan.json').write_text(json.dumps(plan))
            for name in ('optimizer.pt','rng_rank0.pt','rng_rank1.pt'):(cp/name).write_bytes(b'fixture')
            self.assertEqual(validate_resume(cfg,cp,plan),3)
            with self.assertRaisesRegex(ValueError,'Resume recipe differs'):
                validate_resume({**cfg,'thinking':False},cp,plan)


class ReasoningWorkerTests(unittest.TestCase):
    def test_native_thinking_two_gpu_routes_token_accounting_and_gate(self):
        from training.buyer_diagnostics import worker
        from training.buyer_reasoning_data import personas
        from training.buyer_reasoning_gate import validate_continuation,compute
        from training.decision_ablation import fingerprint
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        cases=json.loads((DATA/'cash_probe_cases.json').read_text())
        class Client:
            def __init__(self,**kw):pass
            def complete(inner,**kw):
                self.assertTrue(kw['extra_body']['chat_template_kwargs']['enable_thinking'])
                ChatCompletionRequest.model_validate({k:kw[k] for k in ('model','messages','tools','extra_body') if k!='extra_body'} | kw['extra_body'])
                obs=json.loads(kw['messages'][-1]['content']);state={'bad':[],'pending':None,'last_buy':None}
                if obs['offers_to_you']:
                    row=obs['offers_to_you'][0];tool='accept_offer';args={'offer_id':row['offer_id']};price=row['price']*row['qty']
                else:
                    row=obs['board'][0];tool='buy_listed';args={'seller_id':row['seller'],'product_id':row['product_id'],'qty':1};price=row['price']
                    state['last_buy']=[row['seller'],row['product_id']]
                if 'candidate' in kw['model'] and obs['you']['cash']<price:
                    tool='wait';args={};state['last_buy']=None
                args['private_state']=json.dumps(state)
                action=SimpleNamespace(name=tool,arguments=args,raw_arguments=json.dumps(args),arguments_error=None)
                return SimpleNamespace(tool_call=action,stop_reason='tool_calls',outcome='accepted',error=None,text='',
                    reasoning='Cash and total checked before choosing a tool.',usage={'input_tokens':1200,'output_tokens':80,'reasoning_tokens':70,'total_tokens':1350})
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_reasoning_worker_cpu_') as temp:
            folder=Path(temp);cp=folder/'candidate';cp.mkdir()
            for name in ('adapter_config.json','adapter_model.safetensors'):(cp/name).write_bytes(b'CPU fixture')
            metadata={'experiment':'buyer_operations_reasoning_v1','candidate_checkpoint':str(cp),
                      'candidate_fingerprint':fingerprint(cp),'training_manifest_sha256':'fixture-manifest'}
            plan={'devices':[{'index':str(i),'fraction':.8} for i in (0,1)],'port':8000,'seeds':[0,1],
                  'personas':{f'operations_reasoning_p{i+1}':p for i,p in enumerate(personas())},
                  'adapters':{'base_on':None,'candidate':str(cp)},'calls':768,'analysis':'cash_decision',
                  'thinking':True,'experiment_manifest':metadata}
            (folder/'plan.json').write_text(json.dumps(plan));(folder/'cases.json').write_text(json.dumps(cases))
            with patch('adlmarket.serving.VLLMCluster',return_value=nullcontext()),patch('adlmarket.serving.register_adapters'), \
                 patch('adlmarket.client.ModelClient',Client),redirect_stdout(io.StringIO()):worker(folder)
            gate=json.loads((folder/'reasoning_gate.json').read_text())
            self.assertTrue(gate['passed']);self.assertEqual(gate['token_stats']['candidate']['p95_generated_tokens'],150)
            report=json.loads((folder/'cash_decision_pairs.json').read_text())
            self.assertEqual(report['gpu_calls'],{'0':384,'1':384})
            self.assertEqual(report['models']['candidate']['channel']['offer']['both_correct'],96)
            self.assertTrue(validate_continuation(folder/'reasoning_gate.json',{'data_manifest_sha256':'fixture-manifest'},cp)['passed'])
            rows=[json.loads(s) for s in (folder/'scored_decisions.jsonl').read_text().splitlines()]
            for r in rows:
                if r['adapter']=='candidate':r['usage']['reasoning_tokens']=2000
            (folder/'scored_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertFalse(compute(folder)['passed'])
            with self.assertRaisesRegex(ValueError,'gate did not pass'):
                validate_continuation(folder/'reasoning_gate.json',{'data_manifest_sha256':'fixture-manifest'},cp)


if __name__=='__main__':unittest.main()
