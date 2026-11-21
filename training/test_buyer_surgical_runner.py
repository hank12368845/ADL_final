"""CPU checks for the real launcher boundary, microbatch math and fixed evaluation."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.prepare_buyer_surgical import ROOT,DATA


class SurgicalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.train_buyer_surgical import prepare
        cls.cfg=prepare()
        cls.cases=json.loads((DATA/'evaluation_cases.json').read_text())

    def test_real_torchrun_parser_and_internal_worker_dispatch(self):
        from torch.distributed.run import parse_args
        from training.train_buyer_surgical import worker_command,main
        for kind in ('preflight','train'):
            cmd=worker_command(Path('/tmp/run with spaces'),kind);a=parse_args(cmd[4:])
            self.assertEqual(a.training_script,str(ROOT/'training/train_buyer_surgical.py'))
            self.assertEqual(a.training_script_args,['--worker',kind,'--run','/tmp/run with spaces'])
            with patch('sys.argv',['runner','--worker',kind,'--run','/tmp/example']),patch('training.buyer_surgical_train.worker') as worker:
                main();worker.assert_called_once_with(Path('/tmp/example'),kind=='preflight')
        with patch('sys.argv',['runner','--worker','eval_candidates','--run','/tmp/example','--labels','10,20']),patch('training.buyer_surgical_eval.evaluate') as evaluate:
            main();evaluate.assert_called_once_with(Path('/tmp/example'),['10','20'])

    def test_allocator_override_is_scoped_to_training_workers(self):
        from types import SimpleNamespace
        from training.train_buyer_surgical import stage
        cards=[dict(index=str(i),fraction=.85,budget_gib=20.4) for i in (0,1)]
        for kind in ('preflight','train','eval_parent','eval_candidates'):
            cfg=dict(devices=cards)
            with tempfile.TemporaryDirectory(dir='/tmp') as tmp,redirect_stdout(io.StringIO()), \
                patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                patch('training.train_buyer_surgical.subprocess.Popen',return_value=SimpleNamespace(returncode=0,poll=lambda:0)) as process:
                stage(Path(tmp),cfg,kind,{'CUDA_VISIBLE_DEVICES':'0,1'})
                actual=process.call_args.kwargs['env']
                self.assertEqual(actual.get('PYTORCH_ALLOC_CONF'),'expandable_segments:True' if kind in ('preflight','train') else None)

    def test_frozen_20_step_recipe_microbatches_and_schedule(self):
        from training.buyer_surgical_train import load_data,rate
        train,batches,m=load_data(DATA)
        self.assertEqual(len(train),67);self.assertEqual(len(batches),20)
        self.assertEqual((self.cfg['max_steps'],self.cfg['effective_batch_size'],self.cfg['checkpoint_steps']),(20,4,[10,20]))
        self.assertEqual(self.cfg['parent_adapter'],str(ROOT/'training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084'))
        for b in batches:
            self.assertEqual((len(b[0::2]),len(b[1::2])),(2,2))
            self.assertFalse(set(b[0::2])&set(b[1::2]))
        self.assertEqual([rate(0),rate(1),rate(2),rate(20)],[.5,1.,1.,0.])

    def test_activation_offload_preserves_cpu_autograd_and_short_path(self):
        import torch
        from training.buyer_surgical_train import activation_context
        torch.manual_seed(7);model=torch.nn.Linear(5,3);x=torch.randn(4,5)
        model(x).square().mean().backward();reference=[p.grad.clone() for p in model.parameters()];model.zero_grad()
        original=torch.autograd.graph.save_on_cpu
        with patch('torch.autograd.graph.save_on_cpu',side_effect=lambda **kwargs:original(pin_memory=False)) as hook:
            with activation_context({'input_ids':[1]*16000},self.cfg):model(x).square().mean().backward()
            hook.assert_called_once_with(pin_memory=True)
        for p,g in zip(model.parameters(),reference):torch.testing.assert_close(p.grad,g)
        with patch('torch.autograd.graph.save_on_cpu') as hook:
            with activation_context({'input_ids':[1]*100},self.cfg):pass
            hook.assert_not_called()

    def test_native_qwen_lora_chunking_matches_full_mlp_and_nested_backward(self):
        import torch
        from torch.utils.checkpoint import checkpoint
        from peft import LoraConfig,get_peft_model
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
        from training.buyer_surgical_train import install_chunked_mlp
        torch.manual_seed(13)
        config=Qwen3_5TextConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        original=get_peft_model(torch.nn.Sequential(Qwen3_5MLP(config,32)),
            LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['gate_proj','up_proj','down_proj']))
        for name,p in original.named_parameters():
            if 'lora_B' in name:torch.nn.init.normal_(p,std=.03)
        chunked=copy.deepcopy(original);before={k:v.clone() for k,v in chunked.state_dict().items()}
        self.assertEqual(install_chunked_mlp(chunked,8),1)
        self.assertEqual(set(before),set(chunked.state_dict()))
        x=torch.randn(1,37,16,requires_grad=True);other=x.detach().clone().requires_grad_()
        a=original(x);a.square().mean().backward()
        lengths=[]
        native=next(m for m in chunked.modules() if isinstance(m,Qwen3_5MLP))
        handle=native.gate_proj.register_forward_pre_hook(lambda module,args:lengths.append(args[0].shape[-2]))
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            b=checkpoint(chunked,other,use_reentrant=False)
            b.square().mean().backward()
        handle.remove()
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(x.grad,other.grad,rtol=1e-5,atol=1e-6)
        for (name,p),(other_name,q) in zip(original.named_parameters(),chunked.named_parameters()):
            self.assertEqual(name,other_name)
            if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
        self.assertTrue(lengths);self.assertLessEqual(max(lengths),8)
        for k,v in chunked.state_dict().items():torch.testing.assert_close(v,before[k])
        chunked.eval()
        with torch.no_grad():torch.testing.assert_close(chunked(x.detach()),original(x.detach()))
        # Match the failing deployment precision: bf16 base/inputs, fp32 LoRA.
        mixed=copy.deepcopy(original).to(torch.bfloat16);mixed.zero_grad(set_to_none=True)
        for name,p in mixed.named_parameters():
            if 'lora_' in name:p.data=p.data.float()
        mixed_chunks=copy.deepcopy(mixed);install_chunked_mlp(mixed_chunks,8)
        u=x.detach().to(torch.bfloat16).requires_grad_();v=u.detach().clone().requires_grad_()
        with torch.autocast('cpu',dtype=torch.bfloat16):
            left=mixed(u);left.float().square().mean().backward()
        with torch.autograd.graph.save_on_cpu(pin_memory=False),torch.autocast('cpu',dtype=torch.bfloat16):
            right=checkpoint(mixed_chunks,v,use_reentrant=False);right.float().square().mean().backward()
        torch.testing.assert_close(left,right,rtol=.03,atol=.003)
        torch.testing.assert_close(u.grad,v.grad,rtol=.03,atol=.0003)
        for (name,p),(other_name,q) in zip(mixed.named_parameters(),mixed_chunks.named_parameters()):
            if p.requires_grad:
                self.assertEqual(p.dtype,torch.float32)
                torch.testing.assert_close(p.grad,q.grad,rtol=.03,atol=.0003)

    def test_native_inputs_and_formal_budget_for_all102(self):
        from training.buyer_surgical_eval import make_policy
        from adlmarket.config import ModelSpec
        from adlmarket.client import CallResult,ToolCall
        class Client:
            def complete(inner,**kwargs):
                inner.kwargs=kwargs
                return CallResult(text='',reasoning='Wait.',stop_reason='tool_calls',stop_sequence_hit=None,
                    usage={'output_tokens':10},timing={},outcome='accepted',tool_call=ToolCall('wait',{}))
        for c in self.cases:
            client=Client();policy=make_policy(c,ModelSpec(model='Qwen/Qwen3.5-4B'),client)
            policy.decide(copy.deepcopy(c['observation']))
            self.assertEqual(client.kwargs['messages'],c['messages'])
            self.assertEqual(client.kwargs['max_tokens'],min(8192,c['observation']['output_tokens_left_this_round']))
            self.assertEqual((client.kwargs['temperature'],client.kwargs['top_p']),(.7,.95))

    def test_all_wait_cannot_pass_and_low_gain_does_not_hide_capture_regression(self):
        from training.buyer_surgical_eval import assess,summaries,gates
        fallback={c['name']:c for c in json.loads((ROOT/'training/data/buyer_dollar_lines_v1/evaluation_cases.json').read_text())}
        golden=[];waiting=[]
        for c in self.cases:
            answer=c.get('answer',fallback.get(c['name'],{}).get('answer'))
            fn=answer['tool_calls'][0]['function']
            a=assess(c,fn['name'],fn['arguments'],dict(outcome='accepted',reasoning=answer['reasoning_content']))
            if a['errors']==['persona_cooldown']:
                # Frozen binding-v1 gold has the already documented immediate-retry conflict.
                # Preserve its input, reject that answer, and test a currently allowed alternative.
                self.assertEqual(c.get('family'),'new_listing')
                from training.prepare_buyer_surgical import repair_answer
                corrected=repair_answer(c);fn=corrected['tool_calls'][0]['function']
                a=assess(c,fn['name'],fn['arguments'],dict(outcome='accepted',reasoning=corrected['reasoning_content']))
            self.assertTrue(a['valid'],(c['name'],a['errors']))
            golden.append(a);waiting.append(assess(c,'wait',{},dict(outcome='accepted',reasoning='Wait.')))
        reference=summaries(golden);self.assertFalse(all(gates(reference,summaries(waiting)).values()))
        child=copy.deepcopy(reference)
        child['binding_validation']['personas']['3']['captured_visible_lower_bound_cents']-=1
        self.assertFalse(gates(reference,child)['binding_validation_captured_visible_lower_bound_cents_retained'])
        self.assertEqual(reference['training_market_seen']['calls'],6)

    def test_cpu_check_and_mock_full_pipeline_never_start_gpu_or_promote(self):
        from training.train_buyer_surgical import main
        from training.buyer_surgical_eval import summaries,assess
        with patch('sys.argv',['runner','--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu,patch('training.train_buyer_surgical.stage') as stage,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();stage.assert_not_called()
        protected=[ROOT/f'agents/example_student/buyer/persona_{p}.md' for p in (1,2,3)]+[ROOT/'agents/example_student/buyer/checkpoint/adapter_model.safetensors']
        before={p:p.read_bytes() for p in protected};stages=[]
        def fake_stage(folder,cfg,kind,env,labels=None):
            stages.append((kind,labels))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
            if kind=='train':
                for n in (10,20):
                    cp=folder/'checkpoints'/f'step_{n:04d}';cp.mkdir(parents=True);(cp/'checkpoint_complete.json').write_text('{}')
                (folder/'training_finished.json').write_text('{}')
            for label in labels or []:
                rows=[assess(c,'wait',{},dict(outcome='accepted',reasoning='Wait.')) for c in self.cases]
                (folder/f'evaluation_{label}.json').write_text(json.dumps(dict(summary=summaries(rows),decisions=rows)))
        cards=[dict(index=str(i),fraction=.85,budget_gib=20.4) for i in (0,1)]
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp,patch('sys.argv',['runner','--output-dir',tmp]), \
            patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})),patch('training.local_resources.runtime_cache',return_value=({},None)), \
            patch('training.train_buyer_surgical.stage',side_effect=fake_stage),redirect_stdout(io.StringIO()):
            folder=main();review=json.loads((folder/'surgical_review.json').read_text())
            self.assertTrue(review['complete']);self.assertIsNone(review['development_candidate']);self.assertFalse(review['automatic_promotion'])
        self.assertEqual(stages,[('preflight',None),('eval_parent',['parent']),('train',None),('eval_candidates',['10','20'])])
        for p,content in before.items():self.assertEqual(p.read_bytes(),content)


if __name__=='__main__':unittest.main()
