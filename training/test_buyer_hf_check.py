"""CPU checks of templates, parser, target logits and diagnostic reporting."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.buyer_hf_check import target_encoding, parse_raw, split_loss, score_target, report, verify_loaded_adapter, loaded_adapter_metadata


class HFCheckTests(unittest.TestCase):
    def test_official_template_token_regions_and_production_parser(self):
        from transformers import AutoTokenizer
        from training.buyer_operations_data import BASE, REVISION, make_record, choose, personas
        from training.buyer_diagnostics import replace_persona
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
        for task in ('stale_memory', 'binding_offer'):
            case = make_record(20337802, task, 0)
            tool, args = choose(case['situation'], case['observation'])
            messages = replace_persona(case['messages'], personas()[0], 'clean_fresh')
            original = copy.deepcopy(messages)
            for name, arguments in [(tool,args),('wait',{'private_state':'{"bad":[],"pending":null,"last_buy":null}'})]:
                encoded = target_encoding(tokenizer, messages, case['tools'], name, arguments)
                self.assertEqual(len(encoded['regions']), len(encoded['input_ids'])-encoded['prompt_length'])
                self.assertIn('tool_name', encoded['regions'])
                self.assertIn('private_state', encoded['regions'])
                if name!='wait': self.assertIn('trade_arguments', encoded['regions'])
                parsed = parse_raw(tokenizer, encoded['completion'], case['tools'])
                self.assertIsNone(parsed['parse_error'])
                self.assertEqual(parsed['tool'], name)
                self.assertEqual(parsed['arguments'], arguments)
            self.assertEqual(messages, original)
            malformed = parse_raw(tokenizer, 'Market over, no call.', case['tools'])
            self.assertEqual(malformed['parsed_calls'], 0)
            self.assertIsNone(malformed['tool'])

    def test_tiny_cpu_peft_generation_and_teacher_logits_match_full_forward(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, GenerationConfig
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(7)
        cfg = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=2, layer_types=['linear_attention','full_attention'], tie_word_embeddings=True)
        base = Qwen3_5ForCausalLM(cfg)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'], task_type='CAUSAL_LM'))
        model.eval()
        row = {'input_ids':[4,5,6,7,8,9,10,11], 'prompt_length':4,
               'regions':['tool_name','trade_arguments','private_state','format']}
        ids = torch.tensor([row['input_ids']])
        with torch.inference_mode():
            scores = score_target(model, row, 'cpu')
            full = model(input_ids=ids, use_cache=False).logits
            expected = torch.nn.functional.cross_entropy(full[:,3:-1].reshape(-1,128),ids[:,4:].reshape(-1),reduction='none')
            self.assertTrue(torch.allclose(torch.tensor(scores['token_nll']),expected,atol=1e-6))
            self.assertAlmostEqual(scores['mean_nll'], expected.mean().item(), places=5)
            generated = model.generate(input_ids=ids[:,:4],attention_mask=torch.ones_like(ids[:,:4]),
                generation_config=GenerationConfig(do_sample=True,temperature=.7,top_p=.95,top_k=0,
                    eos_token_id=[0,1],pad_token_id=0,max_new_tokens=2),use_cache=True,logits_to_keep=1)
            self.assertGreater(generated.shape[1],4)
            self.assertLessEqual(generated.shape[1],6)
        self.assertEqual(sum(v['tokens'] for v in scores['regions'].values()),4)
        self.assertAlmostEqual(sum(v['sum_nll'] for v in scores['regions'].values()),sum(scores['token_nll']))
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_hf_adapter_cpu_') as temporary:
            model.save_pretrained(temporary,safe_serialization=True,save_embedding_layers=False)
            self.assertGreater(verify_loaded_adapter(model,temporary)['tensors_checked'],0)
            # Real Qwen attention has float scaling; the old worker introspection crashed here.
            self.assertTrue(any(isinstance(getattr(m,'scaling',None),float) for m in model.modules()))
            with self.assertRaises(TypeError):
                {n:float(m.scaling['default']) for n,m in model.named_modules()
                 if hasattr(m,'scaling') and 'default' in m.scaling}
            metadata = loaded_adapter_metadata(model,temporary)
            self.assertEqual(metadata['r'],2)
            self.assertEqual(metadata['alpha'],4)
            self.assertEqual(len(metadata['scaling']),1)
            self.assertEqual(set(metadata['scaling'].values()),{2.})
            json.dumps(metadata)  # The exact metadata-writing path must be CPU testable.
            with torch.no_grad():
                next(p for name,p in model.named_parameters() if 'lora_A' in name).add_(1.)
            with self.assertRaisesRegex(ValueError,'tensor differs'):
                verify_loaded_adapter(model,temporary)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            split_loss([float('nan')], ['tool_name'])

    def test_report_all_wait_is_not_proof_and_incomplete_or_misrouted_inputs_fail(self):
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_hf_report_cpu_') as temporary:
            folder=Path(temporary); jobs=[]; rows=[]
            for i in range(4):
                for seed in (0,1):
                    job={'id':f'case{i}_s{seed}','input_sha256':f'sha{i}', 'gpu':str(seed),
                         'vllm':{'tool':'wait'}}
                    jobs.append(job)
                    rows.append({**job,'case':f'case{i}','seed':seed,'parsed':{'tool':'wait','parse_error':None},
                        'operation_capture':False,'truncated':False,
                        'scores':{label:{'branch_logprob_reference_minus_wait':value,
                             **{candidate:{'regions':{'tool_name':{'tokens':2,'sum_nll':3.}}}
                                for candidate in ('reference','wait')}} for label,value in [('base',1.),('step8',-1.)]}})
            (folder/'jobs.json').write_text(json.dumps(jobs))
            path=folder/'decisions_0.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with redirect_stdout(io.StringIO()):result=report(folder)
            self.assertEqual(result['tools'],{'wait':8})
            self.assertEqual(result['gpu_calls'],{'0':4,'1':4})
            self.assertEqual(result['loss_by_region']['step8']['reference']['tool_name']['mean_nll'],1.5)
            self.assertIn('does not identify a unique SFT cause',result['finding'])
            rows[0]['operation_capture']=True;rows[0]['parsed']['tool']='buy_listed'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with redirect_stdout(io.StringIO()):result=report(folder)
            self.assertIn('investigate backend',result['finding'])
            rows[0]['gpu']='wrong'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'route changed'):report(folder)
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows[1:]))
            with self.assertRaisesRegex(ValueError,'Incomplete'):report(folder)

    def test_prepare_only_never_queries_or_starts_gpu(self):
        from training.buyer_hf_check import main
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_hf_prepare_cpu_') as temporary:
            with patch('sys.argv',['buyer_hf_check.py','--prepare-only','--output-dir',temporary]), \
                 patch('training.buyer_hf_check.prepare',return_value=([{'id':i} for i in range(8)],{})), \
                 patch('training.local_resources.query_gpus') as query, \
                 patch('training.buyer_hf_check.subprocess.Popen') as launch, redirect_stdout(io.StringIO()):
                main()
                query.assert_not_called();launch.assert_not_called()

    def test_wrapper_surfaces_worker_root_cause_and_keeps_evaluation_invalid(self):
        from types import SimpleNamespace
        from training.buyer_hf_check import main
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_hf_failure_cpu_') as temporary:
            jobs=[{'id':f'case{i}_s{seed}','seed':seed,'vllm':{'gpu':str(seed)}} for i in range(4) for seed in (0,1)]
            cards=[{'index':str(i),'uuid':str(i),'total_gib':24.,'free_gib':24.} for i in range(2)]
            def launch(*args,**kwargs):
                kwargs['stdout'].write("Traceback: worker metadata failure\nTypeError: argument of type 'float' is not iterable\n")
                kwargs['stdout'].flush()
                return SimpleNamespace(returncode=1,poll=lambda:1,wait=lambda **kw:1)
            output=io.StringIO()
            with patch('sys.argv',['buyer_hf_check.py','--output-dir',temporary,'--gpus','0,1']), \
                 patch('training.buyer_hf_check.prepare',return_value=(jobs,{})), \
                 patch('training.local_resources.query_gpus',return_value=cards), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)), \
                 patch('training.buyer_hf_check.subprocess.Popen',side_effect=launch), \
                 patch('training.buyer_hf_check.os.killpg') as stop, redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError,'HF worker failed'):
                    main()
            stop.assert_not_called()
            self.assertIn('ROOT CAUSE GPU0',output.getvalue())
            self.assertIn('ROOT CAUSE GPU1',output.getvalue())
            self.assertIn("argument of type 'float' is not iterable",output.getvalue())
            status=next(Path(temporary).glob('*/evaluation_status.json'))
            self.assertEqual(json.loads(status.read_text())['status'],'invalid')


if __name__=='__main__':unittest.main()
