"""CPU tests of supervision, engine legality, frozen weights and leakage boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from training.buyer_operations_data import make_record, TASKS, encode, BASE, REVISION
from training.buyer_diagnostics import evaluate


class OperationDataTests(unittest.TestCase):
    def test_diagnostic_explicit_two_cards_rejects_single_before_launch(self):
        from unittest.mock import patch
        from training.buyer_diagnostics import main
        record = make_record(13, 'binding_offer', 0)
        case = {**record, 'name': record['id'], 'contexts':['recorded'], 'state_mode':'operations'}
        cards = [{'index':str(i),'uuid':f'fake_{i}','total_gib':24.,'free_gib':free}
                 for i,free in enumerate([24.,6.])]
        with tempfile.TemporaryDirectory(prefix='adl_eval_cards_cpu_',dir='/tmp') as temporary:
            path=Path(temporary)/'cases.json';path.write_text(json.dumps([case]))
            argv=['buyer_diagnostics.py','--versions','operations','--cases-file',str(path),
                  '--output-dir',temporary,'--gpus','0,1']
            with patch('sys.argv',argv), patch('training.local_resources.query_gpus',return_value=cards), \
                 patch('training.buyer_diagnostics.subprocess') as processes:
                with self.assertRaisesRegex(SystemExit,'Requested 2 GPUs but only 1 fit'):
                    main()
                processes.Popen.assert_not_called()

    def test_targets_pass_independent_operation_checks(self):
        for seed in [13, 29, 77]:
            for task in TASKS:
                record = make_record(seed, task, seed%3)
                fn = record['answer']['tool_calls'][0]['function']
                case = {**record, 'name': record['id'], 'state_mode': 'operations'}
                result = evaluate(case, fn['name'], fn['arguments'])
                self.assertEqual(result['errors'], [], (task, result))
                self.assertTrue(record['engine_result']['ok'])
                self.assertNotIn('engine_result', json.loads(record['messages'][-1]['content']))
                if task in ('empty','no_profit','loss_multiunit_offer'):
                    self.assertEqual(fn['name'], 'wait')
                if task == 'binding_offer':
                    self.assertEqual(fn['name'], 'accept_offer')
        record = make_record(13, 'live_pair', 0)
        case = {**record, 'name': record['id'], 'state_mode': 'operations'}
        state = json.dumps({'bad': [], 'pending': None, 'last_buy': None})
        self.assertIn('missed_positive_operation', evaluate(case, 'wait', {'private_state': state})['opportunity'])
        self.assertTrue(evaluate(case, None, {})['positive_reference'])

    def test_template_mask_matches_only_final_tool_call(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
        record = make_record(29, 'failed_pair', 0)
        encoded = encode(tokenizer, record)
        target = tokenizer.decode(encoded['input_ids'][encoded['prompt_length']:])
        self.assertEqual(target.count('<tool_call>'), 1)
        self.assertIn('<|im_end|>', target)
        self.assertNotIn('no_stock', target)
        self.assertIn('no_stock', tokenizer.decode(encoded['input_ids'][:encoded['prompt_length']]))

    def test_adapter_gradient_and_completion_alignment(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from training.buyer_operations_train import adapter_model, completion_loss, save_adapter
        torch.manual_seed(7)
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                     linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                     linear_num_value_heads=2, layer_types=['linear_attention','full_attention'], tie_word_embeddings=True)
        base = Qwen3_5ForCausalLM(config)
        before = base.model.embed_tokens.weight.detach().clone()
        import yaml
        cfg = yaml.safe_load((Path(__file__).parent/'configs/buyer_operations_v1.yaml').read_text())
        cfg.update(rank=2, alpha=4)
        model = adapter_model(base, cfg)
        row = {'input_ids': [4,5,6,7,8,9,10,11], 'prompt_length': 5}
        model.eval()
        loss = completion_loss(model, row, 'cpu')
        ids = torch.tensor([row['input_ids']])
        full = model(input_ids=ids, use_cache=False).logits
        reference = torch.nn.functional.cross_entropy(full[:,4:-1].reshape(-1,128), ids[:,5:].reshape(-1))
        self.assertTrue(torch.allclose(loss, reference, atol=1e-6))
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None), 0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        optimizer.step()
        self.assertTrue(torch.equal(before, base.model.embed_tokens.weight))
        self.assertTrue(any(p.abs().sum() > 0 for n,p in model.named_parameters() if 'lora_B' in n))
        with tempfile.TemporaryDirectory(prefix='adl_adapter_cpu_', dir='/tmp') as folder:
            save_adapter(model, Path(folder), cfg, 1)
            self.assertTrue((Path(folder)/'adapter_model.safetensors').exists())

    def test_paired_evaluator_routes_base_and_candidate_and_penalizes_waiting(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from contextlib import nullcontext
        from training.buyer_diagnostics import worker
        from training.buyer_operations_data import personas
        from adlmarket.setups import market_run
        root = Path(__file__).resolve().parents[1]
        run = market_run(root/'configs/basic_buyer.yaml', root=root)
        adapter = run.models[run.buyers[0].model].lora.path
        record = make_record(13, 'binding_offer', 0)
        case = {**record, 'name': record['id'], 'contexts': ['recorded'], 'state_mode': 'operations'}
        calls = []
        class FakeClient:
            def __init__(self, **kwargs): pass
            def complete(self, **kwargs):
                from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
                ChatCompletionRequest.model_validate({'model':kwargs['model'],'messages':copy.deepcopy(kwargs['messages']),'tools':kwargs['tools']})
                calls.append((kwargs['model'], kwargs['base_url']))
                args = {'private_state': json.dumps({'bad': [], 'pending': None, 'last_buy': None})}
                action = SimpleNamespace(name='wait', arguments=args, raw_arguments=json.dumps(args), arguments_error=None)
                return SimpleNamespace(tool_call=action, stop_reason='tool_calls', outcome='accepted', error=None, text='')
        with tempfile.TemporaryDirectory(prefix='adl_paired_eval_cpu_', dir='/tmp') as temporary:
            folder = Path(temporary)
            plan = {'devices': [{'index':'0','fraction':.8},{'index':'1','fraction':.8}],
                    'port': 8000, 'seeds':[0,1], 'personas':{'operations_p1':personas()[0]},
                    'adapters':{'base':None,'candidate':adapter}}
            (folder/'plan.json').write_text(json.dumps(plan))
            (folder/'cases.json').write_text(json.dumps([case]))
            with patch('adlmarket.serving.VLLMCluster', return_value=nullcontext()), \
                 patch('adlmarket.serving.register_adapters'), patch('adlmarket.client.ModelClient', FakeClient):
                worker(folder)
            self.assertEqual(len(calls), 4)
            self.assertEqual({url for _,url in calls}, {'http://localhost:8000/v1','http://localhost:8001/v1'})
            self.assertTrue(any(name=='Qwen/Qwen3.5-4B' for name,_ in calls))
            self.assertTrue(any('candidate' in name for name,_ in calls))
            scores = json.loads((folder/'scoreboard.json').read_text())
            self.assertEqual(scores['scoring_version'], 'diagnostic_operations_v1')
            self.assertEqual(len(scores['scores']), 2)
            self.assertTrue(all(s['operation_capture']==0 for s in scores['scores'].values()))
            evidence = json.loads((folder/'operation_gate.json').read_text())
            self.assertEqual(evidence['personas']['operations_p1']['finding'], 'mixed_or_inconclusive')


if __name__ == '__main__':
    unittest.main()
