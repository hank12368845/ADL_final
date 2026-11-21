"""CPU-only checks for objective isolation, archived controls and cash pair routing."""
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class DecisionObjectiveTests(unittest.TestCase):
    def test_first_token_loss_and_gradients_match_manual_causal_reference(self):
        import torch
        from training.buyer_operations_train import completion_losses
        class LogitsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.randn(8, 20))
            def forward(self, input_ids, logits_to_keep, **kwargs):
                return SimpleNamespace(logits=self.logits[logits_to_keep].unsqueeze(0))
        torch.manual_seed(9)
        model = LogitsModel()
        row = {'input_ids': [4, 5, 6, 7, 8, 9, 10, 11, 12],
               'prompt_length': 4, 'tool_name_positions': [1, 2, 3]}
        parts = completion_losses(model, row, 'cpu', 1., 'first_token')
        nll = torch.nn.functional.cross_entropy(model.logits[3:8], torch.tensor(row['input_ids'][4:]), reduction='none')
        expected = nll.mean()+nll[1]
        self.assertTrue(torch.allclose(parts['total'], expected))
        self.assertTrue(torch.allclose(parts['tool_name'], nll[1:4].mean()))
        parts['total'].backward()
        actual = model.logits.grad.clone()
        model.zero_grad(set_to_none=True)
        expected.backward()
        self.assertTrue(torch.allclose(actual, model.logits.grad))
        self.assertTrue(torch.equal(actual[:3], torch.zeros_like(actual[:3])))
        old = completion_losses(model, row, 'cpu', 1.)
        explicit = completion_losses(model, row, 'cpu', 1., 'name_mean')
        self.assertTrue(torch.equal(old['total'], explicit['total']))
        self.assertTrue(torch.allclose(old['total'], nll.mean()+nll[1:4].mean()))
        for weight, mode in ((0., 'first_token'), (0., 'name_mean')):
            unweighted = completion_losses(model, row, 'cpu', weight, mode)
            self.assertTrue(torch.equal(unweighted['total'], unweighted['completion']))
        with self.assertRaisesRegex(ValueError, 'Unknown decision'):
            completion_losses(model, row, 'cpu', 1., 'typo')
        with self.assertRaisesRegex(ValueError, 'requires tool-name'):
            completion_losses(model, {**row, 'tool_name_positions': []}, 'cpu', 1., 'first_token')


class AblationControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from training.buyer_operations_train import check_data
        from training.decision_ablation import audit_training
        cls.cfg = yaml.safe_load((ROOT/'training/configs/buyer_operations_v2_2.yaml').read_text())
        cls.train, _ = check_data(cls.cfg, ROOT)
        cls.audit = audit_training(cls.cfg, ROOT, cls.train)

    def test_exact_data_rank_order_and_distinct_first_tokens(self):
        self.assertEqual(self.audit['control_samples_by_rank'], {'0': 64, '1': 64})
        self.assertEqual(len(set(self.audit['first_tool_token_ids'].values())), 3)
        self.assertEqual(self.audit['tool_name_token_lengths'],
                         {'buy_listed': [3], 'accept_offer': [2], 'wait': [1]})

    def test_recipe_drift_rejected_before_training(self):
        from training.decision_ablation import audit_training
        for key, value in (('max_steps', 24), ('learning_rate', 5e-5), ('seed', 42),
                           ('decision_weight', 2.), ('sampling', 'paired_4_2_2')):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Fixed recipe changed'):
                audit_training({**self.cfg, key: value}, ROOT, self.train)

    def test_cpu_entry_does_not_query_or_launch_gpu(self):
        from training.train_buyer_operations import main
        with patch('sys.argv', ['train_buyer_operations.py', '--config',
                               str(ROOT/'training/configs/buyer_operations_v2_2.yaml'), '--cpu-check']), \
             patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_operations.subprocess.Popen') as launch, \
             patch('training.decision_ablation.audit_training', return_value=self.audit), redirect_stdout(io.StringIO()):
            main()
            gpu.assert_not_called(); launch.assert_not_called()

    def test_comparison_uses_matched_step_and_detects_reference_changes(self):
        from training.evaluate_buyer_operations_v2_2 import comparison_spec
        from training.decision_ablation import fingerprint
        cfg = {**self.cfg, 'decision_ablation': self.audit}
        control = Path(self.audit['control_run'])
        old = ROOT/self.cfg['comparison_adapter']
        cfg['comparison_fingerprints'] = fingerprint(old)
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_v22_spec_cpu_') as temporary:
            run = Path(temporary)
            (run/'train_config.json').write_text(json.dumps(cfg))
            candidate = run/'checkpoints/step_0008'; candidate.mkdir(parents=True)
            (candidate/'training_step.json').write_text(json.dumps({'step': 8}))
            adapters, cases, provenance = comparison_spec(run, 'step_0008')
            self.assertEqual(adapters['v21_control'], str(control/'checkpoints/step_0008'))
            self.assertEqual(adapters['v2_step8'], str(old.resolve()))
            self.assertEqual(adapters['candidate'], str(candidate))
            self.assertEqual(provenance['checkpoint'], 'step_0008')
            cfg['comparison_fingerprints']['adapter_model.safetensors'] = 'changed'
            (run/'train_config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'Comparison adapter changed'):
                comparison_spec(run, 'step_0008')

    def test_evaluator_prepare_only_validates_three_native_adapter_routes_without_gpu(self):
        from training.evaluate_buyer_operations_v2_2 import main
        from training.decision_ablation import fingerprint
        cfg = {**self.cfg, 'decision_ablation': self.audit,
               'comparison_fingerprints': fingerprint(ROOT/self.cfg['comparison_adapter'])}
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_v22_prepare_cpu_') as temporary:
            run = Path(temporary); (run/'checkpoints').mkdir()
            # CPU fixture only: use the archived adapter to check real PEFT/request schemas.
            (run/'checkpoints/step_0008').symlink_to(Path(self.audit['control_run'])/'checkpoints/step_0008')
            (run/'train_config.json').write_text(json.dumps(cfg))
            output = run/'reports'; output.mkdir()
            with patch('sys.argv', ['evaluate_buyer_operations_v2_2.py', str(run), '--prepare-only']), \
                 patch('training.local_resources.output_parent', return_value=(output, None)), \
                 patch('training.local_resources.query_gpus') as gpu, \
                 patch('training.buyer_diagnostics.subprocess.Popen') as launch, redirect_stdout(io.StringIO()):
                folder = main()
                gpu.assert_not_called(); launch.assert_not_called()
            plan = json.loads((folder/'plan.json').read_text())
            self.assertEqual(plan['calls'], 576)
            self.assertEqual(plan['analysis'], 'cash_decision')
            self.assertEqual(plan['devices'], [])
            self.assertEqual(set(plan['adapters']), {'v2_step8', 'v21_control', 'candidate'})
            self.assertIn('adapter_fingerprints', plan['experiment_manifest'])
            self.assertFalse((folder/'evaluation_status.json').exists())


class CashPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT/'training/data/buyer_operations_v2_1/cash_probe_cases.json').read_text())

    def test_counterfactual_really_changes_only_cash(self):
        from training.cash_decision_report import case_pairs
        self.assertEqual(len(case_pairs(self.cases)), 16)
        changed = copy.deepcopy(self.cases)
        changed[1]['messages'][0]['content'] += 'Another instruction'
        with self.assertRaisesRegex(ValueError, 'more than current cash'):
            case_pairs(changed)
        with self.assertRaisesRegex(ValueError, 'Incomplete cash pair'):
            case_pairs(self.cases[:-1])

    def test_native_worker_three_models_two_gpus_and_complete_pair_report(self):
        from training.buyer_diagnostics import worker
        from training.cash_decision_report import write_report
        from training.buyer_operations_data import personas
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        class FakeClient:
            def __init__(self, **kwargs): pass
            def complete(self, **kwargs):
                ChatCompletionRequest.model_validate({k: kwargs[k] for k in ('model', 'messages', 'tools')})
                obs = json.loads(kwargs['messages'][-1]['content'])
                state = {'bad': [], 'pending': None, 'last_buy': None}
                if obs.get('offers_to_you'):
                    offer = obs['offers_to_you'][0]
                    tool, args = 'accept_offer', {'offer_id': offer['offer_id']}
                    price = offer['price']*offer['qty']
                else:
                    row = obs['board'][0]
                    tool, args = 'buy_listed', {'seller_id': row['seller'], 'product_id': row['product_id'], 'qty': 1}
                    price = row['price']
                    state['last_buy'] = [row['seller'], row['product_id']]
                model = kwargs['model']
                if 'v21_control' in model or ('candidate' in model and obs['you']['cash'] < price):
                    tool, args = 'wait', {}; state['last_buy'] = None
                args['private_state'] = json.dumps(state)
                action = SimpleNamespace(name=tool, arguments=args, raw_arguments=json.dumps(args), arguments_error=None)
                return SimpleNamespace(tool_call=action, stop_reason='tool_calls', outcome='accepted', error=None, text='')
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_v22_worker_cpu_') as temporary:
            folder = Path(temporary)
            plan = {'devices': [{'index': str(i), 'fraction': .8} for i in (0, 1)],
                    'port': 8000, 'seeds': [0, 1], 'analysis': 'cash_decision',
                    'personas': {f'operations_p{i+1}': p for i, p in enumerate(personas())},
                    'adapters': {k: '/unused/fake_adapter' for k in ('v2_step8', 'v21_control', 'candidate')},
                    'calls': 576}
            (folder/'plan.json').write_text(json.dumps(plan))
            (folder/'cases.json').write_text(json.dumps(self.cases))
            with patch('adlmarket.serving.VLLMCluster', return_value=nullcontext()), \
                 patch('adlmarket.serving.register_adapters'), patch('adlmarket.client.ModelClient', FakeClient), \
                 redirect_stdout(io.StringIO()):
                worker(folder)
            report = json.loads((folder/'cash_decision_pairs.json').read_text())
            self.assertEqual(report['calls'], 576)
            self.assertEqual(report['gpu_calls'], {'0': 288, '1': 288})
            self.assertEqual(report['models']['candidate']['overall']['both_correct'], 96)
            self.assertEqual(report['models']['candidate']['channel']['offer']['both_correct'], 48)
            self.assertEqual(report['models']['candidate']['overall']['unaffordable'], 0)
            self.assertEqual(report['models']['v21_control']['overall']['wait_both_sides'], 96)
            self.assertEqual(report['models']['v21_control']['overall']['both_correct'], 0)
            self.assertEqual(report['models']['v2_step8']['overall']['same_tool_both_sides'], 96)
            self.assertEqual(report['models']['v2_step8']['overall']['unaffordable'], 96)
            self.assertEqual(json.loads((folder/'evaluation_status.json').read_text())['status'], 'complete')
            scored = folder/'scored_decisions.jsonl'
            original = scored.read_text(); rows = [json.loads(x) for x in original.splitlines()]
            rows[0]['input_sha256'] = 'tampered'
            scored.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'different prompts/hardware'):
                write_report(folder)
            scored.write_text('\n'.join(original.splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete cash evaluation'):
                write_report(folder)


if __name__ == '__main__':
    unittest.main()
