"""CPU regression checks; mocked outputs are not evidence about the real LoRA."""
import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.buyer_history_ablation import build_cases, write_report, CONTEXTS
from training.buyer_diagnostics import replace_persona, worker
from training.buyer_operations_data import make_record, personas, choose


class HistoryAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='adl_history_cpu_', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        manifest = {'seed': 20261118, 'groups': {'validation': [76, 41], 'train': [0], 'test': [73]}}
        self.sources = [make_record(manifest['seed']+1009*g, task, 0)
                        for g in manifest['groups']['validation'] for task in ('stale_memory', 'binding_offer')]
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        (self.folder/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.sources))
        self.cases, self.manifest = build_cases(self.folder)

    def test_engine_verified_positive_negative_and_split_boundaries(self):
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(self.manifest['calls'], 144)
        self.assertNotIn(20261118+1009*73, self.manifest['groups'])
        for case in self.cases:
            self.assertTrue(case['engine_result']['ok'])
            if case['task'].endswith('positive'):
                source = next(r for r in self.sources if r['id']==case['origin']['source_id'])
                self.assertEqual(case['observation'], source['observation'])
                self.assertIn(case['expected_tool'], ('buy_listed', 'accept_offer'))
            else:
                self.assertEqual(case['expected_tool'], 'wait')
                negative = case['negative_trade_check']
                if case['task']=='board_cash_negative':
                    self.assertEqual(negative['result']['error'], 'insufficient_cash')
                else:
                    self.assertTrue(negative['result']['ok'])
                    self.assertEqual(negative['realized_surplus_cents'], -1)
            body = json.loads(case['messages'][-1]['content'])
            self.assertNotIn('expected_tool', body)
            self.assertNotIn('negative_trade_check', body)
        manifest = json.loads((self.folder/'manifest.json').read_text())
        manifest['groups']['test'] = [76]
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            build_cases(self.folder)

    def test_context_changes_only_history_and_optional_note_and_http_schema(self):
        from training.chat_payload import api_messages, validate_payloads
        untouched = copy.deepcopy(self.cases)
        for case in self.cases:
            messages = {ctx: replace_persona(case['messages'], personas()[0], ctx) for ctx in CONTEXTS}
            self.assertGreater(len(messages['recorded']), 2)
            self.assertEqual(len(messages['fresh']), 2)
            self.assertEqual(len(messages['clean_fresh']), 2)
            before = json.loads(messages['fresh'][-1]['content'])
            after = json.loads(messages['clean_fresh'][-1]['content'])
            self.assertEqual({**before, 'your_private_state': ''}, after)
            self.assertEqual(messages['recorded'][0], messages['fresh'][0])
            self.assertEqual(messages['recorded'][-1], messages['fresh'][-1])
            if not case['private_state_present']:
                self.assertEqual(api_messages(messages['fresh']), api_messages(messages['clean_fresh']))
            else:
                self.assertNotEqual(before['your_private_state'], '')
        self.assertEqual(self.cases, untouched)
        self.assertEqual(validate_payloads(self.cases, {'operations_p1': personas()[0]})['validated_requests'], 24)

    def test_complete_matrix_routes_pairs_and_reports_distinguish_history_memory_and_weights(self):
        from adlmarket.setups import market_run
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        root = Path(__file__).resolve().parents[1]
        run = market_run(root/'configs/basic_buyer.yaml', root=root)
        adapter = run.models[run.buyers[0].model].lora.path
        plan = {'analysis': 'history_ablation', 'calls': 144, 'port': 8000,
                'devices': [{'index': '0', 'fraction': .8}, {'index': '1', 'fraction': .8}],
                'seeds': [0, 1], 'personas': {'operations_p1': personas()[0]},
                'adapters': {'base': None, 'step4': adapter, 'step8': adapter}}
        cases = self.cases
        def observation_key(body):
            return json.dumps({k: v for k, v in body.items() if k not in ('your_private_state', 'last_action_result')}, sort_keys=True)
        by_observation = {observation_key(c['observation']): c for c in cases}
        calls = []
        class FakeClient:
            def __init__(self, **kwargs): pass
            def complete(self, **kwargs):
                ChatCompletionRequest.model_validate({'model': kwargs['model'], 'messages': kwargs['messages'], 'tools': kwargs['tools']})
                body = json.loads(kwargs['messages'][-1]['content'])
                case = by_observation[observation_key(body)]
                tool, args = choose(case['situation'], case['observation'])
                # Deliberate synthetic sensitivity, solely to verify the report's attribution math.
                if 'step8' in kwargs['model'] and (len(kwargs['messages'])>2 or body['your_private_state']):
                    tool, args = 'wait', {'private_state': json.dumps({'bad': [], 'pending': None, 'last_buy': None})}
                calls.append(kwargs['base_url'])
                action = SimpleNamespace(name=tool, arguments=args, raw_arguments=json.dumps(args), arguments_error=None)
                return SimpleNamespace(tool_call=action, stop_reason='tool_calls', outcome='accepted', error=None, text='')
        (self.folder/'plan.json').write_text(json.dumps(plan))
        (self.folder/'cases.json').write_text(json.dumps(cases))
        with patch('adlmarket.serving.VLLMCluster', return_value=nullcontext()), \
             patch('adlmarket.serving.register_adapters'), patch('adlmarket.client.ModelClient', FakeClient), \
             redirect_stdout(io.StringIO()):
            worker(self.folder)
        self.assertEqual(len(calls), 144)
        self.assertEqual(calls.count('http://localhost:8000/v1'), 72)
        self.assertEqual(calls.count('http://localhost:8001/v1'), 72)
        routes = json.loads((self.folder/'job_routes.json').read_text())
        self.assertEqual([r['gpu'] for r in routes[:4]], ['0', '1', '0', '1'])
        report = json.loads((self.folder/'history_ablation.json').read_text())
        self.assertEqual(report['calls'], 144)
        self.assertTrue(report['valid_evaluation'])
        for context in CONTEXTS:
            self.assertEqual(report['table']['base'][context]['captured_opportunities'], 8)
            self.assertEqual(report['table']['step8'][context]['calls'], 16)
        self.assertEqual(report['table']['step8']['recorded']['captured_opportunities'], 0)
        self.assertIsNone(report['table']['step8']['recorded']['trade_precision'])
        self.assertEqual(report['table']['step8']['fresh']['captured_opportunities'], 4)
        self.assertEqual(report['table']['step8']['clean_fresh']['captured_opportunities'], 8)
        effects = report['paired_context_effects']['step8']
        self.assertEqual(effects['recorded_to_fresh']['matched_pairs'], 16)
        self.assertEqual(effects['recorded_to_fresh']['fixed'], 4)
        self.assertEqual(effects['fresh_to_clean_fresh']['matched_pairs'], 8)
        self.assertEqual(effects['fresh_to_clean_fresh']['fixed'], 4)
        self.assertEqual(report['paired_checkpoint_effects']['step4_to_step8']['recorded']['regressed'], 8)
        for control in report['blank_note_identity_controls'].values():
            self.assertEqual(control['same_input_pairs'], 8)
            self.assertEqual(control['tool_disagreements'], 0)
        scored_path = self.folder/'scored_decisions.jsonl'
        rows = [json.loads(line) for line in scored_path.read_text().splitlines()]
        original = scored_path.read_text()
        scored_path.write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
        with self.assertRaisesRegex(ValueError, 'Missing'):
            write_report(self.folder)
        scored_path.write_text(original)
        rows[0]['input_sha256'] = 'changed'
        scored_path.write_text('\n'.join(json.dumps(r) for r in rows))
        with self.assertRaisesRegex(ValueError, 'changed input'):
            write_report(self.folder)


if __name__ == '__main__':
    unittest.main()
