"""CPU verification of labels, actual supervised spans, metrics and launch boundaries."""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'training/data/buyer_self_checks_v1'


class SelfCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from transformers import AutoTokenizer
        from training.train_buyer_self_checks import prepare
        cls.cfg, cls.manifest, cls.batches = prepare(yaml.safe_load((ROOT/'training/configs/buyer_self_checks_v1.yaml').read_text()))
        cls.train = [json.loads(l) for l in (DATA/'train.jsonl').read_text().splitlines()]
        cls.val = [json.loads(l) for l in (DATA/'validation.jsonl').read_text().splitlines()]
        cls.tokens = [json.loads(l) for l in (DATA/'train.tokens.jsonl').read_text().splitlines()]
        cls.cases = json.loads((DATA/'evaluation_cases.json').read_text())
        cls.t = AutoTokenizer.from_pretrained(cls.cfg['base_model'], revision=cls.cfg['revision'], local_files_only=True)

    def test_frozen_latest_unmodified_observation_and_group_split(self):
        self.assertFalse(set(self.manifest['groups']['train']) & set(self.manifest['groups']['validation']))
        source = {json.loads(l)['id']: json.loads(l) for l in (ROOT/'training/data/buyer_latest_brakes_v1/train.jsonl').read_text().splitlines()}
        for row in self.train:
            original = source[row['id']]
            self.assertEqual(row['messages'], original['messages']); self.assertEqual(row['observation'], original['observation'])
            self.assertEqual(row['tools'], original['tools'])
            self.assertEqual(row['answer']['tool_calls'], original['answer']['tool_calls'])
            persona = row['messages'][0]['content'].split('<<<PERSONA_SECTION>>>')[1].split('<<<END_PERSONA_SECTION>>>')[0].strip()
            self.assertEqual(persona, (ROOT/f'agents/example_student/buyer/persona_{row["persona_index"]}.md').read_text().strip())
            body = json.loads(row['messages'][-1]['content'])
            for key in ('you', 'board', 'offers_to_you'):
                self.assertEqual(body[key], row['observation'][key])
            self.assertNotIn('eligible', body); self.assertNotIn('answer', body)

    def test_every_role_channel_cent_zero_owned_value_and_full_epochs(self):
        from training.buyer_self_checks import truth, target_check
        for p in (1, 2, 3):
            for channel in ('board', 'offer'):
                rows = [r for r in self.train if r['persona_index'] == p and r['channel'] == channel and not r['positive']]
                self.assertEqual(len(rows), 4)
                self.assertTrue(any(truth(r)['cash'] == 0 for r in rows))
                self.assertTrue(any(target_check(r, truth(r))['total']-truth(r)['cash'] == 1 for r in rows))
                self.assertTrue(any(r['mode'] == 'owned' for r in rows))
                self.assertTrue(any(r['mode'] == 'value' for r in rows))
                self.assertTrue(any(truth(r)['cash'] == target_check(r, truth(r))['total'] for r in self.train if r['persona_index'] == p and r['channel'] == channel and r['positive']))
        for start in (0, 6):
            self.assertEqual(Counter(i for b in self.batches[start:start+6] for i in b), Counter(range(48)))
        for b in self.batches:
            self.assertEqual(Counter(self.train[i]['label_action'] for i in b), {'buy_listed': 2, 'accept_offer': 2, 'wait': 4})
            pairs = defaultdict(list)
            for i in b: pairs[self.train[i]['pair_id']].append(self.train[i])
            self.assertEqual(len(pairs), 4)
            self.assertTrue(all({r['positive'] for r in p} == {True, False} for p in pairs.values()))

    def test_truth_matches_engine_evidence_and_current_not_initial_value(self):
        from training.buyer_self_checks import truth, target_check
        from training.official_buyer_train import official_snapshot_review
        for row in self.train+self.val:
            facts = truth(row); target = target_check(row, facts)
            self.assertEqual(target['eligible'], row['positive'])
            self.assertEqual(sum(r['eligible'] for r in facts['checks']), int(row['positive']))
            fn = row['answer']['tool_calls'][0]['function']
            self.assertTrue(official_snapshot_review(row, fn['name'], fn['arguments'])['valid_snapshot'])
            if not row['positive'] and row['mode'] == 'cash':
                self.assertEqual(row['negative_trade_check']['result']['error'], 'insufficient_cash')
            if not row['positive'] and row['mode'] == 'owned':
                self.assertTrue(target['held']); self.assertEqual(target['next_value'], 0)
                self.assertGreater(row['situation']['your_utility_per_unit'][row['target_product']][0], 0)

    def test_native_prompt_target_spans_eos_and_real_tool_parser(self):
        from training.buyer_hf_check import parse_raw
        from training.buyer_self_checks import score, FIELDS
        for row, token in zip(self.train, self.tokens):
            prompt = self.t.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            self.assertEqual(self.t.encode(prompt, add_special_tokens=False), token['input_ids'][:token['prompt_length']])
            tail = self.t.decode(token['input_ids'][token['prompt_length']:], skip_special_tokens=False)
            self.assertTrue(score(row, tail)['exact'])
            parsed = parse_raw(self.t, tail, row['tools'])
            self.assertEqual(parsed['parsed_calls'], 1)
            self.assertEqual(parsed['arguments'], row['answer']['tool_calls'][0]['function']['arguments'])
            self.assertTrue(tail.rstrip().endswith('<|im_end|>'))
            self.assertEqual({s['field'] for s in token['field_spans']}, set(FIELDS))
            self.assertLessEqual(token['reasoning_tokens'], 200)
            self.assertLessEqual(len(token['input_ids']), 12288)
            for span in token['field_spans']:
                self.assertTrue(span['positions'])
                value = self.t.decode([token['input_ids'][token['prompt_length']+i] for i in span['positions']]).strip()
                self.assertEqual(json.loads(value), span['value'])
                self.assertTrue(all(i < token['reasoning_tokens'] for i in span['positions']))

    def test_wrong_fields_ids_missing_coverage_and_malformed_do_not_pass(self):
        from training.buyer_self_checks import score, truth, FIELDS
        row = self.train[0]; facts = truth(row)
        self.assertTrue(score(row, json.dumps(facts)+'\n</think>\n')['exact'])
        for field in FIELDS:
            bad = copy.deepcopy(facts)
            if field == 'cash': bad['cash'] += 1
            else:
                target = next(r for r in bad['checks'] if r['product'] == row['target_product'] and r['source'] == row['channel'])
                target[field] = not target[field] if type(target[field]) is bool else target[field]+1
            result = score(row, json.dumps(bad)+'\n</think>\n')
            self.assertFalse(result['target_fields'][field]); self.assertFalse(result['exact'])
        for mutate in (lambda f: f['checks'].pop(), lambda f: f['checks'].append(f['checks'][0]),
            lambda f: f['checks'][0].update(product='wrong_id'), lambda f: f.update(cash=True)):
            bad = copy.deepcopy(facts); mutate(bad)
            self.assertFalse(score(row, json.dumps(bad)+'\n</think>\n')['exact'])
        for raw in ('', 'cash=100', json.dumps(facts), '[]\n</think>', '{bad}\n</think>'):
            self.assertFalse(score(row, raw)['exact'])

    def test_tools_independent_of_field_format_and_summary_denominators(self):
        from training.buyer_self_checks import score, summarize
        from training.step84_repair_plan import assess
        rows = []
        pair = self.train[:2]
        self.assertEqual({r['positive'] for r in pair}, {True, False})
        for row in pair:
            token = next(t for t in self.tokens if t['id'] == row['id'])
            tail = self.t.decode(token['input_ids'][token['prompt_length']:], skip_special_tokens=False)
            good = assess(row, 'latest', tail, self.t, False, 0); good['self_check'] = score(row, tail)
            self.assertTrue(good['valid']); rows.append(good)
            # A non-JSON reason must fail our field metric without inventing an engine violation.
            alternate = 'I checked the current state.\n</think>'+tail.split('</think>', 1)[1]
            self.assertTrue(assess(row, 'latest', alternate, self.t, False, 0)['valid'])
            self.assertFalse(score(row, alternate)['exact'])
        result = summarize(rows)
        self.assertEqual(result['all_fields_exact'], 2); self.assertEqual(result['pairs'], 1)
        self.assertEqual(result['both_fields_and_tool_correct_pairs'], 1)
        self.assertEqual(result['target_fields']['eligible'], 2)
        self.assertEqual(result['all_candidate_fields']['eligible']['total'], 6)

    def test_semantic_loss_matches_manual_alignment_and_backprop_cpu_only(self):
        import torch
        from training.buyer_self_checks import losses, FIELDS
        # Tiny logits fixture validates shifted labels and actual supervised-value gradients;
        # this is not HF/base-model inference or a GPU job.
        logits = torch.nn.Parameter(torch.zeros(1, 8, 5))
        with torch.no_grad(): logits[0, 0, 2] = 1.5
        class Tiny:
            def __call__(self, **kwargs):
                self_ids = kwargs['input_ids']
                if kwargs['logits_to_keep'].tolist() != list(range(1, 9)):
                    raise AssertionError('Wrong autoregressive alignment')
                return SimpleNamespace(logits=logits)
        row = dict(input_ids=[0, 1, 2, 1, 2, 3, 4, 0, 1, 2], prompt_length=2,
            field_spans=[dict(field=f, target=True, positions=[i]) for i, f in enumerate(FIELDS)])
        got = losses(Tiny(), row, torch.device('cpu'))
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), torch.tensor(row['input_ids'][2:]), reduction='none')
        self.assertTrue(torch.allclose(got['total'], ce[:7].mean()+.25*ce.mean()))
        got['total'].backward(); self.assertTrue(torch.isfinite(logits.grad).all()); self.assertGreater(logits.grad.abs().sum(), 0)

    def test_seen_training_identity_and_fixed_pair_gpu_routes(self):
        original = {r['id']: r for r in self.train}; ranks = defaultdict(set)
        for c in self.cases:
            if c['suite'] == 'self_check_seen': self.assertEqual(c['messages'], original[c['id']]['messages'])
            if c['suite'].startswith('self_check_'): ranks[c['pair_id']].add(c['evaluation_rank'])
        self.assertTrue(all(len(r) == 1 for r in ranks.values()))
        self.assertEqual(Counter(c['evaluation_rank'] for c in self.cases), {0: 37, 1: 36})
        self.assertEqual(sum('cached_parent' in c for c in self.cases), 37)

    def test_cpu_and_two_gpu_wrapper_without_starting_gpu_work(self):
        from training.train_buyer_self_checks import main
        with patch('sys.argv', ['checks', '--cpu-check']), patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_self_checks.subprocess.Popen') as process, redirect_stdout(io.StringIO()):
            main(); gpu.assert_not_called(); process.assert_not_called()
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_checks_cpu_') as tmp:
            def fake(cmd, **kwargs):
                self.assertIn('--nproc_per_node=2', cmd)
                cfg = json.loads((Path(cmd[-1])/'train_config.json').read_text())
                self.assertEqual([c['index'] for c in cfg['devices']], ['0', '1'])
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1')
                return SimpleNamespace(stdout=io.BytesIO(b'CPU subprocess fixture only\n'), wait=lambda: 0)
            cards = [dict(index=str(i), budget_gib=20, fraction=.85) for i in (0, 1)]
            with patch('sys.argv', ['checks', '--output-dir', tmp]), patch('training.local_resources.training_gpu_plan', return_value=(cards, [], {})), \
                 patch('training.local_resources.runtime_cache', return_value=({}, None)), \
                 patch('training.train_buyer_self_checks.subprocess.Popen', side_effect=fake), redirect_stdout(io.StringIO()):
                folder = main()
            self.assertEqual(json.loads((folder/'wrapper_result.json').read_text())['exit'], 0)

    def test_field_first_gate_still_rejects_all_wait_and_correct_fields_wrong_tool(self):
        from training.buyer_self_checks import summarize as field_summary, score, truth
        from training.buyer_self_checks_train import gates, summarize, write_review
        from training.step84_repair_plan import assess
        rows = []
        for case in self.cases:
            if case['suite'].startswith('self_check_'):
                prompt = self.t.apply_chat_template(case['messages'], tools=case['tools'], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                full = self.t.apply_chat_template(case['messages']+[case['answer']], tools=case['tools'], tokenize=False, add_generation_prompt=False, enable_thinking=True)
                raw = full[len(prompt):]
                row = assess(case, 'latest', raw, self.t, False, case['evaluation_rank'])
                row['self_check'] = score(case, raw)
            else:
                raw = case['cached_parent']['text']
                row = assess(case, 'latest', raw, self.t, case['cached_parent']['truncated'], case['evaluation_rank'], True)
            row['persona_index'] = case['persona_index']; rows.append(row)
        parent = summarize(rows); good = copy.deepcopy(parent)
        # Existing cached parent has one truncated case: the full gate must expose it.
        self.assertTrue(gates(parent, good)['seen_all_fields_exact'])
        self.assertTrue(gates(parent, good)['validation_target_fields_exact'])
        self.assertFalse(gates(parent, good)['no_missing_tool_or_truncation'])
        bad = copy.deepcopy(good); bad['self_check_seen']['captured'] = 0
        self.assertFalse(gates(parent, bad)['retain_positive_capture'])
        bad = copy.deepcopy(good); bad['self_check_validation']['target_fields']['cash'] -= 1
        self.assertFalse(gates(parent, bad)['validation_target_fields_exact'])
        wrong = copy.deepcopy([r for r in rows if r['suite']=='self_check_seen'])
        wrong[0]['valid'] = False
        s = field_summary(wrong)
        self.assertEqual(s['all_fields_exact'], 12); self.assertEqual(s['tool_valid_when_fields_exact'], 11)
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_checks_report_cpu_') as tmp:
            folder = Path(tmp)
            (folder/'native_step_0000.json').write_text(json.dumps(dict(step=0, summary=parent)))
            write_review(folder)
            report = (folder/'self_check_review.md').read_text()
            self.assertIn('Candidate coverage', report); self.assertIn('personas:3', report)


if __name__ == '__main__':
    unittest.main()
