"""CPU-only format/field isolation and diagnostic orchestration verification."""
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


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.diagnose_self_check_entry import prepare, DEFAULT
        from transformers import AutoTokenizer
        cls.plan = prepare(DEFAULT)
        cls.t = AutoTokenizer.from_pretrained(cls.plan['base_model'], revision=cls.plan['revision'], local_files_only=True)

    def test_prefix_is_only_structure_original_inputs_and_same_routes(self):
        from training.diagnose_self_check_entry import PREFIX, make_prompt
        self.assertEqual(PREFIX, '{"cash":')
        self.assertEqual(Counter(j['rank'] for j in self.plan['jobs']), {0: 12, 1: 12})
        self.assertEqual(Counter(c['positive'] for c in self.plan['cases']), {True: 6, False: 6})
        pairs = defaultdict(set)
        for c in self.plan['cases']:
            before = copy.deepcopy(c); prompt, prefill = make_prompt(self.t, c)
            self.assertEqual(c, before); self.assertEqual(prefill, prompt+PREFIX)
            self.assertTrue(prompt.endswith('<think>\n')); self.assertNotIn('</think>', prefill[len(prompt):])
            self.assertFalse(any(s in prefill[len(prompt):] for s in ('1959', 'eligible', 'held', 'buy_listed', 'o1')))
            self.assertEqual({j['rank'] for j in self.plan['jobs'] if j['case']==c['name']}, {c['evaluation_rank']})
            pairs[c['pair_id']].add(c['evaluation_rank'])
        self.assertTrue(all(len(p)==1 for p in pairs.values()))

    def test_cash_probe_separates_punctuation_value_and_units(self):
        from training.diagnose_self_check_entry import cash_probe
        c = next(c for c in self.plan['cases'] if c['observation']['you']['cash']==19.59)
        result = cash_probe(c, '{"cash":1959, BROKEN JSON\n</think>')
        self.assertTrue(result['cents_correct']); self.assertFalse(result['dollars_only'])
        result = cash_probe(c, '{"cash":19.59, BROKEN JSON\n</think>')
        self.assertTrue(result['dollars_only']); self.assertFalse(result['cents_correct'])
        for bad in ('P74 cash=1959\n</think>', '{"cash":true}\n</think>', '{"cash":1959oops}',
                    'No cash here\n</think>\n<tool_call>{"cash":1959}</tool_call>'):
            self.assertFalse(cash_probe(c, bad)['cents_correct'])

    def test_native_output_parser_sees_continuation_with_its_prefix(self):
        from training.diagnose_self_check_entry import PREFIX, make_prompt, score_output
        for c in self.plan['cases']:
            prompt, _ = make_prompt(self.t, c)
            full = self.t.apply_chat_template(c['messages']+[c['answer']], tools=c['tools'], tokenize=False,
                add_generation_prompt=False, enable_thinking=True)
            target = full[len(prompt):]; self.assertTrue(target.startswith(PREFIX))
            r = score_output(c, target[len(PREFIX):], self.t, False, c['evaluation_rank'])
            self.assertTrue(r['self_check']['exact']); self.assertTrue(r['cash_probe']['cents_correct'])
            self.assertTrue(r['valid']); self.assertEqual(r['supplied_prefix'], PREFIX)
            self.assertEqual(r['captured'], c['positive'])

    def test_plain_text_all_fields_and_punctuation_are_independent(self):
        from training.buyer_check_lines import render, score
        from training.buyer_self_checks import truth
        for c in self.plan['cases']:
            text = render(truth(c)); self.assertTrue(score(c, text)['exact'])
            text = text.replace(' | ', ', ').replace('=', ' = ')
            self.assertTrue(score(c, text+'\n</think>')['exact'])
        c = self.plan['cases'][0]; text = render(truth(c))
        # Corrupt one target field: cash and the other five candidate fields retain credit.
        facts = truth(c); target = next(r for r in facts['checks'] if r['source']==c['channel'] and r['product']==c['target_product'])
        from training.buyer_check_lines import key
        marker = 'ROW='+':'.join(key(target)); offset = text.index(marker)
        before, after = text[:offset], text[offset:]
        corrupted = before+after.replace('total='+str(target['total']), 'total=BAD', 1)
        r = score(c, corrupted); self.assertFalse(r['target_fields']['total']); self.assertTrue(r['cash_correct'])
        for f in ('held', 'next_value', 'adjusted_value', 'affordable', 'eligible'): self.assertTrue(r['target_fields'][f])
        missing = before+after.replace('eligible='+str(int(target['eligible'])), '', 1)
        self.assertFalse(score(c, missing)['target_fields']['eligible']); self.assertTrue(score(c, missing)['target_fields']['total'])

    def test_plain_text_identity_duplicates_and_private_state_cannot_cheat(self):
        from training.buyer_check_lines import score, render, key
        from training.buyer_self_checks import truth
        c = self.plan['cases'][0]; facts = truth(c); text = render(facts)
        target = next(r for r in facts['checks'] if r['source']==c['channel'] and r['product']==c['target_product'])
        marker = 'ROW='+':'.join(key(target))
        wrong = text.replace(marker, 'ROW='+target['source']+':wrong_id:'+target['product'])
        r = score(c, wrong); self.assertFalse(r['coverage']); self.assertFalse(r['exact']); self.assertTrue(r['cash_correct'])
        doubled = text+' | '+text.split(' | ', 1)[1]
        self.assertFalse(score(c, doubled)['coverage'])
        cash_in_tool = 'Nothing checked\n</think>\n<tool_call>CASH='+str(facts['cash'])+'</tool_call>'
        self.assertFalse(score(c, cash_in_tool)['cash_correct'])
        # Duplicate conflicting fields invalidate that field, not unrelated fields.
        corrupt = text.replace(marker, marker+' held=1', 1)
        r = score(c, corrupt); self.assertFalse(r['target_fields']['held']); self.assertTrue(r['target_fields']['total'])

    def test_prepare_and_mock_launcher_report_without_gpu_execution(self):
        from training.diagnose_self_check_entry import main, PREFIX, score_output, make_prompt, write_report
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_entry_cpu_') as tmp:
            with patch('sys.argv', ['entry', '--prepare-only', '--output-dir', tmp]), patch('training.local_resources.training_gpu_plan') as gpu, \
                 patch('training.diagnose_self_check_entry.subprocess.Popen') as process, redirect_stdout(io.StringIO()):
                main(); gpu.assert_not_called(); process.assert_not_called()
            def fake(cmd, **kwargs):
                folder = Path(cmd[-1]); plan = json.loads((folder/'plan.json').read_text())
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1'); self.assertIn('--nproc_per_node=2', cmd)
                cases = {c['name']: c for c in plan['cases']}
                for rank in (0, 1):
                    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream:
                        for job in plan['jobs']:
                            if job['rank'] != rank: continue
                            c = cases[job['case']]; prompt, _ = make_prompt(self.t, c)
                            full = self.t.apply_chat_template(c['messages']+[c['answer']], tools=c['tools'], tokenize=False, enable_thinking=True)
                            continuation = full[len(prompt)+len(PREFIX):]
                            row = score_output(c, continuation, self.t, False, rank); row['adapter'] = job['adapter']
                            stream.write(json.dumps(row)+'\n')
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock only; zero model execution\n'), wait=lambda:0)
            cards = [dict(index=str(i), fraction=.85) for i in (0, 1)]
            with patch('sys.argv', ['entry', '--output-dir', tmp]), patch('training.local_resources.training_gpu_plan', return_value=(cards, [], {})), \
                 patch('training.local_resources.runtime_cache', return_value=({}, None)), patch('training.diagnose_self_check_entry.subprocess.Popen', side_effect=fake), redirect_stdout(io.StringIO()):
                folder = main()
            report = json.loads((folder/'entry_review.json').read_text())
            for label in self.plan['adapters']:
                s = report['summary'][label]['json_open']; self.assertEqual(s['all_fields_exact'], 12)
                self.assertEqual(s['captured'], 6); self.assertEqual(s['negative_safe'], 6)
                self.assertEqual(s['cash_probe']['cents_correct'], 12)
            path = folder/'decisions_rank0.jsonl'; path.write_text(path.read_text().splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete/repeated'): write_report(folder, self.plan)


if __name__ == '__main__': unittest.main()
