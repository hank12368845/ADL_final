"""CPU-only isolation, provenance, scoring and launcher tests; no GPU inference."""
from collections import Counter
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class BaseStep84Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from training.evaluate_base_step84 import prepare
        cls.plan = prepare()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.plan['base_model'], revision=cls.plan['revision'], local_files_only=True)
        cls.source = {c['name']: c for c in json.loads((ROOT/'training/data/buyer_dollar_lines_v1/evaluation_cases.json').read_text())}

    def test_exact_original_prompts_no_teacher_and_same_case_routes(self):
        from training.evaluate_base_step84 import prompt, stats
        p = self.plan
        self.assertEqual(p['loaded_adapter'], None)
        self.assertEqual(p['training_updates'], 0)
        self.assertEqual(p['new_gpu_calls'], 36)
        self.assertEqual(Counter(c['evaluation_rank'] for c in p['cases']), {0:18, 1:18})
        self.assertEqual(p['generation']['eos_token_id'], [248044, 248046])
        for c in p['cases']:
            original = self.source[c['name']]
            self.assertNotIn('answer', c)
            self.assertNotIn('label_action', c)
            for key in ('messages', 'tools', 'situation', 'observation'):
                self.assertEqual(c[key], original[key])
            text = prompt(self.tokenizer, c)
            self.assertTrue(text.endswith('<think>\n'))
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), c['prompt_sha256'])
        s = stats(p['baseline'])
        self.assertEqual((s['valid'], s['captured'], s['negative_safe'], s['cash_errors'], s['owned_errors']), (19,13,6,7,9))
        self.assertTrue(all(r['cached'] for r in p['baseline']))

    def test_actual_loader_has_no_adapter_and_rejects_adapter_or_incomplete_weights(self):
        import torch
        from training.evaluate_base_step84 import load_bare_base
        model = torch.nn.Linear(2, 2)
        calls = []
        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return model, {}
        result = load_bare_base(self.plan, 'cpu', loader)
        self.assertIs(result, model)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(calls[0][1]['attn_implementation'], 'sdpa')
        self.assertEqual(calls[0][1]['revision'], self.plan['revision'])
        self.assertNotIn('adapter', str(calls))
        model.peft_config = {'default': 'unexpected'}
        with self.assertRaisesRegex(ValueError, 'unexpectedly contains'):
            load_bare_base(self.plan, 'cpu', loader)
        with self.assertRaisesRegex(ValueError, 'Incomplete base'):
            load_bare_base(self.plan, 'cpu', lambda *a, **k: (model, {'missing_keys':['weight']}))

    def golden_rows(self, folder, waiting=False):
        from training.evaluate_base_step84 import prompt, assess_case
        p = json.loads((folder/'plan.json').read_text())
        for rank in (0, 1):
            with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream:
                for c in p['cases']:
                    if c['evaluation_rank'] != rank:
                        continue
                    if waiting:
                        raw = '</think>\n<tool_call>\n<function=wait>\n</function>\n</tool_call>'
                    else:
                        start = prompt(self.tokenizer, c)
                        full = self.tokenizer.apply_chat_template(c['messages']+[self.source[c['name']]['answer']],
                            tools=c['tools'], tokenize=False, enable_thinking=True)
                        self.assertTrue(full.startswith(start))
                        raw = full[len(start):]
                    r = assess_case(c, raw, self.tokenizer, False, rank, False, 'base')
                    self.assertTrue(r['valid'])
                    r.update(prompt_sha256=c['prompt_sha256'], loaded_adapter=None, new_generated_tokens=10)
                    stream.write(json.dumps(r)+'\n')

    def test_prepare_launch_report_and_evidence_preserve_originals(self):
        from training.evaluate_base_step84 import main, write_report
        paths = [ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)]
        paths += [ROOT/'agents/example_student/buyer/checkpoint/adapter_model.safetensors']
        before = {p: p.read_bytes() for p in paths}
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='adl_base_step84_cpu_') as tmp:
            with patch('sys.argv', ['control', '--prepare-only', '--output-dir', tmp]), \
                 patch('training.local_resources.training_gpu_plan') as gpu, \
                 patch('training.evaluate_base_step84.subprocess.Popen') as proc, redirect_stdout(io.StringIO()):
                prepared = main()
                gpu.assert_not_called()
                proc.assert_not_called()
            self.assertTrue((prepared/'plan.json').is_file())
            def fake_launch(cmd, **kwargs):
                self.assertIn('--nproc_per_node=2', cmd)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1')
                self.golden_rows(Path(cmd[-1]))
                return SimpleNamespace()
            cards = [dict(index=str(i), fraction=.85) for i in (0,1)]
            with patch('sys.argv', ['control', '--output-dir', tmp]), \
                 patch('training.local_resources.training_gpu_plan', return_value=(cards, [], {})), \
                 patch('training.local_resources.runtime_cache', return_value=({}, None)), \
                 patch('training.evaluate_base_step84.subprocess.Popen', side_effect=fake_launch), \
                 patch('training.evaluate_base_step84.stream_progress', return_value=0), redirect_stdout(io.StringIO()):
                folder = main()
            result = json.loads((folder/'base_step84_review.json').read_text())
            self.assertEqual(result['summary']['base']['total']['captured'], 18)
            self.assertEqual(result['summary']['base']['paired']['both_correct'], 18)
            self.assertEqual(result['summary']['step84_cached']['total']['valid'], 19)
            self.assertFalse(result['automatic_next_gpu_work'])
            self.assertIsNone(result['selected_market_model'])
            self.assertEqual(result['semantic_review'], 'pending')
            self.assertEqual(len(json.loads((folder/'semantic_review.json').read_text())), 72)
            self.golden_rows(folder, waiting=True)
            wait_result = write_report(folder, self.plan)
            self.assertEqual(wait_result['summary']['base']['total']['valid'], 36)
            self.assertEqual(wait_result['summary']['base']['total']['captured'], 0)
            self.assertEqual(wait_result['summary']['base']['paired']['both_correct'], 0)
            # Good-looking all-wait validity must not select a winner automatically.
            self.assertIsNone(wait_result['selected_market_model'])
            path = folder/'decisions_rank0.jsonl'
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            rows[0]['prompt_sha256'] = 'bad'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'prompt SHA'):
                write_report(folder, self.plan)
            path.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete/repeated'):
                write_report(folder, self.plan)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_source_or_cached_package_change_rejected(self):
        from training.evaluate_base_step84 import verify_files
        changed = copy.deepcopy(self.plan)
        changed['sources']['training/evaluate_base_step84.py'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Source/evidence changed'):
            verify_files(changed)
        changed = copy.deepcopy(self.plan)
        changed['versions']['torch'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Package version differs'):
            verify_files(changed)

    def test_loading_progress_and_stdout_are_saved_without_a_model(self):
        from training.evaluate_base_step84 import stream_progress
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp, redirect_stdout(io.StringIO()) as visible:
            proc = subprocess.Popen([sys.executable, '-u', '-c', 'print("CPU child, no model")'], stdout=subprocess.PIPE)
            log = io.StringIO()
            try:
                code = stream_progress(proc, Path(tmp), log)
            finally:
                proc.stdout.close()
            self.assertEqual(code, 0)
            self.assertIn('0/36', visible.getvalue())
            self.assertIn('[------------------------]', visible.getvalue())
            self.assertIn('ETA estimating', visible.getvalue())
            self.assertIn('CPU child, no model', log.getvalue())


if __name__ == '__main__':
    unittest.main()
