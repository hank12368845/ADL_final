"""CPU checks for symmetric cash/ID supervision and real user entry points."""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'training/data/buyer_symmetric_repair_v1'


class SymmetricRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.cfg = yaml.safe_load((ROOT/'training/configs/buyer_symmetric_repair_v1.yaml').read_text())
        cls.rows = [json.loads(s) for s in (DATA/'train.jsonl').read_text().splitlines()]

    def test_engine_labels_cash_only_pairs_and_identical_holdouts(self):
        from training.buyer_operations_v2_data import diagnostic_case
        from training.cash_decision_report import case_pairs
        from training.decision_ablation import digest
        manifest = json.loads((DATA/'manifest.json').read_text())
        allowed = {manifest['seed']+1009*g for g in manifest['groups']['train']}
        cash_cases = []
        for r in self.rows:
            self.assertIn(r['group'], allowed); self.assertTrue(r['engine_result']['ok'])
            if r['positive']:
                self.assertGreater(r['realized_surplus_cents'], 0)
            else:
                bad = r['negative_trade_check']
                self.assertTrue(not bad['result']['ok'] or bad['realized_surplus_cents'] < 0)
            if r['family'].endswith('_cash') and r['cash_variant'] in ('equal','above_one','wide','zero'):
                c = diagnostic_case(r)
                c['origin'].update({k: r[k] for k in ('cash_variant', 'history_mode')})
                cash_cases.append(c)
        self.assertEqual(len(case_pairs(cash_cases)), 96)
        for split in ('validation', 'test'):
            for suffix in ('jsonl', 'tokens.jsonl'):
                self.assertEqual(digest(DATA/f'{split}.{suffix}'), digest(ROOT/'training/data/buyer_reasoning_v1'/f'{split}.{suffix}'))
        self.assertEqual(digest(DATA/'cash_probe_cases.json'), digest(ROOT/'training/data/buyer_reasoning_v1/cash_probe_cases.json'))
        for p in (1, 2, 3):
            self.assertLessEqual(len((DATA/'personas'/f'persona_{p}.md').read_text()), 8000)

    def test_shared_reasoning_cash_boundaries_and_id_membership(self):
        from training.symmetric_repair_data import symmetric_target
        counts = Counter()
        for r in self.rows:
            text, action = symmetric_target(r)
            self.assertEqual(text, r['answer']['reasoning_content']); self.assertEqual(action, r['label_action'])
            self.assertTrue(text.startswith('SOURCES board=')); self.assertIn('check BOTH', text)
            o = r['observation']; board, offers = o['board'], o['offers_to_you']
            args = r['answer']['tool_calls'][0]['function']['arguments']
            if r['positive']:
                current = (board or offers)[0]
                self.assertIn(current['seller'], text); self.assertIn(current['product_id'], text)
                self.assertIn('C>=P=true', text)
                if offers:
                    self.assertFalse(board)
                    self.assertEqual(args['offer_id'], current['offer_id'])
                else:
                    self.assertEqual((args['seller_id'], args['product_id']), (current['seller'], current['product_id']))
            if r['family'].endswith('_cash') and r['cash_variant'] in ('equal','above_one','wide','zero'):
                counts[r['family'], r['cash_variant'], r['persona_index'], r['history_mode'], r['positive']] += 1
                self.assertIn(f'C>=P={str(r["positive"]).lower()}', text)
            if r['family']=='board_id' or 'wrong_seller_engine_check' in r:
                self.assertFalse(r['wrong_seller_engine_check']['ok'])
                self.assertFalse(r['notice_as_offer_engine_check']['ok'])
        self.assertEqual(len(counts), 96)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(Counter(r['current_seller_is_historical'] for r in self.rows if 'current_seller_is_historical' in r), {True: 16, False: 16})
        # A zero price difference is accepted; a one-cent deficit never is.
        for r in self.rows:
            if r['family'].endswith('_cash') and r['cash_variant']=='equal':
                self.assertIn('C-P=0.00' if r['positive'] else 'C-P=-0.01', r['answer']['reasoning_content'])

    def test_balanced_prefix_and_full_two_gpu_coverage_and_masks(self):
        from training.buyer_operations_train import check_data, sampling_components
        train, val = check_data(self.cfg, ROOT)
        self.assertEqual((len(train), len(val)), (256, 384))
        from training.symmetric_repair_sampling import paired_batches, PairedActionSampler
        batches = paired_batches(train.metadata, 32, self.cfg['seed'])
        self.assertEqual(batches[:16], paired_batches(train.metadata, 16, self.cfg['seed']))
        self.assertEqual(Counter(i for b in batches for i in b), Counter(range(256)))
        self.assertEqual(Counter(r['history_mode'] for r in self.rows), {'fresh':128, 'recorded':128})
        for p, n in ((1,20), (2,22), (3,22)):
            for action in ('buy_listed', 'accept_offer'):
                self.assertEqual(sum(r['persona_index']==p and r['label_action']==action for r in self.rows), n)
        for b in batches:
            self.assertEqual(Counter(train.metadata[i]['label_action'] for i in b), {'buy_listed':2, 'accept_offer':2, 'wait':4})
            self.assertEqual(sorted(Counter(train.metadata[i]['pair_id'] for i in b).values()), [2]*4)
        for action in ('buy_listed','accept_offer'):
            prefix = Counter(train.metadata[i]['persona_index'] for b in batches[:16] for i in b if train.metadata[i]['label_action']==action)
            self.assertLessEqual(max(prefix.values())-min(prefix.values()), 2)
        ranks = [list(PairedActionSampler(train.metadata,32,self.cfg['seed'],rank,2)) for rank in (0,1)]
        self.assertEqual(set(ranks[0]) & set(ranks[1]), set())
        self.assertEqual(len(ranks[0]), 128); self.assertEqual(len(ranks[1]), 128)
        self.assertEqual([ranks[r][m] for start in range(0,128,4) for m in range(start,start+4) for r in (0,1)], [i for b in batches for i in b])
        for i in range(256):
            row = train[i]
            self.assertLess(max(row['reasoning_positions']), min(row['tool_name_positions']))
            self.assertLessEqual(len(row['reasoning_positions']),192)

    def test_cpu_entry_then_mock_ddp_and_evaluation_without_gpu(self):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from training.train_buyer_operations import main
        from training.evaluate_symmetric_repair import resolve_run, main as evaluate
        cfg_path = ROOT/'training/configs/buyer_symmetric_repair_v1.yaml'
        with patch('sys.argv',['train','--config',str(cfg_path),'--cpu-check']), patch('training.local_resources.training_gpu_plan') as gpu, patch('training.train_buyer_operations.subprocess.Popen') as proc, redirect_stdout(io.StringIO()):
            main(); gpu.assert_not_called(); proc.assert_not_called()
        cards = [{'index':str(i), 'fraction':.8} for i in (0,1)]
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_symmetric_cpu_') as tmp:
            def fake_launch(cmd,**kw):
                self.assertIn('--nproc_per_node=2', cmd)
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock, no model loaded\n'), wait=lambda:0)
            with patch('sys.argv',['train','--config',str(cfg_path),'--gpus','0,1','--output-dir',tmp]), patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), patch('training.local_resources.runtime_cache',return_value=({},None)), patch('training.train_buyer_operations.subprocess.Popen',side_effect=fake_launch), redirect_stdout(io.StringIO()):
                main()
            run = next(Path(tmp).glob('*_buyer_operations_*'))
            cfg = json.loads((run/'train_config.json').read_text())
            self.assertIn('134251',cfg['initial_adapter']); self.assertNotIn('resume_checkpoint',cfg)
            with self.assertRaisesRegex(ValueError,'Missing step'):
                resolve_run(latest=True,runs_dir=tmp)
            for step in (16,32):
                cp = run/'checkpoints'/f'step_{step:04d}'; cp.mkdir(parents=True)
                for name in ('adapter_config.json','adapter_model.safetensors'):
                    shutil.copy2(ROOT/'agents/example_student/buyer/checkpoint'/name,cp/name)
                (cp/'training_step.json').write_text(json.dumps({'step':step,'thinking':True}))
            self.assertEqual(resolve_run(latest=True,runs_dir=tmp), run)
            # A newer incomplete run must fail, rather than silently evaluate an old one.
            newer = Path(tmp)/'99999999_buyer_operations_incomplete'; newer.mkdir()
            (newer/'train_config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'not completed'):
                resolve_run(latest=True,runs_dir=tmp)
            output = run/'reports'; output.mkdir()
            with patch('sys.argv',['eval',str(run),'--prepare-only']), patch('training.local_resources.output_parent',return_value=(output,None)), patch('training.local_resources.query_gpus') as gpu, patch('training.buyer_diagnostics.subprocess.Popen') as proc, redirect_stdout(io.StringIO()):
                folder = evaluate(); gpu.assert_not_called(); proc.assert_not_called()
            plan = json.loads((folder/'plan.json').read_text())
            self.assertEqual(plan['calls'],1152); self.assertTrue(plan['thinking'])
            self.assertEqual(plan['experiment_manifest']['experiment'],'buyer_symmetric_repair_v1')

    def test_evaluation_cash_guard_rejects_regression_even_with_high_recall(self):
        from training.evaluate_board_repair import write_review
        source = ROOT/'results/basic_checks/20261119-173122_buyer_diagnostic_1uxajui9'
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_symmetric_review_') as tmp:
            dest = Path(tmp)
            for name in ('cash_decision_pairs.json','cases.json','scored_decisions.jsonl','plan.json'):
                shutil.copy2(source/name,dest/name)
            plan = json.loads((dest/'plan.json').read_text())
            plan['experiment_manifest']['experiment']='buyer_symmetric_repair_v1'
            (dest/'plan.json').write_text(json.dumps(plan))
            with redirect_stdout(io.StringIO()):review=write_review(dest)
            self.assertFalse(review['models']['repair32']['checks']['overall_cash_errors_not_worse_than_parent'])
            self.assertFalse(review['models']['repair32']['checks']['offer/operations_reasoning_p3_positive_capture_not_worse_than_parent'])
            self.assertIsNone(review['candidate_meeting_targets'])


if __name__ == '__main__':
    unittest.main()
