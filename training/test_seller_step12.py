"""CPU-only cache identity,450-action replay, resume and resource retry checks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import evaluate_seller_step12 as step12
from training import compare_seller_models as paired
from training.diagnose_step84_market import sha
from adlmarket.setups import market_run


class SellerStep12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir='/tmp');cls.root=Path(cls.temp.name)
        with patch('sys.argv',['pair','--mock','--output-dir',str(cls.root)]),redirect_stdout(io.StringIO()):paired.main()
        cls.control=next(cls.root.glob('*_seller_models_*'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_frozen_single_arm_matches_cached_world_and_v2_only_adapter_changes(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            plan=step12.prepare(Path(tmp),self.control,mock=True)
            self.assertEqual((plan['new_actions'],plan['cached_actions']),(450,900))
            m=json.loads((Path(plan['arm'])/'suite.json').read_text());loaded=market_run(m['config'],root=step12.ROOT)
            self.assertEqual(paired.initial_signature(loaded),plan['initial_signature'])
            for a in loaded.sellers:
                if a.owner=='ta':continue
                self.assertEqual(Path(loaded.models[a.model].lora.path),step12.CANDIDATE)
                self.assertTrue(a.reasoning)
                self.assertEqual(a.persona_body(),(step12.ROOT/f'training/persona_versions/seller_checkpoint1_v2/persona_{a.id[-1]}.md').read_text().strip())

    def test_new450_mock_and_three_version_report_resume_does_not_replay(self):
        before={str(p):sha(p) for p in self.control.rglob('wrapper_result.json')}
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['step12','--mock','--control',str(self.control),'--output-dir',tmp]),patch('training.local_resources.query_gpus') as gpu,redirect_stdout(io.StringIO()):
                result=step12.main();gpu.assert_not_called()
            self.assertEqual(set(result['results']),{'starter','step12','step46'})
            self.assertEqual(result['new_actions'],450);self.assertFalse(result['valid_real_comparison'])
            root=next(Path(tmp).glob('*_seller_step12_*'));w=json.loads((root/'step12/wrapper_result.json').read_text())
            self.assertEqual(w['actions'],450);self.assertIn('NEW 450/450',(root/'step12/progress.log').read_text())
            with patch('sys.argv',['step12','--mock','--resume',str(root)]),patch.object(step12,'play') as play,redirect_stdout(io.StringIO()):
                step12.main();play.assert_not_called()
        self.assertTrue(all(sha(p)==h for p,h in before.items()))

    def test_prepare_only_does_not_query_wait_or_start_model(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['step12','--mock','--control',str(self.control),'--output-dir',tmp,'--prepare-only','--wait']), \
                 patch('training.local_resources.query_gpus') as gpu,patch('training.queue_seller_personas.wait_ready') as wait, \
                 patch.object(step12,'play') as play,redirect_stdout(io.StringIO()):
                step12.main();gpu.assert_not_called();wait.assert_not_called();play.assert_not_called()

    def test_changed_cached_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            plan=step12.prepare(Path(tmp),self.control,mock=True)
            dummy=Path(tmp)/'evidence';dummy.write_text('original');plan['cached']['starter']['files'][str(dummy)]=sha(dummy)
            dummy.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Cached starter result changed'):step12.verify_cached(plan)

    def test_mock_cannot_reuse_actual_real_model_controls(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with self.assertRaisesRegex(ValueError,'mix real and mock'):step12.prepare(Path(tmp),step12.CONTROL,mock=True)

    def test_startup_memory_retry_and_runtime_failure_stop(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);plan=step12.prepare(root,self.control,mock=True)
            with patch('training.queue_seller_personas.wait_ready',return_value=['1']) as ready, \
                 patch.object(paired,'configure',return_value=('unused',{})), \
                 patch.object(step12,'play',side_effect=[RuntimeError('step12 incomplete'),None]) as play, \
                 patch('training.gpu_startup_errors.startup_memory_races',return_value=[{'detail':'startup free VRAM'}]),redirect_stdout(io.StringIO()):
                step12.execute(root,plan,'0,1',8100,True,False)
            self.assertEqual(play.call_count,2);self.assertEqual(ready.call_count,2)
            with patch('training.queue_seller_personas.wait_ready',return_value=['1']),patch.object(paired,'configure',return_value=('unused',{})), \
                 patch.object(step12,'play',side_effect=RuntimeError('CUDA out of memory')) as play, \
                 patch('training.gpu_startup_errors.startup_memory_races',return_value=[]),redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'CUDA out'):
                step12.execute(root,plan,'0,1',8100,True,False)
            self.assertEqual(play.call_count,1)

    def test_stock_review_uses_pre_tick_actual_input_not_mutated_event(self):
        from training.seller_market_observations import decision_stock
        book=dict(initial_stock={'P1':2});trades=[dict(seller='s1',tick=1,product='P1',qty=1),dict(seller='s1',tick=2,product='P1',qty=1)]
        row=dict(agent='s1',tick=2,observation=dict(you=dict(stock={'P1':0})),
                 llm=dict(messages=[dict(role='user',content=json.dumps(dict(you=dict(stock={'P1':1}))))]))
        stock,source=decision_stock(row,book,trades)
        self.assertEqual(stock,{'P1':1});self.assertIn('agree',source)
        row['llm']['messages']=[]
        self.assertEqual(decision_stock(row,book,trades)[0],{'P1':1})
        row['llm']['messages']=[dict(role='user',content=json.dumps(dict(you=dict(stock={'P1':0}))))]
        with self.assertRaisesRegex(ValueError,'differ'):decision_stock(row,book,trades)


if __name__=='__main__':unittest.main()
