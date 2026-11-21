"""CPU orchestration tests only; no real query, sleep, child job or GPU service."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.overnight_evaluations import main,execute_plan
from training.queue_buyer_economy import main as buyer_main,launch as buyer_launch


class OvernightTests(unittest.TestCase):
    def test_cpu_plan_uses_real_fixed_experiments_and_no_job_launch(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['night','--prepare-only','--output-dir',tmp]),patch('training.overnight_evaluations.run_job') as run,redirect_stdout(io.StringIO()):
                folder=main();run.assert_not_called()
            plan=json.loads((folder/'overnight_plan.json').read_text())
            self.assertEqual(plan['total_actions'],7650)
            self.assertEqual([j['name'] for j in plan['jobs']],['seller_personas','buyer_economy'])
            for j in plan['jobs']:
                self.assertIn('--resume',j['command']);self.assertEqual(j['command'][j['command'].index('--gpus')+1],'0,1')

    def test_sequence_continues_after_failure_and_resume_skips_success(self):
        plan=dict(jobs=[dict(name='seller',command=['fake-seller'],output='seller'),
                        dict(name='buyer',command=['fake-buyer'],output='buyer')])
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            folder=Path(tmp)
            with patch('training.overnight_evaluations.run_job',side_effect=[1,0]) as run:
                state=execute_plan(folder,plan,lambda s:None)
            self.assertEqual([c.args[0] for c in run.call_args_list],[['fake-seller'],['fake-buyer']])
            self.assertEqual(state['jobs']['seller']['status'],'failed');self.assertEqual(state['jobs']['buyer']['status'],'success')
            with patch('training.overnight_evaluations.run_job',return_value=0) as run:
                execute_plan(folder,plan,lambda s:None)
            self.assertEqual(run.call_count,1);self.assertEqual(run.call_args.args[0],['fake-seller'])

    def test_interrupt_stops_before_second_job(self):
        plan=dict(jobs=[dict(name='seller',command=['fake-seller'],output='seller'),dict(name='buyer',command=['fake-buyer'],output='buyer')])
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('training.overnight_evaluations.run_job',return_value=130) as run:
                with self.assertRaises(KeyboardInterrupt):execute_plan(Path(tmp),plan,lambda s:None)
            self.assertEqual(run.call_count,1)
            status=json.loads((Path(tmp)/'overnight_status.json').read_text())
            self.assertEqual(status['jobs']['seller']['status'],'interrupted')

    def test_buyer_queue_cpu_preparation_never_waits_or_launches(self):
        from training.overnight_evaluations import BUYER
        with patch('sys.argv',['buyer-queue','--resume',str(BUYER),'--prepare-only']),patch('training.queue_buyer_economy.wait_ready') as wait,patch('training.queue_buyer_economy.launch') as run,redirect_stdout(io.StringIO()):
            folder=buyer_main();wait.assert_not_called();run.assert_not_called()
        self.assertEqual(folder,BUYER)

    def test_buyer_queue_retries_memory_race_with_same_economy(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            folder=Path(tmp);(folder/'suite.json').write_text(json.dumps({'cpu_fixture':True}))
            with patch('sys.argv',['buyer-queue','--resume',tmp]),patch('training.queue_buyer_economy.check_sources'),patch('training.queue_buyer_economy.finished',return_value=False),patch('training.queue_buyer_economy.wait_ready',side_effect=[['0'],['1']]),patch('training.queue_buyer_economy.launch',side_effect=[(1,True),(0,False)]) as launch,redirect_stdout(io.StringIO()):
                buyer_main()
            self.assertEqual(launch.call_count,2)
            self.assertEqual(launch.call_args_list[0].args[:2],(folder,['0']))
            self.assertEqual(launch.call_args_list[1].args[:2],(folder,['1']))


if __name__=='__main__':unittest.main()
