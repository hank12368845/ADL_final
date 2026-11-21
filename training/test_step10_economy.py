"""CPU-only checks for official casting, replica routing and15-market resume."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from training.evaluate_step10_economy import (ROOT, IDS, CANDIDATE, Progress, prepare,
    configure, report, main, plan_signature)
from adlmarket.setups import economy_run


class OfficialEconomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='test_economy_',dir=ROOT/'.cache')
        cls.folder=Path(cls.tmp.name)
        cls.manifest=prepare(cls.folder,3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.manifest['snapshot'])
        cls.tmp.cleanup()

    def test_01_real_adapter_and_full_official_casting(self):
        m=self.manifest
        self.assertEqual(m['adapter'],str(CANDIDATE));self.assertEqual(len(m['markets']),15)
        self.assertEqual({p['economy'] for p in m['markets']},{0,1,2})
        self.assertEqual({p['rotation'] for p in m['markets']},set(range(5)))
        for p in m['markets']:
            self.assertEqual(len(p['sellers']),5);self.assertEqual(len(p['buyers']),10)
            self.assertTrue(set(IDS)<=set(p['buyers']))
        self.assertFalse(m['automatic_promotion']);self.assertEqual(m['training_updates'],0)

    def test_02_dual_replica_preserves_grouping_plan_and_weights(self):
        from training.evaluate_step10_economy import seating
        with patch('training.local_resources.query_gpus',side_effect=AssertionError('CPU test queried GPU')):
            effective,_=configure(self.folder,self.manifest,'0,1',8100,mock=True)
        after,_=economy_run(effective,'buyer',root=ROOT)
        self.assertEqual(plan_signature(after)[0],self.manifest['plan_signature'])
        resource=json.loads((self.folder/'resource_plan.json').read_text())
        self.assertEqual(set(resource['routes'].values()),{'http://localhost:8100/v1','http://localhost:8101/v1'})
        before,_=economy_run(self.manifest['config'],'buyer',root=ROOT)
        for side in ['sellers','buyers']:
            left={a.id:a.variant_of for a in getattr(before,side)}
            right={a.id:a.variant_of for a in getattr(after,side)}
            self.assertEqual(left,right)
        self.assertEqual([(p.market_id,[a.id for a in p.buyers]) for p in seating(before)],
                         [(p.market_id,[a.id for a in p.buyers]) for p in seating(after)])

    def test_03_official_mock_runs15_markets_audits_and_resumes(self):
        command=['economy','--resume',str(self.folder),'--mock']
        with patch('sys.argv',command),patch('training.local_resources.query_gpus',side_effect=AssertionError('GPU query')),contextlib.redirect_stdout(io.StringIO()):
            result=main()
        self.assertTrue(result['complete']);self.assertFalse(result['valid_real_evaluation'])
        self.assertEqual(result['total_actions'],6750);self.assertEqual(len(result['markets']),3)
        for aid in IDS:
            self.assertEqual(result['summary'][aid]['markets'],15)
            self.assertEqual(sum(r['stats']['turns'] for r in result['markets'][aid]),450)
        run=Path(result['run_dir']);metadata=json.loads((run/'metadata.json').read_text())
        self.assertEqual(metadata['script'],'run_economy.py');self.assertEqual(metadata['phase'],2)
        progress=(self.folder/'progress.log').read_text();self.assertIn('6750/6750',progress)
        original={p:p.stat().st_mtime_ns for p in run.glob('markets/*/scores.json')}
        # A completed wrapper resume must not run another engine or start services.
        with patch('sys.argv',command),patch('training.evaluate_step10_economy.play',side_effect=AssertionError('replayed finished')),contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(original,{p:p.stat().st_mtime_ns for p in original})
        with patch('sys.argv',['economy','--resume',str(self.folder)]),self.assertRaisesRegex(ValueError,'mix mock'):
            main()
        # Official resume keeps14 completed markets and restarts the interrupted one.
        first=next(run.glob('markets/*/scores.json')); first.unlink()
        metadata['status']='running';(run/'metadata.json').write_text(json.dumps(metadata))
        with patch('sys.argv',command),contextlib.redirect_stdout(io.StringIO()):replayed=main()
        self.assertTrue(replayed['complete'])
        for p,stamp in original.items():
            if p!=first:self.assertEqual(p.stat().st_mtime_ns,stamp)
        self.assertNotEqual(first.stat().st_mtime_ns,original[first])

    def test_02b_busy_second_gpu_falls_back_without_changing_seats(self):
        rows=[dict(index='0',uuid='test0',total_gib=24.,free_gib=23.),
              dict(index='1',uuid='test1',total_gib=24.,free_gib=3.)]
        with patch('training.local_resources.query_gpus',return_value=rows),contextlib.redirect_stdout(io.StringIO()):
            effective,_=configure(self.folder,self.manifest,'0,1',8100,mock=False)
        loaded,_=economy_run(effective,'buyer',root=ROOT)
        resource=json.loads((self.folder/'resource_plan.json').read_text())
        self.assertEqual(len(resource['devices']),1)
        self.assertEqual(set(resource['routes'].values()),{'http://localhost:8100/v1'})
        self.assertEqual(plan_signature(loaded)[0],self.manifest['plan_signature'])

    def test_04_progress_resets_after_partial_market_replacement(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=Path(tmp)/'markets/e0_m0_r0/events.jsonl';p.parent.mkdir(parents=True)
            row=dict(type='turn',agent=IDS[0],round=1,tick=1)
            p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
            tracker=Progress();self.assertEqual(tracker.scan(tmp),2)
            p.write_text(json.dumps(row)+'\n')
            self.assertEqual(tracker.scan(tmp),1)


if __name__=='__main__':unittest.main()
