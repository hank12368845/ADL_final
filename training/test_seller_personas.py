"""CPU checks for the persona-only, full-market seller comparison."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.compare_seller_personas import ROOT,IDS,ADAPTER,prepare_pair,configure,initial_state,main
from adlmarket.setups import market_run


class SellerPersonaTests(unittest.TestCase):
    def test_matched_world_and_only_student_personas_differ(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=Path(tmp);plan=prepare_pair(p,4)
            loaded={k:market_run(p/k/'market.yaml',root=ROOT) for k in ['v1','v2']}
            self.assertEqual(initial_state(loaded['v1']),initial_state(loaded['v2']))
            self.assertEqual(plan['total_actions'],900)
            for label,run in loaded.items():
                self.assertEqual((len(run.sellers),len(run.buyers)),(5,10))
                self.assertEqual({a.id for a in run.sellers if a.owner!='ta'},set(IDS))
                self.assertTrue(all(a.owner=='ta' for a in run.buyers))
                for a in run.sellers:
                    if a.owner=='ta':continue
                    self.assertTrue(a.reasoning);self.assertEqual(Path(run.models[a.model].lora.path),ADAPTER)
                    self.assertEqual(a.persona_body(),(ROOT/f'training/persona_versions/seller_checkpoint1_{label}/persona_{a.id[-1]}.md').read_text().strip())
            self.assertNotEqual([a.persona_body() for a in loaded['v1'].sellers[:3]],
                                [a.persona_body() for a in loaded['v2'].sellers[:3]])

    def test_dual_card_routes_preserve_world_and_opponent_settings(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            folder=Path(tmp);prepare_pair(folder,4)
            arm=folder/'v2';m=json.loads((arm/'suite.json').read_text())
            rows=[dict(index=str(i),uuid=f'cpu-test-{i}',total_gib=24.,free_gib=23.) for i in [0,1]]
            with patch('training.local_resources.query_gpus',return_value=rows),redirect_stdout(io.StringIO()):
                path,_=configure(arm,m,'0,1',8100)
            after=market_run(path,root=ROOT)
            self.assertEqual(initial_state(after),m['initial_signature'])
            resource=json.loads((arm/'resource_plan.json').read_text())
            self.assertEqual(set(resource['routes'].values()),{'http://localhost:8100/v1','http://localhost:8101/v1'})

    def test_full900_action_mock_report_and_resume(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['compare','--mock','--output-dir',tmp]),redirect_stdout(io.StringIO()):
                result=main()
            self.assertTrue(result['complete']);self.assertFalse(result['valid_real_comparison'])
            self.assertEqual(result['training_updates'],0);self.assertFalse(result['automatic_promotion'])
            folder=next(Path(tmp).glob('*_seller_personas_*'))
            for label in ['v1','v2']:
                w=json.loads((folder/label/'wrapper_result.json').read_text())
                self.assertEqual(w['actions'],450);self.assertEqual(len(result['arms'][label]['sellers']),3)
                self.assertIn('overall',(folder/label/'progress.log').read_text())
            with patch('sys.argv',['compare','--mock','--resume',str(folder)]),patch('training.compare_seller_personas.play') as play,redirect_stdout(io.StringIO()):
                main();play.assert_not_called()
            with patch('sys.argv',['compare','--resume',str(folder)]),self.assertRaisesRegex(ValueError,'mix mock'):
                main()

    def test_prepare_only_never_queries_or_launches_gpu(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['compare','--prepare-only','--output-dir',tmp]),patch('training.local_resources.query_gpus') as query,patch('training.compare_seller_personas.play') as play,redirect_stdout(io.StringIO()):
                main();query.assert_not_called();play.assert_not_called()


if __name__=='__main__':unittest.main()
