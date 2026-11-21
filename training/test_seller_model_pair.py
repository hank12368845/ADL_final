"""CPU-only paired-model market, routing, reporting and restart checks."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training import compare_seller_models as pair
from adlmarket.setups import market_run


class SellerModelPairTests(unittest.TestCase):
    def test_initial_market_personas_and_opponents_match_only_adapter_differs(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);plan=pair.prepare_pair(root,5)
            loaded={k:market_run(root/k/'market.yaml',root=pair.ROOT) for k in ('starter','step46')}
            self.assertEqual(pair.initial_signature(loaded['starter']),pair.initial_signature(loaded['step46']))
            self.assertEqual(pair.opponent_signature(loaded['starter']),pair.opponent_signature(loaded['step46']))
            self.assertEqual(plan['total_actions'],900)
            for label,model in loaded.items():
                manifest=json.loads((root/label/'suite.json').read_text());pair.validate(model,manifest)
                for a in model.sellers:
                    if a.owner=='ta':continue
                    spec=model.models[a.model]
                    self.assertEqual(Path(spec.lora.path),pair.STARTER if label=='starter' else pair.CANDIDATE)
                    self.assertEqual(a.persona_body(),(pair.ROOT/f'training/persona_versions/seller_checkpoint1_v2/persona_{a.id[-1]}.md').read_text().strip())
            with self.assertRaisesRegex(ValueError,'seed4'):pair.prepare_pair(root/'old',4)

    def test_routing_uses_two_when_available_or_one_without_changing_adapter(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);pair.prepare_pair(root,5);arm=root/'step46';m=json.loads((arm/'suite.json').read_text())
            rows=[dict(index=str(i),uuid=f'test-{i}',total_gib=24.,free_gib=23.) for i in (0,1)]
            with patch('training.local_resources.query_gpus',return_value=rows),redirect_stdout(io.StringIO()):
                path,_=pair.configure(arm,m,'0,1',8100)
            after=market_run(path,root=pair.ROOT);pair.validate(after,m)
            r=json.loads((arm/'resource_plan.json').read_text());self.assertEqual(len(set(r['routes'].values())),2)
            for p in [arm/'routing_agents']:p.rename(arm/'old_routing')
            rows[0]['free_gib']=10.
            with patch('training.local_resources.query_gpus',return_value=rows),redirect_stdout(io.StringIO()):
                path,_=pair.configure(arm,m,'0,1',8100)
            r=json.loads((arm/'resource_plan.json').read_text());self.assertEqual([d['index'] for d in r['devices']],['1'])
            self.assertEqual(len(set(r['routes'].values())),1);pair.validate(market_run(path,root=pair.ROOT),m)

    def test_full_mock900_reports_per_persona_and_resume_skips_both(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['pair','--mock','--output-dir',tmp]),patch('training.local_resources.query_gpus') as gpu,redirect_stdout(io.StringIO()):
                result=pair.main();gpu.assert_not_called()
            self.assertTrue(result['complete']);self.assertFalse(result['valid_real_comparison'])
            root=next(Path(tmp).glob('*_seller_models_*'))
            for label in ('starter','step46'):
                w=json.loads((root/label/'wrapper_result.json').read_text());self.assertEqual(w['actions'],450)
                self.assertEqual(set(result['arms'][label]['sellers']),set(pair.IDS))
                self.assertIn('overall',(root/label/'progress.log').read_text())
                self.assertIn('quote_diagnostics',result['arms'][label]['sellers'][pair.IDS[0]])
            with patch('sys.argv',['pair','--mock','--resume',str(root)]),patch.object(pair.previous,'play') as play,redirect_stdout(io.StringIO()):
                pair.main();play.assert_not_called()
            with patch('sys.argv',['pair','--resume',str(root)]),redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'mix mock'):
                pair.main()

    def test_quote_diagnostics_distinguish_same_listing_grade_change_and_empty_stock(self):
        aid=pair.IDS[0]
        def row(tick,price,grade='high',stock=2,board=True):
            return dict(tick=tick,round=1,tool='set_price',args=dict(product_id='P0',price=price,grade=grade),result=dict(ok=True),
                        observation=dict(turns_left_this_round=6,you=dict(stock={'P0':stock,'P1':1}),
                                         board=[dict(seller=aid,product_id='P0',price=price,in_stock=stock)] if board else []))
        rows=[row(1,3,board=False),row(2,3),row(3,3,grade='low'),row(4,3,stock=0,board=False)]
        r=pair.quote_diagnostics(rows,aid)
        self.assertEqual(r['unchanged_live_listings'],[dict(tick=2,product='P0')])
        self.assertEqual(len(r['observed_insufficient_stock_quotes']),1)
        self.assertTrue(r['observed_insufficient_stock_quotes'][0]['engine_result']['ok'])
        self.assertTrue(r['observed_insufficient_stock_quotes'][0]['other_stock_available'])

    def test_prepare_only_has_no_gpu_query_wait_or_play(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['pair','--prepare-only','--wait','--output-dir',tmp]),patch('training.local_resources.query_gpus') as gpu, \
                 patch('training.queue_seller_personas.wait_ready') as wait,patch.object(pair.previous,'play') as play,redirect_stdout(io.StringIO()):
                pair.main();gpu.assert_not_called();wait.assert_not_called();play.assert_not_called()

    def test_startup_resource_race_requeues_but_runtime_error_stops(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);pair.prepare_pair(root,5);plan=dict(arms={'starter':str(root/'starter')})
            with patch('training.queue_seller_personas.wait_ready',return_value=['1']) as ready, \
                 patch.object(pair,'configure',return_value=('unused',{})), \
                 patch.object(pair.previous,'play',side_effect=[RuntimeError('starter incomplete'),None]) as play, \
                 patch('training.gpu_startup_errors.startup_memory_races',return_value=[{'detail':'startup free VRAM'}]),redirect_stdout(io.StringIO()):
                pair.execute(root,plan,wait=True);self.assertEqual(play.call_count,2);self.assertEqual(ready.call_count,2)
            self.assertEqual(len((root/'resource_requeues.jsonl').read_text().splitlines()),1)
            with patch('training.queue_seller_personas.wait_ready',return_value=['1']),patch.object(pair,'configure',return_value=('unused',{})), \
                 patch.object(pair.previous,'play',side_effect=RuntimeError('CUDA out of memory')) as play, \
                 patch('training.gpu_startup_errors.startup_memory_races',return_value=[]),redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'CUDA out'):
                pair.execute(root,plan,wait=True)
            self.assertEqual(play.call_count,1)


if __name__=='__main__':unittest.main()
