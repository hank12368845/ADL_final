"""CPU configuration, routing, accounting and opportunity-diagnosis regressions."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def fixture(path,mock=False):
    from training.buyer_market_report import IDS
    path=Path(path);folder=path/'markets/e0_m0';folder.mkdir(parents=True)
    (path/'metadata.json').write_text(json.dumps({'status':'finished','mock_llm':mock}))
    buyers=[{'id':aid,'gained_utility':0,'spent':0,'cash':10000,'budget':10000} for aid in IDS+[f'ta_b{i}' for i in range(7)]]
    sellers=[{'id':f'seller_{i}'} for i in range(5)]
    trade={'id':'t1','tick':1,'buyer':'ta_b0','seller':'seller_0','product':'P1','qty':1,'price':500,'utility':1000,'grade':'high'}
    (folder/'final.json').write_text(json.dumps({'buyers':buyers,'sellers':sellers,'trades':[trade]}))
    (folder/'scores.json').write_text(json.dumps([{'agent':b['id'],'raw':0} for b in buyers]))
    agents=[{'agent':a['id'],'owner':'ta' if a['id'].startswith('ta') or a['id'].startswith('seller') else 'example_student','situation':{'market_overview':[{'product_id':'P1','quality_varies':False}]}} for a in buyers+sellers]
    (folder/'agents.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in agents))
    obs={'round':1,'board':[{'seller':'seller_0','product_id':'P1','price':5,'in_stock':1}], 'offers_to_you':[],
         'you':{'cash':100,'utility_of_next_unit':{'P1':10},'units_bought':{}}}
    rows=[]
    for tick in range(1,31):
        for a in buyers+sellers:
            row={'type':'turn','tick':tick,'round':(tick-1)//6+1,'agent':a['id'],'tool':'wait','args':{},'observation':copy.deepcopy(obs),'result':{'ok':True},'llm':{}}
            if tick==1 and a['id'] in IDS[:2]:
                row.update(tool='buy_listed',args={'seller_id':'absent' if a['id']==IDS[0] else 'seller_0','product_id':'P1','qty':1},result={'ok':False,'error':'no_stock'})
            if tick==1 and a['id']==IDS[2]:row['llm']={'outcome':'round_budget_used'}
            rows.append(row)
    (folder/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


class BuyerMarketReviewTests(unittest.TestCase):
    def test_full_preparation_is_paired_and_only_changes_target_adapter(self):
        from training.evaluate_buyer_market import prepare
        from adlmarket.setups import market_run
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_market_prepare_') as tmp:
            m=prepare(tmp)
            self.assertEqual(m['total_actions'],4500);self.assertEqual(m['total_markets'],10)
            self.assertEqual(m['plans']['candidate'],m['plans']['base_on'])
            self.assertEqual([r['rotation'] for r in m['plans']['candidate']],list(range(5)))
            for arm in ('candidate','base_on'):
                loaded=market_run(Path(m['configs'][arm]),root=ROOT)
                self.assertEqual((len(loaded.sellers),len(loaded.buyers)),(5,10))
                self.assertEqual(loaded.config.market.rounds.total,5);self.assertEqual(loaded.config.market.rounds.turns_per_round,6)
                self.assertEqual(loaded.config.market.rounds.output_tokens_per_round,8192)
                for a in loaded.buyers:
                    if a.owner!='ta':self.assertEqual(bool(loaded.models[a.model].lora),arm=='candidate');self.assertTrue(a.reasoning)

    def test_cpu_entry_never_queries_or_launches_gpu(self):
        from training.evaluate_buyer_market import main
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_market_entry_') as tmp:
            with patch('sys.argv',['review','--prepare-only','--output-dir',tmp]),patch('training.local_resources.query_gpus') as query,patch('training.evaluate_buyer_market.subprocess.Popen') as launch,redirect_stdout(io.StringIO()):
                directory=main();query.assert_not_called();launch.assert_not_called()
            config=next((directory/'candidate').glob('market.yaml'));config.write_text(config.read_text()+'\n# changed\n')
            with patch('sys.argv',['review','--resume',str(directory)]),patch('training.evaluate_buyer_market.subprocess.Popen') as launch,self.assertRaisesRegex(ValueError,'Frozen suite input changed'):
                main()
            launch.assert_not_called()

    def test_two_card_routes_preserve_adapter_and_behavior(self):
        import yaml
        from training.evaluate_buyer_market import prepare
        from adlmarket.setups import market_run
        from training.local_resources import gpu_models
        from training.market_load_balance import balance_ta_routes,validate_equivalent
        cards=[{'index':str(i),'fraction':.85} for i in (0,1)]
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_market_routes_') as tmp:
            m=prepare(tmp)
            for label,path in m['configs'].items():
                original=market_run(Path(path),root=ROOT);config=yaml.safe_load(Path(path).read_text())
                models=gpu_models(yaml.safe_load((ROOT/'configs/models.yaml').read_text()),cards,8100)
                out=Path(tmp)/label/'route_check';out.mkdir()
                routes=balance_ta_routes(config,original,models,out,ROOT,cards,8100)
                (out/'models.yaml').write_text(yaml.safe_dump(models));config['models']=str(out/'models.yaml');(out/'market.yaml').write_text(yaml.safe_dump(config))
                after=market_run(out/'market.yaml',root=ROOT);validate_equivalent(original,after)
                self.assertEqual(set(routes['assignments'].values()),{0,1})
                for a in after.everyone():self.assertEqual(after.models[a.model].url_for(after.models[a.model].model),f'http://localhost:{8100+routes["assignments"][a.id]}/v1')

    def test_report_distinguishes_collisions_missing_ids_budget_and_mock(self):
        from training.buyer_market_report import analyze_arm,write_report,IDS
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_market_report_') as tmp:
            p=Path(tmp);run=fixture(p/'run');a=analyze_arm(run,1)
            self.assertEqual(a['personas'][IDS[0]]['counts']['no_stock_missing_current_pair'],1)
            self.assertEqual(a['personas'][IDS[1]]['counts']['no_stock_same_tick_exhausted'],1)
            self.assertEqual(a['personas'][IDS[2]]['counts']['round_budget_waits'],1)
            (p/'suite.json').write_text(json.dumps({'markets_per_arm':1,'description':'CPU fixture'}))
            (p/'state.json').write_text(json.dumps({'completed':{'candidate':{'run_dir':str(run)}}}))
            with redirect_stdout(io.StringIO()):r=write_report(p)
            self.assertFalse(r['complete']);self.assertFalse(r['valid_real_evaluation'])
            (run/'metadata.json').write_text(json.dumps({'status':'finished','mock_llm':True}))
            (p/'state.json').write_text(json.dumps({'completed':{arm:{'run_dir':str(run)} for arm in ('candidate','base_on')}}))
            with redirect_stdout(io.StringIO()):r=write_report(p)
            self.assertTrue(r['complete']);self.assertFalse(r['valid_real_evaluation']);self.assertNotIn('best_persona_per_arm',r)
            scores=run/'markets/e0_m0/scores.json';s=json.loads(scores.read_text());s[0]['raw']=1;scores.write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'Engine surplus mismatch'):analyze_arm(run,1)

    def test_visible_bounds_and_progress_are_explicitly_limited(self):
        from training.buyer_market_report import visible_opportunities
        from training.evaluate_buyer_market import progress_text
        o={'round':1,'board':[{'seller':'s','product_id':'P1','price':5,'in_stock':1}],'offers_to_you':[], 'you':{'cash':5,'utility_of_next_unit':{'P1':10},'units_bought':{}}}
        sit={'market_overview':[{'product_id':'P1','quality_varies':False}]}
        self.assertEqual(visible_opportunities(o,sit),1)
        sit['market_overview'][0]['quality_varies']=True;self.assertEqual(visible_opportunities(o,sit),0)
        m={'actions_per_arm':2250,'total_actions':4500,'total_markets':10}
        self.assertIn('ETA estimating',progress_text(0,0,m,30))
        self.assertIn('2700/4500',progress_text(1,450,m,600));self.assertIn('6/10 markets',progress_text(1,450,m,600))


if __name__=='__main__':unittest.main()
