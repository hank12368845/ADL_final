"""CPU accounting, official-rule preparation, dual-route and mock-engine checks."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class Step84MarketTests(unittest.TestCase):
    def test_stock_race_is_not_confused_with_missing_pair_offer_or_excess_qty(self):
        from training.step84_market_report import stock_error_kind
        row={'seller':'s','product_id':'P','in_stock':1}
        args={'seller_id':'s','product_id':'P','qty':'1'}
        trades=[{'seller':'s','product':'P','tick':4,'qty':1}]
        self.assertEqual(stock_error_kind('buy_listed',args,row,trades,4),'no_stock_same_tick_exhausted')
        self.assertEqual(stock_error_kind('buy_listed',args,None,trades,4),'no_stock_without_current_board_pair')
        self.assertEqual(stock_error_kind('buy_listed',{**args,'qty':'2'},row,trades,4),'requested_quantity_exceeds_visible_stock')
        self.assertEqual(stock_error_kind('accept_offer',{},None,trades,4),'offer_stock_unavailable')
        self.assertEqual(stock_error_kind('buy_listed',args,row,trades,5),'no_stock_visible_unresolved')

    def test_per_unit_replay_counts_upgrade_separately_from_zero_value_rebuy(self):
        from training.step84_market_report import unit_ledger
        trades=[]
        for i,(grade,price,gain) in enumerate([('low',100,400),('high',200,600),('high',300,0)],1):
            trades.append(dict(id=f't{i}',tick=i,product='P',seller='s',qty=1,grade=grade,price=price,utility=gain,via='board'))
        ledger=unit_ledger(trades,{'P':[1000]},.4)
        self.assertEqual([u['utility_cents'] for u in ledger],[400,600,0])
        self.assertEqual([u['repeat_unit'] for u in ledger],[False,True,True])
        self.assertEqual(sum(u['net_cents'] for u in ledger),400)
        self.assertEqual(sum(u['price_cents'] for u in ledger if u['trade_already_owned']),500)
        self.assertEqual(sum(u['price_cents'] for u in ledger if u['utility_cents']==0),300)
        bulk=[dict(id='t1',tick=1,product='P',seller='s',qty=2,grade='high',price=100,utility=1000,via='board')]
        self.assertEqual([u['utility_cents'] for u in unit_ledger(bulk,{'P':[1000]},.4)],[1000,0])
        bad=copy.deepcopy(trades);bad[1]['utility']=1000
        with self.assertRaisesRegex(ValueError,'marginal utility mismatch'):
            unit_ledger(bad,{'P':[1000]},.4)

    def test_visible_bounds_handle_owned_low_grade_and_cash_gap_without_oracle(self):
        from training.step84_market_report import visible_options
        sit={'your_utility_per_unit':{'P':[10]},'market_overview':[{'product_id':'P','quality_varies':True}]}
        obs={'round':1,'you':{'cash':.5,'units_bought':{'P':1},'units_by_grade':{'P':{'low':1}}},
             'board':[{'seller':'s','product_id':'P','price':1,'in_stock':1}], 'offers_to_you':[]}
        self.assertEqual(visible_options(obs,sit,.4,cash_limit=200),[])
        obs['board'][0]['certified_high']=True
        self.assertEqual(visible_options(obs,sit,.4),[])
        option=visible_options(obs,sit,.4,cash_limit=200)[0]
        self.assertEqual(option['marginal_lower_bound_cents'],600)
        self.assertEqual(option['gain_lower_bound_cents'],500)
        obs['you']['units_by_grade']['P']={'high':1}
        self.assertEqual(visible_options(obs,sit,.4,cash_limit=200),[])

    def test_turn_stats_use_attempt_and_llm_denominators_and_all_spent_tokens(self):
        from training.step84_market_report import turn_stats
        rows=[]
        for i in range(3):
            rows.append(dict(tool='buy_listed',result={'ok':False,'error':'insufficient_cash'} if i==0 else {'ok':True},
                observation={'output_tokens_left_this_round':8192-i*100},
                llm={'outcome':'accepted','stop_reason':'length' if i==1 else 'stop','output_tokens_spent':100,
                     'usage':{'output_tokens':10,'reasoning_tokens':90,'input_tokens':200+i},'history_turns_shown':i,
                     'length_retries':[{}] if i==1 else []}))
        rows.append(dict(tool='wait',result={'ok':True},observation={'output_tokens_left_this_round':0},llm={'outcome':'round_budget_used'}))
        s=turn_stats(rows)
        self.assertEqual((s['turns'],s['llm_decision_turns'],s['purchase_attempts']),(4,3,3))
        self.assertEqual(s['cash_errors_per_purchase_attempt'],1/3)
        self.assertEqual(s['truncation_per_llm_turn'],1/3)
        self.assertEqual(s['spent_output_tokens'],300)
        self.assertEqual(s['budget_waits'],1)
        self.assertEqual(s['length_retry_attempts'],1)

    def test_preparation_never_launches_gpu_and_keeps_exact_current_personas(self):
        from training.diagnose_step84_market import main
        from adlmarket.setups import market_run
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_step84_market_test_') as tmp:
            with patch('sys.argv',['market','--prepare-only','--output-dir',tmp]), \
                 patch('training.local_resources.query_gpus') as query, \
                 patch('training.diagnose_step84_market.subprocess.Popen') as popen, redirect_stdout(io.StringIO()):
                directory=main();query.assert_not_called();popen.assert_not_called()
            m=json.loads((directory/'suite.json').read_text());loaded=market_run(Path(m['config']),root=ROOT)
            self.assertEqual((m['total_markets'],m['total_actions'],m['target_actions']),(1,450,90))
            self.assertTrue(m['adapter'].endswith('step_0084'))
            self.assertEqual((len(loaded.sellers),len(loaded.buyers)),(5,10))
            self.assertEqual(loaded.config.market.rounds.output_tokens_per_round,8192)
            self.assertEqual(loaded.config.market.schema_.budget_extra,.15)
            for i in ('1','2','3'):
                self.assertEqual((directory/f'buyer/persona_{i}.md').read_bytes(),(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_bytes())

    def test_dual_gpu_routes_keep_exact_model_persona_and_pin_revision(self):
        from training.diagnose_step84_market import prepare,configure_resources,check_sources
        from adlmarket.setups import market_run
        from adlmarket.serving import vllm_command
        cards=[dict(index=str(i),fraction=.85,free_gib=23,total_gib=24,budget_gib=20.4) for i in (0,1)]
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_step84_routes_test_') as tmp:
            directory=Path(tmp);m=prepare(directory)
            with patch('training.local_resources.query_gpus',return_value=[]), \
                 patch('training.local_resources.select_gpus',return_value=(cards,[])), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),redirect_stdout(io.StringIO()):
                effective,_=configure_resources(directory,m)
            loaded=market_run(effective,root=ROOT)
            resource=json.loads((directory/'resource_plan.json').read_text())
            self.assertEqual(set(resource['routes'].values()),{'http://localhost:8000/v1','http://localhost:8001/v1'})
            self.assertEqual(len(resource['routes']),15)
            self.assertEqual({resource['routes'][f'example_student_buyer_p{i}'] for i in (1,2,3)},{'http://localhost:8000/v1'})
            self.assertEqual(len(loaded.serving),2)
            for name,sv in loaded.serving.items():
                command,_=vllm_command(loaded.models[name],sv,0)
                self.assertEqual(command[command.index('--revision')+1],m['revision'])
                self.assertEqual(command[command.index('--tokenizer-revision')+1],m['revision'])
                self.assertEqual(sv.max_model_len,64000)
            check_sources(m)

    def test_full_cpu_mock_engine_reports_scores_but_is_never_real_evidence(self):
        from training.diagnose_step84_market import main
        from training.step84_market_report import analyze_market,IDS
        originals=[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)]
        originals+=[ROOT/'agents/example_student/buyer/agent.yaml',ROOT/'agents/example_student/buyer/checkpoint/adapter_model.safetensors']
        before={p:p.read_bytes() for p in originals}
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_step84_mock_test_') as tmp:
            with patch('sys.argv',['market','--mock','--output-dir',tmp]),patch('training.local_resources.query_gpus') as query,redirect_stdout(io.StringIO()):
                directory=main();query.assert_not_called()
            report=json.loads((directory/'market_loss_review.json').read_text())
            self.assertTrue(report['complete']);self.assertTrue(report['mock']);self.assertFalse(report['valid_real_diagnostic'])
            self.assertEqual(report['actions'],450)
            self.assertEqual(set(report['buyers']),set(IDS))
            for b in report['buyers'].values():
                self.assertEqual(b['surplus_cents'],b['profitable_unit_net_cents']-b['loss_on_negative_units_cents'])
                self.assertEqual(b['surplus_cents'],b['other_units_net_cents']-b['zero_marginal_spend_cents'])
                self.assertEqual(b['early_turns1_3']['turns'],15)
                self.assertEqual(b['late_turns4_6']['turns'],15)
                self.assertTrue(all(s['turns']==5 for s in b['by_turn_slot'].values()))
            run=Path(report['run_dir']);folder=run/'markets'/report['market'];path=folder/'scores.json'
            scores=json.loads(path.read_text());next(r for r in scores if r['agent']==IDS[0])['raw']+=1
            path.write_text(json.dumps(scores))
            with self.assertRaisesRegex(ValueError,'surplus/spent/utility/cash mismatch'):
                analyze_market(run)
        for p,content in before.items():self.assertEqual(p.read_bytes(),content)


if __name__=='__main__':unittest.main()
