"""CPU data/engine/gate checks; no GPU query, training or model inference."""
from collections import Counter
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from training.seller_consistency_data import DATA,V1,PARENT,ROOT,audit,horizon_findings,row_findings,stock_assertions,control,verify_target
from training.seller_consistency_train import score,summarize,gate,assignments,hf_messages,render_prompt,prompt_audit


class SellerConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=audit()
        cls.train=[json.loads(l) for l in (DATA/'train.jsonl').read_text().splitlines()]
        cls.cases=json.loads((DATA/'evaluation_cases.json').read_text())
        cls.original={r['id']:r for r in (json.loads(l) for l in (V1/'train.jsonl').read_text().splitlines())}

    def test_horizon_quarantine_keeps_expiry_and_valid_carried_note(self):
        quarantine={r['id'] for r in json.loads((DATA/'quarantine.json').read_text())}
        manual=json.loads((ROOT/'docs/SELLER_HORIZON_DATA_AUDIT.json').read_text())
        bad={r['id'] for r in manual['records'] if r['classification']=='future_action_after_fixed_market_close'}
        legal={r['id'] for r in json.loads((DATA/'allowed_expiries.json').read_text())}
        self.assertEqual(len(bad),7);self.assertTrue(bad<=quarantine)
        self.assertEqual(len(legal),3);self.assertFalse(legal&quarantine)
        self.assertNotIn('cedf-f13ce531_turn74',quarantine)
        self.assertTrue(horizon_findings('If unsold R6, lower the price.',5))
        self.assertFalse(horizon_findings('There is no round 6.',5))
        self.assertFalse(horizon_findings('End R4. Next round lower price.',5,check_relative=False))
        self.assertEqual(horizon_findings('o1 expires R6',5)[0]['kind'],'allowed_offer_expiry')
        self.assertEqual(horizon_findings('Wait until R6 to cut the price.',5)[0]['kind'],'future_round_risk')

    def test_stale_input_is_allowed_but_incorrect_target_stock_is_not(self):
        base=next(r for r in self.train if r['source_kind']=='official_replay')
        row=copy.deepcopy(base);pid=next(iter(row['observation']['you']['stock']))
        row['observation']['your_private_state']=f'Stock: {pid}x999.'
        row['messages'][-1]['content']=json.dumps(row['observation'])
        self.assertFalse(row_findings(row)['risks'])
        row['answer']['tool_calls'][0]['function']['arguments']['private_state']=f'Stock: {pid}x999.'
        self.assertTrue(any(r['kind']=='target_inventory_assertion_mismatch' for r in row_findings(row)['risks']))
        self.assertEqual(stock_assertions('Stock:P60:0, P63x2@15.99. Next: P63 price 16.99.'),[('P60',0),('P63',2)])

    def test_public_bluff_not_treated_as_truth_violation(self):
        row=copy.deepcopy(next(r for r in self.train if r['label_action']=='send_message'))
        row['answer']['tool_calls'][0]['function']['arguments']['message']='All my items cost $999; I have buyers waiting until round 8.'
        self.assertFalse(row_findings(row)['risks'])

    def test_original_targets_splits_and_batches(self):
        m=self.manifest;self.assertEqual((m['training']['native'],m['training']['controlled']),(216,72))
        for r in self.train:
            self.assertIn(r['group'],m['groups']['train'])
            if r['source_kind']=='official_replay':self.assertEqual(r['answer'],self.original[r['source_sample_id']]['answer'])
        plan=json.loads((DATA/'batch_plan.json').read_text());self.assertEqual(len(plan),36)
        self.assertEqual(sorted(i for b in plan for i in b),list(range(288)))
        for b in plan:self.assertEqual(Counter(self.train[i]['source_kind'] for i in b),dict(official_replay=6,engine_control=2))
        for c in self.cases:
            if c['kind']=='heldout_controls':self.assertIn(c['group'],m['groups']['validation'])
        self.assertLess(m['training']['actions']['wait']/288,.15)

    def test_engine_controls_flip_with_stock_and_deadline(self):
        families=Counter(r['family'] for r in self.train if r['source_kind']=='engine_control')
        self.assertEqual(families,dict(live_inventory=18,soldout_switch=18,soldout_wait=18,final_round=18))
        for r in self.train:
            if r['source_kind']!='engine_control':continue
            verify_target(r)
            name=r['answer']['tool_calls'][0]['function']['name']
            if r['family']=='soldout_wait':self.assertEqual(name,'wait');self.assertFalse(any(r['observation']['you']['stock'].values()))
            if r['family']=='final_round' and r['observation']['turns_left_this_round']==1:self.assertEqual(name,'wait')
        gold=self.cases[0];bad=copy.deepcopy(gold);bad['observation']['you']['stock']={p:0 for p in bad['observation']['you']['stock']}
        with self.assertRaisesRegex(ValueError,'current stock'):verify_target(bad)
        bad=copy.deepcopy(gold);pid=next(iter(bad['observation']['you']['stock']));bad['observation']['you']['stock'][pid]=bad['situation']['your_stock'].get(pid,0)+1
        with self.assertRaisesRegex(ValueError,'initial allocation'):verify_target(bad)

    def test_native_tokens_history_and_eos_remain_faithful(self):
        from transformers import AutoTokenizer
        from training.official_buyer_data import BASE,MODEL_REVISION
        tok=AutoTokenizer.from_pretrained(BASE,revision=MODEL_REVISION,local_files_only=True)
        encoded=[json.loads(l) for l in (DATA/'train.tokens.jsonl').read_text().splitlines()]
        long=[r for r,e in zip(self.train,encoded) if r['source_kind']=='official_replay' and len(e['input_ids'])>=10000]
        self.assertEqual(len(long),18);self.assertGreaterEqual(sum(r['history_turns_shown']==6 for r in long),12)
        for r,e in zip(self.train,encoded):
            self.assertEqual(json.JSONDecoder().raw_decode(r['messages'][-1]['content'].lstrip())[0],r['observation'])
            tail=tok.decode(e['input_ids'][e['prompt_length']:],skip_special_tokens=False)
            self.assertIn(r['answer']['reasoning_content'],tail);self.assertIn('<tool_call>',tail);self.assertTrue(tail.rstrip().endswith('<|im_end|>'))
            self.assertLessEqual(len(e['input_ids']),32768)

    def test_engine_acceptance_is_separate_from_stock_guard(self):
        case=copy.deepcopy(self.cases[0]);fn=case['answer']['tool_calls'][0]['function']
        pid=fn['arguments']['product_id'];price=fn['arguments']['price']
        case['observation']['you']['stock'][pid]=0
        result=score(case,dict(tool='set_price',arguments=dict(product_id=pid,price=price)), 'Sold out.</think>',False)
        self.assertTrue(result['engine_accepted']);self.assertFalse(result['valid']);self.assertIn('no_stock_or_excess_qty',result['errors'])
        result=score(self.cases[0],dict(tool=fn['name'],arguments={**fn['arguments'],'private_state':'x'*1200}), 'Quote current stock.</think>',False)
        self.assertTrue(result['valid'])

    def test_gate_rejects_all_wait_and_persona_regression(self):
        def rows():
            out=[]
            for kind,count in [('heldout_controls',12),('positive_regression',9),('official_heldout',6),('real_market_development',3)]:
                for i in range(count):out.append(dict(kind=kind,persona_index=i%3+1,valid=True,safe=True,expected_quote=kind=='positive_regression',
                    profitable_quote=True,control_correct=kind=='heldout_controls',truncated=False,tool='set_price',horizon_risks=[],target_stock_mismatches=[]))
            return out
        parent=rows();parent[0]['control_correct']=False
        candidate=rows();self.assertTrue(gate(summarize(parent),summarize(candidate))['passed'])
        waits=copy.deepcopy(candidate)
        for r in waits:r['profitable_quote']=False;r['tool']='wait'
        self.assertFalse(gate(summarize(parent),summarize(waits))['passed'])
        # Total real-safe count is unchanged; losing a different persona must not be hidden.
        parent[-1]['safe']=False;candidate[-2]['safe']=False
        self.assertFalse(gate(summarize(parent),summarize(candidate))['checks']['personas_preserved'])

    def test_cpu_entry_and_resource_wait_never_launch_gpu(self):
        from training.train_seller_consistency import main
        with patch('training.local_resources.training_gpu_plan',side_effect=AssertionError('GPU queried')),redirect_stdout(io.StringIO()):
            cfg=main(['--cpu-check'])
        self.assertEqual(cfg['learning_rate'],5e-6);self.assertEqual(cfg['parent_adapter'],str(PARENT))
        self.assertEqual(cfg['history_turns'],6)
        self.assertEqual(len(cfg['prompt_audit']),30)
        self.assertEqual(sum(r['parsed_history_argument_strings'] for r in cfg['prompt_audit']),18)

    def test_all_actual_evaluation_prompts_render_before_gpu(self):
        from transformers import AutoTokenizer
        from training.official_buyer_data import BASE,MODEL_REVISION
        tokenizer=AutoTokenizer.from_pretrained(BASE,revision=MODEL_REVISION,local_files_only=True)
        before=copy.deepcopy(self.cases);report=prompt_audit(self.cases,tokenizer)
        self.assertEqual(len(report),30)
        real=[r for r in report if r['name'].startswith('real_')]
        self.assertEqual(len(real),3);self.assertTrue(all(r['parsed_history_argument_strings']==6 for r in real))
        for world in [1,2]:
            routed=assignments(self.cases,tokenizer,world)
            self.assertEqual(sorted(i for part in routed for i in part),list(range(30)))
        for case in self.cases:
            normalized=hf_messages(case)
            self.assertEqual(len(normalized),len(case['messages']))
            for original,message in zip(case['messages'],normalized):
                if not original.get('tool_calls'):self.assertEqual(original,message)
                else:
                    for a,b in zip(original['tool_calls'],message['tool_calls']):
                        args=a['function']['arguments'];expected=json.loads(args) if isinstance(args,str) else args
                        self.assertEqual(b['function']['arguments'],expected)
                        self.assertEqual(a['function']['name'],b['function']['name'])
            if not case['name'].startswith('real_'):
                raw=tokenizer.apply_chat_template(case['messages'],tools=case['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                self.assertEqual(raw,render_prompt(case,tokenizer))
        self.assertEqual(self.cases,before)

    def test_history_argument_normalization_rejects_malformed_data(self):
        original=copy.deepcopy(next(c for c in self.cases if c['name'].startswith('real_')))
        for invalid in ['[1,2]','not json',None]:
            case=copy.deepcopy(original)
            call=next(c for m in case['messages'] for c in m.get('tool_calls') or [])
            call['function']['arguments']=invalid;before=copy.deepcopy(case)
            with self.assertRaises(ValueError):hf_messages(case)
            self.assertEqual(case,before)

    def test_long_case_routing_uses_all_replicas_and_is_complete(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return [0]*len(messages)
        result=assignments(self.cases,Tokenizer(),2)
        self.assertEqual(sorted(i for part in result for i in part),list(range(30)))
        self.assertTrue(all(result));self.assertEqual(len(assignments(self.cases,Tokenizer(),1)[0]),30)

    def test_unclosed_reasoning_cannot_pass_even_if_raw_tool_parses(self):
        case=self.cases[0];fn=case['answer']['tool_calls'][0]['function']
        r=score(case,dict(tool=fn['name'],arguments=fn['arguments']),'<tool_call>wait</tool_call>',False)
        self.assertFalse(r['valid']);self.assertIn('thinking_not_closed_before_tool',r['errors'])

    def test_wrapper_wait_single_and_dual_pipeline_with_cpu_stub(self):
        from training.train_seller_consistency import main,prepare,check_run
        cfg,_=prepare()
        def fake_launch(folder,config,env):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],','.join(str(i) for i in range(config['world_size'])))
            b=dict(positive_regression=dict(captured=7,truncated=0),heldout_controls=dict(correct=8,truncated=0),
                   official_heldout=dict(valid=6,truncated=0),real_market_development=dict(safe=1,truncated=1))
            for step in [0,18]:(folder/f'native_step_{step:04d}.json').write_text(json.dumps(dict(summary=dict(step=step,behavior=b))))
            (folder/'training_finished.json').write_text(json.dumps(dict(status='stopped_step18_gate',step=18,gate=dict(passed=False))))
        for world in [1,2]:
            cards=[dict(index=str(i),fraction=.85) for i in range(world)]
            selection=[RuntimeError('No GPU has enough free memory after reserve'),(cards,[],{})]
            with tempfile.TemporaryDirectory(prefix='seller_consistency_test_',dir='/tmp') as tmp:
                with patch('training.train_seller_consistency.prepare',return_value=(copy.deepcopy(cfg),self.manifest)),\
                     patch('training.local_resources.training_gpu_plan',side_effect=selection) as plan,\
                     patch('training.local_resources.runtime_cache',return_value=({},None)),\
                     patch('training.train_seller_consistency.time.sleep') as pause,\
                     patch('training.train_seller_consistency.launch',side_effect=fake_launch),redirect_stdout(io.StringIO()):
                    folder=main(['--wait','--gpus','0,1','--output-dir',tmp])
                self.assertEqual(plan.call_count,2);pause.assert_called_once_with(15)
                self.assertTrue((folder/'seller_consistency_review.md').is_file());saved=check_run(folder)
                self.assertEqual(saved['world_size'],world);self.assertEqual(saved['effective_batch_size'],8)
                saved['learning_rate']=1e-3;(folder/'train_config.json').write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError,'recipe changed'):check_run(folder)


if __name__=='__main__':unittest.main()
