"""Train-seen provenance and failure-sensitive reporting; no real model execution."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io

ROOT=Path(__file__).resolve().parents[1]


class LearningSanityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.buyer_learning_sanity import prepare,DEFAULT_RUN
        cls.cases,cls.adapters,cls.provenance=prepare(DEFAULT_RUN)

    def test_exact_seen_prompts_no_teacher_answers_and_balanced_routing(self):
        from training.cash_decision_report import case_pairs
        self.assertEqual(len(self.cases),32)
        pairs=case_pairs(self.cases);self.assertEqual(len(pairs),16)
        self.assertTrue(self.provenance['prompt_prefix_identical_to_training_tokens'])
        route={c['name']:i%2 for i,c in enumerate(self.cases)}
        for sides in pairs.values():
            self.assertEqual(route[sides[True]['name']],route[sides[False]['name']])
        self.assertEqual(Counter(route.values()),{0:16,1:16})
        for c in self.cases:
            self.assertEqual(c['origin']['split'],'train_seen')
            self.assertEqual(c['messages'][-1]['role'],'user')
            self.assertNotIn('answer',c)
            self.assertNotIn('positive',json.loads(c['messages'][-1]['content']))
        self.assertEqual(set(self.adapters),{'parent_on','repair32'})

    def test_report_separates_capture_from_all_wait_and_rejects_incomplete(self):
        from training.buyer_learning_sanity import write_review
        from training.buyer_diagnostics import evaluate
        rows=[]
        for i,c in enumerate(self.cases):
            for model in self.adapters:
                obs=c['observation'];tool=c['expected_tool'] if model=='parent_on' else 'wait'
                args={'private_state':'{"bad":[],"pending":null,"last_buy":null}'}
                if tool=='buy_listed':
                    b=obs['board'][0];args.update(seller_id=b['seller'],product_id=b['product_id'],qty=1)
                    args['private_state']=json.dumps({'bad':[],'pending':None,'last_buy':[b['seller'],b['product_id']]})
                elif tool=='accept_offer':args['offer_id']=obs['offers_to_you'][0]['offer_id']
                check=evaluate(c,tool,args)
                rows.append({'case':c['name'],'adapter':model,'persona':model+'/operations_reasoning_p3','seed':0,
                    'gpu':str(i%2),'input_sha256':c['name'],'service_ok':True,'action_ok':True,'policy_ok':True,
                    'state_ok':True,'tool':tool,'operation_capture':tool!='wait','reviewed_evaluation':check})
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_seen_review_cpu_') as tmp:
            p=Path(tmp);(p/'plan.json').write_text(json.dumps({'calls':64,'experiment_manifest':self.provenance}))
            (p/'cases.json').write_text(json.dumps(self.cases))
            (p/'scored_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with redirect_stdout(io.StringIO()):report=write_review(p)
            self.assertEqual(report['models']['parent_on']['overall']['both_correct'],16)
            self.assertEqual(report['models']['repair32']['overall']['both_correct'],0)
            self.assertIn('NOT held-out',report['scope'])
            (p/'scored_decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError,'Incomplete'):write_review(p)


if __name__=='__main__':unittest.main()
