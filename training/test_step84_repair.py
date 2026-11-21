"""CPU regression checks for paired interventions, actual engine fixtures and report gates."""
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Step84RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.step84_repair_plan import prepare
        from transformers import AutoTokenizer
        cls.plan=prepare();cls.cases={c['name']:c for c in cls.plan['cases']}
        cls.tokenizer=AutoTokenizer.from_pretrained(cls.plan['base_model'],revision=cls.plan['revision'],local_files_only=True)

    def test_persona_only_interventions_and_original_baseline(self):
        from training.step84_repair_plan import intervention,LATEST
        for c in self.cases.values():
            self.assertEqual(intervention(c,'baseline'),c)
            original=c['messages'][0]['content'];a,rest=original.split('<<<PERSONA_SECTION>>>');_,b=rest.split('<<<END_PERSONA_SECTION>>>')
            for arm in ('latest',):
                changed=intervention(c,arm);new=changed['messages'][0]['content']
                self.assertEqual(changed['messages'][1:],c['messages'][1:]);self.assertEqual(changed['observation'],c['observation'])
                self.assertEqual(changed['tools'],c['tools']);self.assertTrue(new.startswith(a));self.assertTrue(new.endswith(b))
                text=new.split('<<<PERSONA_SECTION>>>')[1].split('<<<END_PERSONA_SECTION>>>')[0]
                self.assertEqual(text.strip(),(LATEST/f'persona_{c["persona_index"]}.md').read_text().strip())
                self.assertLessEqual(len(text),8000)

    def test_new_engine_pairs_have_distinct_worlds_three_personas_and_cash_only_differences(self):
        from training.official_buyer_train import official_snapshot_review
        from training.buyer_operations_v2_data import choose
        fresh=[c for c in self.cases.values() if c['suite']=='fresh'];pairs=defaultdict(list)
        for c in fresh:pairs[c['pair_id']].append(c)
        self.assertEqual(len(pairs),6);self.assertEqual(Counter(c['persona_index'] for c in fresh),{1:4,2:4,3:4})
        self.assertEqual(len({c['origin']['group'] for c in fresh}),6)
        for pair in pairs.values():
            positive=next(c for c in pair if c['positive']);negative=next(c for c in pair if not c['positive'])
            self.assertEqual(positive['evaluation_rank'],negative['evaluation_rank'])
            one=copy.deepcopy(positive['observation']);two=copy.deepcopy(negative['observation'])
            one['you'].pop('cash');two['you'].pop('cash');self.assertEqual(one,two)
            self.assertTrue(positive['observation']['your_unreviewed_trades'])
            self.assertGreater(positive['engine_check']['positive_surplus_cents'],0)
            self.assertEqual(negative['engine_check']['negative_trade_result']['error'],'insufficient_cash')
            tool,args=choose(positive['situation'],positive['observation'])
            self.assertTrue(official_snapshot_review(positive,tool,args)['positive_trade'])
            self.assertEqual(choose(negative['situation'],negative['observation'])[0],'wait')
        self.assertTrue(any(c['observation']['you']['cash']==0 for c in fresh))

    def test_count_cached_provenance_and_pair_routing(self):
        self.assertEqual(self.plan['gpu_calls'],49);self.assertEqual(self.plan['cached_calls'],25)
        self.assertEqual(len(self.plan['jobs']),74)
        routes=defaultdict(set)
        for j in self.plan['jobs']:
            routes[j['case']].add(j['rank'])
            if j['cached'] is not None:self.assertEqual(j['arm'],'baseline');self.assertNotEqual(self.cases[j['case']]['suite'],'fresh')
        self.assertTrue(all(len(r)==1 for r in routes.values()))
        counts=Counter(j['rank'] for j in self.plan['jobs'] if j['cached'] is None)
        self.assertLessEqual(abs(counts[0]-counts[1]),1)

    def native(self,tool,args):
        body='\n'.join(f'<parameter={k}>\n{v}\n</parameter>' for k,v in args.items())
        return f'</think>\n<tool_call>\n<function={tool}>\n{body}\n</function>\n</tool_call><|im_end|>'

    def test_negative_review_is_legal_without_requiring_wait_or_json_private_state(self):
        from training.step84_repair_plan import assess
        negative=next(c for c in self.cases.values() if c['suite']=='fresh' and not c['positive'])
        trade=negative['observation']['your_unreviewed_trades'][0]
        args={'trade_id':trade['trade_id'],'rating':5,'private_state':'ordinary text is permitted'}
        r=assess(negative,'latest',self.native('post_review',args),self.tokenizer,False,negative['evaluation_rank'])
        self.assertTrue(r['valid']);self.assertFalse(r['cash_error']);self.assertFalse(r['captured'])
        positive=next(c for c in self.cases.values() if c['pair_id']==negative['pair_id'] and c['positive'])
        r=assess(positive,'latest',self.native('post_review',args),self.tokenizer,False,positive['evaluation_rank'])
        self.assertTrue(r['valid']);self.assertFalse(r['captured']);self.assertTrue(r['review'])

    def test_cpu_preparation_and_fake_worker_collection_without_gpu(self):
        from training.check_step84_repair import main
        from training.step84_repair_plan import assess,summarize
        from training.buyer_operations_v2_data import choose
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_step84_cpu_') as tmp:
            with patch('sys.argv',['check','--prepare-only','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan') as gpu, \
                 patch('training.check_step84_repair.subprocess.Popen') as process,redirect_stdout(io.StringIO()):
                main();gpu.assert_not_called();process.assert_not_called()
            def fake(cmd,**kwargs):
                folder=Path(cmd[-1]);plan=json.loads((folder/'plan.json').read_text());cases={c['name']:c for c in plan['cases']}
                cached={j['case']:j['cached'] for j in plan['jobs'] if j['cached'] is not None}
                for rank in (0,1):
                    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream:
                        for j in plan['jobs']:
                            if j['cached'] is not None or j['rank']!=rank:continue
                            case=cases[j['case']]
                            if case['suite']=='official':text=cached[j['case']]['text']
                            else:
                                tool,args=choose(case['situation'],case['observation']);text=self.native(tool,args)
                            row=assess(case,j['arm'],text,self.tokenizer,False,rank)
                            stream.write(json.dumps(row)+'\n')
                return SimpleNamespace(stdout=io.BytesIO(b'CPU FAKE WORKERS: no GPU model execution\n'),wait=lambda:0)
            cards=[{'index':str(i),'fraction':.85} for i in (0,1)]
            with patch('sys.argv',['check','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.check_step84_repair.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):
                folder=main()
            rows=json.loads((folder/'decisions.json').read_text());self.assertEqual(len(rows),74)
            self.assertEqual(sum(r['cached'] for r in rows),25)
            report=summarize(rows);self.assertTrue(report['complete']);self.assertTrue(report['eligible_for_small_market'])
            # Missing evidence may never become a passing result.
            self.assertFalse(summarize(rows[:-1])['eligible_for_small_market'])


if __name__=='__main__':unittest.main()
