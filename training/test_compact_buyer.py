"""CPU-only fixed-weight compact-persona evaluation tests."""
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

ROOT=Path(__file__).resolve().parents[1]


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from training.evaluate_compact_buyer import prepare,DEFAULT
        cls.plan=prepare(DEFAULT);cls.t=AutoTokenizer.from_pretrained(cls.plan['base_model'],revision=cls.plan['revision'],local_files_only=True)

    def test_only_diagnostic_personas_change_and_empty_native_think(self):
        from training.evaluate_compact_buyer import prompt
        source=json.loads((ROOT/'training/data/buyer_dollar_lines_v1/evaluation_cases.json').read_text());old={c['name']:c for c in source}
        for c in self.plan['cases']:
            original=old[c['name']]
            for field in ('observation','situation','tools','answer'):self.assertEqual(c[field],original[field])
            self.assertEqual(c['messages'][1:],original['messages'][1:])
            a,b=original['messages'][0]['content'],c['messages'][0]['content']
            self.assertEqual(a.split('<<<PERSONA_SECTION>>>')[0],b.split('<<<PERSONA_SECTION>>>')[0])
            self.assertEqual(a.split('<<<END_PERSONA_SECTION>>>')[1],b.split('<<<END_PERSONA_SECTION>>>')[1])
            self.assertTrue(prompt(self.t,c).endswith('<think>\n'))
        for text in self.plan['compact_personas'].values():
            self.assertLessEqual(len(text),2000);self.assertNotIn('CASH=',text)
        self.assertTrue(all(r['compact']<r['original'] for r in self.plan['prompt_lengths']))

    def test_fixed_step84_exact36_cases_routes_and_cached_controls(self):
        p=self.plan;self.assertTrue(p['adapter'].endswith('step_0084'));self.assertEqual(p['training_updates'],0)
        self.assertEqual(Counter(c['suite'] for c in p['cases']),{'self_check_seen':12,'self_check_validation':24})
        self.assertEqual(Counter(j['rank'] for j in p['jobs']),{0:18,1:18})
        self.assertEqual(Counter(c['positive'] for c in p['cases']),{True:18,False:18})
        pairs=defaultdict(set)
        for c in p['cases']:pairs[c['pair_id']].add(c['evaluation_rank'])
        self.assertTrue(all(len(r)==1 for r in pairs.values()))
        self.assertTrue(all(r['cached'] for r in p['baseline']));self.assertEqual(sum(r['captured'] for r in p['baseline']),13)

    def test_all_wait_cannot_pass_and_one_persona_regression_is_visible(self):
        from training.evaluate_compact_buyer import summary,operation_gates
        parent=summary(self.plan['baseline']);waiting=copy.deepcopy(self.plan['baseline'])
        for r in waiting:r.update(valid=True,captured=False,cash_error=False,errors=[],tool='wait',truncated=False)
        current=summary(waiting);self.assertFalse(all(operation_gates(parent,current).values()))
        good=copy.deepcopy(parent)
        for suite,s in good.items():
            s.update(valid=s['calls'],cash_errors=0,owned_errors=0,not_current_board=0,no_tool=0,truncated=0)
        self.assertTrue(all(operation_gates(parent,good).values()))
        shifted=copy.deepcopy(good);shifted['self_check_seen']['personas']['1']['captured']-=1
        self.assertFalse(operation_gates(parent,shifted)['self_check_seen_each_persona_retained'])

    def test_semantic_quotes_are_pending_not_fake_accuracy_or_private_state(self):
        from training.evaluate_compact_buyer import review_evidence
        c=self.plan['cases'][0];r=copy.deepcopy(self.plan['baseline'][0])
        r['text']='No facts checked.\n</think>\n<tool_call>{"cash":19.59,"held":[]}</tool_call>'
        packet=review_evidence(c,r)
        self.assertTrue(all(f['status']=='pending_review' for f in packet['fields'].values()))
        self.assertEqual(packet['fields']['cash']['candidate_quotes'],[])
        self.assertNotIn('<tool_call>',packet['reason'])

    def test_prepare_and_fake36_launch_report_preserve_real_personas(self):
        from training.evaluate_compact_buyer import main,prompt,assess_case,write_report
        paths=[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)];original={p:p.read_bytes() for p in paths}
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_compact_cpu_') as tmp:
            with patch('sys.argv',['compact','--prepare-only','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan') as gpu, \
                 patch('training.evaluate_compact_buyer.subprocess.Popen') as proc,redirect_stdout(io.StringIO()):
                main();gpu.assert_not_called();proc.assert_not_called()
            def fake(cmd,**kwargs):
                self.assertIn('--nproc_per_node=2',cmd);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                folder=Path(cmd[-1]);p=json.loads((folder/'plan.json').read_text());cases={c['name']:c for c in p['cases']}
                for rank in (0,1):
                    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream:
                        for job in p['jobs']:
                            if job['rank']!=rank:continue
                            c=cases[job['case']];start=prompt(self.t,c)
                            full=self.t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
                            self.assertTrue(full.startswith(start))
                            r=assess_case(c,full[len(start):],self.t,False,rank);self.assertTrue(r['valid'])
                            stream.write(json.dumps(r)+'\n')
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock:36 golden outputs, zero model execution\n'),wait=lambda:0)
            cards=[dict(index=str(i),fraction=.85) for i in (0,1)]
            with patch('sys.argv',['compact','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.evaluate_compact_buyer.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):
                folder=main()
            result=json.loads((folder/'compact_review.json').read_text());self.assertTrue(all(result['operation_gates'].values()))
            self.assertEqual(result['semantic_review'],'pending');self.assertFalse(result['automatic_market_or_promotion'])
            self.assertEqual(len(json.loads((folder/'semantic_review.json').read_text())),72)
            for p,data in original.items():self.assertEqual(p.read_bytes(),data)
            path=folder/'decisions_rank0.jsonl';path.write_text(path.read_text().splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete/repeated'):write_report(folder,self.plan)


if __name__=='__main__':unittest.main()
