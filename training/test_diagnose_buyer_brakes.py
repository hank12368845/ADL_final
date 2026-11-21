from contextlib import redirect_stdout
from collections import Counter
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ReasonToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.diagnose_buyer_brakes import prepare,DEFAULT
        cls.plan=prepare(DEFAULT)

    def test_fact_prefix_has_current_truth_without_action_names_or_mutating_case(self):
        from training.diagnose_buyer_brakes import fact_reason
        for c in self.plan['cases']:
            before=copy.deepcopy(c);text=fact_reason(c);self.assertEqual(c,before)
            self.assertIn('CURRENT cash=',text);self.assertIn('held_count=',text);self.assertIn('cash>=total=',text)
            for word in ('buy_listed','accept_offer','wait','<tool_call>'):self.assertNotIn(word,text)
            if c['mode']=='cash' and not c['positive']:self.assertNotIn('eligible=true',text)
            if c['mode']=='owned' and not c['positive']:self.assertNotIn('eligible=true',text)
            if c['positive']:self.assertIn('eligible=true',text)

    def test_same_case_and_pair_routes_both_checkpoints_balanced(self):
        plan=self.plan;self.assertEqual(len(plan['cases']),12);self.assertEqual(len(plan['jobs']),24)
        self.assertEqual(Counter(j['rank'] for j in plan['jobs']),{0:12,1:12})
        self.assertEqual(Counter(c['positive'] for c in plan['cases']),{True:6,False:6})
        for c in plan['cases']:
            self.assertEqual({j['rank'] for j in plan['jobs'] if j['case']==c['name']},{c['evaluation_rank']})
            self.assertEqual(c['messages'][0]['content'].split('<<<PERSONA_SECTION>>>')[1].split('<<<END_PERSONA_SECTION>>>')[0].strip(),
                Path(f'agents/example_student/buyer/persona_{c["persona_index"]}.md').read_text().strip())

    def test_native_reasoning_boundary_and_tool_parser(self):
        from transformers import AutoTokenizer
        from training.buyer_hf_check import parse_raw
        t=AutoTokenizer.from_pretrained(self.plan['base_model'],revision=self.plan['revision'],local_files_only=True)
        for c in self.plan['cases']:
            prompt=t.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            self.assertTrue(prompt.endswith('<think>\n'))
            whole=t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,add_generation_prompt=False,enable_thinking=True)
            tail=whole[len(prompt):].split('</think>',1)[1]
            parsed=parse_raw(t,tail,c['tools']);self.assertEqual(parsed['parsed_calls'],1)
            self.assertEqual(parsed['tool'],c['answer']['tool_calls'][0]['function']['name'])

    def test_cpu_prepare_and_fake_launcher_collection(self):
        from training.diagnose_buyer_brakes import main
        from training.step84_repair_plan import assess
        from transformers import AutoTokenizer
        t=AutoTokenizer.from_pretrained(self.plan['base_model'],revision=self.plan['revision'],local_files_only=True)
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_reason_tool_cpu_') as tmp:
            with patch('sys.argv',['diag','--prepare-only','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan') as gpu, \
                 patch('training.diagnose_buyer_brakes.subprocess.Popen') as process,redirect_stdout(io.StringIO()):
                main();gpu.assert_not_called();process.assert_not_called()
            def fake(cmd,**kwargs):
                folder=Path(cmd[-1]);plan=json.loads((folder/'plan.json').read_text());cases={c['name']:c for c in plan['cases']}
                for rank in (0,1):
                    with (folder/f'decisions_rank{rank}.jsonl').open('w') as f:
                        for job in plan['jobs']:
                            if job['rank']!=rank:continue
                            c=cases[job['case']];full=t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
                            tail=full.rsplit('</think>',1)[1];r=assess(c,'latest',tail,t,False,rank);r['adapter']=job['adapter'];f.write(json.dumps(r)+'\n')
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock: no model execution\n'),wait=lambda:0)
            cards=[{'index':str(i),'fraction':.85} for i in (0,1)]
            with patch('sys.argv',['diag','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.diagnose_buyer_brakes.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):folder=main()
            result=json.loads((folder/'reason_tool_review.json').read_text())
            self.assertEqual(result['training_updates'],0)
            for model in result['summary'].values():
                self.assertEqual(model['fact_prefix']['positive_captured'],6);self.assertEqual(model['fact_prefix']['negative_safe'],6)


if __name__=='__main__':unittest.main()
