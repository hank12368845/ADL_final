"""CPU checks for answer-free diagnostic personas and native free-generation setup."""
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class InstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from training.diagnose_dollar_instruction import prepare,DEFAULT
        cls.plan=prepare(DEFAULT);cls.t=AutoTokenizer.from_pretrained(cls.plan['base_model'],revision=cls.plan['revision'],local_files_only=True)

    def test_only_persona_copy_changes_original_observation_tools_untouched(self):
        from training.diagnose_dollar_instruction import add_instruction,INSTRUCTION
        source=json.loads((ROOT/'training/data/buyer_dollar_lines_v1/evaluation_cases.json').read_text())
        lookup={c['name']:c for c in source}
        for c in self.plan['cases']:
            old=lookup[c['name']];before=copy.deepcopy(old)
            new,variant=add_instruction(old,self.plan['original_personas'][str(c['persona_index'])])
            self.assertEqual(old,before);self.assertEqual(new['messages'],c['messages'])
            for field in ('observation','situation','tools','answer'):self.assertEqual(new[field],old[field])
            self.assertEqual(new['messages'][1:],old['messages'][1:])
            original=old['messages'][0]['content'];modified=new['messages'][0]['content']
            self.assertEqual(original.split('<<<PERSONA_SECTION>>>')[0],modified.split('<<<PERSONA_SECTION>>>')[0])
            self.assertEqual(original.split('<<<END_PERSONA_SECTION>>>')[1],modified.split('<<<END_PERSONA_SECTION>>>')[1])
            self.assertEqual(variant,self.plan['original_personas'][str(c['persona_index'])]+'\n\n'+INSTRUCTION)
            self.assertLessEqual(len(variant),8000)
        self.assertEqual(self.plan['persona_chars'],{'1':6734,'2':7931,'3':7362})

    def test_instruction_has_no_actual_facts_and_think_starts_empty(self):
        from training.diagnose_dollar_instruction import INSTRUCTION,prompt
        self.assertEqual(len(INSTRUCTION),329)
        self.assertNotRegex(INSTRUCTION,r'\d+\.\d+')
        for c in self.plan['cases']:
            self.assertNotIn(c['target_product'],INSTRUCTION)
            for offer in c['observation']['offers_to_you']:self.assertNotIn(offer['offer_id'],INSTRUCTION)
            text=prompt(self.t,c);self.assertTrue(text.endswith('<think>\n'))
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),c['prompt_sha256'])
            self.assertNotEqual(c['prompt_sha256'],c['original_prompt_sha256'])
            self.assertEqual(text.rsplit('<think>',1)[1],'\n')

    def test_same12_cases_and_pairs_two_adapters_two_cards(self):
        p=self.plan;self.assertEqual(p['training_updates'],0);self.assertEqual(len(p['jobs']),24)
        self.assertEqual(Counter(j['rank'] for j in p['jobs']),{0:12,1:12})
        self.assertEqual(Counter(c['positive'] for c in p['cases']),{True:6,False:6})
        pair_ranks=defaultdict(set)
        for c in p['cases']:
            pair_ranks[c['pair_id']].add(c['evaluation_rank'])
            self.assertEqual({j['rank'] for j in p['jobs'] if j['case']==c['name']},{c['evaluation_rank']})
        self.assertTrue(all(len(r)==1 for r in pair_ranks.values()))
        self.assertEqual(sum(r['captured'] for r in p['free_cached']['parent84'].values()),6)
        self.assertEqual(sum(r['captured'] for r in p['free_cached']['dollar6'].values()),5)

    def test_gold_native_response_and_wrong_verdict_scored_independently(self):
        from training.buyer_dollar_lines import assess,summarize,step6_gate
        from training.diagnose_dollar_instruction import prompt
        rows=[]
        for c in self.plan['cases']:
            start=prompt(self.t,c);full=self.t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
            self.assertTrue(full.startswith(start));text=full[len(start):]
            r=assess(c,text,self.t,False,c['evaluation_rank']);self.assertTrue(r['valid']);self.assertTrue(r['line_check']['exact']);rows.append(r)
        self.assertTrue(all(step6_gate(summarize(rows)).values()))
        c=next(c for c in self.plan['cases'] if c['mode']=='cash' and not c['positive'])
        start=prompt(self.t,c);full=self.t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
        bad=full[len(start):].replace('afford=0','afford=1',1)
        r=assess(c,bad,self.t,False,c['evaluation_rank'])
        self.assertTrue(r['line_check']['self_started']);self.assertTrue(r['valid'])
        self.assertFalse(r['line_check']['target_fields']['affordable']);self.assertFalse(r['line_check']['exact'])

    def test_cpu_prepare_and_mock_two_card_launch_do_not_change_agents(self):
        from training.diagnose_dollar_instruction import main,prompt,write_report
        from training.buyer_dollar_lines import assess
        paths=[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)]
        before={p:p.read_bytes() for p in paths}
        # Initialize the real native parser before mocking subprocess.Popen;
        # its platform detection may itself run the harmless uname command.
        c=self.plan['cases'][0];start=prompt(self.t,c)
        full=self.t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
        self.assertTrue(assess(c,full[len(start):],self.t,False,c['evaluation_rank'])['valid'])
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_instruction_cpu_') as tmp:
            with patch('sys.argv',['instruction','--prepare-only','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan') as gpu, \
                 patch('training.diagnose_dollar_instruction.subprocess.Popen') as proc,redirect_stdout(io.StringIO()):
                main();gpu.assert_not_called();proc.assert_not_called()
            def fake(cmd,**kwargs):
                self.assertIn('--nproc_per_node=2',cmd);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                folder=Path(cmd[-1]);plan=json.loads((folder/'plan.json').read_text());cases={c['name']:c for c in plan['cases']}
                for rank in (0,1):
                    with (folder/f'decisions_rank{rank}.jsonl').open('w') as stream:
                        for job in plan['jobs']:
                            if job['rank']!=rank:continue
                            c=cases[job['case']];start=prompt(self.t,c)
                            full=self.t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
                            r=assess(c,full[len(start):],self.t,False,rank);r.update(adapter=job['adapter'],condition='format_instruction')
                            stream.write(json.dumps(r)+'\n')
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock only; no model inference\n'),wait=lambda:0)
            cards=[dict(index=str(i),fraction=.85) for i in (0,1)]
            with patch('sys.argv',['instruction','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.diagnose_dollar_instruction.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):
                folder=main()
            r=json.loads((folder/'instruction_review.json').read_text());self.assertEqual(r['training_updates'],0)
            for label in self.plan['adapters']:
                self.assertTrue(all(r['gates'][label].values()));self.assertEqual(r['summary'][label]['format_instruction']['all_fields_exact'],12)
            for p,content in before.items():self.assertEqual(p.read_bytes(),content)
            log=folder/'decisions_rank0.jsonl';log.write_text(log.read_text().splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete/repeated'):write_report(folder,self.plan)


if __name__=='__main__':unittest.main()
