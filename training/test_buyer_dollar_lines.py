"""CPU dollar units, independent reason loss, provenance, and mandatory step6 gates."""
from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data/buyer_dollar_lines_v1'


class DollarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from transformers import AutoTokenizer
        from training.train_buyer_dollar_lines import prepare
        cls.cfg,cls.manifest,cls.batches=prepare(yaml.safe_load((ROOT/'training/configs/buyer_dollar_lines_v1.yaml').read_text()))
        cls.train=[json.loads(l) for l in (DATA/'train.jsonl').read_text().splitlines()]
        cls.tokens=[json.loads(l) for l in (DATA/'train.tokens.jsonl').read_text().splitlines()]
        cls.cases=json.loads((DATA/'evaluation_cases.json').read_text())
        cls.t=AutoTokenizer.from_pretrained(cls.cfg['base_model'],revision=cls.cfg['revision'],local_files_only=True)

    def test_inputs_personas_pairs_and_no_new_worlds(self):
        original=[json.loads(l) for l in (ROOT/'training/data/buyer_self_checks_v1/train.jsonl').read_text().splitlines()]
        for old,new in zip(original,self.train):
            for field in ('id','messages','tools','observation','situation','pair_id','persona_index'):self.assertEqual(old[field],new[field])
            self.assertEqual(old['answer']['tool_calls'],new['answer']['tool_calls'])
            reason=new['answer']['reasoning_content'];self.assertTrue(reason.startswith('CASH='));self.assertNotIn('{',reason)
            self.assertEqual(reason.count('ROW='),1)
        self.assertFalse(set(self.manifest['groups']['train'])&set(self.manifest['groups']['validation']))
        for start in (0,6):self.assertEqual(Counter(i for b in self.batches[start:start+6] for i in b),Counter(range(48)))
        for batch in self.batches:
            self.assertEqual(Counter(self.train[i]['label_action'] for i in batch),{'buy_listed':2,'accept_offer':2,'wait':4})
            self.assertTrue(all(n==2 for n in Counter(self.train[i]['pair_id'] for i in batch).values()))

    def test_dollar_units_not_cents_and_one_bad_field_not_all_zero(self):
        from training.buyer_dollar_lines import render,score
        c=next(r for r in self.train if not r['positive'] and r['observation']['you']['cash']==19.59)
        text=render(c);self.assertIn('CASH=19.59',text);self.assertIn('total=19.60',text)
        self.assertTrue(score(c,text)['exact'])
        bad=score(c,text.replace('CASH=19.59','CASH=1959'));self.assertFalse(bad['target_fields']['cash']);self.assertTrue(bad['target_fields']['total'])
        bad=score(c,text.replace('total=19.60','total=19.601'));self.assertFalse(bad['target_fields']['total']);self.assertTrue(bad['target_fields']['cash'])
        self.assertTrue(score(c,text.replace(' | ',', ').replace('=',' = '))['exact'])
        bad=score(c,text.replace('afford=0','afford=1'));self.assertFalse(bad['target_fields']['affordable']);self.assertTrue(bad['target_fields']['held'])

    def test_gold_reason_native_tool_and_state_consistency_all48(self):
        from training.buyer_dollar_lines import assess,validate_gold,state_errors
        for row,token in zip(self.train,self.tokens):
            self.assertTrue(validate_gold(row))
            text=self.t.decode(token['input_ids'][token['prompt_length']:],skip_special_tokens=False)
            case={**row,'suite':'self_check_seen'}
            r=assess(case,text,self.t,False,0)
            self.assertTrue(r['valid']);self.assertTrue(r['line_check']['exact']);self.assertTrue(r['line_action_consistent']);self.assertFalse(r['state_errors'])
        c=self.train[0];fn=c['answer']['tool_calls'][0]['function'];args=copy.deepcopy(fn['arguments']);state=json.loads(args['private_state'])
        state['cash']=999;args['private_state']=json.dumps(state)
        self.assertIn('state_cash',state_errors(c,fn['name'],args))
        state['held']=[['wrong']];args['private_state']=json.dumps(state)
        self.assertIn('state_held',state_errors(c,fn['name'],args))
        self.assertTrue(state_errors(c,fn['name'],None))

    def test_full_check_start_and_closing_think_have_primary_supervision(self):
        for row,tok in zip(self.train,self.tokens):
            p=tok['prompt_length'];check=tok['check_positions'];response=tok['response_positions']
            prompt=self.t.apply_chat_template(row['messages'],tools=row['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            self.assertEqual(self.t.encode(prompt,add_special_tokens=False),tok['input_ids'][:p])
            text=self.t.decode([tok['input_ids'][p+i] for i in check],skip_special_tokens=False)
            self.assertTrue(text.startswith('CASH='));self.assertIn('</think>',text);self.assertNotIn('private_state',text)
            self.assertEqual(set(check)|set(response),set(range(tok['target_length'])));self.assertFalse(set(check)&set(response))
            self.assertTrue(all(set(s['positions'])<=set(check) for s in tok['field_spans']))
            self.assertLessEqual(len(check),60)

    def test_independent_line_field_response_loss_and_cpu_gradient(self):
        import torch
        from training.buyer_dollar_lines import FIELDS,losses
        logits=torch.nn.Parameter(torch.zeros(1,18,5))
        class Tiny:
            def __call__(self,**kwargs):
                if kwargs['logits_to_keep'].tolist()!=list(range(1,19)):raise AssertionError('Wrong shift')
                return SimpleNamespace(logits=logits)
        row=dict(input_ids=[0,1]+[i%5 for i in range(18)],prompt_length=2,check_positions=list(range(8)),response_positions=list(range(8,18)),
            field_spans=[dict(field=f,positions=[i+1]) for i,f in enumerate(FIELDS)])
        got=losses(Tiny(),row,torch.device('cpu'))
        ce=torch.nn.functional.cross_entropy(logits.reshape(-1,5),torch.tensor(row['input_ids'][2:]),reduction='none')
        self.assertTrue(torch.allclose(got['total'],ce[:8].mean()+ce[1:8].mean()+.25*ce[8:].mean()))
        got['total'].backward();self.assertGreater(logits.grad[0,0].abs().sum().item(),0)
        self.assertGreater(logits.grad[0,0].abs().sum().item(),logits.grad[0,-1].abs().sum().item())

    def test_seen12_gate_rejects_missing_line_cent_buy_owned_buy_and_all_wait(self):
        from training.buyer_dollar_lines import assess,summarize,step6_gate
        from training.buyer_dollar_train import summarize as all_summary,write_review
        original={r['id']:t for r,t in zip(self.train,self.tokens)};rows=[]
        seen=[c for c in self.cases if c['suite']=='self_check_seen']
        self.assertEqual(Counter(c['evaluation_rank'] for c in seen),{0:6,1:6})
        for c in seen:
            tok=original[c['id']];text=self.t.decode(tok['input_ids'][tok['prompt_length']:],skip_special_tokens=False)
            rows.append(assess(c,text,self.t,False,c['evaluation_rank']))
        good=summarize(rows);self.assertTrue(all(step6_gate(good).values()))
        for field in ('self_started','all_fields_exact','captured','cash_negative_safe','owned_negative_safe','state_consistent','line_action_consistent'):
            bad=copy.deepcopy(good);bad[field]-=1;self.assertFalse(all(step6_gate(bad).values()),field)
        waiting=copy.deepcopy(good);waiting['captured']=0;self.assertFalse(all(step6_gate(waiting).values()))
        summary=all_summary(rows);self.assertIsNone(summary['self_check_validation']);self.assertIsNone(summary['regression'])
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_dollar_report_cpu_') as tmp:
            folder=Path(tmp);(folder/'native_step_0006.json').write_text(json.dumps(dict(step=6,summary=summary)))
            (folder/'finished.json').write_text(json.dumps(dict(status='stopped_step6_gate',step=6)))
            write_review(folder);self.assertIn('stopped_step6_gate',(folder/'dollar_review.md').read_text())

    def test_cpu_and_stopped_launcher_are_complete_without_any_gpu_execution(self):
        from training.train_buyer_dollar_lines import main
        with patch('sys.argv',['dollar','--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_dollar_lines.subprocess.Popen') as process,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();process.assert_not_called()
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_dollar_launch_cpu_') as tmp:
            def fake(cmd,**kwargs):
                self.assertIn('--nproc_per_node=2',cmd);self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
                folder=Path(cmd[-1]);(folder/'finished.json').write_text(json.dumps(dict(status='stopped_step6_gate',step=6)))
                return SimpleNamespace(stdout=io.BytesIO(b'CPU mock; zero GPU execution\n'),wait=lambda:0)
            cards=[dict(index=str(i),budget_gib=20,fraction=.85) for i in (0,1)]
            with patch('sys.argv',['dollar','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.train_buyer_dollar_lines.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):
                folder=main()
            self.assertEqual(json.loads((folder/'wrapper_result.json').read_text())['exit'],0)


if __name__=='__main__':unittest.main()
