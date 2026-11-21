"""CPU-only full-data audit, native-policy replay, financial guards and controller checks."""
from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.buyer_binding_data import generate,check_data
        from transformers import AutoTokenizer
        cls.tmp=tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_binding_test_')
        cls.data=Path(cls.tmp.name)/'data'
        with redirect_stdout(io.StringIO()):cls.manifest=generate(cls.data)
        cls.train,cls.val,cls.batches,_=check_data(cls.data)
        cls.rows=[json.loads(l) for l in (cls.data/'train.jsonl').read_text().splitlines()]
        cls.cases=json.loads((cls.data/'evaluation_cases.json').read_text())
        cls.t=AutoTokenizer.from_pretrained(cls.manifest['base_model'],revision=cls.manifest['revision'],local_files_only=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_full_coverage_split_native_targets_and_real_cash_history(self):
        from training.buyer_binding_data import candidates
        self.assertEqual(len(self.train),384);self.assertEqual(len(self.val),144)
        self.assertFalse(set(self.manifest['groups']['train'])&set(self.manifest['groups']['validation']))
        self.assertEqual(Counter(i for b in self.batches for i in b),Counter(range(384)))
        self.assertEqual(self.manifest['labels'],{'buy_listed':240,'accept_offer':96,'wait':48})
        for r in self.rows:
            self.assertTrue(r['engine_result']['ok'])
            self.assertEqual(round(r['observation']['you']['cash']*100),r['engine_budget_cents']-r['engine_spent_cents'])
            if r['positive']:self.assertGreater(r['teacher_actual_gain_cents'],0)
            else:self.assertFalse(candidates(r))
            body=r['messages'][0]['content'].split('<<<PERSONA_SECTION>>>',1)[1].split('<<<END_PERSONA_SECTION>>>',1)[0].strip()
            self.assertEqual(body,(ROOT/f'agents/example_student/buyer/persona_{r["persona_index"]}.md').read_text().strip())
        self.assertLessEqual(self.manifest['max_think_tokens'],300)
        self.assertGreater(self.manifest['max_tokens'],12288)

    def test_failed_pair_pivot_and_now_valid_retry_are_both_present(self):
        for family in ('pivot_seller','pivot_product','new_listing'):
            for r in [c for c in self.rows if c['family']==family]:
                previous=next(m for m in reversed(r['messages'][:-1]) if m['role']=='assistant')['tool_calls'][0]['function']['arguments']
                current=r['answer']['tool_calls'][0]['function']['arguments']
                before=(previous['seller_id'],previous['product_id']);after=(current['seller_id'],current['product_id'])
                self.assertEqual(r['last_result']['error'],'not_listed' if family=='new_listing' else 'no_stock')
                if family=='new_listing':self.assertEqual(before,after)
                else:self.assertNotEqual(before,after)
                if family=='pivot_seller':self.assertEqual(before[1],after[1])
                if family=='pivot_product':self.assertNotEqual(before[1],after[1])

    def test_wire_schema_and_native_tool_parser_accept_teacher_closure(self):
        from training.buyer_binding_eval import validate_wire
        from training.buyer_binding_data import FAMILIES
        from training.buyer_hf_check import parse_raw
        self.assertEqual(validate_wire(self.cases),90)
        for family in FAMILIES:
            c=next(c for c in self.rows if c['family']==family)
            prefix=self.t.apply_chat_template(c['messages'],tools=c['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            full=self.t.apply_chat_template(c['messages']+[c['answer']],tools=c['tools'],tokenize=False,enable_thinking=True)
            self.assertTrue(full.startswith(prefix));self.assertIn('<|im_end|>',full[len(prefix):])
            parsed=parse_raw(self.t,full[len(prefix):],c['tools'])
            self.assertEqual(parsed['parsed_calls'],1)
            self.assertEqual(parsed['tool'],c['label_action'])

    def test_all90_policy_inputs_match_frozen_messages_and_official_cap(self):
        from training.buyer_binding_eval import make_policy
        from adlmarket.config import ModelSpec
        from adlmarket.client import CallResult,ToolCall
        class Client:
            def complete(inner,**kwargs):
                inner.kwargs=kwargs
                return CallResult(text='',reasoning='Wait.',stop_reason='tool_calls',stop_sequence_hit=None,
                    usage={'output_tokens':10,'reasoning_tokens':0},timing={},outcome='accepted',tool_call=ToolCall('wait',{}))
        self.assertEqual(len(self.cases),90)
        self.assertEqual(Counter(c['evaluation_rank'] for c in self.cases),{0:45,1:45})
        for c in self.cases:
            client=Client();policy=make_policy(c,ModelSpec(model='Qwen/Qwen3.5-4B'),client)
            d=policy.decide(c['observation'])
            self.assertEqual(client.kwargs['messages'],c['messages'])
            self.assertEqual(client.kwargs['max_tokens'],min(8192,c['observation']['output_tokens_left_this_round']))
            self.assertEqual((client.kwargs['temperature'],client.kwargs['top_p']),(.7,.95))
            self.assertTrue(client.kwargs['extra_body']['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(d.tool,'wait')

    def test_gold_actions_valid_and_all_wait_cannot_pass_capture_gates(self):
        from training.buyer_binding_eval import assess,summaries,gates
        from training.buyer_binding_data import encode
        old={c['name']:c for c in json.loads((ROOT/'training/data/buyer_dollar_lines_v1/evaluation_cases.json').read_text())}
        golden=[];waiting=[]
        for c in self.cases:
            answer=c.get('answer',old.get(c['name'],{}).get('answer'))
            fn=answer['tool_calls'][0]['function']
            r=assess(c,fn['name'],fn['arguments'],{'outcome':'accepted','reasoning':answer['reasoning_content']},self.t)
            self.assertTrue(r['valid'],(c['name'],r['errors']))
            golden.append(r);waiting.append(assess(c,'wait',{},dict(outcome='accepted',reasoning='Wait.'),self.t))
        reference=summaries(golden);current=summaries(waiting)
        self.assertEqual(current['binding_validation']['captured'],0)
        self.assertFalse(all(gates(reference,current).values()))
        child=copy.deepcopy(reference)
        child['legacy36']['personas']['3']['cash_errors']+=1
        self.assertFalse(gates(reference,child)['legacy36_cash_owned_retained'])

    def test_controller_cpu_check_and_full_stage_sequence_never_promote(self):
        from training.train_buyer_binding import main
        from training.buyer_binding_eval import summaries,assess
        protected=[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)]+[ROOT/'agents/example_student/buyer/checkpoint/adapter_model.safetensors']
        before={p:p.read_bytes() for p in protected}
        with patch('sys.argv',['binding','--cpu-check','--data-dir',str(self.data)]),patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_binding.stage') as runner,redirect_stdout(io.StringIO()):
            cfg=main();gpu.assert_not_called();runner.assert_not_called()
        self.assertEqual((cfg['learning_rate'],cfg['warmup_steps'],cfg['max_steps']),(1e-5,4,48))
        self.assertEqual(cfg['rank'],16)
        stages=[]
        def fake_stage(folder,cfg,kind,env,labels=None):
            stages.append((kind,labels));self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
            if kind=='train':
                for n in (24,48):
                    cp=folder/'checkpoints'/f'step_{n:04d}';cp.mkdir(parents=True)
                    (cp/'checkpoint_complete.json').write_text(json.dumps({'step':n}))
                (folder/'training_finished.json').write_text('{}')
            if labels:
                for label in labels:
                    rows=[assess(c,'wait',{},dict(outcome='accepted',reasoning='Wait.'),self.t) for c in self.cases]
                    (folder/f'evaluation_{label}.json').write_text(json.dumps(dict(summary=summaries(rows),decisions=rows)))
        cards=[dict(index=str(i),fraction=.85,budget_gib=20.4) for i in (0,1)]
        with patch('sys.argv',['binding','--data-dir',str(self.data),'--output-dir',self.tmp.name]), \
             patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
             patch('training.local_resources.runtime_cache',return_value=({},None)), \
             patch('training.train_buyer_binding.stage',side_effect=fake_stage),redirect_stdout(io.StringIO()):folder=main()
        self.assertEqual(stages,[('preflight',None),('eval_parent',['parent']),('train',None),('eval_candidates',['24','48'])])
        result=json.loads((folder/'binding_review.json').read_text())
        self.assertTrue(result['complete']);self.assertIsNone(result['development_candidate']);self.assertFalse(result['automatic_promotion'])
        for p,contents in before.items():self.assertEqual(p.read_bytes(),contents)

    def test_stage_rechecks_shared_gpu_capacity_before_starting_process(self):
        from training.train_buyer_binding import stage
        with patch('training.local_resources.training_gpu_plan',side_effect=RuntimeError('busy GPU')), \
             patch('training.train_buyer_binding.subprocess.Popen') as process:
            with self.assertRaisesRegex(RuntimeError,'busy GPU'):
                stage(Path(self.tmp.name),dict(devices=[dict(index='0'),dict(index='1')]),'train',{})
            process.assert_not_called()

    def test_installed_torchrun_parser_preserves_child_run_argument(self):
        from torch.distributed.run import parse_args
        from training.train_buyer_binding import worker_command
        folder=Path(self.tmp.name)/'run with spaces'
        for kind in ('preflight','train'):
            command=worker_command(folder,kind)
            args=parse_args(command[4:])
            self.assertTrue(args.standalone)
            self.assertEqual(args.nproc_per_node,'2')
            self.assertEqual(args.training_script,str(ROOT/'training/train_buyer_binding.py'))
            self.assertEqual(args.training_script_args,['--worker',kind,'--run',str(folder)])
        for kind,labels in (('eval_parent',['parent']),('eval_candidates',['24','48'])):
            command=worker_command(folder,kind,labels)
            self.assertNotIn('torch.distributed.run',command)
            self.assertEqual(command[-2:],['--labels',','.join(labels)])


if __name__=='__main__':unittest.main()
