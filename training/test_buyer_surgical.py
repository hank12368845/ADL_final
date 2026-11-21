"""CPU audits for faithful replay, original persona cooldown and channel-safe repairs."""
from collections import Counter
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from training.prepare_buyer_surgical import ROOT,MARKET,function,audit_answer,cooldown_pair,last_attempt


class SurgicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training.prepare_buyer_surgical import generate
        cls.tmp=tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_surgical_test_')
        cls.folder=Path(cls.tmp.name)/'data'
        with redirect_stdout(io.StringIO()):cls.manifest=generate(cls.folder)
        cls.rows=[json.loads(l) for l in (cls.folder/'train.jsonl').read_text().splitlines()]
        cls.tokens=[json.loads(l) for l in (cls.folder/'train.tokens.jsonl').read_text().splitlines()]
        cls.batches=json.loads((cls.folder/'batch_plan.json').read_text())
        cls.evaluation=json.loads((cls.folder/'evaluation_cases.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_replay_is_actual_profitable_step84_reasoning_and_input_unchanged(self):
        original={(r['agent'],r['tick']):r for r in map(json.loads,(MARKET/'events.jsonl').read_text().splitlines()) if r.get('type')=='turn'}
        replay=[r for r in self.rows if r['family']=='faithful_step84_replay']
        self.assertEqual(len(replay),43)
        for c in replay:
            r=original[f'example_student_buyer_p{c["persona_index"]}',c['source_tick']]
            self.assertEqual(c['messages'],r['llm']['messages'])
            self.assertEqual(c['answer']['reasoning_content'],r['llm']['reasoning'])
            self.assertEqual(c['answer']['content'],r['llm']['text'] or '')
            self.assertEqual(function(c['answer'])[1],json.loads(r['llm']['raw_arguments']))
            self.assertGreater(c['realized_surplus_cents'],0)

    def test_exact_70_30_exposures_twenty_steps_and_truthful_repetition(self):
        self.assertEqual(len(self.batches),20)
        counts=Counter()
        for batch in self.batches:
            self.assertEqual(len(batch),4)
            counts.update(batch)
        self.assertEqual(sum(counts.values()),80)
        self.assertEqual(sum(v for i,v in counts.items() if i<43),56)
        self.assertEqual({i:counts[i] for i in range(43,67)},{i:1 for i in range(43,67)})
        self.assertEqual(set(counts),set(range(67)))
        for i in range(0,20,5):
            self.assertEqual(sum(j<43 for b in self.batches[i:i+5] for j in b),14)
        recipe=json.loads((self.folder/'recipe.json').read_text())
        self.assertEqual((recipe['max_steps'],recipe['learning_rate'],recipe['effective_batch_size']),(20,5e-6,4))

    def test_all_targets_pass_cash_current_source_and_cooldown_audit(self):
        for c in self.rows:audit_answer(c)
        active=[c for c in self.rows if c['family']=='cooldown_active']
        expired=[c for c in self.rows if c['family']=='cooldown_expired']
        self.assertEqual((len(active),len(expired)),(1,1))
        for c in active:
            blocked=cooldown_pair(c);self.assertIsNotNone(blocked)
            tool,args=function(c['answer']);self.assertNotEqual(blocked,(args.get('seller_id'),args.get('product_id')))
            wrong=copy.deepcopy(c);bad=wrong['answer']['tool_calls'][0]['function']['arguments']
            bad.update(seller_id=blocked[0],product_id=blocked[1])
            with self.assertRaisesRegex(ValueError,'cooldown'):audit_answer(wrong)
        for c in expired:
            self.assertIsNone(cooldown_pair(c));self.assertEqual(last_attempt(c)[0],'wait')
            history=[m for m in c['messages'][:-1] if m['role']=='assistant']
            failed=history[-2]['tool_calls'][0]['function']['arguments']
            _,args=function(c['answer'])
            self.assertEqual((failed['seller_id'],failed['product_id']),(args['seller_id'],args['product_id']))
        waiting=next(c for c in self.rows if c['family']=='cooldown_wait')
        self.assertEqual(function(waiting['answer'])[0],'wait')
        self.assertIsNotNone(cooldown_pair(waiting))

    def test_offers_need_no_board_and_first_loss_does_not_end_search(self):
        for c in self.rows:
            if c['family']=='offer_without_board':
                tool,args=function(c['answer']);self.assertEqual(tool,'accept_offer')
                offer=next(o for o in c['observation']['offers_to_you'] if o['offer_id']==args['offer_id'])
                self.assertFalse(any((b['seller'],b['product_id'])==(offer['seller'],offer['product_id']) for b in c['observation']['board']))
            elif c['family']=='first_loss_later_gain':
                tool,args=function(c['answer']);first=c['observation']['board'][0]
                self.assertGreaterEqual(first['price'],c['observation']['you']['utility_of_next_unit'].get(first['product_id'],0))
                self.assertEqual(tool,'buy_listed')
                self.assertNotEqual((first['seller'],first['product_id']),(args['seller_id'],args['product_id']))

    def test_no_source_leakage_and_real_market_probes_are_explicitly_seen(self):
        from training.buyer_binding_eval import validate_wire
        self.assertFalse(set(self.manifest['train_groups'])&set(self.manifest['validation_groups']))
        self.assertEqual(self.manifest['evaluation_suites']['training_market_seen'],6)
        self.assertEqual(self.manifest['evaluation_suites']['surgical_validation'],12)
        self.assertNotIn('market_regression',self.manifest['evaluation_suites'])
        val={c['group'] for c in self.evaluation if c['suite']=='surgical_validation'}
        self.assertFalse(val&{c['group'] for c in self.rows})
        self.assertEqual(Counter(c['evaluation_rank'] for c in self.evaluation),{0:51,1:51})
        self.assertEqual(validate_wire(self.evaluation),102)
        for c in self.rows:
            text=(ROOT/f'agents/example_student/buyer/persona_{c["persona_index"]}.md').read_text().strip()
            self.assertEqual(c['messages'][0]['content'].split('<<<PERSONA_SECTION>>>')[1].split('<<<END_PERSONA_SECTION>>>')[0].strip(),text)

    def test_saved_hash_audit_rejects_mutation_without_touching_historical_data(self):
        from training.prepare_buyer_surgical import audit_folder
        self.assertEqual(audit_folder(self.folder)['unique_replay'],43)
        p=self.folder/'batch_plan.json';contents=p.read_bytes()
        try:
            p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'files changed'):audit_folder(self.folder)
        finally:p.write_bytes(contents)

    def test_native_token_spans_no_input_truncation_and_no_reasoning_rewrite(self):
        from transformers import AutoTokenizer
        from training.prepare_buyer_surgical import encode
        from training.buyer_hf_check import parse_raw
        t=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',revision=self.manifest['model_revision'],local_files_only=True)
        for c,row in zip(self.rows,self.tokens):
            self.assertEqual(row,encode(t,c))
            completion=t.decode(row['input_ids'][row['prompt_length']:])
            self.assertIn(c['answer']['reasoning_content'],completion)
            self.assertIn('</think>',completion);self.assertIn('<|im_end|>',completion)
            parsed=parse_raw(t,completion,c['tools'])
            self.assertEqual(parsed['parsed_calls'],1)
            self.assertEqual(parsed['tool'],function(c['answer'])[0])


if __name__=='__main__':unittest.main()
