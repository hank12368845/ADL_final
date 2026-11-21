"""CPU-only Seller baseline checks; no real model/GPU launch."""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from training.official_seller_data import ROOT, observation, snapshot_review, target_error


class OfficialSellerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from training.train_official_seller import prepare
        cls.cfg,cls.batches,cls.cases,cls.manifest=prepare(yaml.safe_load((ROOT/'training/configs/seller_official_v1.yaml').read_text()))
        cls.rows={s:[json.loads(l) for l in (ROOT/cls.cfg['data_dir']/f'{s}.jsonl').read_text().splitlines()] for s in ('train','validation','test')}

    def test_complete_subset_epoch_and_group_isolation(self):
        cfg=self.cfg;self.assertLessEqual(cfg['train_examples'],1024)
        counts=Counter(i for batch in self.batches for i in batch)
        self.assertEqual(set(counts),set(range(cfg['train_examples'])))
        self.assertLess(len(self.batches)*8-cfg['train_examples'],8)
        self.assertLess(cfg['warmup_steps'],cfg['max_steps'])
        groups={s:{r['group'] for r in rows} for s,rows in self.rows.items()}
        tids={s:{r['trajectory_id'] for r in rows} for s,rows in self.rows.items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(groups[a]&groups[b]);self.assertFalse(tids[a]&tids[b])
        self.assertGreater(self.manifest['splits']['train']['market_groups'],10)

    def test_targets_are_exact_original_official_teacher_completions(self):
        from training.official_buyer_data import read_official,normalize
        wanted=defaultdict(list)
        for rows in self.rows.values():
            for row in rows:wanted[row['trajectory_id']].append(row)
        matched=0
        for episode in read_official(ROOT/'training/data/official_raw'):
            for row in wanted.get(episode['id'],[]):
                answer=normalize(episode['messages'][row['source_message_index']])
                self.assertEqual(row['answer'],answer)
                self.assertIsNone(target_error(row['situation'],row['observation'],answer,row['tools'],row['original_tool_result']))
                self.assertTrue(answer['reasoning_content'])
                self.assertEqual(observation(row['messages'][-1]['content']),row['observation'])
                self.assertTrue(all('reasoning_content' not in m for m in row['messages'][1:]))
                self.assertLessEqual(sum(m['role']=='user' for m in row['messages']),3)
                matched+=1
        self.assertEqual(matched,sum(len(rows) for rows in self.rows.values()))

    def test_current_prompt_has_latest_persona_and_preserves_private_numbers(self):
        from training.official_buyer_data import situation,read_official
        source={}
        for episode in read_official(ROOT/'training/data/official_raw'):
            if episode['role']=='seller':source[episode['id']]=situation(episode['messages'][0]['content'])
        for rows in self.rows.values():
            for row in rows:
                sit=row['situation'];original=source[row['trajectory_id']]
                for key in ('your_stock','your_unit_cost','your_low_grade_unit_cost','your_id'):
                    self.assertEqual(sit.get(key),original.get(key))
                self.assertEqual({p['product_id']:p['list_price'] for p in sit['products']},{p['product_id']:p['list_price'] for p in original['products']})
                self.assertEqual((sit['rounds'],sit['turns_per_round']),(5,6))
                text=(ROOT/f'training/persona_versions/seller_checkpoint1_v2/persona_{row["persona_index"]}.md').read_text().strip()
                self.assertIn(text,row['messages'][0]['content'])
                self.assertNotIn('may close\n  at the end of any round',row['messages'][0]['content'])

    def test_token_supervision_uses_native_complete_answer_and_chat_end(self):
        from transformers import AutoTokenizer
        from training.official_buyer_data import BASE,MODEL_REVISION
        tokenizer=AutoTokenizer.from_pretrained(BASE,revision=MODEL_REVISION,local_files_only=True)
        path=ROOT/self.cfg['data_dir']/'train.tokens.jsonl'
        tokens=[json.loads(l) for l in path.read_text().splitlines()]
        for row,encoded in zip(self.rows['train'][:8],tokens[:8]):
            p=encoded['prompt_length'];self.assertLessEqual(len(encoded['input_ids']),8192)
            target=tokenizer.decode(encoded['input_ids'][p:],skip_special_tokens=False)
            self.assertIn(row['answer']['reasoning_content'],target)
            self.assertIn('<tool_call>',target);self.assertIn('<|im_end|>',target)
            self.assertEqual(encoded['target_length'],len(encoded['input_ids'])-p)

    def test_current_rule_probes_expose_stock_and_offer_limit_errors(self):
        current=[c for c in self.cases if c['kind']=='current_rule']
        self.assertEqual(len(current),12)
        for c in current:
            sit,obs,tools=c['situation'],c['observation'],c['tools']
            if c['family']=='empty_stock':
                self.assertTrue(snapshot_review(sit,obs,'wait',{},tools)['valid'])
                pid=next(iter(sit['your_unit_cost']))
                self.assertIn('no_stock_or_excess_qty',snapshot_review(sit,obs,'set_price',dict(product_id=pid,price=100),tools)['errors'])
            else:
                pid=next(p for p,n in obs['you']['stock'].items() if n)
                args=dict(product_id=pid,price=sit['your_unit_cost'][pid]+1)
                self.assertTrue(snapshot_review(sit,obs,'set_price',args,tools)['profitable_quote'])
                bad={**args,'price':sit['your_unit_cost'][pid]}
                self.assertIn('nonpositive_margin',snapshot_review(sit,obs,'set_price',bad,tools)['errors'])
                if c['family']=='offer_limit':
                    self.assertEqual(len(obs['your_open_offers']),8)
                    self.assertIn('too_many_offers',snapshot_review(sit,obs,'make_offer',{**args,'qty':1,'buyer_id':sit['buyers'][0]},tools)['errors'])

    def test_seen_cases_are_covered_by_first_checkpoint_without_holdout_leak(self):
        rows=self.rows['train'];ids={rows[i]['id'] for b in self.batches[:self.cfg['checkpoint_steps'][0]] for i in b}
        for c in self.cases:
            if c['kind']=='seen':self.assertIn(c['name'].split(':',1)[1],ids)
        self.assertTrue(any(c['kind']=='p1_v1_control' for c in self.cases))

    def test_regression_gate_rejects_wait_collapse(self):
        from training.official_seller_train import catastrophic,summary
        base={'current_rule':dict(captured=7,valid=10)}
        self.assertTrue(catastrophic(base,{'current_rule':dict(captured=0,valid=12)}))
        self.assertFalse(catastrophic(base,{'current_rule':dict(captured=8,valid=11)}))
        rows=[dict(kind='current_rule',valid=True,profitable_quote=False,quote_opportunity=True,truncated=False,tool='wait',errors=[])]
        self.assertEqual(summary(rows)['current_rule']['captured'],0)

    def test_cpu_entry_does_not_query_or_launch_gpu(self):
        from training.train_official_seller import main
        with patch('sys.argv',['seller','--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu,patch('training.train_official_seller.launch') as launch,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();launch.assert_not_called()

    def test_no_stock_observation_suffix_and_price_precision(self):
        self.assertEqual(observation('{"round":1,"you":{}}\n\nannotation')['round'],1)
        c=next(c for c in self.cases if c.get('family')=='stocked');sit,obs=c['situation'],c['observation']
        pid=next(p for p,n in obs['you']['stock'].items() if n)
        out=snapshot_review(sit,obs,'set_price',dict(product_id=pid,price=100.001),c['tools'])
        self.assertIn('price_precision',out['errors'])


if __name__=='__main__':unittest.main()
