"""CPU provenance, actual engine controls, native targets, batching and launcher tests."""
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data/buyer_latest_brakes_v1'


class LatestBrakesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from training.train_buyer_latest_brakes import prepare
        cls.cfg,cls.manifest,cls.batches=prepare(yaml.safe_load((ROOT/'training/configs/buyer_latest_brakes_v1.yaml').read_text()))
        cls.train=[json.loads(l) for l in (DATA/'train.jsonl').read_text().splitlines()]
        cls.val=[json.loads(l) for l in (DATA/'validation.jsonl').read_text().splitlines()]
        cls.cases=json.loads((DATA/'evaluation_cases.json').read_text())

    def test_latest_inputs_original_current_facts_and_group_isolation(self):
        for row in self.train+self.val:
            text=row['messages'][0]['content'].split('<<<PERSONA_SECTION>>>')[1].split('<<<END_PERSONA_SECTION>>>')[0].strip()
            self.assertEqual(text,(ROOT/f'agents/example_student/buyer/persona_{row["persona_index"]}.md').read_text().strip())
            self.assertLessEqual(len(text),8000)
            body=json.loads(row['messages'][-1]['content'])
            for key in ('board','offers_to_you','you'):self.assertEqual(body[key],row['observation'][key])
            self.assertNotIn('positive',body);self.assertNotIn('engine_result',body)
        self.assertFalse(set(self.manifest['groups']['train'])&set(self.manifest['groups']['validation']))
        official_train={json.loads(l)['id'] for l in (ROOT/'training/data/buyer_official_v1/train.jsonl').read_text().splitlines()}
        replay=[r for r in self.train if r['source']=='official_rehearsal']
        self.assertEqual(len(replay),144)
        for r in replay:self.assertIn(r['id'].rsplit('_latest_p',1)[0],official_train)
        old_names={c['name'] for c in self.cases if c['suite'] in ('known','fresh','official')}
        self.assertFalse({r['id'] for r in self.train}&old_names)

    def test_every_batch_has_positive_rehearsal_and_complete_cash_or_owned_pairs(self):
        self.assertEqual(len(self.train),288);self.assertEqual(len(self.batches),36)
        self.assertEqual(Counter(i for b in self.batches for i in b),Counter(range(288)))
        for batch in self.batches:
            rows=[self.train[i] for i in batch]
            self.assertEqual(Counter(r['label_action'] for r in rows),{'buy_listed':3,'accept_offer':3,'wait':2})
            pairs=defaultdict(list)
            for r in rows:
                if r['source']!='official_rehearsal':pairs[r['pair_id']].append(r)
            self.assertEqual(len(pairs),2)
            for pair in pairs.values():self.assertEqual({r['positive'] for r in pair},{True,False})
        from training.official_buyer_data import cents
        first=[self.train[i] for b in self.batches[:12] for i in b]
        for pindex in (1,2):
            cash=[r for r in first if r.get('mode')=='cash' and r['persona_index']==pindex and not r['positive']]
            self.assertTrue(any(r['observation']['you']['cash']==0 for r in cash))
            self.assertTrue(any(cents(next(o['price'] for o in [*r['observation']['board'],*r['observation']['offers_to_you']] if o['product_id']==r['target_product']))-cents(r['observation']['you']['cash'])==1 for r in cash))

    def test_engine_negative_proofs_and_p1_p2_cash_boundaries(self):
        from training.official_buyer_train import official_snapshot_review
        from training.official_buyer_data import cents
        for pindex in (1,2):
            cash_rows=[r for r in self.train if r.get('mode')=='cash' and r['persona_index']==pindex and not r['positive']]
            for channel in ('board','offer'):
                subset=[r for r in cash_rows if r['channel']==channel]
                self.assertTrue(any(r['observation']['you']['cash']==0 for r in subset))
                self.assertTrue(any(cents(next(o for o in [*r['observation']['board'],*r['observation']['offers_to_you']] if o['product_id']==r['target_product'])['price'])-cents(r['observation']['you']['cash'])==1 for r in subset))
        for r in self.train+self.val:
            if r['source']=='official_rehearsal':continue
            fn=r['answer']['tool_calls'][0]['function']
            self.assertTrue(official_snapshot_review(r,fn['name'],fn['arguments'])['valid_snapshot'])
            if r['positive']:self.assertGreater(r['positive_surplus_cents'],0)
            elif r['mode']=='cash':self.assertEqual(r['negative_trade_check']['result']['error'],'insufficient_cash')
            else:
                bad=r['negative_trade_check']
                self.assertTrue(not bad['result']['ok'] or bad['surplus_cents']<0)

    def test_p3_owned_first_and_single_quality_calculation(self):
        rows=[r for r in self.train if r.get('mode')=='owned' and r['persona_index']==3]
        self.assertTrue(rows)
        for r in rows:
            reason=r['answer']['reasoning_content'];pid=r['target_product']
            if not r['positive']:
                self.assertGreater(r['situation']['your_utility_per_unit'][pid][0],0)
                self.assertEqual(r['observation']['you']['units_bought'][pid],1)
                self.assertEqual(r['observation']['you']['utility_of_next_unit'].get(pid,0),0)
                self.assertEqual(r['label_action'],'wait');self.assertNotIn('quality adjustment',reason.lower())
            self.assertEqual(reason.count('Cash>=total'),1)
        quality=[r for r in self.train if r.get('mode') and r['persona_index']==3 and r['positive'] and
                 next(o['quality_varies'] for o in r['situation']['market_overview'] if o['product_id']==r['target_product'])]
        self.assertTrue(quality)
        for r in quality:self.assertEqual(r['answer']['reasoning_content'].count('One adjustment'),1)

    def test_native_tokens_mask_and_one_tool_on_entire_dataset(self):
        from transformers import AutoTokenizer
        from training.buyer_hf_check import parse_raw
        from training.official_buyer_data import schema_error
        tokenizer=AutoTokenizer.from_pretrained(self.cfg['base_model'],revision=self.cfg['revision'],local_files_only=True)
        for split,rows in (('train',self.train),('validation',self.val)):
            tokens=[json.loads(l) for l in (DATA/f'{split}.tokens.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),len(tokens))
            for row,token in zip(rows,tokens):
                prompt=tokenizer.apply_chat_template(row['messages'],tools=row['tools'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                self.assertEqual(tokenizer.encode(prompt,add_special_tokens=False),token['input_ids'][:token['prompt_length']])
                self.assertLessEqual(len(token['input_ids']),12288);self.assertLessEqual(token['reasoning_tokens'],160)
                target=tokenizer.decode(token['input_ids'][token['prompt_length']:],skip_special_tokens=False)
                self.assertEqual(target.count('<tool_call>'),1);self.assertLess(target.index('</think>'),target.index('<function='))
                parsed=parse_raw(tokenizer,target,row['tools']);self.assertEqual(parsed['parsed_calls'],1)
                fn=row['answer']['tool_calls'][0]['function'];self.assertEqual(parsed['tool'],fn['name']);self.assertEqual(parsed['arguments'],fn['arguments'])
                self.assertIsNone(schema_error(fn['name'],fn['arguments'],row['tools']))
                self.assertEqual(next(iter(fn['arguments'])),'private_state')
                if row['persona_index']==2 and fn['name']=='accept_offer':
                    from training.buyer_latest_brakes_data import reference_trade
                    from training.official_buyer_data import cents
                    trade=reference_trade(row);v=row['observation']['you']['utility_of_next_unit'][trade['product_id']]
                    self.assertLessEqual(4*cents(trade['price']),3*cents(v))

    def test_seen_prefix_and_parent_cache_routes_are_explicit(self):
        first_ids={self.train[i]['id'] for b in self.batches[:12] for i in b}
        seen=[c for c in self.cases if c['suite']=='seen_train'];self.assertEqual(len(seen),6)
        for c in seen:self.assertIn(c['id'],first_ids)
        self.assertEqual(sum('cached_parent' in c for c in self.cases),37)
        self.assertEqual(len(self.cases),67)
        self.assertEqual(Counter(c['evaluation_rank'] for c in self.cases),{0:33,1:34})
        pair_ranks=defaultdict(set)
        for c in self.cases:
            if c.get('pair_id'):pair_ranks[(c['suite'],c['pair_id'])].add(c['evaluation_rank'])
        self.assertTrue(all(len(s)==1 for s in pair_ranks.values()))

    def test_cash_safety_cannot_hide_recall_regression(self):
        from training.buyer_latest_brakes_train import gates
        parent={k:{'cash_errors':0,'owned_errors':0,'no_tool':0,'truncated':0,'valid':8} for k in ('current','new_validation','official','seen')}
        parent['current'].update(captured=12,known_captured=8,fresh_captured=4)
        parent['new_validation']['captured']=10
        good=copy.deepcopy(parent);self.assertTrue(all(gates(parent,good).values()))
        swapped=copy.deepcopy(good);swapped['current'].update(known_captured=7,fresh_captured=5)
        self.assertFalse(all(gates(parent,swapped).values()))
        waiting=copy.deepcopy(good);waiting['current'].update(captured=0,known_captured=0,fresh_captured=0)
        self.assertFalse(all(gates(parent,waiting).values()))
        expensive=copy.deepcopy(good);expensive['official']['cash_errors']=1
        self.assertFalse(all(gates(parent,expensive).values()))

    def test_cpu_launcher_and_exact_resume_guards_without_gpu(self):
        from training.train_buyer_latest_brakes import main,resume
        from training.decision_ablation import fingerprint
        with patch('sys.argv',['brakes','--cpu-check']),patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_latest_brakes.subprocess.Popen') as process,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();process.assert_not_called()
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_latest_brakes_cpu_') as tmp:
            def fake(cmd,**kwargs):
                folder=Path(cmd[-1]);self.assertIn('--nproc_per_node=2',cmd)
                (folder/'finished.json').write_text(json.dumps({'status':'CPU_mock','development_candidate':None}))
                return SimpleNamespace(stdout=io.BytesIO(b'CPU fake worker; no model execution\n'),wait=lambda:0)
            cards=[{'index':str(i),'fraction':.85,'budget_gib':20} for i in (0,1)]
            with patch('sys.argv',['brakes','--output-dir',tmp]),patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)),patch('training.train_buyer_latest_brakes.subprocess.Popen',side_effect=fake),redirect_stdout(io.StringIO()):
                folder=main()
            cp=folder/'checkpoints/step_0012';cp.mkdir(parents=True)
            for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(self.manifest['parent_adapter']+'/'+name,cp/name)
            for name in ('optimizer.pt','rng_rank0.pt','rng_rank1.pt'):(cp/name).write_bytes(b'CPU existence fixture only')
            (cp/'checkpoint_complete.json').write_text(json.dumps({'step':12,'baseline_summary':{},'adapter_fingerprints':fingerprint(cp)}))
            (folder/'checkpoint_scores.json').write_text('[]')
            got=resume(copy.deepcopy(self.cfg),cp);self.assertEqual(got['start_step'],12)
            with self.assertRaisesRegex(ValueError,'Exact resume differs'):resume({**self.cfg,'learning_rate':1e-4},cp)


if __name__=='__main__':unittest.main()
