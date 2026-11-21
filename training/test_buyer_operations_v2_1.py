"""CPU evidence for cash equality, context balance and the named step8 comparator."""
from collections import Counter
from contextlib import nullcontext,redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.buyer_operations_v2_1_data import (cash_pair,contexts,make_records,VARIANTS,FAMILIES,
                                               legacy_probe,encode_row,choose)
from training.buyer_operations_v2_1_sampling import paired_batches,PairedActionSampler,validation_selection


class V21Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[r for g in range(8) for p in range(3) for f in FAMILIES
                  for r in make_records(20261119+1009*g,f,p)]
        fields=('id','pair_id','positive','group','family','persona_index','history_mode','label_action',
                'cash_variant','economic_id','curriculum_version')
        cls.meta=[{k:r[k] for k in fields} for r in cls.rows]

    def test_cash_boundaries_and_board_offer_financial_identity(self):
        from training.buyer_diagnostics import evaluate
        for variant in VARIANTS:
            channels={c:cash_pair(20265155,0,c,variant) for c in ('board','offer')}
            for side in (0,1):
                a,b=channels['board'][side],channels['offer'][side]
                self.assertEqual(a['observation']['you'],b['observation']['you'])
                price=a['observation']['board'][0]['price']
                self.assertEqual(price,b['observation']['offers_to_you'][0]['price'])
                cash=a['observation']['you']['cash']
                if side==0:
                    self.assertGreaterEqual(cash,price)
                    if variant=='equal':self.assertEqual(cash,price)
                    if variant=='above_one':self.assertAlmostEqual(cash-price,.01)
                else:
                    self.assertLess(cash,price)
                    if variant in ('equal','above_one'):self.assertAlmostEqual(price-cash,.01)
                    if variant=='zero':self.assertEqual(cash,0)
                for r in (a,b):
                    fn=r['answer']['tool_calls'][0]['function']
                    result=evaluate({**r,'name':r['id'],'state_mode':'operations'},fn['name'],fn['arguments'])
                    self.assertEqual(result['errors'],[])
                    self.assertTrue(r['engine_result']['ok'])
                    if side==1:self.assertEqual(r['negative_trade_check']['result']['error'],'insufficient_cash')
            transformed=contexts(channels['board'])
            for fresh,recorded in zip(transformed[:2],transformed[2:]):
                self.assertEqual(fresh['messages'][0],recorded['messages'][0])
                self.assertEqual(fresh['messages'][-1],recorded['messages'][-1])
                self.assertEqual(fresh['observation'],recorded['observation'])
                self.assertEqual(len(fresh['messages']),2)
                self.assertGreater(len(recorded['messages']),2)

    def test_sampler_balances_cash_contexts_personas_and_real_dataloader_ranks(self):
        from torch.utils.data import DataLoader
        batches=paired_batches(self.meta,16,7)
        self.assertEqual(batches,paired_batches(self.meta,16,7))
        all_actions=Counter();all_personas=Counter();negatives=Counter();cash_modes=Counter()
        for batch in batches:
            self.assertEqual(len(set(batch)),8)
            actions=Counter(self.meta[i]['label_action'] for i in batch)
            self.assertEqual(actions,{'buy_listed':4,'accept_offer':2,'wait':2});all_actions.update(actions)
            modes=Counter((self.meta[i]['label_action'],self.meta[i]['history_mode']) for i in batch)
            for action,n in actions.items():
                for mode in ('fresh','recorded'):self.assertEqual(modes[action,mode],n//2)
            all_personas.update(self.meta[i]['persona_index'] for i in batch)
            for i in batch:
                r=self.meta[i]
                if not r['positive']:
                    negatives[r['family']]+=1
                    if r['family'].endswith('cash'):cash_modes[r['cash_variant'],r['history_mode']]+=1
            cash=[i for i in batch if self.meta[i]['positive'] and self.meta[i]['family'].endswith('cash')]
            # Locate the paired positives, allowing additional cash positive replay.
            pair_ids={self.meta[i]['pair_id'] for i in batch if not self.meta[i]['positive']}
            cash=[i for i in cash if self.meta[i]['pair_id'] in pair_ids]
            if cash:
                self.assertEqual(len(cash),2)
                self.assertEqual(self.meta[cash[0]]['economic_id'],self.meta[cash[1]]['economic_id'])
                self.assertEqual(self.rows[cash[0]]['observation']['you']['cash'],self.rows[cash[1]]['observation']['you']['cash'])
        self.assertEqual(all_actions,{'buy_listed':64,'accept_offer':32,'wait':32})
        self.assertEqual(all_personas,{1:43,2:43,3:42})
        self.assertEqual(sum(v for k,v in negatives.items() if k.endswith('cash')),16)
        self.assertEqual(set(cash_modes.values()),{2})
        ranks=[list(DataLoader(self.meta,batch_size=None,sampler=PairedActionSampler(self.meta,16,7,r,2))) for r in (0,1)]
        for step,batch in enumerate(batches):
            actual=[ranks[r][step*4+m]['id'] for m in range(4) for r in (0,1)]
            self.assertEqual(actual,[self.meta[i]['id'] for i in batch])
        selected=validation_selection(self.meta,16,7)
        self.assertEqual(sum(self.meta[i]['positive'] for i in selected),8)
        self.assertEqual(Counter(self.meta[i]['history_mode'] for i in selected),{'fresh':8,'recorded':8})

    def test_token_masks_legacy_probe_and_unchanged_loss_recipe(self):
        from transformers import AutoTokenizer
        from training.buyer_operations_data import BASE,REVISION
        from training.buyer_operations_train import TokenRows,validation_indices
        import yaml
        root=Path(__file__).resolve().parents[1]
        a=yaml.safe_load((root/'training/configs/buyer_operations_v2.yaml').read_text())
        b=yaml.safe_load((root/'training/configs/buyer_operations_v2_1.yaml').read_text())
        for key in ('decision_weight','rank','alpha','learning_rate','max_steps','effective_batch_size','warmup_steps','seed'):
            self.assertEqual(a[key],b[key])
        self.assertNotIn('initial_adapter',b)
        self.assertEqual(len(legacy_probe()),16)
        tok=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_v21_tokens_cpu_') as temporary:
            path=Path(temporary)/'rows.jsonl'
            # One market, all personas/families and both contexts, for validation integration.
            encoded=[encode_row(tok,r) for r in self.rows[:96]]
            path.write_text(''.join(json.dumps(r)+'\n' for r in encoded))
            data=TokenRows(path,12288,True)
            indices=validation_indices(data,16,7)
            self.assertEqual(len(indices),16)
            for r in encoded:
                name=tok.decode([r['input_ids'][r['prompt_length']+i] for i in r['tool_name_positions']])
                self.assertIn(r['label_action'],name)

    def test_full_cpu_worker_names_step8_reference_and_does_not_treat_it_as_plain_base(self):
        from adlmarket.setups import market_run
        from training.buyer_diagnostics import worker
        from training.buyer_operations_data import personas
        from training.buyer_operations_v2_data import diagnostic_case
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        root=Path(__file__).resolve().parents[1]
        run=market_run(root/'configs/basic_buyer.yaml',root=root)
        adapter=run.models[run.buyers[0].model].lora.path
        records=contexts(cash_pair(20265155,0,'board','equal'))[:2]
        cases=[diagnostic_case(r) for r in records]
        ref=records[0]['answer']['tool_calls'][0]['function']
        class FakeClient:
            def __init__(self,**kwargs):pass
            def complete(self,**kwargs):
                ChatCompletionRequest.model_validate({'model':kwargs['model'],'messages':kwargs['messages'],'tools':kwargs['tools']})
                obs=json.loads(kwargs['messages'][-1]['content'])
                tool,args=(ref['name'],copy.deepcopy(ref['arguments'])) if 'step8' in kwargs['model'] else choose(records[0]['situation'],obs)
                action=SimpleNamespace(name=tool,arguments=args,raw_arguments=json.dumps(args),arguments_error=None)
                return SimpleNamespace(tool_call=action,stop_reason='tool_calls',outcome='accepted',error=None,text='')
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_v21_worker_cpu_') as temporary:
            folder=Path(temporary)
            plan={'devices':[{'index':str(i),'fraction':.8} for i in (0,1)],'port':8000,'seeds':[0,1],
                  'personas':{'operations_p1':personas()[0]},'adapters':{'step8':adapter,'candidate':adapter},'calls':8}
            (folder/'plan.json').write_text(json.dumps(plan));(folder/'cases.json').write_text(json.dumps(cases))
            with patch('adlmarket.serving.VLLMCluster',return_value=nullcontext()),patch('adlmarket.serving.register_adapters'), \
                 patch('adlmarket.client.ModelClient',FakeClient),redirect_stdout(io.StringIO()):worker(folder)
            result=json.loads((folder/'operation_gate.json').read_text())
            self.assertEqual(result['reference_model'],'step8')
            scores=result['personas']['operations_p1']
            self.assertNotIn('base',scores)
            self.assertEqual(scores['step8']['captured_opportunities'],2)
            self.assertEqual(scores['candidate']['captured_opportunities'],2)
            self.assertEqual(scores['candidate']['board_captured'],2)
            self.assertEqual(scores['candidate']['correct_negative_waits'],2)
            self.assertEqual(scores['step8']['correct_negative_waits'],0)
            self.assertEqual(json.loads((folder/'evaluation_status.json').read_text())['status'],'complete')

    def test_changed_reference_fails_before_any_model_evaluation(self):
        from training.evaluate_buyer_operations_v2_1 import main
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_v21_reference_cpu_') as temporary:
            folder=Path(temporary);reference=folder/'reference';reference.mkdir()
            (reference/'adapter_model.safetensors').write_bytes(b'changed reference')
            cfg={'data_version':'buyer_operations_v2_1','comparison_adapter':str(reference),
                 'comparison_fingerprints':{'adapter_model.safetensors':'different hash'}}
            (folder/'train_config.json').write_text(json.dumps(cfg))
            with patch('sys.argv',['evaluate_buyer_operations_v2_1.py',str(folder),'--prepare-only']), \
                 patch('training.buyer_diagnostics.main') as evaluate:
                with self.assertRaisesRegex(ValueError,'Reference step8 changed'):main()
                evaluate.assert_not_called()


if __name__=='__main__':unittest.main()
