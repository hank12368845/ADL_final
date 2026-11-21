"""CPU evidence for source-only board repair, warm start and matched evaluation."""
from collections import Counter
from contextlib import redirect_stdout,nullcontext
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data/buyer_board_repair_v1'


class BoardRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.cfg=yaml.safe_load((ROOT/'training/configs/buyer_board_repair_v1.yaml').read_text())
        with (DATA/'train.jsonl').open() as stream:cls.rows=[json.loads(s) for s in stream]

    def test_original_facts_and_offers_preserved_and_no_validation_or_parent_seen_rows(self):
        from training.board_repair_data import board_target
        from training.decision_ablation import digest
        source=ROOT/'training/data/buyer_reasoning_v1'
        with (source/'train.jsonl').open() as f:old={r['id']:r for r in map(json.loads,f)}
        parent=ROOT/self.cfg['initial_adapter'];seen=set()
        for rank in (0,1):
            seen.update(json.loads(s)['id'] for s in (parent.parent.parent/f'samples_rank{rank}.jsonl').read_text().splitlines())
        self.assertEqual(len(self.rows),256)
        self.assertTrue(not seen & {r['id'] for r in self.rows})
        for row in self.rows:
            before=old[row['id']]
            self.assertEqual(row['messages'],before['messages']);self.assertEqual(row['observation'],before['observation'])
            self.assertEqual(row['answer']['tool_calls'],before['answer']['tool_calls'])
            if row['family'].startswith('offer'):
                self.assertEqual(row['answer'],before['answer'])
            else:
                text=board_target(row);self.assertEqual(row['answer']['reasoning_content'],text)
                available=[b for b in row['observation']['board'] if b['in_stock']>=1]
                if available:
                    b=available[0];first=text.splitlines()[0]
                    self.assertIn('seller_id='+b['seller'],first);self.assertIn('product_id='+b['product_id'],first)
                    self.assertTrue(text.startswith('BOARD:'));self.assertIn('CHECK this product:',text)
                else:self.assertIn('no current listing',text)
        for split in ('validation','test'):
            for suffix in ('jsonl','tokens.jsonl'):
                self.assertEqual(digest(DATA/f'{split}.{suffix}'),digest(source/f'{split}.{suffix}'))
        self.assertEqual(digest(DATA/'cash_probe_cases.json'),digest(source/'cash_probe_cases.json'))
        self.assertEqual(Counter(r['label_action'] for r in self.rows),{'buy_listed':128,'accept_offer':32,'wait':96})
        self.assertEqual(Counter(r['history_mode'] for r in self.rows),{'fresh':128,'recorded':128})

    def test_masks_and_full_coverage_two_rank_sampling(self):
        from training.buyer_operations_train import check_data,validation_indices
        from training.board_repair_sampling import paired_batches,PairedActionSampler
        train,val=check_data(self.cfg,ROOT)
        self.assertEqual((len(train),len(val)),(256,384));self.assertEqual(len(validation_indices(val,32,self.cfg['seed'])),32)
        batches=paired_batches(train.metadata,32,self.cfg['seed'])
        self.assertEqual(batches[:16],paired_batches(train.metadata,16,self.cfg['seed']))
        self.assertEqual(Counter(i for b in batches for i in b),Counter(range(256)))
        rank_rows=[list(PairedActionSampler(train.metadata,32,self.cfg['seed'],r,2)) for r in (0,1)]
        actual=[rank_rows[r][micro] for start in range(0,128,4) for micro in range(start,start+4) for r in (0,1)]
        self.assertEqual(actual,[i for b in batches for i in b])
        for i in range(len(train)):
            row=train[i];self.assertLess(max(row['reasoning_positions']),min(row['tool_name_positions']))
            self.assertTrue(row['thinking'])

    def test_warm_start_verified_and_is_not_optimizer_resume(self):
        from training.board_repair_setup import prepare_warm_start
        cfg=copy.deepcopy(self.cfg);warm=prepare_warm_start(cfg,ROOT)
        self.assertEqual(warm['parent_step'],32);self.assertEqual(warm['local_start_step'],0)
        self.assertEqual(warm['optimizer'],'new AdamW');self.assertNotIn('resume_checkpoint',cfg)
        self.assertEqual(cfg['learning_rate'],5e-6);self.assertEqual(cfg['warmup_steps'],4);self.assertEqual(cfg['eval_every'],16)
        for key,value in (('rank',32),('alpha',64),('thinking',False)):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'recipe mismatch'):
                prepare_warm_start({**self.cfg,key:value},ROOT)

    def test_user_entry_cpu_and_mock_launch_then_prepare_three_model_evaluation(self):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from training.train_buyer_operations import main
        from training.evaluate_board_repair import main as evaluate
        import shutil
        config=ROOT/'training/configs/buyer_board_repair_v1.yaml'
        with patch('sys.argv',['train_buyer_operations.py','--config',str(config),'--cpu-check']), \
             patch('training.local_resources.training_gpu_plan') as gpu,patch('training.train_buyer_operations.subprocess.Popen') as proc,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();proc.assert_not_called()
        cards=[{'index':str(i),'fraction':.8} for i in (0,1)];calls=[]
        def launch(command,**kw):
            calls.append(command);return SimpleNamespace(stdout=io.BytesIO(b'CPU fake worker\n'),wait=lambda:0)
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_board_repair_entry_cpu_') as temp:
            with patch('sys.argv',['train_buyer_operations.py','--config',str(config),'--gpus','0,1','--output-dir',temp]), \
                 patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)), \
                 patch('training.train_buyer_operations.subprocess.Popen',side_effect=launch),redirect_stdout(io.StringIO()):main()
            run=next(Path(temp).glob('*_buyer_operations_*'));cfg=json.loads((run/'train_config.json').read_text())
            self.assertNotIn('resume_checkpoint',cfg);self.assertEqual(cfg['warm_start']['optimizer'],'new AdamW')
            self.assertIn('--nproc_per_node=2',calls[0]);self.assertTrue(cfg['thinking'])
            for step in (16,32):
                cp=run/'checkpoints'/f'step_{step:04d}';cp.mkdir(parents=True)
                for name in ('adapter_config.json','adapter_model.safetensors'):
                    shutil.copy2(ROOT/'agents/example_student/buyer/checkpoint'/name,cp/name)
                (cp/'training_step.json').write_text(json.dumps({'step':step,'thinking':True}))
            output=run/'cpu_reports';output.mkdir()
            with patch('sys.argv',['evaluate_board_repair.py',str(run),'--prepare-only']), \
                 patch('training.local_resources.output_parent',return_value=(output,None)), \
                 patch('training.local_resources.query_gpus') as query,patch('training.buyer_diagnostics.subprocess.Popen') as proc, \
                 redirect_stdout(io.StringIO()):folder=evaluate()
            query.assert_not_called();proc.assert_not_called()
            plan=json.loads((folder/'plan.json').read_text())
            self.assertEqual(plan['calls'],1152);self.assertTrue(plan['thinking'])
            self.assertEqual(set(plan['adapters']),{'parent_on','repair16','repair32'})
            self.assertFalse((folder/'evaluation_status.json').exists())

    def test_three_model_worker_and_review_can_choose_step16_when_step32_regresses(self):
        from training.buyer_diagnostics import worker
        from training.buyer_reasoning_data import personas
        from training.evaluate_board_repair import write_review
        cases=json.loads((DATA/'cash_probe_cases.json').read_text())
        class Client:
            def __init__(self,**kw):pass
            def complete(inner,**kw):
                self.assertTrue(kw['extra_body']['chat_template_kwargs']['enable_thinking'])
                obs=json.loads(kw['messages'][-1]['content']);state={'bad':[],'pending':None,'last_buy':None}
                offers=obs['offers_to_you'];row=(offers or obs['board'])[0]
                if offers:tool='accept_offer';args={'offer_id':row['offer_id']};price=row['price']*row['qty']
                else:
                    tool='buy_listed';args={'seller_id':row['seller'],'product_id':row['product_id'],'qty':1};price=row['price']
                    state['last_buy']=[row['seller'],row['product_id']]
                if ('parent_on' in kw['model'] and not offers) or ('repair32' not in kw['model'] and obs['you']['cash']<price):
                    tool='wait';args={};state['last_buy']=None
                args['private_state']=json.dumps(state)
                action=SimpleNamespace(name=tool,arguments=args,raw_arguments=json.dumps(args),arguments_error=None)
                return SimpleNamespace(tool_call=action,stop_reason='tool_calls',outcome='accepted',error=None,text='',
                    reasoning='Current row IDs and cash checked.',usage={'input_tokens':1000,'output_tokens':80,'reasoning_tokens':60,'total_tokens':1140})
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_board_repair_worker_cpu_') as temp:
            folder=Path(temp)
            plan={'devices':[{'index':str(i),'fraction':.8} for i in (0,1)],'port':8000,'seeds':[0,1],
                  'personas':{f'operations_reasoning_p{i+1}':p for i,p in enumerate(personas())},
                  'adapters':{k:'/unused/CPU_fixture' for k in ('parent_on','repair16','repair32')},
                  'thinking':True,'analysis':'cash_decision','calls':1152,
                  'experiment_manifest':{'experiment':'buyer_board_repair_v1'}}
            (folder/'plan.json').write_text(json.dumps(plan));(folder/'cases.json').write_text(json.dumps(cases))
            with patch('adlmarket.serving.VLLMCluster',return_value=nullcontext()),patch('adlmarket.serving.register_adapters'), \
                 patch('adlmarket.client.ModelClient',Client),redirect_stdout(io.StringIO()):
                worker(folder);review=write_review(folder)
            self.assertEqual(review['candidate_meeting_targets'],'repair16')
            self.assertFalse(review['models']['repair32']['checks']['board_cash_errors_remain_zero'])
            self.assertEqual(review['models']['repair16']['offer']['both_correct'],96)
            report=json.loads((folder/'cash_decision_pairs.json').read_text())
            self.assertEqual(report['gpu_calls'],{'0':576,'1':576})
            self.assertEqual(json.loads((folder/'evaluation_status.json').read_text())['completed_calls'],1152)


if __name__=='__main__':unittest.main()
