"""CPU evidence for historical/current seller counterfactuals and notice-ID traps."""
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

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data/buyer_id_repair_v1'


class IdRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls.cfg=yaml.safe_load((ROOT/'training/configs/buyer_id_repair_v1.yaml').read_text())
        with (DATA/'train.jsonl').open() as f:cls.rows=[json.loads(s) for s in f]

    def test_engine_backing_and_id_sources_and_heldout_boundaries(self):
        from training.decision_ablation import digest
        source=ROOT/'training/data/buyer_reasoning_v1';manifest=json.loads((DATA/'manifest.json').read_text())
        allowed={manifest['seed']+1009*g for g in manifest['groups']['train']}
        board=[r for r in self.rows if r['family'].startswith('board')]
        self.assertEqual(len(board),192)
        self.assertEqual(Counter(r['current_seller_is_historical'] for r in board),{True:96,False:96})
        self.assertEqual(Counter(r['label_action'] for r in self.rows),{'buy_listed':128,'accept_offer':32,'wait':96})
        for r in self.rows:
            self.assertIn(r['group'],allowed)
            self.assertTrue(r['engine_result']['ok'])
            if r['positive']:self.assertGreater(r['realized_surplus_cents'],0)
        for r in board:
            self.assertFalse(r['wrong_seller_engine_check']['ok'])
            self.assertFalse(r['notice_as_offer_engine_check']['ok'])
            self.assertEqual(r['observation']['offers_to_you'],[])
            self.assertTrue(any(m['channel']=='notice' for m in r['observation']['new_messages']))
            if r['positive']:
                b=r['observation']['board'][0];args=r['answer']['tool_calls'][0]['function']['arguments']
                self.assertEqual((args['seller_id'],args['product_id']),(b['seller'],b['product_id']))
                self.assertIn('offers_to_you_count=0',r['answer']['reasoning_content'])
            else:
                trial=r['negative_trade_check'];self.assertTrue(not trial['result']['ok'] or trial['realized_surplus_cents']<0)
        for split in ('validation','test'):
            for suffix in ('jsonl','tokens.jsonl'):
                self.assertEqual(digest(DATA/f'{split}.{suffix}'),digest(source/f'{split}.{suffix}'))
        self.assertEqual(digest(DATA/'cash_probe_cases.json'),digest(source/'cash_probe_cases.json'))

    def test_seller_counterfactual_only_changes_current_tuple_not_history(self):
        grouped=defaultdict(list)
        for r in self.rows:
            if r['family'].startswith('board'):
                grouped[r['group'],r['persona_index'],r['history_mode'],r['cash_variant']].append(r)
        self.assertEqual(len(grouped),96)
        for pair in grouped.values():
            self.assertEqual(len(pair),2);a,b=pair
            self.assertEqual(a['messages'][:-1],b['messages'][:-1])
            self.assertNotEqual(a['current_seller'],b['current_seller'])
            oa,ob=copy.deepcopy(a['observation']),copy.deepcopy(b['observation'])
            for obs in (oa,ob):
                for row in obs['board']:row['seller']='CURRENT_SELLER'
            self.assertEqual(oa,ob)
            self.assertEqual(a['positive'],b['positive'])

    def test_offer_replay_and_original_native_template_masks(self):
        from training.buyer_operations_train import check_data
        from training.board_repair_sampling import paired_batches,PairedActionSampler
        with (ROOT/'training/data/buyer_reasoning_v1/train.jsonl').open() as f:original={r['id']:r for r in map(json.loads,f)}
        for r in self.rows:
            if r['family'].startswith('offer'):
                self.assertEqual(r['messages'],original[r['id']]['messages'])
                self.assertEqual(r['answer'],original[r['id']]['answer'])
        train,val=check_data(self.cfg,ROOT);self.assertEqual((len(train),len(val)),(256,384))
        batches=paired_batches(train.metadata,32,self.cfg['seed'])
        self.assertEqual(Counter(i for b in batches for i in b),Counter(range(256)))
        ranks=[list(PairedActionSampler(train.metadata,32,self.cfg['seed'],r,2)) for r in (0,1)]
        actual=[ranks[r][m] for s in range(0,128,4) for m in range(s,s+4) for r in (0,1)]
        self.assertEqual(actual,[i for b in batches for i in b])
        for i in range(256):
            row=train[i];self.assertLess(max(row['reasoning_positions']),min(row['tool_name_positions']))

    def test_warm_start_and_prepare_three_models_without_gpu(self):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from training.train_buyer_operations import main
        from training.evaluate_board_repair import main as evaluate
        cfg_path=ROOT/'training/configs/buyer_id_repair_v1.yaml'
        with patch('sys.argv',['train_buyer_operations.py','--config',str(cfg_path),'--cpu-check']), \
             patch('training.local_resources.training_gpu_plan') as gpu, \
             patch('training.train_buyer_operations.subprocess.Popen') as launch,redirect_stdout(io.StringIO()):
            main();gpu.assert_not_called();launch.assert_not_called()
        def launch(command,**kw):return SimpleNamespace(stdout=io.BytesIO(b'CPU fixture\n'),wait=lambda:0)
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='adl_id_repair_cpu_') as temp:
            cards=[{'index':str(i),'fraction':.8} for i in (0,1)]
            with patch('sys.argv',['train_buyer_operations.py','--config',str(cfg_path),'--gpus','0,1','--output-dir',temp]), \
                 patch('training.local_resources.training_gpu_plan',return_value=(cards,[],{})), \
                 patch('training.local_resources.runtime_cache',return_value=({},None)), \
                 patch('training.train_buyer_operations.subprocess.Popen',side_effect=launch),redirect_stdout(io.StringIO()):main()
            run=next(Path(temp).glob('*_buyer_operations_*'));cfg=json.loads((run/'train_config.json').read_text())
            self.assertIn('134251',cfg['initial_adapter']);self.assertNotIn('resume_checkpoint',cfg)
            for step in (16,32):
                cp=run/'checkpoints'/f'step_{step:04d}';cp.mkdir(parents=True)
                for name in ('adapter_config.json','adapter_model.safetensors'):
                    shutil.copy2(ROOT/'agents/example_student/buyer/checkpoint'/name,cp/name)
                (cp/'training_step.json').write_text(json.dumps({'step':step,'thinking':True}))
            output=run/'reports';output.mkdir()
            with patch('sys.argv',['evaluate_board_repair.py',str(run),'--prepare-only']), \
                 patch('training.local_resources.output_parent',return_value=(output,None)), \
                 patch('training.local_resources.query_gpus') as gpu,patch('training.buyer_diagnostics.subprocess.Popen') as proc, \
                 redirect_stdout(io.StringIO()):folder=evaluate()
            gpu.assert_not_called();proc.assert_not_called()
            plan=json.loads((folder/'plan.json').read_text());self.assertEqual(plan['calls'],1152)
            self.assertEqual(plan['experiment_manifest']['experiment'],'buyer_id_repair_v1')
            self.assertTrue(plan['thinking'])


if __name__=='__main__':unittest.main()
