"""CPU-only TP2 config checks; no claims about CUDA fit, speed or numerical parity."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from training.compare_seller_personas_tp2 import tp_plan,configure_tp2,main
from training.compare_seller_personas import prepare_pair,ROOT,initial_state
from adlmarket.setups import market_run


def row(i,free):return dict(index=str(i),uuid=f'cpu{i}',total_gib=23.98828125,free_gib=free)


class TP2Tests(unittest.TestCase):
    def test_shared_fraction_keeps_reserve_on_both_ranks(self):
        p=tp_plan([row(0,10.77),row(1,11.13)])
        self.assertEqual(len(p['devices']),2)
        for i,budget in p['per_rank_budget_gib'].items():
            self.assertGreaterEqual(budget,8.)
            self.assertLessEqual(budget,{'0':10.77,'1':11.13}[i]-2.)
        with self.assertRaises(RuntimeError):tp_plan([row(0,9.),row(1,11.13)])
        with self.assertRaisesRegex(ValueError,'two distinct'):tp_plan([row(0,11.)],'0')

    def test_real_official_tp_command_same_market_and_sampling(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            folder=Path(tmp);prepare_pair(folder,4);arm=folder/'v2';m=json.loads((arm/'suite.json').read_text())
            config=configure_tp2(arm,m,[row(0,10.77),row(1,11.13)],'0,1',8100)
            loaded=market_run(config,root=ROOT);sv=loaded.serving['base']
            self.assertEqual(sv.gpus,['0,1']);self.assertEqual(sv.max_model_len,64000)
            self.assertEqual(sv.max_lora_rank,64);self.assertEqual(sv.max_loras,4)
            self.assertEqual(initial_state(loaded),m['initial_signature'])
            r=json.loads((arm/'resource_plan.json').read_text());cmd=r['official_serve_command']
            self.assertEqual(cmd[cmd.index('--tensor-parallel-size')+1],'2')
            self.assertEqual(cmd[cmd.index('--max-num-seqs')+1],'8')
            self.assertEqual(cmd.count('--max-num-seqs'),1)
            self.assertEqual(r['gpu_environment']['CUDA_VISIBLE_DEVICES'],'0,1')
            self.assertEqual(set(r['routes'].values()),{'http://localhost:8100/v1'})

    def test_prepare_only_does_not_query_or_launch_gpu(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['tp','--prepare-only','--output-dir',tmp]),patch('training.local_resources.query_gpus') as query,patch('training.compare_seller_personas_tp2.play') as play,redirect_stdout(io.StringIO()):
                main();query.assert_not_called();play.assert_not_called()


if __name__=='__main__':unittest.main()
