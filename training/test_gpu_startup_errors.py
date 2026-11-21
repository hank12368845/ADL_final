import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from training.gpu_startup_errors import startup_memory_races
from training.queue_seller_personas import launch

ERROR='ValueError: Free memory on device cuda:0 (13.05/23.52 GiB) on startup is less than desired GPU memory utilization (0.896, 21.07 GiB).'


def fixture(root,actions=0,error=ERROR):
    arm=root/'v2';run=arm/'run_market_test';log=run/'vllm/vllm_base_0.log';log.parent.mkdir(parents=True)
    log.write_text(error+'\n')
    wrapper=arm/'wrapper_result.json';wrapper.write_text(json.dumps(dict(exit=2,actions=actions,run_dir=str(run))))
    return wrapper


class StartupErrorTests(unittest.TestCase):
    def test_real_failed_v2_is_identified_without_changing_files(self):
        root=Path(__file__).resolve().parents[1]/'results/basic_checks/20261121-005859_seller_personas_1f4wxek7'
        found=startup_memory_races(root,0)
        self.assertEqual(len(found),2);self.assertTrue(all('/v2/' in r['wrapper'] for r in found))

    def test_does_not_retry_stale_errors_or_real_execution_oom(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);wrapper=fixture(root)
            self.assertEqual(len(startup_memory_races(root,0)),1)
            os.utime(wrapper,(1,1));self.assertFalse(startup_memory_races(root,2))
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);fixture(root,actions=15);self.assertFalse(startup_memory_races(root,0))
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);fixture(root,error='torch.OutOfMemoryError: CUDA out of memory')
            self.assertFalse(startup_memory_races(root,0))

    def test_queue_requeues_generic_wrapper_failure_only_with_fresh_root_evidence(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);fixture(root);proc=Mock(stdout=io.StringIO('RuntimeError: v2 incomplete\n'))
            proc.wait.return_value=1
            with patch('training.queue_seller_personas.subprocess.Popen',return_value=proc),patch('training.queue_seller_personas.time.time',return_value=0):
                code,requeue=launch(root,['0','1'],8100,lambda text:None)
            self.assertEqual((code,requeue),(1,True));self.assertTrue((root/'queue_resource_failures.jsonl').exists())


if __name__=='__main__':unittest.main()
