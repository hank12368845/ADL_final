"""CPU matched-market and kernel-install-plan checks. No real model/GPU work."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class BuyerMarketPairTests(unittest.TestCase):
    def test_matched_initial_state_and_only_target_adapter_differs(self):
        from training.compare_buyer_markets import prepare_pair,CANDIDATE,PARENT
        from adlmarket.setups import market_run
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            plan=prepare_pair(tmp)
            arms={label:json.loads((Path(folder)/'suite.json').read_text()) for label,folder in plan['arms'].items()}
            self.assertEqual(arms['step84']['initial_signature'],arms['step10']['initial_signature'])
            self.assertEqual(arms['step84']['adapter'],str(PARENT));self.assertEqual(arms['step10']['adapter'],str(CANDIDATE))
            self.assertEqual(plan['seed'],1);self.assertEqual(plan['total_actions'],900)
            for label,m in arms.items():
                loaded=market_run(Path(m['config']),root=ROOT)
                self.assertEqual((len(loaded.sellers),len(loaded.buyers)),(5,10))
                self.assertEqual((loaded.config.market.rounds.total,loaded.config.market.rounds.turns_per_round),(5,6))
                self.assertEqual(loaded.config.market.rounds.output_tokens_per_round,8192)
                for i in (1,2,3):
                    self.assertEqual((Path(plan['arms'][label])/f'buyer/persona_{i}.md').read_bytes(),(ROOT/f'agents/example_student/buyer/persona_{i}.md').read_bytes())

    def test_prepare_only_does_not_query_or_launch_gpu_and_seed0_is_rejected(self):
        from training.compare_buyer_markets import main,prepare_pair
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['pair','--prepare-only','--output-dir',tmp]),patch('training.local_resources.query_gpus') as gpu, \
                patch('training.compare_buyer_markets.play_arm') as play,redirect_stdout(io.StringIO()):
                main();gpu.assert_not_called();play.assert_not_called()
            with self.assertRaisesRegex(ValueError,'training'):prepare_pair(Path(tmp)/'seed0',0)

    def test_two_full_mock_markets_and_resume_are_never_real_evidence(self):
        from training.compare_buyer_markets import main
        protected=[ROOT/f'agents/example_student/buyer/persona_{i}.md' for i in (1,2,3)]+[ROOT/'agents/example_student/buyer/checkpoint/adapter_model.safetensors']
        before={p:p.read_bytes() for p in protected}
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            with patch('sys.argv',['pair','--mock','--output-dir',tmp]),patch('training.local_resources.query_gpus') as gpu,redirect_stdout(io.StringIO()):
                folder=main();gpu.assert_not_called()
            result=json.loads((folder/'market_comparison.json').read_text())
            self.assertTrue(result['complete']);self.assertFalse(result['valid_real_comparison'])
            self.assertFalse(result['automatic_promotion']);self.assertFalse(result['automatic_training'])
            self.assertEqual(len(result['by_persona']),3)
            for label in ('step84','step10'):
                info=json.loads((folder/label/'wrapper_result.json').read_text())
                self.assertEqual(info['actions'],450);self.assertEqual(sum(info['target_actions'].values()),90)
                progress=(folder/label/'progress.log').read_text();self.assertIn('overall',progress)
            with patch('sys.argv',['pair','--resume',str(folder),'--mock']),patch('training.compare_buyer_markets.play_arm') as play,redirect_stdout(io.StringIO()):
                main();play.assert_not_called()
            with patch('sys.argv',['pair','--resume',str(folder)]),self.assertRaisesRegex(ValueError,'mix mock'):
                main()
        for p,contents in before.items():self.assertEqual(p.read_bytes(),contents)

    def test_kernel_installer_plan_is_isolated_no_network_install_or_gpu(self):
        from training.install_fast_kernels import main,wheel_url
        with patch('sys.argv',['installer']),patch('training.install_fast_kernels.fetch') as fetch, \
            patch('training.install_fast_kernels.run') as run,redirect_stdout(io.StringIO()):
            plan=main();fetch.assert_not_called();run.assert_not_called()
        self.assertTrue(plan['base_environment_unchanged']);self.assertFalse(plan['gpu_work'])
        self.assertIn('cu13torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl',plan['causal_wheel'])
        self.assertNotIn('torch',','.join(plan['packages']))
        self.assertEqual(plan['signature']['cuda'],'13.0')


class KernelInstallTests(unittest.TestCase):
    def exercise(self,missing_wheel):
        import urllib.error
        from types import SimpleNamespace
        from training.install_fast_kernels import main
        info=dict(versions={},torch_version='2.11.0+cu130',cuda='13.0',abi=True,python='cp311',machine='x86_64')
        calls=[]
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            def fetch(url,path):
                if missing_wheel and url.endswith('.whl'):raise urllib.error.HTTPError(url,404,'not found',{},None)
                Path(path).write_bytes(b'fake download,not an install');return 'fake-checksum'
            def run(command,env,logfile,label):
                calls.append((command,env,label));self.assertIn('--no-deps',command)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
                self.assertTrue(Path(command[command.index('--target')+1]).is_relative_to(tmp))
                if 'toolchain' in label:
                    cuda=Path(tmp)/'cuda13/nvidia/cu13'
                    for filename in ['bin/nvcc','include/cuda_runtime.h','include/cccl/.test','lib/libcudart.so.13']:
                        f=cuda/filename;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fake')
            commit=SimpleNamespace(read=lambda:b'{"sha":"1111111111111111111111111111111111111111"}')
            with patch('sys.argv',['kernels','--install','--prefix',tmp]),patch('training.install_fast_kernels.signature',return_value=info), \
                patch('training.install_fast_kernels.fetch',side_effect=fetch),patch('training.install_fast_kernels.run',side_effect=run), \
                patch('training.install_fast_kernels.urllib.request.urlopen',return_value=commit), \
                patch('training.install_fast_kernels.subprocess.check_output',return_value='release 13.0, V13.0.88'),redirect_stdout(io.StringIO()):
                result=main()
            self.assertEqual(result['status'],'installed_not_gpu_verified')
            self.assertTrue((Path(tmp)/'env.sh').exists())
            if missing_wheel:
                self.assertEqual(result['causal_method'],'source_build')
                self.assertEqual(result['causal_commit'],'1'*40)
                self.assertEqual(calls[-1][1]['CAUSAL_CONV1D_FORCE_BUILD'],'TRUE')
                self.assertTrue((Path(tmp)/'cuda13/nvidia/cu13/lib/libcudart.so').is_symlink())
            else:self.assertEqual(result['causal_method'],'upstream_exact_wheel')
        return calls

    def test_exact_wheel_does_not_build_or_upgrade_core_environment(self):
        calls=self.exercise(False);self.assertEqual(len(calls),2)

    def test_missing_wheel_builds_with_isolated_matching_cuda13(self):
        calls=self.exercise(True);self.assertEqual(len(calls),3)


if __name__=='__main__':unittest.main()
