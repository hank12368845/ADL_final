"""CPU-only regression tests for training GPU selection; no NVML or CUDA calls."""
import unittest
from unittest.mock import Mock
from training.local_resources import training_gpu_plan


def cards(free0=24, free1=24):
    return [{'index':str(i),'uuid':f'test_{i}','total_gib':24.,'free_gib':float(free)}
            for i,free in enumerate([free0,free1])]


class TrainingGPUPlanTests(unittest.TestCase):
    def test_two_free_cards_start_two_workers_without_waiting(self):
        query, pause = Mock(return_value=cards()), Mock()
        devices, skipped, plan = training_gpu_plan(query=query, pause=pause)
        self.assertEqual({d['index'] for d in devices}, {'0','1'})
        self.assertFalse(plan['fallback_to_single']); pause.assert_not_called()

    def test_recheck_recovers_a_card_released_by_previous_job(self):
        query, pause = Mock(side_effect=[cards(6,24), cards()]), Mock()
        devices, skipped, plan = training_gpu_plan(query=query, pause=pause)
        self.assertEqual(len(devices),2)
        self.assertEqual(len(plan['snapshots']),2)
        self.assertEqual(plan['snapshots'][0]['skipped'][0]['index'],'0')
        self.assertFalse(plan['fallback_to_single']); pause.assert_called_once_with(5)

    def test_explicit_two_cards_never_silently_become_one(self):
        with self.assertRaisesRegex(RuntimeError,'Requested 2 usable GPUs'):
            training_gpu_plan('0,1',query=Mock(return_value=cards(6,24)),pause=Mock())

    def test_auto_fallback_preserves_evidence(self):
        devices, skipped, plan = training_gpu_plan(query=Mock(return_value=cards(6,24)),pause=Mock())
        self.assertEqual([d['index'] for d in devices],['1'])
        self.assertTrue(plan['fallback_to_single'])
        self.assertTrue(skipped and plan['snapshots'][1]['skipped'])

    def test_explicit_single_card_is_respected(self):
        pause = Mock()
        devices,_,plan = training_gpu_plan('0',query=Mock(return_value=cards()),pause=pause)
        self.assertEqual([d['index'] for d in devices],['0'])
        self.assertFalse(plan['fallback_to_single']); pause.assert_not_called()


if __name__ == '__main__':
    unittest.main()
