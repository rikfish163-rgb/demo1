"""Offline checks for shared pi0.5 runtime and GPU selection."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(os.environ.get('FR5_PI05_TRAINER_PATH',
                          Path(__file__).resolve().parents[1] / 'training/train_pi05_fr5.py'))
spec = importlib.util.spec_from_file_location('fr5_pi05_training', PATH)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class TrainingContractTests(unittest.TestCase):


    def test_gpu_selection_and_preference(self):
        output = '3, GPU-3, 24000, 0\n8, GPU-8, 23000, 0\n9, GPU-9, 24500, 0\n'
        with patch.object(training.subprocess, 'run') as run:
            run.side_effect = lambda args, **kw: type('Result', (), {'stdout': output if '--query-gpu=index,uuid,memory.free,utilization.gpu' in args else ''})()
            self.assertEqual(training.gpu_status()['index'], 9)
            self.assertEqual(training.gpu_status(8)['index'], 8)
            with self.assertRaises(RuntimeError):
                training.gpu_status(1)

            other_pid = os.getpid() + 1
            run.side_effect = lambda args, **kw: type('Result', (), {'stdout': output if '--query-gpu=index,uuid,memory.free,utilization.gpu' in args else f'GPU-9, {other_pid}\n'})()
            self.assertEqual(training.gpu_status()['index'], 8)
            with self.assertRaises(RuntimeError):
                training.gpu_status(9)

    def test_gpu_recheck_allows_its_own_context_but_not_a_second_process(self):
        result = lambda output: type('Result', (), {'stdout': output})()
        gpu = result('9, GPU-9, 22840, 40\n')
        own = f'GPU-9, {os.getpid()}\n'
        with patch.object(training.subprocess, 'run', side_effect=[gpu, result(own)]):
            self.assertEqual(training.gpu_status(9)['index'], 9)
        with patch.object(training.subprocess, 'run', side_effect=[gpu, result(own + f'GPU-9, {os.getpid()+1}\n')]):
            with self.assertRaises(RuntimeError):
                training.gpu_status(9)


class RuntimeContractTests(unittest.TestCase):
    def test_exporter_runtime_is_not_a_supported_trainer_runtime(self):
        versions = {**training.TESTED_RUNTIME, 'lerobot': '0.4.4'}
        with patch.object(training.importlib.metadata, 'version', side_effect=versions.__getitem__), \
                patch.object(training.sys, 'prefix', str(training.ROOT / '.venv')):
            with self.assertRaisesRegex(RuntimeError, 'lerobot=0.4.4'):
                training.require_runtime_versions()

    def test_exact_versions_and_data4_environment_are_required(self):
        with patch.object(training.importlib.metadata, 'version', side_effect=training.TESTED_RUNTIME.__getitem__), \
                patch.object(training.sys, 'prefix', str(training.ROOT / '.venv')):
            self.assertEqual(training.require_runtime_versions(), training.TESTED_RUNTIME)
        with patch.object(training.importlib.metadata, 'version', side_effect=training.TESTED_RUNTIME.__getitem__), \
                patch.object(training.sys, 'prefix', '/tmp/other-env'):
            with self.assertRaisesRegex(RuntimeError, 'outside the tested'):
                training.require_runtime_versions()


if __name__ == '__main__':
    unittest.main()
