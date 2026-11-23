"""Small /tmp fixtures only; no hardware, real recordings or network access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from lerobot_robot_fairino.demo_writer import DemoWriter
from fr5_lerobot.tests.test_demo_writer import synthetic_sample, synthetic_command


class FaultyJournal:
    def __init__(self, stream, failure):
        self.stream, self.failure = stream, failure

    def write(self, text):
        if self.failure == 'write':
            raise OSError('synthetic journal disk full')
        if self.failure == 'short':
            return self.stream.write(text[:10])
        return self.stream.write(text)

    def flush(self):
        if self.failure == 'flush':
            raise OSError('synthetic journal flush failure')
        self.stream.flush()

    def close(self):
        self.stream.close()


class JournalOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp', prefix='fr5-journal-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'journal'
        self.external = Path(self.tmp.name)/'continuous'

    def writer(self, **kwargs):
        writer = DemoWriter(self.root, fps=6, task='SYNTHETIC ONLY',
                            metadata={'image_width':16, 'image_height':12},
                            journal_only=True, external_capture_root=self.external, **kwargs)
        def close():
            if writer._started and not writer._finished:
                writer.finish('interrupted')
        self.addCleanup(close)
        return writer

    def rows(self):
        return [json.loads(line) for line in (self.root/'meta/capture.jsonl').read_text().splitlines()]

    def test_only_journal_files_are_created_and_external_media_is_untouched(self):
        samples = [synthetic_sample(i) for i in range(2)]
        self.external.mkdir()
        external_rows = []
        for i, sample in enumerate(samples):
            Image.fromarray(sample['camera']['color_rgb']).save(self.external/f'{i}.png')
            np.savez_compressed(self.external/f'{i}.npz', depth_mm=sample['camera']['depth_mm'])
            external_rows.append({'frame_index':40+i, 'camera':sample['camera']['metadata'],
                                  'rgb_file':f'{i}.png', 'depth_file':f'{i}.npz'})
        (self.external/'capture.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in external_rows))
        before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.external.iterdir()}
        commands = [synthetic_command(s, i, 'arm', True) for i, s in enumerate(samples)]
        with patch('lerobot_robot_fairino.demo_writer.LeRobotDataset.create',
                   side_effect=AssertionError('must not create a LeRobot dataset')), \
             patch('lerobot_robot_fairino.demo_writer.np.savez_compressed',
                   side_effect=AssertionError('must not duplicate depth')):
            writer = self.writer().start()
            for sample, command in zip(samples, commands):
                writer.add(sample, command)
            report = writer.finish('pending_review')
        self.assertIsNone(writer.dataset)
        self.assertEqual(sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file()),
                         ['meta/capture.jsonl', 'meta/demo.json'])
        self.assertEqual(before, {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.external.iterdir()})
        for i, row in enumerate(self.rows()):
            self.assertEqual(row['frame_index'], i)
            self.assertEqual(row['camera'], samples[i]['camera']['metadata'])
            self.assertEqual(row['robot'], samples[i]['robot_state'])
            self.assertEqual(row['command'], commands[i])
            self.assertNotIn('depth_file', row)
            self.assertNotIn('rgb_file', row)
            reference = row['external_capture']
            self.assertEqual(reference['capture_log'], str(self.external/'capture.jsonl'))
            matched = [r for r in external_rows
                       if r['camera']['source_timestamp_unix'] == reference['match']['camera_source_timestamp']
                       and r['camera']['source_camera_serial'] == reference['match']['source_camera_serial']]
            self.assertEqual(matched[0]['frame_index'], 40+i)
        self.assertEqual(report['storage_mode'], 'journal_only')
        self.assertEqual(report['recorded_frames'], 2)
        self.assertTrue(report['storage_complete'])
        self.assertTrue(report['data_valid'])
        self.assertFalse(report['materialized_lerobot_dataset'])
        self.assertFalse(report['external_capture_verified'])
        self.assertFalse(report['training_ready'])
        self.assertFalse(report['grasp_success_verified'])
        self.assertIsNone(report['repo_id'])
        self.assertEqual(report['local_rgb_frames_written'], 0)
        self.assertEqual(report['local_depth_frames_written'], 0)
        self.assertEqual(report['outcome_source'], 'awaiting_user_review')
        self.assertEqual(report, json.loads((self.root/'meta/demo.json').read_text()))

    def test_reference_may_precede_external_capture_creation_without_claiming_completeness(self):
        writer = self.writer().start()
        for i in range(2):
            sample = synthetic_sample(i)
            writer.add(sample, synthetic_command(sample, i))
        report = writer.finish('success')
        self.assertFalse(self.external.exists())
        self.assertTrue(report['storage_complete'])
        self.assertIn('journal rows only', report['storage_complete_scope'])
        self.assertFalse(report['external_capture_verified'])
        self.assertFalse(report['materialized_lerobot_dataset'])
        self.assertFalse(report['training_ready'])

    def test_actual_times_and_command_snapshots_survive_wall_clock_step(self):
        writer = self.writer().start()
        expected = []
        for i, wall in enumerate((2000., 1999.6, 2100.)):
            sample = synthetic_sample(i)
            mono = 10.+i/6
            sample['camera']['metadata'].update(source_timestamp_unix=wall, source_timestamp_monotonic=mono)
            sample['robot_state'].update(timestamp=wall+.001, received_monotonic=mono+.001)
            sample['sync_skew_s'] = .001
            sample['sync_clock'] = 'host_monotonic'
            command = synthetic_command(sample, i, 'hold' if i==0 else 'arm', i!=0)
            command.update(issued_unix=2000., issued_monotonic=mono-.01, source='agent_teacher',
                           teacher={'policy':'synthetic', 'decision_id':f'decision-{i}', 'phase':'lift'})
            expected.append((deepcopy(sample['robot_state']), deepcopy(command)))
            writer.add(sample, command)
            sample['robot_state']['joints'][0] = 999.
            command['target'][0] = 999.
        report = writer.finish('pending_review')
        rows = self.rows()
        self.assertEqual([r['camera']['source_timestamp_unix'] for r in rows], [2000.,1999.6,2100.])
        for row, (state, command) in zip(rows, expected):
            self.assertEqual(row['robot'], state)
            self.assertEqual(row['command'], command)
            self.assertEqual(row['external_capture']['match']['timing_clock'], 'host_monotonic')
            self.assertEqual(row['external_capture']['match']['camera_source_timestamp'], row['camera']['source_timestamp_monotonic'])
        self.assertEqual(report['dataset_kind'], 'agent_teacher_action_journal')
        self.assertEqual(report['frames_with_command_executed_false'], 1)
        self.assertEqual(report['frames_with_invalid_gripper_feedback'], 3)
        self.assertAlmostEqual(report['measured_camera_sample_fps'], 6.)
        self.assertAlmostEqual(report['camera_interval_s']['median'], 1/6)
        self.assertAlmostEqual(report['action_age_s']['max'], .01)
        self.assertAlmostEqual(report['sync_abs_skew_ms_max'], 1.)

    def test_prepare_validation_still_rejects_invalid_commands_images_and_depth(self):
        writer = self.writer().start()
        sample = synthetic_sample()
        for key, value in (('target', [0]*6), ('target', [0]*6+[101]),
                           ('target', [float('nan')]+[0]*6), ('sequence', -1),
                           ('executed', 1), ('source', 'invented')):
            command = synthetic_command(sample); command[key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                writer.add(sample, command)
        for key, value in (('color_rgb', np.zeros((1,1,3),np.uint8)),
                           ('color_rgb', np.zeros((12,16,3),np.float32)),
                           ('depth_mm', np.ones((1,1),np.float32)),
                           ('depth_mm', np.full((12,16),-1.,np.float32)),
                           ('depth_mm', np.full((12,16),np.nan,np.float32))):
            bad = deepcopy(sample);bad['camera'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                writer.add(bad, synthetic_command(sample))
        teacher = synthetic_command(sample); teacher['source'] = 'agent_teacher'
        with self.assertRaisesRegex(ValueError,'provenance'):
            writer.add(sample, teacher)
        bad = deepcopy(sample);bad['sync_skew_s'] += .2
        with self.assertRaisesRegex(ValueError,'skew'):
            writer.add(bad, synthetic_command(sample))
        self.assertEqual(self.rows(), [])
        report = writer.finish('discarded')
        self.assertFalse(report['data_valid'])
        self.assertFalse(report['storage_complete'])
        self.assertEqual(report['recorded_frames'], 0)

    def test_future_duplicate_and_changed_camera_are_rejected(self):
        writer = self.writer().start()
        sample = synthetic_sample()
        future = synthetic_command(sample)
        future['issued_unix'] = sample['camera']['metadata']['source_timestamp_unix']+.001
        with self.assertRaisesRegex(ValueError,'Future action'):
            writer.add(sample, future)
        writer.add(sample, synthetic_command(sample))
        with self.assertRaisesRegex(ValueError,'strictly increasing'):
            writer.add(sample, synthetic_command(sample))
        next_sample = synthetic_sample(1)
        next_sample['camera']['metadata']['source_camera_serial'] = 'OTHER_SYNTHETIC_CAMERA'
        with self.assertRaisesRegex(ValueError,'identity changed'):
            writer.add(next_sample, synthetic_command(next_sample))
        report = writer.finish('interrupted')
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(report['recorded_frames'], 1)
        self.assertFalse(report['data_valid'])

    def test_failed_or_short_write_does_not_count_or_allow_further_append(self):
        for failure in ('write','short','flush'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir='/tmp') as tmp:
                writer = DemoWriter(Path(tmp)/'journal', journal_only=True,
                                    external_capture_root=Path(tmp)/'continuous',
                                    metadata={'image_width':16,'image_height':12}).start()
                first = synthetic_sample()
                writer.add(first, synthetic_command(first))
                original = writer._log
                writer._log = FaultyJournal(original, failure)
                sample = synthetic_sample(1)
                with self.assertRaises(OSError):
                    writer.add(sample, synthetic_command(sample))
                with self.assertRaisesRegex(RuntimeError,'Journal storage failed'):
                    writer.add(sample, synthetic_command(sample))
                report = writer.finish('interrupted')
                self.assertTrue(original.closed)
                self.assertEqual(report['recorded_frames'], 1)
                self.assertFalse(report['storage_complete'])
                self.assertFalse(report['data_valid'])
                self.assertTrue(report['errors'])

    def test_mode_and_lifecycle_require_explicit_new_paths(self):
        for kwargs in ({'journal_only':1}, {'journal_only':True},
                       {'journal_only':True,'external_capture_root':''},
                       {'journal_only':True,'external_capture_root':self.root},
                       {'external_capture_root':self.external}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DemoWriter(self.root, **kwargs)
        self.assertFalse(self.root.exists())
        writer = self.writer().start()
        with self.assertRaises(RuntimeError):writer.start()
        with self.assertRaises(FileExistsError):self.writer()
        report = writer.finish('discarded')
        self.assertEqual(report['recorded_frames'], 0)
        self.assertFalse(report['data_valid'])
        self.assertFalse(report['storage_complete'])
        self.assertEqual(writer.finish('discarded'), report)
        with self.assertRaises(ValueError):writer.finish('success')
        with self.assertRaises(RuntimeError):writer.add(synthetic_sample(), synthetic_command(synthetic_sample()))


if __name__ == '__main__':
    unittest.main()
