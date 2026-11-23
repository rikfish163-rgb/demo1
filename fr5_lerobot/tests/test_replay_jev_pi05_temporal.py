"""Focused offline checks for dense recorded-observation replay."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from fr5_lerobot.diagnostics import replay_jev_pi05_temporal as replay


class FakeKinematics:
    def point(self, joints):
        return np.asarray(joints[:3], dtype=float)


class FakeSource:
    normal = np.asarray([0., 0., 1.])

    def __init__(self):
        self.by_index = {
            i: {'frame_index': i, 'camera': {'source_timestamp_monotonic': t},
                'robot': {'received_monotonic': t}}
            for i, t in ((14, 99.), (15, 99.5), (16, 100.))}

    def feedback_at(self, timestamp, cycle, *, preceding=False):
        return {'available': True, 'gripper_feedback_valid': True,
                'raw_gripper_position_percent': 78, 'frame_index': 16,
                'measured_grasp_vs_release_reference': {'height_along_table_normal_mm': 8.}}

    def last_gripper_command(self, timestamp):
        return {'available': True, 'target_percent': 76., 'sequence': 6}


def forecast(frame=16):
    record = {'source_index': 6, 'source_capture_frame': frame, 'source_grid_time_monotonic': 100.,
              'row_index': 42, 'draw': 0, 'seed': 20261493,
              'absolute_actions': [[0., 0., float(i), 0., 0., 0., .76] for i in range(50)],
              'valid_mask': [True] * 50}
    record['absolute_actions'][3][6] = .92
    record['absolute_actions'][7][6] = 1.02
    return record


class TemporalReplayTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket.connect', 'socket.socket.connect_ex'):
            guard = patch(name, side_effect=AssertionError('Offline test cannot connect to a device'))
            guard.start()
            self.addCleanup(guard.stop)
        self.source = FakeSource()
        self.cycle = {'release_command_time': 102., 'hold_start': 1., 'hold_end': 2.,
                      'lift_probe_command_time': 3., 'lift_command_time': 4.,
                      'reference': {'status': 'stable_jaw_contact_proxy', 'point_base_mm': [0., 0., 0.]}}

    def test_all_actual_rows_in_range_are_selected_without_event_filtering(self):
        rows = [{'_physical_source': str(source), '_source_capture_frame': frame, '_split': split}
                for source, frame, split in ((6, 373, 'train'), (6, 374, 'train'), (6, 376, 'train'),
                                             (6, 383, 'train'), (7, 383, 'validation'),
                                             (6, 400, 'train'), (6, 401, 'train'))]
        cases = replay.select_cases(rows, {6: [0, 1, 2, 3, 5, 6], 7: [4]})
        self.assertEqual([case['row_index'] for case in cases], [1, 2, 3, 5])
        self.assertEqual([case['position'] for case in cases], [1, 2, 3, 4])
        self.assertEqual([case['case_index'] for case in cases], [0, 1, 2, 3])
        self.assertEqual(replay.SEEDS, (20261493, 20261494, 20261495))

    def test_source_and_frame_selection_uses_requested_recorded_rows(self):
        rows = [{'_physical_source': str(source), '_source_capture_frame': frame, '_split': split}
                for source, frame, split in ((7, 98, 'validation'), (6, 99, 'train'),
                                             (7, 99, 'validation'), (7, 101, 'validation'),
                                             (7, 102, 'validation'))]
        cases = replay.select_cases(rows, {7: [0, 2, 3, 4], 6: [1]},
                                    source_index=7, first_frame=99, last_frame=101)
        self.assertEqual([case['row_index'] for case in cases], [2, 3])
        self.assertEqual([case['position'] for case in cases], [1, 2])
        self.assertEqual([case['source_index'] for case in cases], [7, 7])
        self.assertEqual([case['split'] for case in cases], ['validation', 'validation'])

    def test_cli_passes_selected_source_and_frames_to_run(self):
        with patch.object(replay, 'run') as run:
            replay.main(['--dataset', 'dataset', '--checkpoint', 'checkpoint', '--output', 'output',
                         '--gpu', '2', '--source-index', '7', '--first-frame', '99', '--last-frame', '101'])
        run.assert_called_once_with(Path('dataset'), Path('checkpoint'), Path('output'), 2,
                                    source_index=7, first_frame=99, last_frame=101)

    def test_cli_defaults_and_saved_predictions_reject_unused_selection(self):
        with patch.object(replay, 'run') as run:
            replay.main(['--dataset', 'dataset', '--checkpoint', 'checkpoint', '--output', 'output', '--gpu', '2'])
        run.assert_called_once_with(Path('dataset'), Path('checkpoint'), Path('output'), 2,
                                    source_index=6, first_frame=374, last_frame=400)
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            replay.main(['--saved-predictions', 'saved', '--output', 'output', '--source-index', '7'])

    def test_selection_rejects_empty_or_reversed_ranges_and_missing_default_anchor(self):
        rows = [{'_physical_source': '6', '_source_capture_frame': 374, '_split': 'train'}]
        for source, first, last in ((7, 374, 400), (6, 400, 374), (6, 374, 400)):
            with self.subTest(source=source, first=first, last=last), self.assertRaises(ValueError):
                replay.select_cases(rows, {6: [0]}, source, first, last)

    def test_prefix_three_marks_only_sealed_execution_ticks(self):
        record = forecast()
        default = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())
        self.assertEqual(default, replay.analyze_draw(record, self.source, self.cycle, FakeKinematics(), 10))
        shorter = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics(), 3)
        self.assertEqual([tick['within_execution_prefix'] for tick in shorter['ticks']], [True] * 3 + [False] * 7)
        self.assertEqual(shorter['prefixes']['6']['90']['tick'], 4)
        self.assertEqual(record, forecast())

    def test_replay_rejects_invalid_prefix_profiles_and_unsealed_checkpoint(self):
        training = replay.evaluation.training
        valid = {'action_representation': training.RELATIVE_REPRESENTATION,
                 'observation_context_profile': 'state-history-physical-v1',
                 'image_profile': training.checkpoints.WORKSPACE_IMAGE_PROFILE, 'n_action_steps': 3}
        invalid = [{'n_action_steps': value} for value in (0, 51, 3.0, True)] + [
            {'action_representation': 'absolute-joint-position-v1'},
            {'observation_context_profile': 'none'}, {'image_profile': training.FULL_IMAGE_PROFILE}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'training-report.json').write_text(json.dumps(
                {'training_data_kind': 'captured', 'steps_completed': 1000}))
            (root / 'diagnostics').mkdir()
            with patch.dict(os.environ), patch.object(replay, 'ROOT', root), \
                 patch.object(replay, 'configure_runtime'), \
                 patch.object(training.legacy, 'require_runtime_versions'), \
                 patch.object(training.legacy, 'gpu_status', return_value={'index': 2}), \
                 patch.object(replay.evaluation.serving, 'inspect_checkpoint') as inspect, \
                 patch.object(training, 'read_dataset') as read_dataset:
                for number, changes in enumerate(invalid):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        inspect.return_value = {**valid, **changes}
                        replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / str(number), 2)
                inspect.side_effect = ValueError('unsealed checkpoint')
                with self.assertRaisesRegex(ValueError, 'unsealed checkpoint'):
                    replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / 'unsealed', 2)
                inspect.side_effect, inspect.return_value = None, valid
                (checkpoint / 'training-report.json').write_text(json.dumps(
                    {'training_data_kind': 'synthetic_diagnostic', 'steps_completed': 1000}))
                with self.assertRaisesRegex(ValueError, 'captured data'):
                    replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / 'synthetic', 2)
                read_dataset.assert_not_called()

    def test_prefix_three_replay_keeps_source_seven_cycles_and_metadata(self):
        import torch

        def cycle(number, base):
            return {'cycle': number, 'close_command_time': base - 10., 'lift_probe_command_time': base - 9.,
                    'lift_command_time': base - 8., 'hold_start': base - 7., 'hold_end': base - 2.,
                    'return_command_time': base - 2., 'release_command_time': base + 5.,
                    'reference': {'status': 'stable_jaw_contact_proxy', 'point_base_mm': [0., 0., number]}}

        source = self.source
        source.path, source.split, source.offset = Path('/recorded/source-7'), 'validation', 12.
        source.cycles = [cycle(1, 100.), cycle(2, 200.)]
        source.cycles[1]['reference'] = {'status': 'unavailable', 'reason': 'no recorded stationary feedback'}
        source.by_index = {frame: {'frame_index': frame, 'camera': {'source_timestamp_monotonic': stamp},
                                   'robot': {'received_monotonic': stamp}}
                           for frame, stamp in ((3407, 99.5), (3408, 100.), (3766, 199.5), (3767, 200.))}

        def feedback(timestamp, cycle, *, preceding=False):
            frames = [frame for frame in source.by_index.values()
                      if not preceding or frame['robot']['received_monotonic'] <= timestamp]
            frame = min(frames, key=lambda item: abs(item['robot']['received_monotonic'] - timestamp))
            result = {'available': True, 'gripper_feedback_valid': True, 'raw_gripper_position_percent': 78,
                      'frame_index': frame['frame_index']}
            if 'point_base_mm' in cycle['reference']:
                result['measured_grasp_vs_release_reference'] = {'height_along_table_normal_mm': float(cycle['cycle'])}
            return result

        source.feedback_at = feedback
        history = [{'age_s': .5, 'state': [1.] * 6 + [.78, 1.]}, None, None, None, None, None]
        rows = [{'_physical_source': str(index), '_source_capture_frame': frame, '_split': split,
                 '_camera_time': stamp, '_grid_time': stamp, '_physical_observation_history': copy.deepcopy(history),
                 'observation.state': [.1] * 6 + [.78, 1.]}
                for index, frame, split, stamp in ((6, 3408, 'train', 100.),
                                                    (7, 3408, 'validation', 100.),
                                                    (7, 3767, 'validation', 200.))]
        proof = {'sources': [{'source_index': 6, 'split': 'train', 'task_instruction': 'other'},
                             {'source_index': 7, 'split': 'validation', 'task_instruction': 'real task'}]}
        training = replay.evaluation.training
        info = {'action_representation': training.RELATIVE_REPRESENTATION,
                'observation_context_profile': 'state-history-physical-v1',
                'image_profile': training.checkpoints.WORKSPACE_IMAGE_PROFILE, 'n_action_steps': 3,
                'manifest': {'dataset_fingerprint': 'fingerprint', 'base': {'weights_path': '/base'}}}
        config = types.SimpleNamespace(n_action_steps=3, num_inference_steps=10, compile_model=False,
                                       use_relative_actions=False)
        native = torch.tensor([[[float(i) / 1000.] * 6 + [.76] for i in range(50)]])
        target = [[float(i)] * 7 for i in range(50)]
        policy = Mock()
        policy.to.return_value = policy
        policy.eval.return_value = policy
        policy.requires_grad_.return_value = policy
        policy.predict_action_chunk.return_value = native
        pre = types.SimpleNamespace(steps=[])
        post = Mock(side_effect=lambda value: value)
        post.steps = []
        pipeline = Mock()
        pipeline.from_pretrained.side_effect = [pre, post]
        modules = {name: types.ModuleType(name) for name in (
            'lerobot', 'lerobot.policies', 'lerobot.policies.pi05', 'lerobot.policies.pi05.configuration_pi05',
            'lerobot.policies.pi05.processor_pi05', 'lerobot.processor', 'lerobot.processor.converters')}
        modules['lerobot.policies.pi05.configuration_pi05'].PI05Config = types.SimpleNamespace(
            from_pretrained=Mock(return_value=config))
        modules['lerobot.policies.pi05'].processor_pi05 = modules['lerobot.policies.pi05.processor_pi05']
        modules['lerobot.processor'].DataProcessorPipeline = pipeline
        modules['lerobot.processor.converters'].policy_action_to_transition = object()
        modules['lerobot.processor.converters'].transition_to_policy_action = object()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'training-report.json').write_text(json.dumps(
                {'training_data_kind': 'captured', 'steps_completed': 1000}))
            (root / 'diagnostics').mkdir()
            output = root / 'diagnostics' / 'replay'
            with patch.dict(sys.modules, modules), patch.dict(os.environ), patch.object(replay, 'ROOT', root), \
                 patch.object(replay, 'configure_runtime'), \
                 patch.object(training.legacy, 'require_runtime_versions'), \
                 patch.object(training.legacy, 'gpu_status', return_value={'index': 2}), \
                 patch.object(replay.evaluation.serving, 'inspect_checkpoint', return_value=info) as inspect, \
                 patch.object(replay.evaluation.serving, 'validate_normalization'), \
                 patch.object(training, 'read_dataset', return_value=(rows, {6: [0], 7: [1, 2]}, None,
                                                                       proof, 'fingerprint')), \
                 patch.object(training, 'chunk_for_anchor', return_value=(target, [True] * 50)), \
                 patch.object(training.legacy, 'register_padding_step'), \
                 patch.object(training, 'load_policy', return_value=policy) as load_policy, \
                 patch.object(replay.evaluation, 'model_input', return_value=({}, 'real task')), \
                 patch.object(replay.support, 'GraspKinematics', return_value=FakeKinematics()), \
                 patch.object(replay.support, 'SourceEvidence', return_value=source) as evidence, \
                 patch.object(torch, 'manual_seed') as seed, \
                 patch.object(torch, 'autocast', side_effect=lambda *args, **kwargs: nullcontext()), \
                 patch.object(torch.cuda, 'get_device_name', return_value='offline test'):
                report = replay.run(root / 'dataset', checkpoint, output, 2,
                                    source_index=7, first_frame=3408, last_frame=3767)
                inspect.assert_called_once_with(checkpoint.resolve())
                evidence.assert_called_once_with(proof['sources'][1], unittest.mock.ANY)
                load_policy.assert_called_once()
                self.assertEqual(seed.call_args_list, [call(value) for value in replay.SEEDS] * 2)
                summary = replay.summarize_saved(output / 'predictions.jsonl', root / 'diagnostics' / 'summary.json')
                pipeline.from_pretrained.side_effect = [pre, post]
                one_cycle = replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / 'one-cycle', 2,
                                       source_index=7, first_frame=3408, last_frame=3408)
                self.assertEqual(one_cycle['recorded_reference'], source.cycles[0]['reference'])
                self.assertEqual(one_cycle['recorded_release_command_time'], 105.)
                self.assertEqual([cycle['cycle'] for cycle in one_cycle['recorded_cycles']], [1])
                config.n_action_steps = 6
                with self.assertRaisesRegex(ValueError, 'sealed prefix'):
                    replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / 'mismatched', 2,
                               source_index=7, first_frame=3408, last_frame=3767)
                with self.assertRaisesRegex(ValueError, 'absent from dataset provenance'):
                    replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / 'missing-source', 2,
                               source_index=8, first_frame=3408, last_frame=3767)
                with self.assertRaisesRegex(ValueError, 'No dataset rows match'):
                    replay.run(root / 'dataset', checkpoint, root / 'diagnostics' / 'missing-frames', 2,
                               source_index=7, first_frame=4000, last_frame=4001)
            items = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual((report['source_index'], report['source_split'], report['source_trial']),
                             (7, 'validation', '/recorded/source-7'))
            self.assertEqual(report['source_frame_range_inclusive'], [3408, 3767])
            self.assertEqual(report['input']['checkpoint_n_action_steps'], 3)
            self.assertEqual(report['input']['checkpoint_steps_completed'], 1000)
            self.assertEqual(report['input']['action_representation'], training.RELATIVE_REPRESENTATION)
            self.assertEqual(report['row_and_frame_mapping'], [
                {'row_index': 1, 'source_capture_frame': 3408, 'recorded_cycle': 1},
                {'row_index': 2, 'source_capture_frame': 3767, 'recorded_cycle': 2}])
            self.assertIsNone(report['recorded_reference'])
            self.assertEqual([cycle['cycle'] for cycle in report['recorded_cycles']], [1, 2])
            self.assertEqual(summary['recorded_cycles'], report['recorded_cycles'])
            self.assertEqual(summary['row_and_frame_mapping'], report['row_and_frame_mapping'])
            self.assertEqual(len(items), 6)
            for item, frame, history_frame, number in zip(items, [3408] * 3 + [3767] * 3,
                                                           [3407] * 3 + [3766] * 3, [1] * 3 + [2] * 3):
                self.assertEqual((item['record']['source_index'], item['record']['source_capture_frame']), (7, frame))
                self.assertEqual(item['physical_history'][0]['source_frame'], history_frame)
                self.assertEqual(item['analysis']['recorded_cycle'], number)
                self.assertEqual(item['analysis']['recorded_reference'], source.cycles[number - 1]['reference'])
                self.assertEqual(item['analysis']['anchor_feedback']['frame_index'], frame)
                self.assertAlmostEqual(item['analysis']['ticks'][0]['seconds_from_release_command'], -5. + 1. / 6.)
                self.assertEqual([tick['within_execution_prefix'] for tick in item['analysis']['ticks']],
                                 [True] * 3 + [False] * 7)
                self.assertEqual(item['record']['native_normalized_actions'], native[0].tolist())
                self.assertEqual(item['record']['model_actions'], native[0].tolist())
                expected_absolute = native[0].clone()
                expected_absolute[:, :6] += .1
                np.testing.assert_array_equal(item['record']['absolute_actions'], expected_absolute.tolist())
                self.assertEqual(item['record']['target_absolute'], target)

    def test_physical_history_identifies_the_recorded_frames_and_state(self):
        item = {'age_s': .5, 'state': [1.] * 6 + [.78, 1.]}
        row = {'_camera_time': 100., '_physical_observation_history': [item, {'age_s': 1.,
                'state': [2.] * 6 + [.78, 1.]}, None, None, None, None]}
        original = copy.deepcopy(row)
        history = replay.history_with_frames(row, self.source)
        self.assertEqual([entry['source_frame'] for entry in history], [15, 14, None, None, None, None])
        self.assertEqual(history[0]['observation'], item)
        self.assertEqual(row, original)
        row['_physical_observation_history'][0]['age_s'] = .6
        with self.assertRaisesRegex(ValueError, 'does not match'):
            replay.history_with_frames(row, self.source)

    def test_prefixes_use_same_unchanged_output_and_real_feedback(self):
        record = forecast()
        original = copy.deepcopy(record)
        result = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())
        self.assertIsNone(result['prefixes']['3']['90'])
        p6 = result['prefixes']['6']['90']
        self.assertEqual(p6['tick'], 4)
        self.assertAlmostEqual(p6['raw_gripper_percent'], 92.)
        self.assertEqual(p6['projected_integer_percent'], 92)
        self.assertEqual(p6['increase_over_anchor_measured_pp'], 14.)
        self.assertEqual(p6['increase_over_anchor_last_command_pp'], 16.)
        self.assertEqual(p6['predicted_grasp_vs_release_reference']['height_along_table_normal_mm'], 3.)
        self.assertEqual(p6['latest_preceding_source_feedback']['measured_grasp_vs_release_reference']
                         ['height_along_table_normal_mm'], 8.)
        self.assertIsNone(result['prefixes']['6']['98'])
        self.assertEqual(result['prefixes']['10']['98']['tick'], 8)
        self.assertAlmostEqual(result['prefixes']['10']['98']['raw_gripper_percent'], 102.)
        self.assertEqual(record, original)

    def test_invalid_tick_and_unavailable_feedback_do_not_invent_evidence(self):
        record = forecast()
        record['valid_mask'][3] = False
        with patch.object(self.source, 'feedback_at', return_value={'available': False}), \
             patch.object(self.source, 'last_gripper_command', return_value={'available': False, 'target_percent': None}):
            result = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())
        self.assertIsNone(result['prefixes']['6']['90'])
        tick = result['prefixes']['10']['90']
        self.assertEqual(tick['tick'], 8)
        self.assertIsNone(tick['increase_over_anchor_measured_pp'])
        self.assertIsNone(tick['increase_over_anchor_last_command_pp'])

    def test_summary_distinguishes_draws_anchors_and_recorded_release_time(self):
        record = forecast()
        first = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())
        second = copy.deepcopy(first)
        second['prefixes']['10']['90']['seconds_from_release_command'] = 1.
        summary = replay.summarize([{'record': record, 'analysis': first},
                                    {'record': record, 'analysis': second}])
        self.assertEqual(summary['3']['90']['draws_with_first_opening'], 0)
        self.assertEqual(summary['10']['90']['draws_with_first_opening'], 2)
        self.assertEqual(summary['10']['90']['distinct_source_anchors'], 1)
        self.assertEqual(summary['10']['90']['draws_before_recorded_release'], 1)
        self.assertEqual(summary['10']['98']['draws_with_first_opening'], 2)

    def test_projected_opening_below_90_compares_both_recorded_baselines(self):
        record = forecast()
        record['absolute_actions'][0][6] = .80
        record['absolute_actions'][1][6] = .79
        analysis = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())
        self.assertIsNone(analysis['prefixes']['3']['90'])
        report = replay.summarize_projected_targets([{'record': record, 'analysis': analysis}])
        measured = report['prefixes']['3']['anchor_measured']
        command = report['prefixes']['3']['anchor_last_move_gripper']
        self.assertEqual((measured['draws_at_least_2pp'], measured['points_at_least_2pp']), (1, 1))
        self.assertEqual(measured['maximum_increase_pp'], 2.)
        self.assertEqual((command['draws_at_least_2pp'], command['points_at_least_2pp']), (1, 2))
        self.assertEqual(command['maximum_increase_pp'], 4.)
        detail = report['anchors'][0]['prefixes']['3']['anchor_measured']['draws'][0]['first_at_least_2pp']
        self.assertEqual((detail['tick'], detail['seed'], detail['projected_integer_percent']), (1, 20261493, 80))
        self.assertEqual(detail['predicted_grasp_vs_release_reference']['height_along_table_normal_mm'], 0.)
        self.assertEqual(detail['latest_preceding_source_feedback']['measured_grasp_vs_release_reference']
                         ['height_along_table_normal_mm'], 8.)

    def test_projected_summary_excludes_missing_baselines(self):
        record = forecast()
        with patch.object(self.source, 'feedback_at', return_value={'available': False}), \
             patch.object(self.source, 'last_gripper_command', return_value={'available': False, 'target_percent': None}):
            analysis = replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())
        result = replay.summarize_projected_targets([{'record': record, 'analysis': analysis}])
        for baseline in ('anchor_measured', 'anchor_last_move_gripper'):
            value = result['prefixes']['3'][baseline]
            self.assertEqual((value['eligible_points'], value['missing_points']), (0, 3))
            self.assertIsNone(value['maximum_increase_pp'])

    def test_saved_replay_summary_uses_cpu_and_does_not_overwrite(self):
        record = forecast()
        item = {'record': record, 'analysis': replay.analyze_draw(record, self.source, self.cycle, FakeKinematics())}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recorded = root / 'recorded'
            recorded.mkdir()
            predictions = recorded / 'predictions.jsonl'
            predictions.write_text(json.dumps(item) + '\n')
            (recorded / 'report.json').write_text(json.dumps({
                'dataset': 'dataset', 'checkpoint': 'checkpoint', 'source_index': 6, 'source_trial': 'trial',
                'seeds_per_anchor': list(replay.SEEDS), 'recorded_reference': self.cycle['reference'], 'prefixes': {}}))
            (root / 'diagnostics').mkdir()
            output = root / 'diagnostics' / 'summary.json'
            with patch.object(replay, 'ROOT', root), patch.object(replay, 'run') as gpu_run:
                replay.main(['--saved-predictions', str(predictions), '--output', str(output)])
                gpu_run.assert_not_called()
                with self.assertRaises(FileExistsError):
                    replay.main(['--saved-predictions', str(predictions), '--output', str(output)])
            self.assertEqual(json.loads(output.read_text())['projected_targets']['anchors'][0]['row_index'], 42)


if __name__ == '__main__':
    unittest.main()
