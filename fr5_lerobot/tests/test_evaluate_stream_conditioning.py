"""CPU-only checks for offline case selection and target-agreement accounting."""
import base64
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from fr5_lerobot.training import evaluate_stream_conditioning as evaluation


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        rows = []
        episodes = {}
        events = {'close': [], 'open': []}
        for source in range(2):
            indices = []
            for pos in range(120):
                index = len(rows)
                indices.append(index)
                grip = 1.0 if pos < 80 or pos >= 95 else .76
                rows.append({'_physical_source': str(source),
                             '_split': 'train' if source == 0 else 'validation',
                             'observation.state': [float(pos), 0, 0, 0, 0, 0, 1., 1.],
                             'action': [0, 0, 0, 0, 0, 0, grip]})
            episodes[source] = indices
            for kind, pos, old, new in (('close', 80, 1., .76), ('open', 95, .76, 1.)):
                events[kind].append({'episode': source, 'position': pos, 'index': indices[pos],
                                     'command_sequence': pos, 'old': old, 'new': new})
        return rows, episodes, events

    def test_approach_requires_valid_actual_open_and_full_future_open(self):
        rows, episodes, events = self.fixture()
        rows[30]['observation.state'][7] = 0
        rows[29]['observation.state'][6] = .8
        cases = evaluation.select_known_cases(rows, episodes, events, [30., 0, 0, 0, 0, 0, 1, 1])
        approaches = [c for c in cases if c['kind'] == 'approach_open']
        self.assertEqual([(c['row_index'], c['split']) for c in approaches],
                         [(28, 'train'), (150, 'validation')])
        self.assertEqual(approaches[0]['first_close_row_index'], 80)
        self.assertEqual(len([c for c in cases if c['kind'] == 'source_start']), 2)
        self.assertEqual(len([c for c in cases if c['kind'] == 'close_transition']), 2)
        self.assertEqual(len([c for c in cases if c['kind'] == 'open_transition']), 2)
        self.assertEqual([c['case_index'] for c in cases], list(range(8)))

    def test_no_eligible_anchor_and_bad_transition_are_rejected(self):
        rows, episodes, events = self.fixture()
        for i in episodes[0]:
            rows[i]['observation.state'][7] = 0
        with self.assertRaisesRegex(ValueError, 'no valid'):
            evaluation.select_known_cases(rows, episodes, events, [30., 0, 0, 0, 0, 0, 1, 1])
        rows, episodes, events = self.fixture()
        events['close'][0]['index'] += 1
        with self.assertRaisesRegex(ValueError, 'does not agree'):
            evaluation.select_known_cases(rows, episodes, events, [30., 0, 0, 0, 0, 0, 1, 1])

    def test_seed_blocks_are_distinct_and_reproducible_across_models(self):
        case0 = evaluation.seeds_for_case(0, 8)
        case1 = evaluation.seeds_for_case(1, 8)
        self.assertEqual(len(set(case0)), 8)
        self.assertFalse(set(case0) & set(case1))
        self.assertEqual(case1, evaluation.seeds_for_case(1, 8))
        with self.assertRaises(ValueError):
            evaluation.seeds_for_case(0, 1001)

    def test_match_is_per_step_and_not_just_first_action(self):
        targets = [[0.] * 6 + [1.]] * 5 + [[0.] * 6 + [.76]] * 5
        actions = [[0.] * 6 + [1.]] * 10
        metrics = evaluation.grip_metrics(actions, targets)
        self.assertTrue(metrics['first_step_matches'])
        self.assertEqual(metrics['matching_first10_steps'], 5)
        self.assertFalse(metrics['first10_all_match'])
        self.assertAlmostEqual(metrics['gripper_first10_mae_fraction'], .12)

    def test_first_gradient_and_exclusive_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log'
            path.write_text('first_gradient was mentioned\n{"status":"training"}\n')
            with self.assertRaisesRegex(RuntimeError, 'first_gradient'):
                evaluation.require_first_gradient(path)
            path.write_text('warning\n{"status":"first_gradient","test":1}\n')
            self.assertEqual(evaluation.require_first_gradient(path)['test'], 1)
            report = Path(directory) / 'report.json'
            evaluation.write_json_exclusive(report, {'a': 1})
            with self.assertRaises(FileExistsError):
                evaluation.write_json_exclusive(report, {'a': 2})
            self.assertEqual(json.loads(report.read_text()), {'a': 1})

    def test_gpu_guard_blocks_busy_or_wrong_gpu(self):
        good = {'index': 8, 'compute_processes': [], 'utilization_percent': 0,
                'memory_total_mib': 24564, 'memory_used_mib': 4}
        evaluation.require_free_gpu(good)
        for changed in ({'index': 9}, {'compute_processes': [['a', '123', '20']]},
                        {'utilization_percent': 10}, {'memory_used_mib': 3000}):
            with self.assertRaises(RuntimeError):
                evaluation.require_free_gpu({**good, **changed})
        gpu9 = {**good, 'index': 9}
        evaluation.require_free_gpu(gpu9, expected_index=9)
        for snapshot, index in ((good, 9), (gpu9, 8), (gpu9, -1),
                                ({**gpu9, 'compute_processes': [['a', '123', '20']]}, 9),
                                ({**gpu9, 'utilization_percent': 10}, 9),
                                ({**gpu9, 'memory_used_mib': 3000}, 9)):
            with self.assertRaises(RuntimeError):
                evaluation.require_free_gpu(snapshot, expected_index=index)

    def test_paths_accept_new_audit_and_reject_escape_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data4'
            root.mkdir()
            audits = root / 'audits'
            audits.mkdir()
            audit = audits / 'new-audit'
            checkpoint, dataset = root / 'checkpoints/model', root / 'datasets/source'
            log, output = audit / 'training.log', audit / 'step-000250'
            with mock.patch.object(evaluation, 'ROOT', root):
                self.assertEqual(evaluation.resolve_paths(checkpoint, dataset, output, audit, log),
                                 tuple(p.resolve() for p in (checkpoint, dataset, output, audit, log)))
                for bad_output, bad_audit, bad_log in (
                        (root / 'outside', audit, log),
                        (audits / 'other/output', audit, log),
                        (output, root / 'outside', log),
                        (output, audit, root / 'outside.log'),
                        (audit, audit, log)):
                    with self.assertRaises(ValueError):
                        evaluation.resolve_paths(checkpoint, dataset, bad_output, bad_audit, bad_log)
                external = Path(directory) / 'external'
                external.mkdir()
                (audits / 'escape').symlink_to(external, target_is_directory=True)
                with self.assertRaises(ValueError):
                    evaluation.resolve_paths(checkpoint, dataset, audits / 'escape/output', audits / 'escape', log)
                with self.assertRaises(ValueError):
                    evaluation.resolve_paths(checkpoint, dataset, output, audit, audits / 'escape/log')
                output.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    evaluation.resolve_paths(checkpoint, dataset, output, audit, log)

    def test_main_passes_requested_gpu_and_training_log_to_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / 'audits/new'
            log = audit / 'training.log'
            sha = 'a' * 64
            argv = ['evaluate', '--checkpoint', str(root / 'checkpoints/model'), '--expected-sha', sha,
                    '--dataset', str(root / 'datasets/source'), '--output', str(audit / 'step'),
                    '--gpu', '9', '--audit-root', str(audit), '--training-log', str(log)]
            snapshot = {'index': 9}
            with (mock.patch.object(evaluation, 'ROOT', root),
                  mock.patch.object(evaluation.sys, 'argv', argv),
                  mock.patch.object(evaluation, 'digest_file', return_value=sha),
                  mock.patch.object(evaluation, 'require_first_gradient', return_value={}) as gradient,
                  mock.patch.object(evaluation, 'gpu_snapshot', return_value=snapshot) as gpu,
                  mock.patch.object(evaluation, 'require_free_gpu', side_effect=RuntimeError('test stop')) as guard):
                with self.assertRaisesRegex(RuntimeError, 'test stop'):
                    evaluation.main()
            gradient.assert_called_once_with(log.resolve())
            gpu.assert_called_once_with(9)
            guard.assert_called_once_with(snapshot, expected_index=9)

    def test_historical_jpeg_is_bound_to_request_journal_and_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            (trial / 'stream-prediction').mkdir()
            (trial / 'continuous/frames').mkdir(parents=True)
            records, links, captures = [], [], []
            for number in (14, 15):
                jpeg = f'jpeg-{number}'.encode()
                sha = evaluation.digest_bytes(jpeg)
                request = {'id': str(number), 'source_time': float(number),
                           'rgb_jpeg_base64': base64.b64encode(jpeg).decode(),
                           'rgb_jpeg': {'sha256': sha, 'bytes': len(jpeg)}}
                reply = {'checkpoint_sha256': evaluation.PARENT_SHA}
                records.append({'request': request, 'reply': reply})
                (trial / f'chunk-{number:05d}.json').write_text(json.dumps(
                    {'request': {'id': str(number), 'source_time': float(number)}, 'reply': reply}))
                filename = f'frames/{number}.jpg'
                (trial / 'continuous' / filename).write_bytes(jpeg)
                captures.append({'frame_index': number, 'rgb_file': filename})
                links.append({'request_id': str(number), 'request_jpeg_hash_verified': True,
                              'rgb_jpeg_sha256': sha, 'source_timestamp_monotonic': float(number),
                              'frame_index': number})
            journal = trial / 'stream-prediction/part.jsonl'
            journal.write_text(''.join(json.dumps(r) + '\n' for r in records))
            index = {'chunks': [{'path': journal.name, 'bytes': journal.stat().st_size,
                                 'sha256': evaluation.digest_file(journal), 'rows': 2}], 'count': 2}
            (trial / 'stream-prediction/index.json').write_text(json.dumps(index))
            (trial / 'stream-prediction-sources.json').write_text(json.dumps(links))
            (trial / 'continuous/capture.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in captures))
            (trial / 'scope.json').write_text(json.dumps({'checkpoint_sha256': evaluation.PARENT_SHA}))
            result, _ = evaluation.load_trial(trial)
            self.assertTrue(result[15]['evidence']['jpeg_equals_capture_file'])
            (trial / 'continuous/frames/15.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'captured bytes'):
                evaluation.load_trial(trial)

    def test_recorded_history_uses_causal_capture_and_preserves_invalid_gripper(self):
        def capture(index, stamp, valid=True):
            return {'frame_index': index, 'camera': {'source_timestamp_monotonic': stamp},
                    'gripper_feedback_valid': valid,
                    'robot': {'joints': [90., 0., 0., 0., 0., 0.], 'received_monotonic': stamp + .01,
                              'gripper_position': 78}}
        captures = [capture(0, 1), capture(1, 2, False), capture(2, 2.5), capture(3, 3), capture(4, 3.5)]
        request = {'source_clock': 'host_monotonic', 'source_time': 3.0}
        history, provenance = evaluation.reconstruct_capture_history(captures, request, 3)
        self.assertEqual(provenance[0]['frame_index'], 2)
        self.assertEqual(provenance[1]['frame_index'], 1)
        self.assertEqual(history[1]['state'][6:], [0., 0.])
        self.assertAlmostEqual(history[1]['state'][0], 3.141592653589793 / 2)
        self.assertEqual(history[1]['age_s'], 1.)
        self.assertEqual(provenance[2]['frame_index'], 0)
        self.assertEqual(history[3:], [None, None, None])
        self.assertTrue(all(p is None or p['camera_time_monotonic'] <= 3 - offset
                            for p, offset in zip(provenance, evaluation.HISTORY_OFFSETS_S)))

    def test_missing_or_not_yet_received_capture_is_not_replaced(self):
        captures = [{'frame_index': i, 'camera': {'source_timestamp_monotonic': float(i)},
                     'gripper_feedback_valid': True,
                     'robot': {'joints': [0.] * 6, 'received_monotonic': float(i), 'gripper_position': 100}}
                    for i in range(1, 4)]
        captures[1]['robot'] = None
        request = {'source_clock': 'host_monotonic', 'source_time': 3.}
        history, _ = evaluation.reconstruct_capture_history(captures, request, 2)
        self.assertIsNone(history[1])
        captures[1]['robot'] = {'joints': [0.] * 6, 'received_monotonic': 3.1, 'gripper_position': 100}
        self.assertIsNone(evaluation.reconstruct_capture_history(captures, request, 2)[0][1])
        captures[1]['camera'] = {}
        self.assertEqual(evaluation.reconstruct_capture_history(captures, request, 2)[0], [None] * 6)

    def test_known_batch_forwards_profile_and_preserves_old_trainer_interface(self):
        trainer = SimpleNamespace(batch_for=mock.Mock(return_value=('batch', None)))
        model = SimpleNamespace(info={'image_profile': 'workspace-only-v1'}, pre='pre')
        evaluation.known_case_batch(trainer, model, 7, 'rows', 'episodes', 'dataset', 'state-history-v1')
        self.assertEqual(trainer.batch_for.call_args.kwargs['observation_context_profile'], 'state-history-v1')
        evaluation.known_case_batch(trainer, model, 7, 'rows', 'episodes', 'dataset', 'none')
        self.assertNotIn('observation_context_profile', trainer.batch_for.call_args.kwargs)

    def test_hover_cases_follow_recorded_hold_evidence_and_keep_original_seed_space(self):
        rows = [{} for _ in range(3510)]
        episodes, sources, examples = {}, [], []
        specs = ((0, 280, 340, ((291, 321),)), (1, 810, 860, ((821, 840),)),
                 (2, 900, 950, ((910, 920),)), (3, 1000, 1100, ((1010, 1020), (1040, 1050))),
                 (4, 1200, 1250, ((1210, 1220),)), (5, 1300, 1350, ((1310, 1320),)),
                 (6, 1400, 1450, ((1410, 1420),)), (7, 3360, 3480, ((3374, 3398), (3430, 3440))))
        for source, begin, end, boundaries in specs:
            split = 'validation' if source == 7 else 'train'
            episodes[source] = list(range(begin, end))
            for index in episodes[source]:
                rows[index] = {'_physical_source': str(source), '_split': split, '_grid_time': float(index)}
            records = []
            for repetition, (early, late) in enumerate(boundaries, start=1):
                start, finish = early - .1, late + .1
                records.append({'repetition': repetition, 'start_monotonic': start, 'end_monotonic': finish})
                if source in (0, 1, 7) and repetition == 1:
                    examples.append({'source': source, 'split': split, 'repetition': repetition, 'task': 'cup',
                                     'early': {'row': early, 'elapsed_s': early-start,
                                               'remaining_s': finish-early, 'frame': 1},
                                     'late': {'row': late, 'elapsed_s': late-start,
                                              'remaining_s': finish-late, 'frame': 2}})
            sources.append({'source_index': source, 'split': split, 'hold_records': records})
        cases = evaluation.select_hover_cases(rows, episodes, {'sources': sources}, {'examples': examples})
        self.assertEqual([c['row_index'] for c in cases[:6]], [291, 321, 821, 840, 3374, 3398])
        self.assertEqual([c['case_index'] for c in cases], list(range(38, 58)))
        self.assertEqual([(c['source_index'], c['hold_index'], c['repetition']) for c in cases[::2]],
                         [(0, 0, 1), (1, 0, 1), (7, 0, 1), (2, 0, 1), (3, 0, 1),
                          (3, 1, 2), (4, 0, 1), (5, 0, 1), (6, 0, 1), (7, 1, 2)])
        for case in cases[:6]:
            self.assertEqual(evaluation.seeds_for_case(case['case_index'], 8),
                             [20261103 + case['case_index'] * 1000 + draw for draw in range(8)])
            self.assertEqual(case['evidence_task'], 'cup')
        self.assertEqual(sum(c['split'] == 'train' for c in cases), 16)
        self.assertEqual(sum(c['split'] == 'validation' for c in cases), 4)
        self.assertFalse(set(evaluation.seeds_for_case(37, 8)) & set(evaluation.seeds_for_case(38, 8)))
        rows[840]['_grid_time'] = 0.
        with self.assertRaisesRegex(ValueError, 'recorded hold'):
            evaluation.select_hover_cases(rows, episodes, {'sources': sources}, {'examples': examples})

    def test_hover_metrics_measure_joint_error_and_return_projection(self):
        state = [0.] * 6 + [.78, 1.]
        target = [[0.] * 6 + [.76] for _ in range(10)]
        predicted = [[math.pi / 180] + [0.] * 5 + [.76] for _ in range(10)]
        metrics = evaluation.hover_joint_metrics(predicted, target, state, [1.] + [0.] * 5)
        self.assertAlmostEqual(metrics['joint_rmse_deg'], math.sqrt(1/6))
        self.assertAlmostEqual(metrics['joint_rmse_deg_per_joint'][0], 1.)
        self.assertAlmostEqual(metrics['predicted_step10_return_projection_deg'], 1.)
        self.assertEqual(metrics['target_step10_return_projection_deg'], 0.)

    def test_hover_output_is_separate_and_contains_per_case_history_and_seeds(self):
        rows, episodes, cases = [], {}, []
        holds = ((0, 0, 0), (1, 0, 0), (7, 0, 0), (3, 0, 0), (3, 1, 1), (7, 1, 2))
        for ep, (source, hold_index, axis) in enumerate(holds):
            episodes[ep] = []
            for pos in range(12):
                action = [0.] * 6 + [.76]
                action[axis] = (pos + 1) * .001
                episodes[ep].append(len(rows))
                rows.append({'observation.state': [0.] * 6 + [.78, 1.], 'action': action,
                             '_observation_history': [None] * 6})
            for pos, kind in enumerate(('hold_start', 'hold_end')):
                case = {'case_index': 38+len(cases), 'kind': kind, 'source_index': source,
                        'hold_index': hold_index, 'repetition': hold_index+1,
                        'split': 'validation' if source == 7 else 'train',
                        'row_index': episodes[ep][pos], 'episode': ep, 'position': pos}
                if ep < 3:
                    case['evidence_task'] = 'cup'
                cases.append(case)
        def chunk(rows_arg, indices, position):
            targets = [list(rows_arg[i]['action']) for i in indices[position:position + 50]]
            return targets, [True] * len(targets)
        trainer = SimpleNamespace(chunk_for_anchor=chunk)
        model = SimpleNamespace(info={'checkpoint_sha256': 'abc'}, pre=None)
        def batch(trainer_arg, model_arg, index, rows_arg, episodes_arg, dataset, profile):
            dataset.sample = {'task': 'cup'}
            self.assertEqual(profile, 'state-history-v1')
            return {'observation.state': SimpleNamespace(shape=(1, 32))}, None
        actions = [[.01, .02, .03] + [0.] * 3 + [.76] for _ in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence.json'
            evidence.write_text('{}')
            with (mock.patch.object(evaluation, 'HOVER_EVIDENCE', evidence),
                  mock.patch.object(evaluation, 'select_hover_cases', return_value=cases),
                  mock.patch.object(evaluation, 'known_case_batch', side_effect=batch),
                  mock.patch.object(evaluation, 'infer_batch', return_value=(actions, .1))):
                evaluation.evaluate_hover(root, trainer, model, None, rows,
                                          episodes, None, {}, 'state-history-v1', 8, evaluation.BASE_SEED)
            output = [json.loads(line) for line in (root / 'hover_cases.jsonl').read_text().splitlines()]
            report = json.loads((root / 'hover_report.json').read_text())
            self.assertEqual(len(output), 12)
            self.assertEqual([d['seed'] for d in output[0]['draws']], evaluation.seeds_for_case(38, 8))
            self.assertEqual(output[-1]['observation_history'], [None] * 6)
            self.assertEqual(len(report['pairs']), 6)
            for case in output:
                original = [rows[i]['action'] for i in
                            episodes[case['episode']][case['position']:case['position'] + 10]]
                self.assertEqual(case['gt_first10'], original)
            for source in (3, 7):
                source_cases = [c for c in output if c['source_index'] == source]
                self.assertEqual([c['hold_index'] for c in source_cases], [0, 0, 1, 1])
                expected_axes = (0, 1 if source == 3 else 2)
                self.assertEqual(source_cases[0]['return_direction_unit'], source_cases[1]['return_direction_unit'])
                self.assertEqual(source_cases[2]['return_direction_unit'], source_cases[3]['return_direction_unit'])
                source_pairs = [p for p in report['pairs'] if p['source_index'] == source]
                for hold_index, pair in enumerate(source_pairs):
                    matching = [c for c in source_cases if c['hold_index'] == hold_index]
                    axis = expected_axes[hold_index]
                    self.assertEqual(matching[0]['return_direction_unit'],
                                     [float(j == axis) for j in range(6)])
                    self.assertEqual((pair['hold_index'], pair['repetition']), (hold_index, hold_index + 1))
                    self.assertEqual([pair['hold_start_row'], pair['hold_end_row']],
                                     [c['row_index'] for c in matching])
                    for value in pair['prediction_mean_projection_deg']:
                        self.assertAlmostEqual(value, actions[0][axis] * 180 / math.pi)
                    for value, target in zip(pair['target_step10_return_projection_deg'], (.010, .011)):
                        self.assertAlmostEqual(value, target * 180 / math.pi)
            self.assertFalse((root / 'cases.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
