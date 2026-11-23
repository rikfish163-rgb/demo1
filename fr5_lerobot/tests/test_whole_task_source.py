"""Offline validation of reviewed teacher sources used by stream export."""
import json
from pathlib import Path
import tempfile
import unittest

from lerobot_robot_fairino.whole_task_source import load_source, sha, source_provenance
from fr5_lerobot.tests.stream_fixtures import TASK, make_source, write_json


def set_fixture_scope(root, role, *, object_description=None):
    scope_path, summary_path = root / 'scope.json', root / 'summary.json'
    scope = json.loads(scope_path.read_text())
    scope['split_role'] = role
    if object_description is not None:
        scope['task_spec']['object_description'] = object_description
    write_json(scope_path, scope)
    summary = json.loads(summary_path.read_text())
    summary['scope_sha256'] = sha(scope_path)
    summary['task_spec'] = scope['task_spec']
    write_json(summary_path, summary)
    scope_sha, summary_sha = sha(scope_path), sha(summary_path)
    manifest_path = root / 'control-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['scope_sha256'], manifest['summary_sha256'] = scope_sha, summary_sha
    for path in (scope_path, summary_path):
        manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    write_json(manifest_path, manifest)
    material_path = root / 'stream-materialization.json'
    material = json.loads(material_path.read_text())
    material['scope_sha256'], material['summary_sha256'] = scope_sha, summary_sha
    write_json(material_path, material)
    review_path = root / 'user-review.json'
    review = json.loads(review_path.read_text())
    review['scope_sha256'], review['summary_sha256'] = scope_sha, summary_sha
    write_json(review_path, review)
    return root


class WholeTaskSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='whole-task-offline-test-')
        self.root = Path(self.temp.name)


    def tearDown(self):
        self.temp.cleanup()


    def test_training_and_held_out_complete_sources_load_with_review(self):
        training = make_source(self.root / 'train')
        heldout = make_source(self.root / 'heldout', trial=1,
                              instruction='抓取桌面上的紫色方块，在空中维持5秒后放下')
        set_fixture_scope(heldout, 'held_out_complete_trial', object_description='紫色方块')
        for path, role in ((training, 'training_complete_trial'), (heldout, 'held_out_complete_trial')):
            with self.subTest(role=role):
                source = load_source(path)
                self.assertEqual(source['scope']['split_role'], role)
                self.assertEqual(source['review']['verdict'], 'success')
                self.assertTrue(source['segments'])
                self.assertEqual(source['data_provenance']['training_data_kind'], 'synthetic_diagnostic')
        source = load_source(heldout)
        self.assertEqual(source['task_spec']['object_description'], '紫色方块')
        self.assertEqual(source['task_instruction'], '抓取桌面上的紫色方块，在空中维持5秒后放下')


    def test_other_source_roles_are_rejected(self):
        for index, role in enumerate(('training_pilot_trial', 'held_out_pilot_trial', '', None)):
            with self.subTest(role=role):
                source = make_source(self.root / str(index), trial=index)
                set_fixture_scope(source, role)
                with self.assertRaisesRegex(ValueError, 'complete-trial split'):
                    load_source(source)


    def test_held_out_source_requires_complete_user_review(self):
        source = make_source(self.root / 'heldout')
        set_fixture_scope(source, 'held_out_complete_trial')
        review_path = source / 'user-review.json'
        review = json.loads(review_path.read_text())
        for verdict in ('pending', 'failure'):
            with self.subTest(verdict=verdict):
                write_json(review_path, {**review, 'verdict': verdict})
                with self.assertRaisesRegex(ValueError, 'User success review'):
                    load_source(source)
        review_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_source(source)
        write_json(review_path, review)
        (source / 'INCOMPLETE').touch()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_source(source)


    def test_actual_commands_get_one_prior_causal_observation(self):
        source = load_source(make_source(self.root / 'a'))
        decisions = source['segments'][0]
        self.assertEqual(len(decisions), 10)
        self.assertEqual(decisions[0]['phase'], 'approach_clear')
        self.assertEqual(decisions[0]['gripper_command_sequence'], 1)
        self.assertEqual(decisions[0]['gripper_target_source'], 'initial_hold')
        self.assertEqual(decisions[0]['arm_command_sequence'], 2)
        self.assertEqual(decisions[0]['arm_target_source'], 'wire')
        self.assertEqual([d['event_kind'] for d in decisions],
                         [0., 0., 0., 1., 0., 0., 0., 1., 0., 0.])
        self.assertTrue(decisions[0]['command_observation_matches_selected_frame'])
        self.assertEqual(len({d['command_sequence'] for d in decisions}), len(decisions))
        self.assertTrue(all(d['camera_time_monotonic'] <= d['command_issued_monotonic'] and
                            d['robot_time_monotonic'] <= d['command_issued_monotonic'] for d in decisions))
        self.assertNotEqual(decisions[0]['state_degrees_percent'][:6], decisions[0]['target_degrees_percent'][:6])
        self.assertAlmostEqual(source['holds'][0]['actual_duration_s'], 5.1)
        self.assertEqual(source['task_origin_basis'], 'initial_hold.issued_monotonic')
        self.assertEqual(source['task_origin_monotonic'], source['initial_hold']['issued_monotonic'])
        self.assertFalse(source['initial_hold']['executed'])
        for i, decision in enumerate(decisions):
            self.assertEqual(decision['completed_event_count'], i)
            self.assertAlmostEqual(decision['elapsed_task_seconds'],
                                   decision['camera_time_monotonic'] - source['task_origin_monotonic'])
            prior = source['initial_hold'] if i == 0 else decisions[i-1]
            prior_target = prior['target'] if i == 0 else prior['target_degrees_percent']
            self.assertEqual(decision['previous_target_degrees_percent'], prior_target)
            self.assertEqual(decision['previous_target_source'], 'initial_hold' if i == 0 else 'wire')
            self.assertEqual(decision['task_complete'], float(i == len(decisions)-1))
        lift_index = next(i for i, d in enumerate(decisions) if d['phase'] == 'lift_full')
        self.assertEqual(decisions[lift_index]['dwell_seconds'], 5.)
        self.assertAlmostEqual(decisions[lift_index]['dwell_actual_seconds'], 5.1)
        self.assertEqual(decisions[lift_index+1]['previous_dwell_seconds'], 5.)
        self.assertGreaterEqual(decisions[lift_index+1]['camera_time_monotonic'],
                                source['holds'][0]['start_monotonic'] + 5.)
        self.assertEqual(source['completion_evidence']['final_command_sequence'], decisions[-1]['command_sequence'])
        self.assertEqual(source['completion_evidence']['user_verdict'], 'success')


    def test_initial_hold_can_supply_arm_target_for_first_gripper_command(self):
        source = load_source(make_source(self.root / 'a', first_gripper_command=True))
        decisions = source['segments'][0]
        self.assertEqual(decisions[0]['command_kind'], 'gripper')
        self.assertEqual(decisions[0]['event_kind'], 1.)
        self.assertEqual(decisions[0]['arm_command_sequence'], 1)
        self.assertEqual(decisions[0]['arm_target_source'], 'initial_hold')
        self.assertEqual(decisions[0]['target_degrees_percent'][:6], [0.] * 6)
        self.assertNotEqual(decisions[0]['state_degrees_percent'][:6], [0.] * 6)
        self.assertEqual(decisions[1]['arm_target_source'], 'wire')


    def test_missing_hold_or_non_user_review_is_rejected(self):
        short = make_source(self.root / 'short', hold_duration=4.9)
        with self.assertRaisesRegex(ValueError, 'requested task duration'):
            load_source(short)
        source = make_source(self.root / 'unconfirmed', trial=1)
        review = json.loads((source / 'user-review.json').read_text())
        review['verdict'] = 'failure'
        write_json(source / 'user-review.json', review)
        with self.assertRaisesRegex(ValueError, 'User success review'):
            load_source(source)


    def test_missing_hold_image_breaks_continuity(self):
        source = make_source(self.root / 'a', skip_frame=30)
        with self.assertRaisesRegex(ValueError, 'continuous recorded frames'):
            load_source(source)


    def test_non_hold_recording_gap_splits_decisions(self):
        source = load_source(make_source(self.root / 'a', skip_frame=8))
        self.assertGreater(len(source['segments']), 1)
        decisions = [d for s in source['segments'] for d in s]
        sequences = [d['command_sequence'] for d in decisions]
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertEqual([d['completed_event_count'] for d in decisions], list(range(len(decisions))))
        self.assertEqual(sum(d['task_complete'] for d in decisions), 1.)


    def test_pre_initial_hold_source_frame_is_rejected(self):
        source = make_source(self.root / 'a', initial_hold_time=100.15)
        with self.assertRaisesRegex(ValueError, 'latched target disagrees|before the initial hold'):
            load_source(source)


    def test_missing_requested_dwell_cannot_be_invented_from_the_instruction(self):
        source = make_source(self.root / 'a')
        hold_path = source / 'hold-0001.json'
        hold = json.loads(hold_path.read_text())
        del hold['requested_seconds']
        write_json(hold_path, hold)
        manifest_path = source / 'control-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['hold-0001.json'] = {'bytes': hold_path.stat().st_size, 'sha256': sha(hold_path)}
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'recorded requested seconds'):
            load_source(source)


    def test_after_hold_decision_requires_observation_after_requested_wait(self):
        source = make_source(self.root / 'a', hold_duration=5., after_hold_delay_s=.001)
        with self.assertRaisesRegex(ValueError, 'observation after the previous completed command and requested dwell'):
            load_source(source)


    def test_wire_tampering_is_rejected(self):
        source = make_source(self.root / 'a')
        wire = json.loads((source / 'command-0002.json').read_text())
        wire['args'][0][0] = 999
        write_json(source / 'command-0002.json', wire)
        with self.assertRaises(ValueError):
            load_source(source)


    def test_recorded_requested_duration_still_must_match_the_task(self):
        source = make_source(self.root / 'a', hold_duration=8.1, hold_seconds=3.,
                             instruction='抓起杯子停3秒后放下')
        path = source / 'hold-0001.json'
        hold = json.loads(path.read_text())
        hold['requested_seconds'] = 5.
        write_json(path, hold)
        manifest = json.loads((source / 'control-manifest.json').read_text())
        manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        write_json(source / 'control-manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'requested task duration'):
            load_source(source)


    def test_long_wait_does_not_relax_clock_or_return_consistency(self):
        cases = (
            ({'actual_duration_s': 9.1}, 'requested task duration'),
            ({'actual_duration_s': 9.1, 'end_monotonic': 113.2}, 'completed lift and return'),
            ({'return_command_issued': True}, 'actual return command'),
            ({'actual_hover_from_lift_completion_s': 99.}, 'actual command clocks'),
            ({'timing_deviation_seconds': 99.}, 'actual command clocks'),
        )
        for i, (changes, message) in enumerate(cases):
            with self.subTest(changes=changes):
                source = make_source(self.root / str(i), hold_duration=8.1, hold_seconds=3.,
                                     instruction='抓起杯子停3秒后放下')
                path = source / 'hold-0001.json'
                hold = json.loads(path.read_text()); hold.update(changes)
                write_json(path, hold)
                manifest = json.loads((source / 'control-manifest.json').read_text())
                manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
                write_json(source / 'control-manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, message):
                    load_source(source)


    def test_current_collector_canonical_hold_file_without_summary_list(self):
        source = make_source(self.root / 'a', hold_duration=3.1, hold_seconds=3.,
                             instruction='抓起杯子停3秒，放回桌面', canonical_hold_without_summary_list=True)
        self.assertEqual(load_source(source)['holds'][0]['path'], 'hold-01.json')


    def test_two_real_holds_attach_to_their_lifts_and_only_last_command_ends_task(self):
        source = load_source(make_source(self.root / 'a', repetitions=2, hold_seconds=3., hold_duration=3.1,
                                         instruction='抓起停3秒放下，再抓起停3秒放下'))
        decisions = [d for segment in source['segments'] for d in segment]
        held = [i for i, decision in enumerate(decisions) if decision['dwell_seconds'] > 0]
        self.assertEqual(len(held), 2)
        self.assertEqual([decisions[i]['dwell_seconds'] for i in held], [3., 3.])
        self.assertEqual([decisions[i]['command_sequence'] for i in held],
                         [hold['lift_command_sequence'] for hold in source['holds']])
        for i in held:
            self.assertEqual(decisions[i+1]['previous_dwell_seconds'], 3.)
            self.assertAlmostEqual(decisions[i]['dwell_actual_seconds'], 3.1)
        self.assertEqual(sum(d['task_complete'] for d in decisions), 1.)
        self.assertEqual(decisions[-1]['task_complete'], 1.)

    def test_source_kind_is_determined_from_scope_and_camera_markers(self):
        def camera(serial, **flags):
            return [{'camera': {'source_camera_serial': serial, **flags}}]
        contract = {'contract': 'fr5-whole-task-collection-v1'}
        captured = source_provenance(contract, camera('CP1E54200015'))
        self.assertEqual(captured['training_data_kind'], 'captured')
        scope_synthetic = source_provenance({**contract, 'synthetic_id': 0}, camera('CP1E54200015'))
        self.assertEqual(scope_synthetic['training_data_kind'], 'synthetic_diagnostic')
        self.assertEqual(scope_synthetic['scope_markers']['synthetic_id'], 0)
        for metadata in (camera('SYNTHETIC'), camera('CP1E54200015', is_synthetic=True),
                         camera('CP1E54200015', simulated=1)):
            with self.subTest(camera=metadata):
                diagnostic = source_provenance(contract, metadata)
                self.assertEqual(diagnostic['training_data_kind'], 'synthetic_diagnostic')
                self.assertTrue(diagnostic['camera_markers'])
        with self.assertRaisesRegex(ValueError, 'capture contract'):
            source_provenance({}, camera('CP1E54200015'))

    def test_long_hold_preserves_requested_wait_and_actual_timing_deviation(self):
        source = make_source(self.root / 'a', hold_duration=8.1, hold_seconds=3.,
                             instruction='抓起杯子停3秒后放下')
        loaded = load_source(source)
        hold = loaded['holds'][0]
        self.assertEqual(hold['requested_seconds'], 3.)
        self.assertAlmostEqual(hold['actual_duration_s'], 8.1)
        # Include latency from lift completion through the actual return RPC,
        # including any delay after an older timer-only hold record ended.
        self.assertAlmostEqual(hold['actual_hover_from_lift_completion_s'], 8.4)
        self.assertAlmostEqual(hold['timing_deviation_seconds'], 5.4)

    def test_three_second_hold_then_replace_and_pick_again(self):
        source = make_source(self.root / 'a', hold_duration=3.1, hold_seconds=3., repetitions=2,
                             finish='held', instruction='抓起杯子停3秒，放下后再拿起来')
        loaded = load_source(source)
        self.assertEqual(loaded['normalized_task_spec']['repetitions'], 2)
        self.assertEqual(loaded['normalized_task_spec']['finish'], 'held')
        phases = [d['phase'] for segment in loaded['segments'] for d in segment]
        self.assertEqual(phases.count('close'), 2)
        self.assertEqual(phases.count('lift_full'), 2)
        self.assertEqual(phases.count('release'), 1)
        self.assertAlmostEqual(loaded['holds'][0]['actual_duration_s'], 3.1)
