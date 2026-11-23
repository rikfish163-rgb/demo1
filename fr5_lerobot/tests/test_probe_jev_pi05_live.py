"""CPU-only tests; no camera, robot, model server or GPU is contacted."""
from contextlib import nullcontext
import io
import json
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from fr5_lerobot.diagnostics import probe_jev_pi05_live as probe
from lerobot_robot_fairino.stream_contract import NAME
from lerobot_robot_fairino.stream_policy import StreamPolicy


def robot_state(timestamp, *, valid=True):
    return {
        'joints': [1., 2., 3., 4., 5., 6.],
        'received_monotonic': timestamp,
        'gripper_position': 70,
        'gripper_active': int(valid),
        'gripper_fault': 0,
        'gripper_fault_id': 0,
        'gripper_motiondone': 1,
        'gripper_speed': 0,
    }


def sample(timestamp, *, valid=True):
    return {
        'camera': {'color_rgb': np.zeros((360, 640, 3), dtype=np.uint8),
                   'metadata': {'source_timestamp_monotonic': timestamp,
                                'source_camera_serial': 'synthetic'}},
        'robot_state': robot_state(timestamp + .003, valid=valid),
        'sync_skew_s': .003,
        'sync_clock': 'host_monotonic',
    }


class Clock:
    def __init__(self, value):
        self.value = value

    def now(self):
        return self.value


class Pipe:
    def __init__(self):
        self.data = bytearray()

    def write(self, value):
        self.data.extend(value)
        return len(value)

    def close(self):
        pass


def policy_with_reply(clock, actions):
    """Use StreamPolicy's actual request transport and reply validation."""
    policy = object.__new__(StreamPolicy)
    policy.checkpoint = '/home/data4/hts/demo1/fr5-learning/checkpoints/example/step-002000'
    policy.checkpoint_sha = 'a' * 64
    policy.observation_context_profile = 'none'
    policy.n_action_steps = 10
    policy.ready = True
    pipe = Pipe()
    policy.process = Mock(stdin=pipe)

    def receive(timeout):
        request = json.loads(pipe.data)
        clock.value = 100.35
        return {'schema': 'fr5-stream-reply-v1', 'id': request['id'],
                'checkpoint_sha256': policy.checkpoint_sha, 'policy_contract': NAME,
                'observation_context_profile': 'none',
                'wire_action_representation': 'absolute-joint-position-v1',
                'inference_seconds': .12, 'actions': actions}

    policy.receive = Mock(side_effect=receive)
    return policy, pipe


class FakeStream:
    def __init__(self, clock, *, before_valid=True, after_valid=True):
        self.clock = clock
        self.before_valid = before_valid
        self.after_valid = after_valid
        self.calls = 0

    def latest(self, *, max_age_s):
        self.calls += 1
        if self.calls == 1:
            return robot_state(100.09, valid=self.before_valid)
        self.clock.value = 100.36
        return robot_state(100.36, valid=self.after_valid)


class ProbeTests(unittest.TestCase):
    def test_paired_policies_start_on_the_same_fresh_frame_before_either_returns(self):
        stamp = time.monotonic() - .01
        current = sample(stamp, valid=False)
        current['observation_history'] = [
            {'age_s': age, 'state': [0.] * 6 + [.7, 1.]}
            for age in (.5, 1., 2., 4., 6., 8.)
        ]
        observations = Mock()
        observations.wait.return_value = current
        robot = SimpleNamespace(stream=SimpleNamespace(
            latest=lambda **_: robot_state(time.monotonic(), valid=False)))
        ready = threading.Barrier(2)
        requests = {}
        actions = {name: [[value] * 6 + [.7] for _ in range(50)]
                   for name, value in (('current', .01), ('history', .02))}

        def policy(name, profile):
            def predict(request):
                requests[name] = request
                ready.wait(timeout=2)
                return {'inference_seconds': .1, 'actions': actions[name]}
            return SimpleNamespace(observation_context_profile=profile, predict=predict)

        policies = {'current': policy('current', probe.NONE_PROFILE),
                    'history': policy('history', probe.PHYSICAL_HISTORY_PROFILE)}
        rows = probe.sample_predictions(policies, robot, observations, '移动物体', after=stamp - 1.)

        observations.wait.assert_called_once_with(probe.PHYSICAL_HISTORY_PROFILE, after=stamp - 1.)
        self.assertEqual(set(requests), set(policies))
        self.assertEqual(requests['current']['source_time'], requests['history']['source_time'])
        self.assertEqual(requests['current']['state'], requests['history']['state'])
        self.assertEqual(requests['current']['rgb_jpeg_base64'], requests['history']['rgb_jpeg_base64'])
        self.assertNotIn('observation_history', requests['current'])
        self.assertEqual(len(requests['history']['observation_history']), 6)
        for name, row in rows.items():
            self.assertEqual(row['reply']['actions'], actions[name])
            self.assertFalse(row['paired_gripper_feedback_valid'])
            self.assertGreater(row['robot_feedback_after']['received_monotonic'],
                               row['reply_received_monotonic'])

    def test_paired_policies_require_complete_history_before_starting_predictions(self):
        policies = {name: Mock(observation_context_profile=profile)
                    for name, profile in (('current', probe.NONE_PROFILE),
                                          ('history', probe.PHYSICAL_HISTORY_PROFILE))}
        observations = Mock()
        observations.wait.side_effect = TimeoutError('incomplete history')
        with self.assertRaisesRegex(TimeoutError, 'incomplete history'):
            probe.sample_predictions(policies, Mock(), observations, '移动物体')
        for policy in policies.values():
            policy.predict.assert_not_called()

    def test_invalid_gripper_is_recorded_as_invalid_without_fabricating_feedback(self):
        clock = Clock(100.1)
        actions = [[float(i), 1., 2., 3., 4., 5., .7] for i in range(50)]
        policy, pipe = policy_with_reply(clock, actions)
        robot = Mock(stream=FakeStream(clock, before_valid=False, after_valid=False))
        observations = Mock()
        observations.wait.return_value = sample(100., valid=False)
        with patch.object(probe.time, 'monotonic', side_effect=clock.now):
            row = probe.sample_prediction(policy, robot, observations, '移动物体')

        self.assertEqual(row['request']['state'][6:], [0., 0.])
        self.assertEqual(row['paired_robot_feedback']['gripper_position'], 70)
        self.assertFalse(row['paired_gripper_feedback_valid'])
        self.assertFalse(row['before_gripper_feedback_valid'])
        self.assertFalse(row['after_gripper_feedback_valid'])
        self.assertEqual(json.loads(pipe.data)['state'], row['request']['state'])
        self.assertNotIn('op', json.loads(pipe.data))

    def test_incomplete_history_rejects_without_calling_the_policy(self):
        window = probe.LiveSamples(Mock())
        with patch.object(probe.time, 'monotonic', return_value=100.1):
            window._accept(sample(100.))
            wait = window.wait
            window.wait = lambda profile, after=None: wait(profile, after=after, timeout=0)
            policy = Mock(observation_context_profile=probe.PHYSICAL_HISTORY_PROFILE)
            with self.assertRaisesRegex(TimeoutError, 'complete history'):
                probe.sample_prediction(policy, Mock(), window, '移动物体')
        policy.predict.assert_not_called()

    def test_eight_seconds_of_actual_history_selects_all_six_offsets(self):
        window = probe.LiveSamples(Mock())
        for index in range(46):
            stamp = 100. + index * .2
            with patch.object(probe.time, 'monotonic', return_value=stamp + .01):
                window._accept(sample(stamp, valid=index != 35))
        with patch.object(probe.time, 'monotonic', return_value=109.02):
            current = window.wait(probe.PHYSICAL_HISTORY_PROFILE, timeout=0)
        history = current['observation_history']
        self.assertEqual(len(history), 6)
        self.assertTrue(all(item is not None for item in history))
        for item, offset in zip(history, (.5, 1., 2., 4., 6., 8.)):
            self.assertGreaterEqual(item['age_s'] + 1e-6, offset)
        self.assertEqual(history[2]['state'][6:], [0., 0.])

    def test_history_gap_is_rejected_instead_of_being_treated_as_continuous(self):
        window = probe.LiveSamples(Mock())
        with patch.object(probe.time, 'monotonic', return_value=100.01):
            window._accept(sample(100.))
        with patch.object(probe.time, 'monotonic', return_value=100.71):
            with self.assertRaisesRegex(ValueError, 'capture gap'):
                window._accept(sample(100.7))

    def test_next_prediction_requires_a_camera_frame_after_the_previous_reply(self):
        window = probe.LiveSamples(Mock())
        with patch.object(probe.time, 'monotonic', return_value=100.01):
            window._accept(sample(100.))
            with self.assertRaisesRegex(TimeoutError, 'fresh observation'):
                window.wait('none', after=100., timeout=0)
        with patch.object(probe.time, 'monotonic', return_value=100.21):
            window._accept(sample(100.2))
            result = window.wait('none', after=100., timeout=0)
        self.assertEqual(result['camera']['metadata']['source_timestamp_monotonic'], 100.2)

    def test_history_sampling_continues_while_a_prediction_is_in_flight(self):
        started, release = threading.Event(), threading.Event()
        errors = []
        robot = SimpleNamespace(
            config=SimpleNamespace(camera=SimpleNamespace(timeout_s=.2)),
            get_sample=lambda: sample(time.monotonic() - .02),
            stream=SimpleNamespace(latest=lambda **_: robot_state(time.monotonic())),
        )
        observations = probe.LiveSamples(robot)

        def predict(request):
            started.set()
            if not release.wait(3):
                raise TimeoutError('prediction was not released')
            return {'inference_seconds': .1, 'actions': [[0.] * 7 for _ in range(50)]}

        policy = SimpleNamespace(observation_context_profile='none', predict=predict)

        def run_prediction():
            try:
                probe.sample_prediction(policy, robot, observations, '移动物体')
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_prediction)
        observations.start()
        try:
            observations.wait('none', timeout=2)
            worker.start()
            self.assertTrue(started.wait(2))
            with observations.condition:
                previous = observations.history[-1][0]
                self.assertTrue(observations.condition.wait_for(
                    lambda: observations.history[-1][0] > previous or observations.error is not None,
                    timeout=2))
                self.assertIsNone(observations.error)
                self.assertGreater(observations.history[-1][0], previous)
        finally:
            release.set()
            if worker.ident is not None:
                worker.join(3)
            observations.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_full_native_chunk_and_separate_real_clock_durations_are_recorded(self):
        clock = Clock(100.1)
        actions = [[float(i), 1., 2., 3., 4., 5., .7] for i in range(50)]
        policy, _ = policy_with_reply(clock, actions)
        robot = Mock(stream=FakeStream(clock))
        observations = Mock()
        observations.wait.return_value = sample(100.)
        with patch.object(probe.time, 'monotonic', side_effect=clock.now):
            row = probe.sample_prediction(policy, robot, observations, '移动物体')
        self.assertEqual(row['reply']['actions'], actions)
        self.assertEqual(len(row['reply']['actions']), 50)
        self.assertTrue(all(len(action) == 7 for action in row['reply']['actions']))
        self.assertEqual(row['request_started_monotonic'], 100.1)
        self.assertEqual(row['reply_received_monotonic'], 100.35)
        self.assertAlmostEqual(row['source_to_reply_s'], .35)
        self.assertAlmostEqual(row['request_roundtrip_s'], .25)
        self.assertEqual(row['server_inference_seconds'], .12)
        self.assertLess(row['robot_feedback_before']['received_monotonic'], row['request_started_monotonic'])
        self.assertGreater(row['robot_feedback_after']['received_monotonic'], row['reply_received_monotonic'])
        restored = json.loads(json.dumps(row))
        self.assertEqual(restored['request']['state'], row['request']['state'])
        self.assertEqual(restored['reply']['actions'], actions)
        self.assertIn('rgb_jpeg_base64', restored['request'])

    def test_wrong_action_shape_is_rejected_by_the_production_client(self):
        for actions in ([[0.] * 7 for _ in range(49)], [[0.] * 6 for _ in range(50)],
                        [[float('nan')] + [0.] * 6 for _ in range(50)]):
            with self.subTest(actions=actions):
                clock = Clock(100.1)
                policy, _ = policy_with_reply(clock, actions)
                robot = Mock(stream=FakeStream(clock))
                observations = Mock()
                observations.wait.return_value = sample(100.)
                with patch.object(probe.time, 'monotonic', side_effect=clock.now):
                    with self.assertRaisesRegex(ValueError, '50 by 7'):
                        probe.sample_prediction(policy, robot, observations, '移动物体')
                self.assertEqual(robot.stream.calls, 1)

    def test_run_uses_only_read_only_observation_and_prediction(self):
        clock = Clock(100.1)
        actions = [[0.] * 6 + [1.] for _ in range(50)]
        policy, pipe = policy_with_reply(clock, actions)
        robot = Mock(stream=FakeStream(clock))
        observations = Mock()
        observations.wait.return_value = sample(100.)
        archive = Mock()
        with patch.object(probe.time, 'monotonic', side_effect=clock.now), \
                patch.object(probe, 'robot_lease', return_value=nullcontext()), \
                patch.object(probe, 'LiveSamples', return_value=observations):
            probe.run(policy, robot, archive, '移动物体', 1)

        robot.connect.assert_called_once_with()
        robot.disconnect.assert_called_once_with()
        robot.send_action.assert_not_called()
        self.assertEqual(robot.method_calls, [call.connect(), call.disconnect()])
        self.assertEqual([call.args[0]['type'] for call in archive.write.call_args_list],
                         ['session', 'prediction', 'complete'])
        self.assertEqual(json.loads(pipe.data)['schema'], 'fr5-stream-request-v1')
        self.assertNotIn('op', json.loads(pipe.data))
        observations.start.assert_called_once_with()
        observations.close.assert_called_once_with()

    def test_repeated_predictions_wait_for_frames_after_the_previous_reply(self):
        policy = Mock(observation_context_profile='none', n_action_steps=10,
                      checkpoint='checkpoint', checkpoint_sha='a' * 64)
        robot, observations, archive = Mock(), Mock(), Mock()
        rows = [{'type': 'prediction', 'reply_received_monotonic': 101.},
                {'type': 'prediction', 'reply_received_monotonic': 103.}]
        with patch.object(probe, 'robot_lease', return_value=nullcontext()), \
                patch.object(probe, 'LiveSamples', return_value=observations), \
                patch.object(probe, 'sample_prediction', side_effect=rows) as predict:
            probe.run(policy, robot, archive, '移动物体', 2)
        self.assertIsNone(predict.call_args_list[0].kwargs['after'])
        self.assertEqual(predict.call_args_list[1].kwargs['after'], 101.)
        self.assertEqual([call.args[0].get('index') for call in archive.write.call_args_list
                          if call.args[0]['type'] == 'prediction'], [0, 1])

    def test_rejects_non_data4_archive_before_starting_a_process(self):
        with patch.object(probe.subprocess, 'Popen') as process:
            with self.assertRaises(ValueError):
                probe.RemoteArchive('/tmp/probe.jsonl')
        process.assert_not_called()

    def test_failed_observation_is_recorded_and_disconnects_the_robot(self):
        policy = Mock(observation_context_profile=probe.PHYSICAL_HISTORY_PROFILE,
                      n_action_steps=10, checkpoint='checkpoint', checkpoint_sha='a' * 64)
        robot = Mock()
        observations = Mock()
        observations.wait.side_effect = TimeoutError('incomplete history')
        archive = Mock()
        with patch.object(probe, 'robot_lease', return_value=nullcontext()), \
                patch.object(probe, 'LiveSamples', return_value=observations):
            with self.assertRaisesRegex(TimeoutError, 'incomplete history'):
                probe.run(policy, robot, archive, '移动物体', 1)
        self.assertEqual([call.args[0]['type'] for call in archive.write.call_args_list],
                         ['session', 'error'])
        self.assertIn('incomplete history', archive.write.call_args.args[0]['error'])
        policy.predict.assert_not_called()
        observations.close.assert_called_once_with()
        robot.disconnect.assert_called_once_with()
        robot.send_action.assert_not_called()

    def test_archive_streams_extractable_request_and_reply_without_local_files(self):
        output = '/home/data4/hts/demo1/fr5-learning/diagnostics/example.jsonl'
        pipe = Pipe()
        stdout = Mock()
        stdout.readline.return_value = b'{"status":"ready"}\n'
        process = Mock(stdin=pipe, stdout=stdout, returncode=0)
        process.communicate.return_value = (b'{"status":"complete","records":1}\n', b'')
        record = {'type': 'prediction', 'request': {'state': [1.] * 8, 'rgb_jpeg_base64': 'AA=='},
                  'reply': {'actions': [[float(i)] * 7 for i in range(50)]}}
        with patch.object(probe.subprocess, 'Popen', return_value=process) as popen, \
                patch.object(probe.select, 'select', return_value=([stdout], [], [])):
            archive = probe.RemoteArchive(output)
            archive.write(record)
            archive.close()
        saved = json.loads(pipe.data)
        self.assertEqual(saved['request'], record['request'])
        self.assertEqual(saved['reply'], record['reply'])
        self.assertEqual(len(saved['reply']['actions']), 50)
        self.assertIn('4090', popen.call_args.args[0])
        self.assertIn(output, popen.call_args.args[0][-1])

    def test_remote_archive_program_writes_framed_json_without_modifying_records(self):
        record = {'type': 'prediction', 'request': {'state': [0.] * 8},
                  'reply': {'actions': [[0.] * 7 for _ in range(50)]}}
        payload = (json.dumps(record) + '\n').encode()

        class MemoryOutput(io.BytesIO):
            def fileno(self):
                return 42

            def __exit__(self, *args):
                return False

        output = MemoryOutput()
        status = io.StringIO()
        with patch.object(sys, 'argv', ['archive', '/home/data4/hts/demo1/fr5-learning/diagnostics/x.jsonl']), \
                patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(payload))), \
                patch.object(sys, 'stdout', status), \
                patch('pathlib.Path.mkdir'), patch('pathlib.Path.open', return_value=output), \
                patch('os.fsync') as fsync:
            exec(probe._REMOTE_ARCHIVE, {})
        self.assertEqual(json.loads(output.getvalue()), record)
        self.assertEqual([json.loads(line) for line in status.getvalue().splitlines()],
                         [{'status': 'ready'}, {'status': 'complete', 'records': 1}])
        fsync.assert_called_once_with(42)


if __name__ == '__main__':
    unittest.main()
