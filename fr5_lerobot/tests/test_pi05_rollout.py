"""Offline direct-action regressions; all state and RPC objects are synthetic."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lerobot_robot_fairino.pi05_bridge import ADAPTER_SHA
from lerobot_robot_fairino.pi05_rollout import (
    DirectSettings, PolicyController, decoded_whole_task_action, direct_action, run,
)


class ProbeDispatched(Exception):
    """The in-memory probe reached the dispatch gate; nothing was sent."""


class FakeRPC:
    def __init__(self):
        self.calls = []
        self.fk_reply = None
        self.on_dispatch = None

    def call(self, method, *args, **kwargs):
        if method == 'GetForwardKin':
            self.calls.append((method, deepcopy(args)))
            return deepcopy(self.fk_reply) if self.fk_reply is not None else [0, *args[0]]
        if method in ('MoveL', 'MoveGripper'):
            with kwargs['before_send']():
                self.calls.append((method, deepcopy(args)))
                if self.on_dispatch is not None:
                    self.on_dispatch(method, args)
                    return 0
                raise ProbeDispatched(method)
        raise AssertionError(f'Unexpected synthetic RPC {method}')


class DirectActionTests(unittest.TestCase):
    def setUp(self):
        # All motion and state are synthetic; any real socket or process is a test error.
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('Offline tests cannot contact devices'))
            guard.start()
            self.addCleanup(guard.stop)
        self.rpc = FakeRPC()
        joints = [10., -8., 5., 4., -3., 2.]
        self.state = dict(joints=joints.copy(), tcp=joints.copy(), flange=joints.copy(),
                          gripper_position=50., gripper_active=1, gripper_fault=0,
                          gripper_fault_id=0, gripper_motiondone=1, gripper_speed=0,
                          main_code=0, sub_code=0, EmergencyStop=0, collisionState=0,
                          safety_stop0_state=0, safety_stop1_state=0, rbtEnableState=1,
                          program_state=1, robot_state=1, motion_done=1, mc_queue_len=0,
                          joint_speeds_deg_s=[0.] * 6, tool=0, user=0, frame_counter=1)
        self.bounds = np.array([[-90., 90.]] * 6)
        self.stream = Mock()
        def read_state(**_):
            self.state['frame_counter'] += 1
            return deepcopy(self.state)
        self.stream.latest.side_effect = read_state
        self.camera_fresh = Mock(return_value=True)
        self.settings = DirectSettings()
        self.controller = PolicyController(self.stream, self.camera_fresh, self.settings,
                                           allow_motion=True, rpc=self.rpc)
        self.controller.armed = True
        self.controller.deadman = False
        self.controller.deadline = 101.
        self.controller.joint_limits = self.bounds.copy()

    def direct(self, raw):
        return direct_action(raw, self.state, self.rpc, self.bounds)

    def test_direct_preserves_full_target_and_uses_fk_without_ik(self):
        for grip in (0., .504, 1.):
            with self.subTest(grip=grip):
                self.rpc.calls.clear()
                raw = [89., -89., 45., -55., 50., -80., grip]
                original_raw, original_state = deepcopy(raw), deepcopy(self.state)
                result = self.direct(raw)
                np.testing.assert_array_equal(result['raw_target'], raw)
                np.testing.assert_array_equal(result['arm_target'][:6], raw[:6])
                self.assertEqual(result['arm_target'][6], self.state['gripper_position'])
                self.assertAlmostEqual(result['gripper_target_percent'], grip * 100.)
                self.assertEqual(result['raw_target_units'], 'joint_degrees_and_gripper_fraction')
                np.testing.assert_array_equal(result['target_pose'], raw[:6])
                self.assertEqual(result['fraction'], 1.)
                self.assertEqual(self.rpc.calls, [('GetForwardKin', (raw[:6],))])
                self.assertEqual(raw, original_raw)
                self.assertEqual(self.state, original_state)

    def test_native_joint_limit_boundaries_are_accepted_and_outside_rejected(self):
        for axis in range(6):
            for value in (-90., 90., -89.9, 89.9):
                with self.subTest(axis=axis, value=value):
                    raw = [*self.state['joints'], .5]
                    raw[axis] = value
                    np.testing.assert_array_equal(self.direct(raw)['arm_target'][:6], raw[:6])
            for value in (-90.001, 90.001):
                raw = [*self.state['joints'], .5]
                raw[axis] = value
                self.rpc.calls.clear()
                with self.subTest(axis=axis, value=value), self.assertRaises((ValueError, RuntimeError)):
                    self.direct(raw)
                self.assertEqual(self.rpc.calls, [])

    def test_bad_target_or_gripper_feedback_rejected_without_dispatch(self):
        good = [*self.state['joints'], .5]
        bad_targets = [[], good[:-1], good + [0.], [[0.] * 7]]
        for i in range(7):
            for bad in (float('nan'), float('inf'), float('-inf')):
                raw = good.copy()
                raw[i] = bad
                bad_targets.append(raw)
        for raw in bad_targets:
            with self.subTest(raw=raw):
                self.rpc.calls.clear()
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    self.direct(raw)
                self.assertEqual(self.rpc.calls, [])
        for key, value in (('gripper_active', 0), ('gripper_fault', 1), ('gripper_fault_id', 1),
                           ('gripper_motiondone', 3), ('gripper_position', float('nan'))):
            with self.subTest(field=key):
                old = self.state[key]
                self.state[key] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    self.direct(good)
                self.state[key] = old

    def test_bad_limits_and_fk_replies_rejected(self):
        raw = [*self.state['joints'], .5]
        for bounds in ([[-90., 90.]] * 5, [[float('nan'), 90.]] * 6,
                       [[float('-inf'), 90.]] * 6, [[90., -90.]] * 6):
            with self.subTest(bounds=bounds), self.assertRaises((ValueError, RuntimeError)):
                direct_action(raw, self.state, self.rpc, bounds)
        for reply in ([1], [0] * 6, [0, float('nan'), 0., 0., 0., 0., 0.]):
            self.rpc.fk_reply = reply
            with self.subTest(reply=reply), self.assertRaises((ValueError, RuntimeError)):
                self.direct(raw)

    def test_finite_gripper_predictions_project_before_percent_and_rounding(self):
        for raw_fraction, fraction, percent, wire in (
            (-1.e308, 0., 0., 0), (-.01, 0., 0., 0), (0., 0., 0., 0),
            (.504, .504, 50.4, 50), (.506, .506, 50.6, 51),
            (1., 1., 100., 100), (1.0219240188598633, 1., 100., 100),
            (1.e308, 1., 100., 100),
        ):
            with self.subTest(raw_fraction=raw_fraction):
                raw = [*self.state['joints'], raw_fraction]
                plan = self.direct(raw)
                adaptation = plan['gripper_adaptation']
                self.assertEqual(adaptation['model_fraction'], raw_fraction)
                self.assertEqual(adaptation['projected_fraction'], fraction)
                self.assertAlmostEqual(adaptation['projected_percent'], percent)
                self.assertEqual(adaptation['planned_wire_integer_percent'], wire)
                self.assertAlmostEqual(plan['gripper_target_percent'], percent)
                self.assertEqual(plan['raw_target'], raw)
                # The persisted plan must not contain an overflowed percentage.
                json.dumps(plan, allow_nan=False)

    def test_model_reply_projection_reaches_real_controller_adapter_and_log(self):
        self.rpc.on_dispatch = lambda method, args: self.state.update(gripper_position=float(args[1]))
        for raw_fraction, expected_percent, expected_wire, sent in (
            (-.01, 0., 0, True), (.504, 50.4, 50, False),
            (.506, 50.6, 51, True), (1.0219240188598633, 100., 100, True),
        ):
            with self.subTest(raw_fraction=raw_fraction):
                self.state['gripper_position'] = 50.
                self.rpc.calls.clear()
                request = {'id': 'offline-request'}
                action = [*np.deg2rad(self.state['joints']).tolist(), raw_fraction]
                reply = {'schema': 'fr5-pi05-reply-v1', 'id': request['id'],
                         'adapter_sha256': ADAPTER_SHA,
                         'action_representation': 'absolute_joint_rad_gripper_fraction',
                         'actions': [action.copy() for _ in range(8)]}
                original = deepcopy(reply)
                decoded = decoded_whole_task_action(reply, request)
                plan = self.direct(decoded)
                with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
                     patch.object(self.controller, '_gripper_complete', return_value=True):
                    result = self.controller.execute(
                        [*self.state['joints'], plan['gripper_target_percent']], kind='gripper')
                record = json.loads(json.dumps({'reply': reply, 'plan': plan, 'commands': [result]},
                                               allow_nan=False))
                self.assertEqual(reply, original)
                self.assertEqual(record['reply']['actions'][0][6], raw_fraction)
                self.assertEqual(record['plan']['gripper_adaptation']['model_fraction'], raw_fraction)
                self.assertEqual(record['plan']['gripper_adaptation']['projected_fraction'],
                                 min(1., max(0., raw_fraction)))
                self.assertEqual(record['plan']['gripper_adaptation']['planned_wire_integer_percent'],
                                 expected_wire)
                self.assertAlmostEqual(result['gripper_device_quantization']['requested_percent'],
                                       expected_percent)
                self.assertEqual(result['gripper_device_quantization']['wire_integer_percent'], expected_wire)
                calls = [args for method, args in self.rpc.calls if method == 'MoveGripper']
                self.assertEqual(len(calls), int(sent))
                if sent:
                    self.assertEqual(result['status'], 'completed')
                    self.assertEqual(calls[0][1], expected_wire)
                    self.assertEqual(result['command']['target'][6], expected_wire)
                else:
                    self.assertEqual(result['status'], 'hold_no_command')

    def test_nonfinite_model_reply_rejected_before_projection(self):
        request = {'id': 'offline-request'}
        for index in (0, 7):
            for bad in (float('nan'), float('inf'), float('-inf')):
                with self.subTest(index=index, value=bad):
                    reply = {'schema': 'fr5-pi05-reply-v1', 'id': request['id'],
                             'adapter_sha256': ADAPTER_SHA,
                             'action_representation': 'absolute_joint_rad_gripper_fraction',
                             'actions': [[0.] * 7 for _ in range(8)]}
                    reply['actions'][index][6] = bad
                    with self.assertRaises(ValueError):
                        decoded_whole_task_action(reply, request)

    def test_very_large_finite_model_fraction_does_not_overflow_decoding(self):
        request = {'id': 'offline-request'}
        for value in (-1.e308, 1.e308):
            reply = {'schema': 'fr5-pi05-reply-v1', 'id': request['id'],
                     'adapter_sha256': ADAPTER_SHA,
                     'action_representation': 'absolute_joint_rad_gripper_fraction',
                     'actions': [[0., 0., 0., 0., 0., 0., value] for _ in range(8)]}
            decoded = decoded_whole_task_action(reply, request)
            self.assertEqual(decoded[6], value)
            self.assertTrue(np.isfinite(decoded).all())
            self.assertEqual(self.direct(decoded)['gripper_target_percent'], 0. if value < 0 else 100.)

    def test_whole_task_run_saves_raw_projection_and_dispatched_wire_value(self):
        task = '抓取桌面上的透明杯子，在空中维持5秒后放下'
        request = {'id': 'offline-request', 'task': task}
        raw_fraction = 1.0219240188598633
        action = [*np.deg2rad(self.state['joints']).tolist(), raw_fraction]
        reply = {'schema': 'fr5-pi05-reply-v1', 'id': request['id'],
                 'adapter_sha256': ADAPTER_SHA,
                 'action_representation': 'absolute_joint_rad_gripper_fraction',
                 'actions': [action.copy() for _ in range(8)]}
        policy = Mock(ready=True)
        policy.predict.return_value = reply
        observer = Mock(is_connected=True)
        observer.stream = self.stream
        observer.get_sample.side_effect = lambda: {'robot_state': deepcopy(self.state)}
        storage = Mock()
        uploaded = []
        storage.sync_files.side_effect = lambda path: uploaded.append(json.loads(Path(path).read_text()))

        def dispatch(method, args):
            self.assertEqual(method, 'MoveGripper')
            self.assertTrue(any(row.get('plan', {}).get('gripper_adaptation', {})
                                .get('planned_wire_integer_percent') == 100 for row in uploaded))
            self.state['gripper_position'] = float(args[1])
        self.rpc.on_dispatch = dispatch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scope.json').write_text(json.dumps({
                'task': task, 'initial_tcp': self.state['tcp'],
                'max_steps': 1, 'max_seconds': 20,
            }), encoding='utf-8')
            with patch('lerobot_robot_fairino.pi05_rollout.SSHPolicy', return_value=policy), \
                 patch('lerobot_robot_fairino.pi05_rollout.validate_scope', return_value=self.settings), \
                 patch('lerobot_robot_fairino.robot_fairino.FairinoFR5', return_value=observer), \
                 patch('fr5_jev.stream_storage.StreamStorage.start_readonly', return_value=storage), \
                 patch('lerobot_robot_fairino.pi05_rollout.PolicyController', return_value=self.controller), \
                 patch('lerobot_robot_fairino.pi05_rollout.observation_request', return_value=request) as observe, \
                 patch('lerobot_robot_fairino.pi05_rollout.feedback_unchanged'), \
                 patch('lerobot_robot_fairino.pi05_rollout.request_age', return_value=0.), \
                 patch('lerobot_robot_fairino.pi05_rollout.signal.signal'), \
                 patch.object(self.controller, 'arm'), patch.object(self.controller, 'stop'), \
                 patch.object(self.controller, '_gripper_complete', return_value=True), \
                 redirect_stdout(io.StringIO()):
                report = run(root, execute=True)
            row = json.loads((root / 'step-0000.json').read_text())
        self.assertEqual(report['status'], 'trial_limit_reached_task_not_verified')
        self.assertEqual(row['status'], 'executed')
        self.assertEqual(row['reply']['actions'][0][6], raw_fraction)
        self.assertEqual(row['plan']['gripper_adaptation'], {
            'model_fraction': raw_fraction, 'projected_fraction': 1.,
            'projected_percent': 100., 'planned_wire_integer_percent': 100,
        })
        result = row['commands'][-1]
        self.assertEqual(result['gripper_device_quantization']['wire_integer_percent'], 100)
        self.assertEqual(result['command']['target'][6], 100)
        self.assertEqual([args[1] for method, args in self.rpc.calls if method == 'MoveGripper'], [100])
        self.assertTrue(all(call.args[1] == task for call in observe.call_args_list))

    def test_direct_settings_and_large_arm_endpoint(self):
        self.assertEqual((self.settings.speed_percent, self.settings.gripper_step_percent,
                          self.settings.gripper_speed_percent, self.settings.gripper_force_percent,
                          self.settings.command_timeout_s), (10., 100., 20, 10, 30.))
        target = [89., -89., 45., -55., 50., -80., 50.]
        np.testing.assert_array_equal(self.controller._validate_arm_target(np.array(target), self.state), target[:6])
        self.assertEqual(self.rpc.calls, [('GetForwardKin', (target[:6],))])

    def test_full_targets_reach_synthetic_dispatch_with_direct_settings(self):
        for kind in ('arm', 'gripper'):
            with self.subTest(kind=kind):
                self.rpc.calls.clear()
                target = ([89., -89., 45., -55., 50., -80., 50.] if kind == 'arm'
                          else [*self.state['joints'], 0.])
                with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
                     patch.object(self.controller, 'stop'):
                    with self.assertRaises(ProbeDispatched):
                        self.controller.execute(target, kind=kind)
                calls = [(method, args) for method, args in self.rpc.calls if method.startswith('Move')]
                self.assertEqual(len(calls), 1)
                method, args = calls[0]
                if kind == 'arm':
                    self.assertEqual(method, 'MoveL')
                    self.assertEqual(args[0], target[:6])
                    self.assertEqual(args[4], 10.)
                else:
                    self.assertEqual(method, 'MoveGripper')
                    self.assertEqual(args[1:4], (0, 20, 10))

    def test_no_space_emulation_and_fault_watchdog(self):
        self.controller.busy = True
        self.controller._heartbeat = -1.e6
        with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
             patch.object(self.controller, 'stop') as stop:
            self.controller._gate()
            self.controller.watchdog()
            stop.assert_not_called()
            self.state['gripper_motiondone'] = 3
            self.controller.watchdog()
            stop.assert_called_once()
            self.assertIn('lost', stop.call_args.args[0])

    def test_nonfinite_current_state_is_rejected(self):
        for key in ('joints', 'tcp', 'flange'):
            with self.subTest(field=key):
                old = self.state[key].copy()
                self.state[key][1] = float('nan')
                with self.assertRaises((ValueError, RuntimeError)):
                    self.controller.ready_state()
                self.state[key] = old

    def test_small_direct_arm_target_is_not_silently_dropped(self):
        target = [*self.state['joints'], self.state['gripper_position']]
        target[0] += .005
        with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
             patch.object(self.controller, 'stop'):
            with self.assertRaises(ProbeDispatched):
                self.controller.execute(target, kind='arm')
        self.assertTrue(any(method == 'MoveL' for method, _ in self.rpc.calls))

    def test_gripper_integer_quantization_hold_and_next_integer(self):
        target = [*self.state['joints'], self.state['gripper_position'] + .4]
        with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.):
            result = self.controller.execute(target, kind='gripper')
        self.assertEqual(result['status'], 'hold_no_command')
        self.assertEqual(result['gripper_device_quantization'],
                         {'requested_percent': 50.4, 'wire_integer_percent': 50})
        self.assertFalse(any(method == 'MoveGripper' for method, _ in self.rpc.calls))

        target[6] = 50.6
        self.rpc.on_dispatch = lambda method, args: self.state.update(gripper_position=float(args[1]))
        # This case isolates integer command mapping. Completion timing and
        # busy feedback are covered independently in test_pi05_completion.
        with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
             patch.object(self.controller, '_gripper_complete', return_value=True):
            result = self.controller.execute(target, kind='gripper')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['gripper_device_quantization'],
                         {'requested_percent': 50.6, 'wire_integer_percent': 51})
        calls = [args for method, args in self.rpc.calls if method == 'MoveGripper']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 51)

    def test_component_kind_handles_opposite_component_feedback_jitter(self):
        target = [*self.state['joints'], self.state['gripper_position']]
        target[0] += .005
        self.state['gripper_position'] += .8
        with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
             patch.object(self.controller, 'stop'):
            with self.assertRaises(ProbeDispatched):
                self.controller.execute(target, kind='arm')
        self.assertTrue(any(method == 'MoveL' for method, _ in self.rpc.calls))

        self.rpc.calls.clear()
        target = [*self.state['joints'], 0.]
        self.state['joints'][0] += .04
        with patch('lerobot_robot_fairino.pi05_rollout.time.monotonic', return_value=100.), \
             patch.object(self.controller, 'stop'):
            with self.assertRaises(ProbeDispatched):
                self.controller.execute(target, kind='gripper')
        self.assertTrue(any(method == 'MoveGripper' for method, _ in self.rpc.calls))


if __name__ == '__main__':
    unittest.main()
