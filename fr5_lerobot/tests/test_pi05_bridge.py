"""Offline bridge tests. All robot/model/executor objects are synthetic."""
from copy import deepcopy
import math
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

import numpy as np

from lerobot_robot_fairino.pi05_bridge import (
    ADAPTER_SHA, Pi05StepBridge, action_target, observation_request, solve_tcp_target)


def state():
    return dict(joints=[0.]*6, tcp=[0., 0., 100., 0., 0., 0.],
                flange=[0., 0., 100., 0., 0., 0.], tool=0, user=0,
                gripper_position=100, gripper_active=1, gripper_motiondone=1,
                gripper_speed=0, gripper_fault=0, gripper_fault_id=0,
                main_code=0, sub_code=0, EmergencyStop=0, collisionState=0,
                safety_stop0_state=0, safety_stop1_state=0, motion_done=1,
                mc_queue_len=0, robot_state=1, rbtEnableState=1)


def sample(s=None):
    return {'robot_state': deepcopy(s or state()), 'sync_skew_s': .01,
            'camera': {'color_rgb': np.zeros((360, 640, 3), np.uint8),
                       'metadata': {'source_timestamp_monotonic': time.monotonic()}}}


def reply(request, grip=1.):
    return {'schema': 'fr5-pi05-reply-v1', 'id': request['id'], 'adapter_sha256': ADAPTER_SHA,
            'action_representation': 'absolute_joint_rad_gripper_fraction',
            'actions': [[math.pi/180, 0., 0., 0., 0., 0., grip] for _ in range(8)],
            'inference_seconds': .1}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.current = state()
        self.observer = SimpleNamespace(get_sample=Mock(side_effect=lambda: sample(self.current)),
            stream=SimpleNamespace(latest=Mock(side_effect=lambda **_: deepcopy(self.current))))
        self.policy = SimpleNamespace(predict=Mock(side_effect=reply))
        self.executor = SimpleNamespace(armed=True, execute=Mock(return_value={'status': 'completed'}))
        self.bridge = Pi05StepBridge(self.observer, self.policy, self.executor)

    def test_preview_converts_units_and_does_not_execute(self):
        report = self.bridge.step('approach_clear', 'Approach the cup')
        self.assertAlmostEqual(report['proposal']['target_deg_percent'][0], 1.)
        self.assertEqual(report['proposal']['target_deg_percent'][-1], 100.)
        self.executor.execute.assert_not_called()
        self.assertFalse(report['task_success_claimed'])

    def test_execute_refreshes_observation_and_uses_new_request_each_step(self):
        self.bridge.step('approach_clear', 'Approach the cup', execute=True)
        self.bridge.step('approach_clear', 'Approach the cup', execute=True)
        self.assertEqual(self.observer.get_sample.call_count, 4)
        self.assertEqual(self.executor.execute.call_count, 2)
        ids = [call.args[0]['id'] for call in self.policy.predict.call_args_list]
        self.assertNotEqual(*ids)

    def test_model_response_cannot_be_used_for_another_frame_or_checkpoint(self):
        req = observation_request(sample(), 'Approach')
        for key, value in [('id', 'other'), ('adapter_sha256', 'other'),
                           ('action_representation', 'delta_tcp')]:
            response = reply(req); response[key] = value
            with self.assertRaises(ValueError):
                action_target(response, req, self.current, 'approach_clear')

    def test_inference_age_and_changed_robot_block_before_dispatch(self):
        def stale(req):
            req['source_time'] -= 3
            return reply(req)
        self.policy.predict.side_effect = stale
        with self.assertRaises(TimeoutError):
            self.bridge.step('approach_clear', 'Approach', execute=True)
        self.executor.execute.assert_not_called()
        def changed(req):
            self.current['joints'][0] += .1
            return reply(req)
        self.policy.predict.side_effect = changed
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.bridge.step('approach_clear', 'Approach', execute=True)
        self.executor.execute.assert_not_called()

    def test_nonfinite_or_out_of_range_model_values_are_rejected(self):
        req = observation_request(sample(), 'Approach')
        for value in (-.1, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                action_target(reply(req, value), req, self.current, 'approach_clear')

    def test_gripper_phase_keeps_joints_and_checks_direction(self):
        req = observation_request(sample(), 'Close')
        target = action_target(reply(req, .95), req, self.current, 'close')
        self.assertEqual(target['target_deg_percent'], [0.]*6+[95.])
        with self.assertRaisesRegex(ValueError, 'direction'):
            action_target(reply(req, .95), req, self.current, 'release')

    def test_uncertain_executor_is_not_retried(self):
        self.executor.execute.side_effect = TimeoutError('uncertain wire result')
        with self.assertRaises(TimeoutError):
            self.bridge.step('approach_clear', 'Approach', execute=True)
        self.executor.execute.assert_called_once()
        self.assertEqual(self.observer.get_sample.call_count, 1)

    def test_unarmed_executor_is_not_implicitly_enabled(self):
        self.executor.armed = False
        with self.assertRaisesRegex(RuntimeError, 'already-armed'):
            self.bridge.step('approach_clear', 'Approach', execute=True)
        self.executor.execute.assert_not_called()

    def test_inactive_gripper_is_visible_in_preview_but_blocks_execution(self):
        self.current.update(gripper_active=0, gripper_motiondone=0, gripper_position=0)
        result = self.bridge.step('close', 'Close the gripper')
        self.assertEqual(result['status'], 'preview_rejected_for_execution')
        self.assertFalse(result['gripper_feedback_valid'])
        request = self.policy.predict.call_args.args[0]
        self.assertEqual(request['state_rad_fraction_valid'][-2:], [0., 0.])
        self.assertEqual(result['physical_commands_sent'], 0)
        with self.assertRaisesRegex(ValueError, 'valid measured gripper'):
            self.bridge.step('close', 'Close the gripper', execute=True)
        self.executor.execute.assert_not_called()

    def test_out_of_range_prediction_is_reported_without_clipping_or_execution(self):
        self.policy.predict.side_effect = lambda request: reply(request, 1.03)
        result = self.bridge.step('approach_clear', 'Approach')
        self.assertAlmostEqual(result['raw_target_deg_percent'][-1], 103.)
        self.assertEqual(result['status'], 'preview_rejected_for_execution')
        with self.assertRaisesRegex(ValueError, 'outside0..100'):
            self.bridge.step('approach_clear', 'Approach', execute=True)
        self.executor.execute.assert_not_called()

    def test_tcp_ik_uses_reference_and_checks_fk_without_motion(self):
        pose = [1., 2., 103., 0., 0., 0.]
        rpc = SimpleNamespace(call=Mock(side_effect=[[0, 1, 2, 3, 4, 5, 6], [0, *pose]]))
        result = solve_tcp_target(pose, self.current, rpc)
        self.assertEqual(result['joint_delta_deg'], [1, 2, 3, 4, 5, 6])
        self.assertEqual(rpc.call.call_args_list[0].args, ('GetInverseKinRef', 0, pose, [0.]*6))
        self.assertEqual([x.args[0] for x in rpc.call.call_args_list], ['GetInverseKinRef', 'GetForwardKin'])
        rpc.call.side_effect = [[0, 1, 2, 3, 4, 5, 6], [0, 10, 2, 103, 0, 0, 0]]
        with self.assertRaisesRegex(ValueError, 'disagree'):
            solve_tcp_target(pose, self.current, rpc)


if __name__ == '__main__':
    unittest.main()
