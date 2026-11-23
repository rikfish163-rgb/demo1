"""Offline cup-relative localization, state and read-only IK tests."""
import io
import json
import math
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from lerobot_robot_fairino.cup_relative_runtime import (
    check_cup_close, check_measured_cup_setdown, cup_relative_state, decode_prefix, detect_initial_cup,
)


class FakeRPC:
    def __init__(self, *, joints=None, fk_offset=None, limits=None, ik_code=0):
        self.calls = []
        self.joints = joints
        self.fk_offset = np.zeros(6) if fk_offset is None else np.asarray(fk_offset)
        self.limits = limits if limits is not None else [0] + [-180., 180.] * 6
        self.ik_code = ik_code
        self.pose_by_joints = {}

    def call(self, method, *args):
        self.calls.append((method, args))
        if method == 'GetJointSoftLimitDeg':
            return self.limits
        if method == 'GetInverseKinRef':
            if self.ik_code:
                return [self.ik_code]
            pose, seed = args[1:]
            joints = list(self.joints) if self.joints is not None else [seed[0] + .5, *seed[1:]]
            self.pose_by_joints[tuple(joints)] = np.asarray(pose)
            return [0, *joints]
        if method == 'GetForwardKin':
            pose = self.pose_by_joints[tuple(args[0])] + self.fk_offset
            return [0, *pose.tolist()]
        raise AssertionError(f'unexpected RPC: {method}')


class OfflineTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket.connect', 'socket.socket.connect_ex'):
            guard = patch(name, side_effect=AssertionError('offline test attempted a connection'))
            guard.start()
            self.addCleanup(guard.stop)
        self.initial_cup = {'base_xyz_mm': [100., 200., 0.],
                            'camera_serial': 'CP1E54200015', 'observed_monotonic': 10.}

    def sample(self):
        robot = {'received_monotonic': 12.02, 'tool': 0, 'user': 0,
                 'tcp': [110., 180., 100., 90., -30., -60.],
                 'flange': [110., 180., 100., 90., -30., -60.],
                 'gripper_position': 76, 'gripper_active': 1, 'gripper_fault': 0,
                 'gripper_fault_id': 0, 'gripper_motiondone': 1, 'gripper_speed': 0}
        return {'camera': {'metadata': {'source_timestamp_monotonic': 12.,
                                        'source_camera_serial': 'CP1E54200015'}},
                'robot_state': robot, 'sync_clock': 'host_monotonic'}

    def test_state_uses_paired_measured_tcp_and_preserves_invalid_gripper(self):
        sample = self.sample()
        np.testing.assert_allclose(cup_relative_state(sample, self.initial_cup),
            [.01, -.02, .1, math.pi/2, -math.pi/6, -math.pi/3, .76, 1.])
        sample['robot_state']['gripper_motiondone'] = 0
        result = cup_relative_state(sample, self.initial_cup)
        self.assertEqual(result[-2:], [0., 0.])
        np.testing.assert_allclose(result[:6], [.01, -.02, .1, math.pi/2, -math.pi/6, -math.pi/3])

    def test_state_rejects_unmatched_or_incompatible_frames(self):
        sample = self.sample()
        sample['robot_state']['received_monotonic'] += .2
        with self.assertRaisesRegex(ValueError, 'matched monotonic'):
            cup_relative_state(sample, self.initial_cup)
        sample = self.sample()
        sample['camera']['metadata']['source_camera_serial'] = 'other'
        with self.assertRaisesRegex(ValueError, 'camera or time'):
            cup_relative_state(sample, self.initial_cup)
        sample = self.sample()
        sample['robot_state']['tool'] = 1
        with self.assertRaisesRegex(ValueError, 'tool0/user0'):
            cup_relative_state(sample, self.initial_cup)
        sample = self.sample()
        sample['robot_state']['flange'][0] += 5
        with self.assertRaisesRegex(ValueError, 'match the flange'):
            cup_relative_state(sample, self.initial_cup)

    def test_decode_prefix_uses_base_axes_and_previous_solution_as_seed(self):
        actions = np.zeros((50, 7))
        actions[0] = [.05, -.02, .08, math.pi/2, -.2, .3, .76]
        actions[1] = [.06, -.01, .09, math.pi/2, -.2, .3, 1.02]
        rpc = FakeRPC()
        results = decode_prefix(actions, self.initial_cup, np.deg2rad([10., 0, 0, 0, 0, 0]), 2, rpc=rpc)
        self.assertEqual([method for method, _ in rpc.calls],
                         ['GetJointSoftLimitDeg', 'GetInverseKinRef', 'GetForwardKin',
                          'GetInverseKinRef', 'GetForwardKin'])
        first, second = results
        np.testing.assert_allclose(first['detail']['target_tcp_base_mm_deg'],
                                   [150., 180., 80., 90., math.degrees(-.2), math.degrees(.3)])
        np.testing.assert_allclose(first['detail']['ik_seed_deg'], [10., 0, 0, 0, 0, 0])
        np.testing.assert_allclose(second['detail']['ik_seed_deg'], first['detail']['ik_joint_deg'])
        np.testing.assert_allclose(first['target'], [math.radians(10.5), 0, 0, 0, 0, 0, .76])
        self.assertEqual(second['target'][6], 1.)
        self.assertEqual(second['detail']['raw_model_action'], actions[1].tolist())
        self.assertEqual(first['detail']['cup_base_xyz_mm'], [100., 200., 0.])
        self.assertEqual(first['detail']['fk_position_error_mm'], 0.)

    def test_decode_rejects_large_jump_soft_limits_and_failed_fk(self):
        actions = np.zeros((50, 7))
        for rpc, message in ((FakeRPC(joints=[4., 0, 0, 0, 0, 0]), 'joint jump'),
                             (FakeRPC(joints=[181., 0, 0, 0, 0, 0]), 'soft limits'),
                             (FakeRPC(fk_offset=[1., 0, 0, 0, 0, 0]), 'IK/FK mismatch'),
                             (FakeRPC(fk_offset=[0, 0, 0, 0, .2, 0]), 'IK/FK mismatch'),
                             (FakeRPC(ik_code=1), 'IK failed')):
            with self.subTest(message=message, rpc=rpc), self.assertRaisesRegex((ValueError, RuntimeError), message):
                decode_prefix(actions, self.initial_cup, [0.]*6, 1, rpc=rpc)

    def test_decode_enforces_one_and_half_degree_step_limit(self):
        actions = np.zeros((50, 7))
        boundary = math.nextafter(1.5, 0.)
        result = decode_prefix(actions, self.initial_cup, [0.]*6, 1,
                               rpc=FakeRPC(joints=[boundary, 0, 0, 0, 0, 0]))
        self.assertLessEqual(result[0]['detail']['ik_joint_step_max_deg'], 1.5)
        self.assertAlmostEqual(result[0]['detail']['ik_joint_step_max_deg'], 1.5, places=12)
        with self.assertRaisesRegex(ValueError, 'joint jump'):
            decode_prefix(actions, self.initial_cup, [0.]*6, 1,
                          rpc=FakeRPC(joints=[1.501, 0, 0, 0, 0, 0]))

    def test_decode_rejects_bad_inputs_before_rpc(self):
        actions = np.zeros((50, 7))
        for bad, count in ((actions[:10], 1), (actions, 0), (actions, 51), (actions, True)):
            rpc = FakeRPC()
            with self.subTest(count=count, shape=bad.shape), self.assertRaises(ValueError):
                decode_prefix(bad, self.initial_cup, [0.]*6, count, rpc=rpc)
            self.assertEqual(rpc.calls, [])
        actions[49, 0] = np.nan
        rpc = FakeRPC()
        with self.assertRaisesRegex(ValueError, 'finite 50 by 7'):
            decode_prefix(actions, self.initial_cup, [0.]*6, 1, rpc=rpc)
        self.assertEqual(rpc.calls, [])


class CupGripperTransitionTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket.connect', 'socket.socket.connect_ex'):
            guard = patch(name, side_effect=AssertionError('offline test attempted a connection'))
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def row(gripper, xyz_mm):
        return {'target': [0.] * 6 + [gripper],
                'detail': {'cup_relative_tcp_m_rad': [*(value / 1000 for value in xyz_mm), 0., 0., 0.]}}

    def test_release_at_support_and_boundary_is_allowed(self):
        for xyz in ([159.7, 71.3, 95.8], [174.7, 71.3, 103.8], [159.7, 71.3, 87.8]):
            with self.subTest(xyz=xyz):
                rows = [self.row(.88, xyz)]
                original = deepcopy(rows)
                self.assertIsNone(check_cup_close(rows, gripper_fraction=.76))
                self.assertEqual(rows, original)

    def test_release_away_from_support_is_rejected(self):
        for xyz in ([159.7, 71.3, 103.81], [159.7, 71.3, 87.79],
                    [174.71, 71.3, 95.8], [170.4, 82., 95.8]):
            with self.subTest(xyz=xyz), self.assertRaisesRegex(RuntimeError, 'release.*support'):
                check_cup_close([self.row(.88, xyz)], gripper_fraction=.76)

    def test_observed_grasp_height_rejects_early_open_without_changing_close(self):
        reference = ([.1597, .0713, .1008], [0., 0., 1.], 5.)
        with self.assertRaisesRegex(RuntimeError, '3.70 mm above'):
            check_cup_close([self.row(1., [159.7, 71.3, 99.5])], gripper_fraction=.76,
                            support_reference=reference)
        self.assertIsNone(check_cup_close([self.row(1., [159.7, 71.3, 95.9])], gripper_fraction=.76,
                                           support_reference=reference))
        self.assertIsNone(check_cup_close([self.row(.76, [159.7, 71.3, 99.5])], gripper_fraction=1.,
                                           support_reference=reference))

    def test_setdown_checks_measured_flange_and_contact_before_open(self):
        from lerobot_robot_fairino.stream_runtime import OrderedChunk

        initial_cup = {'base_xyz_mm': [0., -500., 0.]}
        reference = ([.1597, .0713, .1008], [0., 0., 1.], 5.)
        state = {'gripper_active': 1, 'gripper_fault': 0, 'gripper_fault_id': 0, 'gripper_speed': 0, 'gripper_motiondone': 2,
                 'gripper_position': 78, 'flange': [159.7, -428.7, 99.5, 0., 0., 0.], 'tcp_speed': [0., 0.]}
        with self.assertRaisesRegex(RuntimeError, '3.70 mm from'):
            check_measured_cup_setdown(state, initial_cup, reference)
        state['flange'][2] = 95.8
        check_measured_cup_setdown(state, initial_cup, reference)
        state['tcp_speed'][0] = 5.
        with self.assertRaisesRegex(RuntimeError, 'still moving'):
            check_measured_cup_setdown(state, initial_cup, reference)
        state['gripper_motiondone'] = 1
        with self.assertRaisesRegex(RuntimeError, 'contact was lost'):
            check_measured_cup_setdown(state, initial_cup, reference)
        state['tcp_speed'][0] = 0.
        state['gripper_motiondone'] = 2
        chunk = type('ReleaseChunk', (), {'initial_cup': initial_cup, 'support_reference': reference,
                                          'setdown_release_started': False})()
        OrderedChunk.check_release_feedback(chunk, state)
        state['gripper_motiondone'] = 1
        OrderedChunk.check_release_feedback(chunk, state)
        state['flange'][2] += 3.7
        with self.assertRaisesRegex(RuntimeError, '3.70 mm from'):
            OrderedChunk.check_release_feedback(chunk, state)

    def test_no_release_does_not_create_or_change_a_target(self):
        for initial, target in ((1., 1.), (.76, .76)):
            rows = [self.row(target, [159.7, 71.3, 250.])]
            original = deepcopy(rows)
            self.assertIsNone(check_cup_close(rows, gripper_fraction=initial))
            self.assertEqual(rows, original)

    def test_close_and_later_release_use_their_own_limits(self):
        rows = [self.row(.85, [159.7, 71.3, 106.8]),
                self.row(.90, [159.7, 71.3, 104.8])]
        with self.assertRaisesRegex(RuntimeError, 'release.*support'):
            check_cup_close(rows, gripper_fraction=.88)
        with self.assertRaisesRegex(RuntimeError, 'close.*contact'):
            check_cup_close([self.row(.85, [159.7, 71.3, 108.8])], gripper_fraction=.88)
        rows[1] = self.row(.90, [159.7, 71.3, 95.8])
        self.assertIsNone(check_cup_close(rows, gripper_fraction=.88))

    def test_one_point_deadband_cannot_hide_a_gradual_release(self):
        high = [159.7, 71.3, 136.]
        with self.assertRaisesRegex(RuntimeError, 'release.*support'):
            check_cup_close([self.row(.79, high), self.row(.80, high)], gripper_fraction=.78)
        with self.assertRaisesRegex(RuntimeError, 'close.*contact'):
            check_cup_close([self.row(.99, high), self.row(.98, high)], gripper_fraction=1.)


class CupDetectionTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket.connect', 'socket.socket.connect_ex'):
            guard = patch(name, side_effect=AssertionError('offline test attempted a connection'))
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def scene(centers):
        color = np.full((1080, 1920, 3), 180, np.uint8)
        depth = np.full((1080, 1920), 850, np.float32)
        for center in centers:
            cv2.circle(color, center, 52, (60, 60, 60), -1)
            cv2.circle(depth, center, 52, 750, -1)
        intrinsic = {'fx': 1034., 'fy': 1034., 'cx': 960., 'cy': 540.,
                     'width': 1920, 'height': 1080}
        distortion = {name: 0. for name in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')}
        metadata = {'camera': {'serial': 'CP1E54200015'},
                    'capture': {'alignment': 'depth_to_color', 'depth_unit': 'mm',
                                'time_unix': 1., 'received_monotonic': 100.},
                    'color_intrinsic': intrinsic, 'color_distortion': distortion,
                    'color_distortion_model': 'BROWN_CONRADY',
                    'aligned_depth_profile': {'intrinsic': intrinsic, 'distortion': distortion,
                                              'distortion_model': 'BROWN_CONRADY'}}
        transform = np.diag([1., -1., -1., 1.])
        transform[2, 3] = 850.
        calibration = {'base_from_color_camera_mm': transform.tolist(), 'source': 'synthetic',
                       'K': [[1034., 0, 960.], [0, 1034., 540.], [0, 0, 1.]], 'D': [0.]*8}
        return {'color_bgr': color, 'depth_mm': depth, 'metadata': metadata}, calibration

    def test_bridge_frame_detects_one_cup_and_uses_its_table_plane(self):
        frame, calibration = self.scene([(700, 500)])
        packet = io.BytesIO()
        np.savez(packet, color_bgr=frame['color_bgr'], depth_mm=frame['depth_mm'],
                 metadata=np.array(json.dumps(frame['metadata'])))
        with patch('lerobot_robot_fairino.cup_relative_runtime.urllib.request.urlopen',
                   return_value=io.BytesIO(packet.getvalue())) as bridge, \
             patch('lerobot_robot_fairino.cup_relative_runtime.load_readonly_calibration',
                   return_value=calibration):
            cup = detect_initial_cup(now=100.1)
        bridge.assert_called_once_with('http://127.0.0.1:8765/snapshot.npz', timeout=5)
        np.testing.assert_allclose(cup['uv'], [700., 500.], atol=2.)
        self.assertEqual(cup['camera_serial'], 'CP1E54200015')
        self.assertEqual(cup['observed_monotonic'], 100.)
        self.assertEqual(cup['source'], 'calibrated_rim_and_depth_v1')
        base, rim, normal = map(np.asarray, (cup['base_xyz_mm'], cup['rim_xyz_mm'], cup['table_normal']))
        self.assertAlmostEqual(float(normal @ base), cup['table_offset_mm'], places=5)
        np.testing.assert_allclose(rim - base, 110. * normal, atol=1e-6)

    def test_rejects_missing_or_multiple_cups(self):
        for centers in ([], [(700, 500), (1400, 850)]):
            frame, calibration = self.scene(centers)
            with self.subTest(centers=centers), \
                 patch('lerobot_robot_fairino.cup_relative_runtime.load_readonly_calibration', return_value=calibration), \
                 self.assertRaisesRegex(ValueError, 'Expected one detectable cup rim'):
                detect_initial_cup(frame, now=100.1)

    def test_rejects_stale_frame_and_wrong_calibrated_optics(self):
        frame, calibration = self.scene([(700, 500)])
        with self.assertRaisesRegex(ValueError, 'stale'):
            detect_initial_cup(frame, now=103.)
        frame['metadata']['color_intrinsic']['fx'] += 1
        with patch('lerobot_robot_fairino.cup_relative_runtime.load_readonly_calibration', return_value=calibration), \
             self.assertRaisesRegex(ValueError, 'optics differ'):
            detect_initial_cup(frame, now=100.1)


if __name__ == '__main__':
    unittest.main()
