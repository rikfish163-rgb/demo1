"""Offline executor checks: mock state and RPC only, no robot connections."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fr5_jev.teacher_targets import TeacherSession, check_state, validate_scope


def scope():
    return {"robot_ip": "192.168.58.2", "max_lift_mm": 10, "closing_targets": [90,80,76],
            "closing_speed": 5, "force": 10, "move_speed": 1, "return_to_original_support": True,
            "only_prealigned_instance": True, "reviewed_at_unix": time.time(),
            "same_cup_and_contact_pose": True, "both_fingers_outside_cup": True,
            "vertical_swept_path_clear": True, "no_visible_hands": True,
            "initial_tcp": [0,0,100,0,0,0], "authorization_source": "SYNTHETIC TEST ONLY",
            "evidence": ["SYNTHETIC TEST ONLY"]}


class FakeRobot:
    def __init__(self):
        self.s = dict(program_state=1, robot_state=1, robot_mode=1, main_code=0, sub_code=0,
                      EmergencyStop=0, collisionState=0, safety_stop0_state=0, safety_stop1_state=0,
                      rbtEnableState=1, motion_done=1, mc_queue_len=0, tool=0, user=0,
                      gripper_active=1, gripper_fault=0, gripper_fault_id=0, gripper_motiondone=2,
                      gripper_position=76, gripper_speed=0, gripper_current=0, frame_counter=0,
                      tcp=[0.,0.,100.,0.,0.,0.], flange=[0.,0.,100.,0.,0.,0.],
                      joints=[0.,0.,0.,0.,90.,0.], joint_speeds_deg_s=[0.] * 6, tcp_speed=[0,0])
        self.stream = SimpleNamespace(latest=self.latest)
        self.busy_reads = 0

    def latest(self, **kwargs):
        self.s["frame_counter"] += 1
        self.s["timestamp"] = time.time()
        if self.busy_reads:
            self.busy_reads -= 1
            s = deepcopy(self.s)
            s.update(gripper_position=0, gripper_motiondone=0, gripper_speed=0, gripper_current=0)
            return s
        return deepcopy(self.s)


class FakeRPC:
    def __init__(self, robot, failure=None):
        self.robot, self.failure = robot, failure
        self.calls = []

    def call(self, method, *args, before_send=None, **kwargs):
        with before_send() if before_send else nullcontext():
            self.calls.append((method, args))
            if method == "GetInverseKinRef":
                pose = args[1]
                return [0,0,0,(pose[2]-100)/100,0,90,0]
            if method == "GetForwardKin":
                return [0,0,0,100+args[0][2]*100,0,0,0]
            if method == "MoveL":
                self.robot.s.update(joints=list(args[0]), tcp=list(args[1]), flange=list(args[1]))
                if self.failure == "lost":
                    self.robot.s["gripper_motiondone"] = 3
                if self.failure == "direction":
                    self.robot.s["tcp"][2] = self.robot.s["flange"][2] = 99.9
                if self.failure == "timeout":
                    raise TimeoutError("ambiguous fixture")
            if method == "MoveGripper":
                self.robot.s.update(gripper_position=args[1], gripper_motiondone=2 if args[1] == 76 else 1)
                self.robot.busy_reads = 15
            return 0


class FakeStorage:
    """Explicit fake for executor tests; performs no network or device I/O."""
    def __init__(self):
        self.syncs = []

    def sync_files(self, *paths):
        self.syncs.append(tuple(Path(path).read_bytes() for path in paths))


class TeacherTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "scope.json").write_text(json.dumps(scope()))
        self.robot = FakeRobot()
        self.rpc = FakeRPC(self.robot)
        self.session = TeacherSession(self.root, robot=self.robot, rpc=self.rpc)
        self.session.storage = FakeStorage()
        self.session.capture = SimpleNamespace(assert_healthy=Mock())
        self.session.limits = np.array([[-180,180]] * 6)
        self.session.holding_position = 76
        self.session.append([*self.robot.s["joints"],76], "hold", False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_deferred_media_still_commits_raw_stream_before_robot_is_released(self):
        self.session.capture = Mock()
        self.session.capture.finish.return_value = {'capture_complete':True}
        self.session.storage = Mock(codec_sha256='codec', protocol_sha256='protocol')
        self.session.storage.frame_spool.close.return_value = {'accepted_frames':3}
        self.session.defer_materialization = True
        result = self.session.finish('pending_review')
        self.assertEqual(result['outcome'], 'pending_review')
        self.session.storage.frame_spool.close.assert_called_once()
        self.session.storage.sync_tree.assert_called_once()
        self.session.storage.materialize.assert_not_called()
        self.session.storage.release_checkpoint_caches.assert_not_called()
        self.assertTrue((self.root/'stream-closed.json').is_file())

    def test_exact_targets_from_verified_ik_are_sent_once_and_recorded(self):
        final = self.session.height(1)
        moves = [call for call in self.rpc.calls if call[0] == "MoveL"]
        self.assertEqual(len(moves),1)
        self.assertEqual(self.session.journal[-1]["target"][:6], moves[0][1][0])
        self.assertEqual(self.session.journal[-1]["kind"], "arm")
        self.assertTrue(self.session.journal[-1]["executed"])
        self.assertAlmostEqual(final["tcp"][2],101)
        with self.assertRaises(ValueError):
            self.session.height(1)
        self.assertEqual(len([c for c in self.rpc.calls if c[0] == "MoveL"]),1)

    def test_near_session_expiry_refuses_before_gripper_dispatch(self):
        self.session.returned=True
        self.session.start_mono-=885
        with self.assertRaisesRegex(TimeoutError,'Insufficient session time'):
            self.session.release()
        self.assertFalse(self.rpc.calls)
        self.assertEqual(len(self.session.journal),1)
        data=scope();data['max_session_s']=1800
        (self.root/'scope.json').write_text(json.dumps(data))
        fresh=TeacherSession(self.root,robot=self.robot,rpc=self.rpc)
        self.assertEqual(fresh.max_session_s,1800)
        data['max_session_s']=3600
        (self.root/'scope.json').write_text(json.dumps(data))
        with self.assertRaises(ValueError):TeacherSession(self.root,robot=self.robot,rpc=self.rpc)

    def test_decision_transport_failure_is_journaled_without_motion_or_retry(self):
        from fr5_jev.decision import DecisionError
        self.robot.s.update(gripper_position=100,gripper_motiondone=1)
        self.robot.last_sample={'camera':{'metadata':{'source_camera_serial':'CP1E54200015','source_timestamp_unix':time.time()}}}
        self.session.writer=SimpleNamespace(task='SYNTHETIC ONLY')
        with patch('fr5_jev.decision.call_jev',side_effect=DecisionError('synthetic timeout')) as call, \
             patch('fr5_jev.memory.retrieve',return_value=[]):
            result=self.session.decide('close_gripper','decision.json')
        call.assert_called_once()
        self.assertEqual(result['route'],'needs_llm')
        self.assertFalse(result['physical_execution'])
        self.assertFalse(self.rpc.calls)
        self.assertIsNone(self.session.failed)
        record=json.loads((self.root/'decision.json').read_text())
        self.assertIn('synthetic timeout',record['decision_error'])
        with self.assertRaises(RuntimeError):self.session.review_phase('close','decision.json')

    def test_contact_loss_wrong_direction_and_ambiguous_reply_stop_without_retry(self):
        for failure in ("lost", "direction", "timeout"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                (root/'scope.json').write_text(json.dumps(scope()))
                robot=FakeRobot(); rpc=FakeRPC(robot,failure)
                s=TeacherSession(root,robot=robot,rpc=rpc)
                s.storage=FakeStorage()
                s.limits=np.array([[-180,180]]*6)
                s.holding_position=76
                s.append([*robot.s['joints'],76],'hold',False)
                with self.assertRaises(RuntimeError):
                    s.height(1)
                self.assertEqual(sum(c[0]=='MoveL' for c in rpc.calls),1)
                self.assertEqual(sum(c[0]=='StopMotion' for c in rpc.calls),1)
                self.assertTrue((root/'STOP').exists())

    def test_contact_and_bounds_required_before_any_dispatch(self):
        for height in (-1,11,True,4):
            with self.subTest(height=height), self.assertRaises(ValueError):
                self.session.plan_height(height)
        self.robot.s['gripper_motiondone']=3
        with self.assertRaises(RuntimeError):
            self.session.height(1)
        self.assertFalse(any(c[0]=='MoveL' for c in self.rpc.calls))
        with self.assertRaises(RuntimeError):
            self.session.release()

    def test_invalid_scope_and_stale_review_never_connect(self):
        for changes in ({'force':11},{'max_lift_mm':20},{'reviewed_at_unix':time.time()-301},
                        {'both_fingers_outside_cup':False}):
            data=scope();data.update(changes)
            with self.assertRaises(ValueError):
                validate_scope(data)

    def test_busy_zero_gripper_feedback_does_not_become_completion_or_an_arm_move(self):
        self.robot.s.update(gripper_position=100, gripper_motiondone=1)
        self.session.target[-1]=100
        self.session.holding_position=None
        result=self.session.send('MoveGripper',(1,90,5,10,10000,1,0,0.,0,0),
                                 [*self.robot.s['joints'],90],kind='gripper')
        self.assertEqual(result['gripper_position'],90)
        self.assertEqual(self.session.journal[-1]['target'][-1],90)
        self.assertEqual([c[0] for c in self.rpc.calls],['MoveGripper'])

    def test_recording_failure_prevents_command(self):
        self.session.capture.assert_healthy.side_effect=RuntimeError('writer failed')
        with self.assertRaisesRegex(RuntimeError,'writer failed'):
            self.session.height(1)
        self.assertFalse(self.rpc.calls)

    def test_pre_dispatch_storage_latency_does_not_consume_gripper_feedback_grace(self):
        self.robot.s.update(gripper_position=100, gripper_motiondone=1)
        self.session.target[-1]=100
        self.session.holding_position=None
        original_sync=self.session.storage.sync_files
        first=True
        def delayed_sync(*paths):
            nonlocal first
            if first:
                first=False
                time.sleep(2.05)
            return original_sync(*paths)
        self.session.storage.sync_files=delayed_sync
        result=self.session.send('MoveGripper',(1,90,5,10,10000,1,0,0.,0,0),
                                 [*self.robot.s['joints'],90],kind='gripper')
        self.assertEqual(result['gripper_position'],90)
        self.assertEqual([c[0] for c in self.rpc.calls],['MoveGripper'])

    def test_persistent_busy_feedback_still_fails_after_actual_dispatch(self):
        self.robot.s.update(gripper_position=100, gripper_motiondone=1)
        self.session.target[-1]=100
        self.session.holding_position=None
        original_call=self.rpc.call
        def busy_call(method,*args,**kwargs):
            result=original_call(method,*args,**kwargs)
            if method=='MoveGripper':
                self.robot.busy_reads=10000
            return result
        self.rpc.call=busy_call
        with self.assertRaisesRegex(RuntimeError,'Invalid gripper feedback persisted'):
            self.session.send('MoveGripper',(1,90,5,10,10000,1,0,0.,0,0),
                              [*self.robot.s['joints'],90],kind='gripper')
        self.assertEqual([c[0] for c in self.rpc.calls],['MoveGripper'])
        self.assertTrue((self.root/'STOP').exists())

    def test_stationary_gripper_pose_monitor_is_separate_from_downward_motion(self):
        initial = np.array(self.robot.s['tcp'])
        s = deepcopy(self.robot.s)
        s['tcp'][2] -= .07
        s['flange'][2] -= .07
        # A gripper reaction within the existing0.15mm stationary-pose bound
        # is not permission to command an arm trajectory below the support.
        check_state(s, initial, gripper_reference=initial)
        with self.assertRaisesRegex(RuntimeError, 'height bounds'):
            check_state(s, initial, moving=True)
        s['tcp'][2] = s['flange'][2] = initial[2] - .16
        with self.assertRaisesRegex(RuntimeError, 'stationary gripper pose'):
            check_state(s, initial, gripper_reference=initial)
        s['collisionState'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Fault'):
            check_state(s, initial, gripper_reference=initial)

    def test_gripper_velocity_transient_is_bounded_and_debounced(self):
        ref = list(self.robot.s['tcp'])
        self.robot.s['joint_speeds_deg_s'][4] = -.10877552599026785
        # Post-motion visual holds use the same bounded debounce.
        self.session.state()
        self.session.state(gripper_reference=ref)
        self.robot.s['joint_speeds_deg_s'][4] = 0
        self.session.state(gripper_reference=ref)
        self.assertIsNone(self.session.gripper_velocity_since)
        self.robot.s['joint_speeds_deg_s'][4] = .11
        self.session.gripper_velocity_since = time.time() - .11
        with self.assertRaisesRegex(RuntimeError, 'Sustained joint velocity'):
            self.session.state(gripper_reference=ref)
        self.robot.s['joint_speeds_deg_s'][4] = .201
        with self.assertRaisesRegex(RuntimeError, 'velocity spike'):
            self.session.state(gripper_reference=ref)
        self.robot.s['joint_speeds_deg_s'][4] = .11
        self.robot.s['joints'][0] += .051
        with self.assertRaisesRegex(RuntimeError, 'joint displacement'):
            self.session.state(gripper_reference=ref)

    def test_stationary_hold_debounce_preserves_immediate_motion_and_fault_checks(self):
        self.robot.s['joint_speeds_deg_s'][4] = .11
        self.session.state(held=True)
        self.session.gripper_velocity_since = time.time() - .11
        with self.assertRaisesRegex(RuntimeError, 'Sustained joint velocity'):
            self.session.state(held=True)
        self.robot.s['joint_speeds_deg_s'][4] = .201
        with self.assertRaisesRegex(RuntimeError, 'velocity spike'):
            self.session.state(held=True)
        self.robot.s['joint_speeds_deg_s'][4] = .11
        self.session.gripper_velocity_since = None
        self.robot.s['joints'][0] += .051
        with self.assertRaisesRegex(RuntimeError, 'Unexpected joint displacement'):
            self.session.state(held=True)
        self.robot.s['joints'][0] -= .051
        self.robot.s['collisionState'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Fault'):
            self.session.state(held=True)


if __name__ == '__main__':
    unittest.main()
