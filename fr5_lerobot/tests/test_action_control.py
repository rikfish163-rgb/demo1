"""Offline motion-gate regressions. RPC/state/camera are in-memory fakes only."""

from copy import deepcopy
from contextlib import nullcontext
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lerobot_robot_fairino.action_control import ControlError, FR5ActionController, MotionLimits
from lerobot_robot_fairino.control_transport import NoRetryTransport


MODULE = "lerobot_robot_fairino.action_control"


def stopped_state():
    state = dict.fromkeys(("main_code", "sub_code", "EmergencyStop", "collisionState",
                           "safety_stop0_state", "safety_stop1_state", "gripper_fault",
                           "tool", "user", "mc_queue_len"), 0)
    state.update(joints=[0.] * 6, tcp=[0., 0., 100., 0., 0., 0.],
                 flange=[0., 0., 100., 0., 0., 0.], joint_speeds_deg_s=[0.] * 6,
                 rbtEnableState=1, program_state=1, robot_state=1, motion_done=1,
                 gripper_active=1, gripper_position=50., gripper_motiondone=0,
                 frame_counter=0)
    return state


class ActionControllerTests(unittest.TestCase):
    def setUp(self):
        # Any accidental real transport construction is an immediate test error.
        tripwire = patch("lerobot_robot_fairino.control_transport.xmlrpc.client.ServerProxy",
                         side_effect=AssertionError("Tests must not contact hardware"))
        tripwire.start()
        self.addCleanup(tripwire.stop)
        self.state = stopped_state()
        self.hooks = {}
        self.calls = []
        self.fk_override = None
        self.ik_override = None
        self.camera = Mock(return_value=True)
        self.stream = Mock()
        self.stream.latest.side_effect = self.read_state
        self.rpc = Mock()
        self.rpc.call.side_effect = self.rpc_call
        self.limits = MotionLimits((-10., -10., 90.), (10., 10., 110.), command_timeout_s=.5)
        self.controller = FR5ActionController(self.stream, self.camera, self.limits,
                                               allow_motion=True, rpc=self.rpc)

    def read_state(self, **kwargs):
        self.state["frame_counter"] += 1
        return deepcopy(self.state)

    def rpc_call(self, method, *args, **kwargs):
        before_send = kwargs.pop("before_send", None)
        with before_send() if before_send is not None else nullcontext():
            self.calls.append((method, deepcopy(args), dict(kwargs)))
        if method in self.hooks:
            return self.hooks[method](*args, **kwargs)
        if method == "GetJointSoftLimitDeg":
            return [0] + [-180., 180.] * 6
        if method == "GetForwardKin":
            pose = self.fk_override if self.fk_override is not None else deepcopy(self.state["flange"])
            if self.fk_override is None:
                pose[0] += 1.
            return [0, *pose]
        if method == "GetInverseKinRef":
            return [0, *(self.ik_override if self.ik_override is not None else [.5, 0., 0., 0., 0., 0.])]
        if method == "MoveL":
            self.state.update(joints=list(args[0]), flange=list(args[1]), tcp=list(args[1]))
            return 0
        if method == "MoveGripper":
            self.state["gripper_position"] = args[1]
            return 0
        if method == "StopMotion":
            return 0
        raise AssertionError(f"Unexpected RPC {method}")

    def arm(self):
        self.controller.arm(operator_confirmed=True)
        self.controller.heartbeat(True)
        self.calls.clear()

    def arm_target(self):
        target = [*self.state["joints"], self.state["gripper_position"]]
        target[0] += .5
        return target

    def methods(self):
        return [item[0] for item in self.calls]

    def assert_no_motion(self):
        self.assertNotIn("MoveL", self.methods())
        self.assertNotIn("MoveGripper", self.methods())

    def test_unarmed_and_preview_mode_never_send_motion(self):
        self.controller.heartbeat(True)
        with self.assertRaisesRegex(ControlError, "not armed"):
            self.controller.execute(self.arm_target())
        self.controller.allow_motion = False
        with self.assertRaisesRegex(ControlError, "Preview mode"):
            self.controller.arm(operator_confirmed=True)
        self.assertEqual(self.calls, [])

    def test_arm_requires_explicit_operator_review(self):
        for value in (False, None, 1):
            with self.subTest(value=value), self.assertRaises(ControlError):
                self.controller.arm(operator_confirmed=value)
        self.assertEqual(self.calls, [])

    def test_deadman_release_and_expired_heartbeat_block_commands(self):
        self.arm()
        self.controller.heartbeat(False)
        with self.assertRaisesRegex(ControlError, "Hold Space"):
            self.controller.execute(self.arm_target())
        self.controller.heartbeat(True)
        self.controller._heartbeat -= 2.
        with self.assertRaisesRegex(ControlError, "Hold Space"):
            self.controller.execute(self.arm_target())
        self.assert_no_motion()

    def test_camera_stale_initially_and_after_ik_blocks_motion(self):
        self.arm()
        self.camera.return_value = False
        with self.assertRaisesRegex(ControlError, "Camera"):
            self.controller.execute(self.arm_target())
        self.camera.side_effect = [True, False]
        with self.assertRaisesRegex(ControlError, "Camera"):
            self.controller.execute(self.arm_target())
        self.assert_no_motion()

    def test_faults_moving_state_or_nonzero_tool_block_commands(self):
        self.arm()
        for key, bad in (("EmergencyStop", 1), ("main_code", 8), ("gripper_fault", 2),
                         ("rbtEnableState", 0), ("tool", 1), ("user", 1),
                         ("motion_done", 0), ("mc_queue_len", 1)):
            original = self.state[key]
            self.state[key] = bad
            with self.subTest(key=key), self.assertRaises(ControlError):
                self.controller.execute(self.arm_target())
            self.state[key] = original
        self.assert_no_motion()

    def test_arm_and_gripper_cannot_move_together(self):
        self.arm()
        target = self.arm_target()
        target[6] += 3
        with self.assertRaisesRegex(ControlError, "not both"):
            self.controller.execute(target)
        self.assert_no_motion()

    def test_tool_zero_with_rotation_offset_rejected_before_movement(self):
        self.arm()
        self.state["tcp"][3] = 5.
        with self.assertRaisesRegex(ControlError, "tool0 does not match"):
            self.controller.execute(self.arm_target())
        self.assert_no_motion()

    def test_joint_step_and_joint_margin_rejected(self):
        self.arm()
        target = self.arm_target()
        target[0] = 2.1
        with self.assertRaisesRegex(ControlError, "Joint step"):
            self.controller.execute(target)
        self.state["joints"][0] = 176.
        target = self.arm_target()
        target[0] = 178.
        with self.assertRaisesRegex(ControlError, "joint limit margin"):
            self.controller.execute(target)
        self.assert_no_motion()

    def test_target_workspace_and_cartesian_step_rejected(self):
        self.arm()
        self.state["tcp"][0] = self.state["flange"][0] = 9.5
        with self.assertRaisesRegex(ControlError, "outside configured workspace"):
            self.controller.execute(self.arm_target())
        self.state["tcp"][0] = self.state["flange"][0] = 0.
        self.fk_override = [3., 0., 100., 0., 0., 0.]
        with self.assertRaisesRegex(ControlError, "translation exceeds"):
            self.controller.execute(self.arm_target())
        self.fk_override = [1., 0., 100., 2., 0., 0.]
        with self.assertRaisesRegex(ControlError, "rotation exceeds"):
            self.controller.execute(self.arm_target())
        self.assert_no_motion()

    def test_ik_branch_jump_and_invalid_reply_are_rejected(self):
        self.arm()
        self.ik_override = [3., 0., 0., 0., 0., 0.]
        with self.assertRaisesRegex(ControlError, "IK branch"):
            self.controller.execute(self.arm_target())
        self.hooks["GetInverseKinRef"] = lambda *a, **k: [7]
        with self.assertRaisesRegex(ControlError, "path IK failed"):
            self.controller.execute(self.arm_target())
        self.assert_no_motion()

    def test_valid_arm_step_uses_one_move_and_three_distinct_feedback_frames(self):
        self.arm()
        result = self.controller.execute(self.arm_target())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.methods().count("MoveL"), 1)
        self.assertNotIn("MoveGripper", self.methods())
        self.assertGreaterEqual(result["final_state"]["frame_counter"], 6)
        self.assertFalse(self.controller.busy)
        self.assertTrue(self.controller.armed)

    def test_gripper_step_has_no_arm_motion_and_is_bounded(self):
        self.arm()
        target = [*self.state["joints"], 56.]
        with self.assertRaisesRegex(ControlError, "Gripper step"):
            self.controller.execute(target)
        self.assert_no_motion()
        target[6] = 55.
        result = self.controller.execute(target)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.methods().count("MoveGripper"), 1)
        self.assertNotIn("MoveL", self.methods())

    def test_unknown_motion_outcome_is_not_retried_and_disarms(self):
        self.arm()

        def unknown(*args, **kwargs):
            raise TimeoutError("response lost after possible acceptance")

        self.hooks["MoveL"] = unknown
        with self.assertRaisesRegex(TimeoutError, "response lost"):
            self.controller.execute(self.arm_target())
        self.assertEqual(self.methods().count("MoveL"), 1)
        self.assertEqual(self.methods().count("StopMotion"), 1)
        self.assertFalse(self.controller.armed)
        self.assertFalse(self.controller.busy)
        self.assertIn("outcome uncertain", self.controller.error)
        with self.assertRaisesRegex(ControlError, "not armed"):
            self.controller.execute(self.arm_target())
        self.assertEqual(self.methods().count("MoveL"), 1)

    def test_nonzero_move_result_stops_without_retry(self):
        self.arm()
        self.hooks["MoveL"] = lambda *a, **k: 42
        with self.assertRaisesRegex(ControlError, "returned 42"):
            self.controller.execute(self.arm_target())
        self.assertEqual(self.methods().count("MoveL"), 1)
        self.assertEqual(self.methods().count("StopMotion"), 1)

    def test_stop_during_fk_preparation_prevents_submission(self):
        self.arm()

        def stop_before_preparation_returns(*args, **kwargs):
            self.controller.stop("stopped while preparing")
            return [0, 1., 0., 100., 0., 0., 0.]

        self.hooks["GetForwardKin"] = stop_before_preparation_returns
        with self.assertRaises(ControlError):
            self.controller.execute(self.arm_target())
        self.assert_no_motion()
        self.assertFalse(self.controller.armed)

    def test_external_gripper_change_during_preparation_is_rejected(self):
        self.arm()
        reads = 0

        def changed_gripper(**kwargs):
            nonlocal reads
            reads += 1
            if reads == 2:
                self.state["gripper_position"] = 80
            return self.read_state(**kwargs)

        self.stream.latest.side_effect = changed_gripper
        with self.assertRaisesRegex(ControlError, "changed"):
            self.controller.execute([*self.state["joints"], 55])
        self.assert_no_motion()

    def test_stop_before_transport_write_cancels_actual_endheaders(self):
        self.arm()
        ready, release = threading.Event(), threading.Event()
        connection = Mock()
        errors = []

        def wire_rpc(method, *args, **kwargs):
            if method != "MoveL":
                return self.rpc_call(method, *args, **kwargs)
            ready.set()
            if not release.wait(2):
                raise AssertionError("Test failed to release pending request")
            transport = NoRetryTransport(kwargs["timeout_s"], before_send=kwargs["before_send"])
            # Exercise real XML-RPC content writing, not a mocked controller call.
            transport.send_content(connection, b"synthetic MoveL body")
            return 0

        self.rpc.call.side_effect = wire_rpc

        def worker():
            try:
                self.controller.execute(self.arm_target())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(ready.wait(1))
            self.controller.stop("cancel before HTTP write")
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        connection.endheaders.assert_not_called()
        self.assertTrue(errors)
        self.assertFalse(self.controller.recording_valid)
        self.assertIsNone(self.controller.command_at(time.time() + 1))
        self.assert_no_motion()

    def test_stop_is_independent_while_move_rpc_is_blocked_and_second_move_rejected(self):
        self.arm()
        entered, release, stop_called = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def blocked_move(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Test failed to release fake blocking RPC")
            return 0

        def stop(*args, **kwargs):
            stop_called.set()
            return 0

        self.hooks.update(MoveL=blocked_move, StopMotion=stop)

        def worker():
            try:
                self.controller.execute(self.arm_target())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(ControlError, "in flight"):
                self.controller.execute(self.arm_target())
            self.controller.stop("operator released")
            self.assertTrue(stop_called.is_set(), "StopMotion must not wait for MoveL response")
            self.assertFalse(release.is_set())
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertFalse(self.controller.armed)
        self.assertFalse(self.controller.busy)
        self.assertEqual(self.methods().count("MoveL"), 1)

    def test_stop_during_arm_query_must_not_rearm_controller(self):
        entered, release = threading.Event(), threading.Event()
        errors = []

        def blocked_limits(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Test failed to release fake query")
            return [0] + [-180., 180.] * 6

        self.hooks["GetJointSoftLimitDeg"] = blocked_limits

        def worker():
            try:
                self.controller.arm(operator_confirmed=True)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.controller.stop("cancel arming")
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.controller.armed, "Stop during arm preparation must remain latched")
        self.assertEqual(self.controller.error, "cancel arming")
        self.assertTrue(errors, "Cancelled arming must report failure to its caller")
        self.assert_no_motion()

    def test_journal_selects_hold_or_command_at_camera_source_time(self):
        with patch(f"{MODULE}.time.time", side_effect=[1000., 1001.]):
            self.arm()
            self.controller.execute(self.arm_target())
        self.assertIsNone(self.controller.command_at(999.999))
        hold = self.controller.command_at(1000.999)
        self.assertEqual(hold["kind"], "hold")
        self.assertFalse(hold["executed"])
        self.assertEqual(hold["issued_unix"], 1000.)
        command = self.controller.command_at(1001.)
        self.assertEqual(command["kind"], "arm")
        self.assertTrue(command["executed"])
        self.assertEqual(command["issued_unix"], 1001.)
        self.assertGreater(command["sequence"], hold["sequence"])
        command["target"][0] = 999
        self.assertEqual(self.controller.command_at(1001.)["target"][0], .5)

    def test_stop_failure_does_not_restore_arm_and_is_reported(self):
        self.arm()

        def unknown_stop(*args, **kwargs):
            raise OSError("stop connection lost")

        self.hooks["StopMotion"] = unknown_stop
        self.controller.stop("operator stop")
        self.assertFalse(self.controller.armed)
        self.assertIn("StopMotion unconfirmed", self.controller.error)
        self.assertIn("stop connection lost", self.controller.error)
        self.assertEqual(self.methods(), ["StopMotion"])

    def test_repeated_stop_is_deduplicated_and_invalidates_future_labels(self):
        self.arm()
        held_before_stop = self.controller.command_at(time.time())
        self.controller.stop("operator stop")
        for _ in range(5):
            self.controller.stop("operator stop again")
        self.assertEqual(self.methods(), ["StopMotion"])
        self.assertFalse(self.controller.recording_valid)
        self.assertIsNone(self.controller.command_at(time.time() + .01))
        self.assertEqual(self.controller.command_at(held_before_stop["issued_unix"]), held_before_stop)

    def test_failed_stop_can_be_retried_once_but_not_indefinitely(self):
        self.arm()
        self.controller.busy = True  # A request is still outstanding.
        self.hooks["StopMotion"] = lambda *a, **k: 42
        self.controller.stop("operator stop")
        for _ in range(5):
            self.controller.stop("outcome uncertain", retry_if_unconfirmed=True)
        self.assertEqual(self.methods().count("StopMotion"), 2)
        self.assertFalse(self.controller.recording_valid)

    def test_watchdog_released_deadman_stops_busy_motion(self):
        self.arm()
        self.controller.busy = True
        self.controller.heartbeat(False)
        self.controller.watchdog()
        self.assertFalse(self.controller.armed)
        self.assertEqual(self.methods(), ["StopMotion"])

    def test_invalid_targets_and_limits_do_not_reach_rpc(self):
        self.arm()
        for target in ([0] * 6, [float("nan")] + [0] * 6, [0] * 6 + [101]):
            with self.subTest(target=target), self.assertRaises(ControlError):
                self.controller.execute(target)
        self.assertEqual(self.calls, [])
        for kwargs in ({"translation_step_mm": 3}, {"speed_percent": 6},
                       {"gripper_force_percent": 100}, {"rotation_step_deg": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionLimits((-10., -10., 90.), (10., 10., 110.), **kwargs)


if __name__ == "__main__":
    unittest.main()
