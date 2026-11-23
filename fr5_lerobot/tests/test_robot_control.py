"""Offline control-adapter lifecycle tests; all transports are replaced by mocks."""

from copy import deepcopy
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_fairino import FairinoFR5Control, FairinoFR5ControlConfig, MotionLimits
from lerobot_robot_fairino.action_control import ControlError
from lerobot_robot_fairino.robot_fairino import STATE_NAMES, STATUS_NAMES


class RobotControlTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.stopped = threading.Event()
        self.rpc = Mock()
        self.rpc.call.side_effect = self.rpc_call
        self.stream = Mock()
        self.state = dict.fromkeys(STATUS_NAMES, 0)
        self.state.update(joints=[0.] * 6, gripper_position=50.,
                          tcp=[0., 0., 100., 0., 0., 0.], flange=[0., 0., 100., 0., 0., 0.],
                          joint_speeds_deg_s=[0.] * 6, timestamp=time.time(), frame_counter=1,
                          rbtEnableState=1, program_state=1, robot_state=1,
                          motion_done=1, gripper_active=1, gripper_motiondone=1)
        self.stream.latest.return_value = self.state
        self.stream.nearest.return_value = self.state
        self.camera = Mock()
        for name, replacement in (("lerobot_robot_fairino.robot_fairino.FR5StateStream", self.stream),
                                  ("lerobot_robot_fairino.robot_fairino.GeminiBridgeCamera", self.camera),
                                  ("lerobot_robot_fairino.action_control.FR5RPC", self.rpc)):
            patcher = patch(name, return_value=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = FairinoFR5ControlConfig(
            limits=MotionLimits((-10., -10., 90.), (10., 10., 110.)), allow_motion=True)
        self.robot = FairinoFR5Control(self.config)

    def tearDown(self):
        self.robot.controller.busy = False
        self.robot.disconnect()

    def rpc_call(self, method, *args, **kwargs):
        self.calls.append(method)
        if method == "GetJointSoftLimitDeg":
            return [0] + [-180., 180.] * 6
        if method == "StopMotion":
            self.stopped.set()
            return 0
        raise AssertionError(f"Unexpected RPC: {method}")

    def fresh_sample(self):
        now = time.time()
        self.state["timestamp"] = now
        bundle = {"color_rgb": np.zeros((360, 640, 3), np.uint8),
                  "depth_mm": np.ones((360, 640), np.float32),
                  "metadata": {"source_timestamp_unix": now,
                               "source_camera_serial": self.config.camera.expected_serial,
                               "received_unix": now, "received_monotonic": time.monotonic()}}
        self.camera.read_bundle.return_value = bundle
        return self.robot.get_sample()

    def connect_and_arm(self):
        self.robot.connect()
        self.fresh_sample()
        self.robot.arm(operator_confirmed=True)
        self.robot.heartbeat(True)

    def test_factory_and_default_config_never_connect_or_arm(self):
        config = FairinoFR5ControlConfig(limits=deepcopy(self.config.limits))
        robot = make_robot_from_config(config)
        self.assertIsInstance(robot, FairinoFR5Control)
        self.assertEqual(config.type, "fairino_fr5_control")
        self.assertFalse(config.allow_motion)
        self.assertFalse(robot.is_connected)
        self.stream.connect.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertEqual(set(robot.action_features), set(STATE_NAMES))
        self.robot = robot
        robot.connect()
        with self.assertRaisesRegex(ControlError, "allow_motion=False"):
            robot.arm(operator_confirmed=True)
        with self.assertRaisesRegex(ControlError, "allow_motion=False"):
            robot.send_action(dict(zip(STATE_NAMES, [0.] * 6 + [50.])))
        self.assertEqual(self.calls, [])

    def test_connect_observe_and_unarmed_disconnect_send_no_rpc_or_action_labels(self):
        self.robot.connect(calibrate=True)
        thread = self.robot._watchdog_thread
        self.assertTrue(thread.is_alive())
        sample = self.fresh_sample()
        self.assertNotIn("action", sample)
        self.assertIsNone(self.robot.controller.command_at(time.time()))
        self.assertFalse(self.robot.controller.armed)
        self.robot.disconnect()
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.robot.is_connected)
        self.assertEqual(self.calls, [])

    def test_arm_needs_explicit_review_and_current_camera_sample(self):
        self.robot.connect()
        with self.assertRaises(ControlError):
            self.robot.arm(operator_confirmed=True)
        self.fresh_sample()
        with self.assertRaises(ControlError):
            self.robot.arm()
        self.assertEqual(self.calls, [])
        self.robot.arm(operator_confirmed=True)
        self.assertEqual(self.calls, ["GetJointSoftLimitDeg"])
        self.assertTrue(self.robot.controller.armed)
        self.assertFalse(self.robot.controller.deadman)

    def test_camera_freshness_uses_source_and_monotonic_receipt_age(self):
        self.robot.connect()
        sample = self.fresh_sample()
        self.assertTrue(self.robot.camera_fresh())
        meta = sample["camera"]["metadata"]
        meta["source_timestamp_unix"] -= 2
        self.assertFalse(self.robot.camera_fresh())
        meta = self.fresh_sample()["camera"]["metadata"]
        meta["received_monotonic"] -= 2
        self.assertFalse(self.robot.camera_fresh())
        meta = self.fresh_sample()["camera"]["metadata"]
        meta["source_timestamp_unix"] = float("nan")
        self.assertFalse(self.robot.camera_fresh())
        meta = self.fresh_sample()["camera"]["metadata"]
        meta["source_camera_serial"] = "WRONG_CAMERA"
        self.assertFalse(self.robot.camera_fresh())
        self.assertEqual(self.calls, [])

    def test_send_action_delegates_exact_targets_and_does_not_forge_journal(self):
        self.connect_and_arm()
        target = dict(zip(STATE_NAMES, [.5, 0., 0., 0., 0., 0., 50.]))
        with patch.object(self.robot.controller, "execute", return_value={"status": "completed"}) as execute:
            returned = self.robot.send_action(target)
        execute.assert_called_once_with(list(target.values()))
        self.assertEqual(returned, target)
        self.assertIsNot(returned, target)
        self.assertEqual(self.robot.last_action_result["status"], "completed")
        self.assertEqual(self.robot.controller.command_at(time.time())["kind"], "hold")
        # A preset sitting in memory does not execute or generate a command.
        self.assertEqual(self.calls, ["GetJointSoftLimitDeg"])

    def test_hold_noop_does_not_append_executed_action(self):
        self.connect_and_arm()
        before = self.robot.controller.command_at(time.time())
        target = dict(zip(STATE_NAMES, [0.] * 6 + [50.]))
        self.assertEqual(self.robot.send_action(target), target)
        self.assertEqual(self.robot.last_action_result["status"], "hold_no_command")
        self.assertEqual(self.robot.controller.command_at(time.time()), before)
        self.assertEqual(self.calls, ["GetJointSoftLimitDeg"])

    def test_invalid_action_schema_never_reaches_controller(self):
        self.robot.connect()
        target = dict(zip(STATE_NAMES, [0.] * 6 + [50.]))
        invalid = [list(target.values()), {**target, "extra": 1},
                   {k: v for k, v in target.items() if k != STATE_NAMES[0]},
                   {**target, STATE_NAMES[0]: float("inf")},
                   {**target, STATE_NAMES[0]: True}, {**target, STATE_NAMES[0]: "1"}]
        with patch.object(self.robot.controller, "execute") as execute:
            for action in invalid:
                with self.subTest(action=action), self.assertRaises(ValueError):
                    self.robot.send_action(action)
        execute.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_independent_watchdog_stops_busy_expired_heartbeat(self):
        self.connect_and_arm()
        self.robot.controller.busy = True
        self.robot.controller._heartbeat -= 2
        self.assertTrue(self.stopped.wait(1), "Independent watchdog must run without caller polling")
        self.assertFalse(self.robot.controller.armed)
        self.assertIn("StopMotion", self.calls)
        self.robot.controller.busy = False

    def test_watchdog_exception_latches_and_blocks_future_actions(self):
        self.connect_and_arm()
        with patch.object(self.robot.controller, "watchdog", side_effect=RuntimeError("watchdog fixture")):
            self.assertTrue(self.stopped.wait(1))
        self.assertIn("watchdog fixture", self.robot.watchdog_error)
        with self.assertRaisesRegex(ControlError, "watchdog"):
            self.robot.send_action(dict(zip(STATE_NAMES, [0.] * 6 + [50.])))
        self.assertFalse(self.robot.controller.armed)

    def test_active_disconnect_stops_before_closing_observations(self):
        self.connect_and_arm()
        events = []
        original_call = self.rpc.call.side_effect
        def call(method, *args, **kwargs):
            events.append(method)
            return original_call(method, *args, **kwargs)
        self.rpc.call.side_effect = call
        self.camera.disconnect.side_effect = lambda: events.append("camera_disconnect")
        self.stream.disconnect.side_effect = lambda: events.append("stream_disconnect")
        self.robot.disconnect()
        self.assertEqual(events, ["StopMotion", "camera_disconnect", "stream_disconnect"])
        self.assertFalse(self.robot._watchdog_thread.is_alive())
        self.assertFalse(self.robot.is_connected)

    def test_stop_while_arm_is_pending_cannot_leave_adapter_armed(self):
        self.robot.connect()
        self.fresh_sample()
        def finish_late_arm(**kwargs):
            self.robot.stop("operator cancelled pending arm")
            self.robot.controller.armed = True
        with patch.object(self.robot.controller, "arm", side_effect=finish_late_arm):
            with self.assertRaisesRegex(ControlError, "Arming cancelled"):
                self.robot.arm(operator_confirmed=True)
        self.assertFalse(self.robot.controller.armed)
        self.assertEqual(self.calls, ["StopMotion"])

    def test_busy_only_disconnect_and_stop_failure_still_close_observations(self):
        self.robot.connect()
        self.robot.controller.busy = True
        with patch.object(self.robot.controller, "stop", side_effect=RuntimeError("stop fixture")) as stop:
            with self.assertRaisesRegex(RuntimeError, "stop fixture"):
                self.robot.disconnect()
        stop.assert_called_once()
        self.camera.disconnect.assert_called_once()
        self.stream.disconnect.assert_called_once()
        self.assertFalse(self.robot._watchdog_thread.is_alive())
        self.assertFalse(self.robot.is_connected)
        self.robot.controller.busy = False

    def test_explicit_workspace_and_boolean_opt_in_are_required(self):
        with self.assertRaises(TypeError):
            FairinoFR5ControlConfig()
        with self.assertRaises(ValueError):
            FairinoFR5ControlConfig(limits=None)
        with self.assertRaises(ValueError):
            FairinoFR5ControlConfig(limits=self.config.limits, allow_motion=1)


if __name__ == "__main__":
    unittest.main()
