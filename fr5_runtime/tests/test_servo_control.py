"""Servo driver tests: synthetic state, fake RPC, and no robot connections."""

from contextlib import nullcontext
from copy import deepcopy
import http.client
import io
import math
import threading
import time
import unittest
from unittest.mock import Mock, patch
import xmlrpc.client

from lerobot_robot_fairino.control_transport import NoRetryTransport
from lerobot_robot_fairino.servo_control import ServoControlError, StreamServoController, _PersistentRPC


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.002)
    return predicate()


class Clock:
    def __init__(self):
        self.value = 10.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def advance(self, seconds):
        with self.lock:
            self.value += seconds


class StateStream:
    def __init__(self):
        self.state = {
            "joints": [10., -8., 5., 4., -3., 2.],
            "main_code": 0, "sub_code": 0, "EmergencyStop": 0,
            "collisionState": 0, "safety_stop0_state": 0, "safety_stop1_state": 0,
            "gripper_fault": 0, "gripper_fault_id": 0, "rbtEnableState": 1,
            "gripper_motiondone": 1, "gripper_position": 100., "gripper_active": 1,
            "gripper_speed": 0, "received_monotonic": 10.,
            "robot_state": 1, "program_state": 1, "motion_done": 1, "mc_queue_len": 0,
        }
        self.lock = threading.Lock()
        self.error = None
        self.ages = []

    def latest(self, max_age_s):
        with self.lock:
            self.ages.append(max_age_s)
            if self.error:
                raise self.error
            return deepcopy(self.state)

    def set(self, **values):
        with self.lock:
            self.state.update(values)


class FakeNetwork:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.before = {}
        self.after = {}
        self.results = {}
        self.channels = {}
        self.complete_gripper = True
        self.stream = self.clock = None

    def factory(self, name):
        channel = FakeRPC(self, name)
        self.channels[name] = channel
        return channel

    def add(self, channel, method, args):
        with self.lock:
            self.calls.append((channel, method, deepcopy(args)))

    def matching(self, method):
        with self.lock:
            return [(channel, args) for channel, name, args in self.calls if name == method]


class FakeRPC:
    def __init__(self, network, name):
        self.network = network
        self.name = name
        self.closed = 0

    def call(self, method, *args, timeout_s=3., before_send=None):
        hook = self.network.before.get(method)
        if hook:
            hook()
        with before_send() if before_send is not None else nullcontext():
            self.network.add(self.name, method, args)
        hook = self.network.after.get(method)
        if hook:
            hook()
        result = self.network.results.get(method, 0)
        if isinstance(result, BaseException):
            raise result
        if method == "MoveGripper" and result == 0 and self.network.complete_gripper:
            self.network.stream.set(gripper_position=args[1], gripper_motiondone=1,
                                    received_monotonic=self.network.clock())
        if method == "GetJointSoftLimitDeg":
            return [0] + [-180., 180.] * 6 if result == 0 else result
        return result

    def close(self):
        self.closed += 1


class ServoControllerTests(unittest.TestCase):
    def setUp(self):
        # This suite must remain fully offline, even if a future test regresses.
        for name in ("socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(name, side_effect=AssertionError("offline test attempted a connection"))
            guard.start()
            self.addCleanup(guard.stop)
        self.stream = StateStream()
        self.network = FakeNetwork()
        self.clock = Clock()
        self.network.stream, self.network.clock = self.stream, self.clock
        self.camera = Mock(return_value=True)
        self.records = []
        self.controller = StreamServoController(
            self.stream, self.camera, self.records.append,
            rpc_factory=self.network.factory, clock=self.clock,
            wall_clock=lambda: 1_700_000_000. + self.clock(),
        )
        self.release = []
        self.addCleanup(self.cleanup_controller)

    def cleanup_controller(self):
        for event in self.release:
            event.set()
        self.controller.close()

    def block_hook(self, where, method):
        entered, release = threading.Event(), threading.Event()
        self.release.append(release)

        def hook():
            entered.set()
            if not release.wait(3):
                raise AssertionError("test failed to release fake RPC")

        getattr(self.network, where)[method] = hook
        return entered, release

    def target(self, joints=None, gripper=1.):
        return [*(math.radians(q) for q in (joints or self.stream.state["joints"])), gripper]

    def test_construct_is_inert_and_start_uses_measured_joint_hold(self):
        self.assertEqual(self.network.calls, [])
        with self.assertRaises(ServoControlError):
            self.controller.start()
        self.assertEqual(self.network.calls, [])
        self.controller.start(operator_confirmed=True)
        self.assertTrue(wait_until(lambda: bool(self.network.matching("ServoJ"))))
        args = self.network.matching("ServoJ")[0][1]
        self.assertEqual(args, ([10., -8., 5., 4., -3., 2.], [0.]*4, 0., 0., .008, 0., 0., 0))
        self.assertEqual(self.network.matching("MoveGripper"), [])
        self.assertEqual(self.stream.ages, [.25] * len(self.stream.ages))
        self.controller.close()
        self.assertEqual(len(self.network.matching("ServoMoveStart")), 1)
        self.assertEqual(len(self.network.matching("StopMotion")), 1)
        self.assertEqual(len(self.network.matching("ServoMoveEnd")), 1)
        row = next(row for row in self.records if row["method"] == "ServoJ")
        self.assertEqual(row["issued_monotonic"], 10.)
        self.assertEqual(row["issued_unix"], 1_700_000_010.)
        self.assertEqual(row["return_code"], 0)

    def test_linear_interpolation_and_gripper_share_one_policy_target(self):
        self.controller.start(operator_confirmed=True)
        self.assertTrue(wait_until(lambda: bool(self.network.matching("ServoJ"))))
        start = self.stream.state["joints"]
        goal = [x + 20. for x in start]
        self.controller.submit(self.target(goal, .76), duration=1.)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.clock.advance(.5)
        halfway = [x + 10. for x in start]
        self.assertTrue(wait_until(lambda: any(
            all(abs(a-b) < 1e-8 for a, b in zip(args[0], halfway))
            for _, args in self.network.matching("ServoJ"))))
        self.clock.advance(.5)
        self.assertTrue(wait_until(lambda: any(
            all(abs(a-b) < 1e-8 for a, b in zip(args[0], goal))
            for _, args in self.network.matching("ServoJ"))))
        grip = self.network.matching("MoveGripper")[0]
        self.assertEqual(grip, ("gripper", (1, 76, 20, 10, 10000, 1, 0, 0., 0, 0)))
        snapshot = self.controller.close()
        self.assertEqual(snapshot["pending_callbacks"], 0)
        self.assertEqual(snapshot["workers_alive"], [])
        with self.network.lock:
            expected = [(method, list(args)) for _, method, args in self.network.calls
                        if method != "GetJointSoftLimitDeg"]
        actual = sorted(self.records, key=lambda row: row["sequence"])
        self.assertEqual([(row["method"], row["raw_args"]) for row in actual], expected)
        self.assertEqual({row["method"] for row in actual},
                         {"ServoMoveStart", "ServoJ", "MoveGripper", "StopMotion", "ServoMoveEnd"})
        for row in actual:
            self.assertEqual(row["args"], row["raw_args"])
            self.assertEqual(row["issued_unix"], 1_700_000_000. + row["issued_monotonic"])
            self.assertEqual(row["returned_unix"], 1_700_000_000. + row["returned_monotonic"])

    def test_integer_gripper_deduplicated_but_arm_still_updates(self):
        self.controller.start(operator_confirmed=True)
        target = self.target([20., -8., 5., 4., -3., 2.], .76)
        for _ in range(4):
            self.controller.submit(target)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: any(abs(args[0][0]-20.) < 1e-8
                                              for _, args in self.network.matching("ServoJ"))))
        self.controller.submit(self.target([20., -8., 5., 4., -3., 2.], .7601))
        time.sleep(.03)
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)

    def test_latest_gripper_wins_while_servo_holds_during_peripheral_io(self):
        entered, release = self.block_hook("after", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.70))
        self.assertTrue(entered.wait(1))
        before = len(self.network.matching("ServoJ"))
        self.controller.submit(self.target(gripper=.60))
        self.controller.submit(self.target(gripper=.50))
        time.sleep(.025)
        self.assertEqual(len(self.network.matching("ServoJ")), before)
        self.network.after.pop("MoveGripper")
        release.set()
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 2))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [70, 50])
        self.assertTrue(wait_until(lambda: len(self.network.matching("ServoJ")) > before))

    def test_stop_during_gripper_connect_prevents_late_send(self):
        entered, release = self.block_hook("before", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.5))
        self.assertTrue(entered.wait(1))
        self.controller.stop("user stop")
        release.set()
        self.controller.close()
        self.assertEqual(self.network.matching("MoveGripper"), [])
        self.assertEqual(len(self.network.matching("StopMotion")), 1)

    def test_stop_not_blocked_by_inflight_servo_and_no_later_servo(self):
        entered, release = self.block_hook("after", "ServoJ")
        self.controller.start(operator_confirmed=True)
        self.assertTrue(entered.wait(1))
        self.controller.stop("user stop")
        self.assertEqual(len(self.network.matching("StopMotion")), 1)
        self.assertEqual(self.network.matching("ServoMoveEnd"), [])
        release.set()
        self.controller.close()
        self.assertEqual(len(self.network.matching("ServoJ")), 1)
        self.assertEqual(len(self.network.matching("ServoMoveEnd")), 1)
        self.assertEqual(self.network.matching("StopMotion")[0][0], "stop")

    def test_gripper_response_inflight_after_stop_has_no_queued_successor(self):
        entered, release = self.block_hook("after", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.7))
        self.assertTrue(entered.wait(1))
        self.controller.submit(self.target(gripper=.5))
        self.controller.stop("user stop")
        release.set()
        self.controller.close()
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [70])

    def test_lost_servo_response_stops_once_without_retry(self):
        self.network.results["ServoJ"] = ConnectionError("response lost")
        self.controller.start(operator_confirmed=True)
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        self.controller.close()
        self.assertEqual(len(self.network.matching("ServoJ")), 1)
        self.assertEqual(len(self.network.matching("StopMotion")), 1)
        self.assertEqual(len(self.network.matching("ServoMoveEnd")), 1)
        record = next(row for row in self.records if row["method"] == "ServoJ")
        self.assertIsNone(record["return_code"])
        self.assertIn("response lost", record["error"])

    def test_nonzero_gripper_code_stops_and_is_recorded(self):
        self.network.results["MoveGripper"] = 37
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.5))
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        self.controller.close()
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)
        row = next(row for row in self.records if row["method"] == "MoveGripper")
        self.assertEqual(row["return_code"], 37)
        self.assertIn("returned 37", row["error"])

    def test_fault_or_stale_camera_stops_without_requiring_stationary(self):
        self.controller.start(operator_confirmed=True)
        self.stream.set(motion_done=0, mc_queue_len=5, program_state=2, robot_state=2)
        time.sleep(.12)
        self.assertEqual(self.controller.snapshot()["status"], "running")
        self.camera.return_value = False
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        self.assertIn("camera", self.controller.snapshot()["failure"].lower())
        self.assertEqual(len(self.network.matching("StopMotion")), 1)

    def test_motiondone_three_and_fault_id_alone_do_not_stop_but_fault_does(self):
        self.controller.start(operator_confirmed=True)
        self.stream.set(gripper_motiondone=3, gripper_fault_id=1, gripper_fault=0)
        self.assertTrue(wait_until(lambda: len(self.network.matching("ServoJ")) >= 2))
        time.sleep(.12)
        self.assertEqual(self.controller.snapshot()["status"], "running")
        self.assertEqual(self.network.matching("StopMotion"), [])
        self.stream.set(gripper_fault=3)
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        self.assertIn("gripper_fault=3", self.controller.snapshot()["failure"])
        self.assertEqual(len(self.network.matching("StopMotion")), 1)

    def test_busy_gripper_feedback_cannot_seed_servo_start(self):
        self.stream.set(gripper_position=0, gripper_active=1, gripper_motiondone=0)
        with self.assertRaisesRegex(ServoControlError, 'Settled gripper feedback'):
            self.controller.start(operator_confirmed=True)
        self.assertEqual(self.network.matching('ServoMoveStart'), [])

    def test_invalid_or_outside_native_limits_never_dispatched(self):
        self.controller.start(operator_confirmed=True)
        for bad in ([math.nan]*7, [0.]*6 + [1.1], [math.pi + .1] + [0.]*5 + [.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.controller.submit(bad)
        self.assertEqual(self.network.matching("MoveGripper"), [])

    def test_superseded_gripper_during_connection_does_not_dispatch(self):
        entered, release = self.block_hook("before", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.7))
        self.assertTrue(entered.wait(1))
        self.controller.submit(self.target(gripper=.5))
        self.network.before.pop("MoveGripper")
        release.set()
        self.assertTrue(wait_until(lambda: bool(self.network.matching("MoveGripper"))))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [50])

    def test_command_recorder_runs_off_servo_thread(self):
        entered, release = threading.Event(), threading.Event()
        self.release.append(release)

        def slow_callback(record):
            entered.set()
            release.wait(3)
            self.records.append(record)

        self.controller.command_callback = slow_callback
        self.controller.start(operator_confirmed=True)
        self.assertTrue(entered.wait(1))
        count = len(self.network.matching("ServoJ"))
        self.assertTrue(wait_until(lambda: len(self.network.matching("ServoJ")) > count))
        self.controller.stop()
        release.set()
        self.controller.close()
        self.assertEqual(len([x for x in self.records if x["method"] == "ServoJ"]),
                         len(self.network.matching("ServoJ")))

    def test_start_response_inflight_is_stopped_and_closed_without_late_servo(self):
        entered, release = self.block_hook("after", "ServoMoveStart")
        errors = []

        def start():
            try:
                self.controller.start(operator_confirmed=True)
            except Exception as exc:
                errors.append(exc)

        starter = threading.Thread(target=start)
        starter.start()
        self.assertTrue(entered.wait(1))
        self.controller.stop("user stop during start")
        self.assertEqual(len(self.network.matching("StopMotion")), 1)
        self.assertEqual(self.network.matching("ServoMoveEnd"), [])
        release.set()
        starter.join(2)
        self.assertFalse(starter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ServoControlError)
        self.assertEqual(self.network.matching("ServoJ"), [])
        self.assertEqual(len(self.network.matching("ServoMoveEnd")), 1)
        self.controller.close()
        self.controller.close()
        self.assertTrue(all(channel.closed == 1 for channel in self.network.channels.values()))

    def test_stop_from_other_thread_waits_until_guarded_start_is_sent(self):
        self.controller.close()
        self.network = FakeNetwork()
        stop_entered, stop_done = threading.Event(), threading.Event()
        test = self

        class StartRPC(FakeRPC):
            def call(self, method, *args, timeout_s=3., before_send=None):
                if method != "ServoMoveStart":
                    return super().call(method, *args, timeout_s=timeout_s, before_send=before_send)
                with before_send():
                    def stop():
                        stop_entered.set()
                        test.controller.stop("synthetic separate-thread stop")
                        stop_done.set()
                    threading.Thread(target=stop, daemon=True).start()
                    if not stop_entered.wait(1):
                        raise AssertionError("stop thread did not start")
                    if test.network.matching("StopMotion"):
                        raise AssertionError("stop overtook the guarded start write")
                    test.network.add(self.name, method, args)
                return 0

        def factory(name):
            channel = StartRPC(test.network, name) if name == "servo" else FakeRPC(test.network, name)
            test.network.channels[name] = channel
            return channel

        self.controller = StreamServoController(self.stream, self.camera, self.records.append,
                                                rpc_factory=factory, clock=self.clock)
        try:
            self.controller.start(operator_confirmed=True)
        except ServoControlError:
            pass  # Stop may win immediately after the start write.
        self.assertTrue(stop_done.wait(2))
        self.controller.close()
        with self.network.lock:
            methods = [method for _, method, _ in self.network.calls]
        self.assertLess(methods.index("ServoMoveStart"), methods.index("StopMotion"))
        self.assertFalse(any(method in ("ServoJ", "MoveGripper")
                             for method in methods[methods.index("StopMotion")+1:]))

    def test_stop_during_servo_connection_blocks_first_late_send(self):
        entered, release = self.block_hook("before", "ServoJ")
        self.controller.start(operator_confirmed=True)
        self.assertTrue(entered.wait(1))
        self.controller.stop("user stop")
        release.set()
        self.controller.close()
        self.assertEqual(self.network.matching("ServoJ"), [])
        self.assertEqual(len(self.network.matching("StopMotion")), 1)

    def test_new_joint_target_cancels_a_superseded_unsent_sample(self):
        entered, release = self.block_hook("before", "ServoJ")
        self.controller.start(operator_confirmed=True)
        self.assertTrue(entered.wait(1))
        goal = [20., -8., 5., 4., -3., 2.]
        self.controller.submit(self.target(goal))
        self.clock.advance(1.)
        self.network.before.pop("ServoJ")
        release.set()
        self.assertTrue(wait_until(lambda: bool(self.network.matching("ServoJ"))))
        first = self.network.matching("ServoJ")[0][1][0]
        self.assertTrue(all(abs(a-b) < 1e-8 for a, b in zip(first, goal)))

    def test_bad_native_limits_prevent_servo_start(self):
        self.network.results["GetJointSoftLimitDeg"] = [0] + [30., -30.] * 6
        with self.assertRaises(ServoControlError):
            self.controller.start(operator_confirmed=True)
        self.assertEqual(self.network.matching("ServoMoveStart"), [])
        self.assertEqual(self.network.matching("ServoJ"), [])
        self.assertEqual(self.network.matching("StopMotion"), [])

    def test_callback_failure_requests_stop(self):
        def broken_callback(_record):
            raise OSError("log unavailable")

        self.controller.command_callback = broken_callback
        try:
            self.controller.start(operator_confirmed=True)
        except ServoControlError:
            pass
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        self.assertIn("log unavailable", self.controller.snapshot()["callback_failure"])
        self.assertEqual(len(self.network.matching("StopMotion")), 1)
        self.controller.close()
        pending = self.controller.snapshot()["undelivered_callbacks"]
        self.assertTrue(pending)
        self.assertEqual(pending[0]["method"], "ServoMoveStart")

    def test_start_unknown_result_and_cleanup_are_recorded(self):
        self.network.results["ServoMoveStart"] = ConnectionError("start response lost")
        with self.assertRaises(ServoControlError):
            self.controller.start(operator_confirmed=True)
        snapshot = self.controller.close()
        self.assertEqual(snapshot["pending_callbacks"], 0)
        self.assertEqual(snapshot["workers_alive"], [])
        by_method = {row["method"]: row for row in self.records}
        self.assertEqual(set(by_method), {"ServoMoveStart", "StopMotion", "ServoMoveEnd"})
        self.assertIsNone(by_method["ServoMoveStart"]["return_code"])
        self.assertIn("start response lost", by_method["ServoMoveStart"]["error"])

    def test_logger_waits_for_delayed_stop_and_servo_end_records(self):
        stop_entered, stop_release = self.block_hook("after", "StopMotion")
        end_entered, end_release = self.block_hook("after", "ServoMoveEnd")
        self.controller.start(operator_confirmed=True)
        stopper = threading.Thread(target=self.controller.stop)
        stopper.start()
        self.assertTrue(stop_entered.wait(1))
        time.sleep(.12)
        self.assertIn("fr5-servo-log", self.controller.snapshot()["workers_alive"])
        stop_release.set()
        stopper.join(2)
        self.assertTrue(end_entered.wait(1))
        time.sleep(.12)
        self.assertIn("fr5-servo-log", self.controller.snapshot()["workers_alive"])
        end_release.set()
        snapshot = self.controller.close()
        self.assertEqual(snapshot["pending_callbacks"], 0)
        self.assertEqual(snapshot["workers_alive"], [])
        methods = [row["method"] for row in self.records]
        self.assertEqual(methods.count("StopMotion"), 1)
        self.assertEqual(methods.count("ServoMoveEnd"), 1)

    def test_unknown_stop_and_end_outcomes_are_recorded_without_retry(self):
        self.network.results["StopMotion"] = ConnectionError("stop response lost")
        self.network.results["ServoMoveEnd"] = ConnectionError("end response lost")
        self.controller.start(operator_confirmed=True)
        snapshot = self.controller.close()
        self.assertEqual(snapshot["pending_callbacks"], 0)
        self.assertEqual(snapshot["workers_alive"], [])
        for method in ("StopMotion", "ServoMoveEnd"):
            rows = [row for row in self.records if row["method"] == method]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["raw_args"], [])
            self.assertIsNone(rows[0]["return_code"])
            self.assertIn("response lost", rows[0]["error"])

    def test_start_requires_native_stationary_state_without_moving(self):
        for field, value in (("robot_state", 2), ("robot_state", 3), ("robot_state", 4),
                             ("program_state", 2), ("program_state", 3), ("mc_queue_len", 1)):
            with self.subTest(field=field):
                stream, network = StateStream(), FakeNetwork()
                stream.set(**{field: value})
                controller = StreamServoController(stream, lambda: True, lambda _row: None,
                                                   rpc_factory=network.factory, clock=self.clock)
                try:
                    with self.assertRaisesRegex(ServoControlError, "not stopped"):
                        controller.start(operator_confirmed=True)
                    self.assertEqual(network.matching("ServoMoveStart"), [])
                    self.assertEqual(network.matching("ServoJ"), [])
                finally:
                    controller.close()

    def test_stopped_controller_without_prior_motion_completion_uses_measured_hold(self):
        self.stream.set(motion_done=0)
        measured = self.stream.state["joints"][:]
        self.controller.start(operator_confirmed=True)
        self.assertTrue(wait_until(lambda: bool(self.network.matching("ServoJ"))))
        self.assertEqual(self.network.matching("ServoJ")[0][1][0], measured)
        self.assertEqual(len(self.network.matching("ServoMoveStart")), 1)
        self.assertEqual(self.network.matching("MoveGripper"), [])

    def test_stop_before_start_needs_no_native_stop_rpc(self):
        snapshot = self.controller.stop("operator stop before start")
        self.assertIsNone(snapshot["stop_result"])
        self.assertEqual(snapshot["servo_commands"], 0)
        self.assertEqual(snapshot["gripper_commands"], 0)
        self.assertEqual(self.network.calls, [])
        self.controller.close()
        with self.assertRaises(ServoControlError):
            self.controller.start(operator_confirmed=True)
        self.assertEqual(self.network.calls, [])

    def test_start_rechecks_stationary_immediately_before_wire(self):
        entered, release = self.block_hook("before", "ServoMoveStart")
        errors = []

        def start():
            try:
                self.controller.start(operator_confirmed=True)
            except Exception as exc:
                errors.append(exc)

        starter = threading.Thread(target=start)
        starter.start()
        self.assertTrue(entered.wait(1))
        self.stream.set(motion_done=0, robot_state=2)
        release.set()
        starter.join(2)
        self.assertFalse(starter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("not stopped", str(errors[0]))
        self.assertEqual(self.network.matching("ServoMoveStart"), [])

    def test_ordered_ack_waits_for_endpoint_response_and_blocks_overwrite(self):
        entered, release = threading.Event(), threading.Event()
        self.release.append(release)
        goal = [30., -8., 5., 4., -3., 2.]

        def block_endpoint():
            args = self.network.matching("ServoJ")[-1][1]
            if all(abs(a-b) < 1e-9 for a, b in zip(args[0], goal)):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test failed to release endpoint")

        self.network.after["ServoJ"] = block_endpoint
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(goal), duration=1.)
        self.assertFalse(self.controller.target_acknowledged(token))
        self.clock.advance(.5)
        self.assertTrue(wait_until(lambda: any(abs(args[0][0]-20.) < 1e-8
                                              for _, args in self.network.matching("ServoJ"))))
        self.assertFalse(self.controller.target_acknowledged(token))
        self.clock.advance(.5)
        self.assertTrue(entered.wait(1))
        self.assertFalse(self.controller.target_acknowledged(token))
        with self.assertRaisesRegex(ServoControlError, "awaiting acknowledgement"):
            self.controller.submit_ordered(self.target(gripper=.5))
        with self.assertRaisesRegex(ServoControlError, "awaiting acknowledgement"):
            self.controller.submit(self.target(gripper=.5))
        self.network.after.pop("ServoJ")
        release.set()
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual(self.network.matching("MoveGripper"), [])
        self.controller.close()
        self.assertTrue(any(row["method"] == "ServoJ" and row["return_code"] == 0
                            and all(abs(a-b) < 1e-9 for a, b in zip(row["target_deg_percent"], goal))
                            for row in self.records))

    def test_ordered_ack_waits_for_gripper_response_and_does_not_resend_duplicate(self):
        entered, release = self.block_hook("after", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        first = self.controller.submit_ordered(self.target(gripper=.7))
        self.clock.advance(.2)
        self.assertTrue(entered.wait(1))
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["ordered_pending"]["joint_ack"]))
        self.assertFalse(self.controller.target_acknowledged(first))
        self.network.after.pop("MoveGripper")
        release.set()
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(first)))
        second = self.controller.submit_ordered(self.target(gripper=.7001))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(second)))
        self.assertTrue(self.controller.target_acknowledged(first))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [70])

    def test_release_precheck_stops_before_the_gripper_write(self):
        self.stream.set(gripper_position=78., gripper_motiondone=2)
        self.controller.start(operator_confirmed=True)
        precheck = Mock(side_effect=RuntimeError("measured endpoint above table"))
        token = self.controller.submit_ordered(self.target(gripper=1.), gripper_precheck=precheck)
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["failure"]))
        self.assertIn("measured endpoint above table", self.controller.snapshot()["failure"])
        self.assertEqual(precheck.call_args.args[0]["gripper_motiondone"], 2)
        self.assertEqual(self.network.matching("MoveGripper"), [])
        with self.assertRaisesRegex(ServoControlError, "stopped"):
            self.controller.target_acknowledged(token)

    def test_ordered_one_percent_oscillation_keeps_latched_gripper_and_joint_targets(self):
        self.controller.start(operator_confirmed=True)
        targets = [76] + [75, 76] * 22
        for index, gripper in enumerate(targets):
            joints = [10. + index * .01, -8., 5., 4., -3., 2.]
            token = self.controller.submit_ordered(self.target(joints, gripper / 100.))
            self.assertEqual(self.controller.snapshot()["gripper_target_percent"], 76)
            self.clock.advance(.2)
            self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [76])
        self.assertTrue(any(abs(args[0][0] - joints[0]) < 1e-8
                            for _, args in self.network.matching("ServoJ")))

    def test_ordered_deadband_is_relative_to_last_sent_integer_target(self):
        self.stream.set(gripper_position=40)
        self.controller.start(operator_confirmed=True)
        for proposed, expected in [(39, 40), (41, 40), (38, 38), (37, 38),
                                   (36, 36), (37, 36), (39, 39), (100, 100), (0, 0)]:
            with self.subTest(proposed=proposed, expected=expected):
                token = self.controller.submit_ordered(self.target(gripper=proposed / 100.))
                self.assertEqual(self.controller.snapshot()["gripper_target_percent"], expected)
                self.clock.advance(.2)
                self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")],
                         [38, 36, 39, 100, 0])

    def test_ordered_two_percent_change_waits_for_measured_gripper_completion(self):
        self.network.complete_gripper = False
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.98))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.assertFalse(self.controller.target_acknowledged(token))
        self.stream.set(received_monotonic=self.clock(), gripper_position=98, gripper_motiondone=1)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual(self.network.matching("MoveGripper")[0][1][1], 98)

    def test_ordered_one_percent_hold_waits_for_inflight_gripper_completion(self):
        self.network.complete_gripper = False
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.70))
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        token = self.controller.submit_ordered(self.target(gripper=.71))
        self.assertEqual(self.controller.snapshot()["gripper_target_percent"], 70)
        self.clock.advance(.2)
        self.assertFalse(self.controller.target_acknowledged(token))
        self.stream.set(received_monotonic=self.clock(), gripper_position=70, gripper_motiondone=1)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [70])

    def test_gripper_native_acceptance_and_stale_target_are_not_completion(self):
        self.network.complete_gripper = False
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.79))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["ordered_pending"]["joint_ack"]))
        # A new host frame can still contain the previous device target.
        self.stream.set(received_monotonic=self.clock(), gripper_position=100, gripper_motiondone=1)
        time.sleep(.025)
        self.assertFalse(self.controller.target_acknowledged(token))
        with self.assertRaisesRegex(ServoControlError, "awaiting acknowledgement"):
            self.controller.submit_ordered(self.target(gripper=.76))
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)
        self.stream.set(received_monotonic=self.clock(), gripper_position=79, gripper_motiondone=1)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))

    def test_closing_contact_completes_without_forcing_recorded_target(self):
        self.network.complete_gripper = False
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.70))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.stream.set(received_monotonic=self.clock(), gripper_position=78, gripper_motiondone=2)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)
        self.assertIn("not grasp success",self.controller.snapshot()["gripper_completion_feedback"]["meaning"])

    def test_previous_contact_is_not_completion_of_a_new_close(self):
        self.stream.set(gripper_position=78, gripper_motiondone=2)
        self.network.complete_gripper = False
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.74))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.stream.set(received_monotonic=self.clock(), gripper_position=78, gripper_motiondone=2)
        time.sleep(.025)
        self.assertFalse(self.controller.target_acknowledged(token))
        self.stream.set(received_monotonic=self.clock(), gripper_position=77, gripper_motiondone=2)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))

    def test_stop_interrupts_physical_completion_wait_without_sending_successor(self):
        self.network.complete_gripper = False
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.79))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: len(self.network.matching("MoveGripper")) == 1))
        self.stream.set(received_monotonic=self.clock(), gripper_position=0, gripper_motiondone=0)
        self.controller.stop("operator stop")
        snapshot = self.controller.close()
        self.assertFalse(snapshot["workers_alive"])
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)
        with self.assertRaises(ServoControlError):
            self.controller.target_acknowledged(token)

    def test_ordered_70_60_50_are_all_dispatched_after_each_ack(self):
        entered, release = self.block_hook("after", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        first = self.controller.submit_ordered(self.target(gripper=.7))
        self.clock.advance(.2)
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(ServoControlError, "awaiting acknowledgement"):
            self.controller.submit_ordered(self.target(gripper=.6))
        self.assertFalse(self.controller.target_acknowledged(first))
        self.network.after.pop("MoveGripper")
        release.set()
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(first)))
        for target in (.6, .5):
            token = self.controller.submit_ordered(self.target(gripper=target))
            self.clock.advance(.2)
            self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual([args[1] for _, args in self.network.matching("MoveGripper")], [70, 60, 50])
        snapshot = self.controller.close()
        self.assertEqual(snapshot["ordered_acknowledged_token"], 3)
        self.assertEqual([row["raw_args"][1] for row in self.records if row["method"] == "MoveGripper"],
                         [70, 60, 50])

    def test_ordered_joint_endpoints_survive_a_delayed_servo_response(self):
        entered, release = threading.Event(), threading.Event()
        self.release.append(release)
        first_goal = [20., -8., 5., 4., -3., 2.]
        second_goal = [30., -8., 5., 4., -3., 2.]

        def block_first_endpoint():
            args = self.network.matching("ServoJ")[-1][1]
            if abs(args[0][0] - 20.) < 1e-9:
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test failed to release first endpoint")

        self.network.after["ServoJ"] = block_first_endpoint
        self.controller.start(operator_confirmed=True)
        first = self.controller.submit_ordered(self.target(first_goal))
        self.clock.advance(.2)
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(ServoControlError, "awaiting acknowledgement"):
            self.controller.submit_ordered(self.target(second_goal))
        self.assertFalse(self.controller.target_acknowledged(first))
        self.network.after.pop("ServoJ")
        release.set()
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(first)))
        second = self.controller.submit_ordered(self.target(second_goal))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(second)))
        self.controller.close()
        joint_targets = [row["raw_args"][0][0] for row in self.records
                         if row["method"] == "ServoJ" and row["return_code"] == 0]
        first_index = next(i for i, value in enumerate(joint_targets) if abs(value-20.) < 1e-9)
        second_index = next(i for i, value in enumerate(joint_targets) if abs(value-30.) < 1e-9)
        self.assertLess(first_index, second_index)

    def test_ordered_waits_if_same_gripper_target_was_sent_but_not_acknowledged(self):
        entered, release = self.block_hook("after", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        self.controller.submit(self.target(gripper=.7))
        self.assertTrue(entered.wait(1))
        token = self.controller.submit_ordered(self.target(gripper=.7))
        self.clock.advance(.2)
        self.assertFalse(self.controller.snapshot()["ordered_pending"]["joint_ack"])
        self.assertFalse(self.controller.target_acknowledged(token))
        release.set()
        self.assertTrue(wait_until(lambda: self.controller.target_acknowledged(token)))
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)

    def test_ordered_stop_race_cancels_unsent_gripper_and_fails_poll(self):
        entered, release = self.block_hook("before", "MoveGripper")
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.5))
        self.clock.advance(.2)
        self.assertTrue(entered.wait(1))
        self.controller.stop("operator stop")
        release.set()
        self.controller.close()
        with self.assertRaisesRegex(ServoControlError, "stopped before"):
            self.controller.target_acknowledged(token)
        with self.assertRaises(ServoControlError):
            self.controller.submit_ordered(self.target(gripper=.6))
        self.assertEqual(self.network.matching("MoveGripper"), [])

    def test_ordered_stop_during_endpoint_response_does_not_ack_late_success(self):
        entered, release = threading.Event(), threading.Event()
        self.release.append(release)
        goal = [30., -8., 5., 4., -3., 2.]

        def block_endpoint():
            args = self.network.matching("ServoJ")[-1][1]
            if abs(args[0][0] - 30.) < 1e-9:
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test failed to release endpoint")

        self.network.after["ServoJ"] = block_endpoint
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(goal))
        self.clock.advance(.2)
        self.assertTrue(entered.wait(1))
        self.controller.stop("operator stop")
        sent_at_stop = len(self.network.matching("ServoJ"))
        release.set()
        self.controller.close()
        with self.assertRaisesRegex(ServoControlError, "stopped before"):
            self.controller.target_acknowledged(token)
        self.assertEqual(len(self.network.matching("ServoJ")), sent_at_stop)

    def test_ordered_bad_servo_result_stops_without_false_ack(self):
        goal = [30., -8., 5., 4., -3., 2.]

        def fail_endpoint():
            args = self.network.matching("ServoJ")[-1][1]
            if abs(args[0][0] - 30.) < 1e-9:
                raise ConnectionError("endpoint response lost")

        self.network.after["ServoJ"] = fail_endpoint
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(goal))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        with self.assertRaisesRegex(ServoControlError, "stopped before"):
            self.controller.target_acknowledged(token)
        self.controller.close()
        failed = [row for row in self.records if row["method"] == "ServoJ" and row["error"]]
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0]["return_code"])

    def test_ordered_poll_does_not_wait_for_guarded_rpc_write(self):
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.7), duration=10.)
        results = []
        with self.controller._lock:
            thread = threading.Thread(target=lambda: results.append(self.controller.target_acknowledged(token)))
            thread.start()
            thread.join(.2)
            self.assertFalse(thread.is_alive(), "ack polling waited on the guarded send lock")
        self.assertEqual(results, [False])

    def test_ordered_bad_gripper_result_stops_without_false_ack(self):
        self.network.results["MoveGripper"] = 37
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target(gripper=.5))
        self.clock.advance(.2)
        self.assertTrue(wait_until(lambda: self.controller.snapshot()["status"] == "failed"))
        with self.assertRaisesRegex(ServoControlError, "stopped before"):
            self.controller.target_acknowledged(token)
        self.assertEqual(len(self.network.matching("MoveGripper")), 1)

    def test_ordered_tokens_reject_unknown_values(self):
        self.controller.start(operator_confirmed=True)
        token = self.controller.submit_ordered(self.target())
        for invalid in (True, 0, token+1, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.controller.target_acknowledged(invalid)


class TransportTests(unittest.TestCase):
    def test_reused_proxy_does_not_enable_transport_retries(self):
        proxy = Mock()
        with patch("lerobot_robot_fairino.servo_control.xmlrpc.client.ServerProxy", return_value=proxy) as make_proxy:
            rpc = _PersistentRPC("192.168.58.2", 20003)
            rpc.call("ServoJ", *([0] * 8))
            rpc.call("ServoJ", *([0] * 8))
            self.assertEqual(proxy.ServoJ.call_count, 2)
            make_proxy.assert_called_once()
            self.assertIsInstance(rpc.transport, NoRetryTransport)
            with patch.object(rpc.transport, "single_request",
                              side_effect=http.client.RemoteDisconnected("lost")) as single:
                with self.assertRaises(http.client.RemoteDisconnected):
                    rpc.transport.request("robot", "/RPC2", b"request")
                single.assert_called_once()
            rpc.close()

    def test_http_connection_reuses_socket_when_server_keeps_it_open(self):
        self.assertEqual(self._exercise_http_responses(close_each=False), 1)

    def test_http_connection_reopens_only_when_server_requests_close(self):
        self.assertEqual(self._exercise_http_responses(close_each=True), 2)

    def test_gripper_uses_fresh_connection_after_idle_close_without_retry(self):
        body = xmlrpc.client.dumps((0,), methodresponse=True).encode()
        response = (b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                    + b"\r\n\r\n" + body)
        sockets = []

        class FakeSocket:
            def __init__(self, drop_response):
                self.requests = []
                self.closed = False
                self.peer_closed = False
                self.drop_response = drop_response

            def sendall(self, payload):
                self.requests.append(payload)

            def makefile(self, *_args, **_kwargs):
                return io.BytesIO(b"" if self.peer_closed or self.drop_response else response)

            def settimeout(self, _timeout):
                pass

            def close(self):
                self.closed = True

        def connect(connection):
            sock = FakeSocket(drop_response=len(sockets) == 2)
            sockets.append(sock)
            connection.sock = sock

        commands = [
            (1, 85, 20, 10, 10000, 1, 0, 0., 0, 0),
            (1, 78, 20, 10, 10000, 1, 0, 0., 0, 0),
            (1, 76, 20, 10, 10000, 1, 0, 0., 0, 0),
        ]
        with patch.object(http.client.HTTPConnection, "connect", connect), \
             patch("socket.socket.connect", side_effect=AssertionError("offline test attempted a connection")):
            rpc = _PersistentRPC("robot.test", 20003)
            try:
                self.assertEqual(rpc.call("MoveGripper", *commands[0]), 0)
                first_connection = rpc.transport._connection[1]
                sockets[0].peer_closed = True
                self.assertEqual(rpc.call("MoveGripper", *commands[1]), 0)
                self.assertEqual(len(sockets), 2)
                self.assertTrue(sockets[0].closed)
                self.assertIsNot(rpc.transport._connection[1], first_connection)
                with self.assertRaises(http.client.RemoteDisconnected):
                    rpc.call("MoveGripper", *commands[2])
                self.assertEqual(len(sockets), 3)
                for sock, expected in zip(sockets, commands):
                    request = b"".join(sock.requests)
                    self.assertEqual(request.count(b"POST /RPC2 HTTP/1.1"), 1)
                    args, method = xmlrpc.client.loads(request.split(b"\r\n\r\n", 1)[1])
                    self.assertEqual(method, "MoveGripper")
                    self.assertEqual(args, expected)
            finally:
                rpc.close()

    @staticmethod
    def _exercise_http_responses(*, close_each):
        body = xmlrpc.client.dumps((0,), methodresponse=True).encode()
        header = (b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() +
                  (b"\r\nConnection: close" if close_each else b"") + b"\r\n\r\n")
        response = header + body
        sockets = []

        class FakeSocket:
            def __init__(self):
                self.requests = []
                self.closed = False

            def sendall(self, payload):
                self.requests.append(payload)

            def makefile(self, *_args, **_kwargs):
                return io.BytesIO(response)

            def settimeout(self, _timeout):
                pass

            def close(self):
                self.closed = True

        def connect(connection):
            sock = FakeSocket()
            sockets.append(sock)
            connection.sock = sock

        with patch.object(http.client.HTTPConnection, "connect", connect), \
             patch("socket.socket.connect", side_effect=AssertionError("offline test attempted a connection")):
            rpc = _PersistentRPC("robot.test", 20003)
            rpc.call("ServoJ", [0.]*6, [0.]*4, 0., 0., .008, 0., 0., 0)
            first = rpc.transport._connection[1]
            rpc.call("ServoJ", [1.]*6, [0.]*4, 0., 0., .008, 0., 0., 0)
            assert rpc.transport._connection[1] is first
            count = len(sockets)
            assert sum(b"".join(sock.requests).count(b"POST /RPC2 HTTP/1.1") for sock in sockets) == 2
            rpc.close()
            return count


if __name__ == "__main__":
    unittest.main()
