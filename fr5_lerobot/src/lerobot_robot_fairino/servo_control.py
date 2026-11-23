"""FR5 joint servo streaming with independent, latest-wins gripper commands.

This is an adapter, not a trajectory planner or a collision detector. The
caller owns the robot lease and the lifetime/freshness of ``stream`` and
``camera_fresh``. Creating this controller does not connect to the robot.

``submit`` accepts six absolute joint targets in radians and one gripper
fraction. Targets are interpolated in joint space over the supplied duration;
there is no workspace box, task-completion rule or additional step limit.
``submit_ordered`` preserves one segment until both its joint endpoint and any
changed gripper target have received successful native RPC responses and the
changed gripper has settled in measured feedback. This is not task success.
ServoJ uses the eight wire arguments in sdk-v2.1.4-Robot.py. The controller's
actual support for that signature and 125 Hz over XML-RPC still requires a
separate, read-only version check and a supervised hardware integration test.

The injected ``rpc_factory(channel)`` must return an object with
``call(method, *args, timeout_s=..., before_send=...)`` and ``close()``. It must
invoke the ``before_send`` context around the actual request write and must
never retry. Three separate channels (servo, gripper, stop) are required.
``command_callback`` receives one dictionary for every attempted wire write,
including ServoMoveStart, ServoJ, MoveGripper, StopMotion, and ServoMoveEnd.
``raw_args`` and ``args`` are the exact wire arguments; issue and response times
use both host monotonic and Unix clocks. A lost response has a null return code
and a nonempty error. Callbacks run asynchronously and should return promptly.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import numbers
import queue
import threading
import time
import xmlrpc.client

from .control_transport import FR5RPC, NoRetryTransport
from .feedback import gripper_feedback_valid


class ServoControlError(RuntimeError):
    """Invalid device feedback, command failure, or uncertain command outcome."""


class _Cancelled(Exception):
    pass


class _Superseded(Exception):
    pass


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _vector(values, size, name):
    try:
        values = list(values)
    except TypeError:
        raise ValueError(f"{name} must have {size} finite numeric values") from None
    if len(values) != size or any(
        isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v))
        for v in values
    ):
        raise ValueError(f"{name} must have {size} finite numeric values")
    return tuple(float(v) for v in values)


class _PersistentRPC:
    """Serial XML-RPC channel with fresh gripper connections and no retries."""

    _METHODS = frozenset({"GetJointSoftLimitDeg", "ServoMoveStart", "ServoJ",
                          "ServoMoveEnd", "MoveGripper", "StopMotion"})

    def __init__(self, host, port):
        url = FR5RPC(host, port).url  # Validates authority without connecting.
        self.transport = NoRetryTransport(3.0)
        self.proxy = xmlrpc.client.ServerProxy(url, transport=self.transport,
                                               allow_none=False, use_builtin_types=True)

    def call(self, method, *args, timeout_s=3.0, before_send=None):
        if method not in self._METHODS:
            raise ValueError(f"Unsupported servo RPC method: {method!r}")
        self.transport.timeout_s = _positive(timeout_s, "timeout_s")
        self.transport.write_timeout_s = min(0.25, self.transport.timeout_s)
        if method == "MoveGripper":
            self.transport.close()
        self.transport.before_send = before_send
        try:
            return getattr(self.proxy, method)(*args)
        finally:
            self.transport.before_send = None

    def close(self):
        self.transport.close()


class StreamServoController:
    """Stream absolute six-joint targets and independently command a gripper.

    ``stream`` must already provide ``latest(max_age_s=.25)`` and ``camera_fresh``
    must implement the existing live-camera freshness check. State and camera
    faults stop the stream; this class does not connect the receiver, enable the
    robot, clear faults, change settings, or infer task success.

    One instance is single-use. ``stop`` closes the command gate and sends one
    StopMotion on its own connection without waiting for the servo response.
    ``close`` is idempotent and waits for in-flight calls and log callbacks as
    far as their connection timeouts permit. An already transmitted command
    cannot be retracted; a lost response is reported as an unknown outcome.
    """

    def __init__(self, stream, camera_fresh, command_callback, *, host="192.168.58.2",
                 port=20003, servo_period=.008, rpc_factory=None, clock=time.monotonic,
                 wall_clock=time.time):
        if not all(callable(item) for item in (camera_fresh, command_callback, clock, wall_clock)):
            raise ValueError("camera_fresh, command_callback and clocks must be callable")
        self.stream = stream
        self.camera_fresh = camera_fresh
        self.command_callback = command_callback
        self.clock = clock
        self.wall_clock = wall_clock
        self.servo_period = _positive(servo_period, "servo_period")
        factory = rpc_factory or (lambda _channel: _PersistentRPC(host, port))
        self._servo_rpc = factory("servo")
        self._gripper_rpc = factory("gripper")
        self._stop_rpc = factory("stop")
        if len({id(self._servo_rpc), id(self._gripper_rpc), id(self._stop_rpc)}) != 3:
            raise ValueError("servo, gripper and stop must have separate RPC channels")

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        # This controller's peripheral RPCs stall if ServoJ requests continue
        # arriving. Stop uses its own channel and never takes this mutex.
        self._actuator_io_lock = threading.Lock()
        self._peripheral_pause_count = 0
        self._peripheral_pause_seconds = 0.0
        self._stop_event = threading.Event()
        self._stop_done = threading.Event()
        self._stop_done.set()
        self._startup_done = threading.Event()
        self._startup_done.set()
        self._closed_channels = set()
        self._records = queue.Queue()
        self._record_inflight = 0
        self._sequence = 0
        self._status = "new"
        self._reason = None
        self._failure = None
        self._callback_failure = None
        self._undelivered_records = []
        self._start_issued = False
        self._end_issued = False
        self._stop_sent = False
        self._stop_result = None
        self._end_result = None
        self._limits = None
        self._last_joints = None
        self._segment_from = None
        self._segment_to = None
        self._segment_start = None
        self._segment_duration = None
        self._segment_version = 0
        self._gripper_desired = None
        self._gripper_last_sent = None
        self._gripper_last_succeeded = None
        self._gripper_completion = None
        self._gripper_inflight = False
        self._ordered_token = 0
        self._ordered_ack_token = 0
        self._ordered_pending = None
        self._gripper_precheck = None
        self._producers = 0
        self._threads = []
        self._servo_count = 0
        self._gripper_count = 0
        self._overruns = 0
        self._last_issued = None
        self._last_reply = None

    @staticmethod
    def _assert_stationary(state):
        # Servo entry needs an idle controller, not completion of a prior
        # position command. motion_done may remain zero after a native stop.
        if state["robot_state"] != 1 or state["program_state"] != 1 or state["mc_queue_len"] != 0:
            raise ServoControlError("Previous robot motion has not stopped")

    def _validate_native_state(self, state, *, stationary=False):
        _vector(state["joints"], 6, "measured joints")
        for key in ("main_code", "sub_code", "EmergencyStop", "collisionState",
                    "safety_stop0_state", "safety_stop1_state", "gripper_fault"):
            if state[key] != 0:
                raise ServoControlError(f"Controller reports {key}={state[key]}")
        if state["rbtEnableState"] != 1:
            raise ServoControlError("Robot is not enabled")
        if stationary:
            self._assert_stationary(state)
        # The 3.8.4 state-packet documentation defines motiondone 0/1, not
        # status 3 as a fault. Keep that raw observation; fault is separate.

    def _ready_state(self, *, stationary=False):
        state = self.stream.latest(max_age_s=.25)
        self._validate_native_state(state, stationary=stationary)
        if not self.camera_fresh():
            raise ServoControlError("Live camera unavailable")
        return state

    @staticmethod
    def _decode_limits(reply):
        if not isinstance(reply, (list, tuple)) or len(reply) != 13 or reply[0] != 0:
            raise ServoControlError(f"GetJointSoftLimitDeg failed: {reply!r}")
        values = _vector(reply[1:], 12, "native joint limits")
        limits = tuple((values[2*i], values[2*i+1]) for i in range(6))
        if any(low >= high for low, high in limits):
            raise ServoControlError("Invalid native joint limits")
        return limits

    @staticmethod
    def _require_zero(code, method):
        if isinstance(code, bool) or not isinstance(code, int) or code != 0:
            raise ServoControlError(f"{method} returned {code!r}; command not retried")

    def start(self, operator_confirmed=False):
        """Enter servo mode, initialized from live measured joint angles."""
        if operator_confirmed is not True:
            raise ServoControlError("Explicit operator confirmation is required")
        with self._lock:
            if self._status != "new":
                raise ServoControlError("A controller instance is single-use")
            self._status = "starting"
            self._startup_done.clear()
            logger = threading.Thread(target=self._log_loop, name="fr5-servo-log", daemon=True)
            self._threads = [logger]
            logger.start()
        try:
            self._ready_state(stationary=True)
            limits = self._decode_limits(self._servo_rpc.call("GetJointSoftLimitDeg", 1, timeout_s=3.0))
            state = self._ready_state(stationary=True)
            joints = _vector(state["joints"], 6, "measured joints")
            if any(q < low or q > high for q, (low, high) in zip(joints, limits)):
                raise ServoControlError("Measured joints are outside native joint limits")
            if not gripper_feedback_valid(state):
                raise ServoControlError("Settled gripper feedback is unavailable")
            gripper = state["gripper_position"]
            with self._lock:
                if self._status != "starting":
                    raise _Cancelled("start cancelled")
                self._limits = limits
                self._last_joints = joints
                self._segment_from = self._segment_to = joints
                self._segment_start = self.clock()
                self._segment_duration = 1.0
                self._gripper_desired = self._gripper_last_sent = int(round(gripper))
                self._gripper_last_succeeded = self._gripper_last_sent

            @contextmanager
            def start_gate():
                with self._lock:
                    if self._status != "starting":
                        raise _Cancelled("start cancelled before send")
                    latest = self.stream.latest(max_age_s=.25)
                    self._validate_native_state(latest, stationary=True)
                    joints = _vector(latest["joints"], 6, "measured joints")
                    if any(q < low or q > high for q, (low, high) in zip(joints, self._limits)):
                        raise ServoControlError("Measured joints are outside native joint limits")
                    if not gripper_feedback_valid(latest):
                        raise ServoControlError("Settled gripper feedback is unavailable")
                    gripper = latest["gripper_position"]
                    self._last_joints = self._segment_from = self._segment_to = joints
                    self._segment_start = self.clock()
                    self._gripper_desired = self._gripper_last_sent = int(round(gripper))
                    self._gripper_last_succeeded = self._gripper_last_sent
                    self._start_issued = True
                    yield

            self._invoke(self._servo_rpc, "ServoMoveStart", (), target=None,
                         timeout_s=3.0, gate=start_gate)
            with self._lock:
                if self._status != "starting":
                    raise _Cancelled("start cancelled after send")
                self._status = "running"
                self._producers = 2
                workers = [
                    threading.Thread(target=self._servo_loop, name="fr5-servo", daemon=True),
                    threading.Thread(target=self._gripper_loop, name="fr5-gripper", daemon=True),
                    threading.Thread(target=self._watchdog, name="fr5-servo-watch", daemon=True),
                ]
                self._threads.extend(workers)
                for thread in workers:
                    thread.start()
            self._startup_done.set()
            return self.snapshot()
        except Exception as exc:
            self.stop(f"Servo startup failed: {exc}", failure=not isinstance(exc, _Cancelled))
            if self._start_issued:
                self._finish_servo()
            with self._lock:
                if self._producers == 0 and self._status == "stopping":
                    self._status = "failed" if self._failure else "stopped"
            raise ServoControlError(f"Servo start failed: {exc}") from exc
        finally:
            self._startup_done.set()

    @staticmethod
    def _parse_target(target7, duration):
        target = _vector(target7, 7, "policy target")
        duration = _positive(duration, "duration")
        if not 0 <= target[6] <= 1:
            raise ValueError("Gripper target must be in 0..1")
        joints = tuple(math.degrees(q) for q in target[:6])
        return joints, int(round(target[6] * 100)), duration

    def _set_target_locked(self, joints, gripper, duration):
        if self._status != "running":
            raise ServoControlError("Servo is not running")
        if self._ordered_pending is not None:
            raise ServoControlError("An ordered target is awaiting acknowledgement")
        if any(q < low or q > high for q, (low, high) in zip(joints, self._limits)):
            raise ValueError("Policy target is outside native joint limits")
        # Start from the last target actually sent, not a time-extrapolated
        # target that may never have reached the controller.
        self._segment_from = self._last_joints
        self._segment_to = joints
        self._segment_start = self.clock()
        self._segment_duration = duration
        self._segment_version += 1
        self._gripper_desired = gripper
        self._gripper_precheck = None

    def submit(self, target7, duration=1/6):
        """Set the latest target; cannot override an outstanding ordered target."""
        joints, gripper, duration = self._parse_target(target7, duration)
        with self._condition:
            self._set_target_locked(joints, gripper, duration)
            self._condition.notify_all()

    def submit_ordered(self, target7, duration=1/6, *, gripper_precheck=None):
        """Submit one non-overwritable segment and return its integer token.

        The caller must wait for ``target_acknowledged(token)`` before submitting
        another segment. Changed gripper targets also wait for measured device
        completion; the acknowledgement does not prove a grasp or task success.
        Gripper targets within one percentage point of the last sent target hold
        that target; the initial reference is the measured startup position.
        """
        joints, gripper, duration = self._parse_target(target7, duration)
        with self._condition:
            if self._gripper_last_sent is not None and abs(gripper - self._gripper_last_sent) <= 1:
                gripper = self._gripper_last_sent
            self._set_target_locked(joints, gripper, duration)
            token = self._ordered_token + 1
            unchanged_gripper = (not self._gripper_inflight and
                                 self._gripper_last_sent == self._gripper_last_succeeded == gripper)
            self._ordered_pending = {"token": token, "version": self._segment_version,
                                     "joints": joints, "gripper": gripper,
                                     "joint_ack": False, "gripper_ack": unchanged_gripper}
            self._gripper_precheck = gripper_precheck
            self._ordered_token = token
            self._condition.notify_all()
            return token

    def target_acknowledged(self, token):
        """Poll without waiting for servo/gripper I/O; raise if stopped unacknowledged."""
        if type(token) is not int or not 1 <= token <= self._ordered_token:
            raise ValueError("Unknown ordered target token")
        if token <= self._ordered_ack_token:
            return True
        if self._stop_event.is_set() or self._status != "running":
            if token <= self._ordered_ack_token:
                return True
            raise ServoControlError("Controller stopped before ordered target acknowledgement")
        return False

    def _ack_ordered_locked(self):
        pending = self._ordered_pending
        if pending is not None and pending["joint_ack"] and pending["gripper_ack"]:
            self._ordered_ack_token = pending["token"]
            self._ordered_pending = None
            self._condition.notify_all()

    def _interpolated(self, now):
        alpha = min(1.0, max(0.0, (now - self._segment_start) / self._segment_duration))
        if alpha == 1.0:
            return self._segment_to
        return tuple(a + alpha*(b-a) for a, b in zip(self._segment_from, self._segment_to))

    def _invoke(self, rpc, method, args, *, target, timeout_s, gate):
        issued = issued_unix = sequence = None
        code = None
        error = None

        @contextmanager
        def wire_gate():
            nonlocal issued, issued_unix, sequence
            with gate():
                issued = self.clock()
                issued_unix = self.wall_clock()
                self._sequence += 1
                sequence = self._sequence
                self._record_inflight += 1
                self._last_issued = issued
                yield

        try:
            code = rpc.call(method, *args, timeout_s=timeout_s, before_send=wire_gate)
            if issued is None:
                raise ServoControlError("RPC transport did not apply the send guard")
            self._require_zero(code, method)
            return code
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if issued is not None:
                returned = self.clock()
                returned_unix = self.wall_clock()
                raw_args = deepcopy(list(args))
                record = {"method": method, "raw_args": raw_args, "args": deepcopy(raw_args),
                          "sequence": sequence, "issued_monotonic": issued,
                          "issued_unix": issued_unix, "returned_monotonic": returned,
                          "returned_unix": returned_unix,
                          "target_deg_percent": list(target) if method == "ServoJ" else target,
                          "return_code": code if isinstance(code, int) and not isinstance(code, bool) else None,
                          "error": error}
                with self._condition:
                    self._last_reply = returned
                    self._records.put(record)
                    self._record_inflight -= 1
                    self._condition.notify_all()

    def _send(self, rpc, method, args, target, *, timeout_s, segment_version=None):
        @contextmanager
        def gate():
            with self._lock:
                if self._status != "running":
                    raise _Cancelled("stopped before request write")
                if method == "ServoJ" and segment_version != self._segment_version:
                    raise _Superseded("newer joint target")
                if method == "MoveGripper" and target != self._gripper_desired:
                    raise _Superseded("newer gripper target")
                if method == "ServoJ":
                    self._last_joints = tuple(target)
                    self._servo_count += 1
                else:
                    self._gripper_last_sent = target
                    self._gripper_inflight = True
                    self._gripper_count += 1
                yield
        return self._invoke(rpc, method, args, target=target, timeout_s=timeout_s, gate=gate)

    def _servo_loop(self):
        next_tick = self.clock()
        observed_pauses = 0
        try:
            while not self._stop_event.is_set():
                now = self.clock()
                if now < next_tick and self._stop_event.wait(next_tick - now):
                    break
                try:
                    with self._actuator_io_lock:
                        now = self.clock()
                        with self._lock:
                            if self._status != "running":
                                break
                            if observed_pauses != self._peripheral_pause_count:
                                observed_pauses = self._peripheral_pause_count
                                next_tick = now
                            joints = self._interpolated(now)
                            version = self._segment_version
                            endpoint_due = now - self._segment_start >= self._segment_duration
                        self._send(self._servo_rpc, "ServoJ",
                                   (list(joints), [0.0]*4, 0.0, 0.0, self.servo_period, 0.0, 0.0, 0),
                                   joints, timeout_s=3.0, segment_version=version)
                        with self._condition:
                            pending = self._ordered_pending
                            if (self._status == "running" and pending is not None and endpoint_due
                                    and pending["version"] == version and pending["joints"] == joints):
                                pending["joint_ack"] = True
                                self._ack_ordered_locked()
                                self._condition.notify_all()
                except _Superseded:
                    continue
                next_tick += self.servo_period
                now = self.clock()
                if next_tick < now:
                    with self._lock:
                        self._overruns += 1
                    next_tick = now + self.servo_period
        except _Cancelled:
            pass
        except Exception as exc:
            self.stop(f"ServoJ failed or outcome unknown: {exc}", failure=True)
        finally:
            self._finish_servo()
            self._producer_finished()

    def _gripper_loop(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._status != "running" or
                                             self._gripper_desired != self._gripper_last_sent)
                    if self._status != "running":
                        break
                    target = self._gripper_desired
                    # Finish an ordered joint endpoint before holding it for
                    # peripheral I/O. No policy target or gripper value is lost.
                    self._condition.wait_for(lambda: self._status != "running" or
                        self._ordered_pending is None or self._ordered_pending["joint_ack"])
                    if self._status != "running":
                        break
                try:
                    with self._actuator_io_lock:
                        began = self.clock()
                        with self._lock:
                            self._peripheral_pause_count += 1
                        try:
                            initial = self._ready_state()
                            if self._gripper_precheck is not None:
                                self._gripper_precheck(initial)
                            self._send(self._gripper_rpc, "MoveGripper",
                                       (1, target, 20, 10, 10000, 1, 0, 0.0, 0, 0), target, timeout_s=15.0)
                            completed = self._wait_gripper_completion(target, initial, began, maxtime_ms=10000)
                        finally:
                            with self._lock:
                                self._peripheral_pause_seconds += self.clock()-began
                except _Superseded:
                    continue
                except _Cancelled:
                    with self._condition:
                        self._gripper_inflight = False
                    break
                except Exception:
                    with self._condition:
                        self._gripper_inflight = False
                    raise
                else:
                    with self._condition:
                        self._gripper_inflight = False
                        self._gripper_last_succeeded = target
                        self._gripper_completion = completed
                        pending = self._ordered_pending
                        if (self._status == "running" and pending is not None
                                and pending["gripper"] == target and self._gripper_desired == target):
                            pending["gripper_ack"] = True
                            self._ack_ordered_locked()
        except Exception as exc:
            self.stop(f"MoveGripper failed or outcome unknown: {exc}", failure=True)
        finally:
            self._close_channel("gripper")
            self._producer_finished()

    def _wait_gripper_completion(self, target, initial, began, *, maxtime_ms):
        """Do not overlap accepted peripheral commands on this controller.

        Real 3.8.4.2 feedback showed a 2.29 s opening despite an immediate RPC
        return; another command at 0.18 s timed out. Both block flags returned
        early. Use measured target/contact completion and the same native
        maxtime, while the separate servo and stop channels keep running.
        """
        observed_in_progress = False
        while not self._stop_event.is_set():
            state = self._ready_state()
            stamp = state.get("received_monotonic")
            position = _vector((state["gripper_position"],), 1, "measured gripper position")[0]
            fresh = (isinstance(stamp, numbers.Real) and math.isfinite(stamp) and stamp >= began)
            settled = gripper_feedback_valid(state)
            if fresh and (state["gripper_motiondone"] == 0 or state["gripper_speed"] != 0):
                observed_in_progress = True
            at_target = state["gripper_motiondone"] == 1 and position == target
            closing_contact = (target < initial["gripper_position"]
                               and target <= position <= initial["gripper_position"]
                               and state["gripper_motiondone"] in (2, 3)
                               and (observed_in_progress or position != initial["gripper_position"]
                                    or state["gripper_motiondone"] != initial["gripper_motiondone"]))
            if fresh and settled and (at_target or closing_contact):
                return {"target_percent": target, "position_percent": position,
                        "motiondone": state["gripper_motiondone"],
                        "feedback_monotonic": float(stamp), "command_began_monotonic": began,
                        "completed_monotonic": self.clock(),
                        "meaning": "measured actuator completion, not grasp success"}
            if self.clock() - began >= maxtime_ms/1000:
                raise ServoControlError("No measured gripper completion within native maxtime")
            self._stop_event.wait(.008)
        raise _Cancelled("stopped while awaiting gripper completion")

    def _producer_finished(self):
        with self._condition:
            self._producers -= 1
            if self._producers == 0 and self._status == "stopping":
                self._status = "failed" if self._failure else "stopped"
            self._condition.notify_all()

    def _watchdog(self):
        while not self._stop_event.wait(.05):
            try:
                self._ready_state()
            except Exception as exc:
                self.stop(f"Live device state failed: {exc}", failure=True)
                return

    def _log_loop(self):
        while True:
            try:
                record = self._records.get(timeout=.05)
            except queue.Empty:
                with self._lock:
                    done = (self._stop_event.is_set() and self._producers == 0
                            and self._startup_done.is_set() and self._stop_done.is_set()
                            and self._record_inflight == 0)
                if done and self._records.empty():
                    return
                continue
            try:
                if self._callback_failure is None:
                    self.command_callback(record)
                else:
                    with self._lock:
                        self._undelivered_records.append(record)
            except Exception as exc:
                with self._lock:
                    self._callback_failure = f"{type(exc).__name__}: {exc}"
                    self._undelivered_records.append(record)
                self.stop(f"Command recording failed: {exc}", failure=True)
            finally:
                self._records.task_done()

    def stop(self, reason="operator stop", *, failure=False):
        """Prevent new sends and request StopMotion once, independently."""
        with self._condition:
            if self._reason is None:
                self._reason = str(reason)
            if failure and self._failure is None:
                self._failure = str(reason)
            if self._status not in ("stopped", "failed", "closed"):
                self._status = "stopping"
            self._stop_event.set()
            self._condition.notify_all()
            send = self._start_issued and not self._stop_sent
            if send:
                self._stop_sent = True
                self._stop_done.clear()
            elif not self._start_issued and self._producers == 0 and self._status == "stopping":
                self._status = "failed" if self._failure else "stopped"
        if send:
            try:
                code = self._invoke(self._stop_rpc, "StopMotion", (), target=None,
                                    timeout_s=2.0, gate=self._lifecycle_gate)
                with self._lock:
                    self._stop_result = code
            except Exception as exc:
                with self._lock:
                    self._stop_result = f"unconfirmed: {type(exc).__name__}: {exc}"
                    self._failure = self._failure or self._stop_result
            finally:
                self._close_channel("stop")
                self._stop_done.set()
        return self.snapshot()

    def _finish_servo(self):
        self._stop_done.wait()
        with self._lock:
            if not self._start_issued or self._end_issued:
                return
            self._end_issued = True
        try:
            code = self._invoke(self._servo_rpc, "ServoMoveEnd", (), target=None,
                                timeout_s=3.0, gate=self._lifecycle_gate)
            with self._lock:
                self._end_result = code
        except Exception as exc:
            with self._lock:
                self._end_result = f"unconfirmed: {type(exc).__name__}: {exc}"
                self._failure = self._failure or self._end_result
        finally:
            self._close_channel("servo")

    @contextmanager
    def _lifecycle_gate(self):
        with self._lock:
            yield

    def _close_channel(self, name):
        with self._lock:
            if name in self._closed_channels:
                return
            self._closed_channels.add(name)
        try:
            getattr(self, f"_{name}_rpc").close()
        except Exception as exc:
            with self._lock:
                self._failure = self._failure or f"{name} transport close failed: {exc}"

    def snapshot(self):
        with self._lock:
            return {"status": self._status, "reason": self._reason, "failure": self._failure,
                    "callback_failure": self._callback_failure, "stop_result": self._stop_result,
                    "servo_end_result": self._end_result, "servo_commands": self._servo_count,
                    "gripper_commands": self._gripper_count, "servo_overruns": self._overruns,
                    "last_issued_monotonic": self._last_issued,
                    "last_returned_monotonic": self._last_reply,
                    "last_joint_target_deg": list(self._last_joints) if self._last_joints else None,
                    "gripper_target_percent": self._gripper_desired,
                    "gripper_completion_feedback": deepcopy(self._gripper_completion),
                    "peripheral_servo_pauses": self._peripheral_pause_count,
                    "peripheral_servo_pause_seconds": self._peripheral_pause_seconds,
                    "gripper_command_inflight": self._gripper_inflight,
                    "ordered_last_token": self._ordered_token,
                    "ordered_acknowledged_token": self._ordered_ack_token,
                    "ordered_pending": deepcopy(self._ordered_pending),
                    "pending_callbacks": self._records.qsize(),
                    "inflight_commands": self._record_inflight,
                    "undelivered_callbacks": deepcopy(self._undelivered_records),
                    "startup_inflight": not self._startup_done.is_set(),
                    "workers_alive": [t.name for t in self._threads if t.is_alive()]}

    def close(self):
        """Stop once, wait for workers, and expose unfinished cleanup in snapshot."""
        self.stop("controller closed")
        deadline = time.monotonic() + 16.0
        self._startup_done.wait(timeout=max(0.0, deadline - time.monotonic()))
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            finished = self._startup_done.is_set() and not any(t.is_alive() for t in self._threads)
        if finished:
            for name in ("servo", "gripper", "stop"):
                self._close_channel(name)
            with self._lock:
                self._status = "closed"
        return self.snapshot()
