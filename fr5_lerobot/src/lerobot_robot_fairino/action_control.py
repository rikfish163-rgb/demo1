"""Bounded, single-flight FR5 position commands for supervised demonstration.

No Servo, automatic enable/reset, implicit retries, or automatic homing. This is
a low-speed commissioning transport, not a real-time policy deployment loop.
Workspace checks constrain the flange, not the complete robot/tool geometry.
"""

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .control_transport import FR5RPC
from .feedback import gripper_feedback_valid


class ControlError(RuntimeError):
    pass


@dataclass
class MotionLimits:
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    translation_step_mm: float = 1.0
    rotation_step_deg: float = 0.5
    joint_step_deg: float = 2.0
    speed_percent: float = 2.0
    gripper_step_percent: float = 5.0
    gripper_speed_percent: int = 10
    gripper_force_percent: int = 10
    command_timeout_s: float = 15.0
    heartbeat_timeout_s: float = 0.6

    def __post_init__(self):
        lo, hi = np.asarray(self.workspace_min, float), np.asarray(self.workspace_max, float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
            raise ValueError("Explicit finite workspace_min/max (three millimetre coordinates) required")
        for k, cap in (("translation_step_mm", 2), ("rotation_step_deg", 1),
                       ("joint_step_deg", 3), ("speed_percent", 5),
                       ("gripper_step_percent", 5), ("gripper_speed_percent", 20),
                       ("gripper_force_percent", 20), ("command_timeout_s", 30),
                       ("heartbeat_timeout_s", 1)):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= cap:
                raise ValueError(f"{k} must be positive and <= {cap}")


def finite_vector(values, length, name):
    v = np.asarray(values, dtype=float)
    if v.shape != (length,) or not np.isfinite(v).all():
        raise ControlError(f"Invalid {name}")
    return v


def rpc_vector(reply, length, name):
    if not isinstance(reply, (tuple, list)) or len(reply) != length + 1 or reply[0] != 0:
        raise ControlError(f"{name} failed: {reply}")
    return finite_vector(reply[1:], length, name)


def angle_error(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


class FR5ActionController:
    def __init__(self, stream, camera_fresh, limits: MotionLimits, *, host="192.168.58.2",
                 allow_motion=False, rpc=None):
        self.stream, self.camera_fresh, self.limits = stream, camera_fresh, limits
        self.allow_motion = bool(allow_motion)
        self.rpc = rpc or FR5RPC(host)
        self.armed = False
        self.error = None
        self.busy = False
        self.deadman = False
        self._heartbeat = 0.0
        self._generation = 0
        self._lock = threading.RLock()
        self._flight = threading.Lock()
        self._journal = deque(maxlen=100000)
        self._sequence = 0
        self.joint_limits = None
        self.last_result = None
        self.recording_valid = False
        self._stop_attempts = 0
        self._stop_inflight = False
        self._stop_confirmed = False

    def heartbeat(self, held):
        if type(held) is not bool:
            raise ValueError("deadman must be boolean")
        with self._lock:
            self.deadman = held
            self._heartbeat = time.monotonic()

    def ready_state(self, stationary=True):
        s = self.stream.latest(max_age_s=0.25)
        for key in ("main_code", "sub_code", "EmergencyStop", "collisionState",
                    "safety_stop0_state", "safety_stop1_state", "gripper_fault"):
            if s[key] != 0:
                raise ControlError(f"Controller reports {key}={s[key]}")
        if s["rbtEnableState"] != 1 or s["program_state"] not in (1, 2):
            raise ControlError("Robot must already be enabled; no program may be running")
        if s["tool"] != 0 or s["user"] != 0:
            raise ControlError("First adapter supports existing tool0/user0 only; no coordinate changes are made")
        tcp = finite_vector(s["tcp"], 6, "TCP")
        flange = finite_vector(s["flange"], 6, "flange")
        if (np.linalg.norm(tcp[:3] - flange[:3]) > 1 or
                np.max(np.abs(angle_error(tcp[3:], flange[3:]))) > .1):
            raise ControlError("tool0 does not match the flange")
        if stationary and (s["program_state"] != 1 or s["robot_state"] != 1 or
                           s["motion_done"] != 1 or s["mc_queue_len"] != 0 or
                           max(abs(v) for v in s["joint_speeds_deg_s"]) > 0.1):
            raise ControlError("Robot is not stopped with an empty queue")
        p = flange
        if np.any(p[:3] < self.limits.workspace_min) or np.any(p[:3] > self.limits.workspace_max):
            raise ControlError("Flange is outside the configured workspace")
        if not self.camera_fresh():
            raise ControlError("Camera is unavailable or stale")
        return s

    def arm(self, *, operator_confirmed=False):
        if not self.allow_motion:
            raise ControlError("Preview mode: process was started without --allow-motion")
        if operator_confirmed is not True:
            raise ControlError("Operator must review the workspace and controls")
        if not self._flight.acquire(blocking=False):
            raise ControlError("A command is still in flight")
        try:
            with self._lock:
                if self.armed or self._stop_inflight:
                    raise ControlError("Already armed or a stop request is still in flight")
                generation = self._generation
            self.ready_state()
            # Query only. Activation and reset are intentionally absent.
            limits = rpc_vector(self.rpc.call("GetJointSoftLimitDeg", 1), 12, "joint limits").reshape(6, 2)
            if np.any(limits[:, 0] >= limits[:, 1]):
                raise ControlError("Invalid controller joint limits")
            s = self.ready_state()
            with self._lock:
                if generation != self._generation or self._stop_inflight:
                    raise ControlError("Arming was cancelled by a stop request")
                self.joint_limits = limits
                self.armed = True
                self.error = None
                self.deadman = False
                self._generation += 1
                self._stop_attempts = 0
                self._stop_confirmed = False
                self.recording_valid = True
                self._append([*s["joints"], s["gripper_position"]], "hold", executed=False)
        finally:
            self._flight.release()

    def _append(self, target, kind, *, executed):
        self._sequence += 1
        row = dict(target=list(map(float, target)), sequence=self._sequence,
                   issued_unix=time.time(), issued_monotonic=time.monotonic(),
                   source="keyboard", kind=kind, executed=executed)
        self._journal.append(row)
        return deepcopy(row)

    def command_at(self, timestamp, *, clock_basis='host_unix'):
        from lerobot_camera_gemini.timing import command_clock
        if clock_basis not in ('host_unix','host_monotonic'):
            raise ValueError('Unknown command clock')
        with self._lock:
            for command in reversed(self._journal):
                if command_clock(command,clock_basis) <= timestamp:
                    return None if command["kind"] == "invalid" else deepcopy(command)
        return None

    def _gate(self, generation=None):
        if not self.allow_motion or not self.armed or self.error:
            raise ControlError("Motion is not armed")
        if generation is not None and generation != self._generation:
            raise ControlError("Command was cancelled")
        if not self.deadman or time.monotonic() - self._heartbeat > self.limits.heartbeat_timeout_s:
            raise ControlError("Hold Space and keep the operator page active")

    def plan_cartesian_step(self, axis, direction):
        if type(axis) is not int or axis not in range(6) or direction not in (-1, 1):
            raise ValueError("Invalid axis/direction")
        s = self.ready_state()
        pose = np.asarray(s["tcp"], float)
        pose[axis] += direction * (self.limits.translation_step_mm if axis < 3 else self.limits.rotation_step_deg)
        q = rpc_vector(self.rpc.call("GetInverseKinRef", 0, pose.tolist(), s["joints"]), 6, "IK")
        return [*q.tolist(), float(s["gripper_position"])]

    def _validate_arm_target(self, target, s):
        q = target[:6]
        if np.max(np.abs(q - s["joints"])) > self.limits.joint_step_deg:
            raise ControlError("Joint step exceeds the commissioning limit")
        if np.any(q < self.joint_limits[:, 0] + 3) or np.any(q > self.joint_limits[:, 1] - 3):
            raise ControlError("Target has insufficient joint limit margin")
        pose = rpc_vector(self.rpc.call("GetForwardKin", q.tolist()), 6, "FK")
        if np.linalg.norm(pose[:3] - s["flange"][:3]) > self.limits.translation_step_mm + .15:
            raise ControlError("Cartesian translation exceeds step limit")
        if np.max(np.abs(angle_error(pose[3:], s["flange"][3:]))) > self.limits.rotation_step_deg + .1:
            raise ControlError("Cartesian rotation exceeds step limit")
        if np.any(pose[:3] < self.limits.workspace_min) or np.any(pose[:3] > self.limits.workspace_max):
            raise ControlError("Target outside configured workspace")
        # Check intermediate IK branch continuity for the short linear segment.
        for alpha in (.5, 1.):
            point = np.asarray(s["flange"], float)
            point[:3] += alpha * (pose[:3] - point[:3])
            point[3:] += alpha * angle_error(pose[3:], point[3:])
            qi = rpc_vector(self.rpc.call("GetInverseKinRef", 0, point.tolist(), s["joints"]), 6, "path IK")
            if (np.max(np.abs(qi - s["joints"])) > self.limits.joint_step_deg or
                    np.any(qi < self.joint_limits[:, 0] + 3) or np.any(qi > self.joint_limits[:, 1] - 3)):
                raise ControlError("Linear path changes IK branch or approaches a joint limit")
        return pose

    def _action_changes(self, target, state):
        return (np.max(np.abs(target[:6] - state['joints'])) > .01,
                abs(target[6] - state['gripper_position']) > .5)

    def _gripper_complete(self, state, target, issued_monotonic):
        # Commissioning compatibility; policy deployment overrides this with
        # the device's complete/stable feedback contract.
        return abs(state['gripper_position']-target) <= 1 or state['gripper_motiondone'] == 2

    def _arm_complete(self, state, target, pose):
        return (state['motion_done'] == 1 and state['mc_queue_len'] == 0
                and np.max(np.abs(np.asarray(state['joints']) - target[:6])) < .1)

    def execute(self, action):
        """Execute exactly one bounded target. Call from a worker, never a UI thread."""
        target = finite_vector(action, 7, "action")
        if not 0 <= target[6] <= 100:
            raise ControlError("Gripper target must be 0..100 percent")
        if not self._flight.acquire(blocking=False):
            raise ControlError("Previous command still in flight; commands are never queued")
        submitted = False
        try:
            with self._lock:
                self._gate()
                generation = self._generation
            s = self.ready_state()
            arm_change, grip_change = self._action_changes(target, s)
            if arm_change and grip_change:
                raise ControlError("Commissioning mode accepts either arm or gripper, not both together")
            if not arm_change and not grip_change:
                return {"status": "hold_no_command"}
            if grip_change:
                if s["gripper_active"] == 0:
                    raise ControlError("Gripper is not active; inspect WebApp configuration first")
                if abs(target[6] - s["gripper_position"]) > self.limits.gripper_step_percent:
                    raise ControlError("Gripper step exceeds limit")
                kind = "gripper"
                method = "MoveGripper"
                args = (1, int(round(target[6])), self.limits.gripper_speed_percent,
                        self.limits.gripper_force_percent, 10000, 1, 0, 0., 0, 0)
            else:
                kind = "arm"
                pose = self._validate_arm_target(target, s)
                method = "MoveL"
                args = (target[:6].tolist(), pose.tolist(), 0, 0, self.limits.speed_percent,
                        0., 100., -1., 0, [0.] * 4, 0, 0, [0.] * 6)
            # IK queries may have taken time; reject movement by another client.
            latest = self.ready_state()
            if (np.max(np.abs(np.asarray(latest["joints"]) - s["joints"])) > .05
                    or abs(latest["gripper_position"] - s["gripper_position"]) > .5):
                raise ControlError("Robot changed while preparing the command")

            @contextmanager
            def before_send():
                nonlocal submitted, command
                # This lock remains held through HTTP endheaders(body), but
                # never while awaiting the response. Stop wins before this
                # gate or sends independently after the complete request write.
                with self._lock:
                    self._gate(generation)
                    current = self.ready_state()
                    if (np.max(np.abs(np.asarray(current["joints"]) - s["joints"])) > .05
                            or abs(current["gripper_position"] - s["gripper_position"]) > .5):
                        raise ControlError("Robot changed before the command was sent")
                    self.busy = True
                    command = self._append(target, kind, executed=True)
                    submitted = True
                    yield

            command = None
            result = self.rpc.call(method, *args, timeout_s=self.limits.command_timeout_s,
                                   before_send=before_send)
            if not submitted:
                raise ControlError("Transport did not execute the guarded send")
            if result != 0:
                raise ControlError(f"{method} returned {result}; command will not be retried")
            deadline = time.monotonic() + self.limits.command_timeout_s
            stable, previous_counter = 0, None
            while time.monotonic() < deadline:
                with self._lock:
                    self._gate(generation)
                now = self.ready_state(stationary=False)
                if kind == "arm":
                    done = self._arm_complete(now, target, pose)
                else:
                    if np.max(np.abs(np.asarray(now["joints"]) - s["joints"])) > .1:
                        raise ControlError("Arm moved during gripper operation")
                    done = self._gripper_complete(now,target[6],command['issued_monotonic'])
                    if now["gripper_motiondone"] == 3:
                        raise ControlError("Gripper reports object loss")
                if now["frame_counter"] != previous_counter:
                    stable = stable + 1 if done else 0
                    previous_counter = now["frame_counter"]
                if stable >= 3:
                    self.last_result = {"status": "completed", "command": command,
                                        "return_code": result, "final_state": now}
                    return deepcopy(self.last_result)
                time.sleep(.02)
            raise ControlError("Completion feedback timed out")
        except Exception as exc:
            if submitted:
                self.stop(f"Command failed or outcome uncertain: {exc}", retry_if_unconfirmed=True)
            raise
        finally:
            with self._lock:
                self.busy = False
            self._flight.release()

    def watchdog(self):
        """Call independently of the RPC worker, at least every 50 milliseconds."""
        with self._lock:
            if not self.armed:
                return
            expired = time.monotonic() - self._heartbeat > self.limits.heartbeat_timeout_s
            should_stop = self.busy and (expired or not self.deadman)
        if should_stop:
            self.stop("Operator released Space or heartbeat expired")
            return
        try:
            self.ready_state(stationary=False)
        except Exception as exc:
            self.stop(str(exc))

    def stop(self, reason="Operator stop", *, retry_if_unconfirmed=False):
        with self._lock:
            was_active = self.armed or self.busy
            self.armed = False
            self.deadman = False
            self._generation += 1
            if self.error is None:
                self.error = reason
            if self.recording_valid or not self._journal:
                # Invalidate future labels rather than inventing a measured
                # holding target after a cancelled or uncertain movement.
                self._sequence += 1
                self._journal.append(dict(target=None, sequence=self._sequence,
                    issued_unix=time.time(), issued_monotonic=time.monotonic(), source="keyboard", kind="invalid",
                    executed=False, reason=reason))
            self.recording_valid = False
            send_stop = (self.allow_motion and was_active and not self._stop_inflight
                         and (self._stop_attempts == 0 or
                              (retry_if_unconfirmed and not self._stop_confirmed and self._stop_attempts < 2)))
            if send_stop:
                self._stop_attempts += 1
                self._stop_inflight = True
        # Independent connection can interrupt a blocked MoveL. No gripper open,
        # reset or guessed 'stop' register is sent; hardware E-stop remains needed.
        if send_stop:
            detail = None
            confirmed = False
            try:
                code = self.rpc.call("StopMotion", timeout_s=2)
                confirmed = code == 0
                if not confirmed:
                    detail = f"StopMotion returned {code}"
            except Exception as exc:
                detail = f"StopMotion unconfirmed: {exc}"
            finally:
                with self._lock:
                    self._stop_inflight = False
                    self._stop_confirmed = confirmed
                    if detail:
                        self.error = (self.error or reason) + "; " + detail

