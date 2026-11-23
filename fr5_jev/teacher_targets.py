"""One supervised, prealigned cup episode with actual joint/gripper targets.

The stdin interface has only close, bounded vertical lift, return, release and
observe operations. Fresh scope/phase reviews are required. No arbitrary pose,
enable/reset/mode change, automatic retries, lateral motion or retract exists.
Use the lerobot environment. Construction is inert; --execute starts recording
and accepts one-shot phase requests on stdin, never moves on startup.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import shutil
import sys
import threading
import time

import numpy as np
from lerobot_camera_gemini import GeminiBridgeConfig
from lerobot_robot_fairino.configuration_fairino import FairinoFR5Config
from lerobot_robot_fairino.robot_fairino import FairinoFR5
from lerobot_robot_fairino.control_transport import FR5RPC
from lerobot_robot_fairino.continuous_capture import ContinuousCapture
from lerobot_robot_fairino.demo_writer import DemoWriter
from lerobot_robot_fairino.feedback import gripper_feedback_valid
from lerobot_camera_gemini.timing import camera_clock, command_clock

from fr5_runtime.robot import HOST, require_ready, use_shared_state_stream
from fr5_jev.observe import capture as observe
from fr5_jev.durable_storage import DurableStorage


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def angle_delta(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def review_age(review):
    if 'reviewed_at_monotonic' in review:
        return time.monotonic() - float(review['reviewed_at_monotonic'])
    return time.time() - review.get('reviewed_at_unix', 0)


def validate_scope(scope):
    if (scope.get("robot_ip") != HOST or scope.get("max_lift_mm") != 10
            or scope.get("closing_targets") != [90, 80, 76]
            or scope.get("closing_speed") != 5 or scope.get("force") != 10
            or scope.get("move_speed") != 1 or scope.get("return_to_original_support") is not True
            or scope.get("only_prealigned_instance") is not True):
        raise ValueError("Scope differs from the bounded teacher episode")
    if not 0 <= review_age(scope) <= 300:
        raise ValueError("Scope review is stale")
    if type(scope.get('max_session_s',900)) is not int or scope.get('max_session_s',900) not in (900,1800):
        raise ValueError('Session duration must be explicitly bounded to900 or1800 seconds')
    for flag in ("same_cup_and_contact_pose", "both_fingers_outside_cup",
                 "vertical_swept_path_clear", "no_visible_hands"):
        if scope.get(flag) is not True:
            raise ValueError(f"Missing current evidence: {flag}")
    pose = np.asarray(scope.get("initial_tcp"), float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Invalid scope start pose")
    if not scope.get("authorization_source") or not scope.get("evidence"):
        raise ValueError("Scope needs authorization and evidence provenance")


def check_state(s, origin, *, moving=False, held_position=None, gripper_reference=None):
    require_ready(s, stationary=not moving)
    if s["robot_mode"] != 1 or s["robot_state"] in (3, 4):
        raise RuntimeError("Unexpected controller mode/state")
    if (s["gripper_active"] != 1 or s["gripper_fault"] != 0 or s["gripper_fault_id"] != 0):
        raise RuntimeError("Gripper unavailable or faulted")
    delta = np.asarray(s["tcp"]) - origin
    if (max(abs(delta[:2])) > .15 or max(abs(angle_delta(s["tcp"][3:], origin[3:]))) > .15):
        raise RuntimeError("Departed from reviewed vertical corridor")
    if gripper_reference is None:
        if not -.05 <= delta[2] <= 10.15:
            raise RuntimeError("Departed from reviewed vertical height bounds")
    else:
        # No arm target is issued during this mode. Apply the stationary-pose
        # monitor, rather than interpreting gripper reaction as a commanded
        # downward path. Completion still requires the original support height.
        reference = np.asarray(gripper_reference, float)
        if (moving or reference.shape != (6,) or not np.isfinite(reference).all()
                or abs(reference[2] - origin[2]) > .05
                or np.max(abs(np.asarray(s['tcp'][:3]) - reference[:3])) > .15):
            raise RuntimeError("Arm departed from stationary gripper pose")
    if s["tcp_speed"][0] > 25:
        raise RuntimeError("Measured speed exceeds monitor bound")
    if held_position is not None and (
        not gripper_feedback_valid(s) or s["gripper_motiondone"] != 2
        or abs(s["gripper_position"] - held_position) > 1
    ):
        raise RuntimeError("Stable holding feedback lost")
    return float(delta[2])


class TeacherSession:
    def __init__(self, directory, *, robot=None, rpc=None):
        self.root = Path(directory).resolve()
        raw = (self.root / "scope.json").read_bytes()
        self.scope = json.loads(raw)
        self.validate_scope(self.scope)
        self.scope_sha = hashlib.sha256(raw).hexdigest()
        self.config = FairinoFR5Config(camera=GeminiBridgeConfig(
            url="http://127.0.0.1:8765/snapshot_fast.npz", fps=6))
        self.robot = robot or FairinoFR5(self.config)
        self.rpc = rpc or FR5RPC(HOST)
        self.origin = np.asarray(self.scope["initial_tcp"], float)
        self.expected_start_pose = self.origin.copy()
        self.lock = threading.RLock()
        self.journal = []
        self.command_sequence = 0
        self.failed = None
        self.interrupted = threading.Event()
        self.holding_position = None
        self.peak_height = 0.
        self.returned = False
        self.released = False
        self.closed = False
        self.writer = self.capture = None
        self.storage = None
        self.target = None
        self.phase = "initial_hold"
        self.decision_id = "scope.json"
        self.active_command = False
        self.start_mono = time.monotonic()
        self.max_session_s = self.scope.get('max_session_s',900)
        self.last_read_state = None
        self.gripper_velocity_since = None
        self.policy_name = 'current_codex_chat+jev'
        self.task_description = "Grasp the prealigned cup, lift it vertically 10 mm, return and release"

    @staticmethod
    def validate_scope(scope):
        validate_scope(scope)

    def arm_requires_hold(self):
        return True

    def check_command_permission(self, kind, initial, destination_pose):
        if kind == "arm" and (self.holding_position is None or self.released):
            raise RuntimeError("Arm motion requires stable grasp in this episode")

    def command_timeouts(self, kind, initial, destination_pose):
        return 8., 15.

    def check_controller_state(self, s, **kwargs):
        return check_state(s, self.origin, **kwargs)

    def check_segment_progress(self, s, initial, destination_pose):
        delta = np.asarray(s["tcp"]) - initial["tcp"]
        dz = destination_pose[2] - initial["tcp"][2]
        if not min(0., dz) - .08 <= delta[2] <= max(0., dz) + .08:
            raise RuntimeError("Wrong direction or segment overshoot")

    def state(self, *, moving=False, held=False, gripper_reference=None):
        if self.failed or self.interrupted.is_set() or (self.root / "STOP").exists():
            raise RuntimeError(f"Episode stopped: {self.failed or 'operator interrupt'}")
        if time.monotonic() - self.start_mono > self.max_session_s:
            raise TimeoutError("Teacher session expired")
        if self.capture is not None:
            self.capture.assert_healthy()
        s = self.robot.stream.latest(max_age_s=.25)
        self.last_read_state = deepcopy(s)
        self.check_controller_state(s, moving=moving,
                    held_position=self.holding_position if held else None,
                    gripper_reference=gripper_reference)
        if not moving:
            if self.target is not None and np.max(abs(np.asarray(s["joints"]) - self.target[:6])) > .05:
                raise RuntimeError("Unexpected joint displacement or another command source")
            speed = max(abs(v) for v in s["joint_speeds_deg_s"])
            # Apply the existing bounded debounce to all stationary holds,
            # including post-motion visual checkpoints. Controller done/queue,
            # joint displacement, TCP corridor and grasp checks above remain
            # immediate. Never debounce a larger velocity or a controller fault.
            if speed > .2:
                raise RuntimeError("Joint velocity spike during stationary hold")
            if speed > .1:
                if self.gripper_velocity_since is None:
                    self.gripper_velocity_since = s.get('received_monotonic',s['timestamp'])
                if s.get('received_monotonic',s['timestamp']) - self.gripper_velocity_since >= .1:
                    raise RuntimeError("Sustained joint velocity during stationary hold")
            else:
                self.gripper_velocity_since = None
        else:
            self.gripper_velocity_since = None
        return s

    def append(self, target, kind, executed):
        with self.lock:
            self.command_sequence += 1
            command = {"sequence": self.command_sequence, "issued_unix": time.time(),
                       "issued_monotonic":time.monotonic(),
                       "source": "agent_teacher", "target": list(map(float, target)),
                       "kind": kind, "executed": executed,
                       "teacher": {"policy": self.policy_name, "phase": self.phase,
                                   "decision_id": self.decision_id}}
            self.journal.append(command)
            self.target = command["target"]
            return deepcopy(command)

    def sample(self, sample):
        captured,basis = camera_clock(sample['camera']['metadata'])
        with self.lock:
            command = next((deepcopy(c) for c in reversed(self.journal) if command_clock(c,basis) <= captured), None)
        if command is not None:
            return self.writer.add(sample, command)

    def observe_bounded(self, directory, *, observer=observe):
        check = getattr(self.storage, "ensure_local_capacity", None)
        if callable(check):
            check(reserve_bytes=24 * 1024**2)
        result = observer(directory)
        if callable(check):
            check()
        return result

    def prepare_storage(self):
        """Verify storage before any session connects or claims its directory."""
        self.storage = None
        from fr5_jev.durable_storage import LOCAL_BASE
        from fr5_jev.stream_storage import LOCAL_STREAM_BASE
        if self.root.parent == LOCAL_STREAM_BASE:
            from fr5_jev.stream_storage import StreamStorage
            required_bytes = 512 * 1024**2 + self.max_session_s * 6 * 512 * 1024
            self.storage = StreamStorage.start_for_episode(self.root, self.scope_sha, required_bytes=required_bytes)
            return
        if self.root.parent == LOCAL_BASE:
            raise RuntimeError("Direct GVFS recording failed the real 6 Hz capture test; physical recording is blocked")
        required_bytes = 512 * 1024**2 + self.max_session_s * 6 * 512 * 1024
        self.storage = DurableStorage.preflight(self.root, self.scope_sha, required_bytes=required_bytes)

    def start(self):
        self.prepare_storage()
        # Directory claims survive failures. A previous episode can never be replayed.
        (self.root / "SESSION_STARTED").open("x").close()
        self.robot.connect()
        s = self.state()
        if np.max(np.abs(np.asarray(s["tcp"]) - self.expected_start_pose)) > .1:
            raise RuntimeError("Start differs from the freshly reviewed pose")
        if not gripper_feedback_valid(s) or s["gripper_position"] != 100 or s["gripper_motiondone"] != 1:
            raise RuntimeError("Expected already active, stationary, fully open gripper")
        raw = self.rpc.call("GetJointSoftLimitDeg", 1)
        if len(raw) != 13 or raw[0] != 0:
            raise RuntimeError("Joint limits unavailable")
        self.limits = np.asarray(raw[1:]).reshape(6, 2)
        if self.rpc.call("GetGripperConfig") != [0, 1, 3, 0, 0]:
            raise RuntimeError("Unexpected gripper configuration")
        self.begin_recording(s, 100.)
        return s

    def begin_recording(self, state, gripper_target):
        """Called only after the concrete session's startup state is verified."""
        if self.storage is None:
            raise RuntimeError("Verified durable recording storage is required before recording")
        # Reserve a conservative raw RGB-D budget before any action is sent.
        required_bytes = 512 * 1024**2 + self.max_session_s * 6 * 512 * 1024
        frame_spool = getattr(self.storage, "frame_spool", None)
        local_required = (144 * 1024**2 + frame_spool.max_bytes) if frame_spool else required_bytes
        if shutil.disk_usage(self.root).free < local_required:
            raise RuntimeError("Insufficient free capture storage for the bounded session")
        self.writer = DemoWriter(self.root / "lerobot", fps=6,
            task=self.task_description, journal_only=True,
            sync_writes=True,
            frame_spool=frame_spool,
            external_capture_root=self.root / "continuous",
            metadata={"scope_sha256": self.scope_sha, "scope": self.scope,
                      "image_width": 640, "image_height": 360,
                      "action_semantics": "actual MoveL endpoint joint targets plus explicit gripper targets",
                      "prealigned_only": self.scope.get("only_prealigned_instance", True),
                      "independent_astra_api_calls": 0}).start()
        # This is an explicit teacher hold instruction, not a fabricated move.
        initial_hold = self.append([*state["joints"], gripper_target], "hold", False)
        self.wait_for_initial_observation(initial_hold)
        capture = ContinuousCapture(self.robot, self.root / "continuous", task=self.writer.task,
                                    fps=6, sample_callback=self.sample, sync_writes=True,
                                    frame_spool=frame_spool,
                                    metadata={"external_command_journal": str(self.root / 'target-command-journal.json'),
                                              "external_command_reports": "command-NNNN.json; actual guarded send"})
        self.capture = capture
        capture.start()
        self.capture.phase("initial_hold")

    def wait_for_initial_observation(self, command, timeout_s=2.):
        """Start recording only after the explicit initial hold exists in time.

        The camera may return a cached pre-start frame. It has no causal action
        label, so wait for a newer source observation while the arm stays still.
        These startup observations are not declared recorded demonstration frames.
        """
        started, skipped = time.monotonic(), 0
        while time.monotonic() - started <= timeout_s:
            self.state(held=self.holding_position is not None)
            sample = self.robot.get_sample()
            captured, basis = camera_clock(sample['camera']['metadata'])
            if captured >= command_clock(command, basis):
                save(self.root / 'recording-start.json', {
                    'first_eligible_source_time': captured, 'timing_clock': basis,
                    'initial_hold_issue_time': command_clock(command, basis),
                    'unrecorded_pre_hold_observations': skipped,
                    'physical_commands_sent': 0,
                })
                return
            skipped += 1
            time.sleep(.02)
        raise TimeoutError('No camera observation newer than initial hold; no action dispatched')

    @staticmethod
    def phase_review_record(decision):
        return decision.get("current_chat_llm_review") or decision.get("workflow_review", {})

    def review_phase(self, phase, decision_id):
        if not re.fullmatch(r"[a-z0-9_-]+\.json", decision_id):
            raise ValueError("Decision must be a local episode JSON filename")
        decision = json.loads((self.root / decision_id).read_text())
        review = self.phase_review_record(decision)
        if not decision.get("current_chat_llm_review"):
            from .continuous_workflow import validate_automatic_review
            validate_automatic_review(self, phase, decision)
        if (review.get("phase") != phase or review.get("proceed") is not True
                or decision.get("scope_sha256") != self.scope_sha
                or not 0 <= review_age(review) <= 300):
            raise RuntimeError("Missing fresh phase review for this scope")
        self.state()
        self.phase, self.decision_id = phase, decision_id
        self.capture.phase(phase, teacher={"policy": "current_codex_chat+jev", "decision_id": decision_id})

    def decide(self, skill, filename):
        """One Jev phase selection; never dispatches an action or grants review."""
        from .decision import make_request, call_jev, validate_answer, eligible_skills, DecisionError
        from .policy import gate_table, assess
        from .memory import retrieve
        phase_skill = {
            "approach_clear": "transfer_empty_tool",
            "approach_near": "align_xy",
            "approach_contact": "descend_pregrasp",
            "close": "close_gripper",
            "lift_probe": "lift_verify",
            "lift_full": "lift_verify",
            "return": "return_place",
            "release": "release_gripper",
            "retreat_probe": "retreat_tool",
            "retreat_clear": "retreat_tool",
            "retreat_high": "retreat_tool",
        }
        phase = skill if skill in phase_skill else None
        skill = phase_skill.get(skill, skill)
        if skill not in (
            "transfer_empty_tool", "align_xy", "descend_pregrasp",
            "close_gripper", "lift_verify", "return_place",
            "release_gripper", "retreat_tool",
        ) or not re.fullmatch(r"[a-z0-9_-]+\.json", filename):
            raise ValueError("Invalid phase decision request")
        path = self.root / filename
        if path.exists():
            raise RuntimeError("Decision cannot be overwritten")
        s = self.state(held=skill in ("lift_verify", "return_place", "release_gripper"))
        if skill == "close_gripper" and (self.closed or s["gripper_position"] != 100):
            raise RuntimeError("Closing candidate not eligible")
        if skill in ("lift_verify", "return_place", "release_gripper") and self.holding_position is None:
            raise RuntimeError("Held-object candidate not eligible")
        if skill == "release_gripper" and not self.returned:
            raise RuntimeError("Release candidate requires original-support return")
        if skill == "retreat_tool" and not self.released:
            raise RuntimeError("Retreat candidate requires a released supported cup")
        action_ids = [
            action["id"] for action in self.scope.get("manual_action_plan", {}).get("actions", [])
            if action.get("phase") == phase
        ]
        meta = self.robot.last_sample["camera"]["metadata"]
        state = {"schema": "fr5-cup-shadow-v1", "source": "live_readonly", "robot_id": "J1",
                 "robot_ip": HOST, "captured_at_unix": time.time(),
                 "robot": {"stationary": True, "fault_free": True, "enabled": True,
                           "gripper_position": s["gripper_position"], "gripper_motiondone": s["gripper_motiondone"]},
                 "camera": {"serial": meta["source_camera_serial"], "available": True,
                            "expected_device_and_fresh": True, "capture_time_unix": meta["source_timestamp_unix"]},
                 "candidates": {skill: {"validated": True, "preconditions_met": True,
                                       "evidence": str(self.root / "scope.json"),
                                       "precomputed_actions": action_ids}},
                 "facts": {"task": self.writer.task, "prealigned_only": self.scope.get("only_prealigned_instance", True),
                           "current_scope": self.scope, "stable_gripper_status2": s["gripper_motiondone"] == 2,
                           "actual_target_commands_recorded": True,
                           "requested_phase": phase,
                           "precomputed_action_ids": action_ids,
                           "manual_driven_action_plan": bool(action_ids),
                           "no_automatic_dispatch_from_decision": True},
                 "retrieved_examples": retrieve(["cup", "grasp", "lift", "gripper_feedback"])}
        request = make_request(state, provider="emrank")
        policies = gate_table(eligible_skills(state), evidence_complete=True, matched_successes=0, recent_failures=0)
        result = {"scope_sha256": self.scope_sha, "state": state, "request": request,
                  "policies_frozen_before_inference": policies, "physical_execution": False,
                  "independent_astra_api_calls": 0}
        with path.open("x") as f:
            json.dump(result, f, ensure_ascii=False)
        try:
            response, latency = call_jev(request, Path('/home/dianrobot/.config/emrank/api-key'), provider="emrank")
            answer = validate_answer(request, response, provider="emrank")
        except DecisionError as exc:
            # A read-only choice request cannot make a physical outcome
            # uncertain. Preserve the failed request; require a separate fresh
            # LLM phase review rather than retrying or dispatching any action.
            result.update(decision_error=str(exc),route='needs_llm',reviewed_by_current_chat=False)
            save(path,result)
            return {'decision_id':filename,'decision_error':str(exc),'route':'needs_llm',
                    'physical_execution':False,'retry_performed':False}
        gate = assess(answer, policies[answer["choice"]], hard_checks={
            "specific_authorization": True, "preflight_passed": True, "scene_reviewed": True,
            "bounded_executor_verified": True, "onsite_confirmation": True})
        result.update(response=response, answer=answer, latency_ms=latency, gate=gate,
                      physical_execution=False, reviewed_by_current_chat=False)
        save(path, result)
        return {"decision_id": filename, "answer": answer, "latency_ms": latency, "gate": gate}

    def stop_once(self, reason, *, arm_in_flight):
        # This lock is independent of the guarded-dispatch lock: stopping must
        # not wait for a command worker to finish writing or reading its reply.
        stop_lock = self.__dict__.setdefault("_stop_once_lock", threading.Lock())
        reason_text = str(reason) or type(reason).__name__
        with stop_lock:
            if not self.failed:
                self.failed = reason_text
            failure = self.failed
            if not hasattr(self, 'stop_diagnostic') and self.__dict__.get('last_read_state') is not None:
                self.stop_diagnostic = {
                    'reason': failure, 'state': deepcopy(self.last_read_state),
                    'target': deepcopy(self.__dict__.get('target')),
                    'phase': self.__dict__.get('phase'),
                    'command_sequence': self.__dict__.get('command_sequence'),
                    'captured_unix': time.time(), 'captured_monotonic': time.monotonic(),
                }
            self.__dict__.setdefault("stop_persistence_errors", [])
            send_stop = arm_in_flight and not self.__dict__.get("_stop_attempted", False)
            if send_stop:
                self._stop_attempted = True
                self.stop_result = {"status": "attempted", "reason": failure}

        result_record = None
        stop_interrupt = None
        if send_stop:
            try:
                result = self.rpc.call("StopMotion", timeout_s=2)
                result_record = {"return_code": result, "reason": failure}
            except BaseException as exc:
                result_record = {"unconfirmed": f"{type(exc).__name__}: {exc}", "reason": failure}
                if not isinstance(exc, Exception):
                    stop_interrupt = exc
            with stop_lock:
                self.stop_result = result_record

        # The safety request above must never depend on disk availability.
        try:
            (self.root / "STOP").write_text(failure + "\n")
        except Exception as exc:
            with stop_lock:
                self.stop_persistence_errors.append({"file": "STOP", "error": f"{type(exc).__name__}: {exc}"})
        if result_record is not None:
            try:
                save(self.root / "stop-result.json", result_record)
            except Exception as exc:
                with stop_lock:
                    self.stop_persistence_errors.append({"file": "stop-result.json", "error": f"{type(exc).__name__}: {exc}"})
        if self.__dict__.get('stop_diagnostic') is not None:
            try:
                save(self.root / 'failure-state.json', self.stop_diagnostic)
            except Exception as exc:
                with stop_lock:
                    self.stop_persistence_errors.append({'file':'failure-state.json',
                                                        'error':f'{type(exc).__name__}: {exc}'})
        if stop_interrupt is not None:
            raise stop_interrupt

    def send(self, method, args, target, *, kind, destination_pose=None, action_metadata=None):
        if self.storage is None:
            raise RuntimeError("Verified durable recording storage is required before any command")
        requires_hold = kind == "arm" and self.arm_requires_hold()
        initial = self.state(held=requires_hold)
        rpc_timeout, completion_timeout = self.command_timeouts(kind, initial, destination_pose)
        if self.max_session_s - (time.monotonic()-self.start_mono) < completion_timeout + 5:
            raise TimeoutError('Insufficient session time to dispatch and confirm another command')
        self.check_command_permission(kind, initial, destination_pose)
        if kind == "gripper" and abs(initial["tcp"][2] - self.origin[2]) > .05:
            raise RuntimeError("Gripper may only change at original support height")
        if kind == "gripper" and target[6] < initial["gripper_position"] and not 0 < initial["gripper_position"] - target[6] <= 10:
            raise RuntimeError("Closing step exceeds reviewed bound")
        path = self.root / f"command-{self.command_sequence + 1:04d}.json"
        log = {"status": "prepared", "method": method, "args": args, "initial_state": initial,
               "command_kind": kind, "intended_target": list(map(float, target)),
               "phase": self.phase, "decision_id": self.decision_id, "samples": []}
        if action_metadata is not None:
            log["planned_action"] = action_metadata
        prepared_snapshot = deepcopy(log)
        with path.open("x") as f:
            json.dump(log, f)
        outcome = {}
        submitted = threading.Event()
        cancelled = threading.Event()
        dispatch_gate = threading.Lock()
        persistence_lock = threading.Lock()

        @contextmanager
        def before_send():
            # Persist the prepared wire target without holding the capture or
            # dispatch locks. The command is not yet an issued instruction.
            frame_spool = getattr(self.storage, "frame_spool", None)
            watermark = frame_spool.watermark() if frame_spool is not None else None
            files = [path]
            if frame_spool is not None:
                files.extend((self.root / "continuous/events.jsonl", self.root / self.decision_id))
            self.storage.sync_files(*files)
            if frame_spool is not None:
                frame_spool.wait_for_ack(watermark)
            if cancelled.is_set():
                raise RuntimeError("Command cancelled while recording storage was unavailable")
            checked = self.state(held=requires_hold)
            self.check_command_permission(kind, checked, destination_pose)
            checked_monotonic = time.monotonic()
            attempted_snapshot = journal_snapshot = None
            try:
                with dispatch_gate:
                    if cancelled.is_set():
                        raise RuntimeError("Command cancelled while recording storage was unavailable")
                    with self.lock:
                        if (self.failed or self.interrupted.is_set() or
                                time.monotonic() - checked_monotonic > .25):
                            raise RuntimeError("Dispatch check expired or the session stopped")
                        # This is an in-memory stream read, not a filesystem
                        # check; state() above already checked STOP and capture.
                        latest = self.robot.stream.latest(max_age_s=.25)
                        self.last_read_state = deepcopy(latest)
                        self.check_controller_state(latest, moving=False,
                            held_position=self.holding_position if requires_hold else None,
                            gripper_reference=None)
                        if (np.max(abs(np.asarray(latest["tcp"]) - initial["tcp"])) > .05
                                or abs(latest["gripper_position"] - initial["gripper_position"]) > 1):
                            raise RuntimeError("State changed while verifying command persistence")
                        if self.target is not None and np.max(abs(np.asarray(latest["joints"]) - self.target[:6])) > .05:
                            raise RuntimeError("Unexpected joint displacement or another command source")
                        speed = max(abs(v) for v in latest["joint_speeds_deg_s"])
                        if speed > .2:
                            raise RuntimeError("Joint velocity spike during stationary hold")
                        if speed > .1:
                            received = latest.get('received_monotonic', latest['timestamp'])
                            if self.gripper_velocity_since is None:
                                self.gripper_velocity_since = received
                            if received - self.gripper_velocity_since >= .1:
                                raise RuntimeError("Sustained joint velocity during stationary hold")
                        else:
                            self.gripper_velocity_since = None
                        # Timestamp and snapshot immediately before the bounded
                        # wire write. The sample callback can resume after it.
                        command = self.append(target, kind, True)
                        log["command"], log["status"] = command, "attempted_once"
                        attempted_snapshot = {**prepared_snapshot, "command": command,
                                              "status": "attempted_once"}
                        journal_snapshot = deepcopy(self.journal)
                        submitted.set()
                        yield
            finally:
                if submitted.is_set():
                    journal_path = self.root / "target-command-journal.json"
                    # Serialize competing status updates without blocking the
                    # frame callback or the independent stop request.
                    with persistence_lock:
                        files = []
                        if not cancelled.is_set():
                            save(path, attempted_snapshot)
                            files.append(path)
                        save(journal_path, journal_snapshot)
                        files.append(journal_path)
                        self.storage.sync_files(*files)

        def worker():
            try:
                outcome["return_code"] = self.rpc.call(method, *args, timeout_s=rpc_timeout, before_send=before_send)
            except BaseException as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        self.active_command = True
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        start = time.monotonic()
        stable, previous = 0, None
        try:
            while time.monotonic() - start < completion_timeout:
                elapsed = time.monotonic() - start
                s = self.state(moving=kind == "arm", held=requires_hold,
                               gripper_reference=initial["tcp"] if kind == "gripper" else None)
                log["samples"].append(s)
                if "error" in outcome or outcome.get("return_code", 0) != 0:
                    raise RuntimeError(f"RPC failed or uncertain: {outcome}")
                if kind == "arm":
                    self.check_segment_progress(s, initial, destination_pose)
                    done = (s["motion_done"] == 1 and s["mc_queue_len"] == 0
                            and np.max(abs(np.asarray(s["tcp"][:3]) - destination_pose[:3])) <= .05
                            and np.max(abs(np.asarray(s["joints"]) - target[:6])) < .05)
                else:
                    if np.max(abs(np.asarray(s["tcp"]) - initial["tcp"])) > .15:
                        raise RuntimeError("Unexpected arm motion during gripper command")
                    busy_zero = s["gripper_motiondone"] == 0 and all(s[k] == 0 for k in
                        ("gripper_position", "gripper_speed", "gripper_current"))
                    # Storage preparation precedes the actual wire command.
                    # Its latency must not consume the gripper feedback grace.
                    issued = log.get("command", {}).get("issued_monotonic")
                    dispatched_elapsed = (time.monotonic() - issued
                                          if issued is not None else None)
                    if busy_zero and (dispatched_elapsed is None or dispatched_elapsed > 2):
                        raise RuntimeError("Invalid gripper feedback persisted")
                    if s["gripper_motiondone"] == 3:
                        raise RuntimeError("Gripper reports unstable/lost contact")
                    if not busy_zero and not min(target[6], initial["gripper_position"]) - 1 <= s["gripper_position"] <= max(target[6], initial["gripper_position"]) + 1:
                        raise RuntimeError("Gripper feedback outside commanded interval")
                    done = gripper_feedback_valid(s) and (
                        abs(s["gripper_position"] - target[6]) <= 1 or s["gripper_motiondone"] == 2)
                    done = done and abs(s['tcp'][2] - self.origin[2]) <= .05
                    done = done and max(abs(v) for v in s['joint_speeds_deg_s']) <= .1
                if s["frame_counter"] != previous:
                    stable = stable + 1 if done else 0
                    previous = s["frame_counter"]
                if stable >= 5 and outcome.get("return_code") == 0 and elapsed >= .8:
                    require_ready(s)
                    log.update(status="completed", final_state=s, outcome=outcome, completed_at_unix=time.time(),
                               completed_monotonic=time.monotonic())
                    save(path, log)
                    self.storage.sync_files(path)
                    return s
                time.sleep(.025)
            raise TimeoutError("Completion feedback timed out")
        except BaseException as exc:
            cancelled.set()
            # This gate spans only the bounded socket write, never storage I/O.
            with dispatch_gate:
                was_submitted = submitted.is_set()
            if was_submitted:
                self.stop_once(exc, arm_in_flight=kind == "arm")
            log.update(status="failed_or_uncertain", error=str(exc), outcome=outcome,
                       rejected_or_last_state=self.last_read_state)
            with persistence_lock:
                save(path, log)
            raise
        finally:
            thread.join(timeout=8.5)
            self.active_command = False

    def close_gripper(self):
        if self.closed or self.peak_height > 0 or self.released:
            raise RuntimeError("Closing phase cannot be replayed")
        self.closed = True
        for target in self.scope["closing_targets"]:
            s = self.state()
            command = [*self.target[:6], target]
            s = self.send("MoveGripper", (1, target, 5, 10, 10000, 1, 0, 0., 0, 0), command, kind="gripper")
            if s["gripper_motiondone"] == 2:
                self.holding_position = s["gripper_position"]
                break
        if self.holding_position is None:
            raise RuntimeError("No stable contact within bounded closing targets")
        # At the supported contact pose, acquire a settled reference before
        # applying the unchanged +/-1 held-object monitor to any arm motion.
        # The first status2 packet can precede the final encoder settling.
        initial_contact = self.holding_position
        self.holding_position = None
        samples, window = [], []
        start = time.monotonic()
        try:
            while time.monotonic() - start < 6:
                s = self.state()
                now = time.monotonic()
                samples.append(s)
                if abs(s['tcp'][2] - self.origin[2]) > .05:
                    raise RuntimeError('Contact settling requires original support height')
                if s['gripper_motiondone'] == 3:
                    raise RuntimeError('Object loss reported during supported contact settling')
                valid = gripper_feedback_valid(s) and s['gripper_motiondone'] == 2
                if valid and not self.target[6] <= s['gripper_position'] <= initial_contact + 1:
                    raise RuntimeError('Contact position outside commanded supported settling range')
                if not valid:
                    window = []
                else:
                    window.append((now, s['gripper_position']))
                    while window and max(x[1] for x in window)-min(x[1] for x in window) > 1:
                        window.pop(0)
                    if window and now-window[0][0] >= 3:
                        self.holding_position = float(np.median([x[1] for x in window]))
                        self.state(held=True)
                        save(self.root / 'hold-verification.json', {
                            'samples': samples, 'status': 'stable_status2',
                            'initial_contact_position': initial_contact,
                            'settled_reference_position': self.holding_position,
                            'stable_window_s': now-window[0][0],
                            'reference_acquired_at_support_before_any_lift': True})
                        return s
                time.sleep(.05)
            raise RuntimeError('Contact did not stabilize at support within6seconds')
        except BaseException as exc:
            save(self.root / 'hold-verification.json', {
                'samples': samples, 'status': 'failed_before_lift', 'error': str(exc),
                'rejected_or_last_state': self.last_read_state,
                'initial_contact_position': initial_contact})
            raise

    def plan_height(self, height):
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not 0 <= height <= 10:
            raise ValueError("Height must be 0..10mm above the reviewed support pose")
        s = self.state(held=True)
        current_height = s["tcp"][2] - self.origin[2]
        if not .05 < abs(height - current_height) <= 2.05:
            raise ValueError("Each MoveL segment must be <=2mm")
        if self.holding_position is None:
            raise RuntimeError("No grasp in this episode")
        target_pose = self.origin.copy()
        target_pose[2] += height
        reference = s["joints"]
        checks = []
        for alpha in (.5, 1.):
            pose = np.asarray(s["tcp"]) + alpha * (target_pose - s["tcp"])
            result = self.rpc.call("GetInverseKinRef", 0, pose.tolist(), reference)
            if len(result) != 7 or result[0] != 0:
                raise RuntimeError("IK failed")
            q = np.asarray(result[1:])
            if (not np.isfinite(q).all() or np.any(q < self.limits[:, 0] + 3)
                    or np.any(q > self.limits[:, 1] - 3)
                    or np.max(abs(q - s["joints"])) > 1
                    or abs(math.sin(math.radians(q[4]))) < .2):
                raise RuntimeError("Joint/branch/singularity margin failed")
            fk = self.rpc.call("GetForwardKin", q.tolist())
            if (len(fk) != 7 or fk[0] != 0 or not np.isfinite(fk).all()
                    or np.linalg.norm(np.asarray(fk[1:4]) - pose[:3]) > .2
                    or np.max(abs(angle_delta(fk[4:], pose[3:]))) > .05):
                raise RuntimeError("IK/FK consistency failed")
            checks.append({"pose": pose.tolist(), "q": q.tolist()})
            reference = q.tolist()
        return s, target_pose, reference, checks

    def height(self, target):
        s, pose, q, checks = self.plan_height(target)
        save(self.root / f"path-{self.command_sequence + 1:04d}.json", {"checks": checks})
        action = [*q, self.target[6]]
        s = self.send("MoveL", (q, pose.tolist(), 0, 0, 1., 0., 100., -1., 0, [0.] * 4, 0, 0, [0.] * 6),
                      action, kind="arm", destination_pose=pose)
        self.peak_height = max(self.peak_height, target)
        return s

    def lift(self, target):
        if self.returned or self.released or type(target) is not int or target not in (1, 10):
            raise RuntimeError("Only first 1mm probe and then 10mm lift are allowed")
        start = self.state(held=True)["tcp"][2] - self.origin[2]
        if target <= start + .1:
            raise RuntimeError("Lift target was already reached")
        cursor = int(round(start))
        while cursor < target:
            cursor = min(cursor + 2, target)
            s = self.height(cursor)
        return s

    def setdown(self):
        if self.returned or self.released or self.peak_height < .9:
            raise RuntimeError("Return requires a new completed lift")
        cursor = int(round(self.state(held=True)["tcp"][2] - self.origin[2]))
        while cursor > 0:
            cursor = max(0, cursor - (1 if cursor <= 2 else 2))
            s = self.height(cursor)
        if abs(s["tcp"][2] - self.origin[2]) > .05:
            raise RuntimeError("Did not return to the initial support height")
        self.returned = True
        return s

    def release(self):
        if not self.returned or self.released:
            raise RuntimeError("Release only after this episode returned to support")
        s = self.state(held=True)
        target = [*self.target[:6], 100.]
        s = self.send("MoveGripper", (1, 100, 10, 10, 10000, 1, 0, 0., 0, 0), target, kind="gripper")
        if s["gripper_motiondone"] != 1 or s["gripper_position"] != 100:
            raise RuntimeError("Release not confirmed")
        self.released = True
        self.holding_position = None
        return s

    def final_summary_fields(self):
        """Subclass-owned fields must be fixed before the final snapshot."""
        return {}

    def finish(self, outcome="interrupted", evidence=None):
        if outcome == "success" and (not self.released or self.failed or self.peak_height < 9.9 or not evidence):
            raise RuntimeError("Success needs full lift/return/release plus visual review evidence")
        # Persist issued targets before any optional export/finalization work.
        save(self.root / "target-command-journal.json", self.journal)
        raw_report = self.capture.finish(outcome, evidence=evidence) if self.capture else None
        if raw_report and not raw_report["capture_complete"]:
            outcome = "interrupted"
        report = self.writer.finish(outcome) if self.writer and self.writer._started else None
        save(self.root / "target-command-journal.json", self.journal)
        summary = {"outcome": outcome, "scope_sha256": self.scope_sha, "peak_height_mm": self.peak_height,
                   "released": self.released, "failed": self.failed, "commands": self.command_sequence - 1,
                   "dataset": report, "continuous_capture": raw_report, "evidence": evidence or []}
        summary.update(self.final_summary_fields())
        save(self.root / "summary.json", summary)
        if self.storage is not None:
            files = [self.root / "target-command-journal.json", self.root / "summary.json"]
            if raw_report is not None:
                files.append(self.root / "continuous/report.json")
            if report is not None:
                files.append(self.root / "lerobot/meta/demo.json")
            frame_spool = getattr(self.storage, "frame_spool", None)
            if frame_spool is not None:
                # Do not publish a partial remote summary: close the stream,
                # then copy every final control/evidence file as a snapshot.
                files = []
            try:
                if frame_spool is not None:
                    receipt = frame_spool.close()
                    receipt.update(schema="fr5-stream-closed-v1", codec_sha256=self.storage.codec_sha256,
                                   protocol_sha256=self.storage.protocol_sha256,
                                   local_episode=str(self.root))
                    path = self.root / "stream-closed.json"
                    if receipt.get('accepted_frames') == 0:
                        receipt['materialization_status'] = 'not_attempted_no_frames'
                    save(path, receipt)
                    self.storage.sync_tree()
                    if receipt.get('accepted_frames', 0) > 0 and not getattr(self, 'defer_materialization', False):
                        receipt = self.storage.materialize()
                        self.storage.release_checkpoint_caches(receipt)
                else:
                    self.storage.sync_files(*files)
            except BaseException:
                if frame_spool is not None:
                    frame_spool.abort()
                raise
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        validate_scope(json.loads((args.directory / "scope.json").read_text()))
        print("Scope syntax verified; no connection or motion")
        return
    session = TeacherSession(args.directory)
    signal.signal(signal.SIGINT, lambda *_: session.interrupted.set())
    signal.signal(signal.SIGTERM, lambda *_: session.interrupted.set())
    finished = False
    try:
        state = session.start()
        with use_shared_state_stream(session.robot.stream):
            print(json.dumps({"status": "recording_ready_no_motion", "state": state}), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                op = request["op"]
                if op == "finish":
                    result = session.finish(request.get("outcome", "interrupted"), request.get("evidence"))
                    finished = True
                    print(json.dumps({"status": "finished", "summary": result}), flush=True)
                    break
                if op == "observe":
                    name = request["name"]
                    if not re.fullmatch(r"[a-z0-9-]+", name):
                        raise ValueError("Invalid observation name")
                    report = session.observe_bounded(session.root / name, observer=observe)
                    print(json.dumps({"status": "observed", "path": str(session.root / name / "color.jpg"),
                                      "state": report["robot_after"]}), flush=True)
                    continue
                if op == "decide":
                    result = session.decide(request.get("phase", request.get("skill")), request["decision_id"])
                    print(json.dumps({"status": "jev_decided_no_action", **result}), flush=True)
                    continue
                if op == 'policy_propose' and args.policy_training:
                    print(json.dumps({'status':'policy_proposal_no_action', **session.propose(request['phase'])}),flush=True)
                    continue
                allowed = ('close','release','policy_lift','policy_return') if args.policy_training else ('close','lift','return','release')
                if op not in allowed:
                    raise ValueError("Unknown bounded phase")
                claim = session.root / f"PHASE_{op}_{request.get('height', '')}"
                claim.open("x").close()
                session.review_phase(op.removeprefix('policy_'), request["decision_id"])
                result = {"close": session.close_gripper, "return": session.setdown,
                          "release": session.release,
                          "policy_lift": lambda: session.run_vertical('lift'),
                          "policy_return": lambda: session.run_vertical('return'),
                          "lift": lambda: session.lift(request["height"])}[op]()
                print(json.dumps({"status": "phase_completed", "phase": op, "state": result}), flush=True)
    except BaseException as exc:
        if session.storage is not None or session.active_command:
            session.stop_once(exc, arm_in_flight=session.active_command)
        print(json.dumps({"status": "failed_no_retry", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise
    finally:
        try:
            if not finished and session.writer:
                session.finish("interrupted")
        finally:
            if session.robot.is_connected:
                session.robot.disconnect()


if __name__ == "__main__":
    main()
