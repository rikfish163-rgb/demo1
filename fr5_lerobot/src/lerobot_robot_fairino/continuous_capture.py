"""Continuous RGB-D/telemetry and event evidence around an existing executor.

No motion transport is created here. Raw JOG/RPC events are deliberately not
converted into joint action labels. Use DemoWriter with actual target commands
when the executor provides the declared seven-dimensional target space.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from .feedback import gripper_feedback_valid
from lerobot_camera_gemini.timing import sample_clock, MONOTONIC


def _wait_until_monotonic(stop, deadline):
    """Avoid Event.wait's CLOCK_REALTIME semaphore on this Conda build.

    Short CLOCK_MONOTONIC sleeps keep cancellation responsive without letting
    a host wall-clock step stretch the camera sampling period.
    """
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(.02, remaining))


class ContinuousCapture:
    """Caller owns a connected FairinoFR5 and its single state receiver.

    Start once before any phase, finish after release/verification. Check
    assert_healthy before each command. Capture failure is sticky and preserves
    partial evidence. This recorder itself cannot stop or move the robot.
    """

    def __init__(self, robot, root, *, task, fps=6, metadata=None, sample_callback=None,
                 sync_writes=False, frame_spool=None):
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps <= 30:
            raise ValueError("fps must be in (0, 30]")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task is required")
        self.robot, self.root = robot, Path(root).expanduser().resolve()
        self.task, self.fps = task, float(fps)
        if sample_callback is not None and not callable(sample_callback):
            raise ValueError("sample_callback must be callable")
        if not isinstance(sync_writes, bool):
            raise ValueError("sync_writes must be a boolean")
        self.sync_writes = sync_writes
        self.sample_callback = sample_callback
        if frame_spool is not None and sample_callback is None:
            raise ValueError("A streamed frame requires its validated teacher journal row")
        self.frame_spool = frame_spool
        self.metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        self._stop, self._ready = threading.Event(), threading.Event()
        self._lock = threading.RLock()
        self._thread = None
        self._error = None
        self._clock_basis = None
        self._last_attempt = None
        self._last_cycle = None
        self._times, self._skews = [], []
        self._invalid_gripper = 0
        self._events, self._commands = [], {}
        self._event_file = self._capture_file = None
        self._report = None
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError("Capture can only start once")
        if not self.robot.is_connected:
            raise RuntimeError("Caller must connect the shared observer first")
        self.root.mkdir(parents=True, exist_ok=False)
        self._started = True
        try:
            (self.root / "frames").mkdir()
            self._event_file = (self.root / "events.jsonl").open("x", encoding="utf-8")
            self._capture_file = (self.root / "capture.jsonl").open("x", encoding="utf-8")
            self._thread = threading.Thread(target=self._run, name="fr5-continuous-capture", daemon=True)
            self._thread.start()
            if not self._ready.wait(2 * self.robot.config.camera.timeout_s + 5):
                raise TimeoutError("No first recorded frame")
            self.assert_healthy()
            return self
        except BaseException:
            self.finish("interrupted")
            raise

    def assert_healthy(self):
        # A disconnected network filesystem can block a writer holding this
        # lock. Do not let that block the executor's independent stop path.
        if not self._lock.acquire(timeout=.25):
            raise TimeoutError("Continuous capture storage is unresponsive")
        try:
            if self._error:
                raise RuntimeError(f"Continuous capture failed: {self._error}")
            if (not self._started or self._stop.is_set() or self._thread is None
                    or not self._thread.is_alive() or not self._times):
                raise RuntimeError("Continuous capture is not active")
            # Existing executor checks remain responsible for physical state.
            now = time.monotonic() if self._clock_basis == MONOTONIC else time.time()
            if now - self._times[-1] > self.robot.config.camera.max_age_s + 1 / self.fps:
                raise TimeoutError("No recently stored camera frame")
        finally:
            self._lock.release()
        if self.frame_spool is not None:
            # Storage backlog is separate from camera freshness. Never hold the
            # capture lock while asking the transport about its state.
            self.frame_spool.assert_healthy()

    def _event(self, kind, **fields):
        with self._lock:
            if not self._started or self._stop.is_set():
                raise RuntimeError("Capture is not accepting events")
            row = {"event_id": len(self._events) + 1, "time_unix": time.time(),
                   "time_monotonic":time.monotonic(),
                   "kind": kind, **fields}
            text = json.dumps(row, allow_nan=False, ensure_ascii=False)
            try:
                self._event_file.write(text + "\n")
                self._event_file.flush()
                if self.sync_writes:
                    os.fsync(self._event_file.fileno())
            except BaseException as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                raise
            self._events.append(json.loads(text))
            return deepcopy(row)

    def phase(self, name, *, teacher=None):
        self.assert_healthy()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("phase name is required")
        return self._event("phase", name=name, teacher=teacher or {})

    def call_once(self, method, function, *args, **kwargs):
        """Journal one caller-supplied RPC invocation; never retry on uncertainty.

        Use only for an already gated command. Keep emergency/normal stop on
        the executor's independent path, so recording failure cannot block it.
        A return event is an RPC response, never a task completion assertion.
        """
        self.assert_healthy()
        if not isinstance(method, str) or not method:
            raise ValueError("method is required")
        event = self._event("command_attempt", method=method, args=args, kwargs=kwargs)
        key = event["event_id"]
        with self._lock:
            self._commands[key] = "attempted"
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            with self._lock:
                self._commands[key] = "uncertain"
            self._event("command_uncertain", command_id=key, error=f"{type(exc).__name__}: {exc}")
            raise
        self._event("command_return", command_id=key, response=result)
        with self._lock:
            self._commands[key] = "rpc_returned"
        return result

    def _write_sample(self, sample):
        camera, state = sample["camera"], sample["robot_state"]
        meta = camera["metadata"]
        captured,robot_time,basis = sample_clock(sample)
        skew = float(sample['sync_skew_s'])
        self._last_attempt = {'camera':meta,'robot':state,'sync_clock':basis,
                              'sync_skew_s':skew,'previous_recorded_time':self._times[-1] if self._times else None,
                              'attempted_time':captured,'frames_already_recorded':len(self._times)}
        if self._clock_basis is not None and basis != self._clock_basis:
            raise ValueError('Capture clock domain changed')
        self._clock_basis = basis
        if not math.isfinite(captured) or not math.isfinite(skew):
            raise ValueError("Non-finite capture timing")
        if meta["source_camera_serial"] != self.robot.config.camera.expected_serial:
            raise ValueError("Camera identity changed")
        if self._times and captured <= self._times[-1]:
            raise ValueError("Duplicate or backwards camera timestamp")
        if self._times and captured - self._times[-1] > max(.5, 2.5 / self.fps):
            raise TimeoutError("Camera frame gap exceeds limit")
        if abs(skew) > self.robot.config.max_sync_skew_s:
            raise TimeoutError("Camera/state skew exceeds limit")
        if abs(robot_time - captured - skew) > 1e-6:
            raise ValueError("Inconsistent camera/state timing")
        index = len(self._times)
        encoded = None
        if self.frame_spool is None:
            rgb_path, depth_path = f"frames/{index:06d}.png", f"frames/{index:06d}.npz"
            Image.fromarray(camera["color_rgb"]).save(self.root / rgb_path)
            if self.sync_writes:
                with (self.root / rgb_path).open("rb") as stored:
                    os.fsync(stored.fileno())
            np.savez_compressed(self.root / depth_path, depth_mm=camera["depth_mm"])
            if self.sync_writes:
                with (self.root / depth_path).open("rb") as stored:
                    os.fsync(stored.fileno())
        else:
            # RGB is explicitly lossy JPEG90; depth is exact or capture fails.
            from .frame_codec import encode_media
            encoded = encode_media(camera["color_rgb"], camera["depth_mm"], jpeg_quality=90)
            rgb_path, depth_path = f"frames/{index:06d}.jpg", f"frames/{index:06d}.npz"
        valid = gripper_feedback_valid(state)
        with self._lock:
            # Associate by acquisition time, not whichever phase is current
            # when image compression finishes.
            time_key = 'time_monotonic' if basis == MONOTONIC else 'time_unix'
            events = [e for e in self._events if e[time_key] <= captured]
            phases = [e for e in events if e["kind"] == "phase"]
            attempts = [e for e in events if e["kind"] == "command_attempt"]
            row = {"frame_index": index, "camera": meta, "robot": state,
                   "sync_skew_s": skew, "sync_clock":basis, "gripper_feedback_valid": valid,
                   "rgb_file": rgb_path, "depth_file": depth_path,
                   "phase_event_id": phases[-1]["event_id"] if phases else None,
                   "last_command_attempt_id": attempts[-1]["event_id"] if attempts else None}
            if encoded is not None:
                row["media_codec"] = encoded[0]
            row = json.loads(json.dumps(row, allow_nan=False, ensure_ascii=False))
            if encoded is None:
                self._capture_file.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
                self._capture_file.flush()
                if self.sync_writes:
                    os.fsync(self._capture_file.fileno())
                self._times.append(captured)
                self._skews.append(skew)
                self._invalid_gripper += int(not valid)
        return row, encoded, captured, skew, valid

    def _run(self):
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                if self.frame_spool is not None:
                    # Reject a saturated/disconnected stream before acquiring
                    # another frame that cannot be durably accepted.
                    self.frame_spool.assert_healthy()
                sample = self.robot.get_sample()
                row, encoded, captured, skew, valid = self._write_sample(sample)
                teacher_row = None
                if self.sample_callback is not None:
                    teacher_row = self.sample_callback(sample)
                if encoded is not None:
                    if not isinstance(teacher_row, dict):
                        raise RuntimeError("No validated action journal row for captured frame")
                    media, rgb, depth = encoded
                    self.frame_spool.enqueue(capture=row, teacher=teacher_row,
                                             media=media, rgb=rgb, depth=depth)
                    with self._lock:
                        self._times.append(captured)
                        self._skews.append(skew)
                        self._invalid_gripper += int(not valid)
                self._ready.set()
                wait_start = time.monotonic()
                _wait_until_monotonic(self._stop, tick + 1 / self.fps)
                self._last_cycle = {'started_monotonic':tick,
                                    'processing_s':wait_start-tick,
                                    'wait_s':time.monotonic()-wait_start,
                                    'period_s':1/self.fps}
        except BaseException as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                rejection = {'error':self._error,'last_attempt':self._last_attempt,
                             'previous_cycle':self._last_cycle,
                             'time_unix':time.time(),'time_monotonic':time.monotonic()}
                try:
                    (self.root/'capture-rejection.json').write_text(json.dumps(rejection,indent=2))
                except (OSError, TypeError, ValueError) as diagnostic_error:
                    self._error += f'; rejection diagnostic failed: {diagnostic_error}'
                finally:
                    self._ready.set()

    def finish(self, outcome="observed", *, evidence=None):
        if outcome not in {"observed", "success", "failure", "interrupted", "discarded", "pending_review"}:
            raise ValueError("Invalid outcome label")
        if self._report is not None:
            return deepcopy(self._report)
        if not self._started:
            raise RuntimeError("Capture has not started")
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(2 * self.robot.config.camera.timeout_s + 5)
            if self._thread.is_alive():
                raise RuntimeError("Capture worker did not exit; evidence remains open")
        if self.frame_spool is not None and self._error is None:
            try:
                drain_timeout = max(10., getattr(self.frame_spool, 'max_unacked_age', 4.) + 5.)
                self.frame_spool.wait_for_ack(self.frame_spool.watermark(), timeout=drain_timeout)
            except Exception as exc:
                self._error = f"remote frame durability: {type(exc).__name__}: {exc}"
        for log in (self._capture_file, self._event_file):
            if log is not None:
                log.close()
        times = self._times
        intervals = np.diff(times)
        uncertain = sum(v != "rpc_returned" for v in self._commands.values())
        report = {
            "schema_version": 1, "dataset_kind": "teacher_execution_evidence",
            "task": self.task, "metadata": self.metadata,
            "outcome": "interrupted" if self._error or uncertain else outcome,
            "outcome_source": "awaiting_user_review" if outcome == "pending_review" else "caller_label", "outcome_evidence": evidence or [],
            "capture_complete": not self._error and len(times) >= 2,
            "error": self._error, "frames": len(times), "requested_fps": self.fps,
            "timing_clock":self._clock_basis,
            "unique_camera_fps": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None,
            "max_frame_gap_s": float(max(intervals)) if len(intervals) else None,
            "max_sync_skew_ms": 1000 * max(map(abs, self._skews)) if times else None,
            "frames_with_invalid_gripper_feedback": self._invalid_gripper,
            "command_attempts": len(self._commands),
            "commands_without_rpc_return": uncertain,
            "has_target_action_labels": False, "training_ready": False,
            "not_training_ready_reason": "Raw RPC/JOG events are not continuous target action vectors; outcome and timing also require review.",
            "timestamp_semantics": "timing_clock orders camera/robot/events; original Unix timestamps retained for audit; no hardware synchronization",
            "units": {"joint": "deg", "pose_xyz": "mm", "gripper": "percent", "depth": "mm"},
        }
        if self.frame_spool is not None:
            report["stream_durability"] = self.frame_spool.status()
            report["media_storage"] = "JPEG90 lossy RGB and lossless float32 depth; server stream requires materialization"
        (self.root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        self._report = report
        return deepcopy(report)


def main():
    """Read-only continuous-capture smoke run. No execution option exists."""
    from lerobot_camera_gemini import GeminiBridgeConfig
    from .configuration_fairino import FairinoFR5Config
    from .robot_fairino import FairinoFR5
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--fps", type=float, default=6.)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
        parser.error("seconds must be in (0, 300]")
    config = FairinoFR5Config(camera=GeminiBridgeConfig(url="http://127.0.0.1:8765/snapshot_fast.npz"))
    robot = FairinoFR5(config)
    capture = ContinuousCapture(robot, args.root, fps=args.fps, task="Read-only capture verification")
    try:
        robot.connect()
        capture.start()
        capture.phase("read_only_verification")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            capture.assert_healthy()
            time.sleep(.05)
        print(json.dumps(capture.finish(), indent=2))
    finally:
        if capture._started and capture._report is None:
            capture.finish("interrupted")
        robot.disconnect()


if __name__ == "__main__":
    main()
