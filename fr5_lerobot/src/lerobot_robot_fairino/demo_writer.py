"""Write operator target demonstrations locally; this module sends no commands."""

from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .record import dataset_features, make_frame
from .robot_fairino import STATE_NAMES
from lerobot_camera_gemini.timing import sample_clock, command_clock, MONOTONIC


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json(value):
    return json.dumps(value, default=_json_default, allow_nan=False, ensure_ascii=False)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


class DemoWriter:
    """One writer per episode and new directory, with explicit operator outcome.

    ``metadata.image_width/image_height`` set the image shape before ``start``;
    the default is 640x360. ``metadata.camera_metadata.color_intrinsic`` is also
    accepted as a fallback. Neither acquisition nor robot control occurs here.

    ``journal_only=True`` writes only meta/capture.jsonl and meta/demo.json.
    ``external_capture_root`` must identify the separate ContinuousCapture
    directory; it may be created after this writer starts. That recorder owns
    RGB/depth persistence and reports its completeness independently. Pair its
    capture rows by camera identity and source clock, never by journal index.
    """

    def __init__(self, root: Path, fps: int = 10,
                 task: str = "Pick up the cup and place it down", metadata: dict | None = None,
                 *, journal_only: bool = False, external_capture_root: Path | str | None = None,
                 sync_writes: bool = False, frame_spool=None):
        self.root = Path(root).expanduser().resolve()
        if self.root.exists():
            raise FileExistsError(f"Choose a new episode directory: {self.root}")
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 30:
            raise ValueError("fps must be an integer in 1..30")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if not isinstance(journal_only, bool):
            raise ValueError("journal_only must be an explicit boolean")
        if not isinstance(sync_writes, bool) or (sync_writes and not journal_only):
            raise ValueError("sync_writes is supported for journal-only capture")
        if frame_spool is not None and not journal_only:
            raise ValueError("A frame stream requires journal-only mode")
        self.sync_writes = sync_writes
        self.frame_spool = frame_spool
        self.journal_only = journal_only
        self.external_capture = None
        if journal_only:
            if (not isinstance(external_capture_root, (Path, str))
                    or not str(external_capture_root).strip()):
                raise ValueError("journal_only requires an explicit external_capture_root")
            external = Path(external_capture_root).expanduser().resolve()
            if external == self.root:
                raise ValueError("External capture must use a separate directory")
            self.external_capture = {
                "root": str(external),
                "capture_log": str(external / "capture.jsonl"),
                "report_file": str(external / "report.json"),
                "matching_rule": "Match camera serial and source timestamp in timing_clock; journal and capture frame indices may differ",
                "media_paths": "Read rgb_file and depth_file from the matching external capture row; resolve them relative to root",
            }
        elif external_capture_root is not None:
            raise ValueError("external_capture_root is only used with journal_only=True")
        self.metadata = json.loads(_json(metadata or {}))
        camera_meta = self.metadata.get("camera_metadata", {})
        intrinsic = camera_meta.get("color_intrinsic", {})
        self.width = self.metadata.get("image_width", intrinsic.get("width", 640))
        self.height = self.metadata.get("image_height", intrinsic.get("height", 360))
        for key, value in (("image_width", self.width), ("image_height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        self.fps, self.task = fps, task
        self.repo_id = "local/fr5_operator_demo"
        self.dataset = None
        self._log = None
        self._started = self._finished = False
        self._times, self._robot_times, self._skews, self._action_ages = [], [], [], []
        self._commands_not_executed = 0
        self._invalid_gripper_frames = 0
        self._command_sources = set()
        self._camera_serial = None
        self._clock_basis = None
        self._errors = []
        self._journal_storage_failed = False
        self._report = None
        self._created_utc = datetime.now(timezone.utc).isoformat()

    def start(self):
        if self._started or self._finished:
            raise RuntimeError("DemoWriter can only be started once")
        if self.root.exists():
            raise FileExistsError(f"Choose a new episode directory: {self.root}")
        try:
            if self.journal_only:
                self.root.mkdir(parents=True, exist_ok=False)
                (self.root / "meta").mkdir()
            else:
                features = dataset_features(self.width, self.height)
                features.update({
                    "action": {"dtype": "float32", "shape": (7,), "names": list(STATE_NAMES)},
                    "action.source_time_us": {
                        "dtype": "int64", "shape": (1,), "names": ["operator_command_host_unix_us"]},
                    "action.source_monotonic_us": {"dtype":"int64","shape":(1,),"names":["command_host_monotonic_us_or0_legacy"]},
                    "action.sequence": {"dtype": "int64", "shape": (1,), "names": ["command_sequence"]},
                })
                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id, root=self.root, fps=self.fps,
                    robot_type="fairino_fr5_operator_demo", features=features,
                    use_videos=False, image_writer_threads=2,
                )
                (self.root / "depth/episode_000000").mkdir(parents=True)
            self._log = (self.root / "meta/capture.jsonl").open("x" if self.journal_only else "w", encoding="utf-8")
            self._started = True
        except BaseException:
            self._finished = True
            if self.dataset is not None:
                for cleanup in (self.dataset.finalize, self.dataset.stop_image_writer):
                    try:
                        cleanup()
                    except BaseException:
                        pass
            raise
        return self

    def _prepare(self, sample, command):
        if not isinstance(command, dict):
            raise ValueError("command must be a dictionary")
        target_raw = command.get("target")
        if not isinstance(target_raw, (list, tuple, np.ndarray)) or len(target_raw) != 7:
            raise ValueError("command.target must contain seven joint/gripper targets")
        target = np.asarray([_number(v, "command.target") for v in target_raw], dtype=np.float64)
        if not 0 <= target[-1] <= 100 or np.any(np.abs(target) > np.finfo(np.float32).max):
            raise ValueError("Invalid target range; gripper target must be in 0..100 percent")
        sequence = command.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence <= np.iinfo(np.int64).max:
            raise ValueError("command.sequence must be a nonnegative int64 integer")
        issued = _number(command.get("issued_unix"), "command.issued_unix")
        if issued <= 0 or round(issued * 1_000_000) > np.iinfo(np.int64).max:
            raise ValueError("Invalid command timestamp")
        if command.get("source") not in {"keyboard", "agent_teacher", "pi05_policy"} or command.get("kind") not in {"hold", "arm", "gripper"}:
            raise ValueError("Expected an explicit command source and kind hold, arm or gripper")
        if command['source'] == 'pi05_policy':
            if (self.metadata.get('dataset_role') != 'policy_evaluation'
                    or not isinstance(command.get('policy'), dict)
                    or not command['policy'].get('checkpoint_sha256')):
                raise ValueError('Policy commands require evaluation provenance and checkpoint identity')
        if command["source"] == "agent_teacher":
            teacher = command.get("teacher")
            if not isinstance(teacher, dict) or any(
                not isinstance(teacher.get(k), str) or not teacher[k].strip()
                for k in ("policy", "decision_id", "phase")
            ):
                raise ValueError("Agent teacher requires policy, decision_id and phase provenance")
        if not isinstance(command.get("executed"), bool):
            raise ValueError("command.executed must be an explicit boolean")
        camera, state = sample["camera"], sample["robot_state"]
        meta = camera["metadata"]
        captured,robot_time,basis = sample_clock(sample)
        if self._clock_basis is not None and basis != self._clock_basis:
            raise ValueError('Demonstration clock domain changed')
        issued_audit = issued
        issued = command_clock(command,basis)
        for value in (captured, robot_time):
            if value <= 0 or round(value * 1_000_000) > np.iinfo(np.int64).max:
                raise ValueError("Invalid observation source timestamp")
        if issued - captured > 1e-6:
            raise ValueError("Future action: command issued after camera source timestamp")
        if self._times and (captured <= self._times[-1] or robot_time <= self._robot_times[-1]):
            raise ValueError("Camera and robot source timestamps must be strictly increasing")
        serial = meta.get("source_camera_serial")
        if not isinstance(serial, str) or not serial:
            raise ValueError("Missing camera serial")
        if self._camera_serial is not None and serial != self._camera_serial:
            raise ValueError("Camera identity changed during the episode")
        rgb, depth = np.asarray(camera["color_rgb"]), np.asarray(camera["depth_mm"])
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("RGB shape/dtype does not match episode configuration")
        if (depth.shape != (self.height, self.width) or depth.dtype != np.float32
                or not np.isfinite(depth).all() or np.any(depth < 0)):
            raise ValueError("Depth must be aligned, finite nonnegative HW float32 millimeters")
        skew = _number(sample["sync_skew_s"], "sync_skew_s")
        if abs(skew - (robot_time - captured)) > 1e-6:
            raise ValueError("Synchronization skew disagrees with source timestamps")
        frame = make_frame(sample, self.task)
        for name, value in frame.items():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite frame feature: {name}")
        observed = frame["observation.state"]
        if not 0 <= observed[-1] <= 100:
            raise ValueError("Observed gripper percentage is outside 0..100")
        frame.update({"action": target.astype(np.float32),
                      "action.source_time_us": np.array([round(issued_audit * 1_000_000)], np.int64),
                      "action.source_monotonic_us":np.array([round(issued*1e6) if basis==MONOTONIC else 0],np.int64),
                      "action.sequence": np.array([sequence], np.int64)})
        # Snapshot every buffer now; acquisition may reuse its arrays asynchronously.
        frame = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in frame.items()}
        depth_path = None if self.journal_only else f"depth/episode_000000/frame_{len(self._times):06d}.npz"
        row_data = {"frame_index": len(self._times), "camera": meta, "robot": state,
                    "command": command, "sync_skew_s": skew, "sync_clock":basis}
        if self.journal_only:
            row_data.update(storage_mode="journal_only", external_capture={
                **self.external_capture,
                "match": {"timing_clock": basis, "camera_source_timestamp": captured,
                          "source_camera_serial": serial},
            })
        else:
            row_data["depth_file"] = depth_path
        row = _json(row_data)
        self._clock_basis=basis
        return frame, depth.copy(), depth_path, row, captured, robot_time, issued, serial, skew

    def add(self, sample: dict, command: dict):
        if not self._started or self._finished:
            raise RuntimeError("Call start before add; finished episodes cannot be changed")
        if self.journal_only and self._journal_storage_failed:
            raise RuntimeError("Journal storage failed; finish the writer before further recording")
        try:
            frame, depth, path, row, captured, robot_time, issued, serial, skew = self._prepare(sample, command)
            if self.journal_only:
                try:
                    if self.frame_spool is None:
                        line = row + "\n"
                        if self._log.write(line) != len(line):
                            raise OSError("Incomplete journal row write")
                        self._log.flush()
                        if self.sync_writes:
                            os.fsync(self._log.fileno())
                except BaseException:
                    self._journal_storage_failed = True
                    raise
            else:
                np.savez_compressed(self.root / path, depth_mm=depth)
                self.dataset.add_frame(frame)
            self._times.append(captured)
            self._robot_times.append(robot_time)
            self._skews.append(skew)
            self._action_ages.append(captured - issued)
            self._commands_not_executed += int(not command["executed"])
            self._invalid_gripper_frames += int(not frame["observation.gripper_feedback_valid"][0])
            self._command_sources.add(command["source"])
            self._camera_serial = serial
            if not self.journal_only:
                self._log.write(row + "\n")
                self._log.flush()
            # Return the exact validated journal row to an optional durable
            # frame stream; existing callers may continue to ignore it.
            return json.loads(row)
        except BaseException as exc:
            self._errors.append(f"{type(exc).__name__}: {exc}")
            raise

    def finish(self, outcome: str) -> dict:
        allowed = {"success", "failure", "discarded", "interrupted", "pending_review"}
        if self.metadata.get('dataset_role') == 'policy_evaluation':
            allowed.add('observed')
        if outcome not in allowed:
            raise ValueError("Unknown episode outcome")
        if self._finished:
            if self._report is None:
                raise RuntimeError("Writer did not start successfully")
            if outcome != self._report["outcome"]:
                raise ValueError("An episode outcome cannot be changed after finish")
            return deepcopy(self._report)
        if not self._started:
            raise RuntimeError("Call start before finish")
        self._finished = True
        cleanup_errors = []
        storage_complete = False
        try:
            if self._times:
                if self.journal_only:
                    self._log.flush()
                    if self.sync_writes:
                        os.fsync(self._log.fileno())
                    if self.frame_spool is None:
                        storage_complete = not self._journal_storage_failed
                    else:
                        status = self.frame_spool.status()
                        storage_complete = (not self._journal_storage_failed and status["error"] is None
                            and status["accepted_frames"] == len(self._times)
                            and status["last_durable_sequence"] == len(self._times) - 1
                            and status["unacknowledged_frames"] == 0)
                else:
                    self.dataset.save_episode()
                    storage_complete = True
        except BaseException as exc:
            operation = "flush_capture_log" if self.journal_only else "save_episode"
            cleanup_errors.append(f"{operation}: {type(exc).__name__}: {exc}")
        finally:
            cleanups = [] if self.journal_only else [
                ("finalize", self.dataset.finalize), ("stop_image_writer", self.dataset.stop_image_writer)]
            cleanups.append(("close_capture_log", self._log.close))
            for name, cleanup in cleanups:
                try:
                    cleanup()
                except BaseException as exc:
                    cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        report = {
            "schema_version": 1, "created_utc": self._created_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "repo_id": self.repo_id, "dataset_kind": (
                "policy_execution_evaluation" if "pi05_policy" in self._command_sources else
                "agent_teacher_demonstration" if "agent_teacher" in self._command_sources
                else "operator_demonstration"),
            "lerobot_version": importlib.metadata.version("lerobot"),
            "task": self.task, "has_action_labels": True,
            "outcome": outcome, "outcome_source": "awaiting_user_review" if outcome == "pending_review" else "operator_label",
            "grasp_success_verified": False, "gripper_feedback_verified": False,
            "training_ready": False, "nominal_fps": self.fps,
            "storage_mode": "journal_only" if self.journal_only else "lerobot",
            "materialized_lerobot_dataset": not self.journal_only and storage_complete and not cleanup_errors,
            "recorded_frames": len(self._times), "metadata": self.metadata,
            "source_camera_serial": self._camera_serial,
            "data_valid": len(self._times) >= 2 and not self._errors,
            "storage_complete": storage_complete and not cleanup_errors,
            "errors": list(self._errors), "cleanup_errors": cleanup_errors,
            "frames_with_command_executed_false": self._commands_not_executed,
            "frames_with_invalid_gripper_feedback": self._invalid_gripper_frames,
            "command_sources": sorted(self._command_sources),
            "robot_commands_sent_by_writer": 0,
            "action_definition": {
                "space": "absolute joint target degrees (6), gripper target percent (1)",
                "source": "Explicit keyboard/agent_teacher command.target only; never copied from observed state",
                "executed": "Caller-reported command flag, preserved in capture.jsonl; hold may be false",
            },
            "timing_clock":self._clock_basis,
            "timestamp_semantics": {
                "timestamp": "Nominal LeRobot frame_index/fps, not measured acquisition time",
                "observation.source_time_us": "int64 host UNIX microseconds: camera bridge, robot receive, collector",
                "action.source_time_us": "int64 operator command issued_unix rounded to microseconds",
                "monotonic_features":"Host monotonic microseconds are valid when observation.monotonic_timing_valid=1;0 sentinel for legacy Unix-only records",
                "precision_limit": "Storage preserves microseconds; clocks are not hardware synchronized",
            },
            "pairing_boundary": {
                "rule": "Caller supplies last effective operator target at camera time in timing_clock",
                "validation": "Command issue <= camera capture in the same timing_clock; Unix fields retained for audit",
                "limitation": "Latched target alongside observation; not a future action, completion proof or acknowledgement",
                "training": "Review temporal action horizon, synchronization and operator labels before ACT training",
            },
            "depth_storage": "depth/episode_000000/frame_NNNNNN.npz: aligned depth_mm float32, invalid=0",
            "units": {"joint": "deg", "pose_xyz": "mm", "pose_rotation": "deg",
                      "gripper": "percent", "depth": "mm"},
            "discard_policy": "Discarded episodes remain on disk for audit; exclude from training",
        }
        if len(self._times) >= 2:
            intervals = np.diff(self._times)
            report["measured_camera_sample_fps"] = float((len(self._times) - 1) / (self._times[-1] - self._times[0]))
            report["camera_interval_s"] = {"median": float(np.median(intervals)), "max": float(intervals.max())}
        if self._times:
            report["sync_abs_skew_ms_max"] = float(np.max(np.abs(self._skews)) * 1000)
            report["action_age_s"] = {"min": min(self._action_ages), "max": max(self._action_ages)}
        if len(self._times) < 2:
            report["not_training_ready_reason"] = "Fewer than two paired frames"
        elif self._errors or cleanup_errors:
            report["not_training_ready_reason"] = "Data or storage errors occurred"
        elif outcome == "pending_review":
            report["not_training_ready_reason"] = "Awaiting user outcome review and action/feedback validation"
        elif outcome in {"discarded", "interrupted"}:
            report["not_training_ready_reason"] = "Episode discarded or interrupted by operator"
        else:
            report["not_training_ready_reason"] = "Action alignment and gripper feedback require review"
        if self.journal_only:
            report.update(
                repo_id=None,
                dataset_kind=("policy_execution_journal" if "pi05_policy" in self._command_sources else
                              "agent_teacher_action_journal" if "agent_teacher" in self._command_sources
                              else "operator_action_journal"),
                external_capture=deepcopy(self.external_capture),
                external_capture_verified=False,
                local_rgb_frames_written=0, local_depth_frames_written=0,
                storage_complete_scope="Successfully flushed journal rows only; external capture completeness is reported separately",
                data_valid_scope="Validated observation/issued-command pairs only; external media persistence is not verified here",
                recorded_frames_semantics="Successfully flushed capture.jsonl rows; a failed write may leave an unconfirmed tail",
                depth_storage="External ContinuousCapture only; no depth file is written by this journal",
                timestamp_semantics={
                    "frame_index": "Sequential journal row index; not an external capture frame index",
                    "camera": "Original source_timestamp_unix and source_timestamp_monotonic when supplied",
                    "robot": "Original timestamp and received_monotonic when supplied",
                    "command": "Original issued_unix and issued_monotonic when supplied",
                    "ordering": "sync_clock selects the source clock; no nominal LeRobot timestamps are stored",
                },
                not_training_ready_reason="Journal-only output needs verified external RGB-D evidence and dataset curation; "
                                          + report["not_training_ready_reason"],
            )
            if self.frame_spool is not None:
                report.update(
                    storage_complete_scope="Each validated row was included in a server-fsynced RGB-D stream record",
                    recorded_frames_semantics="Validated rows paired with stream records; local capture.jsonl is a placeholder until remote materialization",
                    stream_durability=self.frame_spool.status(),
                )
        self._report = report
        (self.root / "meta/demo.json").write_text(_json(report) + "\n", encoding="utf-8")
        return deepcopy(report)
