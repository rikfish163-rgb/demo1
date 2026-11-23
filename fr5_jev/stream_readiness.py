"""Check the recorded integration evidence before enabling a fresh stream.

This validates existing files and server hashes; it never connects to a robot,
starts a camera, or issues a command. Any changed code invalidates the proof.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

from .durable_storage import REMOTE_BASE, _server_check


ROOT = Path(__file__).resolve().parent.parent
PROOF = Path.home() / ".local/state/fr5-system/stream-readiness.json"
CODE_FILES = (
    "fr5_jev/async_spool.py", "fr5_jev/stream_protocol.py", "fr5_jev/stream_storage.py",
    "fr5_jev/materialize_stream.py", "fr5_jev/stream_readiness.py", "fr5_jev/stream_recording_probe.py",
    "fr5_jev/stream_control_load_probe.py",
    "fr5_jev/durable_storage.py", "fr5_jev/prepare_varied_session.py", "fr5_jev/teacher_targets.py",
    "fr5_jev/teacher_varied.py",
    "fr5_runtime/robot.py", "fr5_runtime/kinematics.py", "fr5_runtime/session_guard.py",
    "fr5_jev/continuous_workflow.py", "fr5_lerobot/src/lerobot_robot_fairino/continuous_capture.py",
    "fr5_lerobot/src/lerobot_robot_fairino/demo_writer.py", "fr5_lerobot/src/lerobot_robot_fairino/frame_codec.py",
    "fr5_jev/test_stream_finalization.py", "fr5_jev/test_stream_materialization.py",
    "fr5_jev/test_stream_pipeline.py", "fr5_jev/test_async_spool.py",
    "fr5_jev/test_durable_recording_integration.py", "fr5_jev/test_continuous_workflow.py",
    "fr5_jev/test_stream_readiness.py", "fr5_jev/test_stream_storage_timeout.py",
    "fr5_jev/test_durable_storage.py", "fr5_jev/test_stop_persistence.py",
    "fr5_jev/test_teacher_targets.py", "fr5_jev/test_teacher_varied.py",
    "fr5_lerobot/tests/test_frame_codec.py", "fr5_lerobot/tests/test_continuous_capture.py",
    "fr5_lerobot/tests/test_demo_writer.py",
)
DIAGNOSTIC_NAME = re.compile(r"diagnostic-stream-readonly-[0-9a-f]{32}\Z")
SYNTHETIC_NAME = re.compile(r"synthetic-final-control-[0-9a-f]{32}\Z")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_readiness():
    try:
        proof = json.loads(PROOF.read_text())
        if proof.get("schema") != "fr5-stream-readiness-v1":
            raise ValueError("missing readiness proof")
        now, created = time.time(), proof["created_unix"]
        if not 0 <= now - created <= 8 * 3600:
            raise ValueError("readiness proof is stale or from the future")
        recorded = proof["code_sha256"]
        if set(recorded) != set(CODE_FILES) or any(_sha(ROOT / name) != recorded[name] for name in CODE_FILES):
            raise ValueError("code or tests changed since integration verification")
        tests = proof["tests"]
        log_root = PROOF.parent.resolve()
        log = log_root / tests["log"]
        if (not log.resolve().is_relative_to(log_root) or _sha(log) != tests["sha256"]
                or tests.get("status") != "passed" or tests.get("count", 0) < 50):
            raise ValueError("integration test evidence is missing or changed")
        text = log.read_text()
        if not re.search(r"Ran " + str(tests["count"]) + r" tests? in ", text) or not re.search(r"(?m)^OK\s*$", text):
            raise ValueError("integration test log does not show the recorded pass")
        diag_path = Path(proof["diagnostic"]["local_result"])
        if _sha(diag_path) != proof["diagnostic"]["result_sha256"]:
            raise ValueError("read-only recording evidence changed")
        diag = json.loads(diag_path.read_text())
        dname = Path(diag["server_root"]).name
        durability, remote = diag["stream_durability"], diag["remote_integrity"]
        if (not DIAGNOSTIC_NAME.fullmatch(dname) or diag["server_root"] != str(REMOTE_BASE / dname)
                or diag.get("status") != "measurement_complete" or diag.get("physical_commands_sent") != 0
                or diag.get("requested_fps") != 6 or diag.get("requested_seconds", 0) < 30
                # Finite source-clock windows can measure just below nominal6Hz.
                or diag.get("measured_camera_fps", 0) < 5.95 or diag.get("max_frame_gap_s", 1) > .30
                or diag.get("capture_frames", 0) < 180 or diag.get("errors")
                or durability.get("accepted_frames") != diag["capture_frames"]
                or durability.get("last_durable_sequence") != diag["capture_frames"] - 1
                or durability.get("unacknowledged_frames") != 0 or durability.get("error") is not None
                or durability.get("peak_unacknowledged_frames", 24) >= 24
                or remote.get("frames") != diag["capture_frames"]
                or remote.get("codec_sha256") != recorded["fr5_lerobot/src/lerobot_robot_fairino/frame_codec.py"]
                or remote.get("protocol_sha256") != recorded["fr5_jev/stream_protocol.py"]):
            raise ValueError("read-only 6 Hz or durable frame verification is incomplete")
        s = proof["synthetic_finalization"]
        synthetic_local = Path(s["local_episode"])
        if (not SYNTHETIC_NAME.fullmatch(s["server_name"])
                or synthetic_local.name != s["server_name"]
                or s.get("control_verification") != "verified"
                or s.get("target_alignment") != "verified_against_wire_and_frame_clocks"
                or s.get("frames", 0) < 2
                or _sha(synthetic_local / "scope.json") != s.get("scope_sha256")
                or _sha(synthetic_local / "summary.json") != s.get("summary_sha256")):
            raise ValueError("final summary/evidence integration was not verified")
        _server_check(REMOTE_BASE / dname, {
            "diagnostic-result.json": proof["diagnostic"]["result_sha256"],
            "materialized/stream-materialization.json": proof["diagnostic"]["materialization_sha256"]})
        _server_check(REMOTE_BASE / s["server_name"], {
            "materialized/stream-materialization.json": s["materialization_sha256"],
            "materialized/scope.json": s["scope_sha256"], "materialized/summary.json": s["summary_sha256"]})
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Verified stream readiness is unavailable: " + str(exc)) from exc
    return proof
