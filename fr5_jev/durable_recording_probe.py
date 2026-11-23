"""Explicit, read-only camera/state recording and remote hash diagnostic.

No command transport is constructed. The diagnostic hold label is synthetic,
``executed=False``, and is never a training example or a physical command.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .durable_storage import REMOTE_BASE, _server_check, new_recording_path


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def diagnostic_hold(state):
    """Make an explicitly non-executed label before the camera samples."""
    return {"sequence": 1, "issued_unix": time.time(), "issued_monotonic": time.monotonic(),
            "source": "agent_teacher", "kind": "hold", "executed": False,
            "target": [*map(float, state["joints"]), float(state["gripper_position"])],
            "teacher": {"policy": "synthetic_readonly_diagnostic_no_command",
                        "phase": "diagnostic_readonly", "decision_id": "no_actuation"}}


def run_probe(seconds=3):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 2 <= seconds <= 10:
        raise ValueError("Read-only probe duration must be 2..10 seconds")
    # These are observation-only adapters; no FR5 command client is imported.
    from lerobot_camera_gemini import GeminiBridgeConfig
    from lerobot_robot_fairino.configuration_fairino import FairinoFR5Config
    from lerobot_robot_fairino.robot_fairino import FairinoFR5
    from lerobot_robot_fairino.continuous_capture import ContinuousCapture
    from lerobot_robot_fairino.demo_writer import DemoWriter

    name = "diagnostic-readonly-" + uuid.uuid4().hex
    root = new_recording_path(name)
    root.mkdir()
    scope = {"schema": "fr5-readonly-storage-diagnostic-v1", "diagnostic_only": True,
             "training_ready": False, "physical_commands_sent": 0, "nominal_fps": 6,
             "label": "Synthetic non-executed hold based on a prior measured state; not a command or training action"}
    _write_json(root / "scope.json", scope)
    print(json.dumps({"status": "readonly_diagnostic_started", "server_root": str(REMOTE_BASE / name),
                      "physical_commands_sent": 0}), flush=True)
    config = FairinoFR5Config(camera=GeminiBridgeConfig(url="http://127.0.0.1:8765/snapshot_fast.npz", fps=6))
    robot = FairinoFR5(config)
    writer = capture = None
    capture_report = writer_report = None
    errors = []
    begun = time.monotonic()
    try:
        robot.connect()
        previous_state = robot.stream.latest(max_age_s=.25)
        hold = diagnostic_hold(previous_state)
        _write_json(root / "diagnostic-hold.json", hold)
        # Ensure the camera acquisition happens after the synthetic label was
        # formed; no controller instruction is issued or implied.
        time.sleep(.4)
        writer = DemoWriter(root / "lerobot", fps=6, task="Read-only storage diagnostic; never for training",
                            journal_only=True, external_capture_root=root / "continuous", sync_writes=True,
                            metadata={"diagnostic_only": True, "physical_commands_sent": 0,
                                      "image_width": 640, "image_height": 360}).start()
        capture = ContinuousCapture(robot, root / "continuous", fps=6,
                                    task="Read-only storage diagnostic; never for training",
                                    sample_callback=lambda sample: writer.add(sample, hold), sync_writes=True,
                                    metadata={"diagnostic_only": True, "physical_commands_sent": 0})
        capture.start()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            capture.assert_healthy()
            time.sleep(.05)
        capture_report = capture.finish("observed", evidence=["read-only diagnostic; no actuation"])
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if capture is not None and capture._started:
            try:
                capture_report = capture._report or capture.finish("interrupted")
            except Exception as exc:
                errors.append(f"capture finish: {type(exc).__name__}: {exc}")
        if writer is not None and writer._started and not writer._finished:
            try:
                writer_report = writer.finish("interrupted")
            except Exception as exc:
                errors.append(f"journal finish: {type(exc).__name__}: {exc}")
        if robot.is_connected:
            robot.disconnect()

    frame_count = len(capture._times) if capture else 0
    journal_count = len(writer._times) if writer else 0
    capture_done = bool(capture_report and capture_report.get("capture_complete"))
    fps = capture_report.get("unique_camera_fps") if capture_report else None
    gap = capture_report.get("max_frame_gap_s") if capture_report else None
    ready = bool(not errors and capture_done and frame_count == journal_count and
                 fps is not None and fps >= 5.7 and gap is not None and gap <= .30 and
                 writer_report and writer_report.get("storage_complete"))
    result = {"schema": "fr5-readonly-storage-diagnostic-result-v1", "diagnostic_only": True,
              "physical_commands_sent": 0, "training_ready": False,
              "status": "six_hz_observed" if ready else "incomplete_or_too_slow",
              "requested_fps": 6, "measured_fps": fps, "max_frame_gap_s": gap,
              "capture_frames": frame_count, "journal_rows": journal_count,
              "capture_complete": capture_done, "elapsed_s": time.monotonic() - begun,
              "errors": errors, "server_root": str(REMOTE_BASE / name)}
    _write_json(root / "diagnostic-result.json", result)
    worker_alive = bool(capture and capture._thread and capture._thread.is_alive())
    if worker_alive:
        result["remote_hash_verification"] = "not attempted: capture worker is still active"
        return result
    try:
        hashes = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeError("Unexpected symlink in diagnostic output")
            if path.is_file():
                hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if frame_count:
            for key in ("continuous/frames/000000.png", "continuous/frames/000000.npz",
                        "continuous/capture.jsonl", "lerobot/meta/capture.jsonl"):
                if key not in hashes:
                    raise RuntimeError("Missing diagnostic frame or journal: " + key)
        _server_check(REMOTE_BASE / name, hashes)
        verification = {"schema": "fr5-readonly-storage-hash-verification-v1",
                        "status": "all_files_sha256_matched_and_fsynced_on_server",
                        "files": hashes, "server_root": str(REMOTE_BASE / name),
                        "physical_commands_sent": 0, "training_ready": False}
        _write_json(root / "hash-verification.json", verification)
        verification_sha = hashlib.sha256((root / "hash-verification.json").read_bytes()).hexdigest()
        _server_check(REMOTE_BASE / name, {"hash-verification.json": verification_sha})
        result.update(remote_hash_verification="matched", files_hashed=len(hashes),
                      first_rgb_sha256=hashes.get("continuous/frames/000000.png"),
                      first_depth_sha256=hashes.get("continuous/frames/000000.npz"),
                      hash_verification_sha256=verification_sha)
    except Exception as exc:
        result["remote_hash_verification"] = f"failed: {type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-readonly", action="store_true")
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    if not args.execute_readonly:
        print(json.dumps({"status": "inert", "physical_commands_sent": 0}))
        return
    result = run_probe(args.seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "six_hz_observed" or result.get("remote_hash_verification") != "matched":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
