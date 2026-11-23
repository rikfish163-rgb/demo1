"""Explicit read-only 6 Hz end-to-end diagnostic for the asynchronous spool.

Import and the default CLI are inert. Only --run-readonly connects to the
existing camera/state observer; it never creates a command client or moves a
robot. This diagnostic does not authorize a future physical episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid

from .durable_recording_probe import diagnostic_hold
from .stream_storage import REMOTE_BASE, StreamStorage, new_stream_recording_path


def _save(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def run_probe(seconds=30.):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 30 <= seconds <= 300:
        raise ValueError("Choose a read-only sustained duration from 30 to 300 seconds")
    from lerobot_camera_gemini import GeminiBridgeConfig
    from lerobot_robot_fairino.configuration_fairino import FairinoFR5Config
    from lerobot_robot_fairino.robot_fairino import FairinoFR5
    from lerobot_robot_fairino.continuous_capture import ContinuousCapture
    from lerobot_robot_fairino.demo_writer import DemoWriter

    name = "diagnostic-stream-readonly-" + uuid.uuid4().hex
    root = new_stream_recording_path(name)
    root.mkdir(mode=0o700)
    scope = {"schema": "fr5-stream-readonly-diagnostic-v1", "diagnostic_only": True,
             "training_ready": False, "physical_commands_sent": 0, "nominal_fps": 6,
             "label": "Synthetic non-executed hold, not a training action or physical command"}
    _save(root / "scope.json", scope)
    digest = hashlib.sha256((root / "scope.json").read_bytes()).hexdigest()
    storage = robot = writer = capture = None
    capture_report = writer_report = remote = None
    errors = []
    try:
        storage = StreamStorage.start_readonly(root, digest,
                  required_bytes=256 * 1024**2 + int(seconds * 6 * 256 * 1024))
        config = FairinoFR5Config(camera=GeminiBridgeConfig(url="http://127.0.0.1:8765/snapshot_fast.npz", fps=6))
        robot = FairinoFR5(config)
        robot.connect()
        hold = diagnostic_hold(robot.stream.latest(max_age_s=.25))
        _save(root / "diagnostic-hold.json", hold)
        storage.sync_files(root / "diagnostic-hold.json")
        time.sleep(.4)
        writer = DemoWriter(root / "lerobot", fps=6, task="Read-only diagnostic; never for training",
            journal_only=True, external_capture_root=root / "continuous", sync_writes=True,
            frame_spool=storage.frame_spool,
            metadata={"diagnostic_only": True, "physical_commands_sent": 0,
                      "image_width": 640, "image_height": 360}).start()
        capture = ContinuousCapture(robot, root / "continuous", fps=6,
            task="Read-only diagnostic; never for training", sync_writes=True,
            frame_spool=storage.frame_spool, sample_callback=lambda sample: writer.add(sample, hold),
            metadata={"diagnostic_only": True, "physical_commands_sent": 0}).start()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            capture.assert_healthy()
            time.sleep(.05)
        capture_report = capture.finish("observed", evidence=["read-only diagnostic; no actuation"])
        writer_report = writer.finish("interrupted")
        if not capture_report["capture_complete"] or not writer_report["storage_complete"]:
            raise RuntimeError("Capture or journal durability failed")
        storage.sync_files(root / "continuous/events.jsonl", root / "continuous/report.json",
                           root / "lerobot/meta/demo.json")
        receipt = storage.frame_spool.close()
        receipt.update(schema="fr5-stream-closed-v1", codec_sha256=storage.codec_sha256,
                       protocol_sha256=storage.protocol_sha256, local_episode=str(root))
        _save(root / "stream-closed.json", receipt)
        storage.sync_files(root / "stream-closed.json")
        remote = storage.materialize()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if capture is not None and capture._started and capture._report is None:
            try:
                capture_report = capture.finish("interrupted")
            except Exception as exc:
                errors.append(f"capture finish: {type(exc).__name__}: {exc}")
        if writer is not None and writer._started and not writer._finished:
            try:
                writer_report = writer.finish("interrupted")
            except Exception as exc:
                errors.append(f"journal finish: {type(exc).__name__}: {exc}")
        if storage is not None:
            storage.frame_spool.abort()
        if robot is not None and robot.is_connected:
            robot.disconnect()
    result = {"schema": "fr5-stream-readonly-result-v1", "diagnostic_only": True,
              "physical_commands_sent": 0, "training_ready": False,
              "status": "measurement_complete" if not errors and remote else "incomplete",
              "requested_fps": 6, "requested_seconds": seconds,
              "measured_camera_fps": capture_report.get("unique_camera_fps") if capture_report else None,
              "max_frame_gap_s": capture_report.get("max_frame_gap_s") if capture_report else None,
              "capture_frames": capture_report.get("frames") if capture_report else 0,
              "stream_durability": storage.frame_spool.status() if storage else None,
              "remote_integrity": remote, "server_root": str(REMOTE_BASE / name),
              "physical_execution_authorized": False, "errors": errors}
    _save(root / "diagnostic-result.json", result)
    if storage is not None:
        try:
            storage.sync_files(root / "diagnostic-result.json")
        except Exception as exc:
            # The local result remains explicit about the remote diagnostic gap.
            result["diagnostic_result_upload_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-readonly", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.)
    args = parser.parse_args()
    if not args.run_readonly:
        print(json.dumps({"status": "inert", "physical_commands_sent": 0}))
        return
    print(json.dumps(run_probe(args.seconds), ensure_ascii=False))


if __name__ == "__main__":
    main()
