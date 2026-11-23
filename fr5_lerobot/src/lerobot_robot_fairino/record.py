"""Record real observations into a local LeRobot dataset without sending actions."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_camera_gemini import GeminiBridgeConfig
from .configuration_fairino import FairinoFR5Config
from .robot_fairino import FairinoFR5, STATE_NAMES, POSE_NAMES, STATUS_NAMES
from .feedback import gripper_feedback_valid
from lerobot_camera_gemini.timing import camera_clock, sample_clock, MONOTONIC


def dataset_features(width, height):
    return {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.images.front": {
            "dtype": "image", "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.tcp_pose": {"dtype": "float32", "shape": (6,), "names": POSE_NAMES},
        "observation.flange_pose": {"dtype": "float32", "shape": (6,), "names": POSE_NAMES},
        "observation.joint_velocity": {
            "dtype": "float32", "shape": (6,),
            "names": [f"joint_{i}.deg_s" for i in range(1, 7)],
        },
        "observation.robot_status": {
            "dtype": "int64", "shape": (len(STATUS_NAMES),), "names": STATUS_NAMES,
        },
        "observation.source_time_us": {
            "dtype": "int64", "shape": (3,),
            "names": ["camera_host_unix_us", "robot_host_unix_us", "collector_host_unix_us"],
        },
        "observation.source_monotonic_us": {"dtype":"int64","shape":(3,),
            "names":["camera_host_monotonic_us","robot_host_monotonic_us","collector_host_monotonic_us"]},
        "observation.monotonic_timing_valid": {"dtype":"int64","shape":(1,),"names":["monotonic_timing_available"]},
        "observation.sync_skew_s": {"dtype": "float32", "shape": (1,), "names": ["robot_minus_camera"]},
        "observation.gripper_feedback_valid": {
            "dtype": "int64", "shape": (1,), "names": ["measured_position_valid"]},
    }


def make_frame(sample, task):
    state, meta = sample["robot_state"], sample["camera"]["metadata"]
    captured,robot_time,basis=sample_clock(sample)
    return {
        "observation.state": np.asarray([sample["observation"][k] for k in STATE_NAMES], np.float32),
        "observation.images.front": sample["camera"]["color_rgb"],
        "observation.tcp_pose": np.asarray(state["tcp"], np.float32),
        "observation.flange_pose": np.asarray(state["flange"], np.float32),
        "observation.joint_velocity": np.asarray(state["joint_speeds_deg_s"], np.float32),
        "observation.robot_status": np.asarray([state[k] for k in STATUS_NAMES], np.int64),
        "observation.source_time_us": np.asarray(
            [round(t * 1_000_000) for t in
             (meta["source_timestamp_unix"], state["timestamp"], time.time())], np.int64),
        "observation.sync_skew_s": np.asarray([sample["sync_skew_s"]], np.float32),
        "observation.source_monotonic_us":np.asarray(
            [round(t*1e6) for t in (captured,robot_time,time.monotonic())] if basis==MONOTONIC else [0,0,0],np.int64),
        "observation.monotonic_timing_valid":np.asarray([int(basis==MONOTONIC)],np.int64),
        "observation.gripper_feedback_valid": np.asarray([int(gripper_feedback_valid(state))], np.int64),
        "task": task,
    }


def record(config, root, *, repo_id="local/fr5_gemini_observations", fps=2, frames=20,
           task="FR5 Gemini observation pipeline verification"):
    if not isinstance(fps, int) or not 1 <= fps <= 30:
        raise ValueError("fps must be an integer in 1..30")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
        raise ValueError("At least two frames are required")
    root = Path(root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new dataset directory: {root}")
    robot = FairinoFR5(config)
    dataset = None
    times, skews, depths, counters = [], [], [], []
    pending_error = None
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id, "robot_type": robot.name, "robot_host": config.host,
        "state_port": config.state_port, "camera_url": config.camera.url,
        "expected_camera_serial": config.camera.expected_serial,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "dataset_kind": "observation_only", "has_action_labels": False,
        "training_ready": False, "robot_commands_sent": 0,
        "nominal_fps": fps, "requested_frames": frames,
        "timestamp_semantics": {
            "timestamp": "LeRobot nominal frame_index/fps; not a measured sensor clock",
            "observation.source_time_us": "Host UNIX microseconds int64: camera bridge, robot receive, collector; storage precision is not sensor accuracy",
            "observation.source_monotonic_us": "Host monotonic microseconds; valid only when observation.monotonic_timing_valid=1",
            "alignment": "nearest robot host-receive sample to camera frame receipt in timing_clock; Unix retained unchanged for audit",
            "limitation": "No hardware synchronization or exposure-time guarantee",
        },
        "units": {"joint": "deg", "pose_xyz": "mm", "pose_rotation": "deg",
                  "gripper": "percent", "depth": "mm"},
        "hand_eye_validated": False, "max_allowed_sync_skew_s": config.max_sync_skew_s,
        "max_allowed_frame_gap_s": 2.5 / fps,
        "depth_storage": "depth/episode_000000/frame_NNNNNN.npz: depth_mm float32, invalid=0",
        "capture_complete": False,
    }
    try:
        robot.connect()
        sample = robot.get_sample()
        dataset = LeRobotDataset.create(
            repo_id=repo_id, root=root, fps=fps, robot_type=robot.name,
            features=dataset_features(config.camera.width, config.camera.height),
            use_videos=False, image_writer_threads=2,
        )
        depth_dir = root / "depth" / "episode_000000"
        depth_dir.mkdir(parents=True)
        report["camera_metadata"] = sample["camera"]["metadata"]
        start = time.monotonic()
        with (root / "meta" / "capture.jsonl").open("w") as log:
            for i in range(frames):
                if i:
                    time.sleep(max(0.0, start + i / fps - time.monotonic()))
                    sample = robot.get_sample()
                meta, state = sample["camera"]["metadata"], sample["robot_state"]
                captured,basis = camera_clock(meta)
                if i and basis != report['timing_clock']:
                    raise ValueError('Recording clock domain changed')
                report['timing_clock'] = basis
                if times and captured <= times[-1]:
                    raise RuntimeError("Duplicate or backwards camera timestamp")
                if times and captured - times[-1] > report["max_allowed_frame_gap_s"]:
                    raise TimeoutError("Camera frame gap exceeds the recording limit")
                if meta["source_camera_serial"] != report["expected_camera_serial"]:
                    raise RuntimeError("Camera identity changed during recording")
                depth = sample["camera"]["depth_mm"]
                np.savez_compressed(depth_dir / f"frame_{i:06d}.npz", depth_mm=depth)
                dataset.add_frame(make_frame(sample, task))
                times.append(captured)
                skews.append(float(sample["sync_skew_s"]))
                depths.append(float(np.count_nonzero(depth) / depth.size))
                counters.append(int(state["frame_counter"]))
                log.write(json.dumps({
                    "frame_index": i, "camera": meta, "robot": state,
                    "sync_skew_s": skews[-1], "depth_file": f"depth/episode_000000/frame_{i:06d}.npz",
                }) + "\n")
                log.flush()
                if i == 0 or i == frames - 1:
                    print(f"Recorded observation {i + 1}/{frames}", flush=True)
        report["capture_complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        pending_error = exc
    finally:
        try:
            robot.disconnect()
        except BaseException as exc:
            report["capture_complete"] = False
            report["disconnect_error"] = f"{type(exc).__name__}: {exc}"
            pending_error = pending_error or exc
        if dataset is not None:
            try:
                if times:
                    dataset.save_episode()
            except BaseException as exc:
                report["capture_complete"] = False
                report["finalization_error"] = f"{type(exc).__name__}: {exc}"
                pending_error = pending_error or exc
            finally:
                for cleanup in (dataset.finalize, dataset.stop_image_writer):
                    try:
                        cleanup()
                    except BaseException as exc:
                        report["capture_complete"] = False
                        report.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
                        pending_error = pending_error or exc
            report["recorded_frames"] = len(times)
            if len(times) >= 2:
                intervals = np.diff(times)
                report["measured_camera_sample_fps"] = (len(times) - 1) / (times[-1] - times[0])
                report["camera_interval_s"] = {"median": float(np.median(intervals)), "max": float(max(intervals))}
            if times:
                report["sync_abs_skew_ms"] = {"median": float(np.median(np.abs(skews)) * 1000),
                                              "max": float(max(np.abs(skews)) * 1000)}
                report["depth_valid_fraction"] = {"min": min(depths), "mean": float(np.mean(depths))}
                report["robot_frame_counters"] = counters
            (root / "meta" / "recording.json").write_text(json.dumps(report, indent=2) + "\n")
    if pending_error is not None:
        raise pending_error
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo-id", default="local/fr5_gemini_observations")
    p.add_argument("--host", default="192.168.58.2")
    p.add_argument("--camera-url", default="http://127.0.0.1:8765/snapshot.npz")
    p.add_argument("--serial", default="CP1E54200015")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()
    cfg = FairinoFR5Config(id=f"fr5_{args.host.replace('.', '_')}", host=args.host,
        camera=GeminiBridgeConfig(url=args.camera_url, expected_serial=args.serial,
                                 width=args.width, height=args.height, fps=args.fps))
    result = record(cfg, args.root, repo_id=args.repo_id, fps=args.fps, frames=args.frames)
    from .verify import verify_dataset
    verification = verify_dataset(args.root)
    print(json.dumps({"capture": result, "verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
