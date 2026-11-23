"""Read back a local observation dataset without contacting any hardware."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .robot_fairino import STATE_NAMES, STATUS_NAMES
from lerobot_camera_gemini.timing import camera_clock, MONOTONIC


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_dataset(root, *, write_report=True):
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "meta/recording.json").read_text())
    _require(manifest["capture_complete"], "Capture is incomplete")
    _require(manifest["dataset_kind"] == "observation_only", "Unexpected dataset kind")
    _require(manifest["has_action_labels"] is False, "Unexpected action labels")
    _require(manifest["robot_commands_sent"] == 0, "Recorder reports commands")
    rows = [json.loads(s) for s in (root / "meta/capture.jsonl").read_text().splitlines()]
    dataset = LeRobotDataset(repo_id=manifest["repo_id"], root=root)
    count = len(dataset)
    _require(count == manifest["recorded_frames"] == manifest["requested_frames"] == len(rows),
             "Recorded frame counts disagree")
    _require(count >= 2 and dataset.num_episodes == 1, "Expected one episode with at least two frames")
    _require(not any(k == "action" or k.startswith("action.") for k in dataset.features),
             "Observation dataset contains action features")
    height, width, channels = dataset.features["observation.images.front"]["shape"]
    states, captured_times, skews, depth_fractions = [], [], [], []
    first_rgb = None
    for i, row in enumerate(rows):
        frame = dataset[i]
        _require(row["frame_index"] == i and frame["frame_index"].item() == i, "Frame index mismatch")
        _require(frame["episode_index"].item() == 0, "Unexpected episode index")
        rgb = np.asarray(frame["observation.images.front"])
        _require(rgb.shape == (channels, height, width) and np.isfinite(rgb).all(), "Invalid RGB tensor")
        _require(rgb.min() >= 0 and rgb.max() <= 1, "RGB tensor is not normalized to 0..1")
        state = np.asarray(frame["observation.state"])
        _require(state.shape == (7,) and np.isfinite(state).all(), "Invalid joint/gripper vector")
        expected_state = [*row["robot"]["joints"], row["robot"]["gripper_position"]]
        _require(np.allclose(state, expected_state, rtol=0, atol=1e-4), "State disagrees with raw capture")
        _require(0 <= state[-1] <= 100, "Invalid gripper opening")
        for key, source in (("tcp_pose", "tcp"), ("flange_pose", "flange"),
                            ("joint_velocity", "joint_speeds_deg_s")):
            values = np.asarray(frame[f"observation.{key}"])
            _require(values.shape == (6,) and np.isfinite(values).all(), f"Invalid {key}")
            _require(np.allclose(values, row["robot"][source], rtol=0, atol=1e-4), f"{key} mismatch")
        status = np.asarray(frame["observation.robot_status"])
        _require(np.array_equal(status, [row["robot"][k] for k in STATUS_NAMES]), "Status mismatch")
        timestamp_us = np.asarray(frame["observation.source_time_us"])
        _require(timestamp_us.shape == (3,) and timestamp_us.dtype == np.int64, "Invalid source times")
        timestamps = timestamp_us.astype(np.float64) / 1_000_000
        camera_time = row["camera"]["source_timestamp_unix"]
        robot_time = row["robot"]["timestamp"]
        _require(np.allclose(timestamps[:2], [camera_time, robot_time], rtol=0, atol=1e-6),
                 "Source timestamps lost precision or changed")
        timeline,basis=camera_clock(row['camera'])
        _require(basis==manifest.get('timing_clock','host_unix'),'Recording clock domain mismatch')
        if basis==MONOTONIC:
            mono=np.asarray(frame['observation.source_monotonic_us']).astype(np.float64)/1e6
            _require(np.allclose(mono[:2],[timeline,row['robot']['received_monotonic']],rtol=0,atol=1e-6),'Monotonic timestamps mismatch')
            _require(mono[2]>=timeline-1e-6,'Collector monotonic time precedes camera')
            _require(frame['observation.monotonic_timing_valid'].item()==1,'Monotonic validity flag mismatch')
            robot_pair_time=row['robot']['received_monotonic']
        else:
            _require(timestamps[2] >= camera_time - 1e-6, "Collector timestamp precedes camera")
            robot_pair_time=robot_time
        skew = float(frame["observation.sync_skew_s"].item())
        _require(abs(skew - (robot_pair_time - timeline)) < 1e-6, "Synchronization skew mismatch")
        _require(abs(skew) <= manifest["max_allowed_sync_skew_s"], "Synchronization limit exceeded")
        _require(row["camera"]["source_camera_serial"] == manifest["expected_camera_serial"],
                 "Camera identity mismatch")
        depth_path = root / "depth/episode_000000" / f"frame_{i:06d}.npz"
        _require(row["depth_file"] == str(depth_path.relative_to(root)), "Unexpected depth path")
        with np.load(depth_path, allow_pickle=False) as packet:
            depth = packet["depth_mm"]
        _require(depth.shape == (height, width) and depth.dtype == np.float32, "Invalid depth layout")
        _require(np.isfinite(depth).all() and np.all(depth >= 0), "Invalid depth values")
        states.append(state)
        captured_times.append(timeline)
        skews.append(skew)
        depth_fractions.append(float(np.count_nonzero(depth) / depth.size))
        if first_rgb is None:
            first_rgb = np.rint(rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
    gaps = np.diff(captured_times)
    _require(np.all(gaps > 0), "Repeated or backwards camera frames")
    _require(np.max(gaps) <= manifest["max_allowed_frame_gap_s"], "Camera frame gap exceeds limit")
    _require(len(list((root / "depth/episode_000000").glob("frame_*.npz"))) == count,
             "Depth file count mismatch")
    result = {
        "passed": True, "verified_utc": datetime.now(timezone.utc).isoformat(),
        "frames": count, "episodes": dataset.num_episodes, "rgb_shape_hwc": [height, width, channels],
        "depth_frames": count, "camera_serial": manifest["expected_camera_serial"],
        "measured_camera_sample_fps": float((count - 1) / (captured_times[-1] - captured_times[0])),
        "max_host_timestamp_skew_ms": float(np.max(np.abs(skews)) * 1000),
        "joint_gripper_range": dict(zip(STATE_NAMES, np.ptp(np.stack(states), axis=0).tolist())),
        "minimum_depth_valid_fraction": min(depth_fractions),
        "has_action_labels": False, "training_ready": False,
        "limitation": "Host timestamp matching only; no hardware synchronization or hand-eye validation",
    }
    if write_report:
        (root / "meta/verification.json").write_text(json.dumps(result, indent=2) + "\n")
        Image.fromarray(first_rgb).save(root / "preview.png")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.root), indent=2))


if __name__ == "__main__":
    main()
