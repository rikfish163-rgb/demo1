"""Reusable read-only RGB-D and J1 state capture. No actuator RPC interface."""
import io
import json
import time
import urllib.request
from pathlib import Path


def capture(destination, *, decision_record=None):
    import cv2
    import numpy as np
    from fr5_runtime.robot import read_state
    from .rgbd_geometry import color_in_depth_grid, optics
    from .calibration import load_readonly_calibration

    destination = Path(destination)
    if destination.exists():
        raise RuntimeError("Observation already exists; do not overwrite evidence")
    before = read_state()
    raw = urllib.request.urlopen("http://127.0.0.1:8765/snapshot.npz", timeout=5).read()
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        color, depth = data["color_bgr"], data["depth_mm"]
    source_mono = meta['capture'].get('received_monotonic')
    age = time.monotonic()-source_mono if source_mono is not None else time.time()-meta['capture']['time_unix']
    if (meta["camera"]["serial"] != "CP1E54200015" or not 0 <= age < 2
            or color.shape != (1080, 1920, 3) or depth.shape != (1080, 1920)
            or np.mean(depth > 0) < .2):
        raise RuntimeError("Camera identity, native size, depth or freshness check failed")
    after = read_state()
    calibration = load_readonly_calibration(camera_serial=meta["camera"]["serial"])
    K, D = optics(meta["color_intrinsic"], meta["color_distortion"])
    if (not np.allclose(K, calibration["K"], atol=1e-6, rtol=0)
            or not np.allclose(D, calibration["D"], atol=1e-9, rtol=0)):
        raise ValueError("Current native RGB optics differ from calibrated optics")
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"scene.npz").write_bytes(raw)
    if not cv2.imwrite(str(destination/"color.jpg"), color):
        raise RuntimeError("Could not save observation image")
    rectified=color_in_depth_grid(color,meta)
    if not cv2.imwrite(str(destination/"color-in-depth-grid.jpg"),rectified):
        raise RuntimeError("Could not save RGB in depth pixel coordinates")
    report = {"status": "observation_completed", "decision_record": decision_record,
              "calibration": calibration,
              "camera": meta, "camera_age_at_capture_s": age,
              "robot_before": before, "robot_after": after,
              "robot_position_change_mm": (np.array(after["tcp"][:3])-before["tcp"][:3]).tolist(),
              "depth_valid_fraction": float(np.mean(depth > 0)),
              "pixel_geometry": {"raw_rgb": "color.jpg", "rgb_matching_depth_grid": "color-in-depth-grid.jpg",
                                  "depth_profile_explicit": True, "metric_accuracy_validated": False},
              "motion_commands_sent": 0, "gripper_commands_sent": 0,
              "completed_at_unix": time.time()}
    (destination/"observation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
