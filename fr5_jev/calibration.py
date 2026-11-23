"""Pinned camera-to-J1 calibration for read-only observations, never actuation."""
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "hardware/calibration/camera-to-j1-mm.json"
DIRECTION = "p_J1_base_mm = R @ p_color_camera_mm + t_mm"


def load_readonly_calibration(*, camera_serial=None):
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    if (source.get("robot") != "J1" or source.get("camera_serial") != "CP1E54200015"
            or source.get("direction") != DIRECTION
            or source.get("heldout_geometry_pass") is not True):
        raise ValueError("Camera-to-J1 calibration identity, direction or validation mismatch")
    if camera_serial is not None and camera_serial != source["camera_serial"]:
        raise ValueError("Observation camera differs from calibrated camera")
    matrix = np.asarray(source["base_from_color_camera_mm"], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("Invalid rigid camera-to-J1 transform")
    return {
        "source": str(SOURCE), "sha256": hashlib.sha256(raw).hexdigest(),
        "source_status": source["status"], "use": "read_only_localization",
        "robot": source["robot"], "camera_serial": source["camera_serial"],
        "direction": DIRECTION, "base_from_color_camera_mm": matrix.tolist(),
        "K": source["K"], "D": source["D"], "square_mm": source["square_mm"],
        "handeye_geometry_validated_in_sampled_region": True,
        "heldout_rms_mm": source["heldout_rms_mm"],
        "heldout_max_point_mm": source["heldout_max_point_mm"],
        "report": str(SOURCE.with_name("calibration-report.json")),
        "requires_camera_and_robot_base_unchanged": True,
        "depth_accuracy_validated": False, "grasp_tcp_validated": False,
        "end_to_end_grasp_accuracy_validated": False,
        "actuation_authorized_by_this_record": False,
    }
