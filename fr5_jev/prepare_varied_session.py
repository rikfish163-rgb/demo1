"""Prepare a fresh varied-start FR5 cup episode without moving the robot.

The first run captures two fresh RGB-D observations, estimates the cup center
and table plane, builds a bounded fixed-attitude pick/place route, samples the
controller IK/FK, and writes a route overlay for visual review.  A later
``--finalize`` pass creates the one-shot scope only after that review exists.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from fr5_runtime.robot import controller
from fr5_runtime.kinematics import transform
from fr5_runtime.robot import HOST, read_state, require_ready
from fr5_runtime.kinematics import model_chain

from .observe import capture
from .rgbd_geometry import depth_pixels_to_camera_points, models
from .durable_storage import LOCAL_BASE, REMOTE_BASE
from .stream_storage import LOCAL_STREAM_BASE
from .motion_profiles import (CONTINUOUS_PROFILE, LEGACY_PROFILE, PROFILE_SPEEDS,
                             matching_profiles, motion_limits, record_profile,
                             validate_profile)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "hardware/fr5.urdf"
TOOL_REFERENCE = ROOT / "hardware/tool-reference.json"


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_scene(directory: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    report = json.loads((directory / "observation.json").read_text())
    with np.load(directory / "scene.npz", allow_pickle=False) as scene:
        color = scene["color_bgr"].copy()
        depth = scene["depth_mm"].copy()
    calibration = np.asarray(report["calibration"]["base_from_color_camera_mm"], float)
    if calibration.shape != (4, 4) or report["camera"]["camera"]["serial"] != "CP1E54200015":
        raise RuntimeError("Fresh observation lacks the expected calibrated camera")
    return report, color, depth, calibration


def camera_point(uv: np.ndarray, depth_mm: np.ndarray, report: dict, calibration: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, float).reshape(-1, 2)
    points = depth_pixels_to_camera_points(uv, depth_mm, report["camera"])
    return points @ calibration[:3, :3].T + calibration[:3, 3]


def fit_table(color: np.ndarray, depth: np.ndarray, report: dict, calibration: np.ndarray,
              cup_uv: np.ndarray) -> tuple[np.ndarray, float, dict]:
    y, x = np.mgrid[180:1040:12, 80:1840:12]
    uv = np.column_stack([x.ravel(), y.ravel()])
    z = depth[uv[:, 1], uv[:, 0]].astype(float)
    points = camera_point(uv[z > 0], z[z > 0], report, calibration)
    radial_pixels = np.linalg.norm(uv[z > 0] - cup_uv, axis=1)
    keep = (
        (z[z > 0] > 650)
        & (z[z > 0] < 1350)
        & (radial_pixels > 140)
        & (points[:, 2] > -45)
        & (points[:, 2] < 45)
    )
    points = points[keep]
    if len(points) < 2000:
        raise RuntimeError("Not enough unobstructed table depth samples")
    rng = np.random.default_rng(1793195483)
    best = None
    for _ in range(800):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        offset = float(normal @ sample[0])
        residual = np.abs(points @ normal - offset)
        count = int(np.count_nonzero(residual < 2.0))
        if best is None or count > best[0]:
            best = (count, normal, offset)
    if best is None or best[0] < 1000:
        raise RuntimeError("Table plane fit did not converge")
    _, normal, offset = best
    residual = np.abs(points @ normal - offset)
    inliers = points[residual < 2.0]
    for _ in range(4):
        center = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        offset = float(normal @ center)
        residual = np.abs(points @ normal - offset)
        inliers = points[residual < 2.0]
    rms = float(np.sqrt(np.mean((inliers @ normal - offset) ** 2)))
    if rms > 3.0 or len(inliers) < 1000:
        raise RuntimeError("Table plane residual or support is insufficient")
    return normal, offset, {
        "normal": normal.tolist(),
        "offset_mm": offset,
        "inliers": len(inliers),
        "rms_mm": rms,
    }


def find_cup_rim(color: np.ndarray) -> tuple[np.ndarray, float, dict]:
    x0, y0, x1, y1 = 1040, 700, 1570, 1060
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1], (9, 9), 2)
    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=80,
        param2=30,
        minRadius=45,
        maxRadius=100,
    )
    if circles is None:
        raise RuntimeError("Cup rim circle was not found")
    candidates = []
    for x, y, radius in circles[0]:
        u, v = float(x + x0), float(y + y0)
        candidates.append((u, v, float(radius)))
    if not candidates:
        raise RuntimeError("Cup rim candidate list is empty")
    u, v, radius = min(candidates, key=lambda item: abs(item[2] - 62.0))
    if not 1100 < u < 1500 or not 730 < v < 1040:
        raise RuntimeError("Detected cup lies outside the current right-hand work area")
    return np.array([u, v]), radius, {
        "color_uv": [u, v],
        "detected_radius_px": radius,
        "method": "single dominant circular edge in the current right-hand cup region",
    }


def estimate_cup(report: dict, depth: np.ndarray, calibration: np.ndarray, rim_uv: np.ndarray,
                 table_normal: np.ndarray, table_offset: float) -> tuple[np.ndarray, np.ndarray, dict]:
    intr = report["camera"]["color_intrinsic"]
    distortion = report["camera"]["color_distortion"]
    k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    d = np.array([distortion[key] for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], float)
    normalized = cv2.undistortPoints(rim_uv.reshape(1, 1, 2), k, d).reshape(2)
    ray_camera = np.r_[normalized, 1.0]
    origin = calibration[:3, 3]
    ray = calibration[:3, :3] @ ray_camera
    rim_plane_offset = table_offset + 110.0
    denominator = float(ray @ table_normal)
    if abs(denominator) < 1e-6:
        raise RuntimeError("Cup center ray is parallel to the table plane")
    scale = (rim_plane_offset - origin @ table_normal) / denominator
    rim_center = origin + scale * ray
    base_center = rim_center - 110.0 * table_normal
    return base_center, rim_center, {
        "base_center_mm": base_center.tolist(),
        "rim_center_mm": rim_center.tolist(),
        "height_mm": 110.0,
        "rim_outer_diameter_mm": 75.0,
        "method": "current RGB circle ray intersected with measured table normal, minus known 110mm height",
    }


def build_routes(start: np.ndarray, cup_base: np.ndarray, table_normal: np.ndarray,
                 *, profile: str = LEGACY_PROFILE, prealigned: bool = False,
                 grasp_reference: dict | None = None) -> tuple[dict, dict]:
    # Level the tool while preserving the measured approach azimuth.  The
    # controller RPY representation is continuous here: pitch is reduced from
    # its fresh measured value to zero and yaw is retained.
    validate_profile(profile)
    # In the continuous profile preserve the actual attitude throughout.
    # A combined translation/leveling command on this controller produced a
    # measured 0.52mm transient Z deviation. Do not relax the tracking bound.
    orientation = (np.asarray(start[3:], float).copy() if profile == CONTINUOUS_PROFILE
                   else np.array([90.0, 0.0, start[5]], float))
    rotation = transform(np.r_[start[:3], orientation])[:3, :3]
    tool_axis = rotation[:, 2]
    if abs(tool_axis[2]) > 0.15 or np.linalg.norm(tool_axis[:2]) < 0.9:
        raise RuntimeError("Current tool axis is not a bounded horizontal approach")
    boxes = json.loads(TOOL_REFERENCE.read_text())["tool_boxes_for_this_exterior_move"]
    left = np.asarray(boxes['-1']['local_bounds_mm'], float)
    right = np.asarray(boxes['1']['local_bounds_mm'], float)
    if left.shape != (2,3) or right.shape != (2,3) or left[1,0] >= right[0,0]:
        raise RuntimeError("Finger model does not define a lateral opening")
    lateral_center = float((left[1,0] + right[0,0]) / 2.0)
    grip_center = cup_base + 87.0 * table_normal
    # Use the fitted midpoint of the two modeled inner faces rather than
    # silently assuming the flange's local X is the jaw center.  Both this
    # estimate and the 162mm fingertip-as-grasp-center assumption remain
    # unverified physical geometry; this only creates a candidate route.
    contact = np.r_[grip_center - 162.0 * tool_axis - lateral_center * rotation[:,0], orientation]
    if grasp_reference is not None:
        if profile != CONTINUOUS_PROFILE or prealigned:
            raise ValueError('Observed grasp reference requires a new continuous approach')
        offset = np.asarray(grasp_reference['effective_grasp_offset_flange_mm'], float)
        if (offset.shape != (3,) or not np.isfinite(offset).all()
                or abs(offset[0]) > 30 or abs(offset[1]) > 30 or not 120 < offset[2] < 210):
            raise ValueError('Observed grasp offset outside tool-sized candidate bounds')
        contact = np.r_[grip_center - rotation @ offset, orientation]
    if prealigned:
        if profile != CONTINUOUS_PROFILE:
            raise ValueError('Visual prealigned entry requires the continuous workflow')
        contact = np.asarray(start, float).copy()
    clear = contact.copy()
    clear[:3] -= 80.0 * tool_axis
    near = contact.copy()
    near[:3] -= 20.0 * tool_axis
    transfer = clear.copy()
    transfer[2] = max(start[2], clear[2] + 70.0)
    probe = contact.copy()
    probe[2] += 10.0
    lift = contact.copy()
    lift[2] += 50.0
    low = contact.copy()
    low[2] += 2.0
    retreat_probe = contact.copy()
    retreat_probe[:3] -= 10.0 * tool_axis
    routes = {
        "approach_clear": [start, transfer, clear],
        "approach_near": [clear, near],
        "approach_contact": [near, contact],
        "close": [contact, contact],
        "lift_probe": [contact, probe],
        "lift_full": [probe, lift],
        "return": [lift, probe, low, contact],
        "release": [contact, contact],
        "retreat_probe": [contact, retreat_probe],
        "retreat_clear": [retreat_probe, clear],
        "retreat_high": [clear, transfer, start],
    }
    if prealigned:
        for phase in ('approach_clear', 'approach_near', 'approach_contact'):
            routes[phase] = [start.copy(), start.copy()]
        routes['retreat_high'] = [clear, transfer]
    model = {
        "tool_axis_base": tool_axis.tolist(),
        "grip_center_target_mm": grip_center.tolist(),
        "flange_to_grip_center_assumption_mm": 162.0,
        "grip_center_assumption_basis": "162mm was measured to fingertip, not independently verified as grasp-center TCP",
        "lateral_jaw_center_estimate_mm": lateral_center,
        "lateral_jaw_center_basis": "midpoint of fitted inner finger-box faces; not an independently verified TCP",
        "contact_pose": contact.tolist(),
        "clear_pose": clear.tolist(),
        "near_pose": near.tolist(),
        "transfer_pose": transfer.tolist(),
        "fixed_attitude": orientation.tolist(),
        "orientation_policy": "preserve_measured" if profile == CONTINUOUS_PROFILE else "level_before_contact",
        "prealigned_start": prealigned,
        "observed_grasp_reference": grasp_reference,
    }
    return {key: [np.asarray(point).tolist() for point in route] for key, route in routes.items()}, model


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _pose_intervals(source: np.ndarray, target: np.ndarray) -> int:
    distance = float(np.linalg.norm(target[:3] - source[:3]))
    rotation = float(np.max(np.abs(target[3:] - source[3:])))
    return max(2, math.ceil(distance), math.ceil(rotation / 0.25))


def sample_geometry(routes: dict, start_joints: list[float]) -> tuple[dict, dict]:
    rpc = controller()
    raw_limits = rpc.GetJointSoftLimitDeg(1)
    if len(raw_limits) != 13 or raw_limits[0] != 0:
        raise RuntimeError("Joint soft limits unavailable")
    limits = np.asarray(raw_limits[1:], float).reshape(6, 2)
    joints = [joint for joint in ET.parse(URDF).getroot().findall("joint") if joint.get("type") == "revolute"]
    if len(joints) != 6:
        raise RuntimeError("Expected six revolute FR5 joints")
    initial_chain = np.asarray(model_chain(start_joints, joints)[:7], float)
    reference = np.asarray(start_joints, float)
    rows = []
    for phase, route in routes.items():
        for segment, (source, target) in enumerate(zip(route, route[1:])):
            source_array, target_array = np.asarray(source), np.asarray(target)
            steps = _pose_intervals(source_array, target_array)
            for alpha in np.linspace(0.0, 1.0, steps + 1):
                pose = source_array + alpha * (target_array - source_array)
                if alpha == 0.0 and reference is not None and segment == 0:
                    # The first sampled point is the already measured start.
                    ik = [0.0, *reference.tolist()]
                else:
                    ik = rpc.GetInverseKinRef(0, pose.tolist(), reference.tolist())
                if len(ik) != 7 or ik[0] != 0:
                    raise RuntimeError(f"IK failed in {phase} segment {segment}")
                q = np.asarray(ik[1:], float)
                fk = rpc.GetForwardKin(q.tolist())
                if len(fk) != 7 or fk[0] != 0:
                    raise RuntimeError(f"FK failed in {phase} segment {segment}")
                margin = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
                step = float(np.max(np.abs(q - reference)))
                position_error = float(np.linalg.norm(np.asarray(fk[1:4]) - pose[:3]))
                rotation_error = float(np.max(np.abs(angle_delta(np.asarray(fk[4:]), pose[3:]))))
                chain = np.asarray(model_chain(q.tolist(), joints)[:7], float)
                if (margin < 5.0 or step > 1.0 or position_error > 0.2
                        or rotation_error > 0.05 or abs(math.sin(math.radians(q[4]))) < 0.2):
                    raise RuntimeError(
                        f"Kinematic boundary failed in {phase}: margin={margin:.3f}, "
                        f"step={step:.3f}, position={position_error:.3f}, rotation={rotation_error:.3f}"
                    )
                rows.append({
                    "phase": phase,
                    "segment": segment,
                    "fraction": float(alpha),
                    "tcp": pose.tolist(),
                    "q": q.tolist(),
                    "joint_margin_deg": margin,
                    "joint_step_deg": step,
                    "fk_error_mm": position_error,
                    "fk_rotation_error_deg": rotation_error,
                    "link_center_displacement_mm": float(
                        np.linalg.norm(chain - initial_chain, axis=1).max()
                    ),
                    "urdf_flange_error_mm": float(np.linalg.norm(chain[6] - pose[:3])),
                })
                reference = q
    summary = {
        "minimum_joint_margin_deg": min(row["joint_margin_deg"] for row in rows),
        "maximum_joint_step_deg": max(row["joint_step_deg"] for row in rows),
        "max_fk_error_mm": max(row["fk_error_mm"] for row in rows),
        "max_fk_rotation_error_deg": max(row["fk_rotation_error_deg"] for row in rows),
        "max_link_center_displacement_mm": max(row["link_center_displacement_mm"] for row in rows),
        "max_urdf_flange_error_mm": max(row["urdf_flange_error_mm"] for row in rows),
    }
    return rows, summary


def build_manual_action_plan(routes: dict, path_rows: list[dict], initial_joints: list[float],
                             profile: str = LEGACY_PROFILE) -> dict:
    """Materialize every bounded controller command before execution.

    The wire shape follows the local v2.1.4 XML-RPC bindings, checked against
    the vendor 3.8.4-series SDK/manual.  ``acc`` is kept at the historical
    manual's unavailable/default value; speed and override are explicit.
    """
    validate_profile(profile)
    actions = []
    rpc = controller()
    by_phase_segment = {}
    for row in path_rows:
        by_phase_segment.setdefault((row["phase"], row["segment"]), []).append(row)

    def lookup_q(phase: str, segment: int, fraction: float) -> list[float]:
        rows = by_phase_segment[(phase, segment)]
        row = min(rows, key=lambda item: abs(item["fraction"] - fraction))
        # The nearest sampled IK is a seed, not the exact target.  Solve again
        # for the actual wire pose and independently check forward kinematics.
        source, target = map(lambda p: np.asarray(p, float), routes[phase][segment:segment + 2])
        pose = source + fraction * (target - source)
        ik = rpc.GetInverseKinRef(0, pose.tolist(), list(row["q"]))
        if len(ik) != 7 or ik[0] != 0:
            raise RuntimeError(f"Exact action IK failed in {phase}")
        q = np.asarray(ik[1:], float)
        fk = rpc.GetForwardKin(q.tolist())
        if (not np.isfinite(q).all() or len(fk) != 7 or fk[0] != 0
                or np.linalg.norm(np.asarray(fk[1:4]) - pose[:3]) > .2
                or max(abs(angle_delta(fk[4:], pose[3:]))) > .05):
            raise RuntimeError(f"Exact action FK mismatch in {phase}")
        return q.tolist()

    def append_move(phase: str, segment: int, source: np.ndarray, target: np.ndarray,
                    fraction: float, speed: float, target_q: list[float]) -> None:
        action = {
            "id": f"{phase}_{segment + 1:02d}_{len(actions) + 1:03d}",
            "phase": phase,
            "kind": "MoveL",
            "blocking": True,
            "source_pose": source.tolist(),
            "target_pose": target.tolist(),
            "joint_target_deg": target_q,
            "wire": {
                "joint_pos": target_q,
                "desc_pos": target.tolist(),
                "tool": 0,
                "user": 0,
                "vel": speed,
                "acc": 0.0,
                "ovl": 100.0,
                "blendR": -1.0,
                "blendMode": 0,
                "exaxis_pos": [0.0, 0.0, 0.0, 0.0],
                "search": 0,
                "offset_flag": 0,
                "offset_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "manual_basis": [
                "FR5 3.8.4-series operation/programming chapters 9.4-9.7",
                "FR5 3.8.4-series Python SDK MoveL: mm and degrees, tool0/user0, blocking blendR=-1",
            ],
            "realtime_checks": [
                "fresh camera and shared J1 state before dispatch",
                "recomputed IK/FK on the exact wire target",
                "TCP segment distance, direction and overshoot corridor",
                "joint branch, limit and wrist-singularity margins",
                "motion_done=1, command queue empty, collision/stop/fault fields zero",
            ],
            "on_deviation": "StopMotion once, mark uncertain, and require a fresh observation/LLM review",
        }
        actions.append(action)

    for phase, route in routes.items():
        if phase in ("close", "release"):
            continue
        for segment, (source, target) in enumerate(zip(route, route[1:])):
            source_array = np.asarray(source, float)
            target_array = np.asarray(target, float)
            distance = float(np.linalg.norm(target_array[:3] - source_array[:3]))
            if distance <= 0.03:
                continue
            support_z = float(routes["return"][-1][2]) if phase == "return" else 0.0
            boundaries = [0.0, 1.0]
            if profile == CONTINUOUS_PROFILE and phase == "return":
                dz = target_array[2] - source_array[2]
                if abs(dz) > 1e-12:
                    crossing = float((support_z + 2.0 - source_array[2]) / dz)
                    if 0.0 < crossing < 1.0 and min(crossing, 1.0 - crossing) * distance > .03:
                        boundaries.insert(1, crossing)
            previous = source_array
            for start, end in zip(boundaries, boundaries[1:]):
                section_source = source_array + start * (target_array - source_array)
                section_target = source_array + end * (target_array - source_array)
                section_distance = float(np.linalg.norm(section_target[:3] - section_source[:3]))
                step, _ = motion_limits(profile, phase, section_source, section_target, support_z)
                for alpha in np.linspace(start, end, math.ceil(section_distance / step) + 1)[1:]:
                    pose = source_array + alpha * (target_array - source_array)
                    # Legacy chose the speed for the entire original route
                    # segment, even when its intermediate targets differ.
                    speed_target = target_array if profile == LEGACY_PROFILE else pose
                    _, category = motion_limits(profile, phase, previous, speed_target, support_z)
                    speed = PROFILE_SPEEDS[profile][category]
                    append_move(phase, segment, previous, pose, float(alpha), speed,
                                lookup_q(phase, segment, float(alpha)))
                    previous = pose

    close_pose = np.asarray(routes["close"][0], float)
    for index, target in enumerate((90, 80, 76), 1):
        actions.append({
            "id": f"close_{index:02d}",
            "phase": "close",
            "kind": "MoveGripper",
            "blocking": False,
            "stationary_tcp": close_pose.tolist(),
            "wire": {
                "index": 1,
                "pos": target,
                "vel": 5,
                "force": 10,
                "maxtime": 10000,
                "block": 1,
                "type": 0,
                "rotNum": 0.0,
                "rotVel": 0,
                "rotTorque": 0,
            },
            "stop_condition": "gripper_motiondone=2 means stable external contact; do not send the next closing target",
            "realtime_checks": [
                "arm stationary at reviewed contact pose",
                "fresh gripper feedback and no fault",
                "poll GetGripperMotionDone and full state stream",
                "status3 or invalid busy-zero feedback stops the phase",
            ],
            "on_deviation": "stop arm motion, preserve held/uncertain state, request fresh observation",
        })

    release_pose = np.asarray(routes["release"][0], float)
    actions.append({
        "id": "release_01",
        "phase": "release",
        "kind": "MoveGripper",
        "blocking": False,
        "stationary_tcp": release_pose.tolist(),
        "wire": {
            "index": 1,
            "pos": 100,
            "vel": 10,
            "force": 10,
            "maxtime": 10000,
            "block": 1,
            "type": 0,
            "rotNum": 0.0,
            "rotVel": 0,
            "rotTorque": 0,
        },
        "realtime_checks": [
            "only at reviewed original-support return pose",
            "arm stationary and cup visually supported",
            "poll position/status until fully open",
            "any fault or unexpected arm motion stops the phase",
        ],
        "on_deviation": "keep the last pose, do not repeat release blindly, request fresh observation",
    })

    phase_order = list(routes)
    actions.sort(key=lambda action: phase_order.index(action["phase"]))
    for action in actions:
        if action["kind"] == "MoveL":
            if len(action["wire"]["joint_pos"]) != 6 or len(action["wire"]["desc_pos"]) != 6:
                raise ValueError("MoveL action does not have six joint and pose values")
        elif action["kind"] == "MoveGripper" and action["wire"]["block"] != 1:
            raise ValueError("Gripper actions are deliberately nonblocking and feedback-polled")
    return {
        "schema": "fr5-manual-action-plan-v2",
        "profile": profile,
        "controller_family": "FAIRINO FR5 V5.0",
        "software_basis": "controller v3.8.4.2; matched to public 3.8.4-series manual",
        "initial_joints_deg": initial_joints,
        "tool": 0,
        "user": 0,
        "global_speed_changes": None,
        "principle": (
            "Every controller command is precomputed and bounded. Jev selects a task phase; "
            "deterministic code executes its recorded actions with live state/camera checks; "
            "the LLM handles fresh observations, deviations, conflicts and failed completion gates."
        ),
        "actions": actions,
        "emergency_stop_on_arm_uncertainty": {
            "method": "StopMotion",
            "no_retry": True,
        },
        "manual_sources": [
            "https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/coding.html",
            "https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/robot_peripherals.html",
            "https://fairino-doc-zhs.readthedocs.io/3.8.4/SDKManual/index.html",
        ],
    }


def table_clearances(routes: dict, table_normal: np.ndarray, table_offset: float) -> dict:
    boxes = json.loads(TOOL_REFERENCE.read_text())["tool_boxes_for_this_exterior_move"]
    minima = {name: float("inf") for name in boxes}
    for route in routes.values():
        for source, target in zip(route, route[1:]):
            source_array, target_array = np.asarray(source), np.asarray(target)
            for alpha in np.linspace(0.0, 1.0, _pose_intervals(source_array, target_array) + 1):
                pose = source_array + alpha * (target_array - source_array)
                rotation = transform(pose)[:3, :3]
                for name, box in boxes.items():
                    padding = float(box["padding_mm"])
                    bounds = np.asarray(box["local_bounds_mm"], float) + [[-padding] * 3, [padding] * 3]
                    corners = np.asarray(list(itertools.product(*zip(*bounds))), float)
                    points = corners @ rotation.T + pose[:3]
                    clearances = points @ table_normal - table_offset
                    minima[name] = min(minima[name], float(clearances.min()))
    if min(minima.values()) < 5.0:
        raise RuntimeError(f"Conservative tool boxes lack table clearance: {minima}")
    return minima


def cup_tool_collision_checks(routes: dict, cup_base: np.ndarray, table_normal: np.ndarray) -> dict:
    boxes = json.loads(TOOL_REFERENCE.read_text())["tool_boxes_for_this_exterior_move"]
    contact_pose = np.asarray(routes["approach_contact"][-1], float)
    rotation = transform(contact_pose)[:3, :3]
    sampled = {}
    for name, box in boxes.items():
        # Padding is deliberately conservative for table clearance.  For an
        # actual contact-intersection decision, use the measured box itself;
        # inflated finger boxes would always intrude into the nominal 2.5mm
        # per-side cup gap and would incorrectly reject every feasible grasp.
        bounds = np.asarray(box["local_bounds_mm"], float)
        axes = [np.linspace(low, high, 9) for low, high in zip(bounds[0], bounds[1])]
        local = np.asarray(list(itertools.product(*axes)), float)
        world = local @ rotation.T + contact_pose[:3]
        relative = world - cup_base
        height = relative @ table_normal
        horizontal = relative - np.outer(height, table_normal)
        radial = np.linalg.norm(horizontal, axis=1)
        occupied = (height > -3.0) & (height < 113.0) & (radial < 37.5)
        sampled[name] = {
            "sample_points_inside_min_cup_cylinder": int(occupied.sum()),
            "minimum_radial_clearance_mm_when_overlapping_height": float(
                (radial[(height > -3.0) & (height < 113.0)] - 37.5).min()
            ) if np.any((height > -3.0) & (height < 113.0)) else None,
        }
    finger_names = [name for name in sampled if name in ("-1", "1")]
    overlap_count = sum(
        sampled[name]["sample_points_inside_min_cup_cylinder"] for name in finger_names
    )
    # The fitted finger boxes are conservative and were derived from another
    # view; their small nominal overlap cannot replace a fresh near-contact
    # image. Keep this as an explicit warning and require the normal phase
    # review before contact rather than treating an approximate box as proof of
    # either collision or clearance.
    for name in finger_names:
        sampled[name]["near_phase_image_review_required"] = overlap_count > 0
    sampled["summary"] = {
        "finger_sample_overlap_count": overlap_count,
        "coverage": "nominal contact pose only; not a swept-volume check",
        "near_phase_image_review_required": overlap_count > 0,
        "contact_execution_allowed_by_this_check_alone": False,
    }
    # Diagnostic only: sample the complete empty approach at <=2mm intervals.
    # The fitted boxes, cup model and calibration are not independently bounded,
    # and finite point samples do not certify a collision-free swept volume.
    swept = {name: {"sample_points_inside_nominal_cup": 0, "poses_with_overlap": 0,
                    "minimum_radial_clearance_mm": None} for name in boxes}
    local_points = {}
    for name, box in boxes.items():
        bounds = np.asarray(box["local_bounds_mm"], float)
        local_points[name] = np.asarray(list(itertools.product(
            *[np.linspace(low, high, 9) for low, high in zip(bounds[0], bounds[1])])), float)
    pose_count = 0
    for phase in ("approach_clear", "approach_near", "approach_contact"):
        for start, end in zip(routes[phase], routes[phase][1:]):
            start, end = np.asarray(start, float), np.asarray(end, float)
            steps = max(1, math.ceil(np.linalg.norm(end[:3] - start[:3]) / 2.0))
            for alpha in np.linspace(0., 1., steps + 1):
                pose = start + alpha * (end - start)
                rotation = transform(pose)[:3, :3]
                pose_count += 1
                for name, local in local_points.items():
                    relative = local @ rotation.T + pose[:3] - cup_base
                    height = relative @ table_normal
                    horizontal = relative - np.outer(height, table_normal)
                    radial = np.linalg.norm(horizontal, axis=1)
                    inside_height = (height > -3.0) & (height < 113.0)
                    overlaps = int(np.count_nonzero(inside_height & (radial < 37.5)))
                    row = swept[name]
                    row["sample_points_inside_nominal_cup"] += overlaps
                    row["poses_with_overlap"] += int(overlaps > 0)
                    if np.any(inside_height):
                        clearance = float((radial[inside_height] - 37.5).min())
                        old = row["minimum_radial_clearance_mm"]
                        row["minimum_radial_clearance_mm"] = clearance if old is None else min(old, clearance)
    sampled["approach_sweep_diagnostic"] = {
        "pose_count": pose_count,
        "maximum_nominal_translation_step_mm": 2.0,
        "bodies": swept,
        "certifies_clearance": False,
    }
    return sampled


def route_overlay(color: np.ndarray, report: dict, calibration: np.ndarray, cup_base: np.ndarray,
                  rim_center: np.ndarray, rim_uv: np.ndarray, routes: dict) -> np.ndarray:
    image = color.copy()
    _, _, k, d = models(report["camera"])
    inverse = np.linalg.inv(calibration)

    def project(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float).reshape(-1, 3)
        camera = points @ inverse[:3, :3].T + inverse[:3, 3]
        return np.rint(cv2.projectPoints(camera, np.zeros(3), np.zeros(3), k, d)[0]).astype(int).reshape(-1, 2)

    cv2.circle(image, tuple(np.rint(rim_uv).astype(int)), 72, (0, 220, 0), 2)
    colors = {
        "approach_clear": (255, 180, 0),
        "approach_near": (255, 80, 0),
        "approach_contact": (0, 0, 255),
        "lift_full": (0, 255, 255),
        "retreat_high": (180, 0, 255),
    }
    for phase, route in routes.items():
        if phase not in colors:
            continue
        points = project(np.asarray(route)[:, :3])
        cv2.polylines(image, [points], False, colors[phase], 3)
        for point in points:
            cv2.circle(image, tuple(point), 7, colors[phase], -1)
    base_uv = project(cup_base.reshape(1, 3))[0]
    rim_center_uv = project(rim_center.reshape(1, 3))[0]
    cv2.circle(image, tuple(base_uv), 8, (0, 0, 255), -1)
    cv2.circle(image, tuple(rim_center_uv), 8, (0, 255, 255), -1)
    cv2.putText(
        image,
        "Current varied-start route; circles=cup base/rim, red=contact, cyan=lift, green=transfer",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return image


def load_grasp_reference(directory: Path) -> dict:
    """Reuse observed cup/pose correspondence, not old joints or a certified TCP."""
    directory=Path(directory).resolve()
    raw=(directory/'summary.json').read_bytes();summary=json.loads(raw)
    scope_raw=(directory/'scope.json').read_bytes();scope=json.loads(scope_raw)
    geometry_raw=(directory/'geometry.json').read_bytes();geometry=json.loads(geometry_raw)
    if (summary.get('outcome') not in ('pending_review','success') or summary.get('failed')
            or summary.get('released') is not True or summary.get('peak_height_mm',0)<49.9
            or summary.get('scope_sha256')!=hashlib.sha256(scope_raw).hexdigest()
            or scope.get('geometry_sha256')!=hashlib.sha256(geometry_raw).hexdigest()
            or summary.get('completed_stages')!=scope.get('phase_order')):
        raise ValueError('Grasp reference does not contain a completed observed pick/place')
    records=[json.loads(p.read_text()) for p in directory.glob('command-*.json')]
    if not records or any(r.get('status')!='completed' or r.get('outcome',{}).get('return_code')!=0 for r in records):
        raise ValueError('Reference contains uncertain commands')
    closes=[r for r in records if r['phase']=='close' and r['final_state']['gripper_motiondone']==2]
    if not closes or not (directory/'view-return/color.jpg').is_file():
        raise ValueError('Reference lacks stable contact or raised-cup image')
    pose=np.asarray(scope['contact_tcp'],float)
    normal=np.asarray(geometry['table_plane']['normal'],float)
    center=np.asarray(geometry['cup_model']['base_center_mm'],float)+87.*normal
    offset=transform(pose)[:3,:3].T@(center-pose[:3])
    return {'source_episode':str(directory),'source_summary_sha256':hashlib.sha256(raw).hexdigest(),
            'source_scope_sha256':hashlib.sha256(scope_raw).hexdigest(),
            'effective_grasp_offset_flange_mm':offset.tolist(),
            'basis':'observed cup-relative pose completed a50mm lift, replacement and release',
            'not_independent_tcp_calibration':True,'not_training_success_label':True}


def prepare(root: Path, profile: str = LEGACY_PROFILE, *, prealigned: bool = False,
            grasp_reference: Path | None = None) -> dict:
    validate_profile(profile)
    if root.exists():
        raise FileExistsError("A new varied-start root must not already exist")
    start_report = capture(root / "start")
    confirm_report = capture(root / "confirm")
    start_state = np.asarray(start_report["robot_after"]["tcp"], float)
    confirm_state = np.asarray(confirm_report["robot_after"]["tcp"], float)
    if np.max(np.abs(start_state - confirm_state)) > 0.1:
        raise RuntimeError("Robot moved during repeated scene capture")
    live = read_state()
    require_ready(live)
    if (live["robot_mode"] != 1 or live["gripper_active"] != 1
            or live["gripper_position"] != 100 or live["gripper_fault"] != 0):
        raise RuntimeError("Expected enabled J1 with a stationary fully open gripper")
    if np.max(np.abs(np.asarray(live["tcp"]) - confirm_state)) > 0.1:
        raise RuntimeError("Robot changed after scene capture")

    report_a, color_a, depth_a, calibration_a = load_scene(root / "start")
    report_b, color_b, depth_b, calibration_b = load_scene(root / "confirm")
    if not np.allclose(calibration_a, calibration_b, atol=1e-9):
        raise RuntimeError("Camera calibration changed between captures")
    rim_uv, radius_px, rim_detection = find_cup_rim(color_b)
    table_normal, table_offset, table = fit_table(color_b, depth_b, report_b, calibration_b, rim_uv)
    cup_base, rim_center, cup = estimate_cup(
        report_b, depth_b, calibration_b, rim_uv, table_normal, table_offset
    )

    rim_uv_a, _, _ = find_cup_rim(color_a)
    table_normal_a, table_offset_a, _ = fit_table(
        color_a, depth_a, report_a, calibration_a, rim_uv_a
    )
    cup_base_a, _, _ = estimate_cup(
        report_a, depth_a, calibration_a, rim_uv_a, table_normal_a, table_offset_a
    )
    cup_change = float(np.linalg.norm(cup_base - cup_base_a))
    if cup_change > 2.0 or np.linalg.norm(rim_uv - rim_uv_a) > 8.0:
        raise RuntimeError("Cup did not remain stationary across fresh observations")

    reference=load_grasp_reference(grasp_reference) if grasp_reference is not None else None
    routes, model = build_routes(confirm_state, cup_base, table_normal, profile=profile,
                                 prealigned=prealigned,grasp_reference=reference)
    path_rows, kinematics = sample_geometry(routes, live["joints"])
    action_plan = build_manual_action_plan(routes, path_rows, live["joints"], profile=profile)
    save(root / "action-plan.json", action_plan)
    clearances = table_clearances(routes, table_normal, table_offset)
    cup_checks = cup_tool_collision_checks(routes, cup_base, table_normal)
    overlay = route_overlay(
        color_b, report_b, calibration_b, cup_base, rim_center, rim_uv, routes
    )
    if not cv2.imwrite(str(root / "route-review.jpg"), overlay):
        raise RuntimeError("Could not save route review image")

    report = {
        "status": "prepared_for_visual_review",
        "profile": profile,
        "motion_commands_sent": 0,
        "gripper_commands_sent": 0,
        "robot_ip": HOST,
        "initial_tcp": confirm_state.tolist(),
        "initial_joints": live["joints"],
        "cup_change_between_observations_mm": cup_change,
        "rim_detection": rim_detection,
        "table_plane": table,
        "cup_model": cup,
        "route_model": model,
        "routes": routes,
        "action_plan": action_plan,
        "action_plan_sha256": hashlib.sha256((root / "action-plan.json").read_bytes()).hexdigest(),
        "kinematics": kinematics,
        "expanded_tool_min_table_clearances_mm": clearances,
        "cup_tool_collision_samples": cup_checks,
        "path": path_rows,
        "limitations": [
            "Known cup dimensions and circular image fit are not an independent metric accuracy bound.",
            "The 162mm measured flange-to-fingertip distance is only an unverified grasp-center assumption.",
            "Cup/tool overlap samples cover only the nominal contact pose, not the full swept path.",
            "The URDF and tool boxes are not a full arm and cable mesh collision certificate.",
            "Contact and held stages still require fresh image-based phase reviews.",
        ],
        "created_unix": time.time(),
        "route_review_image": str(root / "route-review.jpg"),
    }
    save(root / "geometry.json", report)
    return report


def finalize(root: Path, profile: str | None = None) -> dict:
    if profile is not None:
        validate_profile(profile)
    geometry_path = root / "geometry.json"
    raw = geometry_path.read_bytes()
    geometry = json.loads(raw)
    if geometry.get("status") != "prepared_for_visual_review":
        raise RuntimeError("Geometry is not ready for final review")
    age = time.time() - float(geometry["created_unix"])
    if not 0 <= age <= 300:
        raise RuntimeError("Route review expired before finalization")
    action_plan_path = root / "action-plan.json"
    action_plan_raw = action_plan_path.read_bytes()
    action_plan_sha = hashlib.sha256(action_plan_raw).hexdigest()
    if action_plan_sha != geometry.get("action_plan_sha256"):
        raise RuntimeError("Manual action plan changed after geometry preparation")
    action_plan = json.loads(action_plan_raw)
    selected_profile = matching_profiles(geometry, action_plan)
    if isinstance(geometry.get("action_plan"), dict):
        matching_profiles(geometry, action_plan, geometry["action_plan"])
    if profile is not None and profile != selected_profile:
        raise ValueError("Requested motion profile differs from the reviewed geometry and action plan")
    # A route overlay or a successful IK check cannot certify a hollow-cup
    # exterior grasp.  In particular, do not convert unresolved finger/cup
    # overlaps into a positive review merely by running --finalize.
    cup_check = geometry.get("cup_tool_collision_samples", {}).get("summary", {})
    if cup_check.get("finger_sample_overlap_count") != 0:
        raise RuntimeError("Finger/cup sweep has unresolved overlap; contact route is not validated")
    sweep = geometry.get("cup_tool_collision_samples", {}).get("approach_sweep_diagnostic", {})
    bodies = sweep.get("bodies", {})
    if (not sweep.get("pose_count") or not all(name in bodies for name in ("-1", "1", "housing"))
            or any(body.get("poses_with_overlap") != 0 for body in bodies.values())):
        raise RuntimeError("Approach sweep diagnostic is missing or has unresolved overlap")
    review_path = root / "scene-review.json"
    if not review_path.is_file():
        raise RuntimeError("Missing independent current scene and tool-geometry review")
    review = json.loads(review_path.read_text())
    supervised = review.get("verification_basis") == "operator_supervised_nominal"
    prealigned = geometry['route_model'].get('prealigned_start', False)
    if prealigned and (not supervised or review.get('prealigned_visual_entry_reviewed') is not True):
        raise RuntimeError('Current prealigned cup entry needs an explicit visual review')
    if supervised and (selected_profile != CONTINUOUS_PROFILE
            or review.get('tool_geometry_independently_verified') is not False
            or review.get('exterior_finger_sweep_verified') is not False
            or review.get('nominal_tool_dimensions_reviewed') is not True
            or review.get('visible_empty_approach_reviewed') is not True
            or not review.get('onsite_clearance_source')):
        raise RuntimeError('Supervised trial needs explicit assumptions and the continuous visual workflow')
    if (review.get("geometry_sha256") != hashlib.sha256(raw).hexdigest()
            or review.get("action_plan_sha256") != action_plan_sha
            or not 0 <= time.time() - float(review.get("reviewed_at_unix", 0)) <= 300
            or (not supervised and review.get("tool_geometry_independently_verified") is not True)
            or (not supervised and review.get("exterior_finger_sweep_verified") is not True)
            or review.get("visible_path_reviewed") is not True
            or not isinstance(review.get("evidence"), list) or not review["evidence"]
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"]):
        raise RuntimeError("Scene review lacks fresh, tied, independent geometry evidence")
    scope = {
        "robot_ip": HOST,
        "contract": "varied-cup-fixed-attitude-v1",
        "profile": selected_profile,
        "execution_basis": "operator_supervised_nominal" if supervised else "independent_geometry_review",
        "requires_runtime_visual_alignment": supervised,
        "phase_order": [
            "approach_clear",
            "approach_near",
            "approach_contact",
            "close",
            "lift_probe",
            "lift_full",
            "return",
            "release",
            "retreat_probe",
            "retreat_clear",
            "retreat_high",
        ],
        "max_lift_mm": 50,
        "closing_targets": [90, 80, 76],
        "closing_speed": 5,
        "force": 10,
        "move_speed": PROFILE_SPEEDS[selected_profile]["empty_free"],
        "motion_speed_profile": dict(PROFILE_SPEEDS[selected_profile]),
        "max_session_s": 1800,
        "only_prealigned_instance": prealigned,
        "prealigned_start": prealigned,
        "prealigned_visual_entry_reviewed": review.get('prealigned_visual_entry_reviewed') is True,
        "return_to_original_support": True,
        "split_role": "training_complete_trial",
        "reviewed_at_unix": time.time(),
        "reviewed_at_monotonic": time.monotonic(),
        "initial_tcp": geometry["initial_tcp"],
        "contact_tcp": geometry["route_model"]["contact_pose"],
        "terminal_tcp": geometry['routes']['retreat_high'][-1],
        "authorization_source": (
            "User explicitly requested autonomous large-start-variation cup demonstrations, "
            "complete pickup and replacement, with image-based safety judgment and no repeated permission."
        ),
        "evidence": [
            str(root / "start" / "color.jpg"),
            str(root / "confirm" / "color.jpg"),
            str(root / "route-review.jpg"),
            str(geometry_path),
        ] + review["evidence"],
        "cup_unchanged_visually_verified": True,
        "exterior_contact_route_reviewed": not supervised,
        "visible_local_sweep_clear": not supervised,
        "nominal_contact_route_reviewed": True,
        "unseen_space_user_default_normal": True,
        "no_visible_hands_anywhere": False,
        "geometry_path": str(geometry_path),
        "geometry_sha256": hashlib.sha256(raw).hexdigest(),
        "manual_action_plan_path": str(action_plan_path),
        "manual_action_plan_sha256": action_plan_sha,
        "manual_action_plan": action_plan,
        "current_scene_review": review,
        "near_path_tracking_tolerance_mm": 0.3,
        "storage_destination": (f"4090:{REMOTE_BASE / root.name}" if root.parent in (LOCAL_BASE, LOCAL_STREAM_BASE)
                                else "unverified local preparation; execution requires the 4090 spool"),
        "temporary_storage": ("Direct recording to the 4090 SFTP spool; no local raw-frame staging"
                              if root.parent == LOCAL_BASE else
                              "Bounded persistent local outbox; asynchronous streaming diagnostic only"
                              if root.parent == LOCAL_STREAM_BASE else str(root)),
    }
    save(root / "scope.json", scope)
    scope_sha = hashlib.sha256((root / "scope.json").read_bytes()).hexdigest()
    return {"scope": str(root / "scope.json"), "scope_sha256": scope_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--prealigned", action="store_true", help="Use the actual visually reviewed start for closure; no approach motion")
    parser.add_argument("--grasp-reference",type=Path,help="Observed completed pick/place used to propose a new cup-relative target")
    parser.add_argument("--profile", choices=tuple(PROFILE_SPEEDS), default=None,
                        help="Motion profile (default: legacy; finalization inherits the reviewed profile)")
    args = parser.parse_args()
    if args.finalize:
        result = finalize(args.root.resolve(), profile=args.profile)
    else:
        report = prepare(args.root.resolve(), profile=args.profile or LEGACY_PROFILE,
                         prealigned=args.prealigned,grasp_reference=args.grasp_reference)
        result = {
            "status": report["status"],
            "profile": record_profile(report),
            "root": str(args.root.resolve()),
            "cup_change_between_observations_mm": report["cup_change_between_observations_mm"],
            "geometry": str(args.root.resolve() / "geometry.json"),
            "action_plan": str(args.root.resolve() / "action-plan.json"),
            "route_review_image": report["route_review_image"],
            "actions": len(report["action_plan"]["actions"]),
            "motion_commands_sent": 0,
            "gripper_commands_sent": 0,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
