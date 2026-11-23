"""Local cup-frame observations and read-only IK for continuous pi0.5 actions.

The cup frame has the base frame's axes and an origin at the detected cup base.
The model uses metres for relative XYZ, base-frame radians for RPY, and a
gripper fraction. This module never sends motion or gripper commands.
"""
from __future__ import annotations

import io
import json
import math
import time
import urllib.request

import numpy as np

from fr5_jev.calibration import load_readonly_calibration
from fr5_jev.cup_localization import detect
from fr5_jev.cup_tracking import CupTracker
from fr5_jev.prepare_varied_session import TOOL_REFERENCE, estimate_cup, table_clearances
from fr5_jev.rgbd_geometry import optics
from lerobot_camera_gemini.timing import MONOTONIC, sample_clock

from .action_control import angle_error, finite_vector, rpc_vector
from .control_transport import FR5RPC
from .feedback import gripper_feedback_valid


CONTRACT = 'cup-relative-tcp-v1'
CHUNK_SIZE = 50
MAX_JOINT_STEP_DEG = 1.5
FK_POSITION_TOLERANCE_MM = 0.5
FK_ORIENTATION_TOLERANCE_DEG = 0.1
CAMERA_MAX_AGE_S = 2.0
MAX_CAMERA_ROBOT_SKEW_S = 0.1
TABLE_FK_SAMPLE_STEP_DEG = 0.25


def _bridge_frame():
    with urllib.request.urlopen('http://127.0.0.1:8765/snapshot.npz', timeout=5) as response:
        raw = response.read()
    with np.load(io.BytesIO(raw), allow_pickle=False) as packet:
        return {'color_bgr': packet['color_bgr'].copy(), 'depth_mm': packet['depth_mm'].copy(),
                'metadata': json.loads(str(packet['metadata'].item()))}


def detect_initial_cup(frame=None, *, now=None):
    """Detect exactly one cup in a full-resolution, same-frame RGB-D bundle.

    Without ``frame`` the existing local camera bridge is read once. A supplied
    frame contains ``color_bgr``, ``depth_mm`` and the bridge ``metadata``.
    ``now`` is a host-monotonic reference for replaying a recorded frame.
    The returned table plane is ``normal @ point_base_mm == offset_mm``.
    """
    frame = _bridge_frame() if frame is None else frame
    color, depth, metadata = frame['color_bgr'], frame['depth_mm'], frame['metadata']
    serial = metadata['camera']['serial']
    observed = metadata['capture'].get('received_monotonic')
    if (isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not math.isfinite(observed) or observed <= 0):
        raise ValueError('Cup detection requires a camera host-monotonic timestamp')
    current = time.monotonic() if now is None else float(now)
    if not math.isfinite(current) or not 0 <= current - observed < CAMERA_MAX_AGE_S:
        raise ValueError('Cup detection frame is stale or has a future timestamp')

    calibration = load_readonly_calibration(camera_serial=serial)
    intrinsic, distortion = optics(metadata['color_intrinsic'], metadata['color_distortion'])
    if (not np.allclose(intrinsic, calibration['K'], atol=1e-6, rtol=0)
            or not np.allclose(distortion, calibration['D'], atol=1e-9, rtol=0)):
        raise ValueError('Camera optics differ from the calibrated optics')
    base_from_camera = np.asarray(calibration['base_from_color_camera_mm'], dtype=float)
    result = detect(color, depth, metadata, calibration=base_from_camera)
    if result['status'] != 'found' or len(result['candidates']) != 1:
        raise ValueError(f"Expected one detectable cup rim; detector reported {result['status']}")
    uv = finite_vector(result['cup_uv'], 2, 'cup rim pixels')
    table = result['table']
    normal = finite_vector(table['normal'], 3, 'table normal')
    offset = float(table['offset_mm'])
    if not math.isfinite(offset) or abs(np.linalg.norm(normal) - 1) > .01:
        raise ValueError('Invalid detected table plane')
    base, rim, geometry = estimate_cup({'camera': metadata}, depth, base_from_camera,
                                       uv, normal, offset)
    base = finite_vector(base, 3, 'cup base')
    rim = finite_vector(rim, 3, 'cup rim center')
    rim_camera = base_from_camera[:3, :3].T @ (rim - base_from_camera[:3, 3])
    if rim_camera[2] <= 0:
        raise ValueError('Cup rim ray intersects behind the camera')
    return {'uv': uv.tolist(), 'camera_serial': serial, 'observed_monotonic': float(observed),
            'source': result['source'], 'base_xyz_mm': base.tolist(),
            'table_normal': normal.tolist(), 'table_offset_mm': offset,
            'rim_xyz_mm': rim.tolist(), 'cup_height_mm': geometry['height_mm'],
            'calibration_source': calibration['source'], 'table_fit': table}


def cup_relative_state(sample, initial_cup):
    """Convert a paired sample's measured TCP and gripper to the eight-value state."""
    camera_time, robot_time, basis = sample_clock(sample)
    if basis != MONOTONIC or abs(camera_time - robot_time) > MAX_CAMERA_ROBOT_SKEW_S:
        raise ValueError('Cup-relative state requires a matched monotonic camera/robot sample')
    metadata = sample['camera']['metadata']
    serial = metadata.get('source_camera_serial') or metadata.get('camera', {}).get('serial')
    if serial != initial_cup['camera_serial'] or camera_time < initial_cup['observed_monotonic']:
        raise ValueError('Observation camera or time differs from the initial cup frame')
    state = sample['robot_state']
    if state['tool'] != 0 or state['user'] != 0:
        raise ValueError('Cup-relative IK requires the recorded tool0/user0 coordinates')
    tcp = finite_vector(state['tcp'], 6, 'measured TCP')
    flange = finite_vector(state['flange'], 6, 'measured flange')
    if (np.linalg.norm(tcp[:3] - flange[:3]) > 1.0
            or np.max(np.abs(angle_error(tcp[3:], flange[3:]))) > .1):
        raise ValueError('Cup-relative IK requires tool0 to match the flange')
    cup_base = finite_vector(initial_cup['base_xyz_mm'], 3, 'cup base')
    valid = gripper_feedback_valid(state)
    return [*((tcp[:3] - cup_base) / 1000.).tolist(), *np.deg2rad(tcp[3:]).tolist(),
            float(state['gripper_position']) / 100. if valid else 0., float(valid)]


def new_cup_tracker(initial_cup):
    calibration = load_readonly_calibration(camera_serial=initial_cup['camera_serial'])
    return CupTracker(initial_cup, calibration['base_from_color_camera_mm'])


def track_cup_sample(tracker, sample):
    # Use the same JPEG90 pixels as the original demonstrations.
    import cv2

    camera = sample['camera']
    ok, encoded = cv2.imencode('.jpg', camera['color_rgb'][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError('Cannot encode Gemini RGB for cup tracking')
    color_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    state = sample['robot_state']
    return tracker.update(color_bgr, camera['depth_mm'], camera['metadata'], state,
                          gripper_valid=gripper_feedback_valid(state))


def decode_prefix(actions, initial_cup, anchor_joint_rad, n_steps, *, rpc=None):
    """Decode the first N model actions with read-only soft-limit, IK and FK RPCs.

    Each result has ``target`` (six absolute joint radians and a projected
    gripper fraction) and ``detail`` (model, cup, base-pose and IK/FK evidence).
    The previous IK solution seeds the next one; no robot action is dispatched.
    """
    values = np.asarray(actions, dtype=float)
    if values.shape != (CHUNK_SIZE, 7) or not np.isfinite(values).all():
        raise ValueError('Expected a finite 50 by 7 cup-relative action chunk')
    if type(n_steps) is not int or not 1 <= n_steps <= CHUNK_SIZE:
        raise ValueError('n_steps must be an integer from 1 to 50')
    cup_base = finite_vector(initial_cup['base_xyz_mm'], 3, 'cup base')
    previous_rad = finite_vector(anchor_joint_rad, 6, 'anchor joints')
    seed = np.rad2deg(previous_rad)
    rpc = FR5RPC() if rpc is None else rpc
    limits = rpc_vector(rpc.call('GetJointSoftLimitDeg', 1), 12, 'joint soft limits').reshape(6, 2)
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Invalid controller joint soft limits')
    if np.any(seed < limits[:, 0]) or np.any(seed > limits[:, 1]):
        raise ValueError('Anchor joints are outside controller soft limits')

    decoded = []
    for index, action in enumerate(values[:n_steps]):
        relative_pose = action[:6]
        absolute_pose = np.r_[cup_base + relative_pose[:3] * 1000.,
                              np.rad2deg(relative_pose[3:])]
        if not np.isfinite(absolute_pose).all():
            raise ValueError(f'Action {index} has an invalid absolute TCP target')
        q = rpc_vector(rpc.call('GetInverseKinRef', 0, absolute_pose.tolist(), seed.tolist()), 6, 'IK')
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError(f'Action {index} IK target is outside controller soft limits')
        q_rad = np.deg2rad(q)
        jump = max(abs(math.degrees(float(current - previous)))
                   for current, previous in zip(q_rad, previous_rad))
        if jump > MAX_JOINT_STEP_DEG:
            raise ValueError(f'Action {index} IK joint jump {jump:.3f} deg exceeds {MAX_JOINT_STEP_DEG:.1f} deg')
        fk = rpc_vector(rpc.call('GetForwardKin', q.tolist()), 6, 'FK')
        position_error = float(np.linalg.norm(fk[:3] - absolute_pose[:3]))
        orientation_error = float(np.max(np.abs(angle_error(fk[3:], absolute_pose[3:]))))
        if position_error > FK_POSITION_TOLERANCE_MM or orientation_error > FK_ORIENTATION_TOLERANCE_DEG:
            raise ValueError(f'Action {index} IK/FK mismatch: {position_error:.3f} mm, {orientation_error:.3f} deg')
        gripper = float(np.clip(action[6], 0., 1.))
        decoded.append({'target': [*q_rad.tolist(), gripper], 'detail': {
            'action_index': index, 'policy_contract': CONTRACT,
            'raw_model_action': action.tolist(), 'cup_relative_tcp_m_rad': relative_pose.tolist(),
            'cup_base_xyz_mm': cup_base.tolist(), 'target_tcp_base_mm_deg': absolute_pose.tolist(),
            'ik_seed_deg': seed.tolist(), 'ik_joint_deg': q.tolist(),
            'fk_tcp_base_mm_deg': fk.tolist(), 'ik_joint_step_max_deg': jump,
            'fk_position_error_mm': position_error, 'fk_orientation_error_deg': orientation_error,
            'gripper_projected_fraction': gripper,
        }})
        seed = q
        previous_rad = q_rad
    return decoded


def check_table_clearance(anchor_joint_rad, decoded, initial_cup, *, rpc):
    """Check existing tool boxes against the table along joint-interpolated FK.

    ``decoded`` is the result of ``decode_prefix``; ``rpc`` must provide the
    read-only ``GetForwardKin`` call. FK is sampled at the anchor, endpoints,
    and intermediate joint configurations. The returned poses and box result
    are evidence for review, not a command or a whole-robot collision check.
    """
    normal = finite_vector(initial_cup['table_normal'], 3, 'table normal')
    offset = float(initial_cup['table_offset_mm'])
    if not math.isfinite(offset) or abs(np.linalg.norm(normal) - 1) > .01 or normal[2] <= 0:
        raise ValueError('Invalid table plane')
    anchor = finite_vector(anchor_joint_rad, 6, 'anchor joints')
    if not isinstance(decoded, list) or not decoded:
        raise ValueError('A nonempty decoded action prefix is required')
    joint_radians = [anchor]
    for index, row in enumerate(decoded):
        target = finite_vector(row['target'], 7, f'action {index} target')
        goal = target[:6]
        step = max(abs(math.degrees(float(current - previous)))
                   for current, previous in zip(goal, joint_radians[-1]))
        if step > MAX_JOINT_STEP_DEG:
            raise ValueError(f'Action {index} joint step exceeds {MAX_JOINT_STEP_DEG:.1f} deg')
        joint_radians.append(goal)

    joints = [np.rad2deg(q) for q in joint_radians]

    samples, poses = [], []

    def record_fk(q, segment, fraction):
        fk = rpc_vector(rpc.call('GetForwardKin', q.tolist()), 6, 'FK')
        pose = fk.copy()
        if poses:
            pose[3:] = poses[-1][3:] + angle_error(fk[3:], poses[-1][3:])
        poses.append(pose)
        samples.append({'segment_index': segment, 'fraction': fraction,
                        'joint_deg': q.tolist(), 'fk_tcp_base_mm_deg': fk.tolist(),
                        'table_check_pose_mm_deg': pose.tolist()})

    record_fk(joints[0], None, 0.)
    for index, (start, end) in enumerate(zip(joints, joints[1:])):
        count = max(2, math.ceil(float(np.max(np.abs(end - start))) / TABLE_FK_SAMPLE_STEP_DEG))
        for step in range(1, count + 1):
            fraction = step / count
            record_fk(start + fraction * (end - start), index, fraction)

    error = None
    try:
        minima = table_clearances({'joint_interpolated_fk': poses}, normal, offset)
    except RuntimeError as exc:
        minima, error = None, str(exc)
    return {'passed': error is None, 'clearance_mm_by_box': minima, 'error': error,
            'required_clearance_mm': 5.0, 'joint_sample_max_step_deg': TABLE_FK_SAMPLE_STEP_DEG,
            'table_normal': normal.tolist(), 'table_offset_mm': offset,
            'tool_geometry_source': str(TOOL_REFERENCE), 'fk_samples': samples}


def check_cup_gripper_change(relative_pose, previous_percent, target_percent, *, support_reference=None):
    """Screen actual jaw changes beyond the controller's one-point deadband."""
    if abs(target_percent - previous_percent) <= 1:
        return
    if target_percent < previous_percent and previous_percent >= 88:
        phase, neighborhood, vertical_limit = 'close', 'contact', 12.
    elif target_percent > previous_percent and previous_percent < 98:
        phase, neighborhood, vertical_limit = 'release', 'support', 8.
    else:
        return
    # Demonstrations approached with the jaws open and reached nearly the same
    # cup-relative TCP contact: X≈159.7 mm, Y≈71.3 mm, Z≈95.8 mm.
    error = np.array(relative_pose[:3]) * 1000 - [159.7, 71.3, 95.8]
    horizontal = np.linalg.norm(error[:2])
    if horizontal > 15.0 or abs(error[2]) > vertical_limit:
        raise RuntimeError(f'Cup-relative {phase} is outside the demonstrated {neighborhood} neighborhood: '
                           f'horizontal {horizontal:.1f} mm, vertical {error[2]:.1f} mm')
    if phase == 'release' and support_reference is not None:
        anchor, normal, anchor_gap_mm = support_reference
        gap_mm = anchor_gap_mm + sum((a - b) * n for a, b, n in zip(relative_pose[:3], anchor, normal)) * 1000.
        if gap_mm > 1.5:
            raise RuntimeError(f'Cup-relative release is {gap_mm:.2f} mm above the measured grasp height')


def check_measured_cup_setdown(state, initial_cup, support_reference, *, require_contact=True):
    """Verify fresh FR5 feedback at the release endpoint before writing MoveGripper."""
    anchor, normal, anchor_gap_mm = support_reference
    if not gripper_feedback_valid(state):
        raise RuntimeError('PGI-140 feedback is invalid before release')
    if require_contact and state['gripper_motiondone'] != 2:
        raise RuntimeError('PGI-140 contact was lost before the first release')
    measured = [(x - b) / 1000. for x, b in zip(state['flange'][:3], initial_cup['base_xyz_mm'])]
    gap_mm = anchor_gap_mm + sum((a - b) * n for a, b, n in zip(measured, anchor, normal)) * 1000.
    if abs(gap_mm) > 1.5:
        raise RuntimeError(f'Measured release endpoint is {gap_mm:.2f} mm from the grasp height')
    if abs(state['tcp_speed'][0]) > 4.:
        raise RuntimeError('TCP is still moving at the release endpoint')


def check_cup_close(decoded, *, gripper_fraction, support_reference=None):
    """Check the gripper changes that this prefix would send from its observed state."""
    previous = int(round(100 * gripper_fraction))
    for row in decoded:
        target = int(round(100 * row['target'][6]))
        check_cup_gripper_change(row['detail']['cup_relative_tcp_m_rad'], previous, target,
                                 support_reference=support_reference)
        if abs(target - previous) > 1:
            previous = target
