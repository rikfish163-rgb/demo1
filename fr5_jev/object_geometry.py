"""Measure a selected RGB-D object and prepare a parallel-jaw grasp candidate."""
from __future__ import annotations

import math

import cv2
import numpy as np

from fr5_runtime.kinematics import transform
from .rgbd_geometry import sample_rgb_points


def selected_color_mask(color_bgr, pixel):
    """Select the connected, saturated color region containing an RGB pixel."""
    color = np.asarray(color_bgr)
    u, v = (int(round(float(x))) for x in pixel)
    if color.ndim != 3 or color.shape[2] != 3 or not (0 <= u < color.shape[1] and 0 <= v < color.shape[0]):
        raise ValueError('Target pixel must lie in the current RGB image')
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (int(x) for x in hsv[v, u])
    if saturation < 45 or value < 30:
        raise ValueError('Color selection needs a visible saturated object; use an explicit mask for this target')
    distance = np.abs(hsv[:, :, 0].astype(np.int16) - hue)
    distance = np.minimum(distance, 180 - distance)
    mask = ((distance <= 12) & (hsv[:, :, 1] >= max(35, saturation - 65))
            & (hsv[:, :, 2] >= max(25, value - 90))).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    label = int(labels[v, u])
    if label == 0:
        raise ValueError('Selected pixel has no object component')
    result = labels == label
    if int(result.sum()) < 100:
        raise ValueError('Selected color component is too small to measure')
    return result


def measure_mask(mask, depth_mm, metadata, base_from_camera_mm, table_normal, table_offset_mm):
    """Measure visible surface points using the actual RGB and depth ray models."""
    mask = np.asarray(mask, dtype=np.uint8)
    intrinsic = metadata['color_intrinsic']
    if mask.shape != (intrinsic['height'], intrinsic['width']):
        raise ValueError('Object mask must use the current raw RGB pixel grid')
    matrix = np.asarray(base_from_camera_mm, dtype=float)
    normal = np.asarray(table_normal, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-5)
            or normal[2] <= 0 or not math.isfinite(table_offset_mm)):
        raise ValueError('Expected a measured support plane and camera-to-base transform')
    inside = cv2.erode(mask, np.ones((5, 5), np.uint8))
    v, u = np.nonzero(inside)
    pixels = np.column_stack((u[::3], v[::3]))
    sampled = sample_rgb_points(pixels, depth_mm, metadata, radius=1, maximum_patch_spread_mm=8.)
    camera = sampled['points_camera_mm'][sampled['valid']]
    if len(camera) < 100:
        raise ValueError('Target has too few consistent RGB-D surface points')
    points = camera @ matrix[:3, :3].T + matrix[:3, 3]
    heights = points @ normal - table_offset_mm
    above = heights > 5.
    points, heights = points[above], heights[above]
    if len(points) < 100:
        raise ValueError('Target surface is not distinguishable above the measured support plane')
    center = np.median(points, axis=0)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    tangent_x = np.cross(normal, reference)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    centered = points - center
    planar = np.column_stack((centered @ tangent_x, centered @ tangent_y))
    _, _, angle_degrees = cv2.minAreaRect(planar.astype(np.float32))
    angle = math.radians(angle_degrees)
    axis = math.cos(angle) * tangent_x + math.sin(angle) * tangent_y
    other = np.cross(normal, axis)
    other /= np.linalg.norm(other)
    extent = [float(np.ptp(points @ direction)) for direction in (axis, other)]
    closing_axis = axis if extent[0] <= extent[1] else other
    top_height = float(np.quantile(heights, .9))
    top_center = center + normal * (top_height - (center @ normal - table_offset_mm))
    return {
        'visible_surface_points': len(points), 'selected_rgb_pixels': int(mask.sum()),
        'surface_center_base_mm': center.tolist(), 'top_center_base_mm': top_center.tolist(),
        'surface_height_p10_p90_mm': np.quantile(heights, [.1, .9]).tolist(),
        'visible_planar_extents_mm': extent, 'closing_axis_base': closing_axis.tolist(),
        'visible_closing_width_mm': min(extent), 'table_normal': normal.tolist(),
        'table_offset_mm': float(table_offset_mm),
        'measurement': 'visible selected surface; hidden object volume and support height remain unmeasured',
    }


def _rpy_degrees(rotation):
    return np.rad2deg([
        math.atan2(rotation[2, 1], rotation[2, 2]),
        math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])),
        math.atan2(rotation[1, 0], rotation[0, 0]),
    ])


def top_down_routes(start_tcp, measurement, grasp_offset_flange_mm, *, penetration_mm=10.,
                    open_gap_mm=80., lift_mm=50., transfer_height_mm=350.):
    """Prepare a route from visible geometry without authorizing execution."""
    start = np.asarray(start_tcp, dtype=float)
    offset = np.asarray(grasp_offset_flange_mm, dtype=float)
    normal = np.asarray(measurement['table_normal'], dtype=float)
    closing = np.asarray(measurement['closing_axis_base'], dtype=float)
    if (start.shape != (6,) or offset.shape != (3,) or not np.isfinite(start).all()
            or not np.isfinite(offset).all() or not 120. <= offset[2] <= 210.
            or abs(offset[0]) > 30. or abs(offset[1]) > 30.):
        raise ValueError('Expected a measured start and tool-sized grasp offset')
    if not (0 < penetration_mm <= 20. and 10. < lift_mm <= 80.
            and open_gap_mm > measurement['visible_closing_width_mm'] + 6.):
        raise ValueError('Visible target width or requested grasp motion exceeds the available opening')
    grip = np.asarray(measurement['top_center_base_mm'], dtype=float) - penetration_mm * normal
    if grip @ normal - measurement['table_offset_mm'] < 8.:
        raise ValueError('Candidate grasp lacks fingertip clearance above the support plane')
    closing = closing - normal * (closing @ normal)
    closing /= np.linalg.norm(closing)
    # Choose the equivalent opening direction nearest the measured attitude.
    current_x = transform(start)[:3, 0]
    if closing @ current_x < 0:
        closing = -closing
    tool_z = -normal
    rotation = np.column_stack((closing, np.cross(tool_z, closing), tool_z))
    attitude = _rpy_degrees(rotation)
    attitude = start[3:] + (attitude - start[3:] + 180.) % 360. - 180.
    contact = np.r_[grip - rotation @ offset, attitude]
    clear = contact.copy()
    clear[:3] += 80. * normal
    near = contact.copy()
    near[:3] += 20. * normal
    probe = contact.copy()
    probe[:3] += 10. * normal
    lift = contact.copy()
    lift[:3] += lift_mm * normal
    retreat_probe = contact.copy()
    retreat_probe[:3] += 10. * normal
    raised = start.copy()
    raised[2] = max(start[2], clear[2] + 20., transfer_height_mm)
    oriented = np.r_[raised[:3], attitude]
    transfer = np.r_[clear[:2], raised[2], attitude]
    routes = {
        'approach_clear': [start, raised, oriented, transfer, clear],
        'approach_near': [clear, near], 'approach_contact': [near, contact],
        'close': [contact, contact], 'lift_probe': [contact, probe],
        'lift_full': [probe, lift], 'return': [lift, contact],
        'release': [contact, contact], 'retreat_probe': [contact, retreat_probe],
        'retreat_clear': [retreat_probe, clear], 'retreat_high': [clear, transfer],
    }
    return {phase: [pose.tolist() for pose in poses] for phase, poses in routes.items()}, {
        'contact_tcp': contact.tolist(), 'grasp_center_base_mm': grip.tolist(),
        'grasp_offset_flange_mm': offset.tolist(), 'opening_axis_base': closing.tolist(),
        'tool_axis_base': tool_z.tolist(), 'penetration_mm': penetration_mm,
        'lift_mm': lift_mm, 'open_gap_mm': open_gap_mm,
        'geometry_or_collision_independently_validated': False,
        'requires': ['empty gripper', 'current target volume and clear approach',
                     'controller kinematics', 'tool, arm and cable path review'],
    }


def side_routes(start_tcp, measurement, grasp_offset_flange_mm, *, penetration_mm=10.,
                side_retraction_mm=0., open_gap_mm=80., lift_mm=50., transfer_height_mm=350.):
    """Approach along the visible short edge and close across the other edge."""
    start = np.asarray(start_tcp, dtype=float)
    offset = np.asarray(grasp_offset_flange_mm, dtype=float)
    normal = np.asarray(measurement['table_normal'], dtype=float)
    short_axis = np.asarray(measurement['closing_axis_base'], dtype=float)
    closing_width = max(measurement['visible_planar_extents_mm'])
    if (start.shape != (6,) or offset.shape != (3,) or not np.isfinite(start).all()
            or not np.isfinite(offset).all() or not 120. <= offset[2] <= 210.
            or abs(offset[0]) > 30. or abs(offset[1]) > 30.):
        raise ValueError('Expected a measured start and tool-sized grasp offset')
    if not (0 < penetration_mm <= 20. and 10. < lift_mm <= 80.
            and open_gap_mm > closing_width + 6.):
        raise ValueError('Visible side-grasp width or requested grasp motion exceeds the available opening')
    if not (math.isfinite(side_retraction_mm)
            and 0. <= side_retraction_mm < min(measurement['visible_planar_extents_mm']) / 2.):
        raise ValueError('Side retraction must be nonnegative and less than half the visible short edge')
    grip = np.asarray(measurement['top_center_base_mm'], dtype=float) - penetration_mm * normal
    tool_z = short_axis - normal * (short_axis @ normal)
    tool_z /= np.linalg.norm(tool_z)
    if tool_z @ (grip - start[:3]) < 0:
        tool_z = -tool_z
    grip -= side_retraction_mm * tool_z
    if grip @ normal - measurement['table_offset_mm'] < 8.:
        raise ValueError('Candidate grasp lacks fingertip clearance above the support plane')
    closing = np.cross(normal, tool_z)
    if closing @ transform(start)[:3, 0] < 0:
        closing = -closing
    rotation = np.column_stack((closing, np.cross(tool_z, closing), tool_z))
    attitude = _rpy_degrees(rotation)
    attitude = start[3:] + (attitude - start[3:] + 180.) % 360. - 180.
    contact = np.r_[grip - rotation @ offset, attitude]
    clear = contact.copy()
    clear[:3] -= 80. * tool_z
    near = contact.copy()
    near[:3] -= 20. * tool_z
    probe = contact.copy()
    probe[:3] += 10. * normal
    lift = contact.copy()
    lift[:3] += lift_mm * normal
    retreat_probe = contact.copy()
    retreat_probe[:3] -= 10. * tool_z
    raised = start.copy()
    raised[2] = max(start[2], clear[2] + 20., transfer_height_mm)
    oriented = np.r_[raised[:3], attitude]
    transfer = np.r_[clear[:2], raised[2], attitude]
    routes = {
        'approach_clear': [start, raised, oriented, transfer, clear],
        'approach_near': [clear, near], 'approach_contact': [near, contact],
        'close': [contact, contact], 'lift_probe': [contact, probe],
        'lift_full': [probe, lift], 'return': [lift, contact],
        'release': [contact, contact], 'retreat_probe': [contact, retreat_probe],
        'retreat_clear': [retreat_probe, clear], 'retreat_high': [clear, transfer],
    }
    return {phase: [pose.tolist() for pose in poses] for phase, poses in routes.items()}, {
        'contact_tcp': contact.tolist(), 'grasp_center_base_mm': grip.tolist(),
        'grasp_offset_flange_mm': offset.tolist(), 'opening_axis_base': closing.tolist(),
        'tool_axis_base': tool_z.tolist(), 'visible_closing_width_mm': closing_width,
        'penetration_mm': penetration_mm, 'side_retraction_mm': side_retraction_mm,
        'lift_mm': lift_mm, 'open_gap_mm': open_gap_mm,
        'geometry_or_collision_independently_validated': False,
        'requires': ['empty gripper', 'current target volume and clear side approach',
                     'controller kinematics', 'tool, arm and cable path review'],
    }


def tilted_routes(start_tcp, measurement, grasp_offset_flange_mm, *, tilt_from_down_deg=55.,
                  penetration_mm=10., open_gap_mm=80., lift_mm=50., transfer_height_mm=350.,
                  closing_edge='long'):
    """Tilt toward one measured edge and close across the perpendicular edge."""
    start = np.asarray(start_tcp, dtype=float)
    offset = np.asarray(grasp_offset_flange_mm, dtype=float)
    normal = np.asarray(measurement['table_normal'], dtype=float)
    horizontal = np.asarray(measurement['closing_axis_base'], dtype=float).copy()
    if closing_edge not in ('long', 'short'):
        raise ValueError('Tilted closing edge must be long or short')
    extents = measurement['visible_planar_extents_mm']
    closing_width = max(extents) if closing_edge == 'long' else min(extents)
    if (start.shape != (6,) or offset.shape != (3,) or not np.isfinite(start).all()
            or not np.isfinite(offset).all() or not 120. <= offset[2] <= 210.
            or abs(offset[0]) > 30. or abs(offset[1]) > 30.):
        raise ValueError('Expected a measured start and tool-sized grasp offset')
    if not math.isfinite(tilt_from_down_deg) or not 0. < tilt_from_down_deg < 90.:
        raise ValueError('Tilt from downward must lie strictly between 0 and 90 degrees')
    if not (0 < penetration_mm <= 20. and 10. < lift_mm <= 80.
            and open_gap_mm > closing_width + 6.):
        raise ValueError('Visible tilted-grasp width or requested grasp motion exceeds the available opening')
    grip = np.asarray(measurement['top_center_base_mm'], dtype=float) - penetration_mm * normal
    if grip @ normal - measurement['table_offset_mm'] < 8.:
        raise ValueError('Candidate grasp lacks fingertip clearance above the support plane')
    horizontal -= normal * (horizontal @ normal)
    horizontal /= np.linalg.norm(horizontal)
    if closing_edge == 'short':
        horizontal = np.cross(normal, horizontal)
    if horizontal @ (grip - start[:3]) < 0:
        horizontal = -horizontal
    closing = np.cross(normal, horizontal)
    closing /= np.linalg.norm(closing)
    if closing @ transform(start)[:3, 0] < 0:
        closing = -closing
    theta = math.radians(tilt_from_down_deg)
    tool_z = math.sin(theta) * horizontal - math.cos(theta) * normal
    rotation = np.column_stack((closing, np.cross(tool_z, closing), tool_z))
    attitude = _rpy_degrees(rotation)
    attitude = start[3:] + (attitude - start[3:] + 180.) % 360. - 180.
    contact = np.r_[grip - rotation @ offset, attitude]
    clear, near, probe, lift, retreat_probe = (contact.copy() for _ in range(5))
    clear[:3] -= 80. * tool_z
    near[:3] -= 20. * tool_z
    probe[:3] += 10. * normal
    lift[:3] += lift_mm * normal
    retreat_probe[:3] -= 10. * tool_z
    raised = start.copy()
    raised[2] = max(start[2], clear[2] + 20., transfer_height_mm)
    oriented = np.r_[raised[:3], attitude]
    transfer = np.r_[clear[:2], raised[2], attitude]
    routes = {
        'approach_clear': [start, raised, oriented, transfer, clear],
        'approach_near': [clear, near], 'approach_contact': [near, contact],
        'close': [contact, contact], 'lift_probe': [contact, probe],
        'lift_full': [probe, lift], 'return': [lift, contact],
        'release': [contact, contact], 'retreat_probe': [contact, retreat_probe],
        'retreat_clear': [retreat_probe, clear], 'retreat_high': [clear, transfer],
    }
    return {phase: [pose.tolist() for pose in poses] for phase, poses in routes.items()}, {
        'contact_tcp': contact.tolist(), 'grasp_center_base_mm': grip.tolist(),
        'grasp_offset_flange_mm': offset.tolist(), 'opening_axis_base': closing.tolist(),
        'tool_axis_base': tool_z.tolist(), 'visible_closing_width_mm': closing_width,
        'penetration_mm': penetration_mm, 'tilt_from_down_deg': float(tilt_from_down_deg),
        'closing_edge': closing_edge,
        'lift_mm': lift_mm, 'open_gap_mm': open_gap_mm,
        'geometry_or_collision_independently_validated': False,
        'requires': ['empty gripper', 'current target volume and clear tilted approach',
                     'controller kinematics', 'tool, arm and cable path review'],
    }
