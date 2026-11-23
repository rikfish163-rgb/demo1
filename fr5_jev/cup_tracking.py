"""Track the upright cup in synchronized Gemini RGB-D and FR5 feedback.

The opening-time cup location remains the fixed task/return frame. This
tracker only supplies observations; it neither proposes targets nor estimates
grasp success. The jaw center is the nominal midpoint of the measured tool
model at the finger tips, not a calibrated tool frame.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .rgbd_geometry import models, sample_rgb_points
from .cup_rim_geometry import _ellipse


JAW_TIP_CENTER_FLANGE_MM = np.array([5.208, 8.072, 162.0])
RIM_RADIUS_MM = 37.5
_RIM_HEIGHT_MM = 110.
_PRECONTACT_LATERAL_TOLERANCE_MM = 10.
_RIM_HEIGHT_TOLERANCE_MM = 12.
_PHI = np.linspace(0, 2 * np.pi, 64, endpoint=False)
_UNIT_RING = np.column_stack((np.cos(_PHI), np.sin(_PHI)))


def _jaw_center(robot):
    flange = np.asarray(robot['flange'], dtype=float)
    if (robot['tool'] != 0 or robot['user'] != 0 or flange.shape != (6,)
            or not np.isfinite(flange).all()):
        raise ValueError('Cup tracking requires the measured J1 flange in tool0/user0')
    roll, pitch, yaw = np.deg2rad(flange[3:])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    r = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                  [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return flange[:3] + r @ JAW_TIP_CENTER_FLANGE_MM


def _contact(robot, valid):
    if not valid:
        return None
    return int(robot['gripper_motiondone'] == 2)


class CupTracker:
    """Search around the last seen rim, requiring fresh RGB and depth support.

    In contact, measured jaw motion may move the search window; it never
    supplies a cup location. The tracker returns a missing vector when the
    rim is hidden, too tilted, or otherwise ambiguous.
    """

    def __init__(self, initial_cup, base_from_camera):
        self.serial = initial_cup['camera_serial']
        self.base_from_camera = np.asarray(base_from_camera, dtype=float)
        self.normal = np.asarray(initial_cup['table_normal'], dtype=float)
        self.table_offset = float(initial_cup['table_offset_mm'])
        self.initial_rim = np.asarray(initial_cup['rim_xyz_mm'], dtype=float)
        if (self.base_from_camera.shape != (4, 4) or self.normal.shape != (3,)
                or self.initial_rim.shape != (3,) or not np.isfinite(self.base_from_camera).all()
                or not np.isfinite(self.normal).all() or abs(np.linalg.norm(self.normal) - 1) > .01
                or not np.isfinite(self.initial_rim).all() or not math.isfinite(self.table_offset)):
            raise ValueError('Invalid cup tracking calibration or initial geometry')
        self.last_time = float(initial_cup['observed_monotonic'])
        self.last_rim = self.initial_rim
        self.last_jaw = None
        self.last_contact = None
        self.grasp_contact_seen = False

    def _camera(self, point):
        return self.base_from_camera[:3, :3].T @ (point - self.base_from_camera[:3, 3])

    def _project(self, point, k, d):
        camera = self._camera(point)
        if camera[2] <= 0:
            return None
        uv = cv2.projectPoints(camera[None], np.zeros(3), np.zeros(3), k, d)[0].reshape(2)
        return uv, math.sqrt(k[0, 0] * k[1, 1]) * RIM_RADIUS_MM / camera[2]

    def _candidate(self, gray, edges, gx, gy, xy, radius, depth, metadata, k, d, *, offset):
        # Sample radial arcs, so straight black finger edges alone cannot
        # count as an observed circular cup opening.
        offsets = np.linspace(-2.5, 2.5, 9)
        uv = xy + (radius + offsets[:, None, None]) * _UNIT_RING
        u, v = np.rint(uv).astype(int).transpose(2, 0, 1)
        outward = gx[v, u] * _UNIT_RING[:, 0] + gy[v, u] * _UNIT_RING[:, 1]
        edge_hit = ((edges[v, u] != 0) & (outward > 14)).any(axis=0)
        sectors = edge_hit.reshape(8, 8).sum(axis=1)
        if edge_hit.sum() < 20 or np.count_nonzero(sectors >= 2) < 5:
            return None
        inner = np.rint(xy + radius * .63 * _UNIT_RING).astype(int)
        outer = np.rint(xy + radius * 1.21 * _UNIT_RING).astype(int)
        contrast = float(np.median(gray[outer[:, 1], outer[:, 0]]) - np.median(gray[inner[:, 1], inner[:, 0]]))
        if contrast < 13:
            return None

        # The transparent wall may return depth from the table. Use raised
        # samples near the RGB rim, with depth consistent with its apparent
        # size; do not use the opening's median depth or the gripper's Z.
        uv = xy + offset + radius * .91 * _UNIT_RING
        pts = sample_rgb_points(uv, depth, metadata, radius=0)['points_camera_mm']
        world = pts @ self.base_from_camera[:3, :3].T + self.base_from_camera[:3, 3]
        heights = world @ self.normal - self.table_offset
        apparent_z = math.sqrt(k[0, 0] * k[1, 1]) * RIM_RADIUS_MM / radius
        raised = (heights > 72.) & (heights < 240.) & (np.abs(pts[:, 2] - apparent_z) < 105.)
        if raised.sum() < 11 or np.count_nonzero(raised.reshape(8, 8).sum(axis=1) >= 2) < 4:
            return None
        # The upper part of the supported wall is nearer the rim. Do not
        # rely on a single closest pixel, which can belong to a finger.
        height = float(np.percentile(heights[raised], 80))
        if abs(height - _RIM_HEIGHT_MM) <= _RIM_HEIGHT_TOLERANCE_MM:
            height = _RIM_HEIGHT_MM
        ray_xy = cv2.undistortPoints((xy + offset).reshape(1, 1, 2), k, d).reshape(2)
        origin = self.base_from_camera[:3, 3]
        ray = self.base_from_camera[:3, :3] @ np.r_[ray_xy, 1.]
        denom = float(ray @ self.normal)
        if denom >= -.2:
            return None
        z = (self.table_offset + height - origin @ self.normal) / denom
        ratio = z / apparent_z
        if not .81 <= ratio <= 1.19:
            return None
        score = edge_hit.mean() + min(contrast, 85.) / 135. + .35 * raised.mean() - abs(math.log(ratio))
        return origin + z * ray, float(score), int(edge_hit.sum()), int(raised.sum())

    def update(self, color_bgr, depth_mm, camera, robot, *, gripper_valid):
        stamp = float(camera['source_timestamp_monotonic'])
        robot_time = float(robot['received_monotonic'])
        if (camera['source_camera_serial'] != self.serial or not math.isfinite(stamp)
                or not math.isfinite(robot_time) or stamp < self.last_time or abs(stamp - robot_time) > .1):
            raise ValueError('Cup tracking requires ordered Gemini frames with paired J1 feedback')
        self.last_time = stamp
        jaw = _jaw_center(robot)
        contact = _contact(robot, gripper_valid)
        if contact == 1:
            self.grasp_contact_seen = True
        result = {'rim_minus_jaw_mm': None, 'native_contact': contact, 'reason': 'rim_not_observed',
                  'rim_xyz_mm': None, 'rim_uv': None, 'nominal_jaw_xyz_mm': jaw.tolist(),
                  'rim_edge_samples': 0, 'raised_depth_samples': 0}
        k, d, _, _ = models(camera)
        w, h = camera['color_intrinsic']['width'], camera['color_intrinsic']['height']
        if color_bgr is None or color_bgr.shape != (h, w, 3) or depth_mm.shape != (h, w):
            raise ValueError('Cup tracking RGB-D shape differs from camera metadata')
        predicted = self.last_rim
        if (contact and self.last_jaw is not None and np.linalg.norm(self.last_rim - self.last_jaw) < 80):
            predicted = predicted + jaw - self.last_jaw
        seeds = [self._project(point, k, d) for point in (predicted, self.last_rim, self.initial_rim)]
        seeds = [seed for seed in seeds if seed is not None]
        if not seeds:
            return result
        centers, radii = zip(*seeds)
        centers = np.asarray(centers)
        padding = max(radii) * 2.7
        x0, y0 = np.maximum(0, np.floor(centers.min(axis=0) - padding).astype(int))
        x1, y1 = np.minimum([w, h], np.ceil(centers.max(axis=0) + padding).astype(int))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return result
        # All supported policy recordings have the 640x360 fast RGB-D view.
        gray = cv2.cvtColor(color_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.)
        r_min, r_max = max(6, int(min(radii)*.75)), int(max(radii)*1.36)+1
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, max(8., min(radii)*.65),
                                   param1=65, param2=13, minRadius=r_min, maxRadius=r_max)
        edges = cv2.Canny(gray, 30, 80)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        offset = np.array([x0, y0])
        candidates = [] if circles is None else [tuple(circle) for circle in circles[0]]
        original_candidate_count = len(candidates)
        fit_seeds = candidates + [(center[0]-x0, center[1]-y0, radius)
                                  for center, radius in zip(centers, radii)]
        for x, y, radius in fit_seeds:
            ellipse = _ellipse(gray, edges, gx, gy, float(x), float(y), float(radius))
            if ellipse is None:
                continue
            cx, cy = ellipse['cup_uv']
            a, b = ellipse['axes_px']
            fitted = (cx, cy, math.sqrt(a*b)/2)
            if not any(math.dist(fitted[:2], candidate[:2]) < .5
                       and abs(fitted[2]-candidate[2]) < .5 for candidate in candidates):
                candidates.append(fitted)
        original_found, fitted_found = [], []
        for candidate_index, (x, y, r) in enumerate(candidates):
            xy = np.array([x, y], dtype=float)
            margin = max(r * 1.23, r + 3.)
            gap = float(np.min(np.linalg.norm(centers - (xy + offset), axis=1)))
            if (gap > max(radii)*1.55 or np.any(xy < margin)
                    or np.any(xy + margin >= np.array(gray.shape[::-1]) - 1)):
                continue
            # Use global coordinates only for RGB-to-depth geometry.
            value = self._candidate(gray, edges, gx, gy, xy, float(r), depth_mm, camera, k, d,
                                    offset=offset)
            if value is not None:
                rim, score, edge_count, depth_count = value
                if not self.grasp_contact_seen:
                    from_initial = rim - self.initial_rim
                    normal_offset = float(from_initial @ self.normal)
                    initial_lateral = np.linalg.norm(from_initial - self.normal * normal_offset)
                    if (initial_lateral > _PRECONTACT_LATERAL_TOLERANCE_MM
                            or abs(normal_offset) > _RIM_HEIGHT_TOLERANCE_MM):
                        continue
                lateral = np.linalg.norm((rim-predicted) - self.normal * ((rim-predicted) @ self.normal))
                # In this fixed cup task motion within a rollout comes from
                # the jaw. A competing rim several centimetres away during
                # a close/hold is not evidence that the cup moved there.
                if lateral > 32. and (contact != 0 or self.last_contact != 0):
                    continue
                found = original_found if candidate_index < original_candidate_count else fitted_found
                found.append((score - .012 * gap - .005 * lateral, xy + offset, rim, edge_count, depth_count))
        found = original_found or fitted_found
        if not found:
            return result
        found.sort(key=lambda item: item[0], reverse=True)
        score, uv, rim, edge_count, depth_count = found[0]
        if any(score - other[0] < .12 and np.linalg.norm(uv - other[1]) > min(radii)*.62 for other in found[1:]):
            return dict(result, reason='rim_ambiguous')
        self.last_rim, self.last_jaw, self.last_contact = rim, jaw, contact
        return dict(result, reason='observed', rim_minus_jaw_mm=(rim - jaw).tolist(),
                    rim_xyz_mm=rim.tolist(), rim_uv=uv.tolist(), rim_edge_samples=edge_count,
                    raised_depth_samples=depth_count)
