"""FR5 pose transforms and the controller-verified six-joint chain."""
import math
import cv2
import numpy as np


def transform(pose):
    x, y, z = np.deg2rad(pose[3:])
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rz@Ry@Rx; T[:3, 3] = pose[:3]
    return T


def rotation_error(A, B):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(A@B.T)[0])))


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def model_chain(joints, descriptions, *, tool_offset_mm=(0., 0., 150.)):
    transform = np.eye(4)
    points = [transform[:3, 3].tolist()]
    for i, (angle, description) in enumerate(zip(joints, descriptions)):
        origin = description.find('origin')
        xyz = np.array([float(v) for v in origin.get('xyz').split()]) * 1000
        rpy = [float(v) for v in origin.get('rpy').split()]
        if i == 1:
            xyz[2] = 152.0
        fixed = np.eye(4)
        fixed[:3, :3] = rotation(rpy)
        fixed[:3, 3] = xyz
        revolute = np.eye(4)
        revolute[:3, :3] = rotation([0, 0, math.radians(angle)])
        transform = transform @ fixed @ revolute
        if i > 0:
            points.append(transform[:3, 3].tolist())
    flange = np.eye(4)
    flange[:3, :3] = rotation([0, 0, math.radians(-130)])
    flange[2, 3] = 100
    transform = transform @ flange
    points.append(transform[:3, 3].tolist())
    points.append((transform[:3, 3] + transform[:3, :3] @ tool_offset_mm).tolist())
    return points
