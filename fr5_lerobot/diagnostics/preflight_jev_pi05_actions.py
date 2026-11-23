#!/usr/bin/env python3
"""Read-only joint-prefix and tool/table preflight for an absolute-joint pi0.5 reply.

The controller is queried only for soft limits and forward kinematics. The
report is evidence about the supplied request and scene, not permission to
execute the model output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xmlrpc.client

import numpy as np

if not __package__:
    project_root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(project_root), str(project_root / 'fr5_lerobot' / 'src')]

from fr5_runtime.kinematics import transform
from lerobot_robot_fairino.action_control import finite_vector, rpc_vector
from lerobot_robot_fairino.control_transport import FR5RPC
from lerobot_robot_fairino.cup_relative_runtime import (
    MAX_JOINT_STEP_DEG, TOOL_REFERENCE, check_table_clearance,
)
from lerobot_robot_fairino.stream_contract import ABSOLUTE_JOINT_WIRE, CHUNK


SCHEMA = 'fr5-jev-pi05-joint-preflight-v1'
RPC_ERRORS = (RuntimeError, ValueError, TypeError, OSError, xmlrpc.client.Error)


def table_plane(scene):
    """Read the explicit table plane from an observation or side-candidate."""
    if not isinstance(scene, dict):
        raise ValueError('Scene must contain an explicit table_plane or table')
    if 'table_plane' in scene:
        plane = scene['table_plane']
    elif 'table' in scene:
        plane = scene['table']
    else:
        raise ValueError('Scene must contain an explicit table_plane or table')
    if not isinstance(plane, dict) or 'normal' not in plane or 'offset_mm' not in plane:
        raise ValueError('Table plane requires normal and offset_mm')
    normal = finite_vector(plane['normal'], 3, 'table normal')
    offset = float(plane['offset_mm'])
    if not math.isfinite(offset) or abs(np.linalg.norm(normal) - 1) > .01 or normal[2] <= 0:
        raise ValueError('Invalid table plane')
    return {'table_normal': normal.tolist(), 'table_offset_mm': offset}


def _record_sample_clearances(samples, normal, offset):
    """Add the existing padded boxes' clearance at each controller FK sample."""
    boxes = json.loads(TOOL_REFERENCE.read_text(encoding='utf-8'))['tool_boxes_for_this_exterior_move']
    normal = np.asarray(normal, dtype=float)
    minima = {name: math.inf for name in boxes}
    for sample in samples:
        pose = np.asarray(sample['table_check_pose_mm_deg'], dtype=float)
        rotation = transform(pose)[:3, :3]
        local_normal = rotation.T @ normal
        values = {}
        for name, box in boxes.items():
            low, high = np.asarray(box['local_bounds_mm'], dtype=float)
            center = (low + high) / 2.
            half_size = (high - low) / 2. + float(box['padding_mm'])
            clearance = float(normal @ (pose[:3] + rotation @ center) - offset
                              - np.abs(local_normal) @ half_size)
            values[name] = clearance
            minima[name] = min(minima[name], clearance)
        sample['tool_clearance_mm_by_box'] = values
    return minima


def preflight(request, reply, scene, prefix_steps, *, rpc):
    """Check the untouched absolute-joint action prefix using read-only RPC."""
    if not isinstance(request, dict) or not isinstance(reply, dict):
        raise ValueError('Request and reply must be JSON objects')
    if 'id' in request and 'id' in reply and request['id'] != reply['id']:
        raise ValueError('Request and reply identifiers do not match')
    if reply.get('wire_action_representation', ABSOLUTE_JOINT_WIRE) != ABSOLUTE_JOINT_WIRE:
        raise ValueError('Expected absolute joint position actions')
    state = finite_vector(request.get('state'), 8, 'request state')
    values = np.asarray(reply.get('actions'), dtype=float)
    if values.shape != (CHUNK, 7) or not np.isfinite(values).all():
        raise ValueError('Expected a finite 50 by 7 action chunk')
    if type(prefix_steps) is not int or not 1 <= prefix_steps <= CHUNK:
        raise ValueError('prefix_steps must be an integer from 1 to 50')
    plane = table_plane(scene)
    report = {
        'schema': SCHEMA, 'passed': False, 'execution_authorized': False,
        'request_id': request.get('id'), 'action_shape': [CHUNK, 7],
        'prefix_steps': prefix_steps, 'table_plane': plane,
        'anchor_joint_rad': state[:6].tolist(), 'joint_soft_limits_deg': None,
        'minimum_joint_soft_limit_margin_deg': None,
        'maximum_joint_step_deg': MAX_JOINT_STEP_DEG,
        'segments': [], 'anchor_fk_sample': None,
        'tool_table_clearance': {'status': 'not_checked'}, 'failures': [],
        'limitations': [
            'Checks only the supplied joint prefix and the existing padded tool boxes against the supplied table plane.',
            'Does not check the full robot arm, object or held volume, environment, or cables.',
            'Gripper targets are recorded unchanged and are not checked for actuation or contact.',
            'Assumes controller FK poses correspond to the flange/tool0 model; current coordinates and state are not read.',
            'Controller FK is not independently verified.',
            'The supplied request and scene may not describe the current physical state; this report authorizes no execution.',
        ],
    }
    try:
        limits = rpc_vector(rpc.call('GetJointSoftLimitDeg', 1), 12, 'joint soft limits').reshape(6, 2)
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('Invalid controller joint soft limits')
    except RPC_ERRORS as exc:
        report['failures'].append({'check': 'joint_soft_limits', 'error': f'{type(exc).__name__}: {exc}'})
        return report

    report['joint_soft_limits_deg'] = limits.tolist()
    radians = np.vstack((state[:6], values[:prefix_steps, :6]))
    joints = np.rad2deg(radians)
    margins = np.minimum(joints - limits[:, 0], limits[:, 1] - joints)
    report['minimum_joint_soft_limit_margin_deg'] = float(margins.min())
    if np.any(margins[0] < 0):
        report['failures'].append({'check': 'joint_soft_limits', 'location': 'anchor',
                                   'joint_numbers': (np.flatnonzero(margins[0] < 0) + 1).tolist()})
    for index, action in enumerate(values[:prefix_steps]):
        step = max(abs(math.degrees(float(current - previous)))
                   for current, previous in zip(radians[index + 1], radians[index]))
        segment = {
            'action_index': index, 'raw_model_action': action.tolist(),
            'start_joint_deg': joints[index].tolist(), 'target_joint_deg': joints[index + 1].tolist(),
            'joint_step_max_deg': step,
            'minimum_joint_soft_limit_margin_deg': float(margins[index:index + 2].min()),
            'fk_samples': [],
        }
        report['segments'].append(segment)
        if np.any(margins[index + 1] < 0):
            report['failures'].append({'check': 'joint_soft_limits', 'action_index': index,
                'joint_numbers': (np.flatnonzero(margins[index + 1] < 0) + 1).tolist()})
        if step > MAX_JOINT_STEP_DEG:
            report['failures'].append({'check': 'joint_step', 'action_index': index,
                                       'step_deg': step, 'maximum_deg': MAX_JOINT_STEP_DEG})
    if report['failures']:
        return report

    decoded = [{'target': action.tolist()} for action in values[:prefix_steps]]
    try:
        clearance = check_table_clearance(state[:6], decoded, plane, rpc=rpc)
    except RPC_ERRORS as exc:
        report['tool_table_clearance'] = {'status': 'fk_error', 'passed': False,
                                          'error': f'{type(exc).__name__}: {exc}'}
        report['failures'].append({'check': 'forward_kinematics', 'error': f'{type(exc).__name__}: {exc}'})
        return report

    samples = clearance.pop('fk_samples')
    clearance['minimum_fk_sampled_clearance_mm_by_box'] = _record_sample_clearances(
        samples, plane['table_normal'], plane['table_offset_mm'])
    clearance['status'] = 'checked'
    report['anchor_fk_sample'] = samples[0]
    for sample in samples[1:]:
        report['segments'][sample['segment_index']]['fk_samples'].append(sample)
    report['tool_table_clearance'] = clearance
    if not clearance['passed']:
        report['failures'].append({'check': 'tool_table_clearance', 'error': clearance['error']})
    report['passed'] = not report['failures']
    return report


def main(argv=None, *, rpc=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-json', required=True, type=Path)
    parser.add_argument('--reply-json', required=True, type=Path)
    parser.add_argument('--scene-json', required=True, type=Path)
    parser.add_argument('--prefix-steps', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    request, reply, scene = (json.loads(path.read_text(encoding='utf-8'))
                             for path in (args.request_json, args.reply_json, args.scene_json))
    report = preflight(request, reply, scene, args.prefix_steps, rpc=FR5RPC() if rpc is None else rpc)
    report['sources'] = {'request_json': str(args.request_json), 'reply_json': str(args.reply_json),
                         'scene_json': str(args.scene_json)}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'output': str(args.output)}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
