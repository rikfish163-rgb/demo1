"""Capture, check and record a complete real object demonstration."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time

import cv2
import numpy as np

from lerobot_robot_fairino.feedback import gripper_feedback_valid

from fr5_runtime.kinematics import transform
from fr5_runtime.robot import require_ready
from fr5_runtime import system
from .object_geometry import selected_color_mask
from .observe import capture
from .prepare_object_session import prepare
from .prepare_varied_session import load_scene, sample_geometry, table_clearances
from .rgbd_geometry import sample_rgb_points
from .stream_storage import new_stream_recording_path
from .whole_task_collection import WholeTaskSession, durable_json


def _points_for_mask(mask, depth, source):
    v, u = np.nonzero(mask)
    sampled = sample_rgb_points(np.column_stack((u[::3], v[::3])), depth, source['camera'],
                                radius=1, maximum_patch_spread_mm=8.)
    matrix = np.asarray(source['calibration']['base_from_color_camera_mm'])
    points = sampled['points_camera_mm'][sampled['valid']] @ matrix[:3, :3].T + matrix[:3, 3]
    if len(points) < 100:
        raise ValueError('Too few measured visible points for the object path check')
    return points


def visible_tool_clearances(color, depth, source, geometry, target_mask, support_rgb_pixel=None):
    """Check the open tool against measured target and optional raised support."""
    clouds = {'target': _points_for_mask(target_mask, depth, source)}
    if support_rgb_pixel is not None:
        support = selected_color_mask(color, support_rgb_pixel)
        if np.any(support & target_mask):
            raise ValueError('Support selection overlaps the target selection')
        clouds['support'] = _points_for_mask(support, depth, source)
    boxes = json.loads((system.ROOT / 'hardware/tool-reference.json').read_text())[
        'tool_boxes_for_this_exterior_move']
    poses = []
    for route in geometry['routes'].values():
        for initial, terminal in zip(route, route[1:]):
            initial, terminal = np.asarray(initial), np.asarray(terminal)
            steps = max(2, int(np.ceil(np.linalg.norm(terminal[:3] - initial[:3]) / 2.)) + 1,
                        int(np.ceil(np.max(np.abs(terminal[3:] - initial[3:])) / .5)) + 1)
            poses.extend(initial + fraction * (terminal - initial)
                         for fraction in np.linspace(0., 1., steps))
    distances = {}
    for label, points in clouds.items():
        for part, box in boxes.items():
            bounds = np.asarray(box['local_bounds_mm']) + np.array(
                [[-box['padding_mm']] * 3, [box['padding_mm']] * 3])
            minimum = float('inf')
            for pose in poses:
                local = (points - pose[:3]) @ transform(pose)[:3, :3]
                gap = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0.)
                minimum = min(minimum, float(np.linalg.norm(gap, axis=1).min()))
            if minimum <= 0 or (part == 'housing' and minimum < 5.):
                raise ValueError(f'Measured visible {label} intersects {part} path: {minimum:.3f} mm')
            distances[f'{label}_{part}'] = minimum
    contact = np.asarray(geometry['contact_tcp'])
    local = (clouds['target'] - contact[:3]) @ transform(contact)[:3, :3]
    fingers = [np.asarray(box['local_bounds_mm']) for part, box in boxes.items()
               if part != 'housing']
    lower = np.max([bounds[0, 1:] for bounds in fingers], axis=0)
    upper = np.min([bounds[1, 1:] for bounds in fingers], axis=0)
    closing_span = np.all((local[:, 1:] >= lower) & (local[:, 1:] <= upper), axis=1)
    if not closing_span.any():
        raise ValueError('Visible target does not overlap the actual finger closing span')
    return {'open_padded_tool_visible_point_distance_mm': distances,
            'visible_target_points_in_finger_yz_span': int(closing_span.sum()),
            'visible_points': {label: len(points) for label, points in clouds.items()},
            'scope': 'selected visible surfaces and current open-tool boxes',
            'motion_commands_sent': 0, 'gripper_commands_sent': 0}


def _server_ready(name):
    return system.remote_json('''
import json,os,sys
from pathlib import Path
q=json.load(sys.stdin);root=Path(q['root'])
if not root.is_dir():raise ValueError('data4 storage root unavailable')
for parent in ('observations','recording-spool'):
 if (root/parent/q['name']).exists():raise FileExistsError('Use a fresh object recording name')
disk=os.statvfs(root)
free=disk.f_bavail*disk.f_frsize
if free < 6*1024**3:raise ValueError('Insufficient data4 capacity for whole-task recording')
print(json.dumps({'root':str(root),'free_bytes':free,'storage_available':True}))
''', {'root': system.REMOTE, 'name': name}, timeout=15)


def _upload_observation(observation, name, preflight, rows):
    remote = system.REMOTE + '/observations/' + name
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        for path in sorted(Path(observation).iterdir()):
            archive.add(path, arcname=path.name)
        for filename, value in [('collection-preflight.json', preflight), ('path-samples.json', rows)]:
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            item = tarfile.TarInfo(filename)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    code = '''
import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz').extractall(root,filter='data')
'''
    subprocess.run(system.ssh_args(shlex.join([system.PYTHON, '-B', '-c', code, remote])),
                   input=payload.getvalue(), check=True, timeout=30)
    return remote


def collect(*, name, object_description, rgb_pixel=None, mask_path=None, support_rgb_pixel=None,
            closing_percent, grasp_offset, approach='tilted', grasp_below_top_mm=10.,
            side_retraction_mm=0., tilt_from_down_deg=55., tilted_closing_edge='long',
            split='training_complete_trial', execute=False):
    """Prepare current geometry, then optionally run and archive one whole task."""
    name = system.simple_name(name)
    directory = new_stream_recording_path(name)
    storage = _server_ready(name)
    session = None
    summary = None
    error = None
    with system.robot_lease(), tempfile.TemporaryDirectory(prefix='fr5-object-', dir='/dev/shm') as tmp:
        observation = Path(tmp) / 'observation'
        source = capture(observation)
        state = source['robot_after']
        preflight = {'checked_unix': time.time(), 'server_storage': storage,
                     'source_observation': system.REMOTE + '/observations/' + name,
                     'source_completed_unix': source['completed_at_unix'],
                     'motion_commands_sent': 0, 'gripper_commands_sent': 0}
        rows = []
        try:
            require_ready(state)
            if not gripper_feedback_valid(state) or state['gripper_motiondone'] != 1:
                raise RuntimeError('Object collection needs settled empty-gripper feedback')
            prepare(directory, observation, object_description, rgb_pixel,
                    mask_path=mask_path, closing_percent=closing_percent, grasp_offset=grasp_offset,
                    approach=approach, grasp_below_top_mm=grasp_below_top_mm,
                    side_retraction_mm=side_retraction_mm, tilt_from_down_deg=tilt_from_down_deg,
                    tilted_closing_edge=tilted_closing_edge, split=split)
            report, color, depth, _ = load_scene(observation)
            geometry = json.loads((directory / 'geometry.json').read_text())['geometry_report']
            mask = cv2.imread(str(directory / 'selection-mask.png'), cv2.IMREAD_GRAYSCALE) != 0
            preflight['visible_sweep'] = visible_tool_clearances(
                color, depth, report, geometry, mask, support_rgb_pixel)
            table = geometry['table']
            preflight['tool_table_clearance_mm'] = table_clearances(
                geometry['routes'], np.asarray(table['normal']), table['offset_mm'])
            rows, preflight['kinematics'] = sample_geometry(geometry['routes'], state['joints'])
            preflight.update(status='passed', sample_count=len(rows), checked_unix=time.time())
        except BaseException as exc:
            preflight.update(status='failed', error=str(exc), checked_unix=time.time())
            directory.mkdir(parents=True, exist_ok=True)
            durable_json(directory / 'collection-preflight.json', preflight)
            _upload_observation(observation, name, preflight, rows)
            raise
        durable_json(directory / 'collection-preflight.json', preflight)
        _upload_observation(observation, name, preflight, rows)
        if not execute:
            return {'status': 'prepared_no_motion', 'directory': str(directory),
                    'source_observation': preflight['source_observation'], 'preflight': preflight,
                    'task_success_verified': False, 'motion_commands_sent': 0, 'gripper_commands_sent': 0}
        session = WholeTaskSession(directory)
        session.defer_materialization = True
        prior_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in prior_handlers:
                signal.signal(sig, lambda *_: session.interrupted.set())
            session.start()
            summary = session.run()
        except BaseException as exc:
            error = exc
            session.failed = str(exc)
            if session.controller is not None:
                session.controller.stop(str(exc))
            if session.writer is None or not session.writer._finished:
                summary = session.finish('interrupted')
            elif session.storage is not None:
                session.storage.frame_spool.abort()
        finally:
            session._watchdog_stop.set()
            if session.robot.is_connected:
                session.robot.disconnect()
            for sig, handler in prior_handlers.items():
                signal.signal(sig, handler)
    closed = (directory / 'stream-closed.json').exists()
    result_path = system.LOCAL_STATE / (name + '-collection-result.json')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': summary['outcome'] if summary else 'interrupted',
              'directory': str(directory), 'summary': summary, 'archive': None,
              'result_record': str(result_path),
              'archive_status': 'pending' if closed else 'not_recorded',
              'task_success_verified': False, 'success_requires_user_review': True}
    if error is not None:
        result['error'] = f'{type(error).__name__}: {error}'
    durable_json(result_path, result)
    if closed:
        try:
            result['archive'] = system.archive_stream(name, render=True)
            result['archive_status'] = 'completed'
        except Exception as exc:
            result['status'] = 'archive_pending'
            result['archive_error'] = f'{type(exc).__name__}: {exc}'
        durable_json(result_path, result)
    return result
