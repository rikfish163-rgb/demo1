"""Continuous whole-task teaching with typed phase decisions and the deployment driver.

LLM supplies the current scene/route once. A decision selects a phase once; normal
execution has no per-waypoint chat checkpoint or millimetre subdivision.
This module is inert until ``run --execute``. Native faults/limits and genuine
command completion remain active. Only a subsequent user review labels success.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from lerobot_robot_fairino.action_control import FR5ActionController, finite_vector, rpc_vector, angle_error
from lerobot_robot_fairino.feedback import gripper_feedback_valid
from lerobot_robot_fairino.pi05_rollout import PolicyController, DirectSettings, _camera_fresh
from lerobot_camera_gemini.timing import MONOTONIC, camera_clock
from .teacher_targets import TeacherSession, save
from .teacher_varied import STAGES
from .continuous_workflow import PHASE_SKILLS

TASK = '抓取桌面上的透明杯子，在空中维持5秒后放下'
DEFAULT_OBJECT = '透明杯子'
CONTRACT = 'fr5-whole-task-collection-v1'
PLAN_SOURCES = ('current_codex_chat', 'reused_llm_workflow_with_user_cup_selection',
                'reused_llm_workflow_with_visual_localization')
PHASE_DECISION = {'provider': 'openai', 'endpoint': 'https://api.openai.com/v1/decisions',
                  'model': 'gpt-6-luna'}


def task_stages(retreat_mode='full'):
    if retreat_mode not in ('short', 'full'):
        raise ValueError('Unknown retreat mode')
    return STAGES[:-1] if retreat_mode == 'short' else STAGES.copy()


def open_gripper_complete(state):
    """Match the native driver's integer-percent completion tolerance."""
    return (gripper_feedback_valid(state) and state['gripper_motiondone'] == 1
            and abs(state['gripper_position']-100) <= 1)


def task_instruction(hold_seconds, repetitions, object_description=DEFAULT_OBJECT):
    if (type(hold_seconds) not in (int, float) or not math.isfinite(hold_seconds)
            or hold_seconds <= 0 or type(repetitions) is not int or repetitions not in (1, 2)):
        raise ValueError('A positive finite hold and one or two repetitions are required')
    if (not isinstance(object_description, str) or not object_description.strip()
            or object_description != object_description.strip()
            or any(char in object_description for char in '\r\n')):
        raise ValueError('A nonempty single-line object description is required')
    seconds = repr(float(hold_seconds))
    if seconds.endswith('.0'):
        seconds = seconds[:-2]
    first = f'抓取桌面上的{object_description}，在空中维持{seconds}秒后放下'
    if repetitions == 1:
        return first
    return first + f'并松开夹爪，再次抓起，在空中维持{seconds}秒后放下并松开夹爪'


def durable_json(path, value):
    """Small local journal survives reboot; media streams independently to data4."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def record_llm_route_plan(directory, reason, *, source='current_codex_chat'):
    """Persist the current chat's concrete route choice, separate from Jev.

    Current-chat source requires actual review of this scene. Console reuse
    instead records the user's cup selection and the existing LLM workflow;
    it does not claim a new visual LLM call or current-chat image review.
    """
    root = Path(directory).resolve()
    if (source not in PLAN_SOURCES or not isinstance(reason, str) or not reason.strip()
            or (root/'SESSION_STARTED').exists()):
        raise ValueError('A real planner reason and an unstarted episode are required')
    scope_raw = (root/'scope.json').read_bytes()
    scope = json.loads(scope_raw)
    validate_scope(scope)
    plan = {'schema':'fr5-chat-route-plan-v1','source':source,
        'scope_sha256':hashlib.sha256(scope_raw).hexdigest(),
        'scene_image_sha256':hashlib.sha256((root/'scene.jpg').read_bytes()).hexdigest(),
        'reason':reason.strip(), 'created_unix':time.time(),
        'phase_choices':{phase:PHASE_SKILLS[phase] for phase in scope['phase_order']},
        'independent_llm_api_calls':0,
        'basis':('current chat scene interpretation and route; native feedback still required'
                 if source=='current_codex_chat' else
                 'existing current-chat workflow; target is explicitly identified by plan source; no new LLM visual review')}
    with (root/'llm-route-plan.json').open('x') as handle:
        json.dump(plan,handle,ensure_ascii=False,indent=2)
    return plan


def endpoint_actions(routes, *, closing_percent=76, cycle=1, retreat_mode='full',
                     preserve_return_waypoints=False):
    """Keep task waypoints, remove legacy 1/2/5 mm interpolation and close ramps."""
    if (list(routes) != STAGES or type(closing_percent) not in (int, float)
            or not math.isfinite(closing_percent) or not 0 <= closing_percent < 100
            or type(cycle) is not int or cycle not in (1, 2)):
        raise ValueError('Expected a complete ordered route and physical closing target')
    result = []
    for phase in task_stages(retreat_mode):
        points = [finite_vector(p, 6, 'route pose').tolist() for p in routes[phase]]
        if len(points) < 2:
            raise ValueError('Route needs a source and destination')
        if phase in ('close', 'release'):
            result.append({'id': f'c{cycle:02d}-{phase}', 'cycle': cycle, 'phase': phase, 'kind': 'gripper',
                           'gripper_percent': closing_percent if phase == 'close' else 100})
            continue
        # The return is one straight segment to the original support pose.
        # Other original route corners remain; no artificial subdivisions.
        destinations = points[-1:] if phase == 'return' and not preserve_return_waypoints else points[1:]
        previous = points[0]
        for point in destinations:
            if not np.allclose(point, previous, atol=1e-7, rtol=0):
                result.append({'id': f'c{cycle:02d}-{phase}-{len(result):03d}', 'cycle': cycle, 'phase': phase,
                               'kind': 'arm', 'tcp_mm_deg': point})
            previous = point
    return result


def validate_scope(scope):
    spec = scope.get('task_spec')
    if not isinstance(spec, dict) or spec.get('finish') != 'released':
        raise ValueError('Expected a released whole-task specification')
    seconds, repetitions = spec.get('hold_seconds'), spec.get('repetitions')
    object_description = spec.get('object_description', DEFAULT_OBJECT)
    instruction = task_instruction(seconds, repetitions, object_description)
    stages = task_stages(scope.get('retreat_mode', 'full'))
    if (scope.get('contract') != CONTRACT or scope.get('robot_ip') != '192.168.58.2'
            or scope.get('task_instruction') != instruction or scope.get('phase_order') != stages * repetitions
            or scope.get('hold_seconds') != seconds or scope.get('whole_task') is not True
            or scope.get('gripper_speed') != 20 or scope.get('force') != 10
            or scope.get('split_role') not in ('training_complete_trial', 'held_out_complete_trial')
            or not scope.get('authorization_source')):
        raise ValueError('Invalid whole-task collection contract')
    if 'phase_decision' in scope and scope['phase_decision'] != PHASE_DECISION:
        raise ValueError('Whole-task phase decision differs from the configured official model')
    DirectSettings(speed_percent=scope['move_speed'])
    finite_vector(scope['initial_tcp'], 6, 'start TCP')
    finite_vector(scope['contact_tcp'], 6, 'contact TCP')
    actions = scope.get('actions', [])
    order = [(a.get('cycle'), stages.index(a['phase'])) for a in actions]
    if not actions or order != sorted(order) or any(cycle not in range(1, repetitions + 1) for cycle, _ in order):
        raise ValueError('Actions must follow the full-task order')
    if any({a['phase'] for a in actions if a['cycle'] == cycle} != set(stages)
           for cycle in range(1, repetitions + 1)):
        raise ValueError('A full approach/pick/hold/place/release/retreat route is required')
    ids = [a.get('id') for a in actions]
    if any(not isinstance(name, str) or not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError('Actions require unique IDs')
    for action in actions:
        if (action['phase'] in ('close', 'release')) != (action['kind'] == 'gripper'):
            raise ValueError('Command kind must match the actual phase')
        if action['kind'] == 'arm':
            finite_vector(action['tcp_mm_deg'], 6, 'target TCP')
        elif action['kind'] == 'gripper':
            if type(action['gripper_percent']) not in (int, float) or not 0 <= action['gripper_percent'] <= 100:
                raise ValueError('Invalid physical gripper target')
        else:
            raise ValueError('Unknown endpoint action')
    for cycle in range(1, repetitions + 1):
        if ([a['gripper_percent'] for a in actions if a['cycle'] == cycle and a['phase'] == 'release'] != [100]
                or len([a for a in actions if a['cycle'] == cycle and a['phase'] == 'close']) != 1):
            raise ValueError('Each cycle needs one closure and one full release')
    if 'planned_routes' in scope:
        routes = scope['planned_routes']
        closing_percent = scope.get('closing_percent')
        _validate_object_routes(routes, scope['initial_tcp'], scope['contact_tcp'], repetitions,
                                scope.get('retreat_mode', 'full'))
        if (type(closing_percent) not in (int, float) or not math.isfinite(closing_percent)
                or not 0 <= closing_percent < 100 or not float(closing_percent).is_integer()):
            raise ValueError('Closing target must be an integer physical percentage below 100')
        expected = [action for cycle, route in enumerate(routes, 1)
                    for action in endpoint_actions(route, closing_percent=closing_percent, cycle=cycle,
                        retreat_mode=scope.get('retreat_mode', 'full'), preserve_return_waypoints=True)]
        if (actions != expected or scope.get('manual_action_plan', {}).get('actions') != actions
                or scope.get('terminal_tcp') != routes[-1][stages[-1]][-1]):
            raise ValueError('Actions must match the recorded route and physical targets')


def prepare(root, observation, cup_uv, *, grasp_reference=None, speed=10., split='training_complete_trial',
            hold_seconds=5., repetitions=1, retreat_mode='short'):
    """Prepare from a current LLM-selected cup pixel, without any motion RPC.

    Cup pixels are explicit scene input, not a fixed historical image ROI. The
    empirical grasp offset is retained as such; it is not claimed as a new TCP
    calibration. Preparation does not assert that the cup was actually lifted.
    """
    instruction = task_instruction(hold_seconds, repetitions)
    stages = task_stages(retreat_mode)
    hold_seconds = float(hold_seconds)
    from .prepare_varied_session import load_scene, fit_table, estimate_cup, build_routes, load_grasp_reference
    from .motion_profiles import CONTINUOUS_PROFILE
    from .stream_storage import LOCAL_STREAM_BASE
    root, observation = Path(root).resolve(), Path(observation).resolve()
    if root.parent != LOCAL_STREAM_BASE or root.exists():
        raise ValueError('Choose a fresh directory in the bounded recording spool')
    report, color, depth, calibration = load_scene(observation)
    uv = finite_vector(cup_uv, 2, 'cup pixel')
    if not (0 <= uv[0] < color.shape[1] and 0 <= uv[1] < color.shape[0]):
        raise ValueError('Cup pixel outside actual image')
    normal, offset, table = fit_table(color, depth, report, calibration, uv)
    base, _, cup = estimate_cup(report, depth, calibration, uv, normal, offset)
    start = finite_vector(report['robot_after']['tcp'], 6, 'observed start')
    reference = load_grasp_reference(Path(grasp_reference)) if grasp_reference else None
    routes, model = build_routes(start, base, normal, profile=CONTINUOUS_PROFILE, grasp_reference=reference)
    model['executed_phase_order'] = stages
    model['terminal_pose'] = routes[stages[-1]][-1]
    geometry = {'routes': routes, 'cup_model': cup, 'route_model': model, 'table': table,
                'cup_uv_selected_by_llm': uv.tolist(), 'source_observation': str(observation),
                'source_observation_sha256': hashlib.sha256((observation/'observation.json').read_bytes()).hexdigest(),
                'geometric_assumptions_are_measurements': False}
    actions = [action for cycle in range(1, repetitions + 1)
               for action in endpoint_actions(routes, cycle=cycle, retreat_mode=retreat_mode)]
    scope = {'contract': CONTRACT, 'whole_task': True, 'robot_ip': '192.168.58.2',
             'phase_decision': deepcopy(PHASE_DECISION),
             'task_instruction': instruction, 'phase_order': stages * repetitions, 'hold_seconds': hold_seconds,
             'retreat_mode': retreat_mode, 'terminal_tcp': model['terminal_pose'],
             'task_spec': {'hold_seconds': hold_seconds, 'repetitions': repetitions, 'finish': 'released',
                           'object_description': DEFAULT_OBJECT},
             'initial_tcp': start.tolist(), 'contact_tcp': model['contact_pose'],
             'actions': actions, 'manual_action_plan': {'actions': actions}, 'move_speed': speed,
             'gripper_speed': 20, 'force': 10,
             'storage_estimate_s': max(1800, math.ceil(repetitions * hold_seconds + 120)),
             'split_role': split, 'only_prealigned_instance': False,
             'authorization_source': 'existing user authorization for autonomous onsite-supervised collection',
             'onsite_clearance_source': 'user statement; not an independent collision certification',
             'prepared_unix': time.time(), 'evidence': ['scene.jpg', 'source-observation.json', 'geometry.json']}
    validate_scope(scope)
    root.mkdir(parents=True, exist_ok=False)
    (root/'scene.jpg').write_bytes((observation/'color.jpg').read_bytes())
    (root/'source-observation.json').write_bytes((observation/'observation.json').read_bytes())
    save(root/'geometry.json', geometry)
    save(root/'scope.json', scope)
    return {'status': 'prepared_no_motion', 'directory': str(root), 'planned_actions': len(scope['actions']),
            'task': instruction, 'task_spec': scope['task_spec'], 'success': False}


def _validate_object_routes(routes, initial_tcp, contact_tcp, repetitions, retreat_mode):
    if not isinstance(routes, list) or len(routes) != repetitions:
        raise ValueError('One complete ordered route is required for each cycle')
    start = finite_vector(initial_tcp, 6, 'observed start')
    contact = finite_vector(contact_tcp, 6, 'contact TCP')
    final_phase = task_stages(retreat_mode)[-1]
    for route in routes:
        if not isinstance(route, dict) or list(route) != STAGES:
            raise ValueError('Every cycle requires the complete ordered route')
        previous = start
        poses = {}
        for phase in STAGES:
            if not isinstance(route[phase], list) or len(route[phase]) < 2:
                raise ValueError('Each route phase needs a source and destination')
            points = [finite_vector(point, 6, 'route pose') for point in route[phase]]
            if not np.allclose(points[0], previous, atol=1e-5, rtol=0):
                raise ValueError('Route phases and cycles must join at their actual endpoints')
            previous, poses[phase] = points[-1], points
        for phase in ('close', 'release'):
            if any(not np.allclose(point, contact, atol=1e-5, rtol=0) for point in poses[phase]):
                raise ValueError('Close and release must occur at the contact TCP')
        if (not np.allclose(poses['approach_contact'][-1], contact, atol=1e-5, rtol=0)
                or not np.allclose(poses['return'][-1], contact, atol=1e-5, rtol=0)
                or poses['lift_probe'][-1][2] <= contact[2]
                or poses['lift_full'][-1][2] <= poses['lift_probe'][-1][2]):
            raise ValueError('Route must lift from and return to the original contact TCP')
        start = poses[final_phase][-1]


def _check_observation_age(completed_at):
    if (type(completed_at) not in (int, float) or not math.isfinite(completed_at)
            or not -0.5 <= time.time() - completed_at <= 300):
        raise ValueError('A current observation completed within five minutes is required')


def prepare_from_routes(root, observation, object_description, routes, contact_tcp, geometry, *,
                        closing_percent, speed=10., split='training_complete_trial', hold_seconds=5.,
                        repetitions=1, retreat_mode='full', selection_mask=None):
    """Prepare a recorded object task from current RGB-D evidence and supplied physical routes."""
    import cv2
    from .prepare_varied_session import load_scene
    from .stream_storage import LOCAL_STREAM_BASE, MAX_FILE

    instruction = task_instruction(hold_seconds, repetitions, object_description)
    stages = task_stages(retreat_mode)
    root, observation = Path(root).resolve(), Path(observation).resolve()
    if root.parent != LOCAL_STREAM_BASE or root.exists():
        raise ValueError('Choose a fresh directory in the bounded recording spool')
    report, color, depth, _ = load_scene(observation)
    source_scene = (observation/'scene.npz').read_bytes()
    if len(source_scene) > MAX_FILE:
        raise ValueError('The source RGB-D observation exceeds the streaming control file limit')
    completed_at = report.get('completed_at_unix')
    _check_observation_age(completed_at)
    with np.load(observation/'scene.npz', allow_pickle=False) as scene:
        metadata = json.loads(str(scene['metadata'].item()))
    if metadata != report['camera']:
        raise ValueError('RGB-D metadata and observation report do not match')
    captured_at = metadata['capture']['time_unix']
    _check_observation_age(captured_at)
    capture_age = report.get('camera_age_at_capture_s')
    if (report.get('status') != 'observation_completed' or report.get('motion_commands_sent') != 0
            or report.get('gripper_commands_sent') != 0 or color.shape != (1080, 1920, 3)
            or depth.shape != (1080, 1920) or np.mean(depth > 0) < .2
            or type(capture_age) not in (int, float) or not 0 <= capture_age < 2
            or completed_at < captured_at - .5):
        raise ValueError('A complete read-only RGB-D observation is required')
    start = finite_vector(report['robot_after']['tcp'], 6, 'observed start')
    before = finite_vector(report['robot_before']['tcp'], 6, 'observation start')
    if not np.allclose(start, before, atol=.1, rtol=0):
        raise ValueError('The robot moved during the source observation')
    contact = finite_vector(contact_tcp, 6, 'contact TCP').tolist()
    if not isinstance(geometry, dict) or not geometry:
        raise ValueError('A caller-supplied geometry report is required')
    json.dumps(geometry, ensure_ascii=False, allow_nan=False)
    if isinstance(routes, dict):
        routes = [routes] * repetitions
    if not isinstance(routes, list):
        raise ValueError('Routes must be an ordered route or one route per cycle')
    normalized = []
    for route in routes:
        if not isinstance(route, dict) or list(route) != STAGES:
            raise ValueError('Every cycle requires the complete ordered route')
        normalized.append({phase: [finite_vector(point, 6, 'route pose').tolist() for point in route[phase]]
                           for phase in STAGES})
    _validate_object_routes(normalized, start, contact, repetitions, retreat_mode)
    if 'source_observation' in geometry and Path(geometry['source_observation']).resolve() != observation:
        raise ValueError('Geometry report refers to a different source observation')
    if ('object_description' in geometry and geometry['object_description'] != object_description):
        raise ValueError('Geometry report describes a different object')
    if 'closing_percent' in geometry and geometry['closing_percent'] != closing_percent:
        raise ValueError('Geometry report closing target differs from the requested target')
    if ('contact_tcp' in geometry
            and not np.allclose(finite_vector(geometry['contact_tcp'], 6, 'geometry contact TCP'),
                                contact, atol=1e-5, rtol=0)):
        raise ValueError('Geometry report contact TCP differs from the requested route')
    if 'routes' in geometry:
        geometry_routes = geometry['routes']
        if isinstance(geometry_routes, dict):
            geometry_routes = [geometry_routes] * repetitions
        if geometry_routes != normalized:
            raise ValueError('Geometry report routes differ from the requested routes')
    mask_png = None
    if selection_mask is not None:
        mask = np.asarray(selection_mask)
        if mask.ndim != 2 or mask.shape != color.shape[:2] or not np.any(mask):
            raise ValueError('Selection mask must have foreground on the current raw RGB grid')
        encoded, png = cv2.imencode('.png', (mask != 0).astype(np.uint8) * 255)
        if not encoded or len(png) > MAX_FILE:
            raise ValueError('Selection mask cannot be saved as a bounded PNG')
        mask_png = png.tobytes()
    actions = [action for cycle, route in enumerate(normalized, 1)
               for action in endpoint_actions(route, closing_percent=closing_percent, cycle=cycle,
                   retreat_mode=retreat_mode, preserve_return_waypoints=True)]
    spec = {'object_description': object_description, 'hold_seconds': float(hold_seconds),
            'repetitions': repetitions, 'finish': 'released'}
    scope = {'contract': CONTRACT, 'whole_task': True, 'robot_ip': '192.168.58.2',
             'phase_decision': deepcopy(PHASE_DECISION),
             'task_instruction': instruction, 'phase_order': stages * repetitions,
             'hold_seconds': float(hold_seconds), 'task_spec': spec, 'retreat_mode': retreat_mode,
             'initial_tcp': start.tolist(), 'contact_tcp': contact,
             'terminal_tcp': normalized[-1][stages[-1]][-1], 'planned_routes': normalized,
             'closing_percent': closing_percent, 'actions': actions, 'manual_action_plan': {'actions': actions},
             'move_speed': speed, 'gripper_speed': 20, 'force': 10,
             'storage_estimate_s': max(1800, math.ceil(repetitions * hold_seconds + 120)),
             'split_role': split, 'only_prealigned_instance': False,
             'authorization_source': 'user requested preparation of a real object collection',
             'onsite_clearance_source': 'not assessed during preparation',
             'source_observation_completed_at_unix': completed_at, 'prepared_unix': time.time(),
             'evidence': ['scene.jpg', 'source-observation.json', 'source-scene.npz', 'geometry.json']}
    if mask_png is not None:
        scope['selection_mask'] = 'selection-mask.png'
        scope['evidence'].append(scope['selection_mask'])
    validate_scope(scope)
    evidence = {'source_observation': str(observation), 'routes': normalized,
                'contact_tcp': contact, 'geometry_report': deepcopy(geometry)}
    encoded, scene_jpeg = cv2.imencode('.jpg', color)
    if not encoded:
        raise ValueError('Cannot encode the source RGB-D color image')
    root.mkdir(parents=True, exist_ok=False)
    (root/'scene.jpg').write_bytes(scene_jpeg.tobytes())
    (root/'source-observation.json').write_bytes((observation/'observation.json').read_bytes())
    (root/'source-scene.npz').write_bytes(source_scene)
    if mask_png is not None:
        (root/scope['selection_mask']).write_bytes(mask_png)
    save(root/'geometry.json', evidence)
    save(root/'scope.json', scope)
    return {'status': 'prepared_no_motion', 'directory': str(root), 'planned_actions': len(actions),
            'task': instruction, 'task_spec': spec, 'success': False}


class RecordingRPC:
    """Capture actual wire arguments; never alter, enqueue or retry commands."""
    def __init__(self, rpc, session):
        self.inner, self.session = rpc, session

    def call(self, method, *args, **kwargs):
        session = self.session
        if method in ('MoveL', 'MoveGripper'):
            observed = deepcopy(session.decision_observation)
            if observed is None:
                raise RuntimeError('A recorded decision observation is required before dispatch')
            session.pending_wire = {'method': method, 'args': deepcopy(args),
                'initial_state': session.state(), 'phase': session.phase,
                'cycle': session.cycle, 'decision_id': session.decision_id, 'status': 'prepared', 'samples': [],
                'observation_before_command': observed['camera'], 'decision_observation': observed}
            before_send = kwargs.get('before_send')
            if not callable(before_send):
                raise RuntimeError('Command transport requires the controller send gate')

            @contextmanager
            def recorded_send():
                try:
                    # The original gate holds the controller lock only through
                    # the actual wire write. Persist after it is released so a
                    # slow filesystem cannot block the independent stop path.
                    with before_send():
                        yield
                finally:
                    session.persist_attempted_command()
            kwargs['before_send'] = recorded_send
        return self.inner.call(method, *args, **kwargs)


class TeachingController(PolicyController):
    def __init__(self, session, **kwargs):
        self.session = session
        super().__init__(**kwargs)

    def _append(self, target, kind, *, executed):
        s = self.session
        if executed and s.pending_wire is None:
            raise RuntimeError('Missing prepared wire record')
        target = list(map(float, target))
        # An arm RPC does not change the latched gripper instruction, and a
        # gripper RPC does not change the latched arm endpoint. Measured state
        # remains separately recorded; never replace target labels with it.
        if self._journal:
            previous = self._journal[-1]['target']
            if kind == 'arm':
                target[6] = previous[6]
            elif kind == 'gripper':
                target[:6] = previous[:6]
        row = FR5ActionController._append(self, target, kind, executed=executed)
        row.update(source='agent_teacher', teacher={'policy': s.policy_name,
                   'phase': s.phase, 'cycle': s.cycle, 'decision_id': s.decision_id})
        self._journal[-1] = deepcopy(row)
        with s.lock:
            s.journal.append(deepcopy(row))
            s.command_sequence, s.target = row['sequence'], row['target']
            if executed:
                s.pending_wire.update(command=row, intended_target=target, command_kind=kind,
                                      status='attempted_once')
        return deepcopy(row)


class WholeTaskSession(TeacherSession):
    validate_scope = staticmethod(validate_scope)

    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)
        if self.scope.get('phase_decision') == PHASE_DECISION:
            self.policy_name = 'current_codex_chat+gpt-6-luna'
        # This is a storage reservation estimate, never a task deadline.
        self.max_session_s = self.scope.get('storage_estimate_s', self.max_session_s)
        self.task_description = self.scope['task_instruction']
        self.origin = np.asarray(self.scope['contact_tcp'], float)
        self.completed_stages, self.held_seconds = [], 0.
        self.hold_durations, self.completed_cycles = [], 0
        self.hold_records = []
        self.return_endpoint = None
        self.cycle = 0
        self.controller = None
        self.pending_wire = None
        self.decision_observation = None
        self._last_recorded_sample = None
        self._last_completed_monotonic = None
        self._pending_hold = None
        self._prefetch_thread = None
        self._prefetch_result = None
        self._prefetch_ready = threading.Event()
        self._watchdog_stop = threading.Event()

    def persist_attempted_command(self):
        with self.lock:
            record = deepcopy(self.pending_wire)
            journal = deepcopy(self.journal)
        if record and record.get('command'):
            durable_json(self.root/'target-command-journal.json', journal)
            durable_json(self.root/f"command-{record['command']['sequence']:04d}.json", record)
            if record.get('phase') == 'return' and self._pending_hold is not None:
                self.complete_hold_at_return(record['command']['issued_monotonic'], record['command']['issued_unix'])

    def complete_hold_at_return(self, issued, issued_unix):
        hold = self._pending_hold
        if hold is None or hold['cycle'] != self.cycle or issued < hold['start_monotonic']:
            raise RuntimeError('Return command does not match its pending hold')
        record = {**hold, 'repetition': self.cycle, 'end_monotonic': issued,
                  'end_unix': issued_unix, 'actual_duration_s': issued-hold['start_monotonic'],
                  'actual_hover_from_lift_completion_s': issued-hold['lift_completed_monotonic'],
                  'timing_deviation_seconds': issued-hold['lift_completed_monotonic']-hold['requested_seconds'],
                  'return_command_issued': True}
        path = f'hold-{self.cycle:02d}.json'
        durable_json(self.root/path, record)
        if self.cycle == 1 and hold['requested_seconds'] == 5:
            durable_json(self.root/'hold-5s.json', record)
        self.hold_durations.append(record['actual_duration_s'])
        self.held_seconds = sum(self.hold_durations)
        self.hold_records.append({'repetition': self.cycle, 'path': path})
        self._pending_hold = None

    def interrupt_pending_hold(self, reason):
        hold = self._pending_hold
        if hold is None:
            return
        ended = time.monotonic()
        cycle = hold['cycle']
        incomplete = {**hold, 'repetition': cycle, 'end_monotonic': ended,
            'end_unix': time.time(), 'actual_duration_s': ended-hold['start_monotonic'],
            'actual_hover_from_lift_completion_s': ended-hold['lift_completed_monotonic'],
            'return_command_issued': False, 'measurement': 'interrupted_before_return', 'error': str(reason)}
        path = f'hold-{cycle:02d}.json'
        durable_json(self.root/path, incomplete)
        self.hold_records.append({'repetition': cycle, 'path': path})
        self._pending_hold = None

    def sample(self, sample):
        """Remember the paired observation; capture confirms spool acceptance."""
        row = super().sample(sample)
        if row is not None:
            captured, basis = camera_clock(sample['camera']['metadata'])
            with self.lock:
                self._last_recorded_sample = {
                    'camera': deepcopy(sample['camera']['metadata']),
                    'robot': deepcopy(sample['robot_state']),
                    'color_rgb': sample['camera']['color_rgb'].copy(),
                    'camera_time': captured, 'clock': basis,
                    'robot_time_monotonic': sample['robot_state'].get('received_monotonic'),
                }
        return row

    def fresh_recorded_observation(self, *, after_monotonic=None, image_path=None):
        """Use a committed pair; an observation request waits for a new frame."""
        while True:
            self.state()
            with self.lock:
                stored = self._last_recorded_sample
                sample = ({key: deepcopy(value) for key, value in stored.items() if key != 'color_rgb'}
                          if stored else None)
                rgb = (stored['color_rgb'].copy()
                       if image_path is not None and stored and 'color_rgb' in stored else None)
            with self.capture._lock:
                times = list(self.capture._times)
                basis = self.capture._clock_basis
            if times and basis != MONOTONIC:
                raise RuntimeError('Recorded observation requires the monotonic clock')
            if sample and times and basis == MONOTONIC and sample['clock'] == MONOTONIC:
                camera_time = sample['camera_time']
                robot_time = sample['robot_time_monotonic']
                after = self._last_completed_monotonic
                if (type(robot_time) in (int, float) and math.isfinite(robot_time)
                        and times[-1] == camera_time
                        and (after is None or (camera_time > after and robot_time > after))
                        and (after_monotonic is None
                             or (camera_time > after_monotonic and robot_time > after_monotonic))):
                    sample['frame_index'] = len(times)-1
                    sample['previous_completion_monotonic'] = after
                    if image_path is not None:
                        import cv2
                        if rgb is None:
                            raise RuntimeError('Recorded observation has no RGB image')
                        encoded, jpeg = cv2.imencode('.jpg', rgb[:, :, ::-1],
                                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if not encoded:
                            raise RuntimeError('Cannot encode the recorded RGB image')
                        image_path = Path(image_path)
                        with image_path.open('xb') as handle:
                            handle.write(jpeg.tobytes())
                        sample['image_file'] = image_path.name
                    return sample
            time.sleep(.02)

    def prepare_storage(self):
        from .stream_storage import StreamStorage
        # Storage owns persistence, not motion permission. The historical
        # readiness manifest describes a different executor and is not forged
        # or silently rehashed to certify this implementation.
        self.storage = StreamStorage.start_readonly(self.root, self.scope_sha,
            required_bytes=512*1024**2 + self.max_session_s*6*512*1024,
            spool_settings={'max_frames':120,'max_bytes':32*1024**2,'max_unacked_age':30.})

    def append(self, target, kind, executed):
        # arm() already recorded the explicit initial hold before capture.
        if kind == 'hold' and not executed and self.journal:
            return deepcopy(self.journal[-1])
        raise RuntimeError('Only the controller guarded send may append an action')

    def state(self, *, moving=False, held=False, gripper_reference=None):
        if self.failed or self.interrupted.is_set() or (self.root/'STOP').exists():
            raise RuntimeError('Episode stopped')
        if self.controller is not None and self.controller.error:
            raise RuntimeError(self.controller.error)
        if self.capture is not None:
            self.capture.assert_healthy()
        state = self.controller.ready_state(stationary=not moving)
        if held and (not gripper_feedback_valid(state) or state['gripper_motiondone'] != 2):
            raise RuntimeError('Gripper contact feedback was lost while holding')
        self.last_read_state = deepcopy(state)
        return state

    def watchdog_once(self):
        try:
            if self.interrupted.is_set() or (self.root/'STOP').exists():
                raise RuntimeError('Operator interrupt')
            self.capture.assert_healthy()
            self.controller.watchdog()
            if self.controller.error:
                raise RuntimeError(self.controller.error)
        except Exception as exc:
            self.failed = str(exc)
            self.controller.stop(str(exc))
            return False
        return True

    def start(self):
        if 'source_observation_completed_at_unix' in self.scope:
            _check_observation_age(self.scope['source_observation_completed_at_unix'])
        self.prepare_storage()
        (self.root/'SESSION_STARTED').open('x').close()
        self.robot.connect()
        self.robot.get_sample()
        settings = DirectSettings(speed_percent=self.scope['move_speed'])
        self.controller = TeachingController(self, stream=self.robot.stream,
            camera_fresh=lambda: _camera_fresh(self.robot), limits=settings,
            allow_motion=True, rpc=RecordingRPC(self.rpc, self))
        self.controller.arm(operator_confirmed=True)  # existing user authorization
        durable_json(self.root/'target-command-journal.json', deepcopy(self.journal))
        self.controller.deadline = float('inf')
        state = self.state()
        if 'source_observation_completed_at_unix' in self.scope:
            _check_observation_age(self.scope['source_observation_completed_at_unix'])
            if not np.allclose(state['tcp'], self.expected_start_pose, atol=.1, rtol=0):
                raise RuntimeError('Start differs from the source observation')
        if not gripper_feedback_valid(state) or state['gripper_motiondone'] != 1:
            raise RuntimeError('Collection start requires ready empty-gripper feedback')
        self.begin_recording(state, state['gripper_position'])
        def watch():
            while not self._watchdog_stop.wait(.05):
                if not self.watchdog_once():
                    return
        self._watchdog = threading.Thread(target=watch, daemon=True)
        self._watchdog.start()
        if not open_gripper_complete(state):
            # Opening the currently empty gripper is part of preparing this
            # grasp, not a separate approval or unrecorded hardware action.
            self.phase, self.decision_id = 'initial_open', 'scope.json'
            self.capture.phase('initial_open', teacher={'policy': self.policy_name, 'decision_id': 'scope.json'})
            state = self.dispatch({'kind': 'gripper', 'gripper_percent': 100})
        return state

    def decide(self, phase, filename, *, observe_request=None):
        """Evaluate this cycle's real route using its recorded decision provider."""
        from .decision import CONTRACT_VERSION, make_request, call_jev, validate_answer, DecisionError, provider_config
        from .memory import retrieve
        object_description = self.scope['task_spec'].get('object_description', DEFAULT_OBJECT)
        skill = PHASE_SKILLS[phase]
        actions = [a for a in self.scope['actions'] if a['cycle'] == self.cycle and a['phase'] == phase]
        expected = f'jev-c{self.cycle:02d}-{phase}' + ('-after-observe' if observe_request is not None else '') + '.json'
        if not actions or filename != expected:
            raise ValueError('Decision must refer to this cycle and a declared route')
        held = skill in ('lift_verify', 'return_place', 'release_gripper')
        self.state(held=held)
        observation_name = f'observe-c{self.cycle:02d}-{phase}'
        observation = self.fresh_recorded_observation(
            after_monotonic=observe_request['requested_monotonic'] if observe_request is not None else None,
            image_path=self.root/(observation_name+'.jpg') if observe_request is not None else None)
        if observe_request is not None:
            observation.update(**observe_request, record=observation_name+'.json')
            save(self.root/observation['record'], observation)
        self.state(held=held)
        state = observation['robot']
        if held and (not gripper_feedback_valid(state) or state['gripper_motiondone'] != 2):
            raise RuntimeError('Recorded gripper contact feedback was lost while holding')
        if held and self.holding_position is None:
            raise RuntimeError('A held-object decision requires recorded gripper contact')
        if skill == 'close_gripper' and (self.closed or not open_gripper_complete(state)):
            raise RuntimeError('Closing candidate is not ready')
        if skill == 'release_gripper' and not self.returned:
            raise RuntimeError('Release candidate requires the return command')
        if skill == 'retreat_tool' and not self.released:
            raise RuntimeError('Retreat candidate requires the release command')
        meta = observation['camera']
        ids = [a['id'] for a in actions]
        candidate = {'validated': True, 'preconditions_met': True,
            'evidence': str(self.root/'scope.json'), 'precomputed_actions': ids,
            'validation_basis': 'declared finite endpoint/action structure only',
            'geometry_or_collision_independently_validated': False}
        snapshot = {'schema': CONTRACT_VERSION, 'source': 'live_readonly', 'robot_id': 'J1',
            'robot_ip': '192.168.58.2', 'captured_at_unix': meta['source_timestamp_unix'],
            'robot': {'stationary': True, 'fault_free': True, 'enabled': True,
                      'gripper_position': state['gripper_position'],
                      'gripper_motiondone': state['gripper_motiondone'],
                      'gripper_active': state['gripper_active'],
                      'gripper_feedback_valid': gripper_feedback_valid(state),
                      'received_monotonic': observation['robot_time_monotonic']},
            'camera': {'serial': meta['source_camera_serial'], 'available': True,
                       'expected_device_and_fresh': True, 'capture_time_unix': meta['source_timestamp_unix'],
                       'capture_time_monotonic': observation['camera_time']},
            'candidates': {skill: candidate},
            'facts': {'task': self.scope['task_instruction'], 'current_scope': {'contract':self.scope['contract'],
                      'scope_sha256':self.scope_sha,'task_spec':self.scope['task_spec'],
                      'move_speed':self.scope['move_speed'],'only_prealigned_instance':False},
                      'requested_phase': phase, 'cycle': self.cycle, 'precomputed_action_ids': ids,
                      'completed_stages': self.completed_stages.copy(),
                      'current_cycle_completed_stages': self.completed_stages[
                          (self.cycle-1)*len(task_stages(self.scope.get('retreat_mode', 'full'))):].copy(),
                      'completed_cycles': self.completed_cycles,
                      'current_tcp_mm_deg': deepcopy(state['tcp']),
                      'scene_observation': deepcopy(observation),
                      'contact_reference': {'source': 'scope.json.contact_tcp',
                                            'basis': 'planned grasp and return TCP reference',
                                            'tcp_mm_deg': deepcopy(self.scope['contact_tcp'])},
                      'gripper_contact_position': self.holding_position,
                      'return_completed': self.returned, 'release_completed': self.released,
                      'object_support_visually_verified': None,
                      'peak_tcp_height_above_contact_reference_mm': self.peak_height,
                      'requested_hold_seconds': self.scope['task_spec']['hold_seconds'],
                      'actual_hold_duration_s': None,
                      'manual_driven_action_plan': True, 'actual_target_commands_recorded': True,
                      'stable_gripper_status2': state['gripper_motiondone'] == 2,
                      'geometry_or_collision_independently_validated': False,
                      'no_automatic_dispatch_from_decision': True},
            'retrieved_examples': retrieve(['cup', 'grasp', 'lift', 'gripper_feedback']
                                           if object_description == DEFAULT_OBJECT else [object_description])}
        return_target = [a['tcp_mm_deg'] for a in self.scope['actions']
                         if a['cycle'] == self.cycle and a['phase'] == 'return' and a['kind'] == 'arm'][-1]
        snapshot['facts']['return_target_tcp_mm_deg'] = deepcopy(return_target)
        if self.return_endpoint is not None:
            actual = np.asarray(self.return_endpoint['tcp_mm_deg'], float)
            residual = actual[:3]-np.asarray(return_target[:3], float)
            snapshot['facts']['actual_return_endpoint'] = {
                **deepcopy(self.return_endpoint), 'position_residual_mm': residual.tolist(),
                'position_residual_norm_mm': float(np.linalg.norm(residual)),
                'orientation_residual_deg': angle_error(actual[3:], return_target[3:]).tolist()}
        hold_reference = next((row for row in self.hold_records if row['repetition'] == self.cycle), None)
        if hold_reference is not None:
            hold_record = json.loads((self.root/hold_reference['path']).read_text())
            snapshot['facts']['hold_record'] = {'path': hold_reference['path'], **hold_record}
            if hold_record['return_command_issued']:
                snapshot['facts']['actual_hold_duration_s'] = hold_record['actual_duration_s']
        elif self._pending_hold is not None:
            snapshot['facts']['hold_in_progress'] = {
                **deepcopy(self._pending_hold),
                'elapsed_duration_s': time.monotonic()-self._pending_hold['start_monotonic'],
                'return_dispatch_waits_for_requested_hold': phase == 'return'}
        plan_path = self.root/'llm-route-plan.json'
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text())
            if plan.get('scope_sha256') == self.scope_sha and plan.get('source') in PLAN_SOURCES:
                snapshot['facts']['existing_llm_route_plan'] = {
                    'reason':plan.get('reason'), 'source':plan['source'],
                    'requested_phase_choice':plan.get('phase_choices',{}).get(phase),
                    'user_onsite_supervision':True, 'independent_collision_certificate_claimed':False}
        provider = self.scope.get('phase_decision', {}).get('provider', 'emrank')
        config = provider_config(provider)
        request = make_request(snapshot, provider=provider)
        question = (request['questions'][0] if provider == 'openai'
                    else request['questions']['next_skill'])
        if object_description != DEFAULT_OBJECT:
            question['instructions'] = question['instructions'].replace(
                'grasping the upright cup', 'the current task described in the recorded facts')
            if provider == 'openai':
                question['choices'] = [{**choice, 'description': choice['description'].replace('cup', 'object')}
                                       for choice in question['choices']]
            else:
                question['criteria'] = {name: description.replace('cup', 'object')
                                        for name, description in question['criteria'].items()}
        if 'existing_llm_route_plan' in snapshot['facts']:
            question['instructions'] += (
                ' The recorded LLM workflow and scene input supply this task route under user onsite supervision. '
                'Do not request the same scene review again solely because no independent collision certificate '
                'is claimed. Use observe_scene/request_astra when current feedback is missing or conflicts with the plan.')
        if observe_request is not None and provider == 'openai':
            image = base64.b64encode((self.root/observation['image_file']).read_bytes()).decode('ascii')
            request['input'][0]['content'].append(
                {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,'+image})
            question['instructions'] += (
                ' Inspect the supplied new recorded image for visual evidence. '
                'TCP endpoint completion alone does not establish visible object support.')
        path = self.root/filename
        result = {'scope_sha256': self.scope_sha, 'state': snapshot, 'request': request,
                  'provider': provider, 'endpoint': config['endpoint'], 'requested_model': config['model'],
                  'physical_execution': False, 'independent_astra_api_calls': 0}
        with path.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, allow_nan=False)
        try:
            key_file = (Path.home()/'.config/openai/robot-decisions-api-key' if provider == 'openai'
                        else Path.home()/'.config/emrank/api-key')
            result.update(request_started_monotonic=time.monotonic(), request_started_unix=time.time())
            save(path, result)
            response, latency = call_jev(request, key_file, provider=provider)
            result.update(response=response, latency_ms=latency,
                          response_received_monotonic=time.monotonic(), response_received_unix=time.time())
            answer = validate_answer(request, response, provider=provider)
        except DecisionError as exc:
            result.update(decision_error=str(exc), route='needs_llm', reviewed_by_current_chat=False)
            save(path, result)
            return {'decision_id': filename, 'decision_error': str(exc), 'route': 'needs_llm'}
        result.update(response=response, answer=answer, latency_ms=latency,
                      reviewed_by_current_chat=False,
                      selected_by='openai_decisions' if provider == 'openai' else 'jev')
        save(path, result)
        return {'decision_id': filename, 'answer': answer, 'latency_ms': latency}

    def fetch_phase_decision(self, phase, cycle):
        name = f'jev-c{cycle:02d}-{phase}.json'
        decision = self.decide(phase, name)
        record = json.loads((self.root/name).read_text())
        if record.get('answer', {}).get('choice') == 'observe_scene':
            record.update(automatic_continuation=False, route='observe_scene',
                          observation_record=f'observe-c{cycle:02d}-{phase}.json',
                          followup_decision_id=f'jev-c{cycle:02d}-{phase}-after-observe.json')
            save(self.root/name, record)
            try:
                self.capture.phase('observe_scene', teacher={
                    'policy': self.policy_name, 'decision_id': name,
                    'cycle': cycle, 'requested_phase': phase})
                observe_request = {'requested_by_decision': name,
                    'requested_monotonic': time.monotonic(), 'requested_unix': time.time()}
                name = record['followup_decision_id']
                decision = self.decide(phase, name, observe_request=observe_request)
            except BaseException as exc:
                record.update(route='needs_llm', observation_error=f'{type(exc).__name__}: {exc}')
                save(self.root/f'jev-c{cycle:02d}-{phase}.json', record)
                raise
            record = json.loads((self.root/name).read_text())
        if record.get('answer', {}).get('choice') != PHASE_SKILLS[phase]:
            record.update(automatic_continuation=False, route='needs_llm')
            save(self.root/name, record)
            selector = 'Phase decision' if record.get('provider') == 'openai' else 'Jev'
            reason = (' after consecutive observe_scene' if record.get('answer', {}).get('choice') == 'observe_scene'
                      else '')
            raise RuntimeError(selector+' requests LLM intervention'+reason+': '+json.dumps(decision, ensure_ascii=False))
        record['selected_by'] = ('openai_decisions' if record.get('provider') == 'openai' else 'jev')
        record['continuation_rule'] = 'typed matching phase choice; no additional probability threshold'
        record['automatic_continuation'] = True
        save(self.root/name, record)
        return name

    def select_phase(self, phase, cycle, *, prefetched=None):
        # Jev selects the requested skill; its historical confidence threshold
        # is diagnostic only. A return decision may already be ready in hold.
        self.phase, self.cycle = phase, cycle
        name = prefetched or self.fetch_phase_decision(phase, cycle)
        if name not in (f'jev-c{cycle:02d}-{phase}.json', f'jev-c{cycle:02d}-{phase}-after-observe.json'):
            raise RuntimeError('Prefetched decision does not match this cycle and phase')
        self.decision_id = name
        self.capture.phase(phase, teacher={'policy': self.policy_name, 'decision_id': name, 'cycle': cycle})

    def dispatch(self, action):
        self.decision_observation = self.fresh_recorded_observation()
        state = self.state()
        if action['kind'] == 'arm':
            pose = finite_vector(action['tcp_mm_deg'], 6, 'target pose')
            q = rpc_vector(self.rpc.call('GetInverseKinRef', 0, pose.tolist(), state['joints']), 6, 'endpoint IK')
            fk = rpc_vector(self.rpc.call('GetForwardKin', q.tolist()), 6, 'endpoint FK')
            if np.linalg.norm(fk[:3]-pose[:3]) > .2 or np.max(abs(angle_error(fk[3:], pose[3:]))) > .05:
                raise RuntimeError('Cartesian/endpoint joint representation mismatch')
            target = [*q, state['gripper_position']]
        else:
            target = [*state['joints'], action['gripper_percent']]
        self.pending_wire = None
        try:
            result = self.controller.execute(target, kind=action['kind'])
            if result['status'] == 'hold_no_command':
                return self.state()
            record = self.pending_wire
            final = result['final_state']
            completed = time.monotonic()
            record.update(status='completed', final_state=final, outcome={'return_code': result['return_code']},
                          completed_at_unix=time.time(), completed_monotonic=completed,
                          samples=[record['initial_state'], final], result=result)
            durable_json(self.root/f"command-{record['command']['sequence']:04d}.json", record)
            self._last_completed_monotonic = completed
            return final
        except BaseException as exc:
            if self.pending_wire and self.pending_wire.get('command'):
                self.pending_wire.update(status='failed_or_uncertain', error=str(exc))
                durable_json(self.root/f"command-{self.pending_wire['command']['sequence']:04d}.json", self.pending_wire)
            raise

    def hold(self):
        seconds = self.scope['task_spec']['hold_seconds']
        self.capture.phase('hold', teacher={'policy': self.policy_name, 'decision_id': self.decision_id,
                                            'cycle': self.cycle, 'hold_seconds': seconds})
        started = time.monotonic()
        start_unix = time.time()
        if self._last_completed_monotonic is None:
            raise RuntimeError('Cannot time a hold without a completed lift command')
        self._pending_hold = {'cycle': self.cycle, 'requested_seconds': seconds,
            'start_monotonic': started, 'start_unix': start_unix,
            'lift_completed_monotonic': self._last_completed_monotonic,
            'object_lift_visually_verified': False,
            'basis': 'actual time until return wire issue; user reviews object motion'}
        if self.scope['task_spec'].get('object_description', DEFAULT_OBJECT) == DEFAULT_OBJECT:
            self._pending_hold['cup_lift_visually_verified'] = False
        self._prefetch_result = None
        self._prefetch_ready.clear()
        cycle = self.cycle
        def prefetch():
            try:
                self._prefetch_result = {'name': self.fetch_phase_decision('return', cycle)}
            except BaseException as exc:
                self._prefetch_result = {'error': f'{type(exc).__name__}: {exc}'}
            finally:
                self._prefetch_ready.set()
        self._prefetch_thread = threading.Thread(target=prefetch, daemon=True)
        self._prefetch_thread.start()
        try:
            # A matching Jev answer and the actual hold duration are both
            # required. A stop or failed decision interrupts the hold.
            while True:
                self.state(held=True)
                remaining = started + seconds - time.monotonic()
                if self._prefetch_ready.is_set():
                    result = self._prefetch_result
                    if not result or result.get('error'):
                        raise RuntimeError('Jev return decision failed: '+str((result or {}).get('error')))
                    if remaining <= 0:
                        return result['name']
                time.sleep(min(.02, remaining) if remaining > 0 else .02)
        except BaseException as exc:
            self.interrupt_pending_hold(exc)
            raise

    def run(self):
        for cycle in range(1, self.scope['task_spec']['repetitions'] + 1):
            self.closed = self.returned = self.released = False
            self.holding_position = None
            self.return_endpoint = None
            prefetched_return = None
            for phase in task_stages(self.scope.get('retreat_mode', 'full')):
                self.select_phase(phase, cycle, prefetched=prefetched_return if phase == 'return' else None)
                state = self.state()
                for action in self.scope['actions']:
                    if action['cycle'] == cycle and action['phase'] == phase:
                        state = self.dispatch(action)
                if phase == 'close':
                    self.closed = True
                    if not gripper_feedback_valid(state) or state['gripper_motiondone'] != 2:
                        raise RuntimeError('No actual gripper contact after closure')
                    self.holding_position = state['gripper_position']
                elif phase in ('lift_probe', 'lift_full'):
                    self.peak_height = max(self.peak_height, state['tcp'][2]-self.origin[2])
                    if phase == 'lift_full':
                        has_lift = any(item.get('executed') is True and
                            item.get('teacher', {}).get('cycle') == cycle and
                            item.get('teacher', {}).get('phase') == 'lift_full' for item in self.journal)
                        if not has_lift:
                            raise RuntimeError('No completed lift_full command for this cycle')
                elif phase == 'return':
                    self.returned = True
                    self.return_endpoint = {'tcp_mm_deg': deepcopy(state['tcp']),
                        'source': 'return phase final_state', 'completed_monotonic': self._last_completed_monotonic}
                    if self.pending_wire is not None and self.pending_wire.get('status') == 'completed':
                        self.return_endpoint['command_record'] = f"command-{self.pending_wire['command']['sequence']:04d}.json"
                elif phase == 'release':
                    if not open_gripper_complete(state):
                        raise RuntimeError('Release not completed')
                    self.released, self.holding_position = True, None
                self.completed_stages.append(phase)
                save(self.root/'completed-stages.json', self.completed_stages)
                print(json.dumps({'cycle': cycle, 'phase': phase, 'status': 'completed', 'success': False}), flush=True)
                if phase == 'lift_full':
                    prefetched_return = self.hold()
            self.completed_cycles = cycle
        # The exporter requires a recorded paired observation after the final
        # command completion, not merely the pre-dispatch frame.
        self.fresh_recorded_observation()
        holds = [f'hold-{cycle:02d}.json' for cycle in range(1, self.completed_cycles + 1)]
        return self.finish('pending_review', ['scene.jpg', *holds, 'target-command-journal.json'])

    def final_summary_fields(self):
        return {'completed_stages': self.completed_stages, 'split_role': self.scope['split_role'],
                'start_offset_mm': float(np.linalg.norm(self.scope['initial_tcp'][:3]-self.origin[:3])),
                'whole_task': True, 'task_instruction': self.scope['task_instruction'],
                'task_spec': self.scope['task_spec'], 'held_seconds': self.held_seconds,
                'hold_durations_seconds': self.hold_durations, 'hold_records': self.hold_records,
                'completed_cycles': self.completed_cycles,
                'retreat_mode': self.scope.get('retreat_mode', 'full'),
                'task_success_verified': False, 'success_requires_user_review': True,
                'action_timebase': 'actual command events; camera 6Hz is observation sampling only',
                'per_action_user_approvals': 0, 'per_phase_visual_chat_checkpoints': 0}

    def finish(self, outcome='interrupted', evidence=None):
        if outcome == 'success':
            raise ValueError('Only the user labels the full episode successful')
        self._watchdog_stop.set()
        if getattr(self, '_watchdog', None):
            self._watchdog.join(2.)
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            # The HTTP client's own response handling ends this worker; an
            # extra finalization timer must not discard a completed episode.
            self._prefetch_thread.join()
        if self._pending_hold is not None:
            self.interrupt_pending_hold(self.failed or 'episode ended before return')
            self.failed = self.failed or 'Episode ended before the pending return command'
            outcome = 'interrupted'
        if outcome == 'pending_review':
            repetitions = self.scope['task_spec']['repetitions']
            executed = [item for item in self.journal if item.get('executed') is True]
            try:
                from lerobot_robot_fairino.curate_teacher import validated_commands
                commands = validated_commands(self.root)
                records_complete = (set(commands) == {item['sequence'] for item in executed}
                                    and all(commands[item['sequence']]['command'] == item for item in executed))
            except (OSError, ValueError, KeyError, TypeError):
                records_complete = False
                commands = {}
            holds_complete = False
            if records_complete and self.hold_records == [
                    {'repetition': cycle, 'path': f'hold-{cycle:02d}.json'}
                    for cycle in range(1, repetitions + 1)]:
                try:
                    holds_complete = True
                    for cycle in range(1, repetitions + 1):
                        row = json.loads((self.root/f'hold-{cycle:02d}.json').read_text())
                        lift = [item for item in executed if item['teacher'].get('cycle') == cycle
                                and item['teacher'].get('phase') == 'lift_full']
                        returning = [item for item in executed if item['teacher'].get('cycle') == cycle
                                     and item['teacher'].get('phase') == 'return']
                        if not lift or not returning:
                            holds_complete = False
                            break
                        lift_completed = commands[lift[-1]['sequence']]['completed_monotonic']
                        return_issued = returning[0]['issued_monotonic']
                        start, end, duration = row['start_monotonic'], row['end_monotonic'], row['actual_duration_s']
                        if (row.get('repetition') != cycle or row.get('return_command_issued') is not True
                                or row.get('requested_seconds') != self.scope['task_spec']['hold_seconds']
                                or row.get('lift_completed_monotonic') != lift_completed
                                or not lift_completed <= start <= end or end != return_issued
                                or not math.isclose(duration, end-start, abs_tol=1e-6)
                                or not math.isclose(row['actual_hover_from_lift_completion_s'],
                                                    end-lift_completed, abs_tol=1e-6)
                                or not math.isclose(duration, self.hold_durations[cycle-1], abs_tol=1e-6)):
                            holds_complete = False
                            break
                except (OSError, ValueError, KeyError, TypeError, IndexError):
                    holds_complete = False
            sent = {(item.get('teacher', {}).get('cycle'), item.get('teacher', {}).get('phase'))
                    for item in executed}
            required = {(cycle, phase) for cycle in range(1, repetitions + 1)
                        for phase in ('close', 'lift_full', 'return', 'release')}
            complete = (not self.failed and not self.interrupted.is_set() and not (self.root/'STOP').exists()
                        and self.completed_stages == self.scope['phase_order']
                        and self.completed_cycles == repetitions and self.released and self.returned
                        and len(self.hold_durations) == repetitions
                        and self._pending_hold is None and len(self.hold_records) == repetitions
                        and records_complete and holds_complete
                        and all(value >= self.scope['task_spec']['hold_seconds'] for value in self.hold_durations)
                        and required.issubset(sent))
            if not complete:
                self.failed = self.failed or 'Whole-task cycle or command record is incomplete'
                outcome = 'interrupted'
        return super().finish(outcome, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--directory', type=Path, required=True)
    prep.add_argument('--observation', type=Path, required=True)
    prep.add_argument('--cup-uv', type=float, nargs=2, required=True)
    prep.add_argument('--grasp-reference', type=Path)
    prep.add_argument('--speed', type=float, default=10.)
    prep.add_argument('--split', choices=('training_complete_trial', 'held_out_complete_trial'), default='training_complete_trial')
    prep.add_argument('--hold-seconds', type=float, default=5.)
    prep.add_argument('--repetitions', type=int, choices=(1, 2), default=1)
    prep.add_argument('--retreat', choices=('short', 'full'), default='short')
    run = sub.add_parser('run')
    run.add_argument('--directory', type=Path, required=True)
    run.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.operation == 'prepare':
        print(json.dumps(prepare(args.directory, args.observation, args.cup_uv,
            grasp_reference=args.grasp_reference, speed=args.speed, split=args.split,
            hold_seconds=args.hold_seconds, repetitions=args.repetitions,
            retreat_mode=args.retreat), ensure_ascii=False))
        return
    session = WholeTaskSession(args.directory)
    if not args.execute:
        print(json.dumps({'status': 'validated_no_connection_no_motion', 'task': session.scope['task_instruction'],
                          'task_spec': session.scope['task_spec'],
                          'planned_actions': len(session.scope['actions'])}, ensure_ascii=False))
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: session.interrupted.set())
    try:
        session.start()
        print(json.dumps(session.run(), ensure_ascii=False), flush=True)
    except BaseException as exc:
        session.failed = str(exc)
        if session.controller is not None:
            session.controller.stop(str(exc))
        if session.writer is None or not session.writer._finished:
            session.finish('interrupted')
        elif session.storage is not None:
            session.storage.frame_spool.abort()
        raise
    finally:
        session._watchdog_stop.set()
        if session.robot.is_connected:
            session.robot.disconnect()


if __name__ == '__main__':
    main()
