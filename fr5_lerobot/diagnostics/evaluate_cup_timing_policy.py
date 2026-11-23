#!/usr/bin/env python3
"""Prepare and compare paired cup-timing policy observations offline.

The input set contains two recorded cycles of held-out source 7 and the nine
tick240 trial requests.  It can be prepared before the timing module exists.
Run ``timing`` once the matching source snapshot is available, then use the
same cases and seeds for both checkpoint predictions.  Comparison files also
have the schema consumed by check_cup_tracking_prefixes.py.

All files are read from recordings or checkpoints.  No command in this program
connects to a robot, camera, or controller.
"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys


ROOT = Path('/home/data4/hts/demo1/fr5-learning')
DATASET = ROOT / 'datasets/fr5-joint-stream-20261102-v2-compacted'
PREVIOUS = ROOT / 'audits/pi05-cup-tracking-replay-20261113-v1/inputs-v2.json'
TRACKING_SOURCE = ROOT / 'src/fr5-pi05-cup-tracking-20261113-v1'
TIMING_SOURCE = ROOT / 'src/fr5-pi05-cup-timing-20261113-v1'
CHECKPOINTS = {
    'parent': ROOT / 'checkpoints/pi05-fr5-stream-cup-tracking-n10-20261113-v1/step-000160',
    'candidate': ROOT / 'checkpoints/pi05-fr5-stream-cup-timing-n10-20261113-v1/step-000160',
}
PROFILES = {'parent': 'state-history-physical-cup-tracking-v1',
            'candidate': 'state-history-physical-cup-timing-v1'}
ACTION_CLOCK_PROFILE = 'state-history-physical-cup-action-clock-v1'
ALIGNED_PROFILE = 'state-history-physical-cup-aligned-v1'
SETDOWN_PROFILE = 'state-history-physical-cup-setdown-v1'
SETDOWN_CLEAN_PROFILE = 'state-history-physical-cup-setdown-clean-v1'
SETDOWN_FIXED_PROFILE = 'state-history-physical-cup-setdown-fixed-v1'
SETDOWN_PROFILES = (SETDOWN_PROFILE, SETDOWN_CLEAN_PROFILE, SETDOWN_FIXED_PROFILE)
INPUT_SCHEMA = 'fr5-cup-timing-replay-inputs-v1'
PREDICTION_SCHEMA = 'fr5-cup-timing-replay-predictions-v1'
COMPARISON_SCHEMA = 'fr5-cup-tracking-replay-comparison-v1'
OFFSETS = (0.5, 1., 2., 4., 6., 8.)
CHUNK, FPS = 50, 6
CLOSE_THRESHOLD, OPEN_THRESHOLD = .88, .90
sys.dont_write_bytecode = True


def allowed_profiles(model):
    return PROFILES[model], ACTION_CLOCK_PROFILE, ALIGNED_PROFILE, *SETDOWN_PROFILES


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def near(actual, expected, label, tolerance=1e-5):
    if len(actual) != len(expected) or any(abs(float(a) - float(b)) > tolerance
                                           for a, b in zip(actual, expected)):
        raise ValueError(f'{label} differs from the recorded data')


def state_from_capture(capture, cup_base):
    robot = capture['robot']
    if robot['tool'] != 0 or robot['user'] != 0:
        raise ValueError('Expected the recorded tool0/user0 TCP')
    valid = bool(capture['gripper_feedback_valid'])
    tcp = robot['tcp']
    return ([(tcp[i] - cup_base[i]) / 1000. for i in range(3)]
            + [math.radians(x) for x in tcp[3:]]
            + [robot['gripper_position'] / 100. if valid else 0., float(valid)])


def history_from_captures(captures, current, first, cup_base):
    eligible = [capture for capture in captures
                if capture['camera']['source_timestamp_monotonic'] >= first]
    times = [capture['camera']['source_timestamp_monotonic'] for capture in eligible]
    result = []
    for offset in OFFSETS:
        i = bisect_right(times, current - offset) - 1
        result.append(None if i < 0 else {'age_s': current - times[i],
                                         'state': state_from_capture(eligible[i], cup_base)})
    return result


def replay(root, captures, initial_cup, calibration, selected, start_frame, with_timing, *, aligned=False, setdown=False):
    import cv2
    import numpy as np
    from fr5_jev.cup_tracking import CupTracker
    if with_timing:
        from fr5_jev.cup_timing import CupTiming
        timer = CupTiming(initial_cup, aligned=aligned, setdown=setdown)
    else:
        timer = None
    cv2.setNumThreads(1)
    tracker = CupTracker(initial_cup, calibration['base_from_color_camera_mm'])
    result = {}
    previous_time = -math.inf
    for capture in captures:
        frame = capture['frame_index']
        if frame < start_frame:
            continue
        if frame > max(selected):
            break
        stamp = capture['camera']['source_timestamp_monotonic']
        if stamp <= previous_time:
            raise ValueError('Captured frames are not in camera time order')
        previous_time = stamp
        image = cv2.imread(str(root / 'continuous' / capture['rgb_file']), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Missing RGB frame {frame} in {root}')
        with np.load(root / 'continuous' / capture['depth_file'], allow_pickle=False) as depth:
            tracking = tracker.update(image, depth['depth_mm'], capture['camera'], capture['robot'],
                                      gripper_valid=bool(capture['gripper_feedback_valid']))
        evidence = timer.update(capture['camera'], capture['robot'], tracking) if timer else None
        if evidence is not None and not isinstance(evidence.get('state'), dict):
            raise ValueError('CupTiming.update must return evidence with a state dict')
        if frame in selected:
            result[frame] = (tracking, evidence)
        if (frame - start_frame) % 100 == 0:
            print(f'{root.parent.name}: replayed frame {frame}', file=sys.stderr, flush=True)
    if set(result) != set(selected):
        raise ValueError('A selected frame is absent from the recorded capture')
    return result


def source_data(dataset):
    import pyarrow.parquet as pq
    source = read_json(dataset / 'meta/stream_provenance.json')['sources'][7]
    if source['source_index'] != 7 or source['split'] != 'validation':
        raise ValueError('Source 7 must be the held-out source')
    root = Path(source['trial'])
    geometry = read_json(root / 'geometry.json')
    opening = read_json(root / 'source-observation.json')
    camera, calibration = opening['camera'], opening['calibration']
    cup = {'uv': geometry['cup_uv_selected_by_llm'],
           'camera_serial': camera['camera']['serial'],
           'observed_monotonic': camera['capture']['received_monotonic'],
           'base_xyz_mm': geometry['cup_model']['base_center_mm'],
           'rim_xyz_mm': geometry['cup_model']['rim_center_mm'],
           'table_normal': geometry['table']['normal'],
           'table_offset_mm': geometry['table']['offset_mm'],
           'calibration_source': calibration['source']}
    if calibration['camera_serial'] != cup['camera_serial']:
        raise ValueError('Source camera and calibration do not match')
    captures = read_jsonl(root / 'continuous/capture.jsonl')
    frames = {capture['frame_index']: capture for capture in captures}
    mappings = [row for row in read_jsonl(dataset / 'meta/stream_source_frames.jsonl')
                if row['source_index'] == 7]
    table = pq.read_table(dataset / 'data/chunk-000/file-000.parquet',
                          columns=['index', 'episode_index', 'frame_index', 'observation.state', 'action'])
    rows = {int(row['index']): row for row in table.to_pylist()
            if int(row['episode_index']) == mappings[0]['episode_index']}
    if len(rows) != len(mappings):
        raise ValueError('Held-out compact rows and mappings differ')
    for i, mapping in enumerate(mappings):
        row = rows[mapping['frame']]
        if (mapping['step'] != i or int(row['frame_index']) != i
                or mapping['split'] != 'validation'):
            raise ValueError('Held-out row order or split differs')
        near(row['action'], mapping['derived_action_radians_fraction'], 'Compact action', 1e-6)
    return source, root, cup, calibration, captures, frames, mappings, rows


def select_cases(source, root, captures, frames, mappings):
    commands = read_json(root / 'target-command-journal.json')
    result = []
    for cycle in (1, 2):
        phases = {c['teacher']['phase']: c for c in commands if c.get('executed')
                  and c['teacher']['cycle'] == cycle}
        hold = next(h for h in source['hold_records'] if h['repetition'] == cycle)
        close, release = phases['close'], phases['release']
        lift, ret = phases['lift_probe'], phases['return']
        contact = next(c for c in captures
                       if close['issued_monotonic'] <= c['camera']['source_timestamp_monotonic']
                       < lift['issued_monotonic'] and c['gripper_feedback_valid']
                       and c['robot']['gripper_motiondone'] == 2)
        contact_time = contact['camera']['source_timestamp_monotonic']
        bounds = {'contact_confirmed_monotonic': contact_time,
                  'contact_confirmed_source_frame': contact['frame_index'],
                  'hold_start_monotonic': hold['start_monotonic'],
                  'hold_end_monotonic': hold['end_monotonic'],
                  'hold_record_actual_seconds': hold['actual_duration_s'],
                  'return_issued_monotonic': ret['issued_monotonic'],
                  'close_issued_monotonic': close['issued_monotonic'],
                  'release_issued_monotonic': release['issued_monotonic']}

        def time_of(row):
            return row['observation']['camera_time_monotonic']

        def between(start, end, fraction, predicate=lambda row: True):
            candidates = [row for row in mappings if start <= time_of(row) < end and predicate(row)]
            if not candidates:
                raise ValueError(f'No recorded observations for cycle {cycle} interval')
            desired = start + fraction * (end - start)
            return min(candidates, key=lambda row: (abs(time_of(row) - desired), row['step']))

        def action_transition(command):
            matches = [row for row in mappings
                       if row['gripper_target']['sequence'] == command['sequence']]
            if not matches:
                raise ValueError('No compact gripper target for recorded command')
            return matches[0]['step']

        ci, ri = action_transition(close), action_transition(release)
        close_reference = frames[mappings[ci]['observation']['frame_index']]['robot']['tcp'][:3]
        bounds.update(close_compact_step=ci, release_compact_step=ri,
                      close_reference_tcp_xyz_mm=close_reference)
        start_near = phases['approach_near']['issued_monotonic']
        start_contact = phases['approach_contact']['issued_monotonic']
        end_close = lift['issued_monotonic']
        release_end = phases['retreat_probe']['issued_monotonic']
        retreat_end = phases['retreat_clear']['issued_monotonic']
        contact_rows = [r for r in mappings if contact_time <= time_of(r) < end_close
                        and frames[r['observation']['frame_index']]['robot']['gripper_motiondone'] == 2
                        and r['observation']['gripper_valid']]
        if not contact_rows:
            raise ValueError('No settled contact before lift')
        selections = [
            ('near_early', between(start_near, start_contact, .33)),
            ('near_late', between(start_near, start_contact, .75)),
            ('contact_approach', between(start_contact, close['issued_monotonic'], .5)),
            ('close_before', mappings[ci - 1]), ('close_at', mappings[ci]),
            ('close_after', between(close['issued_monotonic'], end_close, .05)),
            ('contact_on_table', contact_rows[-1]),
            ('hover_early', between(hold['start_monotonic'], hold['end_monotonic'], .1)),
            ('hover_middle', between(hold['start_monotonic'], hold['end_monotonic'], .5)),
            ('hover_late', between(hold['start_monotonic'], hold['end_monotonic'], .9)),
            ('return_early', between(ret['issued_monotonic'], release['issued_monotonic'], .25)),
            ('return_middle', between(ret['issued_monotonic'], release['issued_monotonic'], .5)),
            ('return_late', between(ret['issued_monotonic'], release['issued_monotonic'], .75)),
            ('release_before', mappings[ri - 1]), ('release_at', mappings[ri]),
            ('release_after', between(release['issued_monotonic'], release_end, .05)),
            ('retreat', between(release_end, retreat_end, .5)),
        ]
        for kind, mapping in selections:
            result.append({'cycle': cycle, 'kind': kind, 'mapping': mapping, 'event_times': bounds})
    if len({item['mapping']['frame'] for item in result}) != len(result):
        raise ValueError('Event observations must have distinct compact rows')
    return result


def source_cases(dataset, source_root):
    sys.path.insert(0, str(source_root / 'fr5_lerobot/training'))
    from cup_relative_stream_data import _action
    source, root, cup, calibration, captures, frames, mappings, rows = source_data(dataset)
    selected = select_cases(source, root, captures, frames, mappings)
    first_time = mappings[0]['observation']['camera_time_monotonic']
    first_frame = mappings[0]['observation']['frame_index']
    indices = {item['mapping']['observation']['frame_index'] for item in selected}
    tracked = replay(root, captures, cup, calibration, indices, first_frame, False)
    cases = []
    for item in selected:
        mapping, cycle = item['mapping'], item['cycle']
        capture = frames[mapping['observation']['frame_index']]
        stamp = capture['camera']['source_timestamp_monotonic']
        near([stamp], [mapping['observation']['camera_time_monotonic']], 'Capture time')
        state = state_from_capture(capture, cup['base_xyz_mm'])
        joints = [math.radians(x) for x in capture['robot']['joints']] + state[6:]
        near(joints, rows[mapping['frame']]['observation.state'], 'Compact observation')
        case_id = f"source7-cycle{cycle}-{item['kind']}-{mapping['frame']}"
        request = {'schema': 'fr5-stream-request-v1', 'id': case_id,
                   'task': source['task_instruction'],
                   'rgb_jpeg_base64': base64.b64encode(
                       (root / 'continuous' / capture['rgb_file']).read_bytes()).decode('ascii'),
                   'source_time': stamp, 'source_clock': 'host_monotonic',
                   'policy_contract': 'fr5-joint-position-stream-v1',
                   'state': joints, 'cup_relative_state': state,
                   'observation_history': history_from_captures(captures, stamp, first_time, cup['base_xyz_mm']),
                   'initial_cup_uv': cup['uv'], 'cup_base_xyz_mm': cup['base_xyz_mm']}
        targets = []
        for target in mappings[mapping['step']:mapping['step'] + CHUNK]:
            values = _action(rows[target['frame']], target, frames, cup['base_xyz_mm'])
            targets.append({'compact_row': target['frame'], 'compact_step': target['step'],
                            'source_v1_frame': target['source_v1_frame'],
                            'target_time_monotonic': target['target_time_monotonic'],
                            'cut_after': target['cut_after'], 'cut_id': target['cut_id'],
                            'gripper_command_sequence': target['gripper_target']['sequence'],
                            'joint_action': rows[target['frame']]['action'],
                            'cup_relative_action': values})
        tracking, _ = tracked[capture['frame_index']]
        cases.append({'case_id': case_id, 'group': 'heldout_source_7', 'split': 'validation',
                      'source_index': 7, 'cycle': cycle, 'kind': item['kind'],
                      'row_index': mapping['frame'], 'source_frame_index': capture['frame_index'],
                      'tracker_start_frame_index': first_frame, 'source_root': str(root),
                      'event_times': item['event_times'], 'targets': targets,
                      'distance_to_recorded_close_tcp_mm': math.dist(
                          capture['robot']['tcp'][:3], item['event_times']['close_reference_tcp_xyz_mm']),
                      'height_above_recorded_close_tcp_mm': (
                          capture['robot']['tcp'][2] - item['event_times']['close_reference_tcp_xyz_mm'][2]),
                      'recorded_tcp_mm_deg': capture['robot']['tcp'],
                      'recorded_joints_deg': capture['robot']['joints'],
                      'initial_cup': cup, 'tracking': tracking, 'timing': None, 'request': request})
    return cases, calibration


def trial_cases(previous, calibration):
    prior = read_json(previous)
    cases = [case for case in prior['cases'] if case['group'] == 'failed_trial_evaluation_only']
    if len(cases) != 9 or len({case['case_id'] for case in cases}) != 9:
        raise ValueError('Expected all nine distinct tick240 trial requests')
    trial = Path(prior['failed_trial'])
    captures = read_jsonl(trial / 'continuous/capture.jsonl')
    frames = {capture['frame_index']: capture for capture in captures}
    selected = {case['source_frame_index'] for case in cases}
    cup = cases[0]['initial_cup']
    if calibration['camera_serial'] != cup['camera_serial'] or calibration['source'] != cup['calibration_source']:
        raise ValueError('Trial and held-out source calibration differ')
    tracked = replay(trial, captures, cup, calibration, selected, captures[0]['frame_index'], False)
    for case in cases:
        capture = frames[case['source_frame_index']]
        request = case['request']
        near([request['source_time']], [capture['camera']['source_timestamp_monotonic']], 'Trial time')
        near(request['cup_relative_state'], state_from_capture(capture, cup['base_xyz_mm']), 'Trial state')
        track, _ = tracked[case['source_frame_index']]
        if track != case['tracking']:
            raise ValueError(f"Trial tracking differs from previous replay: {case['case_id']}")
        case.update(source_root=str(trial), timing=None)
    return cases


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.source_root))
    cases, calibration = source_cases(args.dataset, args.source_root)
    cases += trial_cases(args.previous, calibration)
    counts = Counter(case['group'] for case in cases)
    report = {'schema': INPUT_SCHEMA, 'case_count': len(cases),
              'heldout_source_7_cases': counts['heldout_source_7'],
              'failed_trial_cases': counts['failed_trial_evaluation_only'],
              'dataset': str(args.dataset), 'previous_replay': str(args.previous),
              'tracking_source_root': str(args.source_root), 'timing_source_root': None,
              'timing_ready': False, 'profiles': PROFILES,
              'action_origin': 'initial_cup.base_xyz_mm; J1 axes',
              'target_source': 'compact row action and interpolated original TCP feedback at compact target time',
              'target_clock': 'compact 6 Hz steps, with original target times and cut markers',
              'failed_trial_use': 'evaluation only; no ground-truth targets',
              'robot_connections': 0, 'robot_commands': 0, 'cases': cases}
    write_new(args.output, report)
    print(json.dumps({'output': str(args.output), 'case_count': len(cases),
                      'groups': dict(counts), 'kinds': dict(Counter(c['kind'] for c in cases)),
                      'timing_ready': False}, ensure_ascii=False), flush=True)


def add_timing(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = read_json(args.input)
    if report['schema'] != INPUT_SCHEMA or report['timing_ready'] != (args.aligned or args.setdown):
        raise ValueError('Aligned timing requires the existing legacy timing observations')
    sys.path.insert(0, str(args.source_root))
    by_root = defaultdict(list)
    for case in report['cases']:
        by_root[case['source_root']].append(case)
    for name, cases in by_root.items():
        root = Path(name)
        captures = read_jsonl(root / 'continuous/capture.jsonl')
        cup = cases[0]['initial_cup']
        if cases[0]['group'] == 'heldout_source_7':
            calibration = read_json(root / 'source-observation.json')['calibration']
        else:
            heldout_root = Path(next(c['source_root'] for c in report['cases']
                                    if c['group'] == 'heldout_source_7'))
            calibration = read_json(heldout_root / 'source-observation.json')['calibration']
        selected = {case['source_frame_index'] for case in cases}
        starts = {case['tracker_start_frame_index'] for case in cases}
        if len(starts) != 1:
            raise ValueError('Cases from one source have different replay start frames')
        tracked = replay(root, captures, cup, calibration, selected, starts.pop(), True,
                         aligned=args.aligned, setdown=args.setdown)
        for case in cases:
            tracking, timing = tracked[case['source_frame_index']]
            if tracking != case['tracking']:
                raise ValueError(f"Tracking changed for {case['case_id']}")
            case['timing_setdown' if args.setdown else 'timing_aligned' if args.aligned else 'timing'] = timing
    report['timing_source_root'] = str(args.source_root)
    report['timing_ready'] = True
    write_new(args.output, report)
    print(json.dumps({'output': str(args.output), 'timing_ready': True,
                      'cases': len(report['cases'])}, ensure_ascii=False), flush=True)


def add_camera_targets(args):
    inputs = read_json(args.input)
    sys.path.insert(0, str(args.source_root / 'fr5_lerobot/training'))
    from cup_relative_stream_data import attach_camera_aligned_actions

    proof = read_json(args.dataset / 'meta/stream_provenance.json')
    cases = [case for case in inputs['cases'] if case['group'] == 'heldout_source_7']
    rows = [{'_physical_source': case['source_index'], '_camera_time': case['request']['source_time'],
             '_cup_relative_state': case['request']['cup_relative_state']} for case in cases]
    attach_camera_aligned_actions(rows, {0: list(range(len(rows)))}, proof)
    for case, row in zip(cases, rows):
        stamp = case['request']['source_time']
        case['targets_camera_aligned'] = [
            {'compact_row': None, 'target_time_monotonic': stamp + i / FPS, 'cut_after': False,
             'cup_relative_action': point.tolist()}
            for i, point in enumerate(row['_camera_aligned_actions'], 1)]
    write_new(args.output, inputs)
    print(json.dumps({'output': str(args.output), 'camera_aligned_heldout_cases': len(cases)}, ensure_ascii=False))


def model_request(case, model, timing_mode='full', *, profile=None):
    request = dict(case['request'])
    request['task'] = case.get('task_override', request['task'])
    request['observation_context_profile'] = profile or PROFILES[model]
    request['cup_tracking'] = {key: case['tracking'][key] for key in ('rim_minus_jaw_mm', 'native_contact')}
    profile = request['observation_context_profile']
    if profile in (PROFILES['candidate'], ACTION_CLOCK_PROFILE, ALIGNED_PROFILE, *SETDOWN_PROFILES):
        timing_key = ('timing_setdown' if profile in SETDOWN_PROFILES else 'timing_aligned'
                      if profile == ALIGNED_PROFILE or timing_mode == 'aligned-observer' else 'timing')
        evidence = case.get(timing_key)
        if not isinstance(evidence, dict) or not isinstance(evidence.get('state'), dict):
            raise ValueError(f"Missing timing evidence for {case['case_id']}")
        request['cup_timing'] = evidence['state']
        if timing_mode == 'aligned-observer' and profile == ACTION_CLOCK_PROFILE:
            request['cup_timing'] = {k: v for k, v in evidence['state'].items() if k != 'support_gap_mm'}
        if timing_mode == 'unknown':
            request['cup_timing'] = {'phase': 'unknown', 'hover_seconds': None, 'visual': evidence['state']['visual']}
    else:
        request.pop('cup_timing', None)
    return request


def predict(args):
    inputs = read_json(args.input)
    if inputs['schema'] != INPUT_SCHEMA or (args.model == 'candidate' and not inputs['timing_ready']):
        raise ValueError('Expected prepared cases with the candidate timing evidence')
    checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    probes = [(args.task_override, args.output)]
    if args.original_output is not None:
        if args.task_override is None or args.original_output == args.output:
            raise ValueError('A separate original-output path requires a task override')
        probes.append((None, args.original_output))
    for _, output in probes:
        if output.exists():
            raise FileExistsError(output)
    seeds = args.seed or [20261105]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct')
    sys.path.insert(0, str(checkpoint))
    with redirect_stdout(sys.stderr):
        from serve_stream_pi05 import Inference
        model = Inference(checkpoint, gpu=args.gpu)
        model.torch.set_num_threads(4)
    if args.denoising_steps is not None:
        model.policy.config.num_inference_steps = args.denoising_steps
    profile = model.info['observation_context_profile']
    if (profile not in allowed_profiles(args.model)
            or model.info['action_representation'] != 'cup-relative-tcp-tick-delta-v1'):
        raise ValueError('Checkpoint profile or action representation differs')
    if args.timing_mode == 'aligned-observer' and profile != ACTION_CLOCK_PROFILE:
        raise ValueError('The observer-only ablation requires the legacy numeric clock model')
    if args.no_clock_channel_noise or args.noise_scale != 1.:
        if profile != ALIGNED_PROFILE:
            raise ValueError('Noise probes require the aligned observation profile')
        sample_noise = model.policy.model.sample_noise

        def sample_with_probe_noise(shape, device):
            noise = sample_noise(shape, device) * args.noise_scale
            # The first seven noise channels stay identical for a paired seed.
            # The aligned observation contributes 24 values in channels 7:31.
            if args.no_clock_channel_noise:
                noise[..., 7:31] = 0.
            return noise

        model.policy.model.sample_noise = sample_with_probe_noise
    for task_override, output in probes:
        predictions = []
        cases = [case for case in inputs['cases'] if not args.kind or case['kind'] in args.kind]
        if not cases:
            raise ValueError('No replay case matches the selected kinds')
        for i, case in enumerate(cases, 1):
            probe = dict(case, task_override=task_override) if task_override is not None else case
            request = model_request(probe, args.model, args.timing_mode, profile=profile)
            for seed in seeds:
                with redirect_stdout(sys.stderr):
                    model.reset()
                    model.torch.manual_seed(seed)
                    reply = model.predict(request)
                validate_actions(reply['actions'])
                predictions.append({'case_id': case['case_id'], 'seed': seed,
                                    'group': case['group'], 'kind': case['kind'],
                                    'request': request, 'reply': reply})
            print(f'{output.name}: {i}/{len(cases)} cases', file=sys.stderr, flush=True)
        write_new(output, {'schema': PREDICTION_SCHEMA, 'model': args.model, 'seeds': seeds,
                            'input': str(args.input), 'checkpoint': str(checkpoint),
                            'task_override': task_override, 'timing_mode': args.timing_mode,
                            'denoising_steps': model.policy.config.num_inference_steps,
                            'noise_scale': args.noise_scale,
                            'clock_channels_initial_noise': not args.no_clock_channel_noise,
                            'case_kinds': args.kind,
                            'checkpoint_sha256': model.info['checkpoint_sha256'],
                            'observation_context_profile': profile, 'predictions': predictions,
                            'robot_connections': 0, 'robot_commands': 0})
        print(json.dumps({'output': str(output), 'predictions': len(predictions)}), flush=True)


def validate_actions(actions):
    if (len(actions) != CHUNK or any(len(row) != 7 or any(not math.isfinite(float(v)) for v in row)
                                    for row in actions)):
        raise ValueError('Expected a finite 50 x 7 action chunk')


def first_index(values, predicate):
    return next((i for i, value in enumerate(values) if predicate(value)), None)


def tick(index):
    return None if index is None else {'tick': index + 1, 'nominal_seconds': (index + 1) / FPS}


def target_event(index, predictions, targets):
    if index is None:
        return None
    return {**tick(index), 'compact_row': targets[index]['compact_row'],
            'recorded_target_time_monotonic': targets[index]['target_time_monotonic'],
            'predicted_gripper_fraction': predictions[index][6],
            'recorded_gripper_target_fraction': targets[index]['cup_relative_action'][6]}


def movement(actions, anchor):
    points = [anchor[:3]] + [row[:3] for row in actions]
    lengths = [math.dist(a, b) * 1000. for a, b in zip(points, points[1:])]
    return {'net_xyz_mm': [(points[-1][i] - points[0][i]) * 1000. for i in range(3)],
            'net_distance_mm': math.dist(points[0], points[-1]) * 1000.,
            'path_length_mm': sum(lengths),
            'max_axis_excursion_mm': max(abs(p[i] - points[0][i]) * 1000.
                                         for p in points for i in range(3))}


def early_gripper_change(case, pred, labels, *, closing):
    """Quantize as the ordered controller does, using the observed opening as anchor."""
    fraction, valid = case['request']['cup_relative_state'][6:]
    if not valid:
        return None
    previous = round(fraction * 100)
    cup = case['initial_cup']
    pickup = case['event_times']['close_reference_tcp_xyz_mm']
    for i, (action, label) in enumerate(zip(pred, labels)):
        target = round(max(0., min(1., action[6])) * 100)
        if abs(target - previous) <= 1:
            continue
        recorded = label['cup_relative_action'][6]
        early = ((target < previous and previous >= 88 and recorded >= OPEN_THRESHOLD) if closing else
                 (target > previous and previous < 98 and recorded < CLOSE_THRESHOLD))
        if early:
            dz = sum((x * 1000. + base - ref) * n for x, base, ref, n in
                     zip(action[:3], cup['base_xyz_mm'], pickup, cup['table_normal']))
            return {**target_event(i, pred, labels), 'previous_percent': previous,
                    'effective_target_percent': target, 'target_height_above_pickup_mm': dz}
        previous = target
    return None


def score(case, actions, horizon, contact_reference=None):
    pred = actions[:horizon]
    initial = case['request']['cup_relative_state']
    result = {'horizon': horizon, 'predicted_motion': movement(pred, initial),
              'first_predicted_close': tick(first_index(pred, lambda x: x[6] < CLOSE_THRESHOLD)),
              'first_predicted_open': tick(first_index(pred, lambda x: x[6] >= OPEN_THRESHOLD))}
    if case['group'] != 'heldout_source_7':
        if contact_reference is not None:
            start = math.dist(initial[:2], contact_reference[:2]) * 1000.
            end = math.dist(pred[-1][:2], contact_reference[:2]) * 1000.
            close = first_index(pred, lambda action: action[6] < CLOSE_THRESHOLD)
            result['distance_to_heldout_close_reference_xy_mm'] = start
            result['progress_towards_heldout_close_reference_xy_mm'] = start - end
            result['first_close_distance_to_heldout_reference_xy_mm'] = (
                math.dist(pred[close][:2], contact_reference[:2]) * 1000. if close is not None else None)
        return result
    labels = case['targets'][:horizon]
    pred = pred[:len(labels)]
    targets = [row['cup_relative_action'] for row in labels]
    result['valid_targets'] = len(labels)
    result['recorded_target_motion'] = movement(targets, initial)
    result['crosses_compaction_cut'] = any(row['cut_after'] for row in labels)
    result['xyz_rmse_mm'] = math.sqrt(sum((p[i] - t[i]) ** 2 for p, t in zip(pred, targets)
                                           for i in range(3)) / (3 * len(targets))) * 1000.
    result['gripper_mae_pp'] = sum(abs(p[6] - t[6]) for p, t in zip(pred, targets)) * 100. / len(targets)
    for name, expect, hit in (
            ('close', lambda v: v < CLOSE_THRESHOLD, lambda v: v < CLOSE_THRESHOLD),
            ('open', lambda v: v >= OPEN_THRESHOLD, lambda v: v >= OPEN_THRESHOLD)):
        indices = [i for i, target in enumerate(targets) if expect(target[6])]
        first = indices[0] if indices else None
        predicted = first_index(pred, lambda p: hit(p[6]))
        result[name] = {'expected_points': len(indices), 'hit_points': sum(hit(pred[i][6]) for i in indices),
                        'mae_pp': (sum(abs(pred[i][6] - targets[i][6]) for i in indices) * 100. / len(indices)
                                   if indices else None),
                        'first_expected': tick(first), 'first_predicted': tick(predicted),
                        'first_expected_recorded_seconds_from_observation':
                            (labels[first]['target_time_monotonic'] - case['request']['source_time']
                             if first is not None else None),
                        'first_tick_error': (predicted - first if predicted is not None and first is not None else None)}
    kind = case['kind']
    closing = kind in ('near_early', 'near_late', 'contact_approach', 'close_before', 'close_at')
    holding = kind in ('close_after', 'contact_on_table', 'hover_early', 'hover_middle', 'hover_late',
                       'return_early', 'return_middle', 'return_late', 'release_before', 'release_at')
    early_close = first_index(list(zip(pred, targets)),
                              lambda pair: pair[0][6] < CLOSE_THRESHOLD and pair[1][6] >= OPEN_THRESHOLD)
    early_open = first_index(list(zip(pred, targets)),
                             lambda pair: pair[0][6] >= OPEN_THRESHOLD and pair[1][6] < CLOSE_THRESHOLD)
    result['first_early_close_vs_recorded'] = target_event(early_close, pred, labels) if closing else None
    result['first_early_open_vs_recorded'] = target_event(early_open, pred, labels) if holding else None
    result['first_early_effective_close'] = early_gripper_change(case, pred, labels, closing=True) if closing else None
    result['first_early_effective_open'] = early_gripper_change(case, pred, labels, closing=False) if holding else None
    times = case['event_times']
    if closing or holding:
        event = 'close' if closing else 'open'
        command_time = times['close_issued_monotonic' if closing else 'release_issued_monotonic']
        expected_tick = (times['close_compact_step' if closing else 'release_compact_step'] - labels[0]['compact_step'] + 1
                         if labels[0]['compact_row'] is not None else
                         math.ceil((command_time - case['request']['source_time']) * FPS - 1e-7))
        predicted_tick = result[event]['first_predicted']
        due = 1 <= expected_tick <= len(labels)
        value_at_due = pred[expected_tick - 1][6] if due else None
        result['recorded_switch'] = {
            'event': event, 'expected_tick': expected_tick,
            'expected_in_horizon': due,
            'target_tick_hit': ((value_at_due < CLOSE_THRESHOLD if closing else value_at_due >= OPEN_THRESHOLD)
                                if due else None),
            'recorded_command_seconds_from_observation': (
                times['close_issued_monotonic' if closing else 'release_issued_monotonic']
                - case['request']['source_time']),
            'predicted_tick': predicted_tick['tick'] if predicted_tick else None,
            'tick_error': (predicted_tick['tick'] - expected_tick if predicted_tick else None)}
    def restricted(target_time):
        return times['contact_confirmed_monotonic'] <= target_time < times['release_issued_monotonic']
    before_five = first_index(list(zip(pred, labels)),
        lambda pair: pair[0][6] >= OPEN_THRESHOLD and restricted(pair[1]['target_time_monotonic'])
        and pair[1]['target_time_monotonic'] < times['hold_start_monotonic'] + 5.)
    returning = first_index(list(zip(pred, labels)),
        lambda pair: pair[0][6] >= OPEN_THRESHOLD and restricted(pair[1]['target_time_monotonic'])
        and pair[1]['target_time_monotonic'] >= times['return_issued_monotonic'])
    result['first_open_after_contact_before_hover_5s'] = target_event(before_five, pred, labels)
    result['first_open_during_return_before_release'] = target_event(returning, pred, labels)
    nominal_before_five = first_index(list(enumerate(pred)),
        lambda pair: pair[1][6] >= OPEN_THRESHOLD
        and restricted(case['request']['source_time'] + (pair[0] + 1) / FPS)
        and case['request']['source_time'] + (pair[0] + 1) / FPS < times['hold_start_monotonic'] + 5.)
    nominal_return = first_index(list(enumerate(pred)),
        lambda pair: pair[1][6] >= OPEN_THRESHOLD
        and restricted(case['request']['source_time'] + (pair[0] + 1) / FPS)
        and case['request']['source_time'] + (pair[0] + 1) / FPS >= times['return_issued_monotonic'])
    result['first_open_before_hover_5s_on_nominal_clock'] = target_event(nominal_before_five, pred, labels)
    result['first_open_during_return_on_nominal_clock'] = target_event(nominal_return, pred, labels)
    return result


def summarize(cases):
    buckets = defaultdict(list)
    details = []
    references = [case['request']['cup_relative_state'][:3] for case in cases
                  if case['group'] == 'heldout_source_7' and case['kind'] == 'close_at']
    if len(references) != 2:
        raise ValueError('Expected the two recorded close references')
    contact_reference = [sum(row[i] for row in references) / len(references) for i in range(3)]
    for case in cases:
        for pair in case['predictions']:
            for model in ('parent', 'candidate'):
                for horizon in (10, 50):
                    value = score(case, pair[model]['reply']['actions'], horizon, contact_reference)
                    record = {'case_id': case['case_id'], 'group': case['group'], 'kind': case['kind'],
                              'cycle': case.get('cycle'), 'model': model, 'seed': pair['seed'], **value}
                    details.append(record)
                    buckets[(case['group'], case['kind'], model, horizon)].append(record)
    summary = []
    for (group, kind, model, horizon), records in sorted(buckets.items()):
        item = {'group': group, 'kind': kind, 'model': model, 'horizon': horizon, 'draws': len(records),
                'mean_predicted_net_mm': sum(r['predicted_motion']['net_distance_mm'] for r in records) / len(records),
                'mean_predicted_path_mm': sum(r['predicted_motion']['path_length_mm'] for r in records) / len(records),
                'predicted_close_draws': sum(r['first_predicted_close'] is not None for r in records),
                'predicted_open_draws': sum(r['first_predicted_open'] is not None for r in records)}
        if group == 'heldout_source_7':
            item.update({key + '_draws': sum(r[key] is not None for r in records) for key in (
                'first_early_close_vs_recorded', 'first_early_open_vs_recorded',
                'first_early_effective_close', 'first_early_effective_open',
                'first_open_after_contact_before_hover_5s', 'first_open_during_return_before_release',
                'first_open_before_hover_5s_on_nominal_clock', 'first_open_during_return_on_nominal_clock')})
            item['mean_xyz_rmse_mm'] = sum(r['xyz_rmse_mm'] for r in records) / len(records)
            item['mean_gripper_mae_pp'] = sum(r['gripper_mae_pp'] for r in records) / len(records)
            item['mean_recorded_target_net_mm'] = (
                sum(r['recorded_target_motion']['net_distance_mm'] for r in records) / len(records))
            item['mean_recorded_target_path_mm'] = (
                sum(r['recorded_target_motion']['path_length_mm'] for r in records) / len(records))
            for name in ('close', 'open'):
                total = sum(r[name]['expected_points'] for r in records)
                item[name] = {'expected_points': total,
                              'hit_points': sum(r[name]['hit_points'] for r in records),
                              'mae_pp': (sum(r[name]['mae_pp'] * r[name]['expected_points'] for r in records
                                             if r[name]['mae_pp'] is not None) / total if total else None)}
            switches = [r['recorded_switch'] for r in records if 'recorded_switch' in r]
            due = [r for r in switches if r['expected_in_horizon']]
            errors = [r['tick_error'] for r in due if r['tick_error'] is not None]
            item['recorded_switch'] = {
                'expected_in_horizon': len(due), 'predicted_in_horizon': len(errors),
                'missed_in_horizon': len(due) - len(errors),
                'target_tick_hits': sum(r['target_tick_hit'] for r in due),
                'first_prediction_early': sum(e < 0 for e in errors),
                'first_prediction_on_time': sum(e == 0 for e in errors),
                'first_prediction_late': sum(e > 0 for e in errors),
                'mean_signed_tick_error': sum(errors) / len(errors) if errors else None,
                'mean_absolute_tick_error': sum(abs(e) for e in errors) / len(errors) if errors else None}
        else:
            item['mean_progress_towards_heldout_close_reference_xy_mm'] = (
                sum(r['progress_towards_heldout_close_reference_xy_mm'] for r in records) / len(records))
        summary.append(item)
    return summary, details, contact_reference


def compare(args):
    inputs, parent, candidate = (read_json(path) for path in (args.input, args.parent, args.candidate))
    if (inputs['schema'] != INPUT_SCHEMA or not inputs['timing_ready']
            or parent['model'] not in PROFILES or candidate['model'] != 'candidate'
            or parent['seeds'] != candidate['seeds'] or not parent['seeds']):
        raise ValueError('Input or paired predictions do not match')
    seeds = parent['seeds']
    if parent.get('task_override') != candidate.get('task_override'):
        raise ValueError('Paired predictions must use the same task text')
    if candidate.get('task_override'):
        for case in inputs['cases']:
            case['task_override'] = candidate['task_override']
    indices = []
    expected = {(case['case_id'], seed) for case in inputs['cases'] for seed in seeds}
    for model, report in (('parent', parent), ('candidate', candidate)):
        if report.get('observation_context_profile') not in allowed_profiles(model):
            raise ValueError(f'{model} profile differs')
        index = {(row['case_id'], row['seed']): row for row in report['predictions']}
        if len(index) != len(report['predictions']) or set(index) != expected:
            raise ValueError(f'{model} does not contain exactly the paired case/seed set')
        indices.append(index)
    cases = []
    for case in inputs['cases']:
        item = dict(case)
        if args.camera_targets and case['group'] == 'heldout_source_7':
            item['targets'] = case['targets_camera_aligned']
        item['predictions'] = []
        for seed in seeds:
            pair = {'seed': seed}
            for name, index in zip(('parent', 'candidate'), indices):
                draw = index[(case['case_id'], seed)]
                report = parent if name == 'parent' else candidate
                if draw['request'] != model_request(case, name, report.get('timing_mode', 'full'),
                                                    profile=report['observation_context_profile']):
                    raise ValueError(f'{name} input does not match {case["case_id"]}')
                validate_actions(draw['reply']['actions'])
                pair[name] = draw
            item['predictions'].append(pair)
        cases.append(item)
    summary, details, contact_reference = summarize(cases)
    result = {'schema': COMPARISON_SCHEMA, 'evaluation_schema': 'fr5-cup-timing-metrics-v1',
              'case_count': len(cases), 'heldout_source_7_cases': inputs['heldout_source_7_cases'],
              'failed_trial_cases': inputs['failed_trial_cases'], 'seeds': seeds,
              'task_override': candidate.get('task_override'),
              'candidate_timing_mode': candidate.get('timing_mode', 'full'),
              'checkpoints': {'parent': parent['checkpoint'], 'candidate': candidate['checkpoint']},
              'checkpoint_sha256': {'parent': parent['checkpoint_sha256'],
                                    'candidate': candidate['checkpoint_sha256']},
              'action_origin': inputs['action_origin'],
              'thresholds': {'close_below': CLOSE_THRESHOLD, 'open_at_or_above': OPEN_THRESHOLD},
              'heldout_close_reference_cup_relative_xyz_mm': [x * 1000. for x in contact_reference],
              'interpretation': 'Isolated recorded inputs and predicted prefixes, not executed movement or task success.',
              'five_second_rule': 'After recorded contact, before five seconds from recorded hover start; '
                                  'recorded demonstrations requested four seconds. Raw labels remain unchanged.',
              'timing_clock': ('Camera time plus one-sixth second per tick for both predictions and original labels.'
                               if args.camera_targets else
                               'Predicted ticks use 6 Hz; recorded target times retain compacted source gaps.'),
              'gripper_clock': 'Effective changes use integer percent and the one-point deadband, starting '
                               'from the recorded valid gripper position. Height is above the pickup flange '
                               'reference along the measured table normal, not physical cup height.',
              'nominal_clock': 'Observation camera time plus one-sixth second per predicted tick; '
                               'no execution latency or robot motion is assumed.',
              'robot_connections': 0, 'robot_commands': 0, 'metrics': summary, 'metric_details': details,
              'cases': cases}
    write_new(args.output, result)
    print(json.dumps({'output': str(args.output), 'cases': len(cases), 'seeds': seeds,
                      'metric_groups': len(summary)}), flush=True)


def controller_input(args):
    """Join one sealed prediction set to the fixed cup geometry; no model or FR5 I/O."""
    inputs, report = read_json(args.input), read_json(args.predictions)
    if inputs['schema'] != INPUT_SCHEMA or report['schema'] != PREDICTION_SCHEMA:
        raise ValueError('Expected fixed replay inputs and one prediction set')
    seeds = report['seeds']
    index = {(p['case_id'], p['seed']): p for p in report['predictions']}
    if len(index) != len(report['predictions']):
        raise ValueError('Repeated replay prediction')
    cases = []
    for source in inputs['cases']:
        identity = source['case_id']
        if not any((identity, seed) in index for seed in seeds):
            continue
        probe = dict(source, task_override=report['task_override']) if report['task_override'] else source
        request = model_request(probe, report['model'], report['timing_mode'],
                                profile=report['observation_context_profile'])
        group = []
        for seed in seeds:
            prediction = index.pop((identity, seed))
            if request != prediction['request']:
                raise ValueError(f'Prediction request differs from source {identity}')
            validate_actions(prediction['reply']['actions'])
            group.append({'seed': seed, 'candidate': prediction})
        cases.append({k: source[k] for k in ('case_id', 'group', 'kind', 'initial_cup')} | {'predictions': group})
    if index or not cases:
        raise ValueError('Prediction does not match the selected replay inputs')
    write_new(args.output, {'schema': COMPARISON_SCHEMA, 'checkpoints': {'candidate': report['checkpoint']},
                            'checkpoint_sha256': {'candidate': report['checkpoint_sha256']},
                            'seeds': seeds, 'cases': cases})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest='command', required=True)
    cmd = commands.add_parser('prepare', help='prepare fixed observations and compact action labels on CPU')
    cmd.add_argument('--dataset', type=Path, default=DATASET)
    cmd.add_argument('--previous', type=Path, default=PREVIOUS)
    cmd.add_argument('--source-root', type=Path, default=TRACKING_SOURCE)
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=prepare)
    cmd = commands.add_parser('timing', help='add causally replayed CupTiming evidence on CPU')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--source-root', type=Path, default=TIMING_SOURCE)
    cmd.add_argument('--aligned', action='store_true')
    cmd.add_argument('--setdown', action='store_true')
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=add_timing)
    cmd = commands.add_parser('camera-targets', help='derive targets at the actual observation time plus 6 Hz ticks')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    cmd.add_argument('--dataset', type=Path, default=DATASET)
    cmd.add_argument('--source-root', type=Path, required=True)
    cmd.set_defaults(func=add_camera_targets)
    cmd = commands.add_parser('predict', help='predict fixed observations from one checkpoint')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--model', choices=tuple(PROFILES), required=True)
    cmd.add_argument('--checkpoint', type=Path)
    cmd.add_argument('--gpu', type=int, required=True)
    cmd.add_argument('--task-override', help='Counterfactual text probe on the same recorded observations')
    cmd.add_argument('--original-output', type=Path, help='Also replay the recorded task without reloading the model')
    cmd.add_argument('--denoising-steps', type=int, choices=(10, 20, 30), help='Offline sampler convergence probe')
    cmd.add_argument('--no-clock-channel-noise', action='store_true',
                     help='Zero only the initial noise in the aligned observation channels')
    cmd.add_argument('--noise-scale', type=float, choices=(0., .35, 1.), default=1.,
                     help='Offline probe for the effect of the initial flow noise')
    cmd.add_argument('--kind', action='append', help='Replay selected event kinds in an offline probe')
    cmd.add_argument('--timing-mode', choices=('full', 'unknown', 'aligned-observer'), default='full',
                     help='Keep the visual evidence and mask phase and elapsed time for a paired offline probe')
    cmd.add_argument('--seed', type=int, action='append')
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=predict)
    cmd = commands.add_parser('compare', help='score paired predictions and emit a prefix-checker comparison')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--parent', type=Path, required=True)
    cmd.add_argument('--candidate', type=Path, required=True)
    cmd.add_argument('--camera-targets', action='store_true')
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=compare)
    cmd = commands.add_parser('controller-input', help='join one prediction set to its recorded controller geometry')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--predictions', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=controller_input)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
