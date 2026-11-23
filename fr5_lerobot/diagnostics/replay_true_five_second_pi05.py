#!/usr/bin/env python3
"""Replay recorded five-second training demonstrations without robot I/O.

The preparation selects the event observations used by the cup-timing replay,
plus observations around the five-second deadline. It keeps compact targets
and their original times and cuts, as well as a continuous 6 Hz comparison
window made from camera-time TCP samples and latched MoveGripper commands.
Model predictions are open-loop diagnostics, not executed motion.

The source-7 validation split is checked but is not included in this train-fit
report. Run ``prepare --preview`` to inspect the cases without writing output.
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

sys.dont_write_bytecode = True
ROOT = Path('/home/data4/hts/demo1/fr5-learning')
DATASET = ROOT / 'datasets/fr5-joint-stream-20261102-v2-compacted'
OLD_SOURCE = ROOT / 'src/fr5-pi05-cup-action-clock-20261113-v1'
OLD_CHECKPOINT = ROOT / 'checkpoints/pi05-fr5-stream-cup-action-clock-n10-20261113-v1/step-000128'
SOURCES = {0: 1, 3: 2, 4: 1, 5: 1}
OLD_PROFILE = 'state-history-physical-cup-action-clock-v1'
NEW_PROFILE = 'state-history-physical-cup-aligned-v1'
SETDOWN_PROFILE = 'state-history-physical-cup-setdown-v1'
SETDOWN_FIXED_PROFILE = 'state-history-physical-cup-setdown-fixed-v1'
SETDOWN_PROFILES = (SETDOWN_PROFILE, 'state-history-physical-cup-setdown-clean-v1', SETDOWN_FIXED_PROFILE)
INPUT_SCHEMA = 'fr5-five-second-train-fit-inputs-v1'
PREDICTION_SCHEMA = 'fr5-five-second-train-fit-predictions-v1'
REPORT_SCHEMA = 'fr5-five-second-train-fit-report-v1'
OFFSETS = (0.5, 1., 2., 4., 6., 8.)
CHUNK, FPS = 50, 6
DESCENT_MM = 2.0


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path.suffix != '.json':
        raise ValueError(f'Output must be a JSON file under {ROOT}')
    if path.exists():
        raise FileExistsError(path)
    return path


def write_new(path, value):
    path = check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def near(actual, expected, label, tolerance=1e-5):
    if len(actual) != len(expected) or any(abs(float(a) - float(b)) > tolerance
                                           for a, b in zip(actual, expected)):
        raise ValueError(f'{label} differs from the recorded data')


def relative_pose(pose, base):
    return ([(pose[i] - base[i]) / 1000. for i in range(3)]
            + [math.radians(x) for x in pose[3:]])


def state_from_capture(capture, base, pose):
    robot = capture['robot']
    valid = bool(capture['gripper_feedback_valid'])
    return (relative_pose(pose(capture), base)
            + [robot['gripper_position'] / 100. if valid else 0., float(valid)])


def history_from_captures(captures, current, first, base, pose):
    samples = [c for c in captures if c['camera']['source_timestamp_monotonic'] >= first]
    times = [c['camera']['source_timestamp_monotonic'] for c in samples]
    history = []
    for offset in OFFSETS:
        index = bisect_right(times, current - offset) - 1
        history.append(None if index < 0 else {
            'age_s': current - times[index],
            'state': state_from_capture(samples[index], base, pose),
        })
    return history


def select_cases(source, captures, frames, mappings, commands):
    """Use the existing cup-timing event anchors and add deadline anchors."""
    selected = []
    for hold in source['hold_records']:
        cycle = hold['repetition']
        phases = {c['teacher']['phase']: c for c in commands
                  if c.get('executed') and c['teacher']['cycle'] == cycle}
        close, lift, ret, release = (phases[name] for name in
                                     ('close', 'lift_probe', 'return', 'release'))
        contact = next(c for c in captures
                       if close['issued_monotonic'] <= c['camera']['source_timestamp_monotonic']
                       < lift['issued_monotonic'] and c['gripper_feedback_valid']
                       and c['robot']['gripper_motiondone'] == 2)
        contact_time = contact['camera']['source_timestamp_monotonic']

        def time_of(row):
            return row['observation']['camera_time_monotonic']

        def between(start, end, fraction):
            choices = [row for row in mappings if start <= time_of(row) < end]
            if not choices:
                raise ValueError(f'No source {source["source_index"]} cycle {cycle} observation in interval')
            desired = start + fraction * (end - start)
            return min(choices, key=lambda row: (abs(time_of(row) - desired), row['step']))

        def transition(command):
            return next(row['step'] for row in mappings
                        if row['gripper_target']['sequence'] == command['sequence'])

        ci, ri = transition(close), transition(release)
        if ci == 0 or ri == 0:
            raise ValueError('A gripper transition has no previous compact observation')
        contact_rows = [row for row in mappings
                        if contact_time <= time_of(row) < lift['issued_monotonic']
                        and frames[row['observation']['frame_index']]['robot']['gripper_motiondone'] == 2
                        and row['observation']['gripper_valid']]
        if not contact_rows:
            raise ValueError('No settled contact before lift')
        deadline = hold['start_monotonic'] + 5.
        in_hold = [row for row in mappings
                   if hold['start_monotonic'] <= time_of(row) < hold['end_monotonic']]
        before = [row for row in in_hold if time_of(row) < deadline]
        after = [row for row in mappings if deadline <= time_of(row) < release['issued_monotonic']]
        if not before or not after:
            raise ValueError('Missing recorded observations around the five-second deadline')
        closest = lambda rows, t: min(rows, key=lambda row: (abs(time_of(row) - t), row['step']))
        bounds = {
            'contact_confirmed_monotonic': contact_time,
            'contact_confirmed_source_frame': contact['frame_index'],
            'hold_start_monotonic': hold['start_monotonic'],
            'hold_end_monotonic': hold['end_monotonic'],
            'hold_record_actual_seconds': hold['actual_duration_s'],
            'requested_hold_seconds': hold['requested_seconds'],
            'five_second_deadline_monotonic': deadline,
            'return_issued_monotonic': ret['issued_monotonic'],
            'close_issued_monotonic': close['issued_monotonic'],
            'release_issued_monotonic': release['issued_monotonic'],
            'close_compact_step': ci,
            'release_compact_step': ri,
            'close_reference_tcp_xyz_mm': frames[mappings[ci]['observation']['frame_index']]['robot']['tcp'][:3],
        }
        anchors = [
            ('near_early', between(phases['approach_near']['issued_monotonic'],
                                   phases['approach_contact']['issued_monotonic'], .33)),
            ('near_late', between(phases['approach_near']['issued_monotonic'],
                                  phases['approach_contact']['issued_monotonic'], .75)),
            ('contact_approach', between(phases['approach_contact']['issued_monotonic'],
                                         close['issued_monotonic'], .5)),
            ('close_before', mappings[ci - 1]), ('close_at', mappings[ci]),
            ('close_after', between(close['issued_monotonic'], lift['issued_monotonic'], .05)),
            ('contact_on_table', contact_rows[-1]),
            ('hover_early', between(hold['start_monotonic'], hold['end_monotonic'], .1)),
            ('hover_middle', between(hold['start_monotonic'], hold['end_monotonic'], .5)),
            ('hover_late', between(hold['start_monotonic'], hold['end_monotonic'], .9)),
            ('deadline_minus_1s', closest(before, deadline - 1.)),
            ('deadline_before', before[-1]), ('deadline_after', after[0]),
            ('return_early', between(ret['issued_monotonic'], release['issued_monotonic'], .25)),
            ('return_middle', between(ret['issued_monotonic'], release['issued_monotonic'], .5)),
            ('return_late', between(ret['issued_monotonic'], release['issued_monotonic'], .75)),
            ('release_before', mappings[ri - 1]), ('release_at', mappings[ri]),
            ('release_after', between(release['issued_monotonic'],
                                      phases['retreat_probe']['issued_monotonic'], .05)),
            ('retreat', between(phases['retreat_probe']['issued_monotonic'],
                                phases['retreat_clear']['issued_monotonic'], .5)),
        ]
        selected.extend({'cycle': cycle, 'kind': kind, 'mapping': mapping, 'event_times': bounds}
                        for kind, mapping in anchors)
    return selected


def gripper_commands(root, commands):
    initial = next(c for c in commands if c['kind'] == 'hold')
    latched = [{'sequence': initial['sequence'], 'issued_monotonic': initial['issued_monotonic'],
                'fraction': initial['target'][6] / 100., 'source': 'initial_hold'}]
    for command in commands:
        if command['kind'] != 'gripper' or not command.get('executed'):
            continue
        detail = read_json(root / f"command-{command['sequence']:04d}.json")
        if detail['method'] != 'MoveGripper' or detail['args'][1] != command['target'][6]:
            raise ValueError('MoveGripper command and target journal differ')
        latched.append({'sequence': command['sequence'], 'issued_monotonic': command['issued_monotonic'],
                        'fraction': detail['args'][1] / 100., 'source': 'MoveGripper'})
    if any(a['issued_monotonic'] >= b['issued_monotonic'] for a, b in zip(latched, latched[1:])):
        raise ValueError('Gripper commands are not in time order')
    return latched


def build_inputs(dataset, source_root):
    import pyarrow.parquet as pq
    sys.path.insert(0, str(source_root / 'fr5_lerobot/training'))
    from cup_relative_stream_data import _action, _pose

    proof = read_json(dataset / 'meta/stream_provenance.json')
    sources = proof['sources']
    if sources[7]['source_index'] != 7 or sources[7]['split'] != 'validation':
        raise ValueError('Source 7 must remain in the validation split')
    mappings_by_source = defaultdict(list)
    for mapping in read_jsonl(dataset / 'meta/stream_source_frames.jsonl'):
        if mapping['source_index'] in SOURCES:
            mappings_by_source[mapping['source_index']].append(mapping)
    table = pq.read_table(dataset / 'data/chunk-000/file-000.parquet',
                          columns=['index', 'episode_index', 'frame_index', 'observation.state', 'action'])
    rows = {int(row['index']): row for row in table.to_pylist()}
    cases = []
    source_info = []
    for index, expected_cycles in SOURCES.items():
        source = sources[index]
        if (source['source_index'] != index or source['split'] != 'train'
                or len(source['hold_records']) != expected_cycles
                or any(hold['requested_seconds'] != 5. for hold in source['hold_records'])):
            raise ValueError(f'Source {index} is not the expected five-second training demonstration')
        root = Path(source['trial'])
        review = read_json(root / 'user-review.json')
        if review.get('reviewer') != 'user' or review.get('verdict') != 'success':
            raise ValueError(f'Source {index} is not user-confirmed')
        geometry, opening = read_json(root / 'geometry.json'), read_json(root / 'source-observation.json')
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
            raise ValueError(f'Source {index} camera and calibration differ')
        captures = read_jsonl(root / 'continuous/capture.jsonl')
        frames = {c['frame_index']: c for c in captures}
        times = [c['camera']['source_timestamp_monotonic'] for c in captures]
        if (len(frames) != len(captures)
                or any(a >= b for a, b in zip(times, times[1:]))):
            raise ValueError(f'Source {index} capture order is invalid')
        mappings = mappings_by_source[index]
        if not mappings:
            raise ValueError(f'Source {index} has no compact rows')
        for step, mapping in enumerate(mappings):
            row = rows[mapping['frame']]
            if (mapping['step'] != step or mapping['split'] != 'train'
                    or int(row['frame_index']) != step
                    or int(row['episode_index']) != mapping['episode_index']):
                raise ValueError(f'Source {index} compact row order or split differs')
            near(row['action'], mapping['derived_action_radians_fraction'], 'Compact action', 1e-6)
        commands = read_json(root / 'target-command-journal.json')
        grip = gripper_commands(root, commands)
        grip_times = [item['issued_monotonic'] for item in grip]
        selected = select_cases(source, captures, frames, mappings, commands)
        first_time = mappings[0]['observation']['camera_time_monotonic']
        first_frame = mappings[0]['observation']['frame_index']
        for item in selected:
            mapping = item['mapping']
            capture = frames[mapping['observation']['frame_index']]
            stamp = capture['camera']['source_timestamp_monotonic']
            near([stamp], [mapping['observation']['camera_time_monotonic']], 'Capture time')
            state = state_from_capture(capture, cup['base_xyz_mm'], _pose)
            joints = [math.radians(x) for x in capture['robot']['joints']] + state[6:]
            near(joints, rows[mapping['frame']]['observation.state'], 'Compact observation')
            case_id = f"source{index}-cycle{item['cycle']}-{item['kind']}-{mapping['frame']}"
            request = {'schema': 'fr5-stream-request-v1', 'id': case_id,
                       'task': source['task_instruction'],
                       'rgb_jpeg_base64': base64.b64encode(
                           (root / 'continuous' / capture['rgb_file']).read_bytes()).decode('ascii'),
                       'source_time': stamp, 'source_clock': 'host_monotonic',
                       'policy_contract': 'fr5-joint-position-stream-v1',
                       'state': joints, 'cup_relative_state': state,
                       'observation_history': history_from_captures(captures, stamp, first_time,
                                                                    cup['base_xyz_mm'], _pose),
                       'initial_cup_uv': cup['uv'], 'cup_base_xyz_mm': cup['base_xyz_mm']}
            compact_targets = []
            for target in mappings[mapping['step']:mapping['step'] + CHUNK]:
                compact_targets.append({
                    'compact_row': target['frame'], 'compact_step': target['step'],
                    'source_v1_frame': target['source_v1_frame'],
                    'target_time_monotonic': target['target_time_monotonic'],
                    'cut_after': target['cut_after'], 'cut_id': target['cut_id'],
                    'gripper_command_sequence': target['gripper_target']['sequence'],
                    'cup_relative_action': _action(rows[target['frame']], target, frames,
                                                   cup['base_xyz_mm']),
                })
            targets = []
            last_available = min(mappings[-1]['target_time_monotonic'], times[-1])
            for tick in range(1, CHUNK + 1):
                target_time = stamp + tick / FPS
                if target_time > last_available:
                    break
                raw_index = bisect_right(times, target_time) - 1
                if raw_index < 0 or times[raw_index] < stamp:
                    raise ValueError('No camera-time target at or after the observation')
                raw = captures[raw_index]
                grip_index = bisect_right(grip_times, times[raw_index]) - 1
                if grip_index < 0:
                    raise ValueError('No initial gripper target at the raw camera time')
                latched = grip[grip_index]
                target_gripper = grip[bisect_right(grip_times, target_time) - 1]
                targets.append({
                    'tick': tick, 'nominal_time_monotonic': target_time,
                    'raw_camera_frame': raw['frame_index'],
                    'raw_camera_time_monotonic': times[raw_index],
                    'raw_camera_age_s': target_time - times[raw_index],
                    'raw_robot_time_monotonic': raw['robot']['received_monotonic'],
                    'raw_gripper_command': latched,
                    'gripper_command_at_target': target_gripper,
                    'raw_cup_relative_action': relative_pose(_pose(raw), cup['base_xyz_mm'])
                                               + [target_gripper['fraction']],
                })
            cases.append({
                'case_id': case_id, 'group': 'train_fit', 'split': 'train',
                'source_index': index, 'cycle': item['cycle'], 'kind': item['kind'],
                'row_index': mapping['frame'], 'source_frame_index': capture['frame_index'],
                'tracker_start_frame_index': first_frame, 'source_root': str(root),
                'event_times': item['event_times'], 'initial_cup': cup,
                'request': request, 'targets': targets, 'compact_targets': compact_targets,
                'observers': {},
            })
        source_info.append({'source_index': index, 'split': source['split'], 'cycles': expected_cycles,
                            'task_instruction': source['task_instruction'], 'source_root': str(root),
                            'hold_records': source['hold_records']})
    return {'schema': INPUT_SCHEMA, 'dataset': str(dataset), 'preparation_source_root': str(source_root),
            'evaluation': 'training-set fit; source 7 remains validation and is excluded',
            'target_clock': '6 Hz from observation camera time; compact target times and cuts retained separately',
            'raw_labels': 'last camera frame at or before each nominal target; direct recorded TCP and '
                          'MoveGripper target latched at the nominal time, bounded by the last compact '
                          'target and capture',
            'observation_history_offsets_s': list(OFFSETS),
            'sources': source_info, 'observation_replays': {}, 'cases': cases}


def prepare(args):
    if not args.preview and args.output is None:
        raise ValueError('prepare requires --output or --preview')
    if not args.preview:
        check_output(args.output)
    inputs = build_inputs(args.dataset, args.source_root)
    if not args.preview:
        write_new(args.output, inputs)
    print(json.dumps({'output': str(args.output) if not args.preview else None,
                      'evaluation': inputs['evaluation'], 'cases': len(inputs['cases']),
                      'per_source': dict(Counter(c['source_index'] for c in inputs['cases'])),
                      'full_50_targets': sum(len(c['targets']) == CHUNK for c in inputs['cases']),
                      'compact_cross_cut_cases': sum(any(t['cut_after'] for t in c['compact_targets'][:-1])
                                             for c in inputs['cases'])}, ensure_ascii=False), flush=True)


def observer_names(mode):
    return {'old': ('old',), 'both': ('raw', 'aligned'), 'setdown': ('setdown',)}[mode]


def replay_observers(inputs, source_root, mode):
    import cv2
    import numpy as np
    sys.path.insert(0, str(source_root))
    from fr5_jev.cup_tracking import CupTracker
    from fr5_jev.cup_timing import CupTiming

    cv2.setNumThreads(1)
    names = observer_names(mode)
    grouped = defaultdict(list)
    for case in inputs['cases']:
        grouped[case['source_index']].append(case)
    for index, cases in grouped.items():
        root = Path(cases[0]['source_root'])
        captures = read_jsonl(root / 'continuous/capture.jsonl')
        cup = cases[0]['initial_cup']
        calibration = read_json(root / 'source-observation.json')['calibration']
        tracker = CupTracker(cup, calibration['base_from_color_camera_mm'])
        timers = {name: CupTiming(cup, aligned=name == 'aligned', setdown=name == 'setdown') for name in names}
        selected = {case['source_frame_index'] for case in cases}
        starts = {case['tracker_start_frame_index'] for case in cases}
        if len(starts) != 1:
            raise ValueError('Cases have different tracker start frames')
        start, stop = starts.pop(), max(selected)
        results = {}
        previous_time = -math.inf
        for capture in captures:
            frame = capture['frame_index']
            if frame < start:
                continue
            if frame > stop:
                break
            stamp = capture['camera']['source_timestamp_monotonic']
            if stamp <= previous_time:
                raise ValueError('Capture time is not strictly increasing')
            previous_time = stamp
            image = cv2.imread(str(root / 'continuous' / capture['rgb_file']), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Missing source {index} RGB frame {frame}')
            with np.load(root / 'continuous' / capture['depth_file'], allow_pickle=False) as depth:
                tracking = tracker.update(image, depth['depth_mm'], capture['camera'], capture['robot'],
                                          gripper_valid=bool(capture['gripper_feedback_valid']))
            evidence = {name: timer.update(capture['camera'], capture['robot'], tracking)
                        for name, timer in timers.items()}
            if frame in selected:
                results[frame] = (tracking, evidence)
        if set(results) != selected:
            raise ValueError(f'Source {index} is missing selected capture frames')
        for case in cases:
            tracking, evidence = results[case['source_frame_index']]
            if case['observers']:
                previous = next(iter(case['observers'].values()))['tracking']
                if previous != tracking:
                    raise ValueError(f"Tracker changed between source snapshots for {case['case_id']}")
            for name in names:
                state = evidence[name].get('state')
                if not isinstance(state, dict) or (name == 'aligned' and 'support_gap_mm' not in state):
                    raise ValueError(f'{name} observer did not return the expected state')
                model_state = dict(state)
                gap_source = 'observer' if 'support_gap_mm' in state else None
                if name == 'raw' and gap_source is None:
                    model_state['support_gap_mm'] = (
                        evidence[name]['flange_rise_mm']
                        if tracking['native_contact'] == 1 and state['phase'] != 'unknown' else None)
                    gap_source = 'raw_observer_flange_rise_mm'
                case['observers'][name] = {'tracking': tracking, 'timing': evidence[name],
                                           'model_timing': model_state, 'support_gap_source': gap_source}
        print(f'{mode}: replayed source {index}, {len(cases)} cases', file=sys.stderr, flush=True)
    for name in names:
        inputs['observation_replays'][name] = str(source_root)
    return inputs


def observe(args):
    check_output(args.output)
    inputs = read_json(args.input)
    if inputs.get('schema') != INPUT_SCHEMA:
        raise ValueError('Expected prepared five-second inputs')
    names = observer_names(args.mode)
    if any(name in inputs['observation_replays'] for name in names):
        raise ValueError('Input already contains the requested observer replay')
    replay_observers(inputs, args.source_root, args.mode)
    write_new(args.output, inputs)
    print(json.dumps({'output': str(args.output), 'observers': names,
                      'cases': len(inputs['cases'])}), flush=True)


def model_request(case, observer, profile):
    recorded = case['observers'][observer]
    request = dict(case['request'])
    request['observation_context_profile'] = profile
    request['cup_tracking'] = {key: recorded['tracking'][key]
                               for key in ('rim_minus_jaw_mm', 'native_contact')}
    request['cup_timing'] = recorded['model_timing']
    return request


def validate_actions(reply):
    actions = reply['actions']
    if (reply.get('wire_action_representation') != 'cup-relative-tcp-v1'
            or len(actions) != CHUNK
            or any(len(row) != 7 or any(not math.isfinite(float(v)) for v in row) for row in actions)):
        raise ValueError('Expected finite 50 x 7 absolute cup-relative TCP targets')


def predict(args):
    check_output(args.output)
    inputs = read_json(args.input)
    if inputs.get('schema') != INPUT_SCHEMA:
        raise ValueError('Expected prepared five-second inputs')
    if args.aligned_only and args.model != 'new':
        raise ValueError('The aligned observer is only available for the aligned model')
    observers = (('old',) if args.model == 'old' else ('setdown',) if args.model == 'setdown'
                 else ('aligned',) if args.aligned_only else ('raw', 'aligned'))
    if any(name not in inputs['observation_replays'] for name in observers):
        raise ValueError(f'Missing observer replay: expected {observers}')
    seeds = args.seed or [20261105, 20261113]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct')
    checkpoint = args.checkpoint or (OLD_CHECKPOINT if args.model == 'old' else None)
    if checkpoint is None or not checkpoint.is_dir():
        raise ValueError('A matching checkpoint directory is required')
    sys.path.insert(0, str(checkpoint))
    with redirect_stdout(sys.stderr):
        from serve_stream_pi05 import Inference
        model = Inference(checkpoint, gpu=args.gpu)
        model.torch.set_num_threads(4)
    profile = model.info['observation_context_profile']
    if profile not in {'old': (OLD_PROFILE,), 'new': (NEW_PROFILE,), 'setdown': SETDOWN_PROFILES}[args.model]:
        raise ValueError(f'Checkpoint profile {profile!r} does not match {args.model}')
    predictions = []
    for i, case in enumerate(inputs['cases'], 1):
        for observer in observers:
            request = model_request(case, observer, profile)
            for seed in seeds:
                with redirect_stdout(sys.stderr):
                    model.reset()
                    model.torch.manual_seed(seed)
                    reply = model.predict(request)
                validate_actions(reply)
                predictions.append({'case_id': case['case_id'], 'observer': observer, 'seed': seed,
                                    'task': request['task'], 'source_time': request['source_time'],
                                    'cup_tracking': request['cup_tracking'],
                                    'cup_timing': request['cup_timing'], 'reply': reply})
        print(f'{args.model}: {i}/{len(inputs["cases"])} cases', file=sys.stderr, flush=True)
    write_new(args.output, {'schema': PREDICTION_SCHEMA, 'model': args.model, 'seeds': seeds,
                            'input': str(args.input), 'checkpoint': str(checkpoint),
                            'observation_context_profile': profile, 'observers': list(observers),
                            'predictions': predictions})
    print(json.dumps({'output': str(args.output), 'predictions': len(predictions)}), flush=True)


def action_point(case, index, action):
    target = case['targets'][index] if index < len(case['targets']) else None
    compact = case['compact_targets'][index] if index < len(case['compact_targets']) else None
    stamp = case['request']['source_time']
    nominal = stamp + (index + 1) / FPS
    point = {'tick': index + 1, 'nominal_seconds_from_observation': (index + 1) / FPS,
             'nominal_time_monotonic': nominal,
             'nominal_seconds_from_deadline': nominal - case['event_times']['five_second_deadline_monotonic'],
             'cup_relative_action': action}
    if target is not None:
        point.update(raw_camera_time_monotonic=target['raw_camera_time_monotonic'],
                     raw_camera_seconds_from_deadline=(target['raw_camera_time_monotonic']
                                                       - case['event_times']['five_second_deadline_monotonic']),
                     raw_camera_age_s=target['raw_camera_age_s'],
                     raw_cup_relative_action=target['raw_cup_relative_action'],
                     gripper_command_at_camera=target['raw_gripper_command'],
                     gripper_command_at_target=target.get('gripper_command_at_target'))
    if compact is not None:
        point.update(compact_index_target_time_monotonic=compact['target_time_monotonic'],
                     compact_index_row=compact['compact_row'],
                     compact_index_cut_after=compact['cut_after'])
    return point


def normal_delta_mm(case, action):
    initial = case['request']['cup_relative_state']
    normal = case['initial_cup']['table_normal']
    return sum((action[i] - initial[i]) * normal[i] for i in range(3)) * 1000.


def first_descent(case, actions, horizon):
    for i, action in enumerate(actions[:horizon]):
        delta = normal_delta_mm(case, action)
        if delta <= -DESCENT_MM:
            return {**action_point(case, i, action), 'normal_delta_mm': delta}
    return None


def first_descent_from_peak(case, actions, horizon):
    peak, peak_tick = 0., 0
    for i, action in enumerate(actions[:horizon]):
        delta = normal_delta_mm(case, action)
        if peak - delta >= DESCENT_MM:
            return {**action_point(case, i, action), 'normal_delta_mm': delta,
                    'peak_normal_delta_mm': peak, 'peak_tick': peak_tick,
                    'drop_from_peak_mm': peak - delta}
        if delta > peak:
            peak, peak_tick = delta, i + 1
    return None


def effective_events(case, actions, horizon):
    state = case['request']['cup_relative_state']
    if not state[7]:
        return {'first_close': None, 'first_release': None, 'feedback_valid': False}
    previous = round(state[6] * 100)
    first_close = first_release = None
    for i, action in enumerate(actions[:horizon]):
        target = round(max(0., min(1., action[6])) * 100)
        if abs(target - previous) <= 1:
            continue
        event = {**action_point(case, i, action), 'previous_percent': previous,
                 'effective_target_percent': target}
        if target < previous and first_close is None:
            first_close = event
        if target > previous and first_release is None:
            first_release = event
        previous = target
    return {'first_close': first_close, 'first_release': first_release, 'feedback_valid': True}


def metrics(case, actions):
    targets = case['targets']
    raw = [target['raw_cup_relative_action'] for target in targets]
    count = len(targets)
    first_ten = []
    previous_predicted = previous_recorded = 0.
    for i, action in enumerate(actions[:10]):
        row = action_point(case, i, action)
        row['predicted_normal_delta_mm'] = normal_delta_mm(case, action)
        row['predicted_step_normal_delta_mm'] = row['predicted_normal_delta_mm'] - previous_predicted
        previous_predicted = row['predicted_normal_delta_mm']
        if i < count:
            row['recorded_normal_delta_mm'] = normal_delta_mm(case, raw[i])
            row['recorded_step_normal_delta_mm'] = row['recorded_normal_delta_mm'] - previous_recorded
            previous_recorded = row['recorded_normal_delta_mm']
        first_ten.append(row)
    return {'valid_recorded_targets': count,
            'mean_raw_camera_age_s': (sum(t['raw_camera_age_s'] for t in targets) / count if count else None),
            'max_raw_camera_age_s': (max(t['raw_camera_age_s'] for t in targets) if count else None),
            'compact_targets': len(case['compact_targets']),
            'compact_crosses_cut': any(t['cut_after'] for t in case['compact_targets'][:-1]),
            'recorded_camera_span_s': (targets[-1]['raw_camera_time_monotonic']
                                       - case['request']['source_time']) if targets else None,
            'compact_target_span_s': (case['compact_targets'][-1]['target_time_monotonic']
                                      - case['request']['source_time']) if case['compact_targets'] else None,
            'first_10': first_ten,
            'predicted_descent_2mm_first_10': first_descent(case, actions, 10),
            'recorded_descent_2mm_first_10': first_descent(case, raw, 10),
            'predicted_descent_2mm_full': first_descent(case, actions, CHUNK),
            'recorded_descent_2mm_full': first_descent(case, raw, CHUNK),
            'predicted_descent_from_peak_2mm_first_10': first_descent_from_peak(case, actions, 10),
            'recorded_descent_from_peak_2mm_first_10': first_descent_from_peak(case, raw, 10),
            'predicted_descent_from_peak_2mm_full': first_descent_from_peak(case, actions, CHUNK),
            'recorded_descent_from_peak_2mm_full': first_descent_from_peak(case, raw, CHUNK),
            'predicted_effective_first_10': effective_events(case, actions, 10),
            'recorded_effective_first_10': effective_events(case, raw, 10),
            'predicted_effective_full': effective_events(case, actions, CHUNK),
            'recorded_effective_full': effective_events(case, raw, CHUNK),
            'raw_xyz_rmse_mm': (math.sqrt(sum((p[j] - t[j]) ** 2
                                               for p, t in zip(actions, raw) for j in range(3))
                                          / (3 * count)) * 1000. if count else None),
            'raw_gripper_mae_pp': (sum(abs(p[6] - t[6]) for p, t in zip(actions, raw))
                                    * 100. / count if count else None)}


def latch_gripper_at_target(inputs):
    commands_by_root = {}
    for case in inputs['cases']:
        root = Path(case['source_root'])
        if root not in commands_by_root:
            commands = gripper_commands(root, read_json(root / 'target-command-journal.json'))
            commands_by_root[root] = ([c['issued_monotonic'] for c in commands], commands)
        times, commands = commands_by_root[root]
        for target in case['targets']:
            index = bisect_right(times, target['nominal_time_monotonic']) - 1
            if index < 0:
                raise ValueError('No gripper target at the nominal prediction time')
            command = commands[index]
            target['gripper_command_at_target'] = command
            target['raw_cup_relative_action'][6] = command['fraction']


def compare(args):
    check_output(args.output)
    inputs, old, new = (read_json(path) for path in (args.input, args.old, args.new))
    if inputs.get('schema') != INPUT_SCHEMA:
        raise ValueError('Expected prepared five-second inputs')
    latch_gripper_at_target(inputs)
    if (old.get('schema') != PREDICTION_SCHEMA or new.get('schema') != PREDICTION_SCHEMA
            or old['model'] != 'old' or new['model'] not in ('new', 'setdown')
            or not set(old['seeds']) <= set(new['seeds'])
            or old['observation_context_profile'] != OLD_PROFILE
            or new['observation_context_profile'] not in (SETDOWN_PROFILES if new['model'] == 'setdown' else (NEW_PROFILE,))):
        raise ValueError('Expected paired old and new model predictions with the same seeds')
    seeds = new['seeds']
    expected_cases = {case['case_id'] for case in inputs['cases']}
    draws = {}
    new_observers = tuple(new.get('observers', ('raw', 'aligned')))
    if new_observers not in (('raw', 'aligned'), ('aligned',), ('setdown',)):
        raise ValueError('Expected aligned predictions, with or without the raw observer ablation')
    for report, names in ((old, ('old',)), (new, new_observers)):
        expected = {(case, name, seed) for case in expected_cases for name in names for seed in report['seeds']}
        index = {(row['case_id'], row['observer'], row['seed']): row for row in report['predictions']}
        if len(index) != len(report['predictions']) or set(index) != expected:
            raise ValueError('Predictions do not cover the exact paired cases and seeds')
        draws.update(index)
    details = []
    summaries = defaultdict(list)
    for case in inputs['cases']:
        if case['split'] != 'train' or case['group'] != 'train_fit' or case['source_index'] not in SOURCES:
            raise ValueError('A non-training source entered the train-fit report')
        for observer in ('old', *new_observers):
            profile = OLD_PROFILE if observer == 'old' else new['observation_context_profile']
            request = model_request(case, observer, profile)
            for seed in old['seeds'] if observer == 'old' else seeds:
                row = draws[(case['case_id'], observer, seed)]
                if (row['task'] != request['task'] or row['source_time'] != request['source_time']
                        or row['cup_tracking'] != request['cup_tracking']
                        or row['cup_timing'] != request['cup_timing']):
                    raise ValueError(f"Prediction input differs for {case['case_id']}")
                validate_actions(row['reply'])
                value = metrics(case, row['reply']['actions'])
                entry = {'case_id': case['case_id'], 'source_index': case['source_index'],
                         'cycle': case['cycle'], 'kind': case['kind'], 'observer': observer,
                         'seed': seed, 'source_time': request['source_time'],
                         'task': request['task'], 'event_times': case['event_times'],
                         'cup_timing': request['cup_timing'], 'metrics': value}
                details.append(entry)
                summaries[(observer, case['kind'])].append(value)
    summary = []
    for (observer, kind), values in sorted(summaries.items()):
        summary.append({'observer': observer, 'kind': kind, 'draws': len(values),
                        'descent_2mm_first_10': sum(v['predicted_descent_2mm_first_10'] is not None for v in values),
                        'descent_from_peak_2mm_first_10': sum(
                            v['predicted_descent_from_peak_2mm_first_10'] is not None for v in values),
                        'descent_before_deadline_first_10': sum(
                            v['predicted_descent_2mm_first_10'] is not None
                            and v['predicted_descent_2mm_first_10']['nominal_seconds_from_deadline'] < 0
                            for v in values),
                        'valid_gripper_feedback_draws': sum(
                            v['predicted_effective_first_10']['feedback_valid'] for v in values),
                        'effective_close_first_10': sum(
                            v['predicted_effective_first_10']['first_close'] is not None for v in values),
                        'effective_release_first_10': sum(
                            v['predicted_effective_first_10']['first_release'] is not None for v in values),
                        'mean_raw_xyz_rmse_mm': sum(v['raw_xyz_rmse_mm'] for v in values) / len(values)})
    report = {'schema': REPORT_SCHEMA, 'evaluation': inputs['evaluation'],
              'input': str(args.input), 'old_predictions': str(args.old), 'new_predictions': str(args.new),
              'checkpoints': {'old': old['checkpoint'], 'new': new['checkpoint']},
              'seeds': seeds, 'legacy_seeds': old['seeds'], 'sources': inputs['sources'],
              'interpretation': 'Recorded-input open-loop train fit; not robot execution or held-out performance.',
              'timing': 'Deadline is hold start plus 5 s. Prediction times are nominal 6 Hz; '
                        'recorded target and camera times are preserved separately.',
              'raw_observer_support_gap': 'When absent from the raw observer state, the new-model '
                                          'input uses its measured flange rise with the aligned '
                                          'observer contact and known-phase conditions.',
              'raw_label_comparison': 'Direct camera samples are an independent train-fit reference; '
                                      'they are not the interpolated training targets.',
              'descent': 'At least 2 mm downward along the recorded table normal, measured both from '
                         'the observation and from the preceding peak.',
              'gripper': 'First effective integer-percent change greater than one point from the last '
                         'effective target, starting at valid recorded gripper feedback. Recorded '
                         'MoveGripper commands are latched at the nominal target time.',
              'summary': summary, 'details': details}
    write_new(args.output, report)
    print(json.dumps({'output': str(args.output), 'evaluation': report['evaluation'],
                      'draws': len(details), 'cases': len(inputs['cases'])}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    cmd = commands.add_parser('prepare', help='prepare fixed inputs and camera-time labels on CPU')
    cmd.add_argument('--dataset', type=Path, default=DATASET)
    cmd.add_argument('--source-root', type=Path, default=OLD_SOURCE)
    cmd.add_argument('--output', type=Path)
    cmd.add_argument('--preview', action='store_true', help='build and inspect in memory without writing')
    cmd.set_defaults(func=prepare)
    cmd = commands.add_parser('observe', help='replay the old observer or both new observer modes on CPU')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--mode', choices=('old', 'both', 'setdown'), required=True)
    cmd.add_argument('--source-root', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=observe)
    cmd = commands.add_parser('predict', help='run one checkpoint on its fixed observer replay(s)')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--model', choices=('old', 'new', 'setdown'), required=True)
    cmd.add_argument('--aligned-only', action='store_true', help='Omit the raw observer ablation')
    cmd.add_argument('--checkpoint', type=Path)
    cmd.add_argument('--gpu', type=int, required=True)
    cmd.add_argument('--seed', type=int, action='append')
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=predict)
    cmd = commands.add_parser('compare', help='report paired five-second training-set fit')
    cmd.add_argument('--input', type=Path, required=True)
    cmd.add_argument('--old', type=Path, required=True)
    cmd.add_argument('--new', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    cmd.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
