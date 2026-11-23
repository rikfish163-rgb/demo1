#!/usr/bin/env python3
"""Evaluate archived cup-policy prefixes without a robot connection or a GPU.

Inputs are the 43-case replay, one or more saved prediction sets, the held-out
source's original capture and command journal, and a true-five train-fit
comparison.  A path may be local or ``4090:/absolute/path``.  JSON is written
to stdout; a failed gate is data in the report, not a process error.

The selected execution horizon is scored independently of the ten-point
diagnostic.  The latter is always reported even when only three actions will be
executed before replanning.  Both are open-loop checks of isolated observations;
neither establishes actual cup support, controller feasibility, or task success.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True

FPS = 6
FULL_HORIZON = 10
BASE_SEEDS = (20261105, 20261113)
SIX_SEEDS = (*BASE_SEEDS, 20261114, 20261115, 20261116, 20261117)
SIX_SEED_KINDS = {'close_at', 'hover_middle', 'hover_late', 'return_late', 'release_at'}
NEAR_MM = 1.5
SPEED_MM_S = 4.0
CLOSE_PERCENT = 88
OPEN_PERCENT = 98
DEAD_BAND = 1
LATE_TICKS = 1
DESCENT_MM = 2.0
SETDOWN_PROFILES = ('state-history-physical-cup-setdown-v1',
                    'state-history-physical-cup-setdown-clean-v1',
                    'state-history-physical-cup-setdown-fixed-v1')
# Demonstrated cup-relative pickup neighborhood used by the existing controller.
FAILED_PICKUP_CUP_RELATIVE_MM = (159.7, 71.3, 95.8)


def read_text(path):
    if str(path).startswith('4090:/'):
        remote = str(path)[len('4090:'):]
        return subprocess.run(
            ['ssh', '-T', '-o', 'BatchMode=yes', '4090', 'cat -- ' + shlex.quote(remote)],
            text=True, stdout=subprocess.PIPE, check=True,
        ).stdout
    return Path(path).read_text(encoding='utf-8')


def read_json(path):
    return json.loads(read_text(path))


def read_jsonl(path):
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def normal_of(case):
    normal = case['initial_cup']['table_normal']
    if len(normal) != 3 or abs(math.sqrt(dot(normal, normal)) - 1.) > .001:
        raise ValueError(f"{case['case_id']}: invalid table normal")
    return normal


def point_projection(point, normal):
    return dot(point[:3], normal)


def action_tcp(case, action):
    origin = case['initial_cup']['base_xyz_mm']
    return [origin[i] + 1000. * action[i] for i in range(3)]


def percent(action):
    return round(max(0., min(1., action[6])) * 100.)


def target_tick(source_time, event_time):
    return max(1, math.ceil((event_time - source_time) * FPS - 1e-7))


def cases_index(inputs):
    if inputs.get('schema') != 'fr5-cup-timing-replay-inputs-v1':
        raise ValueError('Expected the archived cup replay inputs')
    cases = {case['case_id']: case for case in inputs['cases']}
    if len(cases) != 43 or len(cases) != len(inputs['cases']):
        raise ValueError('Expected 43 unique archived requests')
    counts = Counter(c['group'] for c in cases.values())
    if counts != {'heldout_source_7': 34, 'failed_trial_evaluation_only': 9}:
        raise ValueError('Expected 34 source-7 and nine failed-trial requests')
    for case in cases.values():
        if case['group'] == 'heldout_source_7':
            if case['source_index'] != 7 or case['split'] != 'validation':
                raise ValueError('Held-out source or split changed')
            labels = case['targets_camera_aligned'][:FULL_HORIZON]
            if len(labels) != FULL_HORIZON:
                raise ValueError(f"{case['case_id']}: insufficient camera-clock targets")
            for tick, label in enumerate(labels, 1):
                if abs(label['target_time_monotonic'] - case['request']['source_time'] - tick / FPS) > 1e-5:
                    raise ValueError(f"{case['case_id']}: targets do not use the original 6 Hz clock")
    return cases


def validate_request(case, request, override, profile):
    base = case['request']
    for key, value in base.items():
        if key in ('observation_context_profile', 'cup_tracking', 'cup_timing'):
            continue
        expected = override if key == 'task' and override is not None else value
        if request.get(key) != expected:
            raise ValueError(f"{case['case_id']}: saved request differs in {key}")
    if request.get('observation_context_profile') != profile:
        raise ValueError(f"{case['case_id']}: context profile differs")
    if request.get('cup_tracking') != {k: case['tracking'][k] for k in ('rim_minus_jaw_mm', 'native_contact')}:
        raise ValueError(f"{case['case_id']}: Gemini tracking differs")
    timing_key = ('timing_setdown' if profile in SETDOWN_PROFILES
                  else 'timing_aligned' if profile == 'state-history-physical-cup-aligned-v1'
                  else None)
    if timing_key is None or not case.get(timing_key):
        raise ValueError(f"{case['case_id']}: missing timing for {profile}")
    if request.get('cup_timing') != case[timing_key]['state']:
        raise ValueError(f"{case['case_id']}: saved timing differs from {timing_key}")


def prediction_index(cases, reports, sources):
    metadata_keys = ('checkpoint', 'checkpoint_sha256', 'task_override', 'timing_mode',
                     'denoising_steps', 'observation_context_profile')
    metadata, rows, inputs_by_case, provenance = None, {}, {}, []
    for report, source in zip(reports, sources):
        if report.get('schema') != 'fr5-cup-timing-replay-predictions-v1':
            raise ValueError('Expected saved cup-policy predictions')
        meta = {k: report.get(k) for k in metadata_keys}
        if metadata is not None and meta != metadata:
            raise ValueError('Prediction sets have different checkpoints or inference settings')
        metadata = meta
        seeds = report['seeds']
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate prediction seed')
        kinds = report.get('case_kinds')
        included = {case_id for case_id, case in cases.items() if kinds is None or case['kind'] in kinds}
        expected = {(case_id, seed) for case_id in included for seed in seeds}
        actual = {(row['case_id'], row['seed']) for row in report['predictions']}
        if expected != actual or len(actual) != len(report['predictions']):
            raise ValueError('Prediction file does not cover its declared cases and seeds')
        for row in report['predictions']:
            key = (row['case_id'], row['seed'])
            if key in rows:
                raise ValueError(f'Duplicate prediction {key}')
            case = cases[row['case_id']]
            if row['kind'] != case['kind'] or row['group'] != case['group']:
                raise ValueError(f'{key}: case identity changed')
            request, reply = row['request'], row['reply']
            validate_request(case, request, meta['task_override'], meta['observation_context_profile'])
            if reply.get('checkpoint_sha256') != meta['checkpoint_sha256']:
                raise ValueError(f'{key}: reply checkpoint differs')
            if request != inputs_by_case.setdefault(row['case_id'], request):
                raise ValueError(f'{key}: different requests across seeds')
            actions = reply['actions']
            if len(actions) < FULL_HORIZON or any(len(a) != 7 or not all(math.isfinite(x) for x in a)
                                                  for a in actions[:FULL_HORIZON]):
                raise ValueError(f'{key}: expected at least ten finite 7-D actions')
            rows[key] = row
        provenance.append({'file': str(source), 'input_recorded_in_file': report['input'],
                           'seeds': seeds, 'case_kinds': kinds,
                           'case_count': len(included), 'draws': len(actual)})
    for case_id, case in cases.items():
        required = SIX_SEEDS if (case['group'] == 'heldout_source_7' and case['kind'] in SIX_SEED_KINDS) else BASE_SEEDS
        missing = [seed for seed in required if (case_id, seed) not in rows]
        if missing:
            raise ValueError(f'{case_id}: missing required replay seeds {missing}')
    return metadata, rows, provenance


def raw_support_references(cases, frames, journal):
    frame_map = {f['frame_index']: f for f in frames}
    if len(frame_map) != len(frames):
        raise ValueError('Duplicate raw camera frame')
    result = {}
    for cycle in (1, 2):
        case = next(c for c in cases.values() if c['group'] == 'heldout_source_7'
                    and c['kind'] == 'release_at' and c['cycle'] == cycle)
        events, normal = case['event_times'], normal_of(case)
        commands = {r['teacher']['phase']: r for r in journal
                    if r.get('executed') and r.get('teacher', {}).get('cycle') == cycle
                    and r['teacher'].get('phase') in ('close', 'return', 'release')}
        if set(commands) != {'close', 'return', 'release'}:
            raise ValueError(f'Cycle {cycle}: incomplete original command journal')
        for phase in commands:
            key = {'close': 'close_issued_monotonic', 'return': 'return_issued_monotonic',
                   'release': 'release_issued_monotonic'}[phase]
            if abs(commands[phase]['issued_monotonic'] - events[key]) > 1e-6:
                raise ValueError(f'Cycle {cycle}: original {phase} timestamp differs')
        if commands['close']['target'][6] >= CLOSE_PERCENT or commands['release']['target'][6] < OPEN_PERCENT:
            raise ValueError(f'Cycle {cycle}: original jaw commands are unexpected')
        contact = frame_map[events['contact_confirmed_source_frame']]
        if (not contact['gripper_feedback_valid'] or contact['robot']['gripper_motiondone'] != 2
                or abs(contact['camera']['source_timestamp_monotonic'] - events['contact_confirmed_monotonic']) > 1e-6):
            raise ValueError(f'Cycle {cycle}: recorded gripper contact cannot be verified')
        release = events['release_issued_monotonic']
        eligible = [f for f in frames if release - .5 <= f['camera']['source_timestamp_monotonic'] < release
                    and f['gripper_feedback_valid'] and f['robot']['gripper_motiondone'] == 2
                    and abs(f['robot']['tcp_speed'][0]) <= SPEED_MM_S]
        heights = [point_projection(f['robot']['tcp'], normal) for f in eligible]
        if (len(eligible) < 2 or eligible[-1]['camera']['source_timestamp_monotonic']
                - eligible[0]['camera']['source_timestamp_monotonic'] < .1
                or release - eligible[-1]['camera']['source_timestamp_monotonic'] > .25
                or max(heights) - min(heights) > .5):
            raise ValueError(f'Cycle {cycle}: no stable, contact-confirmed pre-release reference')
        height = statistics.median(heights)
        ready = []
        for f in frames:
            t = f['camera']['source_timestamp_monotonic']
            if (events['return_issued_monotonic'] <= t < release and f['gripper_feedback_valid']
                    and f['robot']['gripper_motiondone'] == 2 and abs(f['robot']['tcp_speed'][0]) <= SPEED_MM_S
                    and abs(point_projection(f['robot']['tcp'], normal) - height) <= NEAR_MM):
                ready.append(f)
        stable_pair = next(((a, b) for a, b in zip(ready, ready[1:])
                            if 0 < b['camera']['source_timestamp_monotonic']
                            - a['camera']['source_timestamp_monotonic'] <= .35), None)
        if stable_pair is None:
            raise ValueError(f'Cycle {cycle}: no supported-height feedback pair')
        ready_time = stable_pair[1]['camera']['source_timestamp_monotonic']
        result[cycle] = {
            'height_projection_mm': height,
            'pre_release_frame_indices': [f['frame_index'] for f in eligible],
            'pre_release_height_spread_mm': max(heights) - min(heights),
            'confirmed_contact_frame': contact['frame_index'],
            'reference_minus_close_command_mm': height - point_projection(events['close_reference_tcp_xyz_mm'], normal),
            'reference_minus_contact_mm': height - point_projection(contact['robot']['tcp'], normal),
            'recorded_first_ready_proxy_monotonic': ready_time,
            'recorded_release_command_monotonic': release,
            'seconds_ready_proxy_before_command': release - ready_time,
        }
    return result, frame_map


def check_capture(case, frame_map):
    frame = frame_map[case['source_frame_index']]
    stamp = frame['camera']['source_timestamp_monotonic']
    request = case['request']
    if abs(stamp - request['source_time']) > 1e-6:
        raise ValueError(f"{case['case_id']}: raw Gemini timestamp differs")
    if frame['camera']['source_camera_serial'] != case['initial_cup']['camera_serial']:
        raise ValueError(f"{case['case_id']}: raw Gemini camera differs")
    if bool(request['cup_relative_state'][7]) != bool(frame['gripper_feedback_valid']):
        raise ValueError(f"{case['case_id']}: raw PGI validity differs")
    if math.dist(action_tcp(case, request['cup_relative_state']), frame['robot']['tcp'][:3]) > .01:
        raise ValueError(f"{case['case_id']}: raw TCP anchor differs")
    return frame


def predicted_points(case, actions, horizon, support=None):
    normal = normal_of(case)
    previous = action_tcp(case, case['request']['cup_relative_state'])
    previous_gap = (point_projection(previous, normal) - support['height_projection_mm']) if support else None
    points = []
    for tick, action in enumerate(actions[:horizon], 1):
        tcp = action_tcp(case, action)
        gap = point_projection(tcp, normal) - support['height_projection_mm'] if support else None
        speed = math.dist(tcp, previous) * FPS
        ready = (gap is not None and abs(gap) <= NEAR_MM and abs(previous_gap) <= NEAR_MM
                 and speed <= SPEED_MM_S)
        points.append({'tick': tick, 'nominal_time_monotonic': case['request']['source_time'] + tick / FPS,
                       'jaw_percent': percent(action), 'support_height_gap_mm': gap,
                       'target_speed_mm_s': speed, 'predicted_ready_proxy': ready})
        previous, previous_gap = tcp, gap
    return points


def score_draw(case, row, frame, support, horizon):
    actions, request = row['reply']['actions'][:horizon], row['request']
    normal = normal_of(case)
    start = request['source_time']
    points = predicted_points(case, actions, horizon, support)
    valid = bool(request['cup_relative_state'][7])
    identity = {'case_id': case['case_id'], 'seed': row['seed'], 'kind': case['kind'],
                'group': case['group']}
    result = {**identity, 'valid_gripper_feedback': valid, 'violations': [], 'observations': {}}
    selection = row['reply'].get('cup_proposal_selection')
    if selection and selection['matched'] is False:
        result['violations'].append({'code': 'no_feasible_complete_proposal',
                                     'draws': selection['draws']})
    if not valid:
        result['observations']['unassessed'] = 'recorded PGI feedback invalid'
        return result
    events = case.get('event_times')
    if events:
        close_ref = events['close_reference_tcp_xyz_mm']
        recorded_contact = bool(frame['gripper_feedback_valid'] and frame['robot']['gripper_motiondone'] == 2
                                and start < events['release_issued_monotonic'])
        actual_gap = point_projection(frame['robot']['tcp'], normal) - support['height_projection_mm']
        actual_ready = recorded_contact and abs(actual_gap) <= NEAR_MM and abs(frame['robot']['tcp_speed'][0]) <= SPEED_MM_S
        labels = case['targets_camera_aligned'][:horizon]
        label_open = next((i for i, label in enumerate(labels, 1)
                           if percent(label['cup_relative_action']) >= OPEN_PERCENT), None)
        command_tick = target_tick(start, events['release_issued_monotonic'])
        result['observations'].update(
            visual=case['timing_aligned']['state']['visual'], recorded_contact=recorded_contact,
            observed_support_gap_mm=actual_gap, observed_tcp_speed_mm_s=frame['robot']['tcp_speed'][0],
            observed_ready_proxy=actual_ready, teacher_open_command_tick=command_tick,
            teacher_open_label_tick=label_open,
        )
    else:
        close_ref = None
        recorded_contact = False
        actual_ready = False
    last = round(request['cup_relative_state'][6] * 100.)
    assumed_contact = recorded_contact
    for action, point in zip(actions, points):
        target = point['jaw_percent']
        if abs(target - last) <= DEAD_BAND:
            continue
        tcp = action_tcp(case, action)
        if target < last and last >= CLOSE_PERCENT:
            if close_ref is None:
                relative = [1000. * a for a in action[:3]]
                dx = math.dist(relative[:2], FAILED_PICKUP_CUP_RELATIVE_MM[:2])
                dz = relative[2] - FAILED_PICKUP_CUP_RELATIVE_MM[2]
            else:
                dx = math.dist(tcp[:2], close_ref[:2])
                dz = point_projection([tcp[i] - close_ref[i] for i in range(3)], normal)
            if dx > 15. or abs(dz) > 12.:
                result['violations'].append({'code': 'far_close', 'tick': point['tick'],
                                             'horizontal_mm': dx, 'normal_gap_mm': dz})
        if target > last and last < OPEN_PERCENT and assumed_contact:
            gap, speed = point['support_height_gap_mm'], point['target_speed_mm_s']
            code = ('open_above_support_proxy' if gap > NEAR_MM else
                    'open_below_support_proxy' if gap < -NEAR_MM else
                    'open_while_target_moving' if speed > SPEED_MM_S else None)
            if code:
                result['violations'].append({'code': code, 'tick': point['tick'],
                                             'support_height_gap_mm': gap, 'target_speed_mm_s': speed,
                                             'from_percent': last, 'to_percent': target})
            if target >= OPEN_PERCENT:
                assumed_contact = False
        last = target
    first_close = next((p['tick'] for p in points if p['jaw_percent'] < CLOSE_PERCENT), None)
    first_open = next((p['tick'] for p in points if p['jaw_percent'] >= OPEN_PERCENT), None)
    ready = next((p['tick'] for p in points if recorded_contact and p['predicted_ready_proxy']), None)
    result['observations'].update(first_close_tick=first_close, first_full_open_tick=first_open,
                                  first_predicted_ready_proxy_tick=ready)
    # The held-out source requested four seconds.  Its five-second prompt is a
    # counterfactual and remains separate from the real five-second train fit.
    task = request.get('task', '').replace(' ', '')
    if events and case['kind'] in ('hover_early', 'hover_middle', 'hover_late') and '5秒' in task:
        begin = events['hold_start_monotonic']
        initial = point_projection(action_tcp(case, request['cup_relative_state']), normal)
        peak, down = 0., 0.
        for action, point in zip(actions, points):
            if begin <= point['nominal_time_monotonic'] < begin + 5.:
                delta = point_projection(action_tcp(case, action), normal) - initial
                down = max(down, -delta, peak - delta)
                peak = max(peak, delta)
        result['observations']['counterfactual_five_second_max_down_mm'] = down
        if down >= DESCENT_MM:
            result['violations'].append({'code': 'early_counterfactual_five_second_descent',
                                         'max_down_mm': down})
    if events and case['kind'] == 'close_at':
        expected = target_tick(start, events['close_issued_monotonic'])
        deadline = expected + LATE_TICKS
        result['observations'].update(teacher_close_command_tick=expected, close_deadline_tick=deadline)
        if deadline <= horizon and (first_close is None or first_close > deadline):
            result['violations'].append({'code': 'late_normal_close', 'deadline_tick': deadline,
                                         'first_close_tick': first_close})
        elif deadline > horizon and first_close is None:
            result['observations']['close_deadline_outside_prefix'] = True
    if events and case['kind'] == 'release_at':
        if not actual_ready:
            result['observations']['unassessed'] = 'recorded release observation lacks contact or stable-height proxy'
        else:
            expected = target_tick(start, events['release_issued_monotonic'])
            deadline = expected + LATE_TICKS
            result['observations']['release_deadline_tick'] = deadline
            if deadline <= horizon and (first_open is None or first_open > deadline):
                result['violations'].append({'code': 'late_supported_open', 'deadline_tick': deadline,
                                             'first_full_open_tick': first_open})
            elif deadline > horizon and first_open is None:
                result['observations']['release_deadline_outside_prefix'] = True
    return result


def fit_scores(report, checkpoint, observer, horizon):
    if (report.get('schema') != 'fr5-five-second-train-fit-report-v1'
            or report['checkpoints']['new'] != checkpoint):
        raise ValueError('True-five comparison is missing or belongs to a different checkpoint')
    seeds = report['seeds']
    if not set(BASE_SEEDS) <= set(seeds):
        raise ValueError('True-five fit lacks the paired baseline seeds')
    rows = [r for r in report['details'] if r['observer'] == observer]
    keys = {(r['case_id'], r['seed']) for r in rows}
    if len(keys) != len(rows) or not rows:
        raise ValueError('Missing or duplicate true-five observer predictions')
    case_ids = {r['case_id'] for r in rows}
    if len(case_ids) != 100 or keys != {(case_id, seed) for case_id in case_ids for seed in seeds}:
        raise ValueError('True-five observer must cover all 100 recorded cases for every seed')
    cycles = {(r['source_index'], r['cycle']) for r in rows}
    if cycles != {(0, 1), (3, 1), (3, 2), (4, 1), (5, 1)}:
        raise ValueError('True-five fit does not contain the five original training cycles')
    failures, eligible, worst = [], 0, 0.
    phases = Counter()
    for row in rows:
        if row['seed'] not in seeds:
            raise ValueError('Unexpected fit seed')
        times = row['event_times']
        begin, deadline = times['hold_start_monotonic'], times['five_second_deadline_monotonic']
        if times['requested_hold_seconds'] != 5. or abs(deadline - begin - 5.) > 1e-6:
            raise ValueError('Fit row is not an original five-second demonstration')
        points = row['metrics']['first_10'][:horizon]
        for i, point in enumerate(points, 1):
            if point['tick'] != i or abs(point['nominal_time_monotonic'] - row['source_time'] - i / FPS) > 1e-5:
                raise ValueError('True-five prediction clock mismatch')
        points = [p for p in points if begin <= p['nominal_time_monotonic'] < deadline]
        if not points:
            continue
        eligible += 1
        phases[row['kind']] += 1
        peak, max_drop = 0., 0.
        for point in points:
            value = point['predicted_normal_delta_mm']
            max_drop = max(max_drop, -value, peak - value)
            peak = max(peak, value)
        worst = max(worst, max_drop)
        if max_drop >= DESCENT_MM:
            failures.append({'case_id': row['case_id'], 'source_index': row['source_index'],
                             'cycle': row['cycle'], 'kind': row['kind'], 'seed': row['seed'],
                             'max_down_mm': max_drop})
    if not eligible:
        raise ValueError('No predictions before the original five-second deadlines')
    return {'observer': observer, 'seeds': seeds, 'cycles': sorted(map(list, cycles)),
            'eligible_draws': eligible, 'draws_by_phase': dict(sorted(phases.items())),
            'worst_down_mm': worst, 'failure_count': len(failures), 'failures': failures}


def summarize(results):
    phases = {}
    for row in results:
        phase = phases.setdefault(row['kind'], {'draws': 0, 'valid_feedback': 0,
                                                'unassessed': 0, 'failure_draws': 0,
                                                'failure_counts': Counter(), 'failures': [],
                                                'physical_and_command': Counter()})
        phase['draws'] += 1
        phase['valid_feedback'] += int(row['valid_gripper_feedback'])
        phase['unassessed'] += int('unassessed' in row['observations'])
        observed = row['observations']
        if row['valid_gripper_feedback']:
            phase['physical_and_command']['observed_ready_proxy'] += int(observed.get('observed_ready_proxy', False))
            phase['physical_and_command']['predicted_ready_proxy_in_prefix'] += int(
                observed.get('first_predicted_ready_proxy_tick') is not None)
            phase['physical_and_command']['teacher_open_label_in_prefix'] += int(
                observed.get('teacher_open_label_tick') is not None)
            phase['physical_and_command']['no_full_open_in_prefix'] += int(
                observed.get('first_full_open_tick') is None)
            phase['physical_and_command']['no_open_and_no_predicted_ready'] += int(
                observed.get('first_full_open_tick') is None
                and observed.get('first_predicted_ready_proxy_tick') is None)
        if row['violations']:
            phase['failure_draws'] += 1
            phase['failure_counts'].update({v['code'] for v in row['violations']})
            phase['failures'].append({'case_id': row['case_id'], 'seed': row['seed'],
                                      'violations': row['violations'], 'observation': observed})
    return {kind: {**phase, 'failure_counts': dict(sorted(phase['failure_counts'].items())),
                   'physical_and_command': dict(sorted(phase['physical_and_command'].items()))}
            for kind, phase in sorted(phases.items())}


def run_horizon(cases, predictions, frame_map, support, fit, meta, observer, horizon):
    results = []
    for (case_id, _), row in sorted(predictions.items()):
        case = cases[case_id]
        held = case['group'] == 'heldout_source_7'
        frame = check_capture(case, frame_map) if held else None
        results.append(score_draw(case, row, frame, support[case['cycle']] if held else None, horizon))
    five = fit_scores(fit, meta['checkpoint'], observer, horizon)
    phases = summarize(results)
    failures = sum(row['failure_draws'] for row in phases.values()) + five['failure_count']
    critical_unknown = [r for r in results if r['kind'] in ('close_at', 'release_at') and
                        ('unassessed' in r['observations'] or
                         r['observations'].get('close_deadline_outside_prefix') or
                         r['observations'].get('release_deadline_outside_prefix'))]
    unknown = [r for r in results if not r['valid_gripper_feedback']]
    assessable_gate = 'fail' if failures else 'inconclusive' if critical_unknown else 'pass'
    gate = 'inconclusive' if assessable_gate == 'pass' and unknown else assessable_gate
    return {'horizon': horizon, 'gate': gate, 'assessable_draws_gate': assessable_gate,
            'failure_draws_including_train_fit': failures,
            'critical_unassessed_draws': len(critical_unknown), 'phases': phases,
            'true_five_second_training_fit': five,
            'unknown_feedback': [{'case_id': r['case_id'], 'seed': r['seed']} for r in unknown]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--predictions', action='append', required=True,
                        help='Saved predictions; repeat for the normal and selective extra seeds')
    parser.add_argument('--fit-comparison', required=True)
    parser.add_argument('--heldout-capture', required=True)
    parser.add_argument('--heldout-journal', required=True)
    parser.add_argument('--execution-prefix', type=int, choices=(3, 10), required=True)
    parser.add_argument('--fit-observer', choices=('aligned', 'setdown'),
                        help='Defaults to the observer matching the prediction context profile')
    args = parser.parse_args()
    cases = cases_index(read_json(args.inputs))
    metadata, predictions, provenance = prediction_index(cases, [read_json(p) for p in args.predictions], args.predictions)
    frames, journal = read_jsonl(args.heldout_capture), read_json(args.heldout_journal)
    support, frame_map = raw_support_references(cases, frames, journal)
    fit = read_json(args.fit_comparison)
    observer = args.fit_observer or ('setdown' if metadata['observation_context_profile'] in SETDOWN_PROFILES else 'aligned')
    short = run_horizon(cases, predictions, frame_map, support, fit, metadata, observer, 3)
    full = run_horizon(cases, predictions, frame_map, support, fit, metadata, observer, 10)
    selected = short if args.execution_prefix == 3 else full
    result = {
        'schema': 'fr5-cup-setdown-offline-gate-v1',
        'scope': 'Isolated recorded requests and predicted actions; no GPU, robot connection or physical success.',
        'input': args.inputs, 'predictions': provenance, 'checkpoint': metadata,
        'train_fit_comparison': args.fit_comparison, 'raw_capture': args.heldout_capture,
        'raw_command_journal': args.heldout_journal,
        'nominal_clock': 'original Gemini source monotonic timestamp plus tick/6 s; execution and inference latency excluded',
        'limits': {'support_reference_mm': NEAR_MM, 'target_speed_mm_s': SPEED_MM_S,
                   'normal_close_below_percent': CLOSE_PERCENT, 'full_open_at_or_above_percent': OPEN_PERCENT,
                   'command_deadband_percent': DEAD_BAND, 'allowed_late_ticks_after_recorded_event': LATE_TICKS,
                   'early_five_second_descent_mm': DESCENT_MM},
        'support_proxy': 'median table-normal TCP height from slow, contact-confirmed raw frames in the last 0.5 s before recorded release; not proof of cup-bottom support',
        'release_timing': 'Require timely full opening at the recorded supported release observation. Earlier return prefixes need not open before they reach support.',
        'heldout_reference': support,
        'coverage': {'fixed_cases': len(cases), 'draws': len(predictions),
                     'all_cases_required_seeds': list(BASE_SEEDS),
                     'six_seed_kinds': sorted(SIX_SEED_KINDS), 'six_seeds': list(SIX_SEEDS),
                     'failed_trial_use': 'evaluation only; no demonstration label or success'},
        'execution_prefix': args.execution_prefix, 'execution_gate': selected['gate'],
        'execution_assessable_draws_gate': selected['assessable_draws_gate'],
        'prefix_3': short, 'prefix_10': full,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
