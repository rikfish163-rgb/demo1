#!/usr/bin/env python3
"""Score saved cup-policy replay comparisons without connecting to a robot.

Use the full held-out comparison and the full true-five training-fit comparison.
Paths may be local or ``4090:/absolute/path``. The latter are read with SSH.
Both reports must contain the predictions/details, not only aggregate summaries.
The result is JSON on stdout; inputs and predictions are never modified.

Example, with A set to the server's pi05-cup-timing-replay-validation directory::

    python score_cup_boundary_replay.py \
        --baseline-heldout "4090:$A/comparison-aligned-events-five-second.json" \
        --baseline-fit "4090:$A/true-five-train-fit-events-vs-legacy-v1.json" \
        --heldout-capture 4090:/path/to/source7/materialized/continuous/capture.jsonl

Add --candidate-heldout and --candidate-fit for a second pair of comparisons.
When both checkpoints occur in the same comparison documents, specify the
corresponding --candidate-heldout-role and --candidate-fit-observer instead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys


FPS = 6
HORIZON = 10
CLOSE = .88
OPEN = .90
SUPPORT_REFERENCE_TOLERANCE_MM = 1.5
CONTACT_REFERENCE_MM = (159.7, 71.3, 95.8)
HISTORICAL_CONTROLLER = (Path(__file__).resolve().parents[1] / 'results' /
                         'pi05-cup-action-clock-five-second-controller-20261113.json')
CLOSING_KINDS = {'near_early', 'near_late', 'contact_approach', 'close_before', 'close_at'}
HOLDING_KINDS = {'close_after', 'contact_on_table', 'hover_early', 'hover_middle',
                 'hover_late', 'return_early', 'return_middle', 'return_late',
                 'release_before', 'release_at'}
HISTORICAL_CASE = 'source7-cycle1-return_late-3426'
HISTORICAL_SEED = 20261113


def read_text(source):
    if source == '-':
        return sys.stdin.read()
    if source.startswith('4090:/'):
        path = source[len('4090:'):]
        return subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '4090',
                               'cat -- ' + shlex.quote(path)], check=True,
                              stdout=subprocess.PIPE, text=True).stdout
    return Path(source).read_text(encoding='utf-8')


def read_json(source):
    return json.loads(read_text(str(source)))


def case_index(report):
    if report.get('schema') != 'fr5-cup-tracking-replay-comparison-v1' or 'cases' not in report:
        raise ValueError('Held-out input must be a full cup replay comparison with cases')
    result = {c['case_id']: c for c in report['cases']}
    if len(result) != len(report['cases']):
        raise ValueError('Duplicate held-out case ID')
    return result


def draw_index(case, role):
    result = {p['seed']: p[role] for p in case['predictions']}
    if len(result) != len(case['predictions']):
        raise ValueError(f"Duplicate seed for {case['case_id']}")
    return result


def fit_index(report, observer):
    if report.get('schema') != 'fr5-five-second-train-fit-report-v1' or 'details' not in report:
        raise ValueError('Train-fit input must be a full five-second comparison with details')
    rows = [d for d in report['details'] if d['observer'] == observer]
    result = {(d['case_id'], d['seed']): d for d in rows}
    if not rows or len(result) != len(rows):
        raise ValueError(f'No unique train-fit draws for observer {observer}')
    for d in rows:
        times = d['event_times']
        if (d['source_index'] not in (0, 3, 4, 5)
                or times.get('requested_hold_seconds') != 5.0
                or abs(times['five_second_deadline_monotonic']
                       - times['hold_start_monotonic'] - 5.) > 1e-6):
            raise ValueError('Train-fit row is not an original five-second training example')
    return result


def validate_models(heldout, role, fit, observer):
    held_checkpoint = heldout['checkpoints'][role]
    fit_checkpoint = fit['checkpoints']['old' if observer == 'old' else 'new']
    if held_checkpoint != fit_checkpoint:
        raise ValueError('Held-out and training-fit reports refer to different checkpoints')
    if heldout['seeds'] != fit['seeds']:
        raise ValueError('Held-out and training-fit reports have different seeds')
    return held_checkpoint


def validate_pair(old_held, old_role, new_held, new_role, old_fit, old_observer,
                  new_fit, new_observer):
    if (old_held['seeds'] != new_held['seeds']
            or old_held.get('task_override') != new_held.get('task_override')):
        raise ValueError('Candidate seeds or task override differ from baseline')
    if not old_fit.get('input') or old_fit['input'] != new_fit.get('input'):
        raise ValueError('Train-fit comparisons must reference the same input document')
    left, right = case_index(old_held), case_index(new_held)
    if set(left) != set(right):
        raise ValueError('Candidate and baseline have different held-out/failed case IDs')
    for name, case in left.items():
        other = right[name]
        for key in ('group', 'kind', 'initial_cup', 'event_times', 'targets'):
            if case.get(key) != other.get(key):
                raise ValueError(f'{name}: baseline and candidate differ in {key}')
        a, b = draw_index(case, old_role), draw_index(other, new_role)
        if set(a) != set(b):
            raise ValueError(f'{name}: baseline and candidate seeds differ')
        for seed in a:
            if a[seed]['request'] != b[seed]['request']:
                raise ValueError(f'{name} seed {seed}: model requests differ')
    a, b = fit_index(old_fit, old_observer), fit_index(new_fit, new_observer)
    if set(a) != set(b):
        raise ValueError('Candidate and baseline have different train-fit case/seed pairs')
    for key in a:
        x, y = a[key], b[key]
        for field in ('source_index', 'cycle', 'kind', 'source_time', 'task', 'event_times', 'cup_timing'):
            if x[field] != y[field]:
                raise ValueError(f'{key}: training-fit inputs differ in {field}')
        # The full comparison stores the recorded camera reference in every point.
        def raw_points(row):
            return [{k: v for k, v in point.items() if k.startswith(('raw_', 'gripper_command_'))}
                    for point in row['metrics']['first_10']]
        if raw_points(x) != raw_points(y):
            raise ValueError(f'{key}: training-fit recorded camera targets differ')


def contact_frames(cases, source):
    if not source:
        return {}
    wanted = {c['event_times']['contact_confirmed_source_frame']:
              c['event_times']['contact_confirmed_monotonic']
              for c in cases.values() if c['group'] == 'heldout_source_7'}
    found = {}
    for line in read_text(source).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frame = row['frame_index']
        if frame not in wanted:
            continue
        if (abs(row['camera']['source_timestamp_monotonic'] - wanted[frame]) > 1e-6
                or not row['gripper_feedback_valid'] or row['robot']['gripper_motiondone'] != 2):
            raise ValueError(f'Frame {frame} does not match confirmed contact feedback')
        found[frame] = row['robot']['tcp'][:3]
    if set(found) != set(wanted):
        raise ValueError('Recorded contact frames missing from capture')
    return found


def geometry(case, action, contacts):
    cup = case['initial_cup']
    tcp = [cup['base_xyz_mm'][i] + 1000. * action[i] for i in range(3)]
    normal = cup['table_normal']
    reference = case['event_times']['close_reference_tcp_xyz_mm']

    def relative(reference_tcp):
        d = [tcp[i] - reference_tcp[i] for i in range(3)]
        return {'xy_distance_mm': math.hypot(d[0], d[1]),
                'normal_gap_mm': sum(d[i] * normal[i] for i in range(3))}

    frame = case['event_times']['contact_confirmed_source_frame']
    return {'relative_to_close_command_tcp': relative(reference),
            'relative_to_confirmed_contact_tcp': relative(contacts[frame]) if frame in contacts else None}


def effective_changes(state, actions):
    """Match the saved evaluator's integer-percent and one-point deadband."""
    if len(state) < 8 or not state[7]:
        return None
    previous = round(state[6] * 100.)
    changes = []
    for i, action in enumerate(actions[:HORIZON]):
        target = round(max(0., min(1., action[6])) * 100.)
        if abs(target - previous) <= 1:
            continue
        changes.append({'tick': i + 1, 'direction': 'close' if target < previous else 'open',
                        'previous_percent': previous, 'target_percent': target})
        previous = target
    return changes


def first_threshold(actions, direction):
    return next((i + 1 for i, a in enumerate(actions[:HORIZON])
                 if (a[6] < CLOSE if direction == 'close' else a[6] >= OPEN)), None)


def event_detail(case, draw, change, contacts):
    tick = change['tick']
    action = draw['reply']['actions'][tick - 1]
    request = draw['request']
    labels = case['targets']
    if tick > len(labels):
        raise ValueError(f"{case['case_id']}: missing camera-aligned label at tick {tick}")
    nominal = request['source_time'] + tick / FPS
    label = labels[tick - 1]
    if abs(label['target_time_monotonic'] - nominal) > 1e-5:
        raise ValueError(f"{case['case_id']}: labels are not aligned to camera time")
    direction = change['direction']
    raw_time = case['event_times']['close_issued_monotonic' if direction == 'close'
                                   else 'release_issued_monotonic']
    return {**change, 'nominal_time_monotonic': nominal,
            'seconds_after_recorded_command': nominal - raw_time,
            'recorded_command_monotonic': raw_time,
            'predicted_fraction': action[6],
            'recorded_target_fraction': label['cup_relative_action'][6],
            **geometry(case, action, contacts)}


def recorded_first(case, draw, direction):
    labels = case['targets'][:HORIZON]
    for i, label in enumerate(labels):
        nominal = draw['request']['source_time'] + (i + 1) / FPS
        if abs(label['target_time_monotonic'] - nominal) > 1e-5:
            raise ValueError(f"{case['case_id']}: expected camera-time labels")
    return first_threshold([label['cup_relative_action'] for label in labels], direction)


def fit_descent(rows):
    """Find the worst dip at any of the first ten points before the raw deadline."""
    values = []
    for row in rows.values():
        times = row['event_times']
        start, deadline = times['hold_start_monotonic'], times['hold_start_monotonic'] + 5.
        if row['source_time'] >= deadline:
            continue
        points = [p for p in row['metrics']['first_10'][:HORIZON]
                  if start <= p['nominal_time_monotonic'] < deadline]
        if not points:
            continue
        peak, peak_tick, max_drop, worst_peak = 0., 0, 0., None
        for point in points:
            delta = point['predicted_normal_delta_mm']
            if peak - delta > max_drop:
                max_drop = peak - delta
                worst_peak = {'tick': point['tick'], 'peak_tick': peak_tick,
                              'nominal_time_monotonic': point['nominal_time_monotonic']}
            if delta > peak:
                peak, peak_tick = delta, point['tick']
        lowest = min(points, key=lambda p: p['predicted_normal_delta_mm'])
        values.append({'case_id': row['case_id'], 'source_index': row['source_index'],
                       'cycle': row['cycle'], 'kind': row['kind'], 'seed': row['seed'],
                       'hold_start_monotonic': start, 'deadline_monotonic': deadline,
                       'points_before_deadline': len(points),
                       'maximum_down_from_observation_mm': max(0., -lowest['predicted_normal_delta_mm']),
                       'lowest_tick': lowest['tick'],
                       'recorded_delta_at_lowest_tick_mm': lowest.get('recorded_normal_delta_mm'),
                       'maximum_drop_from_previous_peak_mm': max_drop,
                       'peak_to_lowest': worst_peak})
    if not values:
        raise ValueError('No true-five points inside the original five-second hold')
    return {'draws': len(values),
            'maximum_down_from_observation': max(values, key=lambda x: x['maximum_down_from_observation_mm']),
            'maximum_drop_from_previous_peak': max(values, key=lambda x: x['maximum_drop_from_previous_peak_mm']),
            'draws_down_at_least_2mm': sum(v['maximum_down_from_observation_mm'] >= 2. for v in values),
            'per_draw': values}


def historical_reference(path):
    if not path:
        return []
    report = read_json(path)
    found = []
    for row in report['results']:
        match = re.search(r'horizontal ([0-9.]+) mm, vertical (-?[0-9.]+) mm', row.get('rejected', ''))
        if (row['group'] == 'heldout_source_7' and match and 'close' in row['rejected']
                and 25. <= float(match.group(1)) <= 27.):
            found.append({'case_id': row['case_id'], 'seed': row['seed'], 'model': row['model'],
                          'horizontal_mm': float(match.group(1)), 'vertical_mm': float(match.group(2)),
                          'source': str(path), 'historical_rejection': row['rejected']})
    return found


def score_heldout(report, role, contacts, historical):
    cases = case_index(report)
    early, late, late_effective, unsupported, normal_close, far, failed, historical_now = ([] for _ in range(8))
    exact = None
    invalid = []
    coverage = {'heldout_draws': 0, 'heldout_valid_feedback_draws': 0,
                'failed_rollout_draws': 0, 'failed_rollout_valid_feedback_draws': 0}
    history_keys = {(x['case_id'], x['seed']) for x in historical}
    for case in cases.values():
        if case['group'] not in ('heldout_source_7', 'failed_trial_evaluation_only'):
            raise ValueError(f"Unexpected evaluation group: {case['group']}")
        for seed, draw in draw_index(case, role).items():
            actions, request = draw['reply']['actions'], draw['request']
            if len(actions) < HORIZON:
                raise ValueError(f"{case['case_id']}: prediction has fewer than ten points")
            group = 'heldout' if case['group'] == 'heldout_source_7' else 'failed_rollout'
            coverage[group + '_draws'] += 1
            changes = effective_changes(request['cup_relative_state'], actions)
            if changes is None:
                invalid.append({'case_id': case['case_id'], 'group': case['group'],
                                'kind': case['kind'], 'cycle': case.get('cycle'), 'seed': seed})
                continue
            coverage[group + '_valid_feedback_draws'] += 1
            identity = {'case_id': case['case_id'], 'cycle': case.get('cycle'),
                        'kind': case['kind'], 'seed': seed}
            if case['group'] != 'heldout_source_7':
                reference = report['heldout_close_reference_cup_relative_xyz_mm']
                for change in changes:
                    if change['direction'] != 'close' or change['previous_percent'] < 88:
                        continue
                    action = actions[change['tick'] - 1]
                    xyz = [action[i] * 1000. for i in range(3)]
                    xy = math.dist(xyz[:2], reference[:2])
                    gap = sum((xyz[i] - reference[i]) * case['initial_cup']['table_normal'][i]
                              for i in range(3))
                    guard_xy = math.dist(xyz[:2], CONTACT_REFERENCE_MM[:2])
                    guard_z = xyz[2] - CONTACT_REFERENCE_MM[2]
                    if guard_xy > 15. or abs(guard_z) > 12.:
                        failed.append({**identity, **change, 'xy_distance_to_heldout_close_reference_mm': xy,
                                       'normal_gap_to_heldout_close_reference_mm': gap,
                                       'guard_reference_xy_mm': guard_xy, 'guard_reference_z_mm': guard_z})
                continue
            details = [event_detail(case, draw, change, contacts) for change in changes]
            kind = case['kind']
            for detail in details:
                direction = detail['direction']
                expected = detail['recorded_target_fraction']
                eligible_close = direction == 'close' and kind in CLOSING_KINDS
                eligible_open = direction == 'open' and kind in HOLDING_KINDS
                if (eligible_close and expected >= OPEN) or (eligible_open and expected < CLOSE):
                    early.append({**identity, **detail})
                contact_gap = detail['relative_to_confirmed_contact_tcp']
                if eligible_open and contact_gap is not None and contact_gap['normal_gap_mm'] > SUPPORT_REFERENCE_TOLERANCE_MM:
                    unsupported.append({**identity, **detail})
                if eligible_close and detail['previous_percent'] >= 88:
                    action = actions[detail['tick'] - 1]
                    guard_xy = math.hypot(action[0] * 1000. - CONTACT_REFERENCE_MM[0],
                                          action[1] * 1000. - CONTACT_REFERENCE_MM[1])
                    guard_z = action[2] * 1000. - CONTACT_REFERENCE_MM[2]
                    if guard_xy > 15. or abs(guard_z) > 12.:
                        far.append({**identity, **detail, 'guard_reference_xy_mm': guard_xy,
                                    'guard_reference_z_mm': guard_z})
            direction = 'close' if kind in CLOSING_KINDS else 'open' if kind in HOLDING_KINDS else None
            if direction:
                expected_tick = recorded_first(case, draw, direction)
                predicted_tick = first_threshold(actions, direction)
                matching = [d for d in details if d['direction'] == direction]
                effective_tick = matching[0]['tick'] if matching else None
                command_key = 'close_issued_monotonic' if direction == 'close' else 'release_issued_monotonic'
                command_time = case['event_times'][command_key]
                if (expected_tick is not None and command_time >= request['source_time']
                        and (effective_tick is None or effective_tick > expected_tick)):
                    late_effective.append({**identity, 'direction': direction, 'recorded_tick': expected_tick,
                                           'first_effective_tick': effective_tick,
                                           'effective_tick_error': (effective_tick - expected_tick
                                                                    if effective_tick is not None else None),
                                           'status': 'late' if matching else 'missing_in_prefix',
                                           'first_effective': matching[0] if matching else None,
                                           'observation_reference': geometry(case, request['cup_relative_state'], contacts)})
                if expected_tick is not None and (predicted_tick is None or predicted_tick > expected_tick):
                    late.append({**identity, 'direction': direction, 'recorded_tick': expected_tick,
                                 'first_effective_tick': effective_tick, 'first_threshold_tick': predicted_tick,
                                 'threshold_tick_error': (predicted_tick - expected_tick
                                                          if predicted_tick is not None else None),
                                 'status': 'late' if predicted_tick is not None else 'missing_in_prefix',
                                 'first_effective': matching[0] if matching else None,
                                 'observation_reference': geometry(case, request['cup_relative_state'], contacts)})
                if kind == 'close_at':
                    error = predicted_tick - expected_tick if predicted_tick is not None and expected_tick is not None else None
                    normal_close.append({**identity, 'recorded_tick': expected_tick,
                                         'first_effective_tick': effective_tick,
                                         'first_below_88_percent_tick': predicted_tick,
                                         'tick_error': error,
                                         'within_zero_to_one_tick_after_recorded': error is not None and 0 <= error <= 1})
            if (case['case_id'], seed) in history_keys:
                matching = [d for d in details if d['direction'] == 'close' and d['previous_percent'] >= 88]
                historical_now.append({**identity, 'first_effective_close': matching[0] if matching else None,
                                       'first_below_88_percent_tick': first_threshold(actions, 'close')})
            if case['case_id'] == HISTORICAL_CASE and seed == HISTORICAL_SEED:
                opening = next((d for d in details if d['direction'] == 'open'), None)
                raw_tick = recorded_first(case, draw, 'open')
                raw_time = case['event_times']['release_issued_monotonic']
                nominal = request['source_time'] + raw_tick / FPS if raw_tick else None
                exact = {**identity, 'observation_camera_monotonic': request['source_time'],
                         'recorded_open_command_monotonic': raw_time,
                         'earliest_recorded_open_nominal_tick': raw_tick,
                         'earliest_recorded_open_nominal_monotonic': nominal,
                         'nominal_open_seconds_after_raw_command': nominal - raw_time if nominal else None,
                         'first_effective_open': opening,
                         'first_at_or_above_90_percent_tick': first_threshold(actions, 'open'),
                         'gemini_visual_at_observation': case.get('timing_aligned', {}).get('state', {}).get('visual'),
                         'gripper_feedback_percent': request['cup_relative_state'][6] * 100.}
    if exact is None:
        raise ValueError('The specified historical return_late case and seed are missing or invalid')
    normal_close.sort(key=lambda x: (x['cycle'], x['seed']))
    expected_normal = {(cycle, seed) for cycle in (1, 2) for seed in (20261105, 20261113)}
    saved_first = {'close': [], 'open': []}
    for row in report['metric_details']:
        if row['group'] != 'heldout_source_7' or row['model'] != role or row['horizon'] != HORIZON:
            continue
        for direction in ('close', 'open'):
            event = row['first_early_effective_' + direction]
            if event is not None:
                saved_first[direction].append({'case_id': row['case_id'], 'seed': row['seed'],
                                               'tick': event['tick']})
    return {'historical_return_late': exact, 'normal_close_at': normal_close,
            'normal_close_at_all_within_tolerance': len(normal_close) == 4 and {(r['cycle'], r['seed']) for r in normal_close} == expected_normal and all(
                r['within_zero_to_one_tick_after_recorded'] for r in normal_close),
            'early_effective_events_vs_recorded_target': early,
            'saved_evaluator_first_early_events': saved_first,
            'late_or_missing_effective_events': late_effective,
            'late_or_missing_threshold_events': late,
            'effective_open_above_confirmed_contact_reference': unsupported if contacts else None,
            'far_close_against_runtime_reference': far,
            'failed_rollout_far_close_evaluation_only': failed,
            'historical_far_close_cases_now': historical_now,
            'coverage': coverage,
            'invalid_gripper_feedback_draws': invalid,
            'counts': {'early_effective': len(early), 'late_or_missing_effective': len(late_effective),
                       'late_or_missing_threshold': len(late),
                       'open_above_confirmed_contact_reference': len(unsupported) if contacts else None,
                       'far_close': len(far), 'failed_rollout_far_close': len(failed),
                       'invalid_gripper_feedback_draws': len(invalid)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline-heldout', required=True)
    parser.add_argument('--baseline-fit', required=True)
    parser.add_argument('--baseline-heldout-role', choices=('parent', 'candidate'), default='candidate')
    parser.add_argument('--baseline-fit-observer', choices=('old', 'raw', 'aligned'), default='aligned')
    parser.add_argument('--candidate-heldout')
    parser.add_argument('--candidate-fit')
    parser.add_argument('--candidate-heldout-role', choices=('parent', 'candidate'))
    parser.add_argument('--candidate-fit-observer', choices=('old', 'raw', 'aligned'))
    parser.add_argument('--heldout-capture', help='Source-7 capture.jsonl with both confirmed contact frames')
    parser.add_argument('--historical-controller', default=str(HISTORICAL_CONTROLLER),
                        help='Saved read-only controller report containing the earlier 26 mm case')
    args = parser.parse_args()
    old_h, old_f = read_json(args.baseline_heldout), read_json(args.baseline_fit)
    task_override = old_h.get('task_override') or ''
    if not any(marker in task_override.replace(' ', '').lower() for marker in ('5秒', '5s', '5seconds')):
        raise ValueError('Expected the saved counterfactual five-second task override')
    if any('4秒' not in c['request']['task'].replace(' ', '')
           for c in case_index(old_h).values() if c['group'] == 'heldout_source_7'):
        raise ValueError('Expected the recorded source-7 four-second instructions')
    checkpoint = validate_models(old_h, args.baseline_heldout_role, old_f, args.baseline_fit_observer)
    contacts = contact_frames(case_index(old_h), args.heldout_capture)
    historical = historical_reference(args.historical_controller)
    result = {
        'schema': 'fr5-cup-boundary-replay-score-v1',
        'scope': {'heldout': 'source 7, two recorded four-second cycles; five-second task is counterfactual when overridden',
                  'train_fit': 'original five-second training demonstrations, not held-out generalization',
                  'failed_rollout': 'evaluation only',
                  'motion': 'recorded-input predictions, not executed motion, contact, support, or task success',
                  'clock': 'camera monotonic plus tick/6 seconds; no inference or execution latency',
                  'effective_gripper': 'integer percent after clamping, changes over one point from last effective target',
                  'event_timing': 'Early effective events disagree with the recorded target; late effective events compare the first effective change against the first nominal recorded target tick.',
                  'gap': 'TCP displacement from recorded reference along the observed table normal; not cup-bottom height or verified support',
                  'support_reference_tolerance_mm': SUPPORT_REFERENCE_TOLERANCE_MM,
                  'normal_close_tolerance': 'first predicted target below 88% is at the recorded target tick or one tick later',
                  'far_close': 'effective closing from at least 88%; the existing fixed runtime reference uses 15 mm XY and 12 mm Z',
                  'task_override': old_h.get('task_override'), 'seeds': old_h['seeds']},
        'historical_far_close_evidence': historical,
        'baseline': {'checkpoint': checkpoint, 'heldout_report': args.baseline_heldout,
                     'fit_report': args.baseline_fit, 'heldout_role': args.baseline_heldout_role,
                     'fit_observer': args.baseline_fit_observer,
                     'heldout_counterfactual_5s': score_heldout(old_h, args.baseline_heldout_role, contacts, historical),
                     'true_five_train_fit': fit_descent(fit_index(old_f, args.baseline_fit_observer))}}
    candidate_requested = any((args.candidate_heldout, args.candidate_fit,
                               args.candidate_heldout_role, args.candidate_fit_observer))
    if candidate_requested:
        new_h = read_json(args.candidate_heldout) if args.candidate_heldout else old_h
        new_f = read_json(args.candidate_fit) if args.candidate_fit else old_f
        role = args.candidate_heldout_role or 'candidate'
        observer = args.candidate_fit_observer or 'aligned'
        new_checkpoint = validate_models(new_h, role, new_f, observer)
        validate_pair(old_h, args.baseline_heldout_role, new_h, role,
                      old_f, args.baseline_fit_observer, new_f, observer)
        result['candidate'] = {'checkpoint': new_checkpoint,
                               'heldout_report': args.candidate_heldout or args.baseline_heldout,
                               'fit_report': args.candidate_fit or args.baseline_fit,
                               'heldout_role': role, 'fit_observer': observer,
                               'heldout_counterfactual_5s': score_heldout(new_h, role, contacts, historical),
                               'true_five_train_fit': fit_descent(fit_index(new_f, observer))}
        result['pairing'] = {'heldout_saved_requests_match': True,
                             'train_fit_input_document_matches': True,
                             'train_fit_recorded_fields_match': True,
                             'train_fit_scope': 'The comparison details do not embed full model requests; '
                                                'the referenced input document and recorded fields are compared.'}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
