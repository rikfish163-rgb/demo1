#!/usr/bin/env python3
"""Score close, release and hover in the five recorded training cycles.

The original camera targets remain unchanged. Command deadlines come from the
original journal; contact comes from valid PGI feedback. Three-point execution
prefixes and ten-point diagnostics are reported separately. This is an offline
training-fit score, not held-out performance or evidence of robot task success.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True

if __package__:
    from . import evaluate_cup_setdown_gate as gate
    from . import replay_true_five_second_pi05 as fit
else:
    import evaluate_cup_setdown_gate as gate
    import replay_true_five_second_pi05 as fit

HORIZONS = (3, gate.FULL_HORIZON)
KINDS = (
    'near_early', 'near_late', 'contact_approach', 'close_before', 'close_at',
    'close_after', 'contact_on_table', 'hover_early', 'hover_middle', 'hover_late',
    'deadline_minus_1s', 'deadline_before', 'deadline_after', 'return_early',
    'return_middle', 'return_late', 'release_before', 'release_at', 'release_after',
    'retreat',
)
MIN_SUPPORT_AGE_S = .30
MAX_SUPPORT_GAP_S = .35
GEOMETRY_CODES = {
    'no_feasible_complete_proposal', 'far_close', 'open_above_support_proxy',
    'open_below_support_proxy', 'open_while_target_moving',
}


def same_time(actual, expected, label):
    if not math.isfinite(actual) or abs(actual - expected) > 1e-6:
        raise ValueError(f'{label}: original timestamp differs')


def cases_index(inputs):
    if inputs.get('schema') != fit.INPUT_SCHEMA:
        raise ValueError('Expected the original five-second train-fit inputs')
    cases, cycles = {}, defaultdict(list)
    for case in inputs['cases']:
        source, cycle = case['source_index'], case['cycle']
        if (case['group'] != 'train_fit' or case['split'] != 'train'
                or source not in fit.SOURCES or not 1 <= cycle <= fit.SOURCES[source]):
            raise ValueError('Expected only original training sources 0, 3, 4 and 5')
        if case['case_id'] in cases:
            raise ValueError('Duplicate train-fit case')
        cases[case['case_id']] = case
        cycles[source, cycle].append(case)
    expected = {(source, cycle) for source, count in fit.SOURCES.items()
                for cycle in range(1, count + 1)}
    if len(cases) != 100 or set(cycles) != expected:
        raise ValueError('Expected 100 original observations in five training cycles')
    sources = {s['source_index']: s for s in inputs['sources']}
    if len(sources) != len(inputs['sources']) or set(sources) != set(fit.SOURCES):
        raise ValueError('Original five-second source metadata differs')
    for (source, cycle), items in cycles.items():
        if Counter(c['kind'] for c in items) != Counter(KINDS):
            raise ValueError(f'Source {source} cycle {cycle}: expected all 20 recorded kinds')
        events = items[0]['event_times']
        if any(c['event_times'] != events or c['source_root'] != sources[source]['source_root']
               for c in items):
            raise ValueError('Cases in one cycle have different original events or roots')
        hold = next(h for h in sources[source]['hold_records'] if h['repetition'] == cycle)
        if events['requested_hold_seconds'] != 5. or hold['requested_seconds'] != 5.:
            raise ValueError('Expected an original requested five-second hold')
        for key, value in (
            ('hold_start_monotonic', hold['start_monotonic']),
            ('hold_end_monotonic', hold['end_monotonic']),
            ('five_second_deadline_monotonic', hold['start_monotonic'] + 5.),
            ('return_issued_monotonic', hold['end_monotonic']),
            ('hold_record_actual_seconds', hold['actual_duration_s']),
        ):
            same_time(events[key], value, f'Source {source} cycle {cycle} {key}')
    return cases


def prediction_index(cases, saved):
    profile, seeds = saved['observation_context_profile'], saved['seeds']
    if (saved.get('schema') != fit.PREDICTION_SCHEMA or saved['model'] != 'setdown'
            or saved['observers'] != ['setdown'] or profile not in gate.SETDOWN_PROFILES
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError('Expected saved setdown predictions with distinct seeds')
    rows = {(r['case_id'], r['seed']): r for r in saved['predictions']}
    if len(rows) != len(saved['predictions']) or set(rows) != {
            (case_id, seed) for case_id in cases for seed in seeds}:
        raise ValueError('Predictions do not cover the exact original cases and seeds')
    for (case_id, _), row in rows.items():
        case = cases[case_id]
        request = fit.model_request(case, 'setdown', profile)
        if row['observer'] != 'setdown':
            raise ValueError(f'{case_id}: saved observer differs')
        for key in ('task', 'source_time', 'cup_tracking', 'cup_timing'):
            if row[key] != request[key]:
                raise ValueError(f'{case_id}: saved request differs in {key}')
        if 'request' in row and row['request'] != request:
            raise ValueError(f'{case_id}: saved full request differs')
        fit.validate_actions(row['reply'])
        selection = row['reply'].get('cup_proposal_selection')
        if selection:
            proposals = selection['proposals']
            if not 1 <= len(proposals) <= 3 or selection['draws'] != len(proposals):
                raise ValueError(f'{case_id}: expected at most three complete proposals')
            for proposal in proposals:
                fit.validate_actions({**row['reply'], 'actions': proposal['actions']})
            selected = selection['selected_index']
            if not 0 <= selected < len(proposals) or proposals[selected]['actions'] != row['reply']['actions']:
                raise ValueError(f'{case_id}: selected complete proposal differs')
    return rows


def source_records(frames, journal):
    frame_map = {f['frame_index']: f for f in frames}
    times = [f['camera']['source_timestamp_monotonic'] for f in frames]
    if len(frame_map) != len(frames) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Original Gemini frames must be unique and time ordered')
    initial = next(r for r in journal if r['kind'] == 'hold')
    commands = [initial] + [r for r in journal if r['kind'] == 'gripper' and r.get('executed')]
    command_times = [r['issued_monotonic'] for r in commands]
    if any(a >= b for a, b in zip(command_times, command_times[1:])):
        raise ValueError('Original jaw commands must be time ordered')
    return {'frames': frames, 'frame_map': frame_map, 'times': times,
            'journal': journal, 'commands': commands, 'command_times': command_times}


def latched_command(records, stamp):
    index = bisect_right(records['command_times'], stamp) - 1
    if index < 0:
        raise ValueError('No recorded jaw target at the requested timestamp')
    command = records['commands'][index]
    return {'sequence': command['sequence'], 'issued_monotonic': command['issued_monotonic'],
            'jaw_percent': command['target'][6],
            'source': 'initial_hold' if command['kind'] == 'hold' else 'MoveGripper'}


def contact(frame):
    return bool(frame['gripper_feedback_valid'] and frame['robot']['gripper_motiondone'] == 2)


def support_reference(case, records):
    events, normal = case['event_times'], gate.normal_of(case)
    commands = [r for r in records['journal'] if r.get('executed')
                and r.get('teacher', {}).get('cycle') == case['cycle']
                and r['teacher'].get('phase') in ('close', 'return', 'release')]
    if Counter(r['teacher']['phase'] for r in commands) != Counter(('close', 'return', 'release')):
        raise ValueError(f"{case['case_id']}: incomplete original close/return/release journal")
    for command in commands:
        phase = command['teacher']['phase']
        same_time(command['issued_monotonic'], events[f'{phase}_issued_monotonic'],
                  f"{case['case_id']} {phase}")
        if ((phase == 'close' and command['target'][6] >= gate.CLOSE_PERCENT)
                or (phase == 'release' and command['target'][6] < gate.OPEN_PERCENT)):
            raise ValueError('Original close/release jaw command differs')
    confirmed = records['frame_map'].get(events['contact_confirmed_source_frame'])
    if confirmed is None:
        return {'status': 'unassessed', 'reason': 'missing_original_contact_frame'}
    same_time(confirmed['camera']['source_timestamp_monotonic'], events['contact_confirmed_monotonic'],
              f"{case['case_id']} contact")
    if not contact(confirmed):
        return {'status': 'unassessed', 'reason': 'invalid_original_contact_feedback'}
    release = events['release_issued_monotonic']
    eligible = [f for f in records['frames']
                if release - .5 <= f['camera']['source_timestamp_monotonic'] < release
                and contact(f) and abs(f['robot']['tcp_speed'][0]) <= gate.SPEED_MM_S]
    heights = [gate.point_projection(f['robot']['tcp'], normal) for f in eligible]
    if (len(eligible) < 2 or eligible[-1]['camera']['source_timestamp_monotonic']
            - eligible[0]['camera']['source_timestamp_monotonic'] < .1
            or release - eligible[-1]['camera']['source_timestamp_monotonic'] > .25
            or max(heights) - min(heights) > .5):
        return {'status': 'unassessed', 'reason': 'missing_stable_pre_release_contact_reference'}
    height = statistics.median(heights)
    ready = [f for f in records['frames']
             if events['return_issued_monotonic'] <= f['camera']['source_timestamp_monotonic'] < release
             and contact(f) and abs(f['robot']['tcp_speed'][0]) <= gate.SPEED_MM_S
             and abs(gate.point_projection(f['robot']['tcp'], normal) - height) <= gate.NEAR_MM]
    pair = next(((a, b) for a, b in zip(ready, ready[1:])
                 if 0 < b['camera']['source_timestamp_monotonic']
                 - a['camera']['source_timestamp_monotonic'] <= MAX_SUPPORT_GAP_S), None)
    if pair is None:
        return {'status': 'unassessed', 'reason': 'missing_stable_contact_pair'}
    ready_time = pair[1]['camera']['source_timestamp_monotonic']
    return {'status': 'assessed', 'height_projection_mm': height,
            'confirmed_contact_frame': confirmed['frame_index'],
            'pre_release_frame_indices': [f['frame_index'] for f in eligible],
            'pre_release_height_spread_mm': max(heights) - min(heights),
            'ready_pair_frame_indices': [f['frame_index'] for f in pair],
            'recorded_first_ready_proxy_monotonic': ready_time,
            'recorded_release_command_monotonic': release,
            'seconds_ready_proxy_before_command': release - ready_time}


def case_evidence(case, records, support):
    frame = gate.check_capture(case, records['frame_map'])
    stamp, robot = case['request']['source_time'], frame['robot']
    state, observer = case['request']['cup_relative_state'], case['observers']['setdown']
    if frame['gripper_feedback_valid'] and abs(state[6] * 100. - robot['gripper_position']) > 1e-5:
        raise ValueError(f"{case['case_id']}: raw PGI jaw feedback differs")
    same_time(observer['timing']['camera_monotonic'], stamp, f"{case['case_id']} observer camera")
    same_time(observer['timing']['feedback_monotonic'], robot['received_monotonic'],
              f"{case['case_id']} observer feedback")
    native_contact = 1 if contact(frame) else 0 if frame['gripper_feedback_valid'] else None
    if observer['tracking']['native_contact'] != native_contact:
        raise ValueError(f"{case['case_id']}: observer PGI contact differs from raw feedback")
    label_mismatches = []
    for tick, target in enumerate(case['targets'], 1):
        nominal = stamp + tick / gate.FPS
        same_time(target['nominal_time_monotonic'], nominal, f"{case['case_id']} target {tick}")
        index = bisect_right(records['times'], nominal) - 1
        raw = records['frame_map'][target['raw_camera_frame']]
        raw_time = raw['camera']['source_timestamp_monotonic']
        if target['tick'] != tick or index < 0 or records['frames'][index]['frame_index'] != raw['frame_index']:
            raise ValueError(f"{case['case_id']}: original camera target selection differs")
        same_time(target['raw_camera_time_monotonic'], raw_time, f"{case['case_id']} target camera")
        same_time(target['raw_camera_age_s'], nominal - raw_time, f"{case['case_id']} target camera age")
        same_time(target['raw_robot_time_monotonic'], raw['robot']['received_monotonic'],
                  f"{case['case_id']} target feedback")
        action = target['raw_cup_relative_action']
        if len(action) != 7 or not all(math.isfinite(v) for v in action):
            raise ValueError('Expected finite original seven-dimensional camera targets')
        if math.dist(gate.action_tcp(case, action), raw['robot']['tcp'][:3]) > .01:
            raise ValueError(f"{case['case_id']}: original target TCP differs")
        command = latched_command(records, nominal)
        if tick <= gate.FULL_HORIZON and abs(action[6] * 100. - command['jaw_percent']) > 1e-5:
            label_mismatches.append({'tick': tick, 'label_jaw_percent': action[6] * 100.,
                                     'nominal_command_jaw_percent': command['jaw_percent'],
                                     'raw_camera_age_s': nominal - raw_time})
    timing = observer['model_timing']
    pair, gap = None, None
    if support['status'] == 'assessed':
        normal = gate.normal_of(case)
        gap = gate.point_projection(robot['tcp'], normal) - support['height_projection_mm']
        index = bisect_right(records['times'], stamp) - 1
        if index > 0:
            previous = records['frames'][index - 1]
            if (0 < stamp - records['times'][index - 1] <= MAX_SUPPORT_GAP_S
                    and all(contact(f) and abs(f['robot']['tcp_speed'][0]) <= gate.SPEED_MM_S
                            and abs(gate.point_projection(f['robot']['tcp'], normal)
                                    - support['height_projection_mm']) <= gate.NEAR_MM
                            for f in (previous, frame))):
                pair = [previous['frame_index'], frame['frame_index']]
    age, timing_gap = timing['setdown_age_s'], timing['support_gap_mm']
    near = (contact(frame) and gap is not None and abs(gap) <= gate.NEAR_MM
            and abs(robot['tcp_speed'][0]) <= gate.SPEED_MM_S
            and timing['phase'] in ('returning', 'at_table_height')
            and timing['visual'] in ('table_height', 'missing')
            and timing_gap is not None and abs(timing_gap) <= gate.NEAR_MM)
    ready = bool(near and pair and age is not None and age >= MIN_SUPPORT_AGE_S)
    command = latched_command(records, stamp)
    return {'frame': frame, 'near': near, 'observations': {
        'raw_camera_monotonic': stamp, 'raw_feedback_monotonic': robot['received_monotonic'],
        'valid_gripper_feedback': bool(frame['gripper_feedback_valid']),
        'raw_feedback_contact': contact(frame), 'gripper_motiondone': robot['gripper_motiondone'],
        'feedback_jaw_percent': robot['gripper_position'] if frame['gripper_feedback_valid'] else None,
        'latched_jaw_command_at_observation': command,
        'contact_excluded_by_release_clock': bool(contact(frame)
                                                 and stamp >= case['event_times']['release_issued_monotonic']),
        'visual': timing['visual'], 'phase': timing['phase'], 'support_gap_mm': timing_gap,
        'raw_support_height_gap_mm': gap, 'setdown_age_s': age,
        'stable_contact_pair_frame_indices': pair, 'observed_ready_proxy': ready,
        'model_hover_seconds': timing['hover_seconds'],
        'original_five_second_label_deadline_monotonic': case['event_times']['five_second_deadline_monotonic'],
        'seconds_to_original_five_second_label_deadline':
            case['event_times']['five_second_deadline_monotonic'] - stamp,
        'label_vs_nominal_command_mismatches_first_10': label_mismatches,
    }}


def check_status(status, reason=None, **values):
    return {'status': status, **({'reason': reason} if reason else {}), **values}


def deadline_check(case, actions, horizon, direction, ready):
    kinds = ('close_before', 'close_at', 'close_after') if direction == 'close' else (
        'release_before', 'release_at', 'release_after')
    if case['kind'] not in kinds:
        return check_status('not_applicable')
    start = case['request']['source_time']
    phase = 'close' if direction == 'close' else 'release'
    expected = gate.target_tick(start, case['event_times'][f'{phase}_issued_monotonic'])
    deadline = expected + gate.LATE_TICKS
    threshold = (lambda a: gate.percent(a) < gate.CLOSE_PERCENT) if direction == 'close' else (
        lambda a: gate.percent(a) >= gate.OPEN_PERCENT)
    first = next((i for i, a in enumerate(actions[:horizon], 1) if threshold(a)), None)
    label = next((i for i, t in enumerate(case['targets'], 1)
                  if threshold(t['raw_cup_relative_action'])), None)
    values = {'original_command_tick': expected, 'deadline_tick': deadline,
              'original_label_tick': label,
              'original_label_deadline_tick': label + gate.LATE_TICKS if label is not None else None,
              'first_predicted_tick': first, 'late_or_missing': None}
    if not case['request']['cup_relative_state'][7]:
        return check_status('unassessed', 'invalid_gripper_feedback', **values)
    if deadline > horizon:
        return check_status('unassessed', f'{phase}_deadline_outside_prefix', **values)
    if len(case['targets']) < deadline:
        return check_status('unassessed', 'recorded_targets_shorter_than_deadline', **values)
    if direction == 'open' and not ready:
        return check_status('unassessed', 'observation_lacks_stable_contact_ready_proxy', **values)
    values['late_or_missing'] = first is None or first > deadline
    return check_status('assessed', **values)


def score_draw(case, row, evidence, support, horizon, profile):
    request = fit.model_request(case, 'setdown', profile)
    adapted = {**case, 'request': request,
               'timing_aligned': copy.deepcopy(case['observers']['setdown']['timing']),
               'targets_camera_aligned': [
                   {'target_time_monotonic': t['nominal_time_monotonic'],
                    'cup_relative_action': list(t['raw_cup_relative_action'])} for t in case['targets']]}
    actions, frame = row['reply']['actions'], evidence['frame']
    observed = copy.deepcopy(evidence['observations'])
    valid, supported = observed['valid_gripper_feedback'], support['status'] == 'assessed'
    violations = []
    geometry = check_status('assessed')
    if not valid or not supported:
        geometry = check_status('unassessed', 'invalid_gripper_feedback' if not valid else support['reason'])
    else:
        base = gate.score_draw(adapted, {**row, 'request': request}, frame, support, horizon)
        violations = [v for v in base['violations'] if v['code'] in GEOMETRY_CODES]
        observed['setdown_gate_recorded_contact'] = base['observations']['recorded_contact']
    opening = check_status('assessed')
    points = gate.predicted_points(adapted, actions, horizon, support if supported else None)
    if not valid or not supported:
        opening = copy.deepcopy(geometry)
    elif not observed['raw_feedback_contact']:
        opening = check_status('not_applicable', 'no_recorded_contact_at_observation')
    else:
        since = (request['source_time'] - observed['setdown_age_s']
                 if evidence['near'] and observed['setdown_age_s'] is not None else None)
        last, held = round(request['cup_relative_state'][6] * 100.), True
        for point in points:
            if point['predicted_ready_proxy']:
                if since is None:
                    since = point['nominal_time_monotonic']
                age = point['nominal_time_monotonic'] - since
            else:
                since, age = None, None
            point['predicted_support_age_s'] = age
            point['predicted_ready_with_age_proxy'] = bool(age is not None and age >= MIN_SUPPORT_AGE_S)
            target = point['jaw_percent']
            if abs(target - last) <= gate.DEAD_BAND:
                continue
            if target > last and last < gate.OPEN_PERCENT and held:
                gap, speed = point['support_height_gap_mm'], point['target_speed_mm_s']
                code = ('open_above_support_proxy' if gap > gate.NEAR_MM else
                        'open_below_support_proxy' if gap < -gate.NEAR_MM else
                        'open_while_target_moving' if speed > gate.SPEED_MM_S else
                        'open_before_stable_support_proxy' if not point['predicted_ready_with_age_proxy'] else None)
                if code and not any(v['code'] == code and v.get('tick') == point['tick'] for v in violations):
                    violations.append({'code': code, 'tick': point['tick'],
                                       'support_height_gap_mm': gap, 'target_speed_mm_s': speed,
                                       'predicted_support_age_s': age, 'from_percent': last, 'to_percent': target})
                if target >= gate.OPEN_PERCENT:
                    held = False
            last = target
    normal_close = deadline_check(case, actions, horizon, 'close', True)
    late_open = deadline_check(case, actions, horizon, 'open', observed['observed_ready_proxy'])
    for check, code in ((normal_close, 'late_normal_close'), (late_open, 'late_supported_open')):
        if check.get('late_or_missing') is True:
            violations.append({'code': code, 'deadline_tick': check['deadline_tick'],
                               'first_predicted_tick': check['first_predicted_tick']})
    events = case['event_times']
    hover_points = [p for p in points if events['hold_start_monotonic'] <= p['nominal_time_monotonic']
                    < events['five_second_deadline_monotonic']]
    hover = check_status('not_applicable', 'no_prediction_in_original_five_second_hold')
    if hover_points:
        if not valid or not observed['raw_feedback_contact']:
            hover = check_status('unassessed', 'no_valid_recorded_contact_for_hover')
        else:
            initial = gate.point_projection(gate.action_tcp(case, request['cup_relative_state']), gate.normal_of(case))
            peak, down = 0., 0.
            for point in hover_points:
                delta = gate.point_projection(gate.action_tcp(case, actions[point['tick'] - 1]), gate.normal_of(case)) - initial
                down, peak = max(down, -delta, peak - delta), max(peak, delta)
            hover = check_status('assessed', ticks=[p['tick'] for p in hover_points], max_down_mm=down,
                                 original_label_deadline_monotonic=events['five_second_deadline_monotonic'])
            if down >= gate.DESCENT_MM:
                violations.append({'code': 'early_true_five_second_descent', 'max_down_mm': down})
    checks = {'geometry': geometry, 'early_open': opening, 'normal_close': normal_close,
              'late_open': late_open, 'hover': hover}
    unassessed = any(c['status'] == 'unassessed' for c in checks.values())
    return {'case_id': case['case_id'], 'source_index': case['source_index'], 'cycle': case['cycle'],
            'kind': case['kind'], 'group': case['group'], 'split': case['split'], 'seed': row['seed'],
            'status': 'unassessed' if unassessed else 'assessed',
            'gate': 'fail' if violations else 'inconclusive' if unassessed else 'pass',
            'checks': checks, 'violations': violations, 'observations': observed, 'points': points}


def counts(rows):
    failures = Counter(code for row in rows for code in {v['code'] for v in row['violations']})
    return {'draws': len(rows), 'assessed': sum(r['status'] == 'assessed' for r in rows),
            'unassessed': sum(r['status'] == 'unassessed' for r in rows),
            'passed': sum(r['gate'] == 'pass' for r in rows),
            'failure_draws': sum(bool(r['violations']) for r in rows),
            'failure_counts': dict(sorted(failures.items())),
            'late_open_assessable': sum(r['checks']['late_open']['status'] == 'assessed' for r in rows),
            'checks': {name: dict(Counter(r['checks'][name]['status'] for r in rows))
                       for name in ('geometry', 'early_open', 'normal_close', 'late_open', 'hover')},
            'unassessed_reasons': dict(sorted(Counter(c['reason'] for r in rows for c in r['checks'].values()
                                                     if c['status'] == 'unassessed').items()))}


def score(inputs, saved, raw_sources):
    cases = cases_index(inputs)
    predictions = prediction_index(cases, saved)
    records = {root: source_records(*raw_sources[root]) for root in {c['source_root'] for c in cases.values()}}
    support, evidence = {}, {}
    for case in cases.values():
        key = (case['source_index'], case['cycle'])
        if key not in support:
            support[key] = support_reference(case, records[case['source_root']])
        evidence[case['case_id']] = case_evidence(case, records[case['source_root']], support[key])
    report = {'schema': 'fr5-five-second-training-timing-fit-score-v1',
              'checkpoint': saved['checkpoint'], 'seeds': saved['seeds'],
              'observation_context_profile': saved['observation_context_profile'],
              'group': 'train_fit', 'split': 'train', 'case_count': len(cases),
              'robot_commands': 0, 'task_success_verified': False,
              'support_references': [{'source_index': key[0], 'cycle': key[1], **value}
                                     for key, value in sorted(support.items())],
              'original_label_vs_nominal_command_mismatch_points_first_10': sum(
                  len(e['observations']['label_vs_nominal_command_mismatches_first_10']) for e in evidence.values()),
              'contact_excluded_by_release_clock_cases': sum(
                  e['observations']['contact_excluded_by_release_clock'] for e in evidence.values()),
              'limitations': [
                  'Original training observations; these scores do not measure held-out generalization or task success.',
                  'Support uses valid PGI contact, recorded table-normal TCP height, a stable raw feedback pair '
                  'and Gemini table-height or occluded support-age state. It does not prove cup-bottom support.',
                  'Predictions are complete 50 x 7 blocks; only prefixes of three and ten points are scored '
                  'at nominal 6 Hz, without robot execution or measured controller timing.',
                  'Saved fit rows retain task, camera time, tracking and timing; full RGB/state requests '
                  'are checked only when embedded in the saved row.',
              ]}
    for horizon in HORIZONS:
        rows = [score_draw(cases[case_id], row, evidence[case_id],
                           support[cases[case_id]['source_index'], cases[case_id]['cycle']],
                           horizon, saved['observation_context_profile'])
                for (case_id, _), row in sorted(predictions.items())]
        groups = defaultdict(list)
        for row in rows:
            groups[row['source_index'], row['cycle'], row['kind']].append(row)
        summary = counts(rows)
        report[f'prefix_{horizon}'] = {
            'horizon': horizon,
            'gate': 'fail' if summary['failure_draws'] else 'inconclusive' if summary['unassessed'] else 'pass',
            'summary': summary,
            'by_source_cycle_kind': [{'source_index': s, 'cycle': c, 'kind': k, **counts(values)}
                                     for (s, c, k), values in sorted(groups.items())],
            'rows': rows}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs, saved = gate.read_json(args.inputs), gate.read_json(args.predictions)
    cases = cases_index(inputs)
    remote = '4090:' if args.inputs.startswith('4090:/') else ''
    raw = {root: (gate.read_jsonl(remote + root + '/continuous/capture.jsonl'),
                  gate.read_json(remote + root + '/target-command-journal.json'))
           for root in {c['source_root'] for c in cases.values()}}
    report = score(inputs, saved, raw)
    report.update(inputs=args.inputs, predictions=args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), **{
        f'prefix_{h}': {k: v for k, v in report[f'prefix_{h}'].items()
                      if k in ('gate', 'summary')} for h in HORIZONS}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
