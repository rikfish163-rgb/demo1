#!/usr/bin/env python3
"""Score archived PGI recovery observations and prepare controller replay inputs."""

import argparse
import base64
from bisect import bisect_right
from collections import Counter
import math
from pathlib import Path
from types import SimpleNamespace

from evaluate_cup_timing_policy import (
    controller_input, history_from_captures, model_request, read_json, read_jsonl,
    state_from_capture, validate_actions, write_new,
)
from score_cup_boundary_replay import effective_changes
from score_cup_proposal_trace import score


FPS, CLOSE_PERCENT, OPEN_PERCENT, LATE_TICKS = 6, 88, 98, 1
HORIZONS = (3, 10)


def verify_recovery(inputs, captures, recovery):
    frames = {f['frame_index']: f for f in captures}
    records = {f"{r['case_id']}-recovered-{r['first_new_valid_frame']}": r
               for r in recovery['observations']}
    if set(records) != {c['case_id'] for c in inputs['cases']}:
        raise ValueError('Recovery observations do not cover the input cases')
    observations = []
    for case in inputs['cases']:
        record = records[case['case_id']]
        previous = frames[record['invalid_source_frame']]
        stamp = previous['camera']['source_timestamp_monotonic']
        if previous['gripper_feedback_valid']:
            raise ValueError('Recovery predecessor has valid PGI feedback')
        frame = next(f for f in captures
                     if f['camera']['source_timestamp_monotonic'] > stamp and f['gripper_feedback_valid'])
        if frame['frame_index'] != case['source_frame_index']:
            raise ValueError('Recovery input is not the first subsequent valid PGI frame')
        request, cup = case['request'], case['initial_cup']
        state = state_from_capture(frame, cup['base_xyz_mm'])
        first = frames[case['tracker_start_frame_index']]['camera']['source_timestamp_monotonic']
        history = history_from_captures(captures, request['source_time'], first, cup['base_xyz_mm'])
        image = Path(case['source_root']) / 'continuous' / frame['rgb_file']
        if (request['source_time'] != frame['camera']['source_timestamp_monotonic']
                or request['cup_relative_state'] != state
                or request['state'] != [math.radians(j) for j in frame['robot']['joints']] + state[6:]
                or request['observation_history'] != history
                or request['rgb_jpeg_base64'] != base64.b64encode(image.read_bytes()).decode('ascii')
                or cup['camera_serial'] != frame['camera']['source_camera_serial']):
            raise ValueError(f"{case['case_id']}: recovery request differs from the recorded observation")
        observations.append({'case_id': case['case_id'], 'invalid_source_frame': previous['frame_index'],
                             'invalid_feedback_status': 'unassessed',
                             'first_valid_source_frame': frame['frame_index'],
                             'wait_s': request['source_time'] - stamp,
                             'gripper_motiondone': frame['robot']['gripper_motiondone'],
                             'gripper_position_percent': frame['robot']['gripper_position']})
    return observations


def timing_metrics(request, actions, journal, horizon):
    changes = effective_changes(request['cup_relative_state'], actions[:horizon])
    if changes is None:
        return None
    commands = sorted((r for r in journal if r['executed']), key=lambda r: r['issued_monotonic'])
    times = [r['issued_monotonic'] for r in commands]
    recorded = []
    for tick in range(1, horizon + 1):
        index = bisect_right(times, request['source_time'] + tick / FPS) - 1
        if index < 0:
            raise ValueError('No recorded jaw target at the prediction clock')
        recorded.append(round(commands[index]['target'][6]))
    early = {'close': [], 'open': []}
    for change in changes:
        target = recorded[change['tick'] - 1]
        if ((change['direction'] == 'close' and change['previous_percent'] >= CLOSE_PERCENT
             and target >= OPEN_PERCENT)
                or (change['direction'] == 'open' and change['previous_percent'] < OPEN_PERCENT
                    and target < CLOSE_PERCENT)):
            early[change['direction']].append({**change, 'recorded_target_percent': target})
    first_open = next((i for i, a in enumerate(actions[:horizon], 1)
                       if round(max(0., min(1., a[6])) * 100) >= OPEN_PERCENT), None)
    expected = next((i for i, target in enumerate(recorded, 1) if target >= OPEN_PERCENT), None)
    late = {'assessed': False, 'late_or_missing': None, 'first_full_open_tick': first_open}
    if round(request['cup_relative_state'][6] * 100) >= OPEN_PERCENT:
        late['status'] = 'already_open_at_observation'
    elif expected is None or expected + LATE_TICKS > horizon:
        late['status'] = 'release_deadline_outside_prefix'
    else:
        late.update(assessed=True, expected_open_tick=expected, deadline_tick=expected + LATE_TICKS,
                    late_or_missing=first_open is None or first_open > expected + LATE_TICKS,
                    status='missing_in_prefix' if first_open is None else 'predicted_in_prefix')
    return {'early_close': early['close'], 'early_open': early['open'], 'late_open': late}


def evaluate(inputs, predictions, journal):
    cases = {c['case_id']: c for c in inputs['cases']}
    if (not cases or len(cases) != len(inputs['cases'])
            or any(c['group'] != 'heldout_source_7' or c['kind'] != 'feedback_recovered'
                   for c in cases.values())):
        raise ValueError('Expected unique held-out PGI recovery cases')
    seeds = predictions['seeds']
    rows = predictions['predictions']
    expected = {(case_id, seed) for case_id in cases for seed in seeds}
    actual = {(r['case_id'], r['seed']) for r in rows}
    if not seeds or len(set(seeds)) != len(seeds) or expected != actual or len(actual) != len(rows):
        raise ValueError('Predictions do not cover the recovery cases and unique seeds')
    valid_ids = {k for k, c in cases.items() if c['request']['cup_relative_state'][7] == 1.}
    for row in rows:
        trace = row['reply'].get('cup_proposal_selection')
        draws = trace['proposals'] if trace else [row['reply']]
        if not 1 <= len(draws) <= 3 or row['draw_count'] != len(draws):
            raise ValueError('Expected at most three complete proposals')
        if trace and (trace['draws'] != len(draws) or trace['matched'] != row['selection_matched']):
            raise ValueError('Recorded proposal matched status or draw count differs')
        if row['case_id'] not in valid_ids:
            case = cases[row['case_id']]
            probe = dict(case, task_override=predictions['task_override']) if predictions['task_override'] else case
            if row['request'] != model_request(probe, 'candidate', 'native',
                                               profile=predictions['observation_context_profile']):
                raise ValueError('Prediction differs from the saved causal request')
            for draw in draws:
                validate_actions(draw['actions'])
    trace_report, raw = score({**inputs, 'cases': [c for c in cases.values() if c['case_id'] in valid_ids]},
                              {**predictions, 'predictions': [r for r in rows if r['case_id'] in valid_ids]})
    trace_index = {(r['case_id'], r['seed']): r for r in trace_report['rows']}
    first_index = {(r['case_id'], r['seed']): r for r in raw['predictions']}
    results = []
    for row in rows:
        key = row['case_id'], row['seed']
        valid = row['case_id'] in valid_ids
        selected = row['reply']['actions']
        item = {'case_id': row['case_id'], 'seed': row['seed'],
                'source_frame_index': cases[row['case_id']]['source_frame_index'],
                'status': 'assessed' if valid else 'unassessed',
                'reported_matched': row['selection_matched'],
                'matched_source': ('proposal_trace' if row['reply'].get('cup_proposal_selection')
                                   else 'single_draw_without_selector'),
                'draw_count': row['draw_count'], 'selected_first_three': selected[:3],
                'first': None, 'selected': None}
        if valid:
            first = first_index[key]['reply']['actions']
            item['first_selected_complete_actions_equal'] = first == selected
            for name, actions in (('first', first), ('selected', selected)):
                item[name] = {'geometry': trace_index[key][name],
                              'timing': {f'prefix_{h}': timing_metrics(row['request'], actions, journal, h)
                                         for h in HORIZONS}}
        results.append(item)
    assessed = [r for r in results if r['status'] == 'assessed']
    summary = {'total': len(results), 'assessed': len(assessed), 'unassessed': len(results) - len(assessed),
               'reported_matched': sum(r['reported_matched'] for r in assessed),
               'trace_matched': sum(r['reported_matched'] and r['matched_source'] == 'proposal_trace'
                                    for r in assessed),
               'single_draw_without_selector': sum(r['matched_source'] == 'single_draw_without_selector'
                                                   for r in assessed),
               'draw_counts': dict(Counter(r['draw_count'] for r in results)),
               'first_selected_complete_actions_equal': sum(r['first_selected_complete_actions_equal']
                                                            for r in assessed)}
    for name in ('first', 'selected'):
        summary[name] = {}
        for h in HORIZONS:
            metrics = [r[name]['timing'][f'prefix_{h}'] for r in assessed]
            summary[name][f'prefix_{h}'] = {
                **{f'early_{direction}_draws': sum(bool(m[f'early_{direction}']) for m in metrics)
                   for direction in ('close', 'open')},
                **{f'early_{direction}_events': sum(len(m[f'early_{direction}']) for m in metrics)
                   for direction in ('close', 'open')},
                'late_open_assessed': sum(m['late_open']['assessed'] for m in metrics),
                'late_or_missing_open': sum(m['late_open']['late_or_missing'] is True for m in metrics),
                'late_open_status': dict(Counter(m['late_open']['status'] for m in metrics))}
    return {'checkpoint': predictions['checkpoint'], 'task_override': predictions['task_override'],
            'seeds': seeds, 'summary': summary, 'proposal_trace_summary': trace_report['summary'], 'rows': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--five-second-predictions', type=Path, required=True)
    parser.add_argument('--original-predictions', type=Path, required=True)
    parser.add_argument('--recovery-record', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    inputs = read_json(args.inputs)
    source = Path(inputs['source_root'])
    capture, journal = source / 'continuous/capture.jsonl', source / 'target-command-journal.json'
    recovery = verify_recovery(inputs, read_jsonl(capture), read_json(args.recovery_record))
    paths = {'five_second': args.five_second_predictions, 'original': args.original_predictions}
    predictions = {label: read_json(path) for label, path in paths.items()}
    for field in ('checkpoint', 'seeds', 'model', 'timing_mode', 'observation_context_profile'):
        if predictions['five_second'][field] != predictions['original'][field]:
            raise ValueError(f'Paired prediction {field} differs')
    recorded_journal = read_json(journal)
    scored = {label: evaluate(inputs, saved, recorded_journal) for label, saved in predictions.items()}
    args.output_dir.mkdir()
    for label, path in paths.items():
        output = args.output_dir / f'{label}-controller-input.json'
        controller_input(SimpleNamespace(input=args.inputs, predictions=path, output=output))
        scored[label]['prediction_source'] = str(path)
        scored[label]['controller_input'] = str(output)
    report = {'schema': 'fr5-cup-feedback-recovery-score-v1', 'input': str(args.inputs),
              'case_count': len(inputs['cases']), 'recovery_record': str(args.recovery_record),
              'capture': str(capture), 'journal': str(journal), 'recovery_observations': recovery,
              'prompts': scored, 'execution_steps': 3, 'robot_connections': 0, 'robot_commands': 0,
              'training_ready': False, 'task_success_verified': False,
              'clock': 'recorded camera monotonic plus tick/6 seconds; original executed jaw targets',
              'timing': 'integer percent, clamped to 0–100, one-point deadband; late opening permits one tick',
              'scope': 'independent held-out PGI recovery supplement; isolated prefixes and recorded targets'}
    write_new(args.output_dir / 'summary.json', report)
    import json
    print(json.dumps({label: value['summary'] for label, value in scored.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
