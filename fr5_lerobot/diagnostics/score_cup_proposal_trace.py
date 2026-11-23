#!/usr/bin/env python3
"""Compare first and selected complete cup predictions without hardware access."""

import argparse
from collections import Counter
import copy
import math
from pathlib import Path
import statistics

from lerobot_robot_fairino.cup_relative_runtime import check_cup_gripper_change
from evaluate_cup_timing_policy import model_request, read_json, validate_actions, write_new


def geometry(request, actions, normal):
    opening = min(100, max(0, round(request['cup_relative_state'][6] * 100)))
    closed = False
    close_reopen = False
    errors = []
    gap = request['cup_timing']['support_gap_mm']
    support = ((request['cup_relative_state'][:3], normal, gap)
               if gap is not None and request['cup_tracking']['native_contact'] == 1 else None)
    for tick, action in enumerate(actions[:3], 1):
        target = min(100, max(0, round(action[6] * 100)))
        try:
            check_cup_gripper_change(action, opening, target, support_reference=support)
        except RuntimeError as exc:
            errors.append({'tick': tick, 'reason': str(exc)})
        if opening >= 98 and target <= 88:
            closed = True
        if closed and target >= 98:
            close_reopen = True
        if abs(target - opening) > 1:
            opening = target
    pickup = [.1597, .0713, .0958]
    return {'geometry_passed': not errors, 'errors': errors,
            'close_then_reopen_in_three_ticks': close_reopen,
            'opening_percent': [min(100, max(0, round(a[6] * 100))) for a in actions[:3]],
            'progress_toward_pickup_mm': 1000. * (math.dist(request['cup_relative_state'][:3], pickup)
                                                - math.dist(actions[2][:3], pickup))}


def score(inputs, predictions):
    cases = {c['case_id']: c for c in inputs['cases']}
    expected = {(case_id, seed) for case_id in cases for seed in predictions['seeds']}
    actual = {(r['case_id'], r['seed']) for r in predictions['predictions']}
    if expected != actual or len(actual) != len(predictions['predictions']):
        raise ValueError('Predictions do not cover the declared inputs and seeds')
    rows = []
    raw_report = copy.deepcopy(predictions)
    raw_report['selector'] = 'first_model_draw_before_selection'
    raw_report['latency_scope'] = 'first draw model time only; total proposal-call wall time omitted'
    for row, raw in zip(predictions['predictions'], raw_report['predictions']):
        case = cases[row['case_id']]
        probe = dict(case, task_override=predictions['task_override']) if predictions['task_override'] else case
        request = model_request(probe, 'candidate', 'native', profile=predictions['observation_context_profile'])
        if request != row['request']:
            raise ValueError('Prediction differs from the saved causal request')
        reply = row['reply']
        trace = reply.get('cup_proposal_selection')
        draws = trace['proposals'] if trace else [{'actions': reply['actions'], 'inference_seconds': reply['inference_seconds'], 'matched': True}]
        selected = trace['selected_index'] if trace else 0
        if reply['actions'] != draws[selected]['actions'] or len(draws) != row['draw_count']:
            raise ValueError('Selected action channels differ from the recorded complete draw')
        for draw in draws:
            validate_actions(draw['actions'])
        matched = trace['matched'] if trace else True
        first = geometry(request, draws[0]['actions'], case['initial_cup']['table_normal'])
        chosen = geometry(request, reply['actions'], case['initial_cup']['table_normal'])
        rows.append({'case_id': row['case_id'], 'seed': row['seed'], 'group': case['group'],
                     'kind': case['kind'], 'draws': len(draws), 'selected_index': selected,
                     'matched': matched, 'first_selector_matched': draws[0]['matched'],
                     'gripper_feedback_valid': request['cup_relative_state'][7] == 1.,
                     'first': first, 'selected': chosen,
                     'predict_elapsed_seconds': row['predict_elapsed_seconds']})
        raw['reply']['actions'] = draws[0]['actions']
        raw['reply']['inference_seconds'] = draws[0]['inference_seconds']
        for key in ('sampling', 'runtime_source'):
            if key in draws[0]:
                raw['reply'][key] = draws[0][key]
            elif trace:
                raw['reply'].pop(key, None)
        raw['reply'].pop('cup_proposal_selection', None)
        raw.update(draw_count=1, new_draw_selected=False, selection_matched=draws[0]['matched'])
        raw.pop('predict_elapsed_seconds', None)
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        elapsed = [r['predict_elapsed_seconds'] for r in subset]
        valid = [r for r in subset if r['gripper_feedback_valid']]
        summary[group] = {
            'total': len(subset),
            'valid_gripper_inputs': len(valid), 'invalid_gripper_inputs': len(subset) - len(valid),
            'first_geometry_rejected': sum(not r['first']['geometry_passed'] for r in valid),
            'first_selector_rejected': sum(not r['first_selector_matched'] for r in valid),
            'selected_geometry_rejected': sum(not r['selected']['geometry_passed'] for r in valid),
            'exhausted_without_match': sum(not r['matched'] for r in valid),
            'first_close_then_reopen': sum(r['first']['close_then_reopen_in_three_ticks'] for r in valid),
            'selected_close_then_reopen': sum(r['selected']['close_then_reopen_in_three_ticks'] for r in valid),
            'draw_counts': dict(Counter(r['draws'] for r in subset)),
            'predict_elapsed_median_s': statistics.median(elapsed),
            'predict_elapsed_max_s': max(elapsed),
        }
    return {'schema': 'fr5-cup-proposal-trace-score-v1', 'checkpoint': predictions['checkpoint'],
            'seeds': predictions['seeds'], 'summary': summary, 'rows': rows,
            'robot_commands': 0,
            'scope': 'isolated recorded inputs and complete model draws; no IK, contact, support or task success established'}, raw_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-first-output', type=Path, required=True)
    args = parser.parse_args()
    report, raw = score(read_json(args.inputs), read_json(args.predictions))
    write_new(args.output, report)
    write_new(args.raw_first_output, raw)
    import json
    print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
