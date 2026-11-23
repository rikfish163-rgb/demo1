#!/usr/bin/env python3
"""Compare recorded training targets with archived complete cup predictions."""

import argparse
import json
import math
from pathlib import Path
import statistics

from evaluate_cup_timing_policy import read_json, write_new


PICKUP_M = (.1597, .0713, .0958)


def metrics(case, row):
    truth = [target['cup_relative_action'] for target in case['targets_camera_aligned'][:3]]
    predicted = row['reply']['actions'][:3]
    state = row['request']['cup_relative_state']
    if len(truth) != 3 or len(predicted) != 3:
        raise ValueError('Training support requires three recorded target points')
    actual_delta = [(truth[2][i] - state[i]) * 1000. for i in range(3)]
    predicted_delta = [(predicted[2][i] - state[i]) * 1000. for i in range(3)]
    actual_move = math.hypot(*actual_delta)
    predicted_move = math.hypot(*predicted_delta)
    dot = sum(a * b for a, b in zip(actual_delta, predicted_delta))
    opening = lambda actions: [min(100, max(0, round(a[6] * 100.))) for a in actions]
    actual_grip, predicted_grip = opening(truth), opening(predicted)
    selection = row['reply'].get('cup_proposal_selection')
    return {
        'case_id': case['case_id'], 'seed': row['seed'], 'source': case['source_index'],
        'row_index': case['row_index'], 'source_frame_index': case['source_frame_index'],
        'kind': case['kind'], 'phase': row['request']['cup_timing']['phase'],
        'xyz_next3_mae_mm': statistics.mean(abs(predicted[t][i] - truth[t][i]) * 1000.
                                           for t in range(3) for i in range(3)),
        'endpoint_error_mm': math.dist(predicted[2][:3], truth[2][:3]) * 1000.,
        'actual_progress_mm': (math.dist(state[:3], PICKUP_M) - math.dist(truth[2][:3], PICKUP_M)) * 1000.,
        'pred_progress_mm': (math.dist(state[:3], PICKUP_M) - math.dist(predicted[2][:3], PICKUP_M)) * 1000.,
        'actual_move_mm': actual_move, 'pred_move_mm': predicted_move,
        'move_alignment_cosine': dot / (actual_move * predicted_move) if actual_move * predicted_move else None,
        'actual_gripper_pct': actual_grip, 'pred_gripper_pct': predicted_grip,
        'early_close_against_recorded_labels': min(actual_grip) >= 98 and min(predicted_grip) < 88,
        'matched': selection['matched'] if selection else True,
    }


def summarize(rows):
    positive = [r for r in rows if r['actual_progress_mm'] > .05]
    moving = [r for r in rows if r['actual_move_mm'] > .5]
    steady = [r for r in rows if r['actual_move_mm'] <= .2]
    return {
        'draws': len(rows), 'unique_anchors': len({(r['source'], r['row_index']) for r in rows}),
        'matched_false': sum(not r['matched'] for r in rows),
        'positive_label_draws': len(positive),
        'wrong_direction_on_positive_labels': sum(r['pred_progress_mm'] < -.05 for r in positive),
        'recorded_move_over_half_mm_draws': len(moving),
        'opposed_recorded_move_draws': sum(r['move_alignment_cosine'] is not None and
                                         r['move_alignment_cosine'] < 0. for r in moving),
        'recorded_move_at_most_point_two_mm_draws': len(steady),
        'predicted_move_over_two_mm_on_steady_draws': sum(r['pred_move_mm'] > 2. for r in steady),
        'early_close_against_recorded_labels': sum(r['early_close_against_recorded_labels'] for r in rows),
        'median_xyz_next3_mae_mm': statistics.median(r['xyz_next3_mae_mm'] for r in rows),
        'median_endpoint_error_mm': statistics.median(r['endpoint_error_mm'] for r in rows),
        'maximum_endpoint_error_mm': max(r['endpoint_error_mm'] for r in rows),
    }


def score(inputs, saved):
    cases = {case['case_id']: case for case in inputs['cases']}
    if any(c['split'] != 'train' or c['source_index'] not in range(7) for c in cases.values()):
        raise ValueError('Training support contains a held-out or failed-trial observation')
    expected = {(case_id, seed) for case_id in cases for seed in saved['seeds']}
    actual = [(row['case_id'], row['seed']) for row in saved['predictions']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Prediction coverage differs from the recorded cases and seeds')
    rows, unique, aliases = [], {}, []
    for prediction in saved['predictions']:
        case = cases[prediction['case_id']]
        result = metrics(case, prediction)
        key = (result['source'], result['row_index'], result['seed'])
        if key in unique:
            prior = unique[key]
            if {k: v for k, v in prior.items() if k not in ('case_id', 'kind')} != {
                    k: v for k, v in result.items() if k not in ('case_id', 'kind')}:
                raise ValueError('Repeated source-row-seed has different targets or predictions')
            aliases.append({'case_id': result['case_id'], 'same_observation_as': prior['case_id'],
                            'seed': result['seed']})
        else:
            unique[key] = result
        rows.append(result)
    independent = list(unique.values())
    return {'prediction': saved.get('input'), 'checkpoint': saved['checkpoint'],
            'case_counts': summarize(rows), 'unique_source_row_seed': summarize(independent),
            'aliases': aliases, 'rows': independent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prediction', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = read_json(args.inputs)
    reports = {}
    for item in args.prediction:
        label, path = item.split('=', 1)
        reports[label] = score(inputs, read_json(Path(path)))
        reports[label]['prediction'] = path
    report = {'schema': 'fr5-cup-recorded-training-support-score-v1', 'input': str(args.inputs),
              'reports': reports, 'robot_commands': 0, 'task_success_verified': False,
              'scope': 'Recorded observations and labels. Fixed-pickup progress is diagnostic; '
                       'submillimetre progress and operator waits do not establish task success.'}
    write_new(args.output, report)
    print(json.dumps({label: {k: value for k, value in entry.items() if k not in ('rows', 'aliases')}
                      for label, entry in reports.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
