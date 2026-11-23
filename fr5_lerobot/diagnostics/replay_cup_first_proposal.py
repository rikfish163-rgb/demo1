#!/usr/bin/env python3
"""Replay saved first draws through the production selector without inference."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from lerobot_robot_fairino.cup_action_selection import choose_proposal

if __package__:
    from . import evaluate_cup_setdown_gate as gate
    from . import replay_true_five_second_pi05 as fit
    from . import score_cup_training_timing_fit as timing_fit
else:
    import evaluate_cup_setdown_gate as gate
    import replay_true_five_second_pi05 as fit
    import score_cup_training_timing_fit as timing_fit


class AdditionalDrawRequired(Exception):
    pass


class SavedFirstDraw:
    def __init__(self, request, reply):
        self.request = request
        self.reply = reply
        self.calls = 0

    def predict(self, request):
        if request != self.request:
            raise ValueError('The selector changed the saved model request')
        self.calls += 1
        if self.calls > 1:
            raise AdditionalDrawRequired
        return copy.deepcopy(self.reply)


def replay(inputs, predictions, score):
    cases = timing_fit.cases_index(inputs)
    rows = timing_fit.prediction_index(cases, predictions)
    if (score['checkpoint'] != predictions['checkpoint'] or score['seeds'] != predictions['seeds']
            or score['observation_context_profile'] != predictions['observation_context_profile']):
        raise ValueError('The raw first-draw score describes different predictions')
    scores = {(row['case_id'], row['seed']): row for row in score['prefix_3']['rows']}
    if set(scores) != set(rows):
        raise ValueError('The raw first-draw score has different cases or seeds')
    results = []
    for key, row in sorted(rows.items()):
        if row['reply'].get('cup_proposal_selection'):
            raise ValueError('Expected raw first draws without prior selection')
        case = cases[key[0]]
        request = fit.model_request(case, 'setdown', predictions['observation_context_profile'])
        model = SavedFirstDraw(request, row['reply'])
        try:
            reply, draws, changed = choose_proposal(
                model, request, gate.normal_of(case), check_boundaries=True)
        except AdditionalDrawRequired:
            status = 'additional_draws_required'
            actions_unchanged = None
        else:
            if draws != 1 or changed or reply['actions'] != row['reply']['actions']:
                raise ValueError('The first-draw selector result differs from the saved actions')
            status = 'first_draw_selected'
            actions_unchanged = True
        results.append({'case_id': key[0], 'seed': key[1], 'source_index': case['source_index'],
                        'cycle': case['cycle'], 'kind': case['kind'], 'selection_status': status,
                        'first_actions_unchanged': actions_unchanged,
                        'prefix3_gate': scores[key]['gate'], 'violations': scores[key]['violations']})
    selected_failures = [row for row in results if row['selection_status'] == 'first_draw_selected'
                         and row['violations']]
    deferred_failures = [row for row in results if row['selection_status'] == 'additional_draws_required'
                         and row['violations']]
    return {
        'schema': 'fr5-cup-first-proposal-replay-v1',
        'checkpoint': predictions['checkpoint'], 'seeds': predictions['seeds'],
        'observation_context_profile': predictions['observation_context_profile'],
        'device': 'cpu', 'model_inference_calls': 0, 'robot_commands': 0,
        'task_success_verified': False, 'case_count': len(cases), 'draw_count': len(results),
        'selection_counts': dict(Counter(row['selection_status'] for row in results)),
        'selected_first_failure_count': len(selected_failures),
        'selected_first_failure_codes': dict(Counter(
            code for row in selected_failures for code in {v['code'] for v in row['violations']})),
        'failed_first_needing_additional_draws_count': len(deferred_failures),
        'selected_first_failures': selected_failures,
        'rows': results,
        'limitations': [
            'Only the recorded first draw is replayed. A request for another draw leaves the selected result unknown.',
            'First draws selected unchanged retain their existing nominal 6 Hz prefix score; '
            'the replay does not establish measured controller timing or physical success.',
            'Support and close readiness retain the original camera and gripper evidence limitations.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--score', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = replay(gate.read_json(args.inputs), gate.read_json(args.predictions), gate.read_json(args.score))
    report.update(inputs=str(args.inputs), predictions=str(args.predictions), score=str(args.score))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'case_count', 'draw_count', 'selection_counts', 'selected_first_failure_count',
        'selected_first_failure_codes', 'failed_first_needing_additional_draws_count',
        'selected_first_failures')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
