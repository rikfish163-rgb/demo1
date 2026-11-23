#!/usr/bin/env python3
"""Replay production cup proposal selection on original five-second fit cases."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from lerobot_robot_fairino.cup_action_selection import choose_proposal

if __package__:
    from . import replay_true_five_second_pi05 as fit
    from . import score_cup_training_timing_fit as timing
else:
    import replay_true_five_second_pi05 as fit
    import score_cup_training_timing_fit as timing


FIT_SCHEMA = fit.PREDICTION_SCHEMA
OUTPUT_ROOT = fit.ROOT / 'evaluations'


def predict_fit_cases(model, inputs, seeds):
    """Return first and selected complete replies for each recorded case/seed."""
    profile = model.info['observation_context_profile']
    if profile not in fit.SETDOWN_PROFILES or model.info['n_action_steps'] != 3:
        raise ValueError('Expected a three-point cup setdown checkpoint')
    if inputs.get('schema') != fit.INPUT_SCHEMA:
        raise ValueError('Expected original five-second train-fit inputs')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and distinct')
    raw, selected = [], []
    predict = model.predict
    had_predict = 'predict' in model.__dict__
    for i, case in enumerate(inputs['cases'], 1):
        request = fit.model_request(case, 'setdown', profile)
        normal = case['initial_cup']['table_normal']
        for seed in seeds:
            first = []

            def capture_first(current_request):
                reply = predict(current_request)
                fit.validate_actions(reply)
                if not first:
                    first.append(reply)
                return reply

            model.reset()
            model.torch.manual_seed(seed)
            model.predict = capture_first
            try:
                with redirect_stdout(sys.stderr):
                    reply, count, changed = choose_proposal(
                        model, request, normal, check_boundaries=True)
            finally:
                if had_predict:
                    model.predict = predict
                else:
                    del model.predict
            fit.validate_actions(reply)
            row = {'case_id': case['case_id'], 'observer': 'setdown', 'seed': seed,
                   'task': request['task'], 'source_time': request['source_time'],
                   'cup_tracking': request['cup_tracking'], 'cup_timing': request['cup_timing'],
                   'request': request}
            raw.append({**row, 'reply': first[0]})
            selected.append({**row, 'reply': reply, 'draw_count': count,
                             'new_draw_selected': changed,
                             'selection_matched': reply.get('cup_proposal_selection', {}).get('matched', True)})
        print(f'fit selection: {i}/{len(inputs["cases"])} cases', file=sys.stderr, flush=True)
    return raw, selected


def prediction_report(model, inputs_path, seeds, predictions):
    return {'schema': FIT_SCHEMA, 'model': 'setdown', 'seeds': seeds,
            'input': str(inputs_path), 'checkpoint': str(model.root),
            'observation_context_profile': model.info['observation_context_profile'],
            'observers': ['setdown'], 'predictions': predictions, 'robot_commands': 0}


def compare_first_predictions(torch, first, reference):
    """Compare all 50 x 7 values exactly, keyed by recorded case and seed."""
    actual = {(row['case_id'], row['seed']): row for row in first}
    expected = {(row['case_id'], row['seed']): row for row in reference}
    common = actual.keys() & expected.keys()
    json_mismatches = tensor_mismatches = mismatches = 0
    for key in common:
        actions, recorded = actual[key]['reply']['actions'], expected[key]['reply']['actions']
        same_json = actions == recorded
        same_tensor = torch.equal(torch.tensor(actions, dtype=torch.float64, device='cpu'),
                                  torch.tensor(recorded, dtype=torch.float64, device='cpu'))
        json_mismatches += not same_json
        tensor_mismatches += not same_tensor
        mismatches += not (same_json and same_tensor)
    missing, extra = len(expected.keys() - actual.keys()), len(actual.keys() - expected.keys())
    return {'expected_draws': len(reference), 'actual_draws': len(first),
            'compared_draws': len(common), 'missing_draws': missing, 'extra_draws': extra,
            'json_action_mismatch_draws': json_mismatches,
            'torch_action_mismatch_draws': tensor_mismatches, 'action_mismatch_draws': mismatches,
            'exactly_paired': (len(first) == len(reference) == len(common)
                               and not mismatches and not missing and not extra)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--raw-predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, action='append', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.suffix != '.json':
        raise ValueError(f'Output must be a JSON file under {OUTPUT_ROOT}')
    if output.exists():
        raise FileExistsError(output)
    inputs, reference = fit.read_json(args.inputs), fit.read_json(args.raw_predictions)
    cases = timing.cases_index(inputs)
    timing.prediction_index(cases, reference)
    if not args.seed or len(set(args.seed)) != len(args.seed) or set(args.seed) != set(reference['seeds']):
        raise ValueError('Seeds must match the distinct original raw prediction seeds')
    if args.checkpoint.resolve() != Path(reference['checkpoint']).resolve():
        raise ValueError('Checkpoint must match the original raw predictions')
    sys.path.insert(0, str(args.checkpoint))
    with redirect_stdout(sys.stderr):
        from serve_stream_pi05 import Inference
        model = Inference(args.checkpoint, gpu=args.gpu)
    model.torch.set_num_threads(4)
    if model.info['observation_context_profile'] != reference['observation_context_profile']:
        raise ValueError('Checkpoint profile differs from the original raw predictions')
    raw, selected = predict_fit_cases(model, inputs, args.seed)
    pairing = compare_first_predictions(model.torch, raw, reference['predictions'])
    report = prediction_report(model, args.inputs, args.seed, selected)
    report.update(selector='bounded_3_complete_proposals_at_current_cup_boundaries_v2',
                  raw_predictions=str(args.raw_predictions), raw_first_pairing=pairing)
    fit.write_new(output, report)
    print(json.dumps({'output': str(output), 'predictions': len(selected),
                      'draw_counts': dict(sorted(Counter(row['draw_count'] for row in selected).items())),
                      'new_draw_selected': sum(row['new_draw_selected'] for row in selected),
                      'unmatched_draws': sum(not row['selection_matched'] for row in selected),
                      'raw_first_pairing': pairing}, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if pairing['exactly_paired'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
