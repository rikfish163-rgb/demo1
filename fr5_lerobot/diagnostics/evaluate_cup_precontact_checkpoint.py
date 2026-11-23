#!/usr/bin/env python3
"""Evaluate cup approach failures before the fixed timing and release suite."""

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import statistics
import sys

from evaluate_cup_timing_policy import SETDOWN_PROFILES, read_json, write_new
from probe_supported_pi05_draws import predict_cases, prediction_report
from evaluate_cup_training_timing_fit import (
    predict_fit_cases, prediction_report as fit_prediction_report,
)
from score_cup_checkpoint_suite import controller_input
from score_cup_proposal_trace import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--heldout-inputs', type=Path, required=True)
    parser.add_argument('--fit-inputs', type=Path, required=True)
    parser.add_argument('--trial-inputs', type=Path, required=True)
    parser.add_argument('--trial02-inputs', type=Path, required=True)
    parser.add_argument('--training-support-inputs', type=Path)
    parser.add_argument('--recovered-feedback-inputs', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, action='append', required=True)
    parser.add_argument('--fit-seed', type=int, action='append', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.checkpoint))
    with redirect_stdout(sys.stderr):
        from serve_stream_pi05 import Inference
        model = Inference(args.checkpoint, gpu=args.gpu)
    model.torch.set_num_threads(4)
    profile = model.info['observation_context_profile']
    if profile not in SETDOWN_PROFILES or model.info['n_action_steps'] != 3:
        raise ValueError('Expected a three-point cup setdown checkpoint')
    trial02 = read_json(args.trial02_inputs)
    predictions = predict_cases(model, trial02, args.seed, None, check_boundaries=True,
                                label='trial02-original.json')
    saved = prediction_report(model, args.trial02_inputs, args.seed, None, predictions,
                              check_boundaries=True)
    write_new(args.output_dir / 'trial02-original.json', saved)
    trace, raw = score(trial02, saved)
    write_new(args.output_dir / 'trial02-trace.json', trace)
    write_new(args.output_dir / 'trial02-raw-first.json', raw)
    write_new(args.output_dir / 'trial02-controller-input.json', controller_input(trial02, saved))
    additional = []
    for inputs_path, label in ((args.training_support_inputs, 'training-support'),
                               (args.recovered_feedback_inputs, 'recovered-feedback')):
        if inputs_path is None:
            continue
        inputs = read_json(inputs_path)
        for task, name in (('抓取桌面上的透明杯子，在空中维持5秒后放下', label + '-five-second.json'),
                           (None, label + '-original.json')):
            predictions = predict_cases(model, inputs, args.seed, task, check_boundaries=True, label=name)
            write_new(args.output_dir / name, prediction_report(model, inputs_path, args.seed, task,
                                                               predictions, check_boundaries=True))
            additional.append(str(args.output_dir / name))
    rows = trace['rows']
    progress = [row['selected']['progress_toward_pickup_mm'] for row in rows]
    summary = {
        'schema': 'fr5-cup-precontact-checkpoint-evaluation-v1',
        'checkpoint': str(args.checkpoint), 'trial02_input': str(args.trial02_inputs),
        'seeds': args.seed, 'trial02_trace': trace['summary'],
        'selected_progress': {'negative_draws': sum(value < 0 for value in progress),
                              'draws': len(progress), 'median_mm': statistics.median(progress),
                              'minimum_mm': min(progress)},
        'timing_suite_completed': False, 'controller_verified': False, 'robot_commands': 0,
        'additional_predictions': additional,
        'scope': 'Recorded-input predictions; progress is distance to the fixed demonstration pickup reference.',
    }
    rejected = any(not row['matched'] or not row['selected']['geometry_passed']
                   for row in rows if row['gripper_feedback_valid'])
    if rejected:
        summary['gate'] = 'failed_trial02_approach'
        write_new(args.output_dir / 'evaluation-status.json', summary)
        print(summary, flush=True)
        return
    for path, task, name in (
        (args.heldout_inputs, '抓取桌面上的透明杯子，在空中维持5秒后放下', 'heldout-five-second.json'),
        (args.heldout_inputs, None, 'heldout-original.json'),
        (args.trial_inputs, None, 'trial-original.json'),
    ):
        predictions = predict_cases(model, read_json(path), args.seed, task,
                                    check_boundaries=True, label=name)
        write_new(args.output_dir / name, prediction_report(model, path, args.seed, task, predictions,
                                                           check_boundaries=True))
    fit_inputs = read_json(args.fit_inputs)
    fit_predictions, fit_selected_predictions = predict_fit_cases(model, fit_inputs, args.fit_seed)
    write_new(args.output_dir / 'fit-predictions.json',
              fit_prediction_report(model, args.fit_inputs, args.fit_seed, fit_predictions))
    write_new(args.output_dir / 'fit-selected-predictions.json',
              fit_prediction_report(model, args.fit_inputs, args.fit_seed, fit_selected_predictions))
    summary.update(gate='pending_timing_score', timing_suite_completed=True)
    write_new(args.output_dir / 'evaluation-status.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
