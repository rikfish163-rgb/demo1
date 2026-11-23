#!/usr/bin/env python3
"""Score complete archived cup predictions and prepare controller-only inputs."""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace

import evaluate_cup_setdown_gate as gate
from evaluate_cup_timing_policy import read_json, write_new
from replay_true_five_second_pi05 import compare
from score_cup_proposal_trace import score


def controller_input(inputs, predictions):
    rows = {(r['case_id'], r['seed']): r for r in predictions['predictions']}
    cases = []
    for case in inputs['cases']:
        pairs = []
        for seed in predictions['seeds']:
            row = rows[(case['case_id'], seed)]
            pairs.append({'seed': seed, 'candidate': {'request': row['request'], 'reply': row['reply']}})
        cases.append({**{k: case[k] for k in ('case_id', 'group', 'kind', 'initial_cup')},
                      'predictions': pairs})
    return {'schema': 'fr5-cup-tracking-replay-comparison-v1',
            'checkpoints': {'candidate': predictions['checkpoint']}, 'seeds': predictions['seeds'],
            'cases': cases, 'robot_commands': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--heldout-inputs', type=Path, required=True)
    parser.add_argument('--trial-inputs', type=Path, required=True)
    parser.add_argument('--fit-inputs', type=Path, required=True)
    parser.add_argument('--legacy-fit', type=Path, required=True)
    parser.add_argument('--heldout-capture', type=Path, required=True)
    parser.add_argument('--heldout-journal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir()
    fit_path = args.output_dir / 'fit-comparison.json'
    with redirect_stdout(io.StringIO()):
        compare(SimpleNamespace(input=args.fit_inputs, old=args.legacy_fit,
                                new=args.evaluation_dir / 'fit-predictions.json', output=fit_path))
    fit = read_json(fit_path)
    heldout = read_json(args.heldout_inputs)
    cases = gate.cases_index(heldout)
    support, frames = gate.raw_support_references(
        cases, gate.read_jsonl(args.heldout_capture), read_json(args.heldout_journal))
    reports = {}
    for label, inputs_path in (('heldout-five-second', args.heldout_inputs),
                               ('heldout-original', args.heldout_inputs),
                               ('trial-original', args.trial_inputs)):
        predictions = read_json(args.evaluation_dir / (label + '.json'))
        inputs = read_json(inputs_path)
        trace, raw = score(inputs, predictions)
        write_new(args.output_dir / (label + '-trace.json'), trace)
        raw_path = args.output_dir / (label + '-raw-first.json')
        write_new(raw_path, raw)
        controller_path = args.output_dir / (label + '-controller-input.json')
        write_new(controller_path, controller_input(inputs, predictions))
        entry = {'trace': trace['summary'], 'controller_input': str(controller_path)}
        if label.startswith('heldout'):
            entry['gates'] = {}
            for name, saved, path in (('selected', predictions, args.evaluation_dir / (label + '.json')),
                                      ('raw_first', raw, raw_path)):
                meta, rows, _ = gate.prediction_index(cases, [saved], [path])
                scored = {f'prefix_{h}': gate.run_horizon(cases, rows, frames, support, fit, meta, 'setdown', h)
                          for h in (3, 10)}
                scored.update(checkpoint=meta, input=str(inputs_path), predictions=str(path),
                              fit_comparison=str(fit_path), raw_capture=str(args.heldout_capture),
                              raw_journal=str(args.heldout_journal),
                              nominal_clock='original Gemini source monotonic timestamp plus tick/6 s; latency excluded')
                write_new(args.output_dir / (label + '-' + name + '-gate.json'), scored)
                entry['gates'][name] = {k: {field: value[field] for field in (
                    'gate', 'assessable_draws_gate', 'failure_draws_including_train_fit',
                    'critical_unassessed_draws', 'true_five_second_training_fit')}
                    for k, value in scored.items() if k.startswith('prefix_')}
        reports[label] = entry
    report = {'schema': 'fr5-cup-checkpoint-suite-score-v1',
              'evaluation_dir': str(args.evaluation_dir), 'checkpoint': fit['checkpoints']['new'],
              'fit_comparison': str(fit_path), 'reports': reports, 'robot_commands': 0,
              'scope': 'isolated recorded observations and model predictions; not controller or task success'}
    write_new(args.output_dir / 'summary.json', report)
    import json
    compact = {'checkpoint': report['checkpoint'], 'output': str(args.output_dir / 'summary.json'),
               'reports': {}}
    for label, entry in reports.items():
        compact['reports'][label] = {'trace': entry['trace']}
        if 'gates' in entry:
            compact['reports'][label]['gates'] = {
                name: {prefix: {**{k: value[k] for k in (
                    'gate', 'assessable_draws_gate', 'failure_draws_including_train_fit')},
                    'fit_failures': value['true_five_second_training_fit']['failure_count'],
                    'fit_worst_down_mm': value['true_five_second_training_fit']['worst_down_mm']}
                       for prefix, value in prefixes.items()}
                for name, prefixes in entry['gates'].items()}
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
