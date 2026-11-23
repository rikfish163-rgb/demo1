#!/usr/bin/env python3
"""Compare two integration step counts on paired recorded cup observations."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

from evaluate_cup_timing_policy import (
    INPUT_SCHEMA, SETDOWN_FIXED_PROFILE, model_request, read_json, validate_actions, write_new,
)
from score_cup_training_support import metrics as support_metrics


SERVER_ROOT = Path('/home/data4/hts/demo1/fr5-learning')
CLOCK = 'observation.cup_action_clock'
FIXED = 'observation.fixed_cup_action_clock'


def paired_predict(model, request, seed, steps):
    torch = model.torch
    predict_chunk = model.training.predict_action_chunk
    sample_noise = model.policy.model.sample_noise
    original_steps = model.policy.config.num_inference_steps
    batches, noises, replies, elapsed = [], [], [], []

    def capture_batch(policy, batch):
        if batch.get(FIXED) is not True or tuple(batch[CLOCK].shape) != (1, 25):
            raise ValueError('Expected fixed cup conditioning for both integration counts')
        tensors = {key: value.detach().cpu().clone() for key, value in batch.items()
                   if torch.is_tensor(value)}
        if batches and (tensors.keys() != batches[0].keys() or any(
                tensors[key].dtype != batches[0][key].dtype or
                not torch.equal(tensors[key], batches[0][key]) for key in tensors)):
            raise ValueError('Paired preprocessed observations differ')
        batches.append(tensors)
        return predict_chunk(policy, batch)

    def capture_noise(shape, device):
        noise = sample_noise(shape, device)
        saved = noise.detach().cpu().clone()
        if noises and (saved.dtype != noises[0].dtype or not torch.equal(saved, noises[0])):
            raise ValueError('Paired initial diffusion noise differs')
        noises.append(saved)
        return noise

    model.training.predict_action_chunk = capture_batch
    model.policy.model.sample_noise = capture_noise
    try:
        for count in steps:
            model.policy.config.num_inference_steps = count
            model.reset()
            torch.manual_seed(seed)
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                reply = model.predict(deepcopy(request))
            elapsed.append(time.perf_counter() - started)
            validate_actions(reply['actions'])
            if len(batches) != len(replies) + 1 or len(noises) != len(replies) + 1:
                raise ValueError('Expected one complete draw and one initial noise per prediction')
            if reply['sampling']['denoising_steps'] != count:
                raise ValueError('Prediction reports a different integration step count')
            if replies and reply['sampling']['cuda_rng_state'] != replies[0]['sampling']['cuda_rng_state']:
                raise ValueError('Paired CUDA sampling RNG states differ')
            replies.append(reply)
    finally:
        model.training.predict_action_chunk = predict_chunk
        model.policy.model.sample_noise = sample_noise
        model.policy.config.num_inference_steps = original_steps
    return replies, elapsed, {
        'preprocessed_tensor_keys': sorted(batches[0]),
        'preprocessed_tensors_elementwise_equal': True,
        'fixed_cup_action_clock': True,
        'pre_sampling_cuda_rng_equal': True,
        'initial_noise_elementwise_equal': True,
        'initial_noise_shape': list(noises[0].shape),
    }


def score(case, request, reply):
    value = support_metrics(case, {'seed': 0, 'request': request, 'reply': reply})
    for key in ('case_id', 'seed', 'matched'):
        value.pop(key)
    value['all_first_three_predicted_full_open'] = min(value['pred_gripper_pct']) >= 98
    value['steady_drift_over_2mm'] = value['actual_move_mm'] <= .2 and value['pred_move_mm'] > 2.
    value['steady_full_open_drift_over_2mm'] = (
        value['steady_drift_over_2mm'] and value['all_first_three_predicted_full_open'])
    value['opposed_recorded_move'] = (value['actual_move_mm'] > .5 and
                                    value['move_alignment_cosine'] is not None and
                                    value['move_alignment_cosine'] < 0.)
    return value


def summarize(rows):
    unique = {}
    for row in rows:
        key = row['metrics']['source'], row['metrics']['row_index'], row['seed']
        if key in unique:
            before = {field: value for field, value in unique[key]['metrics'].items() if field != 'kind'}
            after = {field: value for field, value in row['metrics'].items() if field != 'kind'}
            if before != after or unique[key]['reply']['actions'] != row['reply']['actions']:
                raise ValueError('Repeated original source-row-seed differs')
        else:
            unique[key] = row
    draws = list(unique.values())
    values = [row['metrics'] for row in draws]
    return {
        'draws_with_aliases': len(rows), 'unique_source_row_seed_draws': len(draws),
        'unique_anchors': len({(value['source'], value['row_index']) for value in values}),
        'steady_draws': sum(value['actual_move_mm'] <= .2 for value in values),
        'steady_drift_over_2mm_draws': sum(value['steady_drift_over_2mm'] for value in values),
        'steady_full_open_drift_over_2mm_draws': sum(value['steady_full_open_drift_over_2mm'] for value in values),
        'moving_draws': sum(value['actual_move_mm'] > .5 for value in values),
        'opposed_recorded_move_draws': sum(value['opposed_recorded_move'] for value in values),
        'recorded_all_open_close_label_deviation_draws': sum(value['early_close_against_recorded_labels'] for value in values),
        'median_endpoint_error_mm': statistics.median(value['endpoint_error_mm'] for value in values),
        'maximum_endpoint_error_mm': max(value['endpoint_error_mm'] for value in values),
        'median_inference_seconds': statistics.median(row['reply']['inference_seconds'] for row in draws),
        'median_predict_elapsed_seconds': statistics.median(row['predict_elapsed_seconds'] for row in draws),
        'selector_status': 'not_applied',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, action='append', required=True)
    parser.add_argument('--steps', type=int, nargs=2, default=[10, 20])
    args = parser.parse_args()
    if min(args.steps) < 1 or args.steps[0] == args.steps[1] or len(set(args.seed)) != len(args.seed):
        raise ValueError('Expected two different positive step counts and distinct seeds')
    if not args.output.resolve().is_relative_to(SERVER_ROOT.resolve()) or args.output.exists():
        raise ValueError('Output must be a new file under the server learning directory')
    summary_path = args.output.with_name(args.output.stem + '-summary.json')
    if summary_path.exists():
        raise FileExistsError(summary_path)
    inputs = read_json(args.inputs)
    cases = inputs['cases']
    if inputs.get('schema') != INPUT_SCHEMA or not cases or any(
            case['split'] != 'train' or case['source_index'] not in range(7) or
            len(case['targets_camera_aligned']) < 3 for case in cases):
        raise ValueError('Expected original training observations and three camera-aligned targets')
    sys.path.insert(0, str(args.checkpoint))
    with redirect_stdout(sys.stderr):
        from serve_stream_pi05 import Inference
        model = Inference(args.checkpoint, gpu=args.gpu)
    model.torch.set_num_threads(4)
    if (model.info['observation_context_profile'] != SETDOWN_FIXED_PROFILE or
            model.info['action_representation'] != 'cup-relative-tcp-tick-delta-v1'):
        raise ValueError('Expected a fixed-condition cup-relative tick-delta checkpoint')
    conditions = {str(count): {'predictions': []} for count in args.steps}
    audits = []
    for index, case in enumerate(cases):
        request = model_request(case, 'candidate', 'native', profile=SETDOWN_FIXED_PROFILE)
        if request['task'] != case['request']['task']:
            raise ValueError('Expected the original recorded task')
        for seed in args.seed:
            replies, elapsed, audit = paired_predict(model, request, seed, args.steps)
            audits.append({'case_id': case['case_id'], 'seed': seed, **audit})
            for count, reply, duration in zip(args.steps, replies, elapsed):
                conditions[str(count)]['predictions'].append({
                    'case_id': case['case_id'], 'case_index': index, 'seed': seed,
                    'request': {key: request[key] for key in ('cup_relative_state', 'cup_timing')},
                    'reply': reply, 'predict_elapsed_seconds': duration,
                    'metrics': score(case, request, reply),
                })
        print(f'{args.output.name}: {index + 1}/{len(cases)} cases', file=sys.stderr, flush=True)
    report = {
        'schema': 'fr5-cup-paired-denoising-step-probe-v1', 'checkpoint': str(model.root),
        'input': str(args.inputs), 'seeds': args.seed, 'step_counts': args.steps,
        'observation_context_profile': SETDOWN_FIXED_PROFILE,
        'action_shape': [50, 7], 'scored_prefix_points': 3,
        'scope': 'Raw first complete draws; only integration step count changes.',
        'latency_scope': 'Instrumented offline sampling and predict calls; no network or execution.',
        'selector_status': 'not_applied', 'robot_commands': 0, 'task_success_verified': False,
        'paired_audits': audits, 'conditions': conditions, 'summary_file': str(summary_path),
    }
    write_new(args.output, report)
    summary = {count: summarize(value['predictions']) for count, value in conditions.items()}
    write_new(summary_path, {
        'schema': 'fr5-cup-paired-denoising-step-summary-v1', 'prediction': str(args.output),
        'checkpoint': str(model.root), 'input': str(args.inputs), 'seeds': args.seed,
        'step_counts': args.steps, 'paired_draws': len(audits),
        'all_preprocessed_tensors_equal': all(row['preprocessed_tensors_elementwise_equal'] for row in audits),
        'all_initial_noise_equal': all(row['initial_noise_elementwise_equal'] for row in audits),
        'all_pre_sampling_cuda_rng_equal': all(row['pre_sampling_cuda_rng_equal'] for row in audits),
        'summary': summary, 'robot_commands': 0, 'task_success_verified': False,
    })
    print(json.dumps({'output': str(args.output), 'summary_file': str(summary_path), 'summary': summary},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
