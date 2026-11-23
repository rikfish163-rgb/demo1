#!/usr/bin/env python3
"""Replay recorded live requests through a checkpoint for bounded timing comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path('/home/data4/hts/demo1/fr5-learning')


def action_difference(reference, candidate, kinematics):
    """Compare unchanged absolute 50x7 wire chunks for one observation."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.shape != (50, 7) or b.shape != (50, 7) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected two finite 50 by 7 action chunks')
    distance = np.linalg.norm(kinematics.points(a[:, :6]) - kinematics.points(b[:, :6]), axis=1)
    gripper = np.abs(a[:, 6] - b[:, 6])
    return {
        'all_actions_max_absolute_difference': float(np.abs(a - b).max()),
        'prefix10_grasp_mean_mm': float(distance[:10].mean()),
        'prefix10_grasp_max_mm': float(distance[:10].max()),
        'full50_grasp_mean_mm': float(distance.mean()),
        'full50_grasp_max_mm': float(distance.max()),
        'prefix10_gripper_mean_fraction': float(gripper[:10].mean()),
        'prefix10_gripper_max_fraction': float(gripper[:10].max()),
        'full50_gripper_max_fraction': float(gripper.max()),
    }


def timed_inference(model, serving, request, rng_state, steps):
    """Time the same request, sampling and postprocessor used by the stdio service."""
    t = model.torch
    previous_steps = model.policy.config.num_inference_steps
    t.cuda.synchronize()
    started = time.perf_counter()
    _, task, state, rgb = serving.validate_request(request)
    decoded = time.perf_counter()
    batch = serving.prepare_request_batch(
        model.training, model.pre, state, rgb, task, model.info['image_profile'],
        model.info['observation_context_profile'], request.get('observation_history'))
    t.cuda.synchronize()
    prepared = time.perf_counter()
    model.policy.config.num_inference_steps = steps
    try:
        t.cuda.set_rng_state(t.tensor(rng_state, dtype=t.uint8, device='cpu'))
        t.cuda.synchronize()
        sampling_started = time.perf_counter()
        with t.inference_mode(), t.autocast('cuda', dtype=t.bfloat16):
            normalized = model.policy.predict_action_chunk(batch)
            t.cuda.synchronize()
            sampled = time.perf_counter()
            actions = model.post(normalized.detach().clone()).cpu().float()
            actions = serving.actions_for_wire(actions, state, model.relative,
                                               input_representation=model.info['action_representation'])
            t.cuda.synchronize()
            postprocessed = time.perf_counter()
    finally:
        model.policy.config.num_inference_steps = previous_steps
    if tuple(actions.shape) != (1, 50, 7) or not bool(t.isfinite(actions).all()):
        raise ValueError('Model returned an invalid action chunk')
    return actions[0].tolist(), {
        'decode_validate_s': decoded - started,
        'preprocess_s': prepared - decoded,
        'rng_restore_s': sampling_started - prepared,
        'sampling_s': sampled - sampling_started,
        'postprocess_s': postprocessed - sampled,
        'local_total_s': postprocessed - started,
    }


def timing_summary(records):
    result = {}
    for key in ('decode_validate_s', 'preprocess_s', 'rng_restore_s', 'sampling_s',
                'postprocess_s', 'local_total_s'):
        values = [r['timings'][key] for r in records]
        result[key] = {'count': len(values), 'median': statistics.median(values),
                       'min': min(values), 'max': max(values)}
    return result


def load_requests(path, checkpoint_sha):
    with Path(path).open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    sessions = [r for r in records if r.get('type') == 'session']
    predictions = sorted((r for r in records if r.get('type') == 'prediction'), key=lambda r: r['index'])
    complete = [r for r in records if r.get('type') == 'complete']
    if (len(sessions) != 1 or len(complete) != 1 or any(r.get('type') == 'error' for r in records)
            or sessions[0].get('checkpoint_sha256') != checkpoint_sha
            or len(predictions) != 5 or complete[0].get('predictions') != 5
            or [r['index'] for r in predictions] != list(range(5))):
        raise ValueError('Expected a complete five-request live archive for this checkpoint')
    for row in predictions:
        req, reply = row['request'], row['reply']
        if (req['id'] != reply['id'] or reply['checkpoint_sha256'] != checkpoint_sha
                or reply.get('wire_action_representation') != 'absolute-joint-position-v1'
                or reply.get('model_action_representation') != 'joint-delta-from-anchor-v1'
                or reply.get('sampling', {}).get('denoising_steps') != 10
                or not isinstance(reply['sampling'].get('cuda_rng_state'), list)):
            raise ValueError('Live request, model output or RNG state differs')
        values = np.asarray(reply['actions'], dtype=float)
        if values.shape != (50, 7) or not np.isfinite(values).all():
            raise ValueError('Live archive contains an invalid action chunk')
    return predictions


def run(checkpoint, source, archive, output, gpu):
    checkpoint, source, archive, output = map(lambda p: Path(p).resolve(), (checkpoint, source, archive, output))
    if not all(p.is_relative_to(ROOT) for p in (checkpoint, source, archive)):
        raise ValueError('Checkpoint, source and live archive must be on data4')
    if not output.is_relative_to(ROOT / 'diagnostics'):
        raise ValueError('Benchmark output must be under data4 diagnostics')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source / 'fr5_lerobot/training'))
    sys.path.insert(0, str(source))
    import serve_stream_pi05 as serving
    from fr5_lerobot.training.summarize_jev_pi05 import GraspKinematics

    loaded = time.monotonic()
    model = serving.Inference(checkpoint, gpu=gpu)
    load_seconds = time.monotonic() - loaded
    if (model.info['action_representation'] != serving.checkpoints.RELATIVE_REPRESENTATION
            or model.info['observation_context_profile'] not in ('none', 'state-history-physical-v1')
            or model.info['image_profile'] != 'full-and-workspace-v1'):
        raise ValueError('Expected the FR5 joint-delta full/workspace stream checkpoint')
    predictions = load_requests(archive, model.info['checkpoint_sha256'])
    kinematics = GraspKinematics(source / 'hardware/fr5.urdf', [5.208, 8.072, 162.])
    torch = model.torch
    default_threads = torch.get_num_threads()
    thread_counts = list(dict.fromkeys((default_threads, 1, 4)))
    warmups = []
    for steps in (10, 5):
        _, timings = timed_inference(model, serving, predictions[0]['request'],
                                     predictions[0]['reply']['sampling']['cuda_rng_state'], steps)
        warmups.append({'steps': steps, 'timings': timings})
    records, grouped = [], defaultdict(list)
    for threads in thread_counts:
        torch.set_num_threads(threads)
        for row in predictions:
            for steps in (10, 5):
                rng = row['reply']['sampling']['cuda_rng_state']
                actions, timings = timed_inference(model, serving, row['request'], rng, steps)
                record = {'index': row['index'], 'request_id': row['request']['id'],
                          'source_time': row['request']['source_time'], 'task': row['request']['task'],
                          'threads': threads, 'denoising_steps': steps, 'cuda_rng_state': rng,
                          'timings': timings, 'actions': actions}
                records.append(record)
                grouped[(threads, steps)].append(record)
    with (output / 'runs.jsonl').open('x', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    comparisons = []
    for threads in thread_counts:
        for index, captured in enumerate(predictions):
            ten, five = grouped[(threads, 10)][index], grouped[(threads, 5)][index]
            comparisons.append({
                'index': index, 'threads': threads,
                'five_vs_ten': action_difference(ten['actions'], five['actions'], kinematics),
                'ten_vs_recorded_v3': action_difference(captured['reply']['actions'], ten['actions'], kinematics),
            })
    summary = {'schema': 'fr5-pi05-live-timing-benchmark-v1', 'checkpoint': str(checkpoint),
               'checkpoint_sha256': model.info['checkpoint_sha256'], 'source': str(source),
               'live_archive': str(archive), 'gpu': gpu, 'load_seconds': load_seconds,
               'default_torch_threads': default_threads, 'thread_counts': thread_counts,
               'request_count': len(predictions), 'warmups': warmups,
               'timing_scope': 'local service path; preprocessing includes device transfer, no SSH/network',
               'same_request_and_recorded_cuda_rng_state': True,
               'grasp_offset_flange_mm': [5.208, 8.072, 162.],
               'timings': {f'{threads}_threads_{steps}_steps': timing_summary(grouped[(threads, steps)])
                           for threads in thread_counts for steps in (10, 5)},
               'comparisons': comparisons, 'runs': str(output / 'runs.jsonl'),
               'training_or_robot_commands': 0}
    with (output / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': 'complete', 'output': str(output), 'summary': str(output / 'summary.json'),
                      'thread_counts': thread_counts, 'timings': summary['timings']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    run(args.checkpoint, args.source, args.archive, args.output, args.gpu)


if __name__ == '__main__':
    main()
