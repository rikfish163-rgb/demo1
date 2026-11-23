#!/usr/bin/env python3
"""Benchmark RTC on saved real camera requests without connecting to a robot."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np


DATA_ROOT = Path('/home/data4/hts/demo1/fr5-learning')
FPS = 6
PREFIX = 10


def prior_prefix(previous, current):
    """Align a prior prediction on its nominal grid, with no execution claim."""
    old = previous['request']
    new = current['request']
    elapsed = float(new['source_time']) - float(old['source_time'])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('Adjacent requests must have increasing source times')
    actions = np.asarray(previous['reply']['actions'], dtype=float)
    if actions.shape != (50, 7) or not np.isfinite(actions).all():
        raise ValueError('Expected the previous complete 50 by 7 prediction')
    consumed = math.floor(elapsed * FPS)
    retained = actions[min(consumed, PREFIX):PREFIX].copy()
    # Apply the stream adapter's device projection, without inventing a
    # previously sent target for the controller's deadband.
    from lerobot_robot_fairino.stream_contract import decode_action
    for target in retained:
        target[6] = decode_action(target.tolist())['gripper_integer_percent'] / 100.
    return {
        'previous_request_id': old['id'],
        'current_request_id': new['id'],
        'source_interval_s': elapsed,
        'nominal_past_steps': consumed,
        'retained_steps': len(retained),
        'first_retained_time_minus_current_source_s': (
            (consumed + 1) / FPS - elapsed if len(retained) else None),
        'absolute_actions': retained.tolist(),
        'commitment': 'hypothetical_prior_prediction_not_executed',
        'gripper_projection': 'native_clamp_round_without_controller_deadband',
    }


def summarize_timings(rows):
    if not rows:
        return {}
    keys = ('sampling_s', 'postprocess_s', 'local_total_s', 'peak_allocated_mib')
    return {key: {
        'count': len(rows),
        'min': min(row['timings'][key] for row in rows),
        'median': statistics.median(row['timings'][key] for row in rows),
        'max': max(row['timings'][key] for row in rows),
    } for key in keys}


def sample(model, serving, batch, state, noise, *, processor, prefix=None, delay=0):
    """Use the native sampler and the checkpoint's unchanged postprocessor."""
    torch = model.torch
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    model.policy.config.rtc_config = (
        RTCConfig(enabled=True, execution_horizon=PREFIX) if processor is not None else None)
    model.policy.rtc_processor = processor
    model.policy.model.rtc_processor = processor
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = torch.no_grad() if processor is not None else torch.inference_mode()
    kwargs = {'noise': noise}
    if processor is not None:
        kwargs.update(prev_chunk_left_over=prefix, inference_delay=delay,
                      execution_horizon=prefix.shape[-2] if prefix is not None else PREFIX)
    with context, torch.autocast('cuda', dtype=torch.bfloat16):
        normalized = model.policy.predict_action_chunk(batch, **kwargs)
        torch.cuda.synchronize()
        sampled = time.perf_counter()
        actions = model.post(normalized.detach().clone()).cpu().float()
        actions = serving.actions_for_wire(
            actions, state, model.relative,
            input_representation=model.info['action_representation'])
        torch.cuda.synchronize()
        finished = time.perf_counter()
    if tuple(actions.shape) != (1, 50, 7) or not bool(torch.isfinite(actions).all()):
        raise ValueError('Expected a finite complete native action chunk')
    return {
        'normalized_actions': normalized.detach().cpu().float()[0].tolist(),
        'absolute_actions': actions[0].tolist(),
        'timings': {
            'sampling_s': sampled - started,
            'postprocess_s': finished - sampled,
            'local_total_s': finished - started,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        },
    }


def run(checkpoint, source, archive, output, gpu):
    checkpoint, source, archive, output = (
        Path(path).resolve() for path in (checkpoint, source, archive, output))
    if not all(path.is_relative_to(DATA_ROOT) for path in (checkpoint, source, archive)):
        raise ValueError('Checkpoint, source and real input archive must be on data4')
    if not output.is_relative_to(DATA_ROOT / 'diagnostics'):
        raise ValueError('Output must be under data4 diagnostics')
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(source), str(source / 'fr5_lerobot/training'),
                    str(source / 'fr5_lerobot/src')]
    import serve_stream_pi05 as serving
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.processor import NormalizerProcessorStep
    from fr5_lerobot.diagnostics.benchmark_pi05_live import action_difference, load_requests
    from fr5_lerobot.training.stream_rtc import FR5RTCProcessor, normalize_absolute_prefix
    from fr5_lerobot.training.summarize_jev_pi05 import GraspKinematics

    started = time.monotonic()
    print(json.dumps({'status': 'model_loading', 'gpu': gpu,
                      'robot_commands_sent': 0}), flush=True)
    model = serving.Inference(checkpoint, gpu=gpu)
    load_seconds = time.monotonic() - started
    if (model.info['action_representation'] != serving.checkpoints.RELATIVE_REPRESENTATION
            or model.info['observation_context_profile'] != 'state-history-physical-v1'
            or model.policy.config.num_inference_steps != 10):
        raise ValueError('Expected the history joint-delta checkpoint with native ten-step sampling')
    captured = load_requests(archive, model.info['checkpoint_sha256'])
    normalizer = next(step for step in model.pre.steps
                      if isinstance(step, NormalizerProcessorStep))
    processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=PREFIX), action_dim=7)
    kin = GraspKinematics(source / 'hardware/fr5.urdf', [5.208, 8.072, 162.])
    torch = model.torch
    grouped = {name: [] for name in ('eager', 'rtc_without_prefix', 'rtc_with_prefix')}
    comparisons = []
    with (output / 'runs.jsonl').open('x', encoding='utf-8') as stream:
        for previous, current in zip(captured, captured[1:]):
            window = prior_prefix(previous, current)
            if not window['retained_steps']:
                comparisons.append({'index': current['index'], 'prefix': window,
                                    'status': 'no_remaining_nominal_prefix'})
                continue
            _, task, state, rgb = serving.validate_request(current['request'])
            batch = serving.prepare_request_batch(
                model.training, model.pre, state, rgb, task, model.info['image_profile'],
                model.info['observation_context_profile'],
                current['request'].get('observation_history'))
            rng = current['reply']['sampling']['cuda_rng_state']
            torch.cuda.set_rng_state(torch.tensor(rng, dtype=torch.uint8, device='cpu'))
            noise = model.policy.model.sample_noise((1, 50, 32), 'cuda')
            prefix = normalize_absolute_prefix(
                torch.tensor(window['absolute_actions'], dtype=torch.float32),
                torch.tensor(state, dtype=torch.float32),
                model.relative, normalizer).to('cuda')
            delay = math.ceil(current['source_to_reply_s'] * FPS)
            runs = {}
            for name, use_processor, condition in (
                ('eager', None, None),
                ('rtc_without_prefix', processor, None),
                ('rtc_with_prefix', processor, prefix),
            ):
                row = sample(model, serving, batch, state, noise.clone(),
                             processor=use_processor, prefix=condition, delay=delay)
                row.update(mode=name, index=current['index'],
                           request_id=current['request']['id'],
                           source_time=current['request']['source_time'],
                           reference_inference_delay_steps=delay,
                           input_prefix=window if condition is not None else None)
                grouped[name].append(row)
                runs[name] = row
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps({'status': 'sampled', 'mode': name,
                                  'index': current['index'], 'timings': row['timings'],
                                  'robot_commands_sent': 0}), flush=True)
            comparison = {
                'index': current['index'],
                'prefix': window,
                'cuda_rng_state': rng,
                'noise': noise.cpu().tolist(),
                'eager_vs_saved_live': action_difference(
                    current['reply']['actions'], runs['eager']['absolute_actions'], kin),
                'rtc_without_prefix_vs_eager': action_difference(
                    runs['eager']['absolute_actions'],
                    runs['rtc_without_prefix']['absolute_actions'], kin),
                'rtc_with_prefix_vs_eager': action_difference(
                    runs['eager']['absolute_actions'],
                    runs['rtc_with_prefix']['absolute_actions'], kin),
                'reference_inference_delay_steps': delay,
                'rtc_local_delay_steps_lower_bound': math.ceil(
                    runs['rtc_with_prefix']['timings']['local_total_s'] * FPS),
            }
            comparisons.append(comparison)
    report = {
        'schema': 'fr5-pi05-rtc-inference-benchmark-v1',
        'checkpoint': str(checkpoint),
        'source': str(source),
        'real_input_archive': str(archive),
        'gpu': gpu,
        'load_seconds': load_seconds,
        'denoising_steps': 10,
        'action_shape': [50, 7],
        'internal_noise_shape': [1, 50, 32],
        'timings': {key: summarize_timings(rows) for key, rows in grouped.items()},
        'comparisons': comparisons,
        'runs': str(output / 'runs.jsonl'),
        'robot_commands_sent': 0,
        'physical_task_successes_confirmed': 0,
        'limitations': [
            'Prior prefixes are projected model predictions; no controller committed or executed them.',
            'The nominal source-time grid is used here; execution requires actual controller consumption times.',
            'Local timing excludes JPEG preprocessing, SSH and controller preflight.',
            'The reference delay comes from eager live replies; RTC delay must be measured on the complete execution chain.',
            'Gripper deadband and physical contact are not simulated.',
            'Recorded input replay does not establish task success or resolve the current gripper fault.',
        ],
    }
    (output / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'timings': report['timings'],
                      'robot_commands_sent': 0}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    run(args.checkpoint, args.source, args.archive, args.output, args.gpu)


if __name__ == '__main__':
    main()
