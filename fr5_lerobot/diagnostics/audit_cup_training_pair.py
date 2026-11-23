#!/usr/bin/env python3
"""Compare saved clean and terminal cup training runs on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


SHARED_FIELDS = (
    'dataset_path', 'source_provenance', 'train_episodes', 'heldout_episodes',
    'heldout_physical_source', 'camera_aligned_targets', 'cup_relative_data',
    'cup_tracking_data', 'relative_action_statistics', 'anchor_coverage',
    'initialized_from', 'initialized_from_steps', 'resumed_from',
    'optimizer_restored', 'optimizer_migration', 'seed', 'batch_size',
    'gradient_accumulation', 'effective_batch_size', 'target_steps',
    'learning_rate', 'lr_schedule', 'rank', 'adaptation', 'trainable_parameters',
    'trainable_expert_parameters', 'vision_lora', 'vision_lora_initialization',
    'action_representation', 'action_label_origin', 'gripper_target_origin',
    'chunk_size', 'n_action_steps', 'target_timebase_hz', 'image_profile',
    'observation_context_profile', 'sampling', 'sampling_profile',
    'sampler_report', 'sampler_migration', 'normalization_from',
    'normalization_fallbacks', 'cup_roll_state_bounds_rad',
    'first_action_weight', 'executed_prefix_weight', 'vertical_weight',
    'gripper_weight', 'tail_attention_and_loss_mask', 'prefix_training',
    'prefix_cache',
)
PROCESSOR_FILES = (
    'policy_preprocessor_step_2_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
)
CONFIG_FILES = (
    'normalization.json', 'policy_preprocessor.json', 'policy_postprocessor.json',
    'action_representation.json', 'observation_context.json', 'image_profile.json',
)


def read_json(path):
    return json.loads(path.read_text())


def metric_summary(rows):
    sources = Counter(item['source'] for row in rows for item in row['sampling'])
    finite = all(math.isfinite(row[key]) for row in rows
                 for key in ('loss', 'gradient_norm', 'learning_rate'))
    result = {
        'optimizer_step_count': len(rows),
        'steps_are_contiguous_1_to_256': [row['step'] for row in rows] == list(range(1, 257)),
        'loss_gradient_lr_finite': finite,
        'gradient_positive_count': sum(row['gradient_norm'] > 0 for row in rows),
        'learning_rates': sorted({row['learning_rate'] for row in rows}),
        'sample_count': sum(sources.values()),
        'sample_count_by_source': dict(sorted(sources.items())),
        'last_loss': rows[-1]['loss'],
    }
    if 'optimizer_moments' in rows[0]:
        result['moment_bytes'] = sorted({row['optimizer_moments']['moment_bytes'] for row in rows})
        result['moment_devices_between_updates'] = sorted({
            device for row in rows for device in row['optimizer_moments']['devices']})
        result['peak_allocated_mib_recorded_across_all_steps'] = max(
            row['gpu_peak_allocated_mib'] for row in rows)
    return result


def processor_comparison(paths):
    result = []
    for name in PROCESSOR_FILES:
        with safe_open(str(paths['parent'] / name), framework='pt', device='cpu') as parent:
            keys = list(parent.keys())
            for label in ('clean', 'terminal'):
                with safe_open(str(paths[label] / name), framework='pt', device='cpu') as current:
                    keys_equal = list(current.keys()) == keys
                    unequal = [key for key in keys if key not in current.keys()
                               or not torch.equal(parent.get_tensor(key), current.get_tensor(key))]
                    result.append({'model': label, 'file': name, 'tensor_count': len(keys),
                                   'keys_equal_to_parent': keys_equal,
                                   'all_tensors_equal_to_parent': keys_equal and not unequal,
                                   'unequal_tensors': unequal})
    return result


def parameter_updates(paths):
    root = paths['terminal']
    with safe_open(str(root / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
        keys = list(f.keys())
    representatives = [
        ('vision_lora', 'adapter_model.safetensors', next(
            key for key in keys if 'vision' in key and 'lora_B' in key)),
        ('language_lora', 'adapter_model.safetensors', next(
            key for key in keys if 'gemma_expert' not in key and 'vision' not in key
            and 'lora_B' in key)),
        ('action_expert', 'expert_model.safetensors',
         'model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.base_layer.weight'),
        ('action_projection', 'projections.safetensors', 'model.action_out_proj.weight'),
    ]
    result = []
    for component, name, key in representatives:
        with safe_open(str(paths['parent'] / name), framework='pt', device='cpu') as f:
            parent = f.get_tensor(key)
        parent_norm = float(torch.linalg.vector_norm(parent.float()))
        for label in ('clean', 'terminal'):
            with safe_open(str(paths[label] / name), framework='pt', device='cpu') as f:
                current = f.get_tensor(key)
            delta = current.float() - parent.float()
            result.append({'model': label, 'component': component, 'file': name,
                           'tensor': key, 'shape': list(current.shape),
                           'changed_from_parent': not torch.equal(current, parent),
                           'finite': bool(torch.isfinite(current).all()),
                           'max_absolute_change': float(delta.abs().max()),
                           'delta_l2': float(torch.linalg.vector_norm(delta)),
                           'parent_l2': parent_norm})
    return result


def audit(clean_path, terminal_path):
    paths = {'clean': clean_path, 'terminal': terminal_path}
    reports = {key: read_json(path / 'training-report.json') for key, path in paths.items()}
    paths['parent'] = Path(reports['clean']['initialized_from'])
    metrics = {key: [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()
                     if line.strip()] for key, path in paths.items() if key != 'parent'}
    equal = [key for key in SHARED_FIELDS if reports['clean'][key] == reports['terminal'][key]]
    different = [key for key in SHARED_FIELDS if key not in equal]
    source = reports['terminal']['source_provenance']
    coverage = reports['terminal']['cup_relative_data']['coverage']
    configs = [{'file': name,
                'clean_equals_terminal': read_json(clean_path / name) == read_json(terminal_path / name),
                'clean_equals_parent': read_json(clean_path / name) == read_json(paths['parent'] / name),
                'terminal_equals_parent': read_json(terminal_path / name) == read_json(paths['parent'] / name)}
               for name in CONFIG_FILES]
    result = {
        'schema': 'fr5-cup-training-pair-audit-v1',
        'device': 'cpu', 'paths': {key: str(path) for key, path in paths.items()},
        'data': {'source_count': source['source_count'], 'train_sources': reports['terminal']['train_episodes'],
                 'heldout_sources': reports['terminal']['heldout_episodes'],
                 'rows': coverage['rows'], 'train_rows': coverage['train_rows'],
                 'validation_rows': coverage['validation_rows']},
        'shared_field_count': len(SHARED_FIELDS), 'shared_fields_equal': equal,
        'shared_fields_different': different,
        'sampling_equal_at_every_optimizer_step': len(metrics['clean']) == len(metrics['terminal'])
        and all(a['step'] == b['step'] and a['sampling'] == b['sampling']
                for a, b in zip(metrics['clean'], metrics['terminal'])),
        'metrics': {key: metric_summary(rows) for key, rows in metrics.items()},
        'configuration_values': configs,
        'normalizer_tensors': processor_comparison(paths),
        'representative_parameter_updates': parameter_updates(paths),
        'method_difference': {
            'clean': reports['clean']['xyz_loss_mode'],
            'terminal': reports['terminal']['xyz_loss_mode'],
            'terminal_coefficient': reports['terminal']['xyz_terminal_coefficient'],
            'terminal_margin_mm': reports['terminal']['xyz_terminal_margin_mm'],
            'native_denoising_steps': reports['terminal']['xyz_terminal_steps'],
            'terminal_memory_method': reports['terminal']['terminal_memory_method'],
        },
        'robot_commands': 0, 'task_success_verified': False,
        'limitations': [
            'Saved training metadata and metrics establish the paired run configuration; '
            'they do not establish inference quality.',
            'Four representative components are compared directly; this is not a scan of every saved parameter.',
            'The optimizer moment equivalence result is recorded separately from this CPU audit.',
        ],
    }
    completed = all(row['steps_are_contiguous_1_to_256'] and row['loss_gradient_lr_finite']
                    and row['gradient_positive_count'] == 256 and row['learning_rates'] == [1e-6]
                    and row['sample_count'] == 1024 and '7' not in row['sample_count_by_source']
                    for row in result['metrics'].values())
    valid = (completed and not different and result['sampling_equal_at_every_optimizer_step']
             and all(row['clean_equals_terminal'] for row in configs)
             and all(row['all_tensors_equal_to_parent'] for row in result['normalizer_tensors'])
             and all(row['changed_from_parent'] and row['finite']
                     for row in result['representative_parameter_updates'])
             and result['data'] == {'source_count': 8, 'train_sources': list(range(7)),
                                    'heldout_sources': [7], 'rows': 4193,
                                    'train_rows': 3443, 'validation_rows': 750})
    result['status'] = 'paired_training_consistent' if valid else 'paired_training_differs'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', required=True, type=Path)
    parser.add_argument('--terminal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = audit(args.clean, args.terminal)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'status', 'data', 'shared_field_count', 'shared_fields_different',
        'sampling_equal_at_every_optimizer_step', 'metrics', 'representative_parameter_updates')},
                     ensure_ascii=False))
    return 0 if report['status'] == 'paired_training_consistent' else 1


if __name__ == '__main__':
    raise SystemExit(main())
