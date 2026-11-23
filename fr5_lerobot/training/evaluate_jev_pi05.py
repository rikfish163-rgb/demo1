"""Open-loop pi0.5 fit to recorded FR5/Jev labels; this is not task success.

Each source contributes uniformly spaced frames plus recorded gripper-switch
and hold endpoints. Predictions use only the current RGB/state/task and, when
configured, past physical state history. Results go to a new evaluations folder.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import train_stream_pi05 as training
import serve_stream_pi05 as serving

EVALUATIONS_ROOT = Path('/home/data4/hts/demo1/fr5-learning/evaluations')
STATIONARY_DEG = 0.5
GROUPS = ('all', 'uniform', 'close_before', 'close_at', 'open_before', 'open_at', 'hold_start', 'hold_end')


def _synthetic_marker(value):
    if not isinstance(value, dict):
        return False
    for key in ('synthetic', 'is_synthetic', 'simulated', 'is_simulated', 'mock', 'fake'):
        flag = value.get(key)
        if flag is True or type(flag) is int and flag == 1 or (
                isinstance(flag, str) and flag.strip().lower() in ('true', 'yes', '1')):
            return True
    return 'synthetic_id' in value


def require_captured_jev(proof):
    """Require the same captured Jev teacher evidence used in dataset preparation."""
    if proof.get('training_data_kind') != 'captured' or not proof.get('sources'):
        raise ValueError('Evaluation requires captured fr5_jev sources; synthetic data is rejected')
    for source in proof['sources']:
        provenance = source.get('data_provenance', {})
        if (provenance.get('training_data_kind') != 'captured'
                or provenance.get('scope_contract') != 'fr5-whole-task-collection-v1'
                or provenance.get('scope_markers') or provenance.get('camera_markers')
                or not provenance.get('camera_serials') or _synthetic_marker(provenance)):
            raise ValueError('Source is not a captured fr5_jev whole-task recording')
        root = Path(source['trial'])
        scope = json.loads((root / 'scope.json').read_text(encoding='utf-8'))
        if _synthetic_marker(scope):
            raise ValueError('Synthetic or simulated source evidence is rejected')
        with (root / 'continuous/capture.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                if any(_synthetic_marker(item) for item in
                       (frame, frame.get('robot'), frame.get('camera'), frame.get('media_codec'))):
                    raise ValueError('Synthetic or simulated capture evidence is rejected')
        manifest = json.loads((root / 'control-manifest.json').read_text(encoding='utf-8'))
        journal = json.loads((root / 'target-command-journal.json').read_text(encoding='utf-8'))
        jev = 0
        for command in journal:
            teacher = command.get('teacher', {})
            if (_synthetic_marker(command) or command.get('source') != 'agent_teacher'
                    or teacher.get('policy') != 'current_codex_chat+jev'):
                raise ValueError('Command journal is not from the recorded fr5_jev teacher')
            if command.get('executed') is not True:
                continue
            name = teacher.get('decision_id')
            if name == 'scope.json' and teacher.get('phase') == 'initial_open':
                continue
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.startswith(('jev-c', 'llm-c')) or name not in manifest.get('files', {})):
                raise ValueError('Executed command lacks a recorded fr5_jev decision')
            decision = json.loads((root / name).read_text(encoding='utf-8'))
            if (decision.get('scope_sha256') != source.get('source_scope_sha256')
                    or not decision.get('selected_by')):
                raise ValueError('Recorded decision does not match the source')
            jev += name.startswith('jev-c')
        if not jev:
            raise ValueError('Source has no recorded Jev decision')


def check_profiles(info, report):
    if report.get('training_data_kind') != 'captured':
        raise ValueError('Checkpoint was not trained on captured data')
    if info['action_representation'] != training.RELATIVE_REPRESENTATION:
        raise ValueError('Expected six joint deltas from the anchor and absolute gripper')
    if info['image_profile'] not in (training.FULL_IMAGE_PROFILE, training.checkpoints.WORKSPACE_IMAGE_PROFILE):
        raise ValueError('Unsupported image profile')
    if info['observation_context_profile'] not in ('none', 'state-history-physical-v1'):
        raise ValueError('Unsupported observation context profile')


def output_directory(path, dataset, checkpoint):
    path = Path(path).expanduser().resolve()
    root = EVALUATIONS_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Output must be a new directory inside data4/fr5-learning/evaluations')
    for source in (Path(dataset).resolve(), Path(checkpoint).resolve()):
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError('Output must not overlap the dataset or checkpoint')
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    return path


def select_cases(rows, episodes, proof, samples_per_source):
    """Use each physical source's actual frames and recorded event boundaries."""
    if samples_per_source < 1:
        raise ValueError('samples-per-source must be positive')
    by_source, offsets = defaultdict(list), {}
    sources = {str(s['source_index']): s for s in proof['sources']}
    for episode, indices in episodes.items():
        for position, index in enumerate(indices):
            source = rows[index]['_physical_source']
            if source not in sources or rows[index]['_split'] != sources[source]['split']:
                raise ValueError('Frame source/split does not match provenance')
            offsets[index] = (episode, position)
            by_source[source].append(index)
    if set(by_source) != set(sources):
        raise ValueError('A physical source has no exported frames')
    cases = {}

    def add(index, tag, details=None):
        ep, pos = offsets[index]
        row = rows[index]
        case = cases.setdefault(index, {'row_index': index, 'episode': ep, 'position': pos,
            'source_index': int(row['_physical_source']), 'split': row['_split'], 'tags': [],
            'events': [], 'holds': []})
        if tag not in case['tags']:
            case['tags'].append(tag)
        if details is not None:
            case['events' if tag.startswith(('close_', 'open_')) else 'holds'].append(details)

    for source in sorted(by_source, key=int):
        indices = by_source[source]
        count = min(samples_per_source, len(indices))
        if count == 1:
            positions = [len(indices) // 2]
        else:
            positions = [round(i * (len(indices)-1) / (count-1)) for i in range(count)]
        for position in positions:
            add(indices[position], 'uniform')

    for kind, events in training.transition_events(rows, episodes).items():
        for event in events:
            indices = episodes[event['episode']]
            for where, offset in (('before', -1), ('at', 0)):
                index = indices[event['position'] + offset]
                add(index, f'{kind}_{where}', {'kind': kind, 'where': where,
                    'event_row_index': event['index'], 'command_sequence': event['command_sequence'],
                    'old_target': event['old'], 'new_target': event['new']})

    missing_holds = []
    for source, info in sources.items():
        for number, hold in enumerate(info.get('hold_records', [])):
            start, end = float(hold['start_monotonic']), float(hold['end_monotonic'])
            inside = [i for i in by_source[source] if start <= rows[i]['_grid_time'] <= end]
            if not inside:
                missing_holds.append({'source_index': int(source), 'hold_index': number})
                continue
            for tag, index in (('hold_start', inside[0]), ('hold_end', inside[-1])):
                add(index, tag, {'hold_index': number, 'start_monotonic': start, 'end_monotonic': end})
    result = sorted(cases.values(), key=lambda c: (c['source_index'], c['row_index']))
    for number, case in enumerate(result):
        case['case_index'] = number
    return result, missing_holds


class HorizonMetrics:
    def __init__(self):
        self.steps = 0
        self.joint_abs = self.joint_sq = self.grip_abs = 0.0
        self.pred_delta = self.target_delta = 0.0
        self.gripper_state_steps = 0
        self.pred_gripper_delta = self.target_gripper_delta = 0.0
        self.still_steps = self.moving_steps = 0
        self.still_pred = self.still_target = self.moving_pred = self.moving_target = 0.0
        self.switch_steps = self.switch_matches = self.switch_ties = 0

    def add(self, prediction, target, valid, state, event=None):
        import numpy as np
        p, t = prediction[valid], target[valid]
        if not len(p):
            return
        error = np.degrees(p[:, :6] - t[:, :6])
        pred_delta = np.abs(np.degrees(p[:, :6] - state[:6]))
        target_delta = np.abs(np.degrees(t[:, :6] - state[:6]))
        self.steps += len(p)
        self.joint_abs += float(np.abs(error).sum())
        self.joint_sq += float((error ** 2).sum())
        self.grip_abs += float(np.abs(p[:, 6] - t[:, 6]).sum()) * 100
        self.pred_delta += float(pred_delta.sum())
        self.target_delta += float(target_delta.sum())
        if state[7] == 1:
            self.gripper_state_steps += len(p)
            self.pred_gripper_delta += float(np.abs(p[:, 6] - state[6]).sum()) * 100
            self.target_gripper_delta += float(np.abs(t[:, 6] - state[6]).sum()) * 100
        still = np.max(target_delta, axis=1) <= STATIONARY_DEG
        for condition, name in ((still, 'still'), (~still, 'moving')):
            setattr(self, name + '_steps', getattr(self, name + '_steps') + int(condition.sum()))
            setattr(self, name + '_pred', getattr(self, name + '_pred') + float(pred_delta[condition].sum()))
            setattr(self, name + '_target', getattr(self, name + '_target') + float(target_delta[condition].sum()))
        if event is not None:
            old, new = event['old_target'], event['new_target']
            is_old = np.isclose(t[:, 6], old, rtol=0, atol=1e-5)
            is_new = np.isclose(t[:, 6], new, rtol=0, atol=1e-5)
            eligible = is_old | is_new
            old_distance, new_distance = np.abs(p[:, 6] - old), np.abs(p[:, 6] - new)
            ties = eligible & np.isclose(old_distance, new_distance, rtol=0, atol=1e-7)
            matches = eligible & ~ties & ((is_old & (old_distance < new_distance)) |
                                         (is_new & (new_distance < old_distance)))
            self.switch_steps += int(eligible.sum())
            self.switch_matches += int(matches.sum())
            self.switch_ties += int(ties.sum())

    def result(self):
        def ratio(value, denominator):
            return value / denominator if denominator else None
        joint_values = self.steps * 6
        return {'valid_steps': self.steps, 'joint_values': joint_values,
            'joint_mae_deg': ratio(self.joint_abs, joint_values),
            'joint_rmse_deg': math.sqrt(self.joint_sq / joint_values) if joint_values else None,
            'gripper_mae_percentage_points': ratio(self.grip_abs, self.steps),
            'predicted_joint_abs_from_state_deg': ratio(self.pred_delta, joint_values),
            'target_joint_abs_from_state_deg': ratio(self.target_delta, joint_values),
            'gripper_state_valid_steps': self.gripper_state_steps,
            'predicted_gripper_abs_from_state_percentage_points': ratio(self.pred_gripper_delta, self.gripper_state_steps),
            'target_gripper_abs_from_state_percentage_points': ratio(self.target_gripper_delta, self.gripper_state_steps),
            'near_anchor_stationary': {'valid_steps': self.still_steps,
                'predicted_abs_from_state_deg': ratio(self.still_pred, self.still_steps * 6),
                'target_abs_from_state_deg': ratio(self.still_target, self.still_steps * 6)},
            'moving_from_anchor': {'valid_steps': self.moving_steps,
                'predicted_abs_from_state_deg': ratio(self.moving_pred, self.moving_steps * 6),
                'target_abs_from_state_deg': ratio(self.moving_target, self.moving_steps * 6)},
            'switch': {'eligible_valid_steps': self.switch_steps, 'matched_steps': self.switch_matches,
                'ties': self.switch_ties, 'match_fraction': ratio(self.switch_matches, self.switch_steps)}}


class MetricGroup:
    def __init__(self, prefix):
        self.horizons = {name: HorizonMetrics() for name in ('first', 'prefix', 'full')}
        self.prefix = prefix
        self.anchors, self.events = set(), set()
        self.draws = 0

    def add(self, record, event=None):
        import numpy as np
        self.anchors.add((record['source_index'], record['episode'], record['row_index']))
        if event is not None:
            self.events.add((record['episode'], event['event_row_index'], event['kind']))
        self.draws += 1
        prediction = np.asarray(record['absolute_actions'], dtype=np.float64)
        target = np.asarray(record['target_absolute'], dtype=np.float64)
        valid = np.asarray(record['valid_mask'], dtype=bool)
        state = np.asarray(record['state'], dtype=np.float64)
        for name, stop in (('first', 1), ('prefix', self.prefix), ('full', training.CHUNK)):
            self.horizons[name].add(prediction[:stop], target[:stop], valid[:stop], state, event)

    def result(self):
        return {'unique_anchors': len(self.anchors), 'recorded_events': len(self.events),
                'prediction_draws': self.draws,
                **{name: metric.result() for name, metric in self.horizons.items()}}


class Summary:
    def __init__(self, prefix):
        self.groups = {split: {name: MetricGroup(prefix) for name in GROUPS}
                       for split in ('train', 'validation')}

    def add(self, record):
        groups = self.groups[record['split']]
        groups['all'].add(record)
        for tag in record['tags']:
            if tag.startswith(('close_', 'open_')):
                for event in record['events']:
                    if tag == event['kind'] + '_' + event['where']:
                        groups[tag].add(record, event)
            else:
                groups[tag].add(record)

    def result(self):
        return {split: {name: group.result() for name, group in groups.items()}
                for split, groups in self.groups.items()}


def load_base_policy(config, weights):
    """Load the complete original base directly, without any adapter or expert restore."""
    from safetensors.torch import load_file
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy(config)
    original = load_file(str(weights), device='cpu')
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    mapped = {key if key.startswith('model.') else 'model.' + key: value for key, value in fixed.items()}
    policy.load_state_dict(mapped, strict=True)
    return policy


def model_input(case, rows, episodes, dataset, pre, info):
    """Expose the model to the recorded image/task and past observations only."""
    index = case['row_index']
    sample = dataset[index]
    task = sample['task']
    sample_input = {index: {training.IMAGE_KEY: sample[training.IMAGE_KEY], 'task': task}}
    batch, valid = training.batch_for([index], rows, episodes, sample_input, pre, labels=False,
        image_profile=info['image_profile'], observation_context_profile=info['observation_context_profile'])
    if valid is not None or batch.get('action') is not None or any(
            'hold' in key or 'future' in key for key in batch):
        raise ValueError('Evaluation model input must not contain labels or future hold information')
    batch = {key: value for key, value in batch.items() if key == 'task' or key.startswith('observation.')}
    return batch, task


def make_record(case, row, task, draw, draw_seed, native, model_actions, absolute, target, mask, model):
    return {**case, 'model': model, 'draw': draw, 'seed': draw_seed,
        'source_capture_frame': int(row['_source_capture_frame']),
        'source_camera_time_monotonic': float(row['_camera_time']),
        'source_grid_time_monotonic': float(row['_grid_time']),
        'state': [float(x) for x in row['observation.state']], 'task': task,
        'native_normalized_actions': native.tolist(), 'model_actions': model_actions.tolist(),
        'absolute_actions': absolute.tolist(), 'target_absolute': target,
        'valid_mask': mask}


def evaluate_model(model_name, policy, pre, post, info, rows, episodes, dataset, cases, draws, seed, output):
    import torch
    summary = Summary(info['n_action_steps'])
    converter = training.load_relative_helper()
    with output.open('x', encoding='utf-8') as stream:
        for case in cases:
            index = case['row_index']
            row = rows[index]
            batch, task = model_input(case, rows, episodes, dataset, pre, info)
            if task != info['source_tasks'][str(case['source_index'])]:
                raise ValueError('Recorded task differs from source provenance')
            target, mask = training.chunk_for_anchor(rows, episodes[case['episode']], case['position'])
            for draw in range(draws):
                draw_seed = seed + case['case_index'] * draws + draw
                torch.manual_seed(draw_seed)
                policy.reset()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    native = policy.predict_action_chunk(batch)
                    model_actions = post(native.detach().clone()).cpu().float()
                    absolute = converter.to_absolute_actions(model_actions,
                        torch.tensor([row['observation.state']], dtype=torch.float32),
                        input_representation=training.RELATIVE_REPRESENTATION)
                native = native.detach().cpu().float()
                if (tuple(native.shape) != (1, 50, 7) or tuple(model_actions.shape) != (1, 50, 7)
                        or tuple(absolute.shape) != (1, 50, 7)
                        or not all(bool(torch.isfinite(t).all()) for t in (native, model_actions, absolute))):
                    raise ValueError('Expected finite native 50x7 action chunks')
                record = make_record(case, row, task, draw, draw_seed, native[0], model_actions[0],
                                     absolute[0], target, mask, model_name)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                summary.add(record)
    return summary.result()


def run(args):
    if args.samples_per_source < 1 or args.draws < 1 or args.gpu < 0 or args.seed < 0:
        raise ValueError('samples-per-source and draws must be positive; gpu and seed must be nonnegative')
    output = output_directory(args.output, args.dataset, args.checkpoint)
    dataset_path, root = Path(args.dataset).resolve(), Path(args.checkpoint).resolve()
    report = json.loads((root / 'training-report.json').read_text(encoding='utf-8'))
    info = serving.inspect_checkpoint(root)
    check_profiles(info, report)
    declared = json.loads((dataset_path / 'meta/stream_provenance.json').read_text(encoding='utf-8'))
    require_captured_jev(declared)
    training.legacy.require_runtime_versions()
    training.legacy.configure_environment()
    selected = training.legacy.gpu_status(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected['index'])
    import torch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    from lerobot.processor import DataProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    rows, episodes, dataset, proof, fingerprint = training.read_dataset(dataset_path)
    if proof != declared or fingerprint != info['manifest']['dataset_fingerprint']:
        raise ValueError('Dataset does not match the checkpoint training dataset')
    cases, missing_holds = select_cases(rows, episodes, proof, args.samples_per_source)
    if args.seed + len(cases) * args.draws >= 2**63:
        raise ValueError('Seed and draw count exceed the supported range')
    info['source_tasks'] = {str(s['source_index']): s['task_instruction'] for s in proof['sources']}
    serving.validate_normalization(root)
    config = PI05Config.from_pretrained(root, local_files_only=True, cli_overrides=['--device=cpu'])
    training.legacy.register_padding_step()
    pre = DataProcessorPipeline.from_pretrained(root, config_filename='policy_preprocessor.json',
        local_files_only=True, overrides={'tokenizer_processor': {'tokenizer_name': str(root / 'tokenizer')},
                                          'device_processor': {'device': 'cuda'}})
    post = DataProcessorPipeline.from_pretrained(root, config_filename='policy_postprocessor.json',
        local_files_only=True, to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    training.load_relative_helper().assert_manual_pipeline(config, preprocessor=pre, postprocessor=post)
    output.mkdir(parents=True, exist_ok=False)
    models = ('fine_tuned', 'base_control') if args.base_control else ('fine_tuned',)
    results = {}
    for model_name in models:
        weights = Path(info['manifest']['base']['weights_path'])
        config.device = 'cpu'
        if model_name == 'fine_tuned':
            policy = training.load_policy(config, weights, checkpoint=root)
        else:
            policy = load_base_policy(config, weights)
        training.legacy.gpu_status(selected['index'])
        policy.to('cuda').eval().requires_grad_(False)
        config.device = 'cuda'
        results[model_name] = evaluate_model(model_name, policy, pre, post, info, rows, episodes,
            dataset, cases, args.draws, args.seed, output / (model_name + '.jsonl'))
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    result = {'evaluation': 'open_loop_recorded_label_fit_not_task_success',
        'dataset': str(dataset_path), 'checkpoint': str(root), 'dataset_matches_checkpoint': True,
        'samples_per_source': args.samples_per_source, 'draws_per_anchor': args.draws,
        'seed': args.seed, 'prefix_steps': info['n_action_steps'],
        'image_profile': info['image_profile'], 'observation_context_profile': info['observation_context_profile'],
        'action_representation': info['action_representation'], 'missing_hold_frames': missing_holds,
        'stationary_definition': f'all six recorded target joints within {STATIONARY_DEG} deg of anchor state',
        'denominators': 'valid target steps across draws; joint metrics use six joint values per valid step',
        'switch_definition': 'target equals recorded old/new command; prediction nearest that command; ties count as unmatched',
        'meaning': 'Recorded open-loop label fit only; draws are not physical attempts or object generalization.',
        'models': results}
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'cases': len(cases), 'models': list(results)}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, help='physical GPU index')
    parser.add_argument('--output', required=True, type=Path, help='new directory under data4/fr5-learning/evaluations')
    parser.add_argument('--samples-per-source', type=int, default=12)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--base-control', action='store_true', help='also evaluate the unadapted complete base')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
