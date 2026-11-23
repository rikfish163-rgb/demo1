#!/usr/bin/env python3
"""Replay recorded source observations through a saved pi0.5 checkpoint.

The output compares independent open-loop predictions with the recorded
source trajectory. It does not connect to or command a robot.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import os
from pathlib import Path

from fr5_lerobot.diagnostics import assess_jev_pi05_support as support
from fr5_lerobot.training import evaluate_jev_pi05 as evaluation


SOURCE_INDEX = 6
FIRST_FRAME, LAST_FRAME = 374, 400
SEEDS = (20261493, 20261494, 20261495)
PREFIXES = (3, 6, 10)
HISTORY_OFFSETS_S = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
ROOT = Path('/home/data4/hts/demo1/fr5-learning')


def select_cases(rows, episodes, source_index=SOURCE_INDEX, first_frame=FIRST_FRAME, last_frame=LAST_FRAME):
    """Select dataset rows by their recorded source-frame mapping."""
    if first_frame > last_frame:
        raise ValueError('First source frame must not exceed last source frame')
    cases = []
    for episode, indices in episodes.items():
        for position, index in enumerate(indices):
            row = rows[index]
            if (int(row['_physical_source']) == source_index
                    and first_frame <= int(row['_source_capture_frame']) <= last_frame):
                cases.append({'row_index': index, 'episode': episode, 'position': position,
                              'source_index': source_index, 'split': row['_split'],
                              'tags': ['dense_recorded_observation'], 'events': [], 'holds': []})
    cases.sort(key=lambda case: case['row_index'])
    if not cases:
        raise ValueError('No dataset rows match the selected source and frame range')
    if ((source_index, first_frame, last_frame) == (SOURCE_INDEX, FIRST_FRAME, LAST_FRAME)
            and not any(rows[case['row_index']]['_source_capture_frame'] == 383 for case in cases)):
        raise ValueError('The selected source rows do not include the recorded frame 383')
    for number, case in enumerate(cases):
        case['case_index'] = number
    return cases


def cycle_for_row(source, row):
    """Associate each anchor with the latest recorded close command."""
    cycles = sorted(source.cycles, key=lambda cycle: cycle['close_command_time'])
    if not cycles:
        raise ValueError('Selected source has no recorded release cycle')
    index = bisect_right([cycle['close_command_time'] for cycle in cycles], row['_grid_time']) - 1
    return cycles[max(index, 0)]


def history_with_frames(row, source):
    """Identify the actual capture frames used by the existing history input."""
    history = row['_physical_observation_history']
    if len(history) != len(HISTORY_OFFSETS_S):
        raise ValueError('Expected six physical observation history entries')
    captures = sorted((float(frame['camera']['source_timestamp_monotonic']), frame['frame_index'])
                      for frame in source.by_index.values())
    times = [item[0] for item in captures]
    result = []
    for offset, item in zip(HISTORY_OFFSETS_S, history):
        if item is None:
            result.append({'requested_age_s': offset, 'source_frame': None, 'observation': None})
            continue
        timestamp = float(row['_camera_time']) - float(item['age_s'])
        pos = bisect_left(times, timestamp)
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(times)]
        if not candidates:
            raise ValueError('Physical history frame is unavailable')
        index = min(candidates, key=lambda i: abs(times[i] - timestamp))
        if abs(times[index] - timestamp) > 1e-5:
            raise ValueError('Physical history does not match a source camera frame')
        result.append({'requested_age_s': offset, 'source_frame': captures[index][1],
                       'camera_time_monotonic': captures[index][0], 'observation': item})
    return result


def analyze_draw(record, source, cycle, kin, execution_prefix=max(PREFIXES)):
    """Describe unchanged output ticks and their corresponding source feedback."""
    anchor = source.by_index[record['source_capture_frame']]
    stamp = float(anchor['robot']['received_monotonic'])
    anchor_feedback = source.feedback_at(stamp, cycle)
    last_command = source.last_gripper_command(stamp)
    measured = (float(anchor_feedback['raw_gripper_position_percent'])
                if anchor_feedback.get('gripper_feedback_valid') is True else None)
    commanded = last_command.get('target_percent')
    ticks = []
    for index in range(max(PREFIXES)):
        if not record['valid_mask'][index]:
            continue
        tick = support.prediction_detail(record, index, source, cycle, kin, execution_prefix)
        projected = support.projected_gripper_percent(tick['raw_gripper_fraction'])
        tick['projected_integer_percent'] = projected
        tick['increase_over_anchor_measured_pp'] = None if measured is None else projected - measured
        tick['increase_over_anchor_last_command_pp'] = None if commanded is None else projected - commanded
        tick['last_move_gripper_at_nominal_time'] = source.last_gripper_command(tick['nominal_time_monotonic'])
        ticks.append(tick)
    prefixes = {}
    for prefix in PREFIXES:
        found = {}
        for threshold in support.OPEN_THRESHOLDS:
            found[str(threshold)] = next((tick for tick in ticks if tick['tick'] <= prefix
                                          and tick['raw_gripper_percent'] >= threshold), None)
        prefixes[str(prefix)] = found
    return {'anchor_feedback': anchor_feedback, 'anchor_last_move_gripper': last_command,
            'ticks': ticks, 'prefixes': prefixes}


def summarize(samples):
    result = {}
    for prefix in PREFIXES:
        by_threshold = {}
        for threshold in support.OPEN_THRESHOLDS:
            matched = [(item, item['analysis']['prefixes'][str(prefix)][str(threshold)]) for item in samples]
            matched = [(item, tick) for item, tick in matched if tick is not None]
            before = [(item, tick) for item, tick in matched if tick['seconds_from_release_command'] < 0]
            heights = [tick['predicted_grasp_vs_release_reference']['height_along_table_normal_mm']
                       for _, tick in matched if tick['predicted_grasp_vs_release_reference'] is not None]
            measured = [tick['latest_preceding_source_feedback']['measured_grasp_vs_release_reference']
                        ['height_along_table_normal_mm'] for _, tick in matched
                        if tick['latest_preceding_source_feedback'].get('gripper_feedback_valid') is True
                        and 'measured_grasp_vs_release_reference' in tick['latest_preceding_source_feedback']]
            by_threshold[str(threshold)] = {
                'draws_with_first_opening': len(matched),
                'distinct_source_anchors': len({item['record']['source_capture_frame'] for item, _ in matched}),
                'draws_before_recorded_release': len(before),
                'predicted_height_minus_reference_mm': support.distribution(heights),
                'preceding_valid_source_height_minus_reference_mm': support.distribution(measured),
            }
        result[str(prefix)] = by_threshold
    return result


def projected_comparison(samples, prefix, field, *, include_draws=False):
    """Compare projected model targets with one fixed recorded anchor baseline."""
    eligible, opened, draw_details = [], [], []
    missing = 0
    for item in samples:
        record, analysis = item['record'], item['analysis']
        points = [tick for tick in analysis['ticks'] if tick['tick'] <= prefix]
        available = [tick for tick in points if tick[field] is not None]
        eligible.extend((item, tick) for tick in available)
        missing += len(points) - len(available)
        changes = [tick for tick in available if tick[field] >= 2.]
        opened.extend((item, tick) for tick in changes)
        if include_draws:
            draw_details.append({'draw': record['draw'], 'seed': record['seed'],
                                 'eligible_points': len(available),
                                 'points_at_least_2pp': len(changes),
                                 'first_at_least_2pp': (projected_witness(item, changes[0], field)
                                                        if changes else None)})
    max_pair = max(eligible, key=lambda pair: pair[1][field]) if eligible else None
    result = {
        'eligible_draws': sum(any(tick['tick'] <= prefix and tick[field] is not None
                                  for tick in item['analysis']['ticks']) for item in samples),
        'eligible_points': len(eligible), 'missing_points': missing,
        'draws_at_least_2pp': sum(any(tick['tick'] <= prefix and tick[field] is not None and tick[field] >= 2.
                                       for tick in item['analysis']['ticks']) for item in samples),
        'points_at_least_2pp': len(opened),
        'anchors_at_least_2pp': len({item['record']['source_capture_frame'] for item, _ in opened}),
        'maximum_increase_pp': None if max_pair is None else max_pair[1][field],
        'maximum_witness': None if max_pair is None else projected_witness(*max_pair, field),
    }
    if include_draws:
        result['draws'] = draw_details
    return result


def projected_witness(item, tick, field):
    record = item['record']
    return {'source_index': record['source_index'], 'source_capture_frame': record['source_capture_frame'],
            'row_index': record['row_index'], 'draw': record['draw'], 'seed': record['seed'],
            'tick': tick['tick'], 'raw_gripper_fraction': tick['raw_gripper_fraction'],
            'raw_gripper_percent': tick['raw_gripper_percent'],
            'projected_integer_percent': tick['projected_integer_percent'], 'increase_pp': tick[field],
            'seconds_from_release_command': tick['seconds_from_release_command'],
            'predicted_grasp_vs_release_reference': tick['predicted_grasp_vs_release_reference'],
            'nearest_source_feedback': tick['nearest_source_feedback'],
            'latest_preceding_source_feedback': tick['latest_preceding_source_feedback']}


def summarize_projected_targets(samples):
    baselines = {'anchor_measured': 'increase_over_anchor_measured_pp',
                 'anchor_last_move_gripper': 'increase_over_anchor_last_command_pp'}

    def compare(rows, include_draws=False):
        return {str(prefix): {name: projected_comparison(rows, prefix, field, include_draws=include_draws)
                              for name, field in baselines.items()} for prefix in PREFIXES}

    frames = sorted({item['record']['source_capture_frame'] for item in samples})
    anchors = []
    for frame in frames:
        group = [item for item in samples if item['record']['source_capture_frame'] == frame]
        first = group[0]
        anchors.append({'source_index': first['record']['source_index'], 'source_capture_frame': frame,
                        'row_index': first['record']['row_index'],
                        'anchor_feedback': first['analysis']['anchor_feedback'],
                        'anchor_last_move_gripper': first['analysis']['anchor_last_move_gripper'],
                        'prefixes': compare(group, include_draws=True)})
    return {'prefixes': compare(samples), 'anchors': anchors}


def summarize_saved(predictions_path, output):
    """Summarize a completed replay without loading a model or using a GPU."""
    predictions_path, output = Path(predictions_path).resolve(), Path(output).resolve()
    diagnostics = (ROOT / 'diagnostics').resolve()
    if not output.is_relative_to(diagnostics) or output == diagnostics:
        raise ValueError('Output must be a new file under data4/fr5-learning/diagnostics')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    source_report = predictions_path.parent / 'report.json'
    meta = json.loads(source_report.read_text(encoding='utf-8'))
    with predictions_path.open(encoding='utf-8') as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    if not samples:
        raise ValueError('Recorded replay contains no predictions')
    report = {'analysis': 'recorded_replay_projected_targets_not_closed_loop',
              'source_report': str(source_report), 'source_predictions': str(predictions_path),
              'dataset': meta['dataset'], 'checkpoint': meta['checkpoint'],
              'source_index': meta['source_index'], 'source_trial': meta['source_trial'],
              'seeds_per_anchor': meta['seeds_per_anchor'], 'recorded_reference': meta['recorded_reference'],
              'raw_threshold_prefixes': meta['prefixes'],
              'projection': 'clamp raw fraction to 0..1 and round 100*fraction to integer percent',
              'baselines': 'valid recorded anchor feedback and last recorded MoveGripper wire target at anchor',
              'projected_targets': summarize_projected_targets(samples),
              'limitations': ['Recorded baselines do not simulate the last actual model target or contact.',
                  'These are independent recorded-observation predictions, not a closed-loop rollout or task success.',
                  'Feedback at predicted timestamps is source evidence, not evidence the prediction was executed.']}
    if 'recorded_cycles' in meta:
        report['recorded_cycles'] = meta['recorded_cycles']
        report['row_and_frame_mapping'] = meta['row_and_frame_mapping']
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'draws': len(samples),
                      'anchors': len(report['projected_targets']['anchors'])}, ensure_ascii=False))
    return report


def configure_runtime(output):
    """Keep runtime caches and temporary files inside this new diagnostic."""
    paths = {'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
             'HF_DATASETS_CACHE': 'huggingface/datasets', 'HF_LEROBOT_HOME': 'huggingface/lerobot',
             'HF_XET_CACHE': 'xet', 'TORCH_HOME': 'torch', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
             'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'XDG_CACHE_HOME': 'xdg',
             'TMPDIR': 'tmp'}
    for name, relative in paths.items():
        path = output / 'runtime_cache' / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')


def run(dataset_path, checkpoint, output, gpu, source_index=SOURCE_INDEX,
        first_frame=FIRST_FRAME, last_frame=LAST_FRAME):
    dataset_path, checkpoint, output = (Path(p).expanduser().resolve() for p in (dataset_path, checkpoint, output))
    diagnostics = (ROOT / 'diagnostics').resolve()
    if not output.is_relative_to(diagnostics) or output == diagnostics:
        raise ValueError('Output must be a new directory under data4/fr5-learning/diagnostics')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if gpu < 0:
        raise ValueError('GPU index must be nonnegative')
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(output)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    training, serving = evaluation.training, evaluation.serving
    training.legacy.require_runtime_versions()
    selected = training.legacy.gpu_status(gpu)
    info = serving.inspect_checkpoint(checkpoint)
    training_report = json.loads((checkpoint / 'training-report.json').read_text(encoding='utf-8'))
    evaluation.check_profiles(info, training_report)
    training.checkpoints.configured_action_steps({'n_action_steps': info['n_action_steps']}, 50)
    if (info['observation_context_profile'] != 'state-history-physical-v1'
            or info['image_profile'] != training.checkpoints.WORKSPACE_IMAGE_PROFILE):
        raise ValueError('Expected physical history with full and workspace camera views')

    import torch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    from lerobot.processor import DataProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    rows, episodes, dataset, proof, fingerprint = training.read_dataset(dataset_path)
    if fingerprint != info['manifest']['dataset_fingerprint']:
        raise ValueError('Dataset does not match the checkpoint training dataset')
    source_info = next((source for source in proof['sources'] if source['source_index'] == source_index), None)
    if source_info is None:
        raise ValueError(f'Physical source {source_index} is absent from dataset provenance')
    cases = select_cases(rows, episodes, source_index, first_frame, last_frame)
    info['source_tasks'] = {str(source['source_index']): source['task_instruction'] for source in proof['sources']}
    kin = support.GraspKinematics(support.PROJECT / 'hardware/fr5.urdf', support.OFFSET_MM)
    source = support.SourceEvidence(source_info, kin)
    if (source_index == SOURCE_INDEX and
            (len(source.cycles) != 1 or source.cycles[0]['reference']['status'] != 'stable_jaw_contact_proxy')):
        raise ValueError('Source 6 requires its recorded stable release-reference cycle')
    case_cycles = {case['row_index']: cycle_for_row(source, rows[case['row_index']]) for case in cases}
    selected_cycles = {cycle['cycle']: cycle for cycle in case_cycles.values()}
    serving.validate_normalization(checkpoint)
    config = PI05Config.from_pretrained(checkpoint, local_files_only=True, cli_overrides=['--device=cpu'])
    if (config.num_inference_steps != 10 or config.compile_model is not False
            or config.n_action_steps != info['n_action_steps']):
        raise ValueError('Expected the sealed prefix and ten denoising steps without model compilation')
    training.legacy.register_padding_step()
    pre = DataProcessorPipeline.from_pretrained(checkpoint, config_filename='policy_preprocessor.json',
        local_files_only=True, overrides={'tokenizer_processor': {'tokenizer_name': str(checkpoint / 'tokenizer')},
                                          'device_processor': {'device': 'cuda'}})
    post = DataProcessorPipeline.from_pretrained(checkpoint, config_filename='policy_postprocessor.json',
        local_files_only=True, to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    converter = training.load_relative_helper()
    converter.assert_manual_pipeline(config, preprocessor=pre, postprocessor=post)
    policy = training.load_policy(config, Path(info['manifest']['base']['weights_path']), checkpoint=checkpoint)
    training.legacy.gpu_status(selected['index'])
    policy.to('cuda').eval().requires_grad_(False)
    config.device = 'cuda'

    samples = []
    with (output / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for case in cases:
            row = rows[case['row_index']]
            cycle = case_cycles[case['row_index']]
            history = history_with_frames(row, source)
            batch, task = evaluation.model_input(case, rows, episodes, dataset, pre, info)
            if task != source_info['task_instruction']:
                raise ValueError('Recorded task differs from source provenance')
            target, mask = training.chunk_for_anchor(rows, episodes[case['episode']], case['position'])
            for draw, seed in enumerate(SEEDS):
                torch.manual_seed(seed)
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
                        or not all(bool(torch.isfinite(value).all()) for value in (native, model_actions, absolute))):
                    raise ValueError('Expected finite native 50x7 action chunks')
                record = evaluation.make_record(case, row, task, draw, seed, native[0], model_actions[0],
                                                absolute[0], target, mask, 'fine_tuned')
                analysis = analyze_draw(record, source, cycle, kin, info['n_action_steps'])
                if len(source.cycles) > 1:
                    analysis.update(recorded_cycle=cycle['cycle'], recorded_reference=cycle['reference'],
                                    recorded_release_command_time=cycle['release_command_time'])
                item = {'record': record, 'physical_history': history,
                        'model_input_task': training.observation_history.render_task(
                            task, row['_physical_observation_history'], info['observation_context_profile']),
                        'analysis': analysis}
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                samples.append(item)
            print(json.dumps({'source_frame': row['_source_capture_frame'],
                              'row_index': case['row_index'], 'completed_draws': len(samples)}), flush=True)

    cycle = next(iter(selected_cycles.values())) if len(selected_cycles) == 1 else None
    report = {'analysis': 'independent_recorded_observation_replay_not_closed_loop',
              'dataset': str(dataset_path), 'checkpoint': str(checkpoint), 'source_index': source_index,
              'source_trial': str(source.path), 'source_split': source.split,
              'source_frame_range_inclusive': [first_frame, last_frame],
              'row_and_frame_mapping': [{'row_index': case['row_index'],
                                         'source_capture_frame': rows[case['row_index']]['_source_capture_frame']}
                                        for case in cases],
              'seeds_per_anchor': list(SEEDS), 'seed_policy': 'torch.manual_seed(seed), then policy.reset() per draw',
              'rng': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                      'physical_gpu_index': selected['index'], 'device': torch.cuda.get_device_name(0),
                      'autocast_dtype': 'bfloat16'},
              'input': {'observation_context_profile': info['observation_context_profile'],
                        'image_profile': info['image_profile'], 'action_representation': info['action_representation'],
                        'denoising_steps': config.num_inference_steps,
                        'compile_model': config.compile_model, 'native_action_shape': [50, 7],
                        'checkpoint_n_action_steps': info['n_action_steps'],
                        'checkpoint_steps_completed': training_report.get('steps_completed'),
                        'history_offsets_s': list(HISTORY_OFFSETS_S)},
              'recorded_reference': cycle['reference'] if cycle else None,
              'recorded_release_command_time': cycle['release_command_time'] if cycle else None,
              'table_normal': source.normal.tolist(), 'table_offset_mm': source.offset,
              'grasp_offset_flange_mm': list(support.OFFSET_MM), 'prefixes': summarize(samples),
              'projected_targets': summarize_projected_targets(samples),
              'limitations': ['Independent recorded observations are not a closed-loop rollout or task success.',
                  'Opening thresholds describe raw model targets, not physical gripper motion, contact or a drop.',
                  'Recorded source feedback and the stable jaw reference do not prove object support.',
                  'Each prefix truncates the same unmodified 50-step prediction; no action was executed.']}
    if len(source.cycles) > 1:
        report['recorded_cycles'] = list(selected_cycles.values())
        for mapping in report['row_and_frame_mapping']:
            mapping['recorded_cycle'] = case_cycles[mapping['row_index']]['cycle']
    if len(source.cycles) > 1 or cycle['reference']['status'] != 'stable_jaw_contact_proxy':
        report['limitations'][2] = 'Recorded source feedback and jaw references do not prove object support.'
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'anchors': len(cases), 'draws': len(samples)}, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, help='idle physical GPU index')
    parser.add_argument('--source-index', type=int, default=None)
    parser.add_argument('--first-frame', type=int, default=None)
    parser.add_argument('--last-frame', type=int, default=None)
    parser.add_argument('--saved-predictions', type=Path,
                        help='completed replay predictions.jsonl to summarize on CPU')
    args = parser.parse_args(argv)
    if args.saved_predictions is not None:
        if any(value is not None for value in (args.dataset, args.checkpoint, args.gpu,
                                               args.source_index, args.first_frame, args.last_frame)):
            parser.error('--saved-predictions cannot be combined with replay arguments')
        return summarize_saved(args.saved_predictions, args.output)
    if args.dataset is None or args.checkpoint is None or args.gpu is None:
        parser.error('replay requires --dataset, --checkpoint and --gpu')
    run(args.dataset, args.checkpoint, args.output, args.gpu,
        source_index=SOURCE_INDEX if args.source_index is None else args.source_index,
        first_frame=FIRST_FRAME if args.first_frame is None else args.first_frame,
        last_frame=LAST_FRAME if args.last_frame is None else args.last_frame)


if __name__ == '__main__':
    main()
