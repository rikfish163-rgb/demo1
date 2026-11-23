"""Evaluate recorded hold, release, and episode-end observations offline."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path('/home/data4/hts/demo1/fr5-learning')
DATASET = ROOT / 'datasets/fr5-joint-stream-20261102-v2-compacted'
BASE_SEED = 20261105
sys.dont_write_bytecode = True


def select_cases(rows, episodes, proof, events):
    """Select observed hold endpoints, open transitions, and episode ends."""
    sources = {int(source['source_index']): source for source in proof['sources']}
    splits = {source: info['split'] for source, info in sources.items()}
    if (len(proof['sources']) != 8 or set(sources) != set(range(8))
            or splits != {source: 'validation' if source == 7 else 'train'
                          for source in range(8)}):
        raise ValueError('Expected the original seven train sources and validation source 7')

    source_episodes = defaultdict(list)
    for episode, indices in episodes.items():
        if not indices:
            raise ValueError('Empty episode')
        source = int(rows[indices[0]]['_physical_source'])
        if source not in sources or any(
                int(rows[index]['_physical_source']) != source
                or rows[index]['_split'] != splits[source] for index in indices):
            raise ValueError('Episode source or split differs from provenance')
        source_episodes[source].append((episode, indices))
    if set(source_episodes) != set(sources) or any(len(value) != 1 for value in source_episodes.values()):
        raise ValueError('Expected one episode for each physical source')

    cases = []

    def add_case(kind, episode, position, **details):
        index = episodes[episode][position]
        row = rows[index]
        cases.append({'kind': kind, 'source_index': int(row['_physical_source']),
                      'split': row['_split'], 'episode': episode, 'position': position,
                      'row_index': index, 'grid_time': float(row['_grid_time']), **details})

    hold_count = 0
    for source, info in sorted(sources.items()):
        for hold_index, hold in enumerate(info['hold_records']):
            start, end = hold['start_monotonic'], hold['end_monotonic']
            matches = []
            for episode, indices in source_episodes[source]:
                inside = [(position, index) for position, index in enumerate(indices)
                          if start <= rows[index]['_grid_time'] <= end]
                if inside:
                    matches.append((episode, inside))
            if len(matches) != 1 or len(matches[0][1]) < 2:
                raise ValueError('Recorded hold must contain two observed endpoints in one episode')
            episode, inside = matches[0]
            for kind, (position, _) in (('hold_start', inside[0]), ('hold_end', inside[-1])):
                add_case(kind, episode, position, hold_index=hold_index,
                         repetition=hold['repetition'], hold_start_time=float(start),
                         hold_end_time=float(end))
            hold_count += 1

    for event_number, event in enumerate(events['open']):
        episode, position = event['episode'], event['position']
        indices = episodes[episode]
        if position < 1 or position >= len(indices) or indices[position] != event['index']:
            raise ValueError('Open transition does not match its episode')
        for kind, offset in (('open_before', -1), ('open_at', 0)):
            add_case(kind, episode, position + offset, open_event=event_number,
                     open_row_index=event['index'], command_sequence=event['command_sequence'],
                     previous_gripper_target=float(event['old']),
                     new_gripper_target=float(event['new']))

    for source in sorted(source_episodes):
        episode, indices = source_episodes[source][0]
        add_case('episode_end', episode, len(indices) - 1)

    if hold_count != 10 or len(events['open']) != 10 or len(cases) != 48:
        raise ValueError('Expected ten holds, ten open transitions, and eight episode ends')
    for case_index, case in enumerate(cases):
        case['case_index'] = case_index
    return cases


def return_directions(cases, rows, episodes, trainer, *, cup_relative=False, aligned=False):
    """Use the real tenth target after each hold end to define its return axis."""
    directions = {}
    for case in cases:
        if case['kind'] != 'hold_end':
            continue
        targets, valid = trainer.chunk_for_anchor(rows, episodes[case['episode']], case['position'], **(
            {'action_key': '_camera_aligned_actions' if aligned else '_cup_relative_action'} if cup_relative else {}))
        key = (case['source_index'], case['hold_index'])
        if not all(valid[:10]):
            directions[key] = (None, 'hold end has fewer than ten real future targets')
            continue
        state = rows[case['row_index']]['_cup_relative_state' if cup_relative else 'observation.state']
        delta = [float(targets[9][joint]) - float(state[joint]) for joint in range(3 if cup_relative else 6)]
        magnitude = math.sqrt(sum(value * value for value in delta))
        if magnitude == 0:
            directions[key] = (None, 'hold end tenth real target has zero joint displacement')
        else:
            directions[key] = ([value / magnitude for value in delta], None)
    return directions


def action_metrics(actions, targets, valid, state, kind, direction=None, *, cup_relative=False):
    """Compare predictions with valid targets and describe hold movement."""
    scale = 1000. if cup_relative else 180.0 / math.pi
    axes = range(3 if cup_relative else 6)
    error_suffix = 'tcp_xyz_rmse_mm' if cup_relative else 'joint_rmse_deg'
    unit = 'mm' if cup_relative else 'deg'
    result = {}
    for horizon in (10, 50):
        positions = [index for index in range(horizon) if valid[index]]
        result[f'first{horizon}_valid_steps'] = len(positions)
        if positions:
            errors = [(float(actions[i][j]) - float(targets[i][j])) * scale
                      for i in positions for j in axes]
            result[f'first{horizon}_{error_suffix}'] = math.sqrt(
                sum(value * value for value in errors) / len(errors))
            result[f'first{horizon}_gripper_mae_fraction'] = sum(
                abs(float(actions[i][6]) - float(targets[i][6])) for i in positions) / len(positions)
        else:
            result[f'first{horizon}_{error_suffix}'] = None
            result[f'first{horizon}_gripper_mae_fraction'] = None

    if kind in ('hold_start', 'hold_end'):
        pred_delta = [(float(actions[9][j]) - float(state[j])) * scale for j in axes]
        target_delta = ([(float(targets[9][j]) - float(state[j])) * scale for j in axes]
                        if valid[9] else None)
        valid_first10 = [i for i in range(10) if valid[i]]
        result.update({
            f'predicted_step10_delta_{unit}': pred_delta,
            f'target_step10_delta_{unit}': target_delta,
            f'predicted_first10_max_abs_delta_{unit}': max(
                abs((float(actions[i][j]) - float(state[j])) * scale)
                for i in range(10) for j in axes),
            f'target_first10_max_abs_delta_{unit}': (max(
                abs((float(targets[i][j]) - float(state[j])) * scale)
                for i in valid_first10 for j in axes) if valid_first10 else None),
            f'predicted_step10_return_projection_{unit}': (sum(x * y for x, y in zip(pred_delta, direction))
                                                    if direction is not None else None),
            f'target_step10_return_projection_{unit}': (sum(x * y for x, y in zip(target_delta, direction))
                                                 if direction is not None and target_delta is not None else None)})
    return result


def group_summary(summaries, *, cup_relative=False):
    groups = {}
    error_metrics = ('first10_joint_rmse_deg', 'first50_joint_rmse_deg',
                     'first10_gripper_mae_fraction', 'first50_gripper_mae_fraction')
    hold_metrics = ('predicted_first10_max_abs_delta_deg', 'target_first10_max_abs_delta_deg',
                    'predicted_step10_return_projection_deg', 'target_step10_return_projection_deg')
    if cup_relative:
        error_metrics = ('first10_tcp_xyz_rmse_mm','first50_tcp_xyz_rmse_mm', *error_metrics[2:])
        hold_metrics = tuple(key.replace('_deg','_mm') for key in hold_metrics)
    for case in summaries:
        group = groups.setdefault(case['split'], {}).setdefault(case['kind'], [])
        group.append(case)
    result = {}
    for split, kinds in groups.items():
        result[split] = {}
        for kind, cases in kinds.items():
            draws = [draw for case in cases for draw in case['draws']]
            values = {}
            metric_names = error_metrics + (hold_metrics if kind in ('hold_start', 'hold_end') else ())
            for name in metric_names:
                present = [draw[name] for draw in draws if draw[name] is not None]
                values[name + '_mean'] = sum(present) / len(present) if present else None
            result[split][kind] = {'observations': len(cases), 'draws': len(draws), **values}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--draws', type=int, default=4)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error('--draws must be positive')
    checkpoint, dataset_path, output = (path.resolve() for path in
                                        (args.checkpoint, args.dataset, args.output))
    if dataset_path != DATASET:
        parser.error('--dataset must be the original eight-source dataset')
    if ROOT / 'audits' not in output.parents or output.exists():
        parser.error('--output must be a new directory under data4 audits')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(checkpoint))
    import serve_stream_pi05 as service

    model = service.Inference(checkpoint, gpu=args.gpu)
    trainer = model.training
    torch = model.torch
    torch.set_num_threads(4)
    rows, episodes, dataset, proof, fingerprint = trainer.read_dataset(dataset_path)
    representation = model.info['action_representation']
    cup_delta = representation in ('cup-relative-tcp-delta-v1', 'cup-relative-tcp-tick-delta-v1')
    cup_relative = cup_delta or representation == 'cup-relative-tcp-v1'
    if cup_relative:
        from cup_relative_stream_data import attach_cup_relative
        attach_cup_relative(rows, episodes, proof, dataset_path)
    history = service.observation_history
    profile = model.info['observation_context_profile']
    aligned = profile == getattr(history, 'CUP_ALIGNED_PROFILE', None)
    if profile in history.CUP_HISTORY_PROFILES:
        trainer.attach_initial_cup(rows, episodes, proof)
    if profile in getattr(history, 'CUP_TRACKING_PROFILES', (getattr(history, 'CUP_TRACKING_PROFILE', None),)):
        from cup_relative_stream_data import attach_cup_tracking
        if aligned:
            from cup_relative_stream_data import attach_camera_aligned_actions
            attach_camera_aligned_actions(rows, episodes, proof)
            attach_cup_tracking(rows, proof, with_timing=True, aligned=True)
        elif profile in getattr(history, 'CUP_TIMING_PROFILES', (getattr(history, 'CUP_TIMING_PROFILE', None),)):
            attach_cup_tracking(rows, proof, with_timing=True)
        else:
            attach_cup_tracking(rows, proof)
    cases = select_cases(rows, episodes, proof, trainer.transition_events(rows, episodes))
    directions = return_directions(cases, rows, episodes, trainer, cup_relative=cup_relative, aligned=aligned)
    train_fingerprint = model.info['manifest'].get('dataset_fingerprint')
    settings = {
        'checkpoint': str(checkpoint), 'dataset': str(dataset_path), 'runtime_source': str(checkpoint),
        'image_profile': model.info['image_profile'], 'observation_context_profile': profile,
        'action_representation': representation,
        'draws_per_observation': args.draws,
        'seeds': [BASE_SEED + draw for draw in range(args.draws)],
        'observations': len(cases),
        'evaluation_dataset_matches_checkpoint_training_dataset': (
            fingerprint == train_fingerprint if train_fingerprint is not None else None),
        'split_definition': 'physical-source split in the evaluation dataset',
        'targets': '50 entries padded after the real episode end; valid marks real targets',
        'return_direction': ('TCP XYZ unit vector from hold end to its real tenth target' if cup_relative
                             else 'unit vector from hold-end state to its real tenth joint target'),
        'diagnostic': 'offline recorded-target comparison, not task success or a runtime threshold',
    }
    output.mkdir(exist_ok=False)
    (output / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n',
                                          encoding='utf-8')
    summaries = []
    with (output / 'cases.jsonl').open('x', encoding='utf-8') as stream:
        for case in cases:
            index, episode, position = case['row_index'], case['episode'], case['position']
            row = rows[index]
            state = [float(value) for value in row['_cup_relative_state' if cup_relative else 'observation.state']]
            targets, valid = trainer.chunk_for_anchor(rows, episodes[episode], position, **(
                {'action_key': '_camera_aligned_actions' if aligned else '_cup_relative_action'} if cup_relative else {}))
            task = dataset.meta.tasks.iloc[int(row['task_index'])].name
            batch, _ = trainer.batch_for([index], rows, episodes, dataset, model.pre, labels=False,
                                         image_profile=settings['image_profile'],
                                         observation_context_profile=profile)
            direction, reason = (directions[(case['source_index'], case['hold_index'])]
                                 if case['kind'] in ('hold_start', 'hold_end') else (None, None))
            draws = []
            for seed in settings['seeds']:
                torch.manual_seed(seed)
                model.reset()
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    raw = (trainer.predict_action_chunk(model.policy, batch) if 'observation.cup_action_clock' in batch
                           else model.policy.predict_action_chunk(batch))
                    actions = model.post(raw.detach().clone()).cpu().float()
                    if cup_delta:
                        actions = service.actions_for_wire(actions, state, model.relative,
                                                           input_representation=representation)
                    else:
                        actions = service.actions_for_wire(actions, state, model.relative)
                torch.cuda.synchronize()
                if tuple(actions.shape) != (1, 50, 7) or not bool(torch.isfinite(actions).all()):
                    raise ValueError('Invalid 50 by 7 prediction')
                values = actions[0].tolist()
                draws.append({'seed': seed, 'seconds': time.monotonic() - started,
                              'actions': values,
                              'metrics': action_metrics(values, targets, valid, state,
                                                        case['kind'], direction, cup_relative=cup_relative)})
            detail = {**case, 'task': task, 'state': state, 'targets': targets, 'valid': valid,
                      'draws': draws}
            if profile != 'none':
                detail['observation_history'] = row[
                    '_cup_relative_history' if cup_relative else '_physical_observation_history'
                    if profile in ('state-history-physical-v1', 'state-history-physical-initial-cup-v1')
                    else '_observation_history']
            if case['kind'] in ('hold_start', 'hold_end'):
                detail.update(return_direction_unit=direction, return_direction_unavailable=reason)
            stream.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            summaries.append({**case, 'draws': [draw['metrics'] for draw in draws]})
            print(json.dumps({'status': 'case_evaluated', 'case': case['case_index'],
                              'observations': len(cases)}), flush=True)
    summary = {**settings, 'groups': group_summary(summaries, cup_relative=cup_relative), 'cases': summaries}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'output': str(output), 'observations': len(cases)}), flush=True)


if __name__ == '__main__':
    main()
