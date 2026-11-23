"""Inspect approach motion and gripper timing on recorded FR5 observations.

This offline evaluation uses the sealed checkpoint's own preprocessing and
reports complete 50-step predictions. It never connects to the robot.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def approach_cases(rows, episodes, events):
    cases = []
    for number, event in enumerate(events['close']):
        ep, close = event['episode'], event['position']
        indices = episodes[ep]
        previous = [item['position'] for item in events['open']
                    if item['episode'] == ep and item['position'] < close]
        start = max(previous) + 1 if previous else 0
        positions = {start, (start + close) // 2, close}
        positions.update(close - offset for offset in (6, 12, 24, 48)
                         if close - offset >= start)
        for pos in sorted(positions):
            row = rows[indices[pos]]
            cases.append({'case_index': len(cases), 'close_event': number,
                          'episode': ep, 'position': pos, 'row_index': indices[pos],
                          'source_index': int(row['_physical_source']),
                          'split': row['_split'], 'close_row_index': indices[close],
                          'recorded_close_offset': close - pos,
                          'open_target': float(event['old']),
                          'close_target': float(event['new'])})
    return cases


def metrics(prediction, targets, valid, case, close_state, *, cup_relative=False):
    import numpy as np
    pred, truth = np.asarray(prediction), np.asarray(targets)
    count = sum(valid)
    error = np.degrees(pred[:count, :6] - truth[:count, :6])
    grip = pred[:count, 6]
    close, opened = case['close_target'], case['open_target']
    # Label proximity describes the recorded command, not an execution rule.
    closing = np.flatnonzero(np.abs(grip - close) < np.abs(grip - opened))
    index = int(closing[0]) if len(closing) else None
    result = {'real_target_steps': count,
            'first10_joint_rmse_deg': float(np.sqrt(np.mean(error[:10] ** 2))),
            'full_chunk_joint_rmse_deg': float(np.sqrt(np.mean(error ** 2))),
            'per_joint_rmse_deg': np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            'gripper_mae_fraction': float(np.abs(grip - truth[:count, 6]).mean()),
            'predicted_close_offset': index,
            'recorded_close_within_chunk': case['recorded_close_offset'] < count,
            'close_index_difference': (index - case['recorded_close_offset']
                                       if index is not None else None),
            'predicted_close_joint_deg': (np.degrees(pred[index, :6]).tolist()
                                          if index is not None else None),
            'close_joint_minus_recorded_close_deg':
                (np.degrees(pred[index, :6] - np.asarray(close_state[:6])).tolist()
                 if index is not None else None)}
    if cup_relative:
        xyz = (pred[:count, :3] - truth[:count, :3])*1000
        rpy = error[:, 3:6]
        for k in ('first10_joint_rmse_deg', 'full_chunk_joint_rmse_deg', 'per_joint_rmse_deg',
                  'predicted_close_joint_deg', 'close_joint_minus_recorded_close_deg'):
            del result[k]
        result.update(first10_tcp_xyz_rmse_mm=float(np.sqrt(np.mean(xyz[:10]**2))),
                      full_chunk_tcp_xyz_rmse_mm=float(np.sqrt(np.mean(xyz**2))),
                      first10_tcp_rpy_rmse_deg=float(np.sqrt(np.mean(rpy[:10]**2))),
                      close_tcp_minus_recorded_close_xyz_mm=(
                          ((pred[index, :3] - np.asarray(close_state[:3]))*1000).tolist() if index is not None else None))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--draws', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261105)
    args = p.parse_args()
    if args.draws < 1:
        p.error('--draws must be positive')
    data4 = Path('/home/data4/hts/demo1/fr5-learning')
    checkpoint, dataset_path, output = (x.resolve() for x in
                                        (args.checkpoint, args.dataset, args.output))
    if data4 / 'audits' not in output.parents:
        p.error('--output must be a new audit directory on data4')
    output.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(checkpoint))
    import serve_stream_pi05 as service
    model = service.Inference(checkpoint, gpu=args.gpu)
    torch, trainer = model.torch, model.training
    torch.set_num_threads(4)
    rows, episodes, dataset, proof, fingerprint = trainer.read_dataset(dataset_path)
    if fingerprint != model.info['manifest']['dataset_fingerprint']:
        raise ValueError('Evaluation dataset differs from the trained dataset')
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
    cases = approach_cases(rows, episodes, trainer.transition_events(rows, episodes))
    settings = {'checkpoint': str(checkpoint), 'dataset': str(dataset_path),
                'checkpoint_sha256': model.info['checkpoint_sha256'],
                'image_profile': model.info['image_profile'],
                'observation_context_profile': model.info.get('observation_context_profile', 'none'),
                'action_representation': model.info['action_representation'],
                'n_action_steps': model.info['n_action_steps'],
                'draws': args.draws, 'seed': args.seed, 'cases': len(cases),
                'physical_sources': len(proof['sources']), 'robot_commands_sent': 0,
                'kind': 'offline_approach_diagnostic_not_task_success'}
    (output / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
    summaries = []
    with (output / 'cases.jsonl').open('x') as progress:
        for case in cases:
            index, ep, position = case['row_index'], case['episode'], case['position']
            action_key = '_camera_aligned_actions' if aligned else '_cup_relative_action' if cup_relative else 'action'
            state_key = '_cup_relative_state' if cup_relative else 'observation.state'
            targets, valid = trainer.chunk_for_anchor(rows, episodes[ep], position, **(
                {'action_key': action_key} if cup_relative else {}))
            state = list(rows[index][state_key])
            batch, _ = trainer.batch_for([index], rows, episodes, dataset, model.pre,
                labels=False, image_profile=settings['image_profile'],
                observation_context_profile=settings['observation_context_profile'])
            records = []
            for draw in range(args.draws):
                seed = args.seed + case['case_index'] * 100 + draw
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
                records.append({'seed': seed, 'seconds': time.monotonic() - started,
                    'metrics': metrics(values, targets, valid, case,
                                       rows[case['close_row_index']][action_key if cup_relative else state_key],
                                       cup_relative=cup_relative),
                    'actions': values})
            detail = {**case, 'state': state, 'targets': targets,
                      'valid': valid, 'draws': records}
            progress.write(json.dumps(detail, allow_nan=False) + '\n')
            progress.flush()
            summaries.append({**case, 'draws': [r['metrics'] for r in records]})
            print(json.dumps({'status': 'approach_case_evaluated',
                              'case': case['case_index'], 'of': len(cases)}), flush=True)
    summary = {**settings, 'cases': summaries,
               'limitations': ['Pose similarity and recorded-command timing are diagnostic metrics, not task success.',
                               'These are teacher-forced observations; the robot did not execute the predictions.',
                               'Cup positions and robot states remain those of the original eight sources.']}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'output': str(output),
                      'cases': len(cases), 'robot_commands_sent': 0}), flush=True)


if __name__ == '__main__':
    main()
