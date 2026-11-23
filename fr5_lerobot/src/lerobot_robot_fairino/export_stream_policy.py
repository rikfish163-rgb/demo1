"""Retarget reviewed FR5 recordings onto the fixed-rate stream policy contract.

The source arm commands are MoveL endpoints, not ServoJ commands.  Joint
labels here are interpolated *measured feedback* at the next grid tick.
Gripper labels are latched from issued MoveGripper commands.  If a recording
starts before its first such command, the caller may explicitly opt into its
declared, non-RPC initial hold; the distinction is retained for every row.

Run the CLI on the data4 server, supplying one --source per reviewed trial.
The output is new; neither the source nor an existing dataset is modified.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot_camera_gemini.timing import MONOTONIC, camera_clock, command_clock
from .feedback import gripper_feedback_valid
from .stream_contract import ACTION_DIM, ACTION_NAMES, CHUNK, CONTRACT, FPS, STATE_DIM, STATE_NAMES
from .whole_task_source import MAX_FRAME_GAP_S, load_source, read_json, read_jsonl, sha

DATA_ROOT = Path('/home/data4/hts/demo1/fr5-learning')
SCHEMA = 'fr5-stream-retarget-export-v1'
_EPS = 1e-8


def _features():
    return {
        'observation.state': {'dtype': 'float32', 'shape': (STATE_DIM,), 'names': STATE_NAMES},
        'observation.images.front': {'dtype': 'video', 'shape': (360, 640, 3),
                                     'names': ['height', 'width', 'channels']},
        'action': {'dtype': 'float32', 'shape': (ACTION_DIM,), 'names': ACTION_NAMES},
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                          encoding='utf-8')


def _splits(count, validation_indices):
    if count < 1:
        raise ValueError('At least one reviewed source is required')
    if validation_indices is None:
        if count < 2:
            raise ValueError('Default source split needs at least two sources')
        validation_indices = [count - 1]
    if (not isinstance(validation_indices, (list, tuple))
            or any(type(i) is not int or not 0 <= i < count for i in validation_indices)
            or len(set(validation_indices)) != len(validation_indices)):
        raise ValueError('Validation indices must be unique zero-based source indices')
    if len(validation_indices) == count:
        raise ValueError('At least one source must remain in the training split')
    selected = set(validation_indices)
    return ['validation' if i in selected else 'train' for i in range(count)]


def _check_sources(sources):
    for key in ('source_scope_sha256', 'review_sha256'):
        values = [s[key] for s in sources]
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate reviewed source: {key}')
    for key in ('session', 'source_stream_sha256'):
        values = [s['materialization'][key] for s in sources]
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate physical recording: {key}')


def _source_data(source):
    root = source['root']
    capture_path = root / 'continuous/capture.jsonl'
    # load_source has already checked each original JPEG and all review/control
    # evidence. These small files are checked again after loading to catch edits.
    captures = read_jsonl(capture_path)
    journal = read_json(root / 'target-command-journal.json')
    media = read_jsonl(root / 'stream-frames.jsonl')
    manifest = read_json(root / 'control-manifest.json')
    if (sha(capture_path) != source['materialization']['capture_journal_sha256']
            or sha(root / 'target-command-journal.json') != manifest['files']['target-command-journal.json']['sha256']
            or len(captures) != source['capture_frames'] or len(media) != len(captures)):
        raise ValueError('Source changed after validation')
    return captures, journal, media


def _joint_degrees(frame):
    values = np.asarray(frame['robot']['joints'], dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('Measured joint feedback must have six finite angles')
    return values


def _record_ref(frame):
    camera_t, basis = camera_clock(frame['camera'])
    if basis != MONOTONIC:
        raise ValueError('Source clock must be monotonic')
    robot_t = float(frame['robot']['received_monotonic'])
    if not math.isfinite(robot_t):
        raise ValueError('Invalid robot feedback timestamp')
    return {'frame_index': frame['frame_index'], 'camera_time_monotonic': camera_t,
            'robot_time_monotonic': robot_t}


def _wire_ref(command):
    return {'sequence': command['sequence'], 'kind': command['kind'],
            'executed': command['executed'], 'issued_monotonic': command_clock(command, MONOTONIC),
            'target_degrees_percent': list(command['target']),
            'source': 'wire_command' if command['executed'] is True else 'initial_hold_not_rpc'}


def _build_clips(source, allow_initial_hold=False):
    """Return grid rows, with a distinct clip at every source time gap."""
    captures, journal, media = _source_data(source)
    refs = [_record_ref(row) for row in captures]
    camera_times = [r['camera_time_monotonic'] for r in refs]
    robot_times = [r['robot_time_monotonic'] for r in refs]
    if any(b <= a for a, b in zip(robot_times, robot_times[1:])):
        raise ValueError('Robot feedback clocks are not strictly increasing')
    if any(b <= a for a, b in zip(camera_times, camera_times[1:])):
        raise ValueError('Camera clocks are not strictly increasing')
    for row in captures:
        _joint_degrees(row)
    command_times = [command_clock(c, MONOTONIC) for c in journal]
    if not journal or journal[0].get('kind') != 'hold' or journal[0].get('executed') is not False:
        raise ValueError('Missing explicitly identified initial hold')
    arm = [journal[0]] + [c for c in journal if c.get('executed') is True and c.get('kind') == 'arm']
    grip = [c for c in journal if c.get('executed') is True and c.get('kind') == 'gripper']
    arm_times = [command_clock(c, MONOTONIC) for c in arm]
    grip_times = [command_clock(c, MONOTONIC) for c in grip]
    spans, start = [], 0
    for i in range(1, len(captures)):
        if camera_times[i] - camera_times[i-1] > MAX_FRAME_GAP_S or robot_times[i] - robot_times[i-1] > MAX_FRAME_GAP_S:
            spans.append((start, i)); start = i
    spans.append((start, len(captures)))
    clips = []
    for begin, end in spans:
        if end - begin < 2:
            continue
        origin = max(camera_times[begin], robot_times[begin])
        rows = []
        k = 0
        while True:
            grid = origin + k / FPS
            target_t = origin + (k + 1) / FPS
            if target_t > robot_times[end - 1]:
                break
            # Both elements of the selected pair must already exist at t.
            obs = bisect.bisect_right(camera_times, grid, begin, end) - 1
            while obs >= begin and robot_times[obs] > grid:
                obs -= 1
            if obs < begin or grid - min(camera_times[obs], robot_times[obs]) > MAX_FRAME_GAP_S + _EPS:
                raise ValueError('No sufficiently recent causal camera/state pair')
            lo = bisect.bisect_right(robot_times, target_t, begin, end) - 1
            if lo < begin:
                raise ValueError('Missing feedback before future action time')
            if robot_times[lo] == target_t:
                hi, weight = lo, 0.0
            else:
                hi = lo + 1
                if hi >= end or robot_times[hi] - robot_times[lo] > MAX_FRAME_GAP_S + _EPS:
                    raise ValueError('Cannot interpolate across a recording gap')
                weight = (target_t - robot_times[lo]) / (robot_times[hi] - robot_times[lo])
                if not 0 <= weight <= 1:
                    raise ValueError('Future interpolation is not bracketed')
            jlo, jhi = _joint_degrees(captures[lo]), _joint_degrees(captures[hi])
            joint_target = np.deg2rad((1. - weight) * jlo + weight * jhi)
            gi = bisect.bisect_right(grip_times, target_t) - 1
            if gi < 0:
                if not allow_initial_hold:
                    raise ValueError('No issued gripper command before the first grid target; '
                                     'initial hold is not an RPC. Explicitly opt in to the recorded initial latch '
                                     'or provide earlier verified wire evidence.')
                gripper = journal[0]
            else:
                gripper = grip[gi]
            ai = bisect.bisect_right(arm_times, target_t) - 1
            wi = bisect.bisect_right(command_times, target_t) - 1
            if ai < 0 or wi < 0:
                raise ValueError('No initial target before future action time')
            wire = journal[wi]
            valid = gripper_feedback_valid(captures[obs]['robot'])
            measured = captures[obs]['robot']
            state = np.asarray([*np.deg2rad(_joint_degrees(captures[obs])),
                                measured['gripper_position'] / 100. if valid else 0., float(valid)], np.float32)
            action = np.asarray([*joint_target, float(gripper['target'][6]) / 100.], np.float32)
            if not np.isfinite(state).all() or not np.isfinite(action).all() or not 0 <= action[6] <= 1:
                raise ValueError('Non-finite state/action or invalid gripper target')
            mapping = {
                'grid_time_monotonic': grid, 'target_time_monotonic': target_t,
                'observation': {**refs[obs], 'rgb_file': captures[obs]['rgb_file'],
                                'rgb_jpeg_sha256': media[obs]['rgb_jpeg_sha256'], 'gripper_valid': valid},
                'joint_retarget': {'method': 'linear_interpolation_of_measured_feedback_not_wire_servoj',
                                   'lower': refs[lo], 'upper': refs[hi], 'upper_weight': weight,
                                   'lower_joint_degrees': jlo.tolist(), 'upper_joint_degrees': jhi.tolist()},
                'gripper_target': _wire_ref(gripper),
                'active_arm_wire_target': _wire_ref(arm[ai]),
                'active_wire_target': _wire_ref(wire),
                'derived_action_radians_fraction': action.tolist(),
            }
            rows.append({'state': state, 'action': action, 'mapping': mapping,
                         'rgb_path': source['root'] / 'continuous' / captures[obs]['rgb_file']})
            k += 1
        if rows:
            clips.append(rows)
    if not clips:
        raise ValueError('Source contains no complete fixed-rate observation/action pair')
    coverage = {'source_frames': len(captures), 'source_spans': len(spans), 'exported_clips': len(clips),
                'source_elapsed_seconds': camera_times[-1] - camera_times[0],
                'source_continuous_seconds': sum(camera_times[b-1] - camera_times[a] for a, b in spans),
                'exported_action_seconds': sum(len(clip) / FPS for clip in clips),
                'exported_frames': sum(map(len, clips)),
                'invalid_gripper_observations': sum(not row['mapping']['observation']['gripper_valid']
                                                    for clip in clips for row in clip),
                'initial_hold_action_frames': sum(row['mapping']['gripper_target']['executed'] is False
                                                  for clip in clips for row in clip),
                'gap_threshold_seconds': MAX_FRAME_GAP_S}
    return clips, coverage


def _episode_stats(episodes, indices):
    from lerobot.datasets.compute_stats import aggregate_stats
    selected = []
    for index in indices:
        stats = {}
        for key, value in episodes[index].items():
            if key.startswith('stats/') and value is not None:
                _, feature, name = key.split('/', 2)
                stats.setdefault(feature, {})[name] = np.asarray(value)
        if not stats:
            raise ValueError('LeRobot episode statistics are missing')
        selected.append(stats)
    if not selected:
        raise ValueError('No training episodes for statistics')
    result = aggregate_stats(selected)
    return {feature: {key: np.asarray(value).tolist() for key, value in values.items()}
            for feature, values in result.items()}


def _exact_stats(rows):
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Training values are empty or non-finite')
    return {'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
            'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist(), 'count': [len(values)],
            **{f'q{int(q * 100):02d}': np.quantile(values, q, axis=0).tolist()
               for q in (.01, .10, .50, .90, .99)}}


def _stats(root, train_episode_indices, train_states, train_actions):
    import pyarrow.parquet as pq
    files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    episodes = pq.read_table(files).to_pylist()
    result = _episode_stats(episodes, train_episode_indices)
    result['observation.state'] = _exact_stats(train_states)
    result['action'] = _exact_stats(train_actions)
    return result


def _source_record(source, split, coverage, index):
    return {'source_index': index, 'trial': str(source['root']), 'split': split,
            'source_scope_sha256': source['source_scope_sha256'],
            'source_summary_sha256': source['source_summary_sha256'],
            'source_stream_sha256': source['materialization']['source_stream_sha256'],
            'capture_journal_sha256': source['materialization']['capture_journal_sha256'],
            'session': source['materialization']['session'],
            'user_review_sha256': source['review_sha256'],
            'user_review_file': f'meta/user_reviews/trial-{index:04d}.json',
            'task_instruction': source['task_instruction'], 'task_spec': source['task_spec'],
            'hold_records': source['holds'], 'data_provenance': source['data_provenance'],
            'coverage': coverage}


def export(sources, output, validation_indices=None, *, allow_initial_hold=False):
    """Write a fresh data4 LeRobot dataset; return the integrity report.

    validation_indices selects whole input sources, not individual frames.
    ``allow_initial_hold`` is an explicit exception for an unissued, declared
    initial latch, never a claim that it was a MoveGripper RPC.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    output = Path(output).resolve()
    if not output.is_relative_to(DATA_ROOT.resolve()) or output == DATA_ROOT.resolve():
        raise ValueError('Output must be a new directory inside data4/fr5-learning')
    if output.exists():
        raise FileExistsError('Output already exists')
    paths = [Path(p).resolve() for p in sources]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in paths):
        raise ValueError('Output and source directories must not overlap')
    splits = _splits(len(paths), validation_indices)
    loaded = [load_source(path) for path in paths]
    _check_sources(loaded)
    planned = [_build_clips(source, allow_initial_hold) for source in loaded]
    create = {'repo_id': 'local/fr5_pi05_stream', 'fps': FPS, 'root': output,
              'features': _features(), 'robot_type': 'fairino_fr5', 'use_videos': True,
              'image_writer_threads': 2}
    params = inspect.signature(LeRobotDataset.create).parameters
    if 'vcodec' in params:
        create['vcodec'] = 'h264'
    elif 'rgb_encoder' in params:
        from lerobot.configs.video import RGBEncoderConfig
        create['rgb_encoder'] = RGBEncoderConfig(vcodec='h264')
    else:
        raise RuntimeError('Unsupported LeRobot video encoder API')
    ds = LeRobotDataset.create(**create)
    mapping, train_states, train_actions, source_records = [], [], [], []
    episodes, train_episodes, validation_episodes = [], [], []
    try:
        for i, (source, split, (clips, coverage)) in enumerate(zip(loaded, splits, planned)):
            source_records.append(_source_record(source, split, coverage, i))
            for clip_index, clip in enumerate(clips):
                episode_index = len(episodes)
                (train_episodes if split == 'train' else validation_episodes).append(episode_index)
                episodes.append({'episode_index': episode_index, 'source_index': i, 'source_clip': clip_index,
                                 'split': split, 'length': len(clip),
                                 'grid_start_monotonic': clip[0]['mapping']['grid_time_monotonic']})
                for step, row in enumerate(clip):
                    with Image.open(row['rgb_path']) as image:
                        rgb = np.asarray(image.convert('RGB'))
                    if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
                        raise ValueError('Source image must be actual recorded 640x360 RGB')
                    ds.add_frame({'observation.state': row['state'], 'action': row['action'],
                                  'observation.images.front': rgb, 'task': source['task_instruction']})
                    if split == 'train':
                        train_states.append(row['state']); train_actions.append(row['action'])
                    mapping.append({'frame': len(mapping), 'episode_index': episode_index, 'step': step,
                                    'source_index': i, 'split': split, **row['mapping']})
                ds.save_episode()
    finally:
        try:
            ds.finalize()
        finally:
            stop = getattr(ds, 'stop_image_writer', None)
            if callable(stop):
                stop()
    meta = output / 'meta'
    (meta / 'user_reviews').mkdir(exist_ok=False)
    for source, record in zip(loaded, source_records):
        raw = (source['root'] / 'user-review.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['user_review_sha256']:
            raise ValueError('User review changed while exporting')
        (output / record['user_review_file']).write_bytes(raw)
    stats = _stats(output, train_episodes, train_states, train_actions)
    _write_json(meta / 'stats.json', stats)
    _write_json(meta / 'stream_stats.json', {'schema': 'fr5-stream-training-stats-v1',
                'source_indices': [i for i, split in enumerate(splits) if split == 'train'],
                'episode_indices': train_episodes, 'frame_count': len(train_states),
                'note': 'Global normalization uses train only; LeRobot retains per-episode metadata.',
                'stats': stats})
    proof = {'schema': SCHEMA, 'policy_contract': CONTRACT, 'fps': FPS, 'chunk_size': CHUNK,
             'source_count': len(loaded), 'frames': len(mapping), 'sources': source_records,
             'episodes': episodes, 'train_episode_indices': train_episodes,
             'validation_episode_indices': validation_episodes,
             'training_data_kind': 'synthetic_diagnostic' if any(
                 s['data_provenance']['training_data_kind'] != 'captured' for s in loaded) else 'captured',
             'allow_initial_hold': allow_initial_hold,
             'observation_semantics': 'latest camera and paired measured state both no later than grid time',
             'joint_action_semantics': 'linear interpolation of real feedback at grid time plus 1/fps; '
                                       'derived targets, not recorded ServoJ commands',
             'gripper_action_semantics': 'last actually issued gripper command before target time, '
                                         'or explicit non-RPC initial hold if opted in',
             'wire_semantics': 'separately retained journal targets in degrees and percent, not servo actions',
             'gap_semantics': 'independent episodes at source camera or robot feedback gaps',
             'statistics_semantics': 'global stats and quantiles computed from training sources only',
             'task_semantics': 'one full user-reviewed instruction per source; actual hold times preserved',
             'policy_trained': False, 'physical_success_verified': False}
    _write_json(meta / 'stream_provenance.json', proof)
    with (meta / 'stream_source_frames.jsonl').open('w', encoding='utf-8') as stream:
        for row in mapping:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    result = verify(output, _loaded=loaded, _planned=planned)
    _write_json(meta / 'stream_verification.json', result)
    return result


def _scalar(value):
    return np.asarray(value).reshape(-1)[0].item()


def verify(output, *, _loaded=None, _planned=None):
    """Reconstruct every label from reviewed sources and read the encoded video."""
    import av
    import pyarrow.parquet as pq
    root = Path(output).resolve()
    proof = read_json(root / 'meta/stream_provenance.json')
    info = read_json(root / 'meta/info.json')
    if (proof.get('schema') != SCHEMA or proof.get('policy_contract') != CONTRACT
            or info.get('fps') != FPS or proof.get('fps') != FPS or proof.get('chunk_size') != CHUNK):
        raise ValueError('Dataset contract or timebase mismatch')
    for name, dimension, names in (('observation.state', STATE_DIM, STATE_NAMES), ('action', ACTION_DIM, ACTION_NAMES)):
        f = info['features'].get(name, {})
        if f.get('dtype') != 'float32' or f.get('shape') != [dimension] or f.get('names') != names:
            raise ValueError('Dataset feature mismatch: ' + name)
    f = info['features'].get('observation.images.front', {})
    if f.get('dtype') != 'video' or f.get('shape') != [360, 640, 3]:
        raise ValueError('Expected recorded 640x360 RGB video')
    records = proof['sources']
    loaded = _loaded if _loaded is not None else [load_source(item['trial']) for item in records]
    _check_sources(loaded)
    planned = _planned if _planned is not None else [_build_clips(s, proof['allow_initial_hold']) for s in loaded]
    if len(records) != len(loaded) or len(planned) != len(loaded) or proof['source_count'] != len(loaded):
        raise ValueError('Source count mismatch')
    for i, (source, record, (_, coverage)) in enumerate(zip(loaded, records, planned)):
        if record != _source_record(source, record['split'], coverage, i):
            raise ValueError('Source identity, review, task or coverage differs')
        copied = (root / record['user_review_file']).resolve()
        if root not in copied.parents or sha(copied) != source['review_sha256']:
            raise ValueError('Copied user review hash mismatch')
    data_files = sorted((root / 'data').rglob('*.parquet'))
    video_files = sorted((root / 'videos').rglob('*.mp4'))
    episode_files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    if not data_files or not video_files or not episode_files:
        raise ValueError('Missing LeRobot data, episode metadata or video')
    rows = pq.read_table(data_files).to_pylist()
    ep_rows = pq.read_table(episode_files).to_pylist()
    maps = read_jsonl(root / 'meta/stream_source_frames.jsonl')
    task_table = pq.read_table(root / 'meta/tasks.parquet').to_pandas()
    tasks = {int(row['task_index']): str(index) for index, row in task_table.iterrows()}
    expected_images, train_states, train_actions, episode_proof = [], [], [], []
    train_episodes, val_episodes, cursor = [], [], 0
    for i, (source, record, (clips, _)) in enumerate(zip(loaded, records, planned)):
        split = record['split']
        if split not in ('train', 'validation'):
            raise ValueError('Invalid source split')
        for ci, clip in enumerate(clips):
            ep = len(episode_proof)
            (train_episodes if split == 'train' else val_episodes).append(ep)
            episode_proof.append({'episode_index': ep, 'source_index': i, 'source_clip': ci,
                                  'split': split, 'length': len(clip),
                                  'grid_start_monotonic': clip[0]['mapping']['grid_time_monotonic']})
            if ep >= len(ep_rows):
                raise ValueError('Missing episode metadata')
            ep_row = ep_rows[ep]
            if any(_scalar(ep_row[k]) != v for k, v in (('episode_index', ep), ('length', len(clip)),
                    ('dataset_from_index', cursor), ('dataset_to_index', cursor + len(clip)))):
                raise ValueError('Episode boundary differs from source clip')
            for step, expected in enumerate(clip):
                if cursor >= len(rows) or cursor >= len(maps):
                    raise ValueError('Missing frame')
                row = rows[cursor]
                mapping = {'frame': cursor, 'episode_index': ep, 'step': step,
                           'source_index': i, 'split': split, **expected['mapping']}
                if (maps[cursor] != mapping or _scalar(row['index']) != cursor
                        or _scalar(row['episode_index']) != ep or _scalar(row['frame_index']) != step
                        or abs(float(row['timestamp']) - step / FPS) > .001
                        or tasks.get(int(_scalar(row['task_index']))) != source['task_instruction']):
                    raise ValueError('Frame/task/provenance or fixed-rate timestamp mismatch')
                if (not np.array_equal(np.asarray(row['observation.state'], np.float32), expected['state'])
                        or not np.array_equal(np.asarray(row['action'], np.float32), expected['action'])):
                    raise ValueError('State or derived action does not match source feedback and commands')
                expected_images.append(expected['rgb_path'])
                if split == 'train':
                    train_states.append(expected['state']); train_actions.append(expected['action'])
                cursor += 1
    if (cursor != len(rows) or cursor != len(maps) or cursor != info['total_frames'] or cursor != proof['frames']
            or len(episode_proof) != len(ep_rows) or len(episode_proof) != info['total_episodes']
            or episode_proof != proof['episodes'] or train_episodes != proof['train_episode_indices']
            or val_episodes != proof['validation_episode_indices']):
        raise ValueError('Source/episode/frame split counts mismatch')
    stats = _stats(root, train_episodes, train_states, train_actions)
    if read_json(root / 'meta/stats.json') != stats:
        raise ValueError('LeRobot statistics are not the recomputed training-only statistics')
    stream_stats = read_json(root / 'meta/stream_stats.json')
    if (stream_stats.get('stats') != stats or stream_stats.get('episode_indices') != train_episodes
            or stream_stats.get('frame_count') != len(train_states)
            or stream_stats.get('source_indices') != [i for i, s in enumerate(records) if s['split'] == 'train']):
        raise ValueError('Stream statistics or source split mismatch')
    decoded, max_error = 0, 0.
    for path in video_files:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if decoded >= len(expected_images):
                    raise ValueError('Video has extra frames')
                rgb = frame.to_ndarray(format='rgb24')
                with Image.open(expected_images[decoded]) as img:
                    source_rgb = np.asarray(img.convert('RGB'))
                if rgb.shape != (360, 640, 3) or source_rgb.shape != rgb.shape:
                    raise ValueError('Video frame shape mismatch')
                error = float(np.abs(rgb.astype(np.int16) - source_rgb.astype(np.int16)).mean())
                if error > 25.:
                    raise ValueError('Decoded video does not match causal source image')
                max_error = max(max_error, error); decoded += 1
    if decoded != len(expected_images):
        raise ValueError('Decoded video frame count mismatch')
    return {'status': 'passed', 'source_count': len(loaded), 'frames': cursor, 'episodes': len(episode_proof),
            'train_episode_indices': train_episodes, 'validation_episode_indices': val_episodes,
            'train_frames': len(train_states), 'decoded_frames': decoded, 'max_mean_rgb_error': max_error,
            'source_coverage': [record['coverage'] for record in records],
            'training_data_kind': proof['training_data_kind'], 'allow_initial_hold': proof['allow_initial_hold'],
            'meaning': 'Source/derived-label/video integrity only; no training or robot success is implied'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-source-index', type=int, action='append', default=None)
    parser.add_argument('--allow-recorded-initial-hold', action='store_true',
                        help='explicitly permit the declared non-RPC initial gripper latch')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(DATA_ROOT.resolve()):
        parser.error('Output must be inside data4/fr5-learning')
    result = verify(args.output) if args.verify_only else export(
        args.source, args.output, args.validation_source_index, allow_initial_hold=args.allow_recorded_initial_hold)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
