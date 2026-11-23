"""Attach cup-relative TCP observations and targets to the eight FR5 demonstrations.

``attach_cup_relative`` consumes the rows, episodes and proof returned by the
sealed stream trainer's ``read_dataset``. Each row receives:

* ``_cup_relative_state``: measured [dx, dy, dz] in metres from the initial
  cup base, measured TCP [rx, ry, rz] in radians, measured gripper and validity.
* ``_cup_relative_action``: [dx, dy, dz] in metres and TCP RPY in radians,
  interpolated from the recorded feedback references at the action target time,
  followed by the original gripper target.
* ``_cup_relative_history``: six entries at 0.5, 1, 2, 4, 6 and 8 seconds,
  each ``None`` or ``{'age_s': actual_age, 'state': state8}``. The TCP and
  gripper are the recorded historical observation, never an action target.

Cup origins are estimated initial cup-base positions in J1 base coordinates.
Only translation is cup-relative: RPY remains the measured base-frame TCP
orientation. Returned source records and coverage contain JSON-native values
and can be incorporated into a training report or checkpoint.
"""

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path


HISTORY_OFFSETS_S = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
SOURCE_EPISODES = (
    'cup-system-20261031-06',
    'cup-ui-20261031-183700-cc83dc',
    'cup-ui-20261031-190231-0c1c42',
    'cup-ui-20261031-191451-8c9277',
    'cup-ui-20261031-192509-307a80',
    'cup-ui-20261101-165538-d95608',
    'cup-chat-20261101-171627-cbdf83',
    'cup-ui-20261101-172307-9c19b2',
)
_CAMERA_SERIAL = 'CP1E54200015'
_CALIBRATION_DIRECTION = 'p_J1_base_mm = R @ p_color_camera_mm + t_mm'
_TIME_TOLERANCE = 1e-5


def _finite_vector(values, size, label):
    if (not isinstance(values, (list, tuple)) or len(values) != size
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values)):
        raise ValueError(f'Expected {size} finite values for {label}')
    return [float(value) for value in values]


def _recorded_file(root, name, manifest):
    raw = (root / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files'][name]['sha256']:
        raise ValueError(f'{name} differs from its existing control receipt')
    return raw


def _source_observation(source, source_index, episode_index):
    root = Path(source['trial'])
    if root.name != 'materialized' or root.parent.name != SOURCE_EPISODES[source_index]:
        raise ValueError(f'Source {source_index} is not the expected original demonstration')
    control = json.loads((root / 'control-manifest.json').read_bytes())
    if control['scope_sha256'] != source['source_scope_sha256']:
        raise ValueError(f'Source {source_index} scope differs from the exported provenance')
    geometry = json.loads(_recorded_file(root, 'geometry.json', control))
    observation_raw = _recorded_file(root, 'source-observation.json', control)
    if hashlib.sha256(observation_raw).hexdigest() != geometry['source_observation_sha256']:
        raise ValueError(f'Source {source_index} cup and initial observation differ')
    observation = json.loads(observation_raw)
    camera = observation['camera']
    calibration = observation['calibration']
    observed_time = camera['capture']['received_monotonic']
    if (camera['camera']['serial'] != _CAMERA_SERIAL or calibration['robot'] != 'J1'
            or calibration['direction'] != _CALIBRATION_DIRECTION
            or not isinstance(observed_time, (int, float)) or not math.isfinite(observed_time)):
        raise ValueError(f'Source {source_index} initial cup observation has the wrong frame or clock')
    cup_model = geometry['cup_model']
    base = _finite_vector(cup_model['base_center_mm'], 3, 'initial cup base in J1 millimetres')
    uv = _finite_vector(geometry['cup_uv_selected_by_llm'], 2, 'initial cup rim pixel')
    return {
        'source_index': source_index,
        'episode_index': int(episode_index),
        'split': source['split'],
        'trial': str(root),
        'cup_base_xyz_mm': base,
        'initial_cup_uv': uv,
        'source_observation_monotonic': float(observed_time),
        'source_observation': geometry['source_observation'],
        'camera_serial': _CAMERA_SERIAL,
        'cup_base_method': cup_model['method'],
        'source_scope_sha256': source['source_scope_sha256'],
        'capture_journal_sha256': source['capture_journal_sha256'],
        'geometry_sha256': control['files']['geometry.json']['sha256'],
        'source_observation_sha256': control['files']['source-observation.json']['sha256'],
    }


def _captures(source, source_index):
    raw = (Path(source['trial']) / 'continuous/capture.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['capture_journal_sha256']:
        raise ValueError(f'Source {source_index} capture journal differs from the exported receipt')
    captures = [json.loads(line) for line in raw.splitlines() if line.strip()]
    frames = {}
    previous_time = -math.inf
    for capture in captures:
        frame = capture['frame_index']
        camera_time = capture['camera']['source_timestamp_monotonic']
        if (not isinstance(frame, int) or frame in frames
                or not isinstance(camera_time, (int, float)) or not math.isfinite(camera_time)
                or camera_time <= previous_time):
            raise ValueError(f'Source {source_index} capture frames or camera clock are invalid')
        frames[frame] = capture
        previous_time = camera_time
    return captures, frames


def _pose(capture):
    robot = capture['robot']
    if robot['tool'] != 0 or robot['user'] != 0:
        raise ValueError('Cup-relative TCP requires the recorded tool 0 and user 0 frame')
    return _finite_vector(robot['tcp'], 6, 'recorded TCP pose in millimetres and degrees')


def _reference(frames, reference):
    capture = frames[reference['frame_index']]
    if (abs(float(capture['camera']['source_timestamp_monotonic'])
            - float(reference['camera_time_monotonic'])) > _TIME_TOLERANCE
            or abs(float(capture['robot']['received_monotonic'])
                   - float(reference['robot_time_monotonic'])) > _TIME_TOLERANCE):
        raise ValueError('Capture frame and provenance timestamps differ')
    return capture


def _relative_pose(pose, cup_base):
    return ([(pose[index] - cup_base[index]) / 1000. for index in range(3)]
            + [math.radians(angle) for angle in pose[3:]])


def _measured_state(capture, state, cup_base):
    state = _finite_vector(state, 8, 'recorded joint/gripper state')
    if state[7] not in (0., 1.) or not 0. <= state[6] <= 1.:
        raise ValueError('Invalid recorded gripper state or validity')
    robot = capture['robot']
    joints = _finite_vector(robot['joints'], 6, 'recorded joint feedback')
    if any(abs(math.radians(joint) - state[index]) > 1e-5 for index, joint in enumerate(joints)):
        raise ValueError('Recorded joint state and selected capture frame differ')
    if state[7] == 0.:
        if state[6] != 0.:
            raise ValueError('Invalid gripper feedback must retain its zero placeholder')
    elif abs(float(robot['gripper_position']) / 100. - state[6]) > 1e-5:
        raise ValueError('Recorded gripper state and selected capture frame differ')
    return _relative_pose(_pose(capture), cup_base) + state[6:]


def _action(row, mapping, frames, cup_base):
    retarget = mapping['joint_retarget']
    lower = _reference(frames, retarget['lower'])
    upper = _reference(frames, retarget['upper'])
    weight = retarget['upper_weight']
    target_time = mapping['target_time_monotonic']
    lo_time = lower['robot']['received_monotonic']
    hi_time = upper['robot']['received_monotonic']
    if (not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0. <= weight <= 1.
            or not isinstance(target_time, (int, float)) or not math.isfinite(target_time)
            or lo_time > hi_time or target_time < lo_time - _TIME_TOLERANCE
            or target_time > hi_time + _TIME_TOLERANCE
            or abs((lo_time + weight * (hi_time - lo_time)) - target_time) > _TIME_TOLERANCE):
        raise ValueError('TCP action references do not bracket the mapped target time')
    for capture, key in ((lower, 'lower_joint_degrees'), (upper, 'upper_joint_degrees')):
        recorded = _finite_vector(capture['robot']['joints'], 6, 'recorded joint feedback')
        mapped = _finite_vector(retarget[key], 6, 'mapped joint feedback')
        if any(abs(a - b) > 1e-5 for a, b in zip(recorded, mapped)):
            raise ValueError('Joint-retarget references differ from the source feedback')
    action = _finite_vector(row['action'], 7, 'original stream action')
    mapped_action = _finite_vector(mapping['derived_action_radians_fraction'], 7, 'mapped stream action')
    if any(abs(a - b) > 1e-6 for a, b in zip(action, mapped_action)):
        raise ValueError('Original action and source mapping differ')
    lo_pose, hi_pose = _pose(lower), _pose(upper)
    pose = [(1. - weight) * lo + weight * hi for lo, hi in zip(lo_pose, hi_pose)]
    return _relative_pose(pose, cup_base) + [action[6]]


def attach_cup_relative(rows, episodes, proof, dataset_path):
    """Attach source-derived cup-relative state, action and physical history.

    The input must be an eight-source measured-trajectory dataset returned by
    ``read_dataset``; its physical history must already be attached. This reads
    existing source journals, observation receipts and compact frame mappings,
    performs no device I/O and does not write any file. Rows are updated only
    after all eight sources and all referenced observations have been checked.

    Return ``{'cup_sources': [...], 'coverage': {...}}``. ``cup_sources`` is in
    source-index order and records each cup origin, episode, split and initial
    observation time; ``coverage`` records rows and valid/missing history slots.
    """
    sources = proof['sources']
    if (proof.get('schema') not in ('fr5-stream-retarget-export-v1', 'fr5-stream-retarget-compact-v2')
            or proof.get('training_data_kind') != 'captured'
            or proof.get('source_count') != len(SOURCE_EPISODES)
            or len(sources) != len(SOURCE_EPISODES) or len(episodes) != len(SOURCE_EPISODES)):
        raise ValueError('Expected the original eight captured demonstrations')
    if any(source.get('source_index') != index or source.get('split') not in ('train', 'validation')
           for index, source in enumerate(sources)):
        raise ValueError('Physical source order or split differs from the eight-source contract')
    with (Path(dataset_path) / 'meta/stream_source_frames.jsonl').open(encoding='utf-8') as stream:
        mappings = [json.loads(line) for line in stream if line.strip()]
    if len(mappings) != len(rows) or proof.get('frames') != len(rows):
        raise ValueError('Source mappings and loaded rows have different lengths')
    all_indices = [index for indices in episodes.values() for index in indices]
    if sorted(all_indices) != list(range(len(rows))):
        raise ValueError('Episodes must cover each row exactly once')

    updates = [None] * len(rows)
    observations = []
    per_source = []
    seen_sources = set()
    for episode, indices in episodes.items():
        if not indices:
            raise ValueError('An original demonstration episode is empty')
        source_index = int(rows[indices[0]]['_physical_source'])
        if source_index not in range(len(sources)) or source_index in seen_sources:
            raise ValueError('Each original physical source must have exactly one episode')
        seen_sources.add(source_index)
        source = sources[source_index]
        cup = _source_observation(source, source_index, episode)
        first_time = float(rows[indices[0]]['_camera_time'])
        if cup['source_observation_monotonic'] > first_time + _TIME_TOLERANCE:
            raise ValueError(f'Source {source_index} cup observation follows the episode start')
        captures, frames = _captures(source, source_index)
        history_samples = [capture for capture in captures
                           if capture['camera']['source_timestamp_monotonic'] >= first_time]
        history_times = [capture['camera']['source_timestamp_monotonic'] for capture in history_samples]
        observed_frames = {rows[index]['_source_capture_frame'] for index in indices}
        present = missing = unretained = 0
        for index in indices:
            row, mapping = rows[index], mappings[index]
            if (int(row['_physical_source']) != source_index or row['_split'] != source['split']
                    or int(row['episode_index']) != int(episode)
                    or mapping['frame'] != index or mapping['episode_index'] != int(episode)
                    or mapping['source_index'] != source_index or mapping['split'] != source['split']
                    or mapping['step'] != int(row['frame_index'])
                    or mapping['observation']['frame_index'] != row['_source_capture_frame']):
                raise ValueError('Row, episode and original source do not agree')
            current = _reference(frames, mapping['observation'])
            current_time = float(row['_camera_time'])
            if abs(current['camera']['source_timestamp_monotonic'] - current_time) > _TIME_TOLERANCE:
                raise ValueError('Current camera time differs from its source frame')
            state = _measured_state(current, row['observation.state'], cup['cup_base_xyz_mm'])
            if bool(state[7]) != mapping['observation']['gripper_valid']:
                raise ValueError('Current gripper validity differs from source provenance')
            action = _action(row, mapping, frames, cup['cup_base_xyz_mm'])
            recorded_history = row['_physical_observation_history']
            if not isinstance(recorded_history, list) or len(recorded_history) != len(HISTORY_OFFSETS_S):
                raise ValueError('Physical observation history must be attached before cup-relative history')
            history = []
            for offset, previous in zip(HISTORY_OFFSETS_S, recorded_history):
                position = bisect_right(history_times, current_time - offset) - 1
                if position < 0:
                    if previous is not None:
                        raise ValueError('Physical history contains an unavailable source observation')
                    history.append(None)
                    missing += 1
                    continue
                capture = history_samples[position]
                stamp = capture['camera']['source_timestamp_monotonic']
                age = current_time - stamp
                if (not isinstance(previous, dict) or abs(previous['age_s'] - age) > _TIME_TOLERANCE
                        or capture['robot']['received_monotonic'] > current_time + _TIME_TOLERANCE):
                    raise ValueError('Physical history differs from causal source feedback')
                history.append({'age_s': age,
                                'state': _measured_state(capture, previous['state'], cup['cup_base_xyz_mm'])})
                present += 1
                unretained += int(capture['frame_index'] not in observed_frames)
            updates[index] = (state, action, history)
        observations.append(cup)
        per_source.append({'source_index': source_index, 'episode_index': int(episode),
                           'split': source['split'], 'rows': len(indices),
                           'history_entries': present, 'missing_history_entries': missing,
                           'history_entries_not_observation_frames': unretained})
    if seen_sources != set(range(len(SOURCE_EPISODES))):
        raise ValueError('The original eight physical sources are not all represented')
    for row, (state, action, history) in zip(rows, updates):
        row['_cup_relative_state'] = state
        row['_cup_relative_action'] = action
        row['_cup_relative_history'] = history
    per_source.sort(key=lambda item: item['source_index'])
    observations.sort(key=lambda item: item['source_index'])
    coverage = {
        'source_count': len(observations), 'episode_count': len(episodes), 'rows': len(rows),
        'state_rows': len(rows), 'action_rows': len(rows),
        'train_rows': sum(item['rows'] for item in per_source if item['split'] == 'train'),
        'validation_rows': sum(item['rows'] for item in per_source if item['split'] == 'validation'),
        'history_entries': sum(item['history_entries'] for item in per_source),
        'missing_history_entries': sum(item['missing_history_entries'] for item in per_source),
        'history_entries_not_observation_frames': sum(item['history_entries_not_observation_frames']
                                                       for item in per_source),
        'history_offsets_s': list(HISTORY_OFFSETS_S),
        'history_clock': 'host_monotonic_camera_time',
        'translation': 'J1 base axes; metres from estimated initial cup base',
        'orientation': 'recorded TCP RPY in radians; action RPY interpolated componentwise in degrees',
        'action_source': 'recorded TCP feedback at joint-retarget lower/upper frames and weight',
        'per_source': per_source,
    }
    return {'cup_sources': observations, 'coverage': coverage}


def pickup_wait_mask(targets, request):
    """Mask a leading operator wait after the observed pickup is ready."""
    from lerobot_robot_fairino.cup_action_selection import close_at_pickup

    ready = close_at_pickup(request)
    before_close, mask = True, []
    for target in targets:
        if target[6] < .88:
            before_close = False
        mask.append(bool(ready and before_close and target[6] >= .98))
    return mask


def attach_cup_tracking(rows, proof, *, with_timing=False, aligned=False, setdown=False, execution_steps=10,
                       mask_settled_wait_in_prefix=False, mask_pickup_wait=False):
    """Replay all causal RGB-D frames, then join by the sealed observation mapping.

    Run only after ``attach_cup_relative`` validated the eight source journals
    and the row bindings. Repeated compact rows share the original observation.
    """
    import cv2
    import numpy as np
    from fr5_jev.cup_tracking import CupTracker
    from fr5_jev.cup_timing import CupTiming
    from collections import Counter
    from bisect import bisect_right

    grouped = {index: {} for index in range(len(SOURCE_EPISODES))}
    for row in rows:
        grouped[int(row['_physical_source'])].setdefault(row['_source_capture_frame'], []).append(row)
    coverage = []
    for index, source in enumerate(proof['sources']):
        root = Path(source['trial'])
        geometry = json.loads((root / 'geometry.json').read_bytes())
        opening = json.loads((root / 'source-observation.json').read_bytes())
        camera = opening['camera']
        initial_cup = {'camera_serial': camera['camera']['serial'],
                       'observed_monotonic': camera['capture']['received_monotonic'],
                       'table_normal': geometry['table']['normal'],
                       'table_offset_mm': geometry['table']['offset_mm'],
                       'rim_xyz_mm': geometry['cup_model']['rim_center_mm'],
                       'base_xyz_mm': geometry['cup_model']['base_center_mm']}
        tracker = CupTracker(initial_cup, opening['calibration']['base_from_color_camera_mm'])
        timer = CupTiming(initial_cup, aligned=aligned, setdown=setdown) if with_timing else None
        timing_counts = Counter()
        selected = grouped[index]
        frames, visible, selected_visible, selected_rows, visible_rows = 0, 0, 0, 0, 0
        start, end = min(selected), max(selected)
        setdown_stamps, setdown_ages = [], []
        with (root / 'continuous/capture.jsonl').open(encoding='utf-8') as journal:
            for line in journal:
                capture = json.loads(line)
                frame = capture['frame_index']
                if frame < start:
                    continue
                if frame > end and not setdown:
                    break
                image = cv2.imread(str(root / 'continuous' / capture['rgb_file']), cv2.IMREAD_COLOR)
                with np.load(root / 'continuous' / capture['depth_file'], allow_pickle=False) as depth:
                    track = tracker.update(image, depth['depth_mm'], capture['camera'], capture['robot'],
                                           gripper_valid=capture['gripper_feedback_valid'])
                observed = track['rim_minus_jaw_mm'] is not None
                timing = timer.update(capture['camera'], capture['robot'], track)['state'] if timer else None
                if setdown:
                    setdown_stamps.append(capture['camera']['source_timestamp_monotonic'])
                    setdown_ages.append(timing['setdown_age_s'])
                frames += 1
                visible += observed
                if frame in selected:
                    selected_visible += observed
                    selected_rows += len(selected[frame])
                    visible_rows += observed * len(selected[frame])
                    if timing is not None:
                        timing_counts[timing['phase']] += len(selected[frame])
                    for row in selected[frame]:
                        row['_cup_tracking'] = {k: track[k] for k in ('rim_minus_jaw_mm', 'native_contact')}
                        if timing is not None:
                            row['_cup_timing'] = timing
        if setdown:
            from fr5_jev.cup_timing import SETDOWN_MIN_AGE_S
            suppressed = 0
            for group in selected.values():
                for row in group:
                    anchor = row['_cup_timing']
                    holding_return = (row['_cup_tracking']['native_contact'] == 1 and
                                      anchor['phase'] in ('returning', 'at_table_height'))
                    mask, before = [], []
                    for tick, target in enumerate(row['_camera_aligned_actions'], 1):
                        stamp = row['_camera_time'] + tick / 6.
                        j = bisect_right(setdown_stamps, stamp) - 1
                        # Exact commands remain in the dataset. Only the goal's
                        # gripper loss ignores human waiting after stable return.
                        stable = (j >= 0 and stamp - setdown_stamps[j] <= .25
                                  and setdown_ages[j] is not None and setdown_ages[j] >= SETDOWN_MIN_AGE_S)
                        ignore_wait = tick > execution_steps or (mask_settled_wait_in_prefix
                                                                  and anchor['setdown_age_s'] is not None
                                                                  and anchor['setdown_age_s'] >= SETDOWN_MIN_AGE_S)
                        mask.append(bool(ignore_wait and target[6] < .88 and stable))
                        # Restrict the extra loss to demonstrated held points;
                        # a lost/invalid observation cannot erase an opening.
                        before.append(bool(holding_return and target[6] < .88 and not stable))
                    row['_settled_wait_mask'] = mask
                    row['_before_setdown_mask'] = before
                    suppressed += sum(mask)
        pickup_suppressed = pickup_ready_rows = 0
        if mask_pickup_wait:
            for group in selected.values():
                for row in group:
                    row['_pickup_wait_mask'] = pickup_wait_mask(row['_camera_aligned_actions'], {
                        'cup_relative_state': row['_cup_relative_state'],
                        'observation_history': row['_cup_relative_history'],
                        'cup_tracking': row['_cup_tracking'], 'cup_timing': row['_cup_timing'],
                    })
                    pickup_ready_rows += int(any(row['_pickup_wait_mask']))
                    pickup_suppressed += sum(row['_pickup_wait_mask'])
        coverage.append({'source_index': index, 'split': source['split'], 'raw_frames': frames,
                         'raw_observed': visible, 'selected_frames': len(selected),
                         'selected_observed': selected_visible, 'rows': selected_rows,
                         'rows_observed': visible_rows})
        if with_timing:
            coverage[-1]['timing_observation_rows'] = dict(timing_counts)
        if setdown:
            key = ('settled_wait_gripper_targets_masked' if mask_settled_wait_in_prefix
                   else 'settled_wait_gripper_targets_masked_after_prefix')
            coverage[-1][key] = suppressed
        if mask_pickup_wait:
            coverage[-1]['pickup_wait_observation_rows'] = pickup_ready_rows
            coverage[-1]['pickup_wait_gripper_targets_masked'] = pickup_suppressed
        if not selected_visible:
            raise ValueError(f'Cup tracker found no rim in original source {index}')
    if any('_cup_tracking' not in row for row in rows):
        raise ValueError('A compact training observation lacks its RGB-D tracking result')
    return {'rows': len(rows), 'sources': coverage,
            'original_capture_order': True, 'cup_action_origin': 'unchanged_initial_cup',
            **({'pickup_wait_mask': 'leading_recorded_open_targets_at_observed_stable_pickup_v1'}
               if mask_pickup_wait else {})}


def attach_camera_aligned_actions(rows, episodes, proof, *, fps=6, horizon=50, selected=None):
    """Resample the original TCP and command latch on each observation's clock."""
    import numpy as np

    coverage = []
    for indices in episodes.values():
        if selected is not None:
            indices = [i for i in indices if i in selected]
            if not indices:
                continue
        source_id = int(rows[indices[0]]['_physical_source'])
        source = proof['sources'][source_id]
        root = Path(source['trial'])
        geometry = json.loads((root / 'geometry.json').read_bytes())
        cup_base = np.asarray(geometry['cup_model']['base_center_mm'], dtype=float)
        table_normal = np.asarray(geometry['table']['normal'], dtype=np.float32)
        captures, _ = _captures(source, source_id)
        stamps = np.asarray([c['robot']['received_monotonic'] for c in captures], dtype=float)
        tcp = np.asarray([_pose(c) for c in captures], dtype=float)
        if (np.any(np.diff(stamps) <= 0.) or not np.isfinite(stamps).all()):
            raise ValueError('Expected strictly ordered source feedback for camera-clock targets')
        poses = np.column_stack(((tcp[:, :3] - cup_base) / 1000., np.unwrap(np.deg2rad(tcp[:, 3:]), axis=0)))
        commands = json.loads((root / 'target-command-journal.json').read_bytes())
        gripper = [commands[0], *(c for c in commands[1:] if c['kind'] == 'gripper' and c['executed'])]
        issued = np.asarray([g['issued_monotonic'] for g in gripper], dtype=float)
        opening = np.asarray([g['target'][6] / 100. for g in gripper], dtype=float)
        if not len(gripper) or not np.isfinite(issued).all() or np.any(np.diff(issued) <= 0.):
            raise ValueError('Expected ordered original gripper targets')
        counts = []
        for i in indices:
            row = rows[i]
            times = row['_camera_time'] + np.arange(1, horizon + 1, dtype=float) / fps
            upper = np.searchsorted(stamps, times, side='left')
            grip_index = np.searchsorted(issued, times, side='right') - 1
            if grip_index[0] < 0:
                raise ValueError('Aligned target precedes the original gripper latch')
            safe_upper = np.clip(upper, 1, len(stamps) - 1)
            supported = ((upper > 0) & (upper < len(stamps))
                         & (stamps[safe_upper] - stamps[safe_upper - 1] <= .25))
            invalid = np.flatnonzero(~supported)
            count = int(invalid[0]) if len(invalid) else horizon
            if count == 0:
                raise ValueError(f'No real next TCP target for source {source_id}, row {i}')
            xyz_rpy = np.column_stack([np.interp(times[:count], stamps, p) for p in poses.T])
            anchor_rpy = np.asarray(row['_cup_relative_state'][3:6])
            delta = xyz_rpy[:, 3:] - anchor_rpy
            xyz_rpy[:, 3:] = anchor_rpy + (delta + np.pi) % (2. * np.pi) - np.pi
            row['_camera_aligned_actions'] = np.column_stack((xyz_rpy, opening[grip_index[:count]])).astype(np.float32)
            row['_cup_table_normal'] = table_normal
            stamp = row['_camera_time']
            row['_recorded_hold_remaining_s'] = next(
                (h['start_monotonic'] + h['requested_seconds'] - stamp for h in source['hold_records']
                 if h['start_monotonic'] <= stamp <= h['end_monotonic']), None)
            counts.append(count)
        coverage.append({'source_index': source_id, 'rows': len(indices), 'targets': sum(counts),
                         'full_chunks': counts.count(horizon), 'split': source['split']})
    return coverage
