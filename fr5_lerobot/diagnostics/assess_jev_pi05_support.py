#!/usr/bin/env python3
"""Compare saved pi0.5 joint forecasts with recorded hold and release evidence.

This CPU-only analysis reads existing evaluations and source recordings. It
does not run a policy, contact a robot, or establish task success.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

if not __package__ and __file__ != '<stdin>':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fr5_lerobot.training.summarize_jev_pi05 import (
    CHUNK, FPS, PROJECT, GraspKinematics, switch_timing,
)


OFFSET_MM = (5.208, 8.072, 162.)
FEEDBACK_MAX_AGE_S = .25
OPEN_THRESHOLDS = (90, 98)
PHASES = ('after_close_before_lift_probe', 'lift_probe_before_full',
          'lift_full_before_hold', 'recorded_hold', 'return_before_release')


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0, 'min': None, 'mean': None, 'median': None, 'p95': None, 'max': None}
    return {'count': len(values), 'min': float(values.min()), 'mean': float(values.mean()),
            'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'max': float(values.max())}


def target_time(record, index):
    return float(record['source_grid_time_monotonic']) + (index + 1) / FPS


def first_open(record, percent, stop=CHUNK):
    return next((i for i in range(stop) if record['valid_mask'][i]
                 and record['absolute_actions'][i][6] >= percent / 100.), None)


def projected_gripper_percent(raw_fraction):
    """Match the native stream adapter's clamp and integer rounding."""
    return int(round(min(1., max(0., float(raw_fraction))) * 100.))


def displacement(point, reference, normal):
    delta = np.asarray(point, float) - np.asarray(reference, float)
    height = float(delta @ normal)
    return {'delta_xyz_mm': delta.tolist(), 'distance_mm': float(np.linalg.norm(delta)),
            'height_along_table_normal_mm': height, 'base_z_difference_mm': float(delta[2]),
            'lateral_to_table_normal_mm': float(np.linalg.norm(delta - height * normal))}


def phase_at(timestamp, cycle):
    if timestamp < cycle['lift_probe_command_time']:
        return PHASES[0]
    if timestamp < cycle['lift_command_time']:
        return PHASES[1]
    if timestamp < cycle['hold_start']:
        return PHASES[2]
    if timestamp <= cycle['hold_end']:
        return PHASES[3]
    if timestamp < cycle['release_command_time']:
        return PHASES[4]
    return 'after_release'


class SourceEvidence:
    def __init__(self, source, kin):
        self.index, self.split, self.kin = source['source_index'], source['split'], kin
        self.path = Path(source['trial'])
        plane = json.loads((self.path / 'geometry.json').read_text())['table']
        self.normal = np.asarray(plane['normal'], dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.offset = float(plane['offset_mm'])
        journal = json.loads((self.path / 'target-command-journal.json').read_text())
        self.gripper_commands = []
        for item in journal:
            if item.get('kind') != 'gripper' or item.get('executed') is not True:
                continue
            sequence = item['sequence']
            path = self.path / f'command-{sequence:04d}.json'
            entry = {'sequence': sequence, 'issued_monotonic': float(item['issued_monotonic']),
                     'command_file': str(path), 'available': False, 'target_percent': None}
            if path.is_file():
                command = json.loads(path.read_text())
                args = command.get('args', [])
                if command.get('method') == 'MoveGripper' and len(args) >= 2:
                    entry.update(available=True, target_percent=float(args[1]),
                                 recorded_command_status=command.get('status'))
            self.gripper_commands.append(entry)
        self.gripper_commands.sort(key=lambda item: item['issued_monotonic'])
        self.gripper_command_times = [item['issued_monotonic'] for item in self.gripper_commands]
        with (self.path / 'continuous/capture.jsonl').open() as stream:
            self.frames = sorted((json.loads(line) for line in stream if line.strip()),
                                 key=lambda frame: float(frame['robot']['received_monotonic']))
        self.times = [float(frame['robot']['received_monotonic']) for frame in self.frames]
        self.by_index = {frame['frame_index']: frame for frame in self.frames}
        self.points = {}
        self.cycles = []
        for hold in source['hold_records']:
            number = hold['repetition']

            def command(phase):
                found = [item for item in journal if item.get('executed')
                         and item.get('teacher', {}).get('cycle') == number
                         and item['teacher'].get('phase') == phase]
                if len(found) != 1:
                    raise ValueError(f'source {self.index}, cycle {number}: expected one {phase} command')
                return found[0]

            close, probe, lift, returning, release = (
                command(phase) for phase in ('close', 'lift_probe', 'lift_full', 'return', 'release'))
            if lift['sequence'] != hold['lift_command_sequence'] or returning['sequence'] != hold['return_command_sequence']:
                raise ValueError(f'source {self.index}, cycle {number}: hold command association differs')
            cycle = {'cycle': number, 'hold_start': float(hold['start_monotonic']),
                     'hold_end': float(hold['end_monotonic']),
                     'hold_source_frames': [hold['start_source_frame'], hold['end_source_frame']],
                     'close_command_time': float(close['issued_monotonic']),
                     'lift_probe_command_time': float(probe['issued_monotonic']),
                     'lift_command_time': float(lift['issued_monotonic']),
                     'return_command_time': float(returning['issued_monotonic']),
                     'release_command_time': float(release['issued_monotonic']),
                     'release_sequence': release['sequence'],
                     'close_target_percent': float(close['target'][6]),
                     'release_target_percent': float(release['target'][6])}
            cycle['reference'] = self.release_reference(cycle)
            self.cycles.append(cycle)

    def point(self, frame):
        index = frame['frame_index']
        if index not in self.points:
            self.points[index] = self.kin.point(np.deg2rad(frame['robot']['joints']))
        return self.points[index]

    def last_gripper_command(self, timestamp):
        index = bisect.bisect_right(self.gripper_command_times, timestamp) - 1
        if index < 0:
            return {'available': False, 'target_percent': None, 'reason': 'no_prior_move_gripper'}
        return self.gripper_commands[index]

    def release_reference(self, cycle):
        release = cycle['release_command_time']
        frames = [f for f in self.frames if 0 <= release - float(f['robot']['received_monotonic']) <= .5
                  and f.get('gripper_feedback_valid') is True and f['robot'].get('gripper_motiondone') == 2
                  and f['robot'].get('motion_done') == 1 and abs(float(f['robot']['tcp_speed'][0])) <= 4.]
        if not frames:
            return {'status': 'unavailable', 'reason': 'no_stationary_valid_jaw_contact_before_release'}
        frame = frames[-1]
        heights = [float(self.point(f) @ self.normal) for f in frames]
        span = float(frames[-1]['robot']['received_monotonic']) - float(frames[0]['robot']['received_monotonic'])
        stable = (len(frames) >= 2 and span >= .1
                  and release - float(frame['robot']['received_monotonic']) <= FEEDBACK_MAX_AGE_S
                  and max(heights) - min(heights) <= .5)
        return {'status': 'stable_jaw_contact_proxy' if stable else 'unstable_or_insufficient_feedback',
                'frame_index': frame['frame_index'], 'robot_time_monotonic': frame['robot']['received_monotonic'],
                'camera_time_monotonic': frame['camera']['source_timestamp_monotonic'],
                'point_base_mm': self.point(frame).tolist(),
                'jaw_feedback_percent': frame['robot']['gripper_position'],
                'gripper_motiondone': frame['robot']['gripper_motiondone'],
                'robot_motion_done': frame['robot']['motion_done'],
                'tcp_speed_mm_s': frame['robot']['tcp_speed'][0],
                'pre_release_candidate_frames': [f['frame_index'] for f in frames],
                'candidate_span_s': span, 'normal_height_spread_mm': max(heights) - min(heights)}

    def feedback_at(self, timestamp, cycle, *, preceding=False):
        index = bisect.bisect_right(self.times, timestamp)
        candidates = [index - 1] if preceding else [index - 1, index]
        candidates = [i for i in candidates if 0 <= i < len(self.times)]
        if not candidates:
            return {'available': False, 'reason': 'no_feedback_within_0.25_s'}
        best = min(candidates, key=lambda i: abs(self.times[i] - timestamp))
        if abs(self.times[best] - timestamp) > FEEDBACK_MAX_AGE_S:
            return {'available': False, 'reason': 'no_feedback_within_0.25_s'}
        frame, robot = self.frames[best], self.frames[best]['robot']
        valid = frame.get('gripper_feedback_valid') is True
        result = {'available': True, 'frame_index': frame['frame_index'],
                  'robot_time_monotonic': float(robot['received_monotonic']),
                  'robot_time_minus_nominal_s': float(robot['received_monotonic']) - timestamp,
                  'camera_time_monotonic': frame['camera']['source_timestamp_monotonic'],
                  'gripper_feedback_valid': valid, 'raw_gripper_position_percent': robot['gripper_position'],
                  'gripper_motiondone': robot['gripper_motiondone'],
                  'jaw_contact_reported': robot['gripper_motiondone'] == 2 if valid else None,
                  'robot_motion_done': robot['motion_done'], 'tcp_speed_mm_s': robot['tcp_speed'][0],
                  'before_recorded_release_command': self.times[best] < cycle['release_command_time']}
        reference = cycle['reference']
        if 'point_base_mm' in reference:
            result['measured_grasp_vs_release_reference'] = displacement(
                self.point(frame), reference['point_base_mm'], self.normal)
        return result


def sources_for(dataset, kin):
    proof = json.loads((Path(dataset) / 'meta/stream_provenance.json').read_text())
    if proof['fps'] != FPS or proof['chunk_size'] != CHUNK:
        raise ValueError('Dataset timebase or chunk size differs from the saved evaluation')
    return {s['source_index']: SourceEvidence(s, kin) for s in proof['sources']}


def identity(record, cycle):
    return {**{k: record[k] for k in ('source_index', 'split', 'episode', 'row_index', 'case_index',
                                      'source_capture_frame', 'draw', 'seed', 'tags')},
            'cycle': cycle['cycle'], 'anchor_camera_time_monotonic': record['source_camera_time_monotonic'],
            'anchor_grid_time_monotonic': record['source_grid_time_monotonic'],
            'anchor_gripper_feedback_fraction': record['state'][6],
            'anchor_gripper_feedback_valid': bool(record['state'][7])}


def prediction_detail(record, index, source, cycle, kin, prefix):
    action, stamp = record['absolute_actions'][index], target_time(record, index)
    point = kin.point(action[:6])
    reference = cycle['reference']
    return {'tick': index + 1, 'index': index, 'raw_gripper_fraction': float(action[6]),
            'raw_gripper_percent': float(action[6]) * 100., 'nominal_time_monotonic': stamp,
            'seconds_from_release_command': stamp - cycle['release_command_time'],
            'inside_recorded_hold': cycle['hold_start'] <= stamp <= cycle['hold_end'],
            'recorded_source_phase_at_nominal_time': phase_at(stamp, cycle),
            'within_execution_prefix': index < prefix,
            'predicted_grasp_point_base_mm': point.tolist(), 'reference_status': reference['status'],
            'predicted_grasp_vs_release_reference': (
                displacement(point, reference['point_base_mm'], source.normal) if 'point_base_mm' in reference else None),
            'nearest_source_feedback': source.feedback_at(stamp, cycle),
            'latest_preceding_source_feedback': source.feedback_at(stamp, cycle, preceding=True)}


def opening_detail(record, source, cycle, kin, prefix):
    result = identity(record, cycle)
    frame = source.by_index.get(record['source_capture_frame'])
    valid = frame is not None and frame.get('gripper_feedback_valid') is True
    result['anchor_phase'] = phase_at(record['source_grid_time_monotonic'], cycle)
    result['anchor_jaw_contact_reported'] = frame['robot']['gripper_motiondone'] == 2 if valid else None
    if frame is not None:
        result['anchor_raw_gripper_position_percent'] = frame['robot']['gripper_position']
        result['anchor_gripper_motiondone'] = frame['robot']['gripper_motiondone']
    for percent in OPEN_THRESHOLDS:
        index = first_open(record, percent)
        result[f'first_at_least_{percent}_percent'] = (
            None if index is None else prediction_detail(record, index, source, cycle, kin, prefix))
    return result


def release_detail(record, event, source, cycle, kin, prefix):
    result = opening_detail(record, source, cycle, kin, prefix)
    result['event_where'] = event['where']
    result['recorded_release_command_sequence'] = event['command_sequence']
    result['reference_status'] = cycle['reference']['status']
    timing = switch_timing(record, event, prefix)
    result['label_switch_timing_prefix'] = timing
    result['at_first_open_label'] = None
    result['waiting_for_label_switch'] = []
    if timing['target_in_horizon']:
        target = timing['target_index']
        result['at_first_open_label'] = prediction_detail(record, target, source, cycle, kin, prefix)
        predicted = timing['predicted_index']
        end = prefix if predicted is None else predicted
        for index in range(target, end):
            if not record['valid_mask'][index]:
                continue
            point = kin.point(record['absolute_actions'][index][:6])
            reference = cycle['reference']
            result['waiting_for_label_switch'].append({
                'tick': index + 1, 'raw_gripper_percent': float(record['absolute_actions'][index][6]) * 100.,
                'predicted_grasp_vs_release_reference': (
                    displacement(point, reference['point_base_mm'], source.normal)
                    if 'point_base_mm' in reference else None),
            })
    return result


def hold_detail(record, source, cycle, kin, stop, prefix):
    indices = [i for i in range(stop) if record['valid_mask'][i]
               and cycle['hold_start'] <= target_time(record, i) <= cycle['hold_end']]
    if not indices:
        return None
    values = [float(record['absolute_actions'][i][6]) * 100. for i in indices]
    maximum = indices[int(np.argmax(values))]
    result = identity(record, cycle)
    anchor = float(record['source_grid_time_monotonic'])
    result['anchor_relation_to_recorded_hold'] = (
        'before_hold' if anchor < cycle['hold_start'] else
        'within_hold' if anchor <= cycle['hold_end'] else 'after_hold')
    frame = source.by_index.get(record['source_capture_frame'])
    valid = (frame is not None and frame.get('gripper_feedback_valid') is True
             and record['state'][7] == 1)
    result['anchor_jaw_contact_reported'] = frame['robot']['gripper_motiondone'] == 2 if valid else None
    measured = float(frame['robot']['gripper_position']) if valid else None
    anchor_time = float(frame['robot']['received_monotonic']) if frame is not None else None
    last_anchor = (source.last_gripper_command(anchor_time) if anchor_time is not None
                   else {'available': False, 'target_percent': None, 'reason': 'missing_anchor_frame'})
    result['anchor_measured_gripper_percent'] = measured
    result['anchor_measured_robot_time_monotonic'] = anchor_time
    result['anchor_last_move_gripper'] = last_anchor
    result['hold_points'] = []
    for index in indices:
        raw = float(record['absolute_actions'][index][6])
        projected = projected_gripper_percent(raw)
        nominal_last = source.last_gripper_command(target_time(record, index))
        anchor_target, nominal_target = last_anchor['target_percent'], nominal_last['target_percent']
        result['hold_points'].append({
            'tick': index + 1, 'nominal_time_monotonic': target_time(record, index),
            'raw_gripper_fraction': raw, 'projected_integer_percent': projected,
            'last_move_gripper_at_nominal_time': {
                'sequence': nominal_last.get('sequence'), 'available': nominal_last['available'],
                'target_percent': nominal_target},
            'increase_over_anchor_measured_pp': None if measured is None else projected - measured,
            'increase_over_anchor_last_command_pp': None if anchor_target is None else projected - anchor_target,
            'increase_over_nominal_last_command_pp': None if nominal_target is None else projected - nominal_target,
        })
    result.update({'horizon': stop, 'hold_tick_count': len(indices), 'hold_ticks': [i + 1 for i in indices],
                   'min_raw_gripper_percent': min(values), 'max_raw_gripper_percent': max(values),
                   'mean_raw_gripper_percent': float(np.mean(values)),
                   'max_detail': prediction_detail(record, maximum, source, cycle, kin, prefix)})
    for percent in OPEN_THRESHOLDS:
        first = next((i for i in indices if record['absolute_actions'][i][6] >= percent / 100.), None)
        result[f'first_hold_at_least_{percent}_percent'] = (
            None if first is None else prediction_detail(record, first, source, cycle, kin, prefix))
    return result


def summarize_release(rows):
    result = {'draws': len(rows),
              'stable_reference_draws': sum(r['reference_status'] == 'stable_jaw_contact_proxy' for r in rows)}
    for percent in OPEN_THRESHOLDS:
        key = f'first_at_least_{percent}_percent'
        found = [r[key] for r in rows if r[key] is not None and r[key]['within_execution_prefix']]
        result[f'first_ge_{percent}'] = {
            'within_prefix': len(found),
            'before_recorded_release_command': sum(x['seconds_from_release_command'] < 0 for x in found),
            'height_minus_reference_mm': distribution([
                x['predicted_grasp_vs_release_reference']['height_along_table_normal_mm']
                for x in found if x['reference_status'] == 'stable_jaw_contact_proxy'
                and x['predicted_grasp_vs_release_reference'] is not None]),
            'distance_from_reference_mm': distribution([
                x['predicted_grasp_vs_release_reference']['distance_mm']
                for x in found if x['reference_status'] == 'stable_jaw_contact_proxy'
                and x['predicted_grasp_vs_release_reference'] is not None]),
        }
    timing = [r['label_switch_timing_prefix'] for r in rows]
    matched = [t for t in timing if t.get('target_in_horizon') and not t.get('never_switched_in_horizon')]
    result['nearest_label_switch'] = {
        'matched_draws': len(matched), 'censored_draws': sum(t.get('never_switched_in_horizon', False) for t in timing),
        'target_outside_horizon': sum(not t['target_in_horizon'] for t in timing),
        'timing_error_s': distribution([t['timing_error_s'] for t in matched]),
        'at_label_distance_from_reference_mm': distribution([
            r['at_first_open_label']['predicted_grasp_vs_release_reference']['distance_mm']
            for r in rows if r['at_first_open_label'] is not None
            and r['reference_status'] == 'stable_jaw_contact_proxy'
            and r['at_first_open_label']['predicted_grasp_vs_release_reference'] is not None]),
        'at_label_raw_gripper_percent': distribution([
            r['at_first_open_label']['raw_gripper_percent'] for r in rows if r['at_first_open_label'] is not None]),
        'max_wait_distance_from_reference_mm': distribution([
            max(x['predicted_grasp_vs_release_reference']['distance_mm'] for x in r['waiting_for_label_switch']
                if x['predicted_grasp_vs_release_reference'] is not None)
            for r in rows if r['reference_status'] == 'stable_jaw_contact_proxy'
            and any(x['predicted_grasp_vs_release_reference'] is not None
                                 for x in r['waiting_for_label_switch'])]),
    }
    return result


def summarize_hold(rows):
    result = {'draws_with_recorded_hold_ticks': len(rows),
            'hold_ticks_across_draws': sum(r['hold_tick_count'] for r in rows),
            'draws_by_anchor_relation': dict(Counter(r['anchor_relation_to_recorded_hold'] for r in rows)),
            **{f'draws_ge_{p}_by_anchor_relation': dict(Counter(
                r['anchor_relation_to_recorded_hold'] for r in rows if r[f'first_hold_at_least_{p}_percent'] is not None))
               for p in OPEN_THRESHOLDS},
            **{f'draws_with_hold_target_at_least_{p}_percent': sum(
                r[f'first_hold_at_least_{p}_percent'] is not None for r in rows) for p in OPEN_THRESHOLDS},
            'per_draw_min_raw_gripper_percent': distribution([r['min_raw_gripper_percent'] for r in rows]),
            'per_draw_max_raw_gripper_percent': distribution([r['max_raw_gripper_percent'] for r in rows])}
    for label, field in (
        ('projected_vs_anchor_measured', 'increase_over_anchor_measured_pp'),
        ('projected_vs_anchor_last_command', 'increase_over_anchor_last_command_pp'),
        ('projected_vs_nominal_last_command', 'increase_over_nominal_last_command_pp'),
    ):
        eligible = [(row, point) for row in rows for point in row['hold_points'] if point[field] is not None]
        over = [(row, point) for row, point in eligible if point[field] >= 2.]
        witness = None
        if eligible:
            row, point = max(eligible, key=lambda pair: pair[1][field])
            witness = {**{key: row[key] for key in ('source_index', 'cycle', 'source_capture_frame',
                                                     'row_index', 'draw', 'seed')},
                       'tick': point['tick'], 'raw_gripper_fraction': point['raw_gripper_fraction'],
                       'projected_integer_percent': point['projected_integer_percent'],
                       'increase_percentage_points': point[field]}
        result[label] = {
            'eligible_draws': sum(any(point[field] is not None for point in row['hold_points']) for row in rows),
            'eligible_points': len(eligible), 'missing_points': result['hold_ticks_across_draws'] - len(eligible),
            'draws_at_least_2pp': sum(any(point[field] is not None and point[field] >= 2.
                                           for point in row['hold_points']) for row in rows),
            'points_at_least_2pp': len(over),
            'maximum_increase_pp': None if witness is None else witness['increase_percentage_points'],
            'maximum_witness': witness,
        }
    return result


def summarize_phases(rows):
    result = {}
    for phase in PHASES:
        phase_rows = [r for r in rows if r['anchor_phase'] == phase]
        confirmed = [r for r in phase_rows if r['anchor_jaw_contact_reported'] is True]
        item = {'anchor_draws': len(phase_rows), 'jaw_contact_reported_anchor_draws': len(confirmed),
                'unknown_anchor_jaw_feedback_draws': sum(r['anchor_jaw_contact_reported'] is None for r in phase_rows)}
        for percent in OPEN_THRESHOLDS:
            key = f'first_at_least_{percent}_percent'
            found = [r[key] for r in confirmed if r[key] is not None and r[key]['within_execution_prefix']]
            item[f'first_ge_{percent}_within_prefix'] = len(found)
            item[f'first_ge_{percent}_source_phase_at_nominal_time'] = dict(Counter(
                x['recorded_source_phase_at_nominal_time'] for x in found))
            item[f'first_ge_{percent}_height_minus_reference_mm'] = distribution([
                x['predicted_grasp_vs_release_reference']['height_along_table_normal_mm']
                for x in found if x['reference_status'] == 'stable_jaw_contact_proxy'
                and x['predicted_grasp_vs_release_reference'] is not None])
        result[phase] = item
    return result


def assess_model(records, sources, kin, prefix):
    releases, anchors, holds_prefix, holds_full = [], [], [], []
    for record in records:
        source = sources[record['source_index']]
        for event in record['events']:
            if event['kind'] == 'open':
                matches = [c for c in source.cycles if c['release_sequence'] == event['command_sequence']]
                if len(matches) != 1:
                    raise ValueError('Open event does not match exactly one recorded cycle')
                releases.append(release_detail(record, event, source, matches[0], kin, prefix))
        for cycle in source.cycles:
            grid = float(record['source_grid_time_monotonic'])
            if cycle['close_command_time'] <= grid < cycle['release_command_time']:
                anchors.append(opening_detail(record, source, cycle, kin, prefix))
            for stop, destination in ((prefix, holds_prefix), (CHUNK, holds_full)):
                detail = hold_detail(record, source, cycle, kin, stop, prefix)
                if detail is not None:
                    destination.append(detail)
    summaries = {}
    for split in ('all', 'train', 'validation'):
        def selected(rows):
            return [r for r in rows if split == 'all' or r['split'] == split]
        opening = selected(releases)
        summaries[split] = {
            'release_anchors': {where: summarize_release([r for r in opening if r['event_where'] == where])
                                for where in ('before', 'at')},
            'recorded_hold_prefix': summarize_hold(selected(holds_prefix)),
            'recorded_hold_full': summarize_hold(selected(holds_full)),
            'first_open_from_recorded_contact_anchors': summarize_phases(selected(anchors)),
        }
    return {'summary': summaries, 'release_details': releases,
            'hold_details_prefix': holds_prefix, 'hold_details_full': holds_full,
            'cycle_anchor_details': [r for r in anchors if any(
                r[f'first_at_least_{p}_percent'] is not None and
                r[f'first_at_least_{p}_percent']['within_execution_prefix'] for p in OPEN_THRESHOLDS)]}


def assess(evaluations, *, kin=None):
    paths = [Path(path).resolve() for path in evaluations]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Provide distinct completed evaluation directories')
    metadata = [json.loads((path / 'summary.json').read_text()) for path in paths]
    datasets = {m['dataset'] for m in metadata}
    if len(datasets) != 1:
        raise ValueError('Compared evaluations must use the same recorded dataset')
    if any(m.get('evaluation') != 'open_loop_recorded_label_fit_not_task_success'
           or m.get('dataset_matches_checkpoint') is not True for m in metadata):
        raise ValueError('Expected completed recorded-output evaluations')
    dataset = datasets.pop()
    kin = kin if kin is not None else GraspKinematics(PROJECT / 'hardware/fr5.urdf', OFFSET_MM)
    sources = sources_for(dataset, kin)
    result = {
        'analysis': 'offline_recorded_outputs_not_task_success', 'dataset': dataset,
        'grasp_offset_flange_mm': list(OFFSET_MM), 'fps': FPS,
        'method': {
            'time': 'source grid time + (0-based output index + 1)/6 seconds',
            'opening': 'first valid raw gripper target >=90% or >=98%; values are not clipped',
            'hold': 'all valid forecast ticks inside provenance-recorded hold intervals',
            'hold_gripper_projection': 'clamp raw fraction to 0..1 and round 100*fraction to integer percent',
            'hold_baselines': 'valid measured anchor position and recorded MoveGripper wire target at anchor and nominal tick',
            'deadband_comparison': 'integer target at least 2 percentage points above baseline may request opening',
            'reference': 'latest stationary valid jaw-contact feedback within 0.5 seconds before recorded release',
            'feedback': 'nearest and latest preceding robot feedback within 0.25 seconds; invalid feedback is unknown',
            'distance': 'GraspKinematics of forecast joints against measured pre-release joints and recorded table normal',
            'label_switch': 'nearest old/new target timing is a label comparison, not proof of contact or release',
        },
        'limitations': [
            'Raw gripper thresholds describe model targets; they do not establish gripper motion, contact or a drop.',
            'Deadband comparisons use fixed recorded baselines; they do not simulate previous model targets or actual commands.',
            'A stable pose and jaw contact signal do not prove that the object is supported by the table.',
            'Full-horizon outputs are open-loop forecasts; only the configured prefix is the prospective execution prefix.',
            'The grasp offset is estimated and recorded future feedback does not prove the model would reach that state.',
            'Prediction draws are not physical attempts; offline fit cannot establish task success.',
        ],
        'sources': {str(i): {'trial': str(s.path), 'split': s.split, 'table_normal': s.normal.tolist(),
                             'table_offset_mm': s.offset, 'gripper_commands': s.gripper_commands,
                             'cycles': s.cycles} for i, s in sources.items()},
        'evaluations': {},
    }
    for path, meta in zip(paths, metadata):
        prefix = meta['prefix_steps']
        if type(prefix) is not int or not 1 <= prefix <= CHUNK:
            raise ValueError(f'{path}: invalid prefix length')
        models = {}
        for model in meta['models']:
            with (path / f'{model}.jsonl').open() as stream:
                models[model] = assess_model((json.loads(line) for line in stream if line.strip()), sources, kin, prefix)
        result['evaluations'][path.name] = {'path': str(path), 'checkpoint': meta['checkpoint'],
                                             'prefix_steps': prefix, 'models': models}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', type=Path, required=True,
                        help='Completed evaluation directory; repeat to compare checkpoints')
    parser.add_argument('--output', type=Path, required=True, help='New report file; never overwritten')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    report = assess(args.evaluation)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'evaluations': list(report['evaluations'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
