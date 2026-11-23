"""Summarize recorded pi0.5 outputs in FR5 millimeters and gripper timing."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from fr5_runtime.kinematics import model_chain

EVALUATIONS = Path('/home/data4/hts/demo1/fr5-learning/evaluations')
FPS = 6
CHUNK = 50


def distribution(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {'count': 0, 'mean': None, 'p95': None, 'max': None}
    return {'count': len(array), 'mean': float(array.mean()),
            'p95': float(np.quantile(array, .95)), 'max': float(array.max())}


class GraspKinematics:
    def __init__(self, urdf, offset_mm):
        self.offset = np.asarray(offset_mm, dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError('Expected a finite flange-frame grasp offset in millimeters')
        self.joints = [joint for joint in ET.parse(urdf).getroot().findall('joint')
                       if joint.get('type') == 'revolute']
        if len(self.joints) != 6:
            raise ValueError('Expected the six-joint FR5 URDF')

    def point(self, radians):
        q = np.asarray(radians, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Expected six finite joint angles in radians')
        return np.asarray(model_chain(np.degrees(q), self.joints, tool_offset_mm=self.offset)[-1])

    def points(self, radians):
        return np.asarray([self.point(q) for q in radians])


def measured_hold_mask(record, valid):
    """Restrict hold metrics to future labels inside the actual recorded hold."""
    times = float(record['source_grid_time_monotonic']) + (np.arange(CHUNK) + 1) / FPS
    mask = np.zeros(CHUNK, dtype=bool)
    for hold in record.get('holds', []):
        mask |= ((times >= float(hold['start_monotonic']))
                 & (times <= float(hold['end_monotonic'])))
    return mask & valid


def switch_timing(record, event, stop):
    """Compare the first predicted nearest-new target with the real command."""
    prediction = np.asarray(record['absolute_actions'], dtype=float)[:stop, 6]
    target = np.asarray(record['target_absolute'], dtype=float)[:stop, 6]
    valid = np.asarray(record['valid_mask'], dtype=bool)[:stop]
    old, new = float(event['old_target']), float(event['new_target'])
    actual = np.flatnonzero(valid & np.isclose(target, new, atol=1e-5, rtol=0))
    if not len(actual):
        return {'target_in_horizon': False}
    target_index = int(actual[0])
    predicted_new = valid & (np.abs(prediction - new) < np.abs(prediction - old))
    predicted = np.flatnonzero(predicted_new)
    predicted_index = int(predicted[0]) if len(predicted) else None
    last_index = int(np.flatnonzero(valid)[-1])
    age = float(record['source_grid_time_monotonic']) - float(record['source_camera_time_monotonic'])
    return {'target_in_horizon': True, 'target_index': target_index,
            'predicted_index': predicted_index,
            'target_time_from_camera_s': age + (target_index + 1) / FPS,
            'predicted_time_from_camera_s': (
                None if predicted_index is None else age + (predicted_index + 1) / FPS),
            'timing_error_s': None if predicted_index is None else (predicted_index - target_index) / FPS,
            'never_switched_in_horizon': predicted_index is None,
            'late_lower_bound_s': (last_index - target_index) / FPS if predicted_index is None else None}


class PositionMetrics:
    def __init__(self):
        self.errors = []
        self.vertical_errors = []
        self.still_predicted = []
        self.still_recorded = []
        self.hold_errors = []
        self.hold_vertical = []

    def add(self, prediction, target, anchor, valid, hold):
        p, t = prediction[valid], target[valid]
        errors = np.linalg.norm(p - t, axis=1)
        vertical = p[:, 2] - t[:, 2]
        recorded = np.linalg.norm(t - anchor, axis=1)
        predicted = np.linalg.norm(p - anchor, axis=1)
        still = recorded <= 1.
        self.errors.extend(errors.tolist())
        self.vertical_errors.extend(vertical.tolist())
        self.still_predicted.extend(predicted[still].tolist())
        self.still_recorded.extend(recorded[still].tolist())
        self.hold_errors.extend(np.linalg.norm(prediction[hold] - target[hold], axis=1).tolist())
        self.hold_vertical.extend((prediction[hold, 2] - target[hold, 2]).tolist())

    def result(self):
        return {'grasp_position_error_mm': distribution(self.errors),
                'absolute_vertical_error_mm': distribution(np.abs(self.vertical_errors)),
                'signed_vertical_error_mean_mm': (
                    float(np.mean(self.vertical_errors)) if self.vertical_errors else None),
                'target_within_1mm_of_anchor': {
                    'predicted_displacement_mm': distribution(self.still_predicted),
                    'recorded_displacement_mm': distribution(self.still_recorded)},
                'recorded_hold': {'grasp_position_error_mm': distribution(self.hold_errors),
                                  'absolute_vertical_error_mm': distribution(np.abs(self.hold_vertical))}}


class TimingMetrics:
    def __init__(self):
        self.count = self.missing_target = self.early = self.on_time = self.late = self.never = 0
        self.errors = []
        self.lower_bounds = []

    def add(self, timing):
        self.count += 1
        if not timing['target_in_horizon']:
            self.missing_target += 1
        elif timing['never_switched_in_horizon']:
            self.never += 1
            self.lower_bounds.append(timing['late_lower_bound_s'])
        else:
            error = timing['timing_error_s']
            self.errors.append(error)
            self.early += int(error < 0)
            self.on_time += int(error == 0)
            self.late += int(error > 0)

    def result(self):
        return {'prediction_draws': self.count, 'target_not_in_horizon': self.missing_target,
                'early': self.early, 'same_label_step': self.on_time, 'late': self.late,
                'never_switched_in_horizon': self.never,
                'timing_error_when_switched_s': distribution(self.errors),
                'censored_late_lower_bound_s': distribution(self.lower_bounds)}


class Summary:
    def __init__(self, kinematics, prefix):
        if type(prefix) is not int or not 1 <= prefix <= CHUNK:
            raise ValueError('Expected the evaluated execution prefix')
        self.kinematics, self.prefix = kinematics, prefix
        self.groups = {}
        self.timings = {}
        self.anchors = defaultdict(set)
        self.draws = defaultdict(int)
        self.details = []

    def add(self, record):
        p = np.asarray(record['absolute_actions'], dtype=float)
        t = np.asarray(record['target_absolute'], dtype=float)
        valid = np.asarray(record['valid_mask'], dtype=bool)
        state = np.asarray(record['state'], dtype=float)
        if (p.shape != (CHUNK, 7) or t.shape != (CHUNK, 7) or valid.shape != (CHUNK,)
                or state.shape != (8,) or not np.isfinite(state).all()
                or not np.isfinite(p[valid]).all() or not np.isfinite(t[valid]).all()
                or record['split'] not in ('train', 'validation')):
            raise ValueError('Expected saved finite FR5 50x7 outputs and source labels')
        prediction = np.full((CHUNK, 3), np.nan)
        target = np.full((CHUNK, 3), np.nan)
        prediction[valid] = self.kinematics.points(p[valid, :6])
        target[valid] = self.kinematics.points(t[valid, :6])
        anchor = self.kinematics.point(state[:6])
        hold = measured_hold_mask(record, valid)
        split = record['split']
        self.anchors[split].add((record['source_index'], record['row_index']))
        self.draws[split] += 1
        for tag in dict.fromkeys(['all', *record['tags']]):
            key = (split, tag)
            group = self.groups.setdefault(key, {name: PositionMetrics() for name in ('first', 'prefix', 'full')})
            for name, stop in (('first', 1), ('prefix', self.prefix), ('full', CHUNK)):
                group[name].add(prediction[:stop], target[:stop], anchor, valid[:stop], hold[:stop])
        for event in record.get('events', []):
            key = (split, event['kind'] + '_' + event['where'])
            timing_group = self.timings.setdefault(key, {name: TimingMetrics() for name in ('prefix', 'full')})
            detail = {'source_index': record['source_index'], 'row_index': record['row_index'],
                      'draw': record['draw'], 'kind': event['kind'], 'where': event['where'],
                      'event_row_index': event['event_row_index']}
            for name, stop in (('prefix', self.prefix), ('full', CHUNK)):
                timing = switch_timing(record, event, stop)
                timing_group[name].add(timing)
                detail[name] = timing
            self.details.append(detail)

    def result(self):
        result = {}
        for split in ('train', 'validation'):
            result[split] = {
                'unique_anchors': len(self.anchors[split]), 'prediction_draws': self.draws[split],
                'position': {tag: {name: metrics.result() for name, metrics in group.items()}
                             for (group_split, tag), group in self.groups.items() if group_split == split},
                'switch_timing': {tag: {name: metrics.result() for name, metrics in group.items()}
                                  for (group_split, tag), group in self.timings.items() if group_split == split}}
        return result


def summarize(evaluation, output, offset_mm, *, urdf=PROJECT / 'hardware/fr5.urdf'):
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    if not output.is_relative_to(EVALUATIONS.resolve()):
        raise ValueError('Output must be inside data4/fr5-learning/evaluations')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    original = json.loads((evaluation / 'summary.json').read_text())
    if (original.get('evaluation') != 'open_loop_recorded_label_fit_not_task_success'
            or original.get('dataset_matches_checkpoint') is not True):
        raise ValueError('Expected a complete captured FR5/Jev output evaluation')
    kinematics = GraspKinematics(urdf, offset_mm)
    models, details = {}, {}
    for model in original['models']:
        summary = Summary(kinematics, original['prefix_steps'])
        with (evaluation / (model + '.jsonl')).open() as stream:
            for line in stream:
                if line.strip():
                    summary.add(json.loads(line))
        models[model], details[model] = summary.result(), summary.details
    result = {'evaluation': 'open_loop_modeled_grasp_position_and_command_timing_not_task_success',
              'source_evaluation': str(evaluation), 'checkpoint': original['checkpoint'],
              'prefix_steps': original['prefix_steps'], 'grasp_offset_flange_mm': kinematics.offset.tolist(),
              'kinematics': 'controller-verified FR5 model; grip offset is an estimate',
              'position_definition': 'Euclidean and base-Z prediction errors against real future joint-feedback labels',
              'stationary_definition': 'modeled real target within 1 mm of modeled anchor grasp point',
              'hold_definition': 'valid future labels inside the recorded physical hold interval',
              'switch_definition': 'first valid prediction closer to new than old recorded gripper command',
              'censoring': 'no switch in the evaluated horizon gives a delay lower bound, not an exact switch time',
              'denominators': 'prediction draws and valid target steps; draws are not physical attempts',
              'models': models, 'switch_details': details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--grasp-offset', required=True, type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--urdf', type=Path, default=PROJECT / 'hardware/fr5.urdf')
    args = parser.parse_args()
    result = summarize(args.evaluation, args.output, args.grasp_offset, urdf=args.urdf)
    print(json.dumps({'output': str(args.output), 'models': list(result['models']),
                      'prefix_steps': result['prefix_steps']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
