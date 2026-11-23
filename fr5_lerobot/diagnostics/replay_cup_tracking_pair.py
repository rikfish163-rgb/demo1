#!/usr/bin/env python3
"""Compare two cup observers on all original eight RGB-D recordings."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from fr5_jev.cup_tracking import CupTracker as AfterTracker


def before_tracker(path):
    spec = importlib.util.spec_from_file_location('fr5_jev._cup_tracking_before', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CupTracker


def observed(value):
    return value['rim_minus_jaw_mm'] is not None


def compare(training_report, old_source):
    sources = training_report['cup_relative_data']['cup_sources']
    if [source['source_index'] for source in sources] != list(range(8)):
        raise ValueError('Expected only the original eight source recordings')
    BeforeTracker = before_tracker(old_source)
    results, total = [], []
    for source in sources:
        started = time.monotonic()
        root = Path(source['trial'])
        opening = json.loads((root / 'source-observation.json').read_text())
        geometry = json.loads((root / 'geometry.json').read_text())
        initial = {'camera_serial': opening['camera']['camera']['serial'],
                   'observed_monotonic': opening['camera']['capture']['received_monotonic'],
                   'table_normal': geometry['table']['normal'],
                   'table_offset_mm': geometry['table']['offset_mm'],
                   'rim_xyz_mm': geometry['cup_model']['rim_center_mm']}
        transform = opening['calibration']['base_from_color_camera_mm']
        before, after = BeforeTracker(initial, transform), AfterTracker(initial, transform)
        rows, durations = [], []
        for line in (root / 'continuous' / 'capture.jsonl').read_text().splitlines():
            frame = json.loads(line)
            image = cv2.imread(str(root / 'continuous' / frame['rgb_file']), cv2.IMREAD_COLOR)
            with np.load(root / 'continuous' / frame['depth_file'], allow_pickle=False) as packet:
                depth = packet['depth_mm']
                one = before.update(image, depth, frame['camera'], frame['robot'],
                                    gripper_valid=frame['gripper_feedback_valid'])
                clock = time.monotonic()
                two = after.update(image, depth, frame['camera'], frame['robot'],
                                   gripper_valid=frame['gripper_feedback_valid'])
                durations.append(time.monotonic() - clock)
            both = observed(one) and observed(two)
            delta = float(np.linalg.norm(np.array(two['rim_xyz_mm']) - one['rim_xyz_mm'])) if both else None
            rows.append({'source_index': source['source_index'], 'split': source['split'],
                         'frame_index': frame['frame_index'],
                         'camera_monotonic': frame['camera']['source_timestamp_monotonic'],
                         'gripper_feedback_valid': frame['gripper_feedback_valid'],
                         'before': one, 'after': two, 'both_observed_delta_mm': delta})
        summary = {
            'source_index': source['source_index'], 'split': source['split'], 'source_root': str(root),
            'frames': len(rows), 'before_observed': sum(observed(row['before']) for row in rows),
            'after_observed': sum(observed(row['after']) for row in rows),
            'recovered_frames': [row['frame_index'] for row in rows if not observed(row['before']) and observed(row['after'])],
            'lost_frames': [row['frame_index'] for row in rows if observed(row['before']) and not observed(row['after'])],
            'both_observed_max_delta_mm': max((row['both_observed_delta_mm'] for row in rows
                                               if row['both_observed_delta_mm'] is not None), default=None),
            'after_processing_median_s': statistics.median(durations),
            'after_processing_max_s': max(durations), 'elapsed_s': time.monotonic() - started,
        }
        results.append(summary)
        total.extend(rows)
        print(json.dumps({key: summary[key] for key in (
            'source_index', 'frames', 'before_observed', 'after_observed', 'recovered_frames',
            'lost_frames', 'both_observed_max_delta_mm', 'after_processing_median_s', 'elapsed_s')},
                         ensure_ascii=False), flush=True)
    return results, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-report', required=True, type=Path)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output_dir.mkdir()
    sources, rows = compare(json.loads(args.training_report.read_text()), args.before)
    summary = {
        'schema': 'fr5-cup-tracking-pair-replay-v1', 'status': 'offline_observer_assessed',
        'device': 'cpu', 'training_report': str(args.training_report), 'before_source': str(args.before),
        'after_source': str(Path(__import__('fr5_jev.cup_tracking', fromlist=['__file__']).__file__)),
        'source_count': len(sources), 'frame_count': len(rows),
        'before_observed': sum(source['before_observed'] for source in sources),
        'after_observed': sum(source['after_observed'] for source in sources),
        'recovered_count': sum(len(source['recovered_frames']) for source in sources),
        'lost_count': sum(len(source['lost_frames']) for source in sources),
        'sources': sources, 'new_demonstrations': 0, 'failed_trials_training': False,
        'model_inference_calls': 0, 'robot_commands': 0, 'task_success_verified': False,
        'limitations': [
            'Detection counts do not establish millimetre accuracy, grasp success or cup-bottom support.',
            'Both observers replay all raw frames; training and evaluation selections remain unchanged.',
            'The saved first-draw predictions were produced with the original observer inputs.',
        ],
    }
    with (args.output_dir / 'frames.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    print(json.dumps({key: summary[key] for key in (
        'status', 'source_count', 'frame_count', 'before_observed', 'after_observed', 'recovered_count', 'lost_count')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
