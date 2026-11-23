#!/usr/bin/env python3
"""Inspect rim evidence in the recorded close frames without changing the tracker."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from fr5_jev.cup_tracking import CupTracker

CASES = ('source3-cycle1-close_at-1468', 'source5-cycle1-close_at-2524')


def candidate_trace(frame, event, value):
    if frame.f_code != CupTracker._candidate.__code__:
        return None
    if event == 'return':
        local = frame.f_locals
        trace = {'uv': (local['xy'] + local['offset']).tolist(),
                 'radius_px': local['radius'], 'accepted': value is not None}
        if 'edge_hit' in local:
            trace['outward_edge_samples'] = int(local['edge_hit'].sum())
            trace['covered_edge_sectors'] = int(np.count_nonzero(local['sectors'] >= 2))
        if 'contrast' in local:
            trace['contrast'] = local['contrast']
        if 'raised' in local:
            trace['raised_depth_samples'] = int(local['raised'].sum())
            trace['raised_depth_sectors'] = int(np.count_nonzero(local['raised'].reshape(8, 8).sum(axis=1) >= 2))
            heights = local['heights'][np.isfinite(local['heights'])]
            trace['depth_height_percentiles_mm'] = np.percentile(heights, [10, 50, 90]).tolist() if len(heights) else []
        if 'ratio' in local:
            trace['apparent_depth_ratio'] = local['ratio']
        if value is not None:
            reason = None
        elif 'contrast' not in local:
            reason = 'radial_edge_support'
        elif 'raised' not in local:
            reason = 'interior_contrast'
        elif 'height' not in local:
            reason = 'raised_depth_support'
        elif 'ratio' not in local:
            reason = 'camera_ray_geometry'
        else:
            reason = 'apparent_depth_ratio'
        trace['rejected_by'] = reason
        TRACE.append(trace)
    return candidate_trace


TRACE = []


def diagnose(inputs):
    result = []
    for case_id in CASES:
        case = next(case for case in inputs['cases'] if case['case_id'] == case_id)
        root = Path(case['source_root'])
        opening = json.loads((root / 'source-observation.json').read_text())
        tracker = CupTracker(case['initial_cup'], opening['calibration']['base_from_color_camera_mm'])
        history = []
        captures = [json.loads(line) for line in (root / 'continuous' / 'capture.jsonl').read_text().splitlines()
                    if line.strip()]
        for capture in captures:
            index = capture['frame_index']
            if index < case['tracker_start_frame_index']:
                continue
            if index > case['source_frame_index']:
                break
            image = cv2.imread(str(root / 'continuous' / capture['rgb_file']), cv2.IMREAD_COLOR)
            with np.load(root / 'continuous' / capture['depth_file'], allow_pickle=False) as depth:
                inspected = index >= case['source_frame_index'] - 2
                TRACE.clear()
                if inspected:
                    sys.settrace(candidate_trace)
                try:
                    tracking = tracker.update(image, depth['depth_mm'], capture['camera'], capture['robot'],
                                              gripper_valid=capture['gripper_feedback_valid'])
                finally:
                    sys.settrace(None)
            if inspected:
                history.append({'frame_index': index, 'camera_monotonic': capture['camera']['source_timestamp_monotonic'],
                                'feedback_monotonic': capture['robot']['received_monotonic'],
                                'image_shape': list(image.shape), 'rgb_file': capture['rgb_file'],
                                'depth_file': capture['depth_file'], 'tracking': tracking,
                                'candidate_traces': list(TRACE)})
        observed = history[-1]['tracking']['rim_minus_jaw_mm']
        saved = case['observers']['setdown']['tracking']['rim_minus_jaw_mm']
        result.append({'case_id': case_id, 'source_root': str(root),
                       'original_frame_index': case['source_frame_index'],
                       'matches_saved_missing_rim': observed == saved,
                       'last_three_frames': history})
    rejections = Counter(candidate['rejected_by'] for case in result for frame in case['last_three_frames']
                         for candidate in frame['candidate_traces'] if candidate['rejected_by'])
    return {'schema': 'fr5-cup-close-tracking-diagnostic-v1', 'device': 'cpu',
            'model_inference_calls': 0, 'robot_commands': 0, 'task_success_verified': False,
            'candidate_rejections': dict(rejections), 'cases': result,
            'limitations': ['The original policy RGB-D is 640 x 360; current full-resolution localization '
                            'does not establish tracking quality in those recorded frames.',
                            'Candidate traces describe detector rejection; they do not establish grasp or support.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = diagnose(json.loads(args.inputs.read_text()))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'candidate_rejections': report['candidate_rejections'], 'cases': [
        {'case_id': case['case_id'], 'matches_saved_missing_rim': case['matches_saved_missing_rim'],
         'frames': [{'frame_index': frame['frame_index'], 'tracking_reason': frame['tracking']['reason'],
                     'candidate_traces': frame['candidate_traces']} for frame in case['last_three_frames']]}
        for case in report['cases']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
