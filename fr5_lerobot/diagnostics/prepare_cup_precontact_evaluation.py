#!/usr/bin/env python3
"""Join fresh cup observers to the original evaluation images and action labels."""

import argparse
import base64
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

from evaluate_cup_timing_policy import read_json, read_jsonl, replay, write_new


def same_tracking(left, right):
    if left['native_contact'] != right['native_contact']:
        return False
    a, b = left['rim_minus_jaw_mm'], right['rim_minus_jaw_mm']
    if a is None or b is None:
        return a is None and b is None
    return all(abs(x - y) <= 1e-9 for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--replay-details', type=Path, required=True)
    parser.add_argument('--heldout-inputs', type=Path, required=True)
    parser.add_argument('--fit-inputs', type=Path, required=True)
    parser.add_argument('--trial-inputs', type=Path, required=True)
    parser.add_argument('--trial02', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    details = read_json(args.replay_details)
    if not details['complete'] or details['failures']:
        raise ValueError('Original-frame tracker replay is incomplete')
    evidence = {str(Path(row['trial'])): row for row in details['results']}
    heldout = read_json(args.heldout_inputs)
    fit = read_json(args.fit_inputs)
    trial = read_json(args.trial_inputs)
    source7 = Path(next(c['source_root'] for c in heldout['cases'] if c['group'] == 'heldout_source_7'))
    calibration = read_json(source7 / 'source-observation.json')['calibration']
    scope = read_json(args.trial02 / 'scope.json')
    review = read_json(args.trial02 / 'user-review.json')
    if scope['dataset_role'] != 'policy_evaluation' or review['verdict'] != 'failure':
        raise ValueError('trial02 must remain a failed evaluation')
    captures = read_jsonl(args.trial02 / 'continuous/capture.jsonl')
    by_stamp = {row['camera']['source_timestamp_monotonic']: row for row in captures}
    trial02_cases = []
    for path in sorted(args.trial02.glob('chunk-*.json')):
        saved = read_json(path)
        request = deepcopy(saved['request'])
        capture = by_stamp[request['source_time']]
        request['rgb_jpeg_base64'] = base64.b64encode(
            (args.trial02 / 'continuous' / capture['rgb_file']).read_bytes()).decode('ascii')
        trial02_cases.append({
            'case_id': args.trial02.parent.name + '-' + path.stem,
            'group': 'failed_trial_evaluation_only', 'kind': 'failed_real_approach',
            'source_root': str(args.trial02), 'source_frame_index': capture['frame_index'],
            'tracker_start_frame_index': 0, 'initial_cup': scope['initial_cup'],
            'request': request, 'tracking': request['cup_tracking'],
            'timing_setdown': {'state': request['cup_timing']},
        })
    trial02 = {'schema': heldout['schema'], 'case_count': len(trial02_cases),
               'timing_ready': True, 'failed_trial': str(args.trial02),
               'failed_trial_use': 'evaluation_only', 'user_review': review,
               'cases': trial02_cases, 'robot_commands': 0}
    reports = {'heldout-inputs.json': heldout, 'fit-inputs.json': fit,
               'trial01-inputs.json': trial, 'trial02-inputs.json': trial02}
    groups = defaultdict(list)
    for name, report in reports.items():
        for case in report['cases']:
            groups[case['source_root']].append((name, case))
    counts = defaultdict(lambda: {'cases': 0, 'tracking_changed': 0, 'timing_changed': 0})
    for root_name, cases in groups.items():
        root = Path(root_name)
        selected = {case['source_frame_index'] for _, case in cases}
        starts = {case.get('tracker_start_frame_index', 0) for _, case in cases}
        if len(starts) != 1:
            raise ValueError('Cases from one source have different observer starts')
        start = starts.pop()
        tracked, frames = {}, {}
        if root_name in evidence and start == 0:
            item = evidence[root_name]
            for line in Path(item['evidence']).read_text().splitlines():
                row = json.loads(line)
                frame = row['capture']['frame_index']
                if frame in selected:
                    tracked[frame] = row['tracking']['new'], row['timing']['new']
                    frames[frame] = row['capture']
        else:
            raw = read_jsonl(root / 'continuous/capture.jsonl')
            tracked = replay(root, raw, cases[0][1]['initial_cup'], calibration, selected,
                             start, True, aligned=True, setdown=True)
            frames = {row['frame_index']: row for row in raw if row['frame_index'] in selected}
        if set(tracked) != selected:
            raise ValueError(f'Missing original evaluation frames: {root}')
        for name, case in cases:
            frame = frames[case['source_frame_index']]
            if frame['camera']['source_timestamp_monotonic'] != case['request']['source_time']:
                raise ValueError('Observer and request camera clocks differ')
            track, timing = tracked[case['source_frame_index']]
            counter = counts[name]
            counter['cases'] += 1
            if name == 'fit-inputs.json':
                old = case['observers']['setdown']
                counter['tracking_changed'] += not same_tracking(old['tracking'], track)
                counter['timing_changed'] += old['model_timing'] != timing['state']
                case['observers']['setdown'] = {'tracking': track, 'timing': timing,
                                                'model_timing': timing['state'],
                                                'support_gap_source': 'observer'}
            else:
                counter['tracking_changed'] += not same_tracking(case['tracking'], track)
                counter['timing_changed'] += case['timing_setdown']['state'] != timing['state']
                case['tracking'] = track
                case['timing_setdown'] = timing
    fit['observation_replays']['setdown'] = str(args.source_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, report in reports.items():
        report['observer_source_root'] = str(args.source_root)
        report['observer_replay_details'] = str(args.replay_details)
        write_new(args.output_dir / name, report)
    summary = {'schema': 'fr5-cup-precontact-evaluation-inputs-v1',
               'source_root': str(args.source_root), 'replay_details': str(args.replay_details),
               'outputs': dict(counts), 'failed_trial_use': 'evaluation_only',
               'preserved': ['original JPEG', 'camera timestamp', 'robot and gripper feedback',
                             'history', 'task instruction', 'original action targets'],
               'robot_commands': 0}
    write_new(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
