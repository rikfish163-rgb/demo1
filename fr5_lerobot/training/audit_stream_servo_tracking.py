"""Read-only comparison of recorded native ServoJ targets and real feedback.

The camera capture journal samples robot feedback much slower than ServoJ.
These diagnostics do not establish high-frequency tracking or task success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected finite, nonempty six-joint errors')
    a = np.abs(a)
    return {'samples': len(a), 'mean_absolute_deg': float(a.mean()),
            'p95_absolute_deg': float(np.quantile(a, .95)), 'max_absolute_deg': float(a.max()),
            'per_joint_mean_absolute_deg': a.mean(axis=0).tolist(),
            'per_joint_max_absolute_deg': a.max(axis=0).tolist()}


def audit(materialized):
    import numpy as np
    root = Path(materialized).resolve()
    wire = root / 'stream-wire'
    index = json.loads((wire / 'index.json').read_text())
    records = []
    for part in index['chunks']:
        path = (wire / part['path']).resolve()
        if wire not in path.parents or path.stat().st_size != part['bytes'] or sha(path) != part['sha256']:
            raise ValueError('Native wire archive differs from its manifest')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != part['rows']:
            raise ValueError('Wire row count differs')
        records.extend(rows)
    if len(records) != index['count']:
        raise ValueError('Wire total count differs')
    servo = [row for row in records if row['method'] == 'ServoJ']
    if not servo or any(row.get('return_code') != 0 or row.get('error') for row in servo):
        raise ValueError('This audit requires a complete successful native ServoJ series')
    stamps = np.asarray([row['issued_monotonic'] for row in servo], dtype=float)
    targets = np.asarray([row['raw_args'][0] for row in servo], dtype=float)
    if (targets.shape != (len(servo), 6) or not np.isfinite(targets).all()
            or not np.isfinite(stamps).all() or not np.all(np.diff(stamps) > 0)):
        raise ValueError('Invalid native target positions or clocks')
    capture = root / 'continuous/capture.jsonl'
    frames = [json.loads(line)['robot'] for line in capture.read_text().splitlines() if line.strip()]
    times = np.asarray([frame['received_monotonic'] for frame in frames], dtype=float)
    joints = np.asarray([frame['joints'] for frame in frames], dtype=float)
    if (joints.shape != (len(frames), 6) or not np.isfinite(joints).all()
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)):
        raise ValueError('Invalid actual feedback positions or clocks')
    results = []
    for offset in (0., .008, .016, .032):
        # Fixed alternatives describe clock/actuator lag sensitivity; no fitted lag.
        eligible = (times-offset >= stamps[0]) & (times <= servo[-1]['returned_monotonic'])
        frame_times, measured = times[eligible], joints[eligible]
        indices = np.searchsorted(stamps, frame_times-offset, side='right')-1
        error = measured-targets[indices]
        age = frame_times-offset-stamps[indices]
        recent = age <= .032
        item = {'fixed_command_offset_seconds': offset, 'all': distribution(error),
                'recent_native_command_at_most_32ms_old': distribution(error[recent]) if recent.any() else None,
                'older_or_intentional_hold': distribution(error[~recent]) if (~recent).any() else None,
                'command_age_seconds': {'median': float(np.median(age)), 'maximum': float(age.max())}}
        results.append(item)
    return {'schema': 'fr5-recorded-servo-tracking-diagnostic-v1', 'materialized': str(root),
            'wire_index_sha256': sha(wire / 'index.json'), 'capture_journal_sha256': sha(capture),
            'native_servo_commands': len(servo), 'capture_robot_feedback_samples': len(frames),
            'feedback_time_basis': 'host_monotonic_receive_time',
            'target_pairing': 'latest issued native joint target at feedback time minus fixed offset; zero-order hold',
            'offsets_are_fitted': False, 'comparisons': results,
            'robot_commands_sent_by_audit': 0, 'task_success_verified': False,
            'limitations': ['Capture feedback is sampled at approximately 6 Hz, not at the native 125 Hz command rate.',
                            'Host receive times include transport and controller feedback latency.',
                            'The preceding native command is held during recorded peripheral pauses.',
                            'Position tracking does not establish correct cup targeting or successful grasp.'],
            'runner_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materialized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    allowed = Path('/home/data4/hts/demo1/fr5-learning').resolve()
    if allowed not in args.materialized.resolve().parents or allowed / 'audits' not in args.output.resolve().parents:
        raise ValueError('Use saved evidence and a new audit output under data4')
    result = audit(args.materialized)
    with args.output.open('x') as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(args.output), 'sha256': sha(args.output),
                      'comparisons': result['comparisons'], 'robot_commands_sent': 0}))


if __name__ == '__main__':
    main()
