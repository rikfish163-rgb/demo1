"""Synthetic reviewed teacher recordings for stream data tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot_robot_fairino.human_review import record_review
from lerobot_robot_fairino.whole_task_source import sha

TASK = '抓取桌面上的透明杯子，在空中维持5秒后放下'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values), encoding='utf-8')


def make_source(root, trial=0, *, skip_frame=None, hold_duration=5.1, hold_seconds=5.,
                repetitions=1, finish='released', instruction=TASK, first_gripper_command=False,
                canonical_hold_without_summary_list=False, initial_hold_time=99.9, after_hold_delay_s=.2,
                retreat_mode=None):
    (root / 'continuous/frames').mkdir(parents=True)
    (root / 'lerobot/meta').mkdir(parents=True)
    (root / 'proof.jpg').write_bytes(b'synthetic-review-evidence')
    hold_repetitions = [1, 2] if repetitions == 2 and finish == 'released' else [1]
    task_spec = {'hold_seconds': hold_seconds, 'repetitions': repetitions, 'finish': finish,
                 'hold_repetitions': hold_repetitions}
    phase_order = ['approach_clear', 'close', 'lift_full', 'return', 'release', 'retreat_high']
    if repetitions == 2:
        phase_order += (['approach_clear', 'close', 'lift_full'] if finish == 'held'
                        else ['approach_clear', 'close', 'lift_full', 'return', 'release', 'retreat_high'])
    if retreat_mode is not None:
        if retreat_mode != 'short' or repetitions != 1 or finish != 'released':
            raise ValueError('The short-retreat fixture requires one released cycle')
        phase_order = ['approach_clear', 'approach_near', 'approach_contact', 'close', 'lift_probe',
                       'lift_full', 'return', 'release', 'retreat_probe', 'retreat_clear']
    scope = {'task_instruction': instruction, 'whole_task': True, 'task_spec': task_spec,
             'phase_order': phase_order,
             'split_role': 'training_complete_trial', 'synthetic_id': trial}
    if retreat_mode is not None:
        scope['retreat_mode'] = retreat_mode
    write_json(root / 'scope.json', scope)
    hold = {'repetition': 1, 'requested_seconds': hold_seconds, 'start_monotonic': 104.1,
            'end_monotonic': 104.1 + hold_duration, 'actual_duration_s': hold_duration}
    hold_path = 'hold-01.json' if canonical_hold_without_summary_list else 'hold-0001.json'
    write_json(root / hold_path, hold)
    returning = 104.1 + hold_duration + after_hold_delay_s
    hold_names = [hold_path]
    if len(hold_repetitions) == 2:
        hold_start = returning + 4.4
        write_json(root / 'hold-0002.json', {'repetition': 2, 'requested_seconds': hold_seconds,
                   'start_monotonic': hold_start, 'end_monotonic': hold_start + hold_duration,
                   'actual_duration_s': hold_duration})
        hold_names.append('hold-0002.json')
    summary = {'outcome': 'pending_review', 'released': finish == 'released', 'failed': None, 'evidence': ['proof.jpg'],
               'completed_stages': scope['phase_order'], 'scope_sha256': sha(root / 'scope.json'),
               'whole_task': True, 'task_instruction': instruction, 'task_spec': task_spec,
               'hold_records': [{'repetition': rep, 'path': path}
                                for rep, path in zip(hold_repetitions, hold_names)],
               'held_seconds': hold_duration,
               'continuous_capture': {'capture_complete': True}}
    if canonical_hold_without_summary_list:
        del summary['hold_records']
    write_json(root / 'summary.json', summary)
    journal = [{'sequence': 1, 'issued_monotonic': initial_hold_time, 'issued_unix': initial_hold_time + 1900,
                'kind': 'hold', 'executed': False, 'target': [0.] * 6 + [100.]}]
    specs = [
        (100.60, 'arm', 'approach_clear', 10),
        (101.10, 'arm', 'approach_clear', 20),
        (101.70, 'arm', 'approach_clear', 30),
        (102.40, 'gripper', 'close', 76),
        (103.10, 'arm', 'lift_probe', 35),
        (103.80, 'arm', 'lift_full', 40),
        (returning, 'arm', 'return', 30),
        (returning + .7, 'gripper', 'release', 100),
        (returning + 1.4, 'arm', 'retreat_clear', 20),
        (returning + 2.0, 'arm', 'retreat_high', 10),
    ]
    if retreat_mode == 'short':
        specs[1] = (101.10, 'arm', 'approach_near', 20)
        specs[2] = (101.70, 'arm', 'approach_contact', 30)
        specs[-2] = (returning + 1.4, 'arm', 'retreat_probe', 25)
        specs[-1] = (returning + 2.0, 'arm', 'retreat_clear', 20)
    if first_gripper_command:
        specs.insert(0, (100.25, 'gripper', 'initial_open', 100))
    if repetitions == 2:
        specs += [(returning + 2.7, 'arm', 'approach_clear', 20),
                  (returning + 3.4, 'gripper', 'close', 76),
                  (returning + 4.1, 'arm', 'lift_full', 40)]
        if finish == 'released':
            second_return = hold_start + hold_duration + .2
            specs += [(second_return, 'arm', 'return', 30),
                      (second_return + .7, 'gripper', 'release', 100),
                      (second_return + 1.4, 'arm', 'retreat_high', 10)]
    command_names = []
    for index, (issued, kind, phase, value) in enumerate(specs, 2):
        last = journal[-1]['target']
        target = [float(value)] * 6 + last[6:] if kind == 'arm' else last[:6] + [float(value)]
        command = {'sequence': index, 'issued_monotonic': issued, 'issued_unix': issued + 1900,
                   'kind': kind, 'executed': True, 'target': target,
                   'teacher': {'phase': phase}}
        journal.append(command)
        method, args = ('MoveL', [target[:6], [0] * 6]) if kind == 'arm' else ('MoveGripper', [1, target[6]])
        record = {'status': 'completed', 'command': command, 'intended_target': target,
                  'method': method, 'args': args, 'phase': phase, 'outcome': {'return_code': 0},
                  'completed_monotonic': issued + .2, 'completed_at_unix': issued + 1900.2}
        name = f'command-{index:04d}.json'
        write_json(root / name, record); command_names.append(name)
    write_json(root / 'target-command-journal.json', journal)
    hold_events = [{'event_id': 1, 'kind': 'phase', 'name': f'hold_{hold_seconds:g}s', 'time_monotonic': 104.05}]
    if len(hold_repetitions) == 2:
        hold_events.append({'event_id': 2, 'kind': 'phase', 'name': f'hold_{hold_seconds:g}s',
                            'time_monotonic': hold_start - .05})
    write_jsonl(root / 'continuous/events.jsonl', hold_events)
    jpeg = root / 'example.jpg'
    Image.new('RGB', (640, 360), (40 + trial, 120, 70)).save(jpeg, format='JPEG')
    raw = jpeg.read_bytes(); jpeg.unlink()
    captures, teachers, media = [], [], []
    for n in range(int((specs[-1][0] + 1.0 - 100) * 6) + 2):
        if n == skip_frame:
            continue
        t = 100 + n / 6
        index = len(captures)
        camera = {'source_timestamp_monotonic': t, 'source_timestamp_unix': t + 1900,
                  'source_camera_serial': 'SYNTHETIC'}
        state = {'received_monotonic': t + .005, 'timestamp': t + 1900.005,
                 'joints': [float(trial) + n / 100] * 6, 'gripper_position': 82,
                 'gripper_active': 1, 'gripper_fault': 0, 'gripper_fault_id': 0,
                 'gripper_motiondone': 1, 'gripper_speed': 0}
        rgb_file = f'frames/{index:06d}.jpg'
        (root / 'continuous' / rgb_file).write_bytes(raw)
        capture = {'frame_index': index, 'camera': camera, 'robot': state, 'sync_clock': 'host_monotonic',
                   'sync_skew_s': .005, 'rgb_file': rgb_file}
        eligible_commands = [c for c in journal if c['issued_monotonic'] <= t]
        command = eligible_commands[-1] if eligible_commands else journal[0]
        captures.append(capture)
        teachers.append({**{k: capture[k] for k in ('frame_index', 'camera', 'robot', 'sync_clock', 'sync_skew_s')},
                         'command': command})
        media.append({'sequence': index, 'camera': camera, 'rgb_file': rgb_file,
                      'rgb_jpeg_sha256': hashlib.sha256(raw).hexdigest()})
    write_jsonl(root / 'continuous/capture.jsonl', captures)
    write_jsonl(root / 'lerobot/meta/capture.jsonl', teachers)
    write_jsonl(root / 'stream-frames.jsonl', media)
    for name in command_names:
        wire = json.loads((root / name).read_text())
        issued = wire['command']['issued_monotonic']
        eligible = [c for c in captures if c['camera']['source_timestamp_monotonic'] <= issued
                    and c['robot']['received_monotonic'] <= issued]
        wire['observation_before_command'] = eligible[-1]['camera']
        write_json(root / name, wire)
    names = ['scope.json', 'summary.json', *hold_names, 'continuous/events.jsonl',
             'target-command-journal.json', *command_names]
    manifest = {'session': f'synthetic-{trial}', 'scope_sha256': sha(root / 'scope.json'),
                'summary_sha256': sha(root / 'summary.json'),
                'files': {name: {'bytes': (root / name).stat().st_size, 'sha256': sha(root / name)} for name in names}}
    write_json(root / 'control-manifest.json', manifest)
    material = {'schema': 'fr5-stream-materialization-v1', 'session': manifest['session'],
                'control_verification': 'verified', 'target_alignment': 'verified_against_wire_and_frame_clocks',
                'scope_sha256': manifest['scope_sha256'], 'summary_sha256': manifest['summary_sha256'],
                'frames': len(captures), 'source_stream_sha256': hashlib.sha256(f'stream-{trial}'.encode()).hexdigest(),
                'capture_journal_sha256': sha(root / 'continuous/capture.jsonl'),
                'teacher_journal_sha256': sha(root / 'lerobot/meta/capture.jsonl'),
                'original_local_episode': str(root.resolve())}
    write_json(root / 'stream-materialization.json', material)
    if finish == 'released':
        record_review(root, verdict='success', user_statement='Synthetic test, whole task succeeded')
    else:
        write_json(root / 'user-review.json', {'schema': 'fr5-user-review-v1', 'reviewer': 'user',
            'source': 'direct_user_message', 'verdict': 'success', 'user_statement': 'Synthetic test only',
            'scope_sha256': sha(root / 'scope.json'), 'summary_sha256': sha(root / 'summary.json')})
    return root
