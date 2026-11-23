"""Read reviewed whole-task stream materializations without contacting devices.

Collection contract: scope task_instruction/whole_task/task_spec; completed
summary and user-review.json; summary.hold_records lists {repetition, path}
for recorded holds, each with start_monotonic, end_monotonic and
actual_duration_s; ordinary command records and the materialized journals.

One decision sample is the latest *causally prior* observation after the
previous command completed, paired with the next actual issued command. A
latched target is not counted as a new command. Source times remain authoritative;
LeRobot's nominal video frame times are only an index clock for these samples.
The initial hold is an explicit non-RPC target latch, individually identified
until each component has been replaced by an actual issued wire command.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .curate_teacher import validated_commands, validate_journal
from .feedback import gripper_feedback_valid
from .human_review import resolve_review_evidence
from lerobot_camera_gemini.timing import MONOTONIC, camera_clock, command_clock, completion_clock

MAX_FRAME_GAP_S = .30
TASK_ORIGIN = 'initial_hold.issued_monotonic'
COLLECTION_CONTRACT = 'fr5-whole-task-collection-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def source_provenance(scope, captures):
    """Classify declared source metadata; this is not proof of hardware identity."""
    flag_names = ('synthetic', 'is_synthetic', 'simulated', 'is_simulated', 'mock', 'fake')
    def marked(value):
        return (value is True or (type(value) is int and value == 1)
                or (isinstance(value, str) and value.strip().lower() in ('true', 'yes', '1')))
    scope_markers = {key: scope[key] for key in flag_names if marked(scope.get(key))}
    if 'synthetic_id' in scope:
        scope_markers['synthetic_id'] = scope['synthetic_id']
    camera_markers, serials = set(), set()
    for frame in captures:
        camera = frame.get('camera', {})
        if not isinstance(camera, dict):
            raise ValueError('Invalid camera metadata for source provenance')
        serial = camera.get('source_camera_serial')
        if isinstance(serial, str) and serial:
            serials.add(serial)
            if serial.strip().upper().startswith(('SYNTHETIC', 'SIMULATED', 'SIMULATION', 'SIMULATOR', 'MOCK', 'FAKE')):
                camera_markers.add('source_camera_serial=' + serial)
        for key in flag_names:
            if marked(camera.get(key)):
                camera_markers.add(key + '=' + str(camera[key]))
        if 'synthetic_id' in camera:
            camera_markers.add('synthetic_id=' + str(camera['synthetic_id']))
    synthetic = bool(scope_markers or camera_markers)
    if not synthetic and scope.get('contract') != COLLECTION_CONTRACT:
        raise ValueError('Unmarked source does not declare the whole-task capture contract')
    return {'training_data_kind': 'synthetic_diagnostic' if synthetic else 'captured',
            'scope_contract': scope.get('contract'), 'scope_markers': scope_markers,
            'camera_markers': sorted(camera_markers), 'camera_serials': sorted(serials),
            'classification_basis': 'declared scope and validated camera metadata; not hardware attestation'}


def contained(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Source path is not relative to the episode')
    resolved = (root / path).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError('Source file is missing or escapes the episode')
    return resolved


def _verify_control(root, manifest, name):
    item = manifest.get('files', {}).get(name)
    path = contained(root, name)
    if (not isinstance(item, dict) or item.get('bytes') != path.stat().st_size
            or item.get('sha256') != sha(path)):
        raise ValueError(f'Missing or changed control evidence: {name}')
    return path


def task_specification(scope):
    """Validate typed intent; hold_repetitions uses one-based trial indices."""
    instruction, spec = scope.get('task_instruction'), scope.get('task_spec')
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(spec, dict):
        raise ValueError('Every episode requires a nonempty task instruction and typed task_spec')
    duration = number(spec.get('hold_seconds'), 'task_spec hold_seconds')
    repetitions, finish = spec.get('repetitions'), spec.get('finish')
    if duration < 0 or type(repetitions) is not int or repetitions not in (1, 2) or finish not in ('held', 'released'):
        raise ValueError('Invalid task_spec hold_seconds, repetitions or finish')
    default_holds = (list(range(1, repetitions + 1)) if duration > 0 else [])
    if duration > 0 and repetitions > 1 and finish == 'held':
        default_holds = list(range(1, repetitions))
    holds = spec.get('hold_repetitions', default_holds)
    if (not isinstance(holds, list) or any(type(i) is not int or not 1 <= i <= repetitions for i in holds)
            or sorted(set(holds)) != holds or (duration == 0 and holds) or (duration > 0 and not holds)):
        raise ValueError('Invalid hold_repetitions for the task specification')
    return {'hold_seconds': duration, 'repetitions': repetitions, 'finish': finish, 'hold_repetitions': holds}


def _verify_review(root, scope, summary, materialized):
    if (summary.get('outcome') != 'pending_review' or summary.get('failed')
            or not summary.get('evidence') or not summary.get('continuous_capture', {}).get('capture_complete')
            or not scope.get('phase_order') or summary.get('completed_stages') != scope.get('phase_order')
            or summary.get('scope_sha256') != sha(root / 'scope.json')):
        raise ValueError('Episode is incomplete or not awaiting whole-task user review')
    resolve_review_evidence(root, summary, original_episode=materialized.get('original_local_episode'))
    review_path = root / 'user-review.json'
    review = read_json(review_path)
    if (review.get('schema') != 'fr5-user-review-v1' or review.get('reviewer') != 'user'
            or review.get('source') != 'direct_user_message' or review.get('verdict') != 'success'
            or not isinstance(review.get('user_statement'), str) or not review['user_statement'].strip()
            or review.get('scope_sha256') != sha(root / 'scope.json')
            or review.get('summary_sha256') != sha(root / 'summary.json')):
        raise ValueError('User success review is missing, rejected or refers to another episode')
    return review, sha(review_path)


def _task_runs(commands, task):
    """Match real command order to each requested pickup/place cycle."""
    grouped = []
    for row in sorted(commands.values(), key=lambda x: command_clock(x['command'], MONOTONIC)):
        phase = 'lift_full' if row.get('phase') == 'lift' else row.get('phase')
        if phase not in ('close', 'lift_full', 'return', 'release'):
            continue
        if not grouped or grouped[-1][0] != phase:
            grouped.append((phase, []))
        grouped[-1][1].append(row)
    expected = []
    for rep in range(1, task['repetitions'] + 1):
        expected += ['close', 'lift_full']
        if rep < task['repetitions'] or task['finish'] == 'released':
            expected += ['return', 'release']
    if [phase for phase, _ in grouped] != expected:
        raise ValueError('Actual close/lift/return/release command order differs from task_spec')
    cycles, position = [], 0
    for rep in range(1, task['repetitions'] + 1):
        close, lift = grouped[position][1], grouped[position + 1][1]; position += 2
        returning = release = []
        if rep < task['repetitions'] or task['finish'] == 'released':
            returning, release = grouped[position][1], grouped[position + 1][1]; position += 2
        if (any(r['method'] != 'MoveGripper' for r in [*close, *release])
                or any(r['method'] != 'MoveL' for r in [*lift, *returning])):
            raise ValueError('Task stages do not match actual gripper/arm command types')
        cycles.append({'lift': lift, 'return': returning})
    return cycles


def _holds(root, manifest, summary, commands, frames, task):
    cycles = _task_runs(commands, task)
    entries = summary.get('hold_records')
    if entries is None and task['hold_repetitions']:
        # Current collector names per-cycle records deterministically. They
        # remain bound to the final control manifest just like explicit entries.
        names = [f'hold-{rep:02d}.json' for rep in task['hold_repetitions']]
        if all(name in manifest.get('files', {}) for name in names):
            entries = [{'repetition': rep, 'path': name} for rep, name in zip(task['hold_repetitions'], names)]
        elif len(task['hold_repetitions']) == 1 and (root / 'hold-5s.json').is_file():
            entries = [{'repetition': task['hold_repetitions'][0], 'path': 'hold-5s.json'}]
    if entries is None and not task['hold_repetitions']:
        entries = []
    if not isinstance(entries, list) or any(not isinstance(x, dict) for x in entries):
        raise ValueError('summary.hold_records must list each recorded hold and its repetition')
    by_rep = {item.get('repetition'): item.get('path') for item in entries}
    if len(by_rep) != len(entries) or sorted(by_rep) != task['hold_repetitions']:
        raise ValueError('Hold evidence count/repetitions differ from task_spec')
    events = read_jsonl(root / 'continuous/events.jsonl')
    times = [camera_clock(row['camera'])[0] for row in frames]
    holds, used_markers = [], set()
    for rep in task['hold_repetitions']:
        path = _verify_control(root, manifest, by_rep[rep])
        data = read_json(path)
        start, end = number(data.get('start_monotonic'), 'hold start'), number(data.get('end_monotonic'), 'hold end')
        duration = number(data.get('actual_duration_s'), 'hold duration')
        target = number(data.get('requested_seconds'), 'recorded requested seconds')
        if (data.get('repetition', data.get('cycle', rep)) != rep or abs(duration - (end - start)) > .05
                or target != task['hold_seconds'] or duration < target or end - start < target):
            raise ValueError('Recorded hold duration does not match the requested task duration')
        cycle = cycles[rep - 1]
        lift = cycle['lift'][-1]
        returning = cycle['return'][0] if cycle['return'] else None
        return_time = command_clock(returning['command'], MONOTONIC) if returning is not None else None
        lift_completed = completion_clock(lift, MONOTONIC)
        if lift_completed > start or (return_time is not None and return_time < end):
            raise ValueError('Hold is not between the corresponding completed lift and return')
        # Current collector records through the actual return wire issue.
        # Older records can end at the timer; include their remaining return
        # latency in the derived hover measurement without rewriting sources.
        actual_hover = (return_time if return_time is not None else end) - lift_completed
        deviation = actual_hover - target
        if 'return_command_issued' in data:
            expected_return = return_time is not None
            if (data['return_command_issued'] is not expected_return or
                    (expected_return and not math.isclose(end, return_time, rel_tol=0., abs_tol=1e-6))):
                raise ValueError('Recorded hold end does not match the actual return command')
        for field, expected in (('lift_completed_monotonic', lift_completed),
                                ('actual_hover_from_lift_completion_s', actual_hover),
                                ('timing_deviation_seconds', deviation)):
            if field in data and not math.isclose(number(data[field], field), expected, rel_tol=0., abs_tol=1e-6):
                raise ValueError(f'Recorded hold {field} differs from actual command clocks')
        marker_names = {'hold', 'hold_' + format(target, 'g') + 's'}
        markers = [(i, e) for i, e in enumerate(events) if e.get('kind') == 'phase' and e.get('name') in marker_names
                   and completion_clock(lift, MONOTONIC) <= number(e.get('time_monotonic'), 'hold event time') <= start]
        if len(markers) != 1 or markers[0][0] in used_markers:
            raise ValueError('Hold event does not identify this lift-to-hold transition')
        used_markers.add(markers[0][0])
        left, right = bisect.bisect_right(times, start) - 1, bisect.bisect_left(times, end)
        if left < 0 or right >= len(times) or any(b - a > MAX_FRAME_GAP_S for a, b in zip(times[left:right], times[left + 1:right + 1])):
            raise ValueError('The hold is not covered by continuous recorded frames')
        holds.append({'repetition': rep, 'path': by_rep[rep], 'requested_seconds': target,
                      'start_monotonic': start, 'end_monotonic': end, 'actual_duration_s': duration,
                      'actual_hover_from_lift_completion_s': actual_hover,
                      'timing_deviation_seconds': deviation,
                      'start_source_frame': frames[left]['frame_index'], 'end_source_frame': frames[right]['frame_index'],
                      'lift_command_sequence': lift['command']['sequence'],
                      'return_command_sequence': returning['command']['sequence'] if returning else None,
                      'sha256': sha(path)})
    return holds


def load_source(path):
    """Validate one reviewed, single-session materialized recording in full."""
    root = Path(path).resolve()
    if (root / 'INCOMPLETE').exists() or (root / 'composition-manifest.json').exists():
        raise ValueError('Incomplete or composite recordings cannot be a continuous whole-task source')
    scope, summary = read_json(root / 'scope.json'), read_json(root / 'summary.json')
    task = task_specification(scope)
    if (scope.get('whole_task') is not True
            or summary.get('task_instruction') != scope['task_instruction'] or summary.get('whole_task') is not True
            or ('task_spec' in summary and summary['task_spec'] != scope['task_spec'])
            or scope.get('split_role') not in ('training_complete_trial', 'held_out_complete_trial')):
        raise ValueError('Scope and summary must agree on the whole task and complete-trial split')
    materialized = read_json(root / 'stream-materialization.json')
    manifest = read_json(root / 'control-manifest.json')
    if (materialized.get('schema') != 'fr5-stream-materialization-v1'
            or materialized.get('control_verification') != 'verified'
            or materialized.get('target_alignment') != 'verified_against_wire_and_frame_clocks'
            or manifest.get('session') != materialized.get('session')
            or manifest.get('scope_sha256') != sha(root / 'scope.json')
            or manifest.get('summary_sha256') != sha(root / 'summary.json')
            or materialized.get('scope_sha256') != manifest.get('scope_sha256')
            or materialized.get('summary_sha256') != manifest.get('summary_sha256')):
        raise ValueError('Materialization is incomplete or source identity changed')
    for name in ('scope.json', 'summary.json', 'target-command-journal.json', 'continuous/events.jsonl'):
        _verify_control(root, manifest, name)
    review, review_sha = _verify_review(root, scope, summary, materialized)
    if review['scope_sha256'] != sha(root / 'scope.json'):
        raise ValueError('Review is not tied to the whole-task scope')
    journal = read_json(root / 'target-command-journal.json')
    commands = validated_commands(root)
    validate_journal(journal, commands, MONOTONIC)
    actual = [entry for entry in journal if entry.get('executed') is True]
    if set(commands) != {entry['sequence'] for entry in actual}:
        raise ValueError('Missing, failed or uncertain actual command')
    for entry in actual:
        _verify_control(root, manifest, f"command-{entry['sequence']:04d}.json")
    captures_path = root / 'continuous/capture.jsonl'
    teachers_path = root / 'lerobot/meta/capture.jsonl'
    if (sha(captures_path) != materialized.get('capture_journal_sha256')
            or sha(teachers_path) != materialized.get('teacher_journal_sha256')):
        raise ValueError('Materialized capture journal changed')
    captures, teachers = read_jsonl(captures_path), read_jsonl(teachers_path)
    media = read_jsonl(root / 'stream-frames.jsonl')
    if not captures or len(captures) != len(teachers) or len(captures) != len(media) or len(captures) != materialized['frames']:
        raise ValueError('Capture, action and media frame counts differ')
    times = []
    for index, (capture, teacher, medium) in enumerate(zip(captures, teachers, media)):
        timestamp, basis = camera_clock(capture['camera'])
        robot = capture['robot']
        robot_time = number(robot.get('received_monotonic'), 'robot observation time')
        skew = number(capture.get('sync_skew_s'), 'observation skew')
        if (basis != MONOTONIC or capture.get('sync_clock') != MONOTONIC
                or capture.get('frame_index') != index or teacher.get('frame_index') != index
                or medium.get('sequence') != index or medium.get('camera') != capture['camera']
                or any(teacher.get(key) != capture.get(key) for key in ('camera', 'robot', 'sync_clock', 'sync_skew_s'))
                or abs(robot_time - timestamp - skew) > 1e-6 or (times and timestamp <= times[-1])):
            raise ValueError('Capture clocks or synchronized state are inconsistent')
        if capture.get('composition_source') is not None or teacher.get('composition_source') is not None:
            raise ValueError('Source contains remapped recording segments')
        target_index = bisect.bisect_right([command_clock(c, MONOTONIC) for c in journal], timestamp) - 1
        if target_index < 0 or teacher.get('command') != journal[target_index]:
            raise ValueError('Captured latched target disagrees with command journal')
        rgb_name = capture.get('rgb_file')
        rgb = contained(root, 'continuous/' + str(rgb_name))
        if medium.get('rgb_file') != rgb_name or sha(rgb) != medium.get('rgb_jpeg_sha256'):
            raise ValueError('Captured JPEG identity changed')
        times.append(timestamp)
    provenance = source_provenance(scope, captures)
    holds = _holds(root, manifest, summary, commands, captures, task)
    if not actual or times[-1] < completion_clock(commands[actual[-1]['sequence']], MONOTONIC):
        raise ValueError('Capture ends before the final physical command completed')
    decisions = []
    # The recorded initial hold is a real declared latch, not an RPC. Retain
    # it for either component until that component receives an actual command.
    initial_hold = journal[0]
    origin = command_clock(initial_hold, MONOTONIC)
    if times[0] < origin:
        raise ValueError('Recording starts before the initial hold time origin')
    holds_by_lift = {hold['lift_command_sequence']: hold for hold in holds}
    if len(holds_by_lift) != len(holds):
        raise ValueError('Multiple holds cannot be assigned to the same lift command')
    last_kind = {'arm': initial_hold, 'gripper': initial_hold}
    previous = None
    previous_command = initial_hold
    previous_hold = None
    for command in actual:
        row = commands[command['sequence']]
        kind = command.get('kind')
        if kind not in ('arm', 'gripper'):
            raise ValueError('Unsupported command kind')
        last_kind[kind] = command
        issued = command_clock(command, MONOTONIC)
        completed = completion_clock(row, MONOTONIC)
        if previous is not None and issued < completion_clock(previous, MONOTONIC):
            raise ValueError('Overlapping commands cannot be represented by the sequential policy executor')
        if len(last_kind) == 2:
            lower = completion_clock(previous, MONOTONIC) if previous is not None else origin
            if previous_hold is not None:
                # A decision after a timed hold must observe only after the
                # requested dwell has actually elapsed. The wire return can be
                # later; its extra jitter is source evidence, not the target.
                lower = max(lower, previous_hold['start_monotonic'] + previous_hold['requested_seconds'])
            index = bisect.bisect_right(times, issued) - 1
            while index >= 0 and (not lower <= captures[index]['robot']['received_monotonic'] <= issued
                                  or times[index] < lower):
                index -= 1
            if index < 0 or times[index] < lower:
                raise ValueError('No causal camera/state observation after the previous completed command and requested dwell')
            image, state = captures[index], captures[index]['robot']
            valid = gripper_feedback_valid(state)
            measured = [*state['joints'], state['gripper_position'] if valid else 0., float(valid)]
            target = command['target']
            if (not np.array_equal(np.asarray(target[:6]), np.asarray(last_kind['arm']['target'][:6]))
                    or target[6] != last_kind['gripper']['target'][6]):
                raise ValueError('A latched target component differs from its initial hold or actual wire command')
            decisions.append({'source_frame': index, 'camera_time_monotonic': times[index],
                              'robot_time_monotonic': float(state['received_monotonic']),
                              'command_sequence': command['sequence'], 'command_kind': kind,
                              'event_kind': float(kind == 'gripper'),
                              'command_issued_monotonic': issued, 'command_completed_monotonic': completed,
                              'arm_command_sequence': last_kind['arm']['sequence'],
                              'gripper_command_sequence': last_kind['gripper']['sequence'],
                              'arm_target_source': 'wire' if last_kind['arm']['executed'] else 'initial_hold',
                              'gripper_target_source': 'wire' if last_kind['gripper']['executed'] else 'initial_hold',
                              'command_observation_before': row.get('observation_before_command'),
                              'command_observation_matches_selected_frame':
                                  row.get('observation_before_command') == image['camera'],
                              'phase': row.get('phase'), 'rgb_file': image['rgb_file'],
                              'state_degrees_percent': measured, 'target_degrees_percent': target,
                              'previous_target_degrees_percent': previous_command['target'],
                              'previous_target_command_sequence': previous_command['sequence'],
                              'previous_target_source': 'wire' if previous_command['executed'] else 'initial_hold',
                              'completed_event_count': len(decisions),
                              'previous_dwell_seconds': previous_hold['requested_seconds'] if previous_hold else 0.,
                              'previous_dwell_source': previous_hold['path'] if previous_hold else None,
                              'task_origin_monotonic': origin,
                              'elapsed_task_seconds': times[index] - origin,
                              'dwell_seconds': holds_by_lift[command['sequence']]['requested_seconds']
                                  if command['sequence'] in holds_by_lift else 0.,
                              'dwell_source': holds_by_lift[command['sequence']]['path']
                                  if command['sequence'] in holds_by_lift else None,
                              'dwell_actual_seconds': holds_by_lift[command['sequence']]['actual_duration_s']
                                  if command['sequence'] in holds_by_lift else None,
                              'task_complete': float(command['sequence'] == actual[-1]['sequence'])})
        previous = row
        previous_command = command
        previous_hold = holds_by_lift.get(command['sequence'])
    if not decisions or decisions[0]['command_sequence'] != actual[0]['sequence']:
        raise ValueError('The first actual command is absent from the decision series')
    # Never train across a recording gap. A missing frame may split an otherwise
    # accepted physical trial into separate policy decision sequences.
    segments = []
    start = 0
    for i in range(1, len(decisions)):
        a, b = decisions[i-1]['source_frame'], decisions[i]['source_frame']
        if any(times[j+1] - times[j] > MAX_FRAME_GAP_S for j in range(a, b)):
            segments.append(decisions[start:i]); start = i
    segments.append(decisions[start:])
    return {'root': root, 'scope': scope, 'summary': summary, 'materialization': materialized,
            'review': review, 'review_sha256': review_sha, 'holds': holds,
            'data_provenance': provenance,
            'initial_hold': initial_hold, 'task_origin_monotonic': origin, 'task_origin_basis': TASK_ORIGIN,
            'completion_evidence': {'final_command_sequence': actual[-1]['sequence'],
                'final_command_completed_monotonic': completion_clock(commands[actual[-1]['sequence']], MONOTONIC),
                'summary_outcome': summary['outcome'], 'summary_completed_stages': summary['completed_stages'],
                'user_verdict': review['verdict'], 'summary_sha256': sha(root / 'summary.json'),
                'review_sha256': review_sha},
            'task_instruction': scope['task_instruction'], 'task_spec': scope['task_spec'],
            'normalized_task_spec': task,
            'source_scope_sha256': sha(root / 'scope.json'), 'source_summary_sha256': sha(root / 'summary.json'),
            'segments': segments, 'capture_frames': len(captures),
            'initial_joints_deg': captures[decisions[0]['source_frame']]['robot']['joints']}
