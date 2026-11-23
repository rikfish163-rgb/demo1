"""Verify a closed binary spool and build a new conventional episode tree.

Nothing here controls hardware or writes to a pre-existing dataset. The
server must have the exact codec and protocol files recorded for this stream.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sync_file(path):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stream_index(root, entries, kind):
    directory = 'stream-' + kind
    index_name = directory + '/index.json'
    index = json.loads((root / index_name).read_text())
    if (index.get('schema') != 'fr5-evaluation-' + kind + '-index-v1'
            or type(index.get('count')) is not int or index['count'] < 0
            or not isinstance(index.get('chunks'), list)):
        raise ValueError('Invalid stream ' + kind + ' index')
    expected_names, rows = set(), []
    for number, entry in enumerate(index['chunks']):
        name = f'{number:06d}.jsonl'
        if not isinstance(entry, dict) or entry.get('path') != name:
            raise ValueError('Stream journal chunks must be contiguous')
        full_name = directory + '/' + name
        if full_name not in entries:
            raise ValueError('Stream journal chunk is missing from manifest')
        raw = (root / full_name).read_bytes()
        if (entry.get('bytes') != len(raw) or entry.get('sha256') != _sha(raw)
                or entry.get('rows') != raw.count(b'\n') or (raw and not raw.endswith(b'\n'))):
            raise ValueError('Stream journal index hash or row count mismatch')
        expected_names.add(full_name)
        for line in raw.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('callback_sequence') != len(rows):
                raise ValueError('Stream callback journal sequence mismatch')
            rows.append(row)
    present = {name for name in entries if name.startswith(directory + '/')}
    if present != expected_names | {index_name} or len(rows) != index['count']:
        raise ValueError('Stream journal has missing or extra chunks')
    return rows


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_prefix_ack(item, ack, chunk, scope, gripper_wire, steps):
    """Pair native target completion with its consumed prefix and wire command."""
    consumed = item['action_index'] + 1
    stamp, cause = ack.get('acknowledged_monotonic'), ack.get('prefix_retired_cause')
    before, after = ack.get('gripper_commands_before'), ack.get('gripper_commands_after')
    if (ack.get('sequence') != item['sequence']
            or type(item.get('delivery_token')) is not int or item['delivery_token'] != item['sequence'] + 1
            or type(ack.get('delivery_token')) is not int or ack['delivery_token'] != item['delivery_token']
            or not _finite(stamp) or stamp < item['target_monotonic']
            or type(item.get('prefix_consumed_steps')) is not int or item['prefix_consumed_steps'] != consumed
            or type(ack.get('prefix_consumed_steps')) is not int or ack['prefix_consumed_steps'] != consumed
            or 'prefix_retired_cause' not in ack
            or cause not in (None, 'execution_prefix_completed', 'native_gripper_command_completed')
            or type(before) is not int or type(after) is not int or not 0 <= before <= after):
        raise ValueError('Native prefix acknowledgement disagrees with submitted target')
    if ((cause == 'execution_prefix_completed' and consumed != steps)
            or (cause is None and consumed == steps)):
        raise ValueError('Native prefix retirement disagrees with consumed actions')
    joint_profiles = ('none', 'state-history-v1', 'state-history-physical-v1')
    gripper_reobserve = (
        scope.get('observation_rule') == 'prefix_or_completed_native_gripper_command'
        and scope.get('observation_context_profile', 'none') in joint_profiles
        and chunk['request'].get('observation_context_profile', 'none') in joint_profiles
        and scope.get('wire_action_representation') == 'absolute-joint-position-v1'
        and chunk['reply'].get('wire_action_representation', 'absolute-joint-position-v1')
        == 'absolute-joint-position-v1')
    if ((cause == 'native_gripper_command_completed' and not gripper_reobserve)
            or (gripper_reobserve and after > before and cause != 'native_gripper_command_completed')):
        raise ValueError('Native gripper retirement does not match the joint observation rule')
    result = {key: ack[key] for key in (
        'delivery_token', 'acknowledged_monotonic', 'prefix_consumed_steps',
        'prefix_retired_cause', 'gripper_commands_before', 'gripper_commands_after')}
    if cause == 'native_gripper_command_completed':
        submitted = item['submitted_monotonic']
        actual_before = sum(row['issued_monotonic'] < submitted for row in gripper_wire)
        actual_after = sum(row['issued_monotonic'] <= stamp for row in gripper_wire)
        commands = gripper_wire[actual_before:actual_after]
        effective = item.get('effective_gripper_target_percent')
        if (before != actual_before or after != actual_after or len(commands) != 1
                or type(effective) is not int or not 0 <= effective <= 100
                or effective != int(round(item['target'][6] * 100.))
                or commands[0]['target_deg_percent'] != effective
                or commands[0]['return_code'] != 0 or commands[0].get('error')
                or commands[0]['returned_monotonic'] > stamp):
            raise ValueError('Native gripper retirement lacks its completed MoveGripper wire command')
        result.update(effective_gripper_target_percent=effective,
                      gripper_wire_sequence=commands[0]['sequence'])
    return result


def _validate_stream_targets(path, chunks, ordered, prefix_steps=None, *, scope=None, wire=(), predictions=()):
    """Verify submitted model actions and retain their two distinct clocks."""
    timing, previous = [], None
    chunk_order = {request_id: index for index, request_id in enumerate(chunks)}
    if not path.exists():
        return timing
    scope = scope or {}
    acknowledgements = {}
    ack_path = path.with_name('acknowledged-targets.jsonl')
    native = ordered and scope.get('target_delivery') == 'ordered_native_acknowledgements'
    if native and ack_path.exists():
        for line in ack_path.read_text().splitlines():
            ack = json.loads(line)
            if 'prefix_consumed_steps' not in ack and 'prefix_retired_cause' not in ack:
                continue
            sequence = ack.get('sequence')
            if type(sequence) is not int or sequence < 0 or sequence in acknowledgements:
                raise ValueError('Native prefix acknowledgement sequence is invalid')
            acknowledgements[sequence] = ack
    gripper_wire = [row for row in wire if row['method'] == 'MoveGripper']
    native_replies = {row['request']['id']: row['reply'] for row in predictions}
    steps = prefix_steps if prefix_steps is not None else 50
    for n, line in enumerate(path.read_text().splitlines()):
        item = json.loads(line)
        request_id = item.get('request_id')
        chunk = chunks.get(request_id)
        index = item.get('action_index')
        if (item.get('sequence') != n or chunk is None or type(index) is not int
                or not 0 <= index < 50 or item.get('raw_action') != chunk['reply']['actions'][index]
                or not _finite(item.get('submitted_monotonic'))
                or item['submitted_monotonic'] < chunk['request']['source_time']):
            raise ValueError('Submitted target disagrees with prediction chunk')
        if prefix_steps is not None and (index >= prefix_steps
                or type(item.get('execution_prefix_steps')) is not int
                or item['execution_prefix_steps'] != prefix_steps
                or type(item.get('prediction_chunk_steps')) is not int
                or item['prediction_chunk_steps'] != 50):
            raise ValueError('Submitted target disagrees with configured policy prefix')
        raw = item['raw_action']
        joints = raw[:6]
        if chunk['reply'].get('wire_action_representation') == 'cup-relative-tcp-v1':
            ik = item.get('cup_inverse_kinematics', {})
            origin = chunk['request'].get('cup_base_xyz_mm')
            if (not isinstance(origin, list) or len(origin) != 3 or not all(map(_finite, origin))
                    or ik.get('cup_base_xyz_mm') != origin or ik.get('raw_model_action') != raw
                    or ik.get('action_index') != index):
                raise ValueError('Submitted target does not preserve its cup action and origin')
            pose = [x+1000*y for x, y in zip(origin, raw[:3])] + [math.degrees(x) for x in raw[3:6]]
            ik_pose, fk, q = (ik.get(k) for k in ('target_tcp_base_mm_deg', 'fk_tcp_base_mm_deg', 'ik_joint_deg'))
            if (any(not isinstance(v, list) or len(v) != 6 or not all(map(_finite, v)) for v in (ik_pose, fk, q))
                    or max(abs(a-b) for a,b in zip(pose, ik_pose)) > 1e-6
                    or math.dist(fk[:3], pose[:3]) > .5
                    or max(abs((a-b+180)%360-180) for a,b in zip(fk[3:], pose[3:])) > .1):
                raise ValueError('Submitted cup target disagrees with its inverse/forward kinematics')
            joints = [math.radians(x) for x in q]
        expected = joints + [int(round(min(1., max(0., raw[6])) * 100.)) / 100.]
        actual_target = item.get('target_monotonic')
        predicted_target = chunk['request']['source_time'] + (index+1)/6
        got = item.get('target')
        if (not isinstance(got, list) or len(got) != 7 or not all(map(_finite, got))
                or max(abs(a-b) for a,b in zip(got,expected)) > 1e-9 or not _finite(actual_target)):
            raise ValueError('Submitted target differs from decoded policy action')
        if not ordered:
            # Historical TimedChunk recordings use the source time as their
            # target time and must retain their original verification rules.
            if abs(actual_target - predicted_target) > 1e-6:
                raise ValueError('Submitted target timing differs from predicted source time')
            continue
        recorded_prediction_time = item.get('predicted_source_target_monotonic')
        submitted = item['submitted_monotonic']
        received = chunk.get('received_monotonic')
        position = chunk_order[request_id]
        if (not _finite(recorded_prediction_time)
                or abs(recorded_prediction_time-predicted_target) > 1e-6
                or not _finite(received) or received < chunk['request']['source_time']
                or submitted < received or actual_target < submitted):
            raise ValueError('Ordered target predicted or actual timing is inconsistent')
        entry = {'sequence': n, 'request_id': request_id, 'action_index': index,
                 'submitted_monotonic': submitted, 'target_monotonic': actual_target,
                 'predicted_source_target_monotonic': recorded_prediction_time,
                 'target_time_offset_s': actual_target-recorded_prediction_time}
        if 'prefix_consumed_steps' in item:
            if type(item['prefix_consumed_steps']) is not int or item['prefix_consumed_steps'] != index + 1:
                raise ValueError('Submitted target consumed prefix disagrees with action index')
            entry['prefix_consumed_steps'] = item['prefix_consumed_steps']
        ack = acknowledgements.pop(n, None)
        if ack is not None:
            entry.update(_validate_prefix_ack(item, ack, chunk, scope, gripper_wire, steps))
            if (entry['prefix_retired_cause'] == 'native_gripper_command_completed'
                    and native_replies.get(request_id) != chunk['reply']):
                raise ValueError('Retired prefix differs from its native 50-action prediction')
        if previous is None:
            if position != 0 or index != 0:
                raise ValueError('Ordered action sequence must start at the first chunk prefix')
        else:
            if (submitted <= previous['submitted_monotonic']
                    or actual_target <= previous['target_monotonic']):
                raise ValueError('Ordered action submission times must increase')
            prior_ack = previous.get('acknowledged_monotonic')
            if native and 'prefix_consumed_steps' in previous:
                if prior_ack is None or submitted <= prior_ack:
                    raise ValueError('Ordered target lacks its prior native acknowledgement')
            if request_id == previous['request_id']:
                if index != previous['action_index'] + 1 or previous.get('prefix_retired_cause') is not None:
                    raise ValueError('Ordered action sequence skipped or repeated a chunk action')
            else:
                if ((previous['action_index'] != steps - 1
                        and previous.get('prefix_retired_cause') != 'native_gripper_command_completed')
                        or index != 0 or position != chunk_order[previous['request_id']] + 1):
                    raise ValueError('Ordered action sequence changed chunks before completing the prior chunk')
        timing.append(entry)
        previous = entry
    if acknowledgements:
        raise ValueError('Native prefix acknowledgement has no submitted target')
    return timing


def _stream_controls(root, receipt, scope, summary, entries):
    """Validate evaluation records independently from teacher MoveL rules."""
    metadata = scope.get('metadata')
    readonly = (scope.get('model_execution') is False
                or isinstance(metadata, dict) and metadata.get('model_execution') is False)
    if (scope.get('dataset_role') != 'policy_evaluation'
            or scope.get('policy_contract', {}).get('name') != 'fr5-joint-position-stream-v1'
            or summary.get('schema') != 'fr5-stream-policy-evaluation-summary-v1'
            or summary.get('dataset_role') != 'policy_evaluation'
            or summary.get('task_instruction') != scope.get('task_instruction')
            or summary.get('checkpoint_sha256') != scope.get('checkpoint_sha256')
            or summary.get('task_success_verified') is not False
            or summary.get('training_ready') is not False
            or summary.get('awaiting_whole_user_review') is not (not readonly)
            or summary.get('read_only_diagnostic') is not readonly
            or summary.get('model_execution') is not (not readonly)
            or summary.get('storage_status') != 'server_stream_closed'
            or summary.get('summary_stage') != 'final_control_snapshot'
            or summary.get('termination_source') != summary.get('reason')
            or summary.get('failed') != summary.get('error')
            or summary.get('recorded_frames') != receipt['accepted_frames']
            or summary.get('continuous_capture', {}).get('frames') != receipt['accepted_frames']):
        raise ValueError('Invalid stream evaluation scope or final summary')
    wire = _stream_index(root, entries, 'wire')
    predictions = _stream_index(root, entries, 'prediction')
    if len(wire) != summary.get('wire_records') or len(predictions) != summary.get('prediction_records'):
        raise ValueError('Evaluation journal count disagrees with summary')
    sequences, counts = set(), {'ServoJ': 0, 'MoveGripper': 0}
    for row in wire:
        method, args, target = row.get('method'), row.get('args'), row.get('target_deg_percent')
        issued, returned = row.get('issued_monotonic'), row.get('returned_monotonic')
        code, sequence = row.get('return_code'), row.get('sequence')
        if (method not in ('ServoMoveStart', 'ServoJ', 'MoveGripper', 'StopMotion', 'ServoMoveEnd')
                or not isinstance(args, list) or row.get('raw_args') != args
                or type(sequence) is not int or sequence < 0 or sequence in sequences
                or not _finite(issued) or issued <= 0 or not _finite(returned) or returned < issued
                or not _finite(row.get('issued_unix')) or row['issued_unix'] <= 0
                or not _finite(row.get('returned_unix')) or row['returned_unix'] <= 0
                or not _finite(row.get('callback_monotonic')) or row['callback_monotonic'] < returned
                or (code is not None and type(code) is not int)
                or (code is None and not row.get('error'))):
            raise ValueError('Invalid stream native wire event')
        sequences.add(sequence)
        if method == 'ServoJ':
            if (len(args) != 8 or not isinstance(target, list) or len(target) != 6
                    or any(not _finite(x) for x in target) or args[0] != target
                    or not isinstance(args[1], list) or len(args[1]) != 4
                    or any(not _finite(x) for x in args[1] + args[2:])):
                raise ValueError('ServoJ target disagrees with native wire arguments')
        elif method == 'MoveGripper':
            if (len(args) != 10 or type(target) is not int or not 0 <= target <= 100
                    or args[1] != target or any(not _finite(x) for x in args)):
                raise ValueError('MoveGripper target disagrees with native wire arguments')
        elif args:
            raise ValueError('Invalid native lifecycle command arguments')
        if method in counts:
            counts[method] += 1
    wire.sort(key=lambda item: (item['issued_monotonic'], item['sequence']))
    if any(a['sequence'] >= b['sequence'] for a, b in zip(wire, wire[1:])):
        raise ValueError('Native command sequence disagrees with issued monotonic time')
    if (summary.get('wire_method_counts') != counts or summary.get('commands') != sum(counts.values())
            or (not readonly and type(summary.get('wire_callbacks_drained')) is not bool)):
        raise ValueError('Native command counts disagree with final summary')
    if readonly and (wire or predictions or summary.get('wire_callbacks_drained') is not None
            or summary.get('controller_snapshot') is not None
            or summary.get('controller_recording') != 'not_applicable_read_only'
            or summary.get('outcome') not in ('observed', 'interrupted')):
        raise ValueError('Read-only diagnostic contains a controller, command or prediction')
    if summary['wire_callbacks_drained']:
        snap = summary.get('controller_snapshot')
        if (not isinstance(snap, dict) or snap.get('servo_commands') != counts['ServoJ']
                or snap.get('gripper_commands') != counts['MoveGripper']
                or snap.get('pending_callbacks') != 0 or snap.get('inflight_commands', 0) != 0
                or snap.get('startup_inflight') or snap.get('undelivered_callbacks')
                or snap.get('callback_failure') or snap.get('workers_alive')):
            raise ValueError('Claimed drained callbacks disagree with controller snapshot')

    chunks = {}
    names = sorted(name for name in entries if name.startswith('chunk-') and name.endswith('.json'))
    if readonly and (names or 'submitted-targets.jsonl' in entries):
        raise ValueError('Read-only diagnostic contains policy actions or predictions')
    for n, name in enumerate(names):
        if name != f'chunk-{n:05d}.json':
            raise ValueError('Prediction chunk files are not contiguous')
        item = json.loads((root / name).read_text())
        request, reply = item.get('request'), item.get('reply')
        _validate_prediction(request, reply, scope)
        if request['id'] in chunks:
            raise ValueError('Duplicate prediction request id')
        chunks[request['id']] = item
    for item in predictions:
        _validate_prediction(item.get('request'), item.get('reply'), scope)
        media = item['request'].get('rgb_jpeg')
        encoded = item['request'].get('rgb_jpeg_base64')
        if (not isinstance(media, dict) or type(media.get('bytes')) is not int or media['bytes'] <= 0
                or not isinstance(media.get('sha256'), str) or len(media['sha256']) != 64
                or media.get('storage') != 'exact_base64_in_this_request' or not isinstance(encoded, str)):
            raise ValueError('Prediction input JPEG provenance is missing')
        try:
            raw_jpeg = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('Prediction input JPEG encoding is invalid') from exc
        if len(raw_jpeg) != media['bytes'] or _sha(raw_jpeg) != media['sha256']:
            raise ValueError('Prediction input JPEG bytes or hash mismatch')
    mode = scope.get('execution_mode')
    ordered = mode in ('ordered_complete_chunk', 'ordered_policy_prefix')
    prefix_steps = None
    if mode == 'ordered_complete_chunk' and (scope.get('execution_chunk_steps', 50) != 50 or scope.get('prefix_skip', False) is not False):
        raise ValueError('Ordered policy scope does not match complete 50-action chunks')
    if mode == 'ordered_policy_prefix':
        prefix_steps = scope.get('execution_chunk_steps')
        if (type(prefix_steps) is not int or not 1 <= prefix_steps < 50
                or type(scope.get('prediction_chunk_steps')) is not int
                or scope['prediction_chunk_steps'] != 50 or scope.get('prefix_skip') is not False):
            raise ValueError('Ordered policy scope does not match configured prediction prefix')
        if any(type(chunk.get('execution_prefix_steps')) is not int
               or chunk['execution_prefix_steps'] != prefix_steps for chunk in chunks.values()):
            raise ValueError('Prediction chunk disagrees with configured policy prefix')
    timing = _validate_stream_targets(root / 'submitted-targets.jsonl', chunks, ordered, prefix_steps,
                                     scope=scope, wire=wire, predictions=predictions)
    retired = {row['request_id']: row for row in timing
               if row.get('prefix_retired_cause') is not None}
    request_ids = list(chunks)
    retirements = {current: retired[previous] for previous, current in zip(request_ids, request_ids[1:])
                   if previous in retired}
    for request_id, previous in retirements.items():
        chunk = chunks[request_id]
        stamp = previous['acknowledged_monotonic']
        if chunk['request']['source_time'] <= stamp or chunk.get('observation_after_monotonic') != stamp:
            raise ValueError('Next prediction observation does not follow the prior native acknowledgement')
    return {'kind': 'stream', 'events': wire, 'chunks': chunks, 'predictions': predictions,
            'callbacks_drained': summary['wire_callbacks_drained'], 'read_only_diagnostic': readonly,
            'execution_mode': scope.get('execution_mode'), 'ordered_timing': timing,
            'prefix_retirements': {request_id: previous for request_id, previous in retirements.items()
                                   if previous['prefix_retired_cause'] == 'native_gripper_command_completed'}}


def _validate_prediction(request, reply, scope):
    actions = reply.get('actions') if isinstance(reply, dict) else None
    if (not isinstance(request, dict) or not isinstance(reply, dict)
            or request.get('schema') != 'fr5-stream-request-v1'
            or request.get('policy_contract') != 'fr5-joint-position-stream-v1'
            or reply.get('schema') != 'fr5-stream-reply-v1'
            or not isinstance(request.get('id'), str) or not request['id']
            or request.get('id') != reply.get('id')
            or request.get('source_clock') != 'host_monotonic'
            or not _finite(request.get('source_time')) or request['source_time'] <= 0
            or request.get('task') != scope.get('task_instruction')
            or reply.get('checkpoint_sha256') != scope.get('checkpoint_sha256')
            or reply.get('policy_contract') != 'fr5-joint-position-stream-v1'
            or not isinstance(actions, list) or len(actions) != 50
            or any(not isinstance(action, list) or len(action) != 7
                   or any(not _finite(value) for value in action) for action in actions)):
        raise ValueError('Prediction request or reply provenance is invalid')
    if scope.get('observation_context_profile') in ('state-history-physical-initial-cup-v1',
                                                    'state-history-physical-cup-tcp-v1'):
        cup = scope.get('initial_cup', {})
        uv = cup.get('uv')
        stamp = cup.get('observed_monotonic')
        if (not isinstance(uv, list) or len(uv) != 2
                or any(not _finite(x) or not 0 <= x < bound for x, bound in zip(uv, (1920, 1080)))
                or not _finite(stamp) or not 0 < stamp <= request['source_time']
                or cup.get('camera_serial') != 'CP1E54200015' or request.get('initial_cup_uv') != uv):
            raise ValueError('Prediction initial cup observation differs from episode scope')
        if scope.get('observation_context_profile') == 'state-history-physical-cup-tcp-v1':
            base = cup.get('base_xyz_mm')
            if (not isinstance(base, list) or len(base) != 3 or not all(map(_finite, base))
                    or request.get('cup_base_xyz_mm') != base
                    or reply.get('wire_action_representation') != 'cup-relative-tcp-v1'):
                raise ValueError('Prediction cup-relative wire and base frame differ from scope')


def _control_snapshot(root, receipt):
    """Verify the frozen small-file tree before publishing materialization."""
    scope_path = root / "scope.json"
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    manifest_path = root / "control-manifest.json"
    if not manifest_path.exists():
        if scope.get("diagnostic_only") is True:
            return None, scope, None, None
        raise ValueError("Final control/evidence manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("files")
    if (manifest.get("schema") != "fr5-final-control-manifest-v1"
            or manifest.get("session") != receipt.get("session") or not isinstance(entries, dict)
            or manifest.get("local_episode") != receipt.get("local_episode")):
        raise ValueError("Invalid final control manifest")
    for name, detail in entries.items():
        relative = Path(name)
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or not isinstance(detail, dict) or type(detail.get("bytes")) is not int
                or not 0 <= detail["bytes"] <= 16 * 1024 * 1024):
            raise ValueError("Invalid control manifest path or size")
        path = root / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != detail["bytes"] or _sha(path.read_bytes()) != detail.get("sha256")):
            raise ValueError("Server control/evidence hash mismatch: " + name)
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in {"wire", "materialized"} or relative.parts[0].startswith(".materializing-"):
            continue
        if path.is_symlink():
            raise ValueError("Server control/evidence contains a symlink")
        if path.is_file() and relative.as_posix() not in {"frames.stream", "INCOMPLETE", "control-manifest.json"}:
            actual.add(relative.as_posix())
    if actual != set(entries):
        raise ValueError("Server control/evidence tree differs from final manifest")
    is_stream = (scope.get('dataset_role') == 'policy_evaluation'
                 and isinstance(scope.get('policy_contract'), dict)
                 and scope['policy_contract'].get('name') == 'fr5-joint-position-stream-v1')
    required = ({'scope.json', 'summary.json', 'stream-closed.json', 'SESSION_STARTED',
                 'stream-wire/index.json', 'stream-prediction/index.json',
                 'continuous/report.json', 'continuous/events.jsonl'} if is_stream else
                {"scope.json", "summary.json", "stream-closed.json", "target-command-journal.json",
                 "continuous/report.json", "continuous/events.jsonl", "lerobot/meta/demo.json"})
    if not required.issubset(entries):
        raise ValueError("Final control/evidence tree is incomplete")
    scope_sha = entries["scope.json"]["sha256"]
    summary = json.loads((root / "summary.json").read_text())
    if (manifest.get("scope_sha256") != scope_sha or summary.get("scope_sha256") != scope_sha
            or manifest.get("summary_sha256") != entries["summary.json"]["sha256"]):
        raise ValueError("Final scope or summary hash mismatch")
    if is_stream:
        return manifest, scope, summary, _stream_controls(root, receipt, scope, summary, entries)
    if "completed_stages" in summary:
        stages, declared = summary["completed_stages"], scope.get("phase_order")
        offset = summary.get("start_offset_mm")
        if (not isinstance(stages, list) or not isinstance(declared, list)
                or stages != declared[:len(stages)]
                or (summary.get("outcome") == "pending_review" and stages != declared)
                or summary.get("split_role") != scope.get("split_role")
                or type(offset) not in (int, float) or not math.isfinite(offset)):
            raise ValueError("Final varied-session fields disagree with scope")
    journal = json.loads((root / "target-command-journal.json").read_text())
    if not isinstance(journal, list) or not journal:
        raise ValueError("Missing initial hold and target journal")
    seen = set()
    last_sequence = -1
    last_unix = last_monotonic = None
    monotonic_journal = "issued_monotonic" in journal[0]
    for index, command in enumerate(journal):
        target, sequence = command.get("target"), command.get("sequence")
        issued = command.get("issued_unix")
        issued_mono = command.get("issued_monotonic")
        if (type(sequence) is not int or sequence in seen or sequence <= last_sequence
                or type(issued) not in (int, float) or not math.isfinite(issued) or issued <= 0
                or (not monotonic_journal and last_unix is not None and issued <= last_unix)
                or (monotonic_journal and (type(issued_mono) not in (int, float)
                    or not math.isfinite(issued_mono) or issued_mono <= 0
                    or (last_monotonic is not None and issued_mono <= last_monotonic)))
                or (not monotonic_journal and issued_mono is not None)
                or not isinstance(target, list) or len(target) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in target)
                or not 0 <= target[6] <= 100 or type(command.get("executed")) is not bool):
            raise ValueError("Invalid or duplicate target journal entry")
        seen.add(sequence)
        last_sequence, last_unix, last_monotonic = sequence, issued, issued_mono
        if index == 0:
            if command.get("kind") != "hold" or command["executed"]:
                raise ValueError("The first target must be the explicit unexecuted hold")
            continue
        if not command["executed"]:
            if command.get("kind") != "hold" or target != journal[index - 1]["target"]:
                raise ValueError("An unexecuted hold cannot change the actual latched target")
            continue
        name = f"command-{sequence:04d}.json"
        if name not in entries:
            raise ValueError("Executed target lacks its durable wire record")
        wire = json.loads((root / name).read_text())
        if wire.get("command") != command or wire.get("intended_target") != target:
            raise ValueError("Target journal differs from the actual command record")
        args, previous = wire.get("args"), journal[index - 1]["target"]
        if wire.get("method") == "MoveL":
            if command.get("kind") != "arm" or args[0] != target[:6] or target[6] != previous[6]:
                raise ValueError("Joint target does not match MoveL or preceding gripper target")
        elif wire.get("method") == "MoveGripper":
            if command.get("kind") != "gripper" or args[1] != target[6] or target[:6] != previous[:6]:
                raise ValueError("Gripper target does not match MoveGripper or preceding joint target")
        else:
            raise ValueError("Unsupported executed command")
        if (summary.get("outcome") == "pending_review" and
                (wire.get("status") != "completed" or wire.get("outcome", {}).get("return_code") != 0)):
            raise ValueError("Pending review contains an unconfirmed physical command")
        if (wire.get("status") == "completed" and
                ((monotonic_journal and (type(wire.get("completed_monotonic")) not in (int, float)
                                         or wire["completed_monotonic"] < issued_mono))
                 or (not monotonic_journal and (type(wire.get("completed_at_unix")) not in (int, float)
                                                 or wire["completed_at_unix"] < issued)))):
            raise ValueError("Command completion predates its actual issue time")
    if summary.get("outcome") == "pending_review":
        if (summary.get("released") is not True or summary.get("failed") or not summary.get("evidence")
                or summary.get("continuous_capture", {}).get("frames") != receipt["accepted_frames"]
                or summary.get("continuous_capture", {}).get("capture_complete") is not True
                or summary.get("dataset", {}).get("recorded_frames") != receipt["accepted_frames"]
                or summary.get("dataset", {}).get("storage_complete") is not True):
            raise ValueError("Final summary frame count or durability disagrees with stream")
        if "WORKFLOW_STARTED" in entries:
            if ("workflow-metrics.json" not in entries or
                    json.loads((root / "workflow-metrics.json").read_text()).get("outcome")
                    != "pending_user_whole_episode_review"):
                raise ValueError("Final workflow metrics are missing or stale")
        for name in entries:
            if name.startswith("command-") and name.endswith(".json"):
                wire = json.loads((root / name).read_text())
                if (wire.get("status") != "completed" or wire.get("outcome", {}).get("return_code") != 0
                        or wire.get("command", {}).get("sequence") not in seen):
                    raise ValueError("Pending review contains an unconfirmed command file")
    return manifest, scope, summary, journal


def _evidence_in_snapshot(temp, receipt, scope, summary):
    original = Path(receipt["local_episode"])
    if not original.is_absolute():
        raise ValueError("Original local episode path must be absolute")
    for name in [*(scope.get("evidence") or []), *(summary.get("evidence") or [])]:
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid evidence reference")
        path = Path(name)
        if path.is_absolute():
            try:
                path = path.relative_to(original)
            except ValueError as exc:
                raise ValueError("Evidence is outside the original session") from exc
        if ".." in path.parts or not (temp / path).is_file() or not (temp / path).resolve().is_relative_to(temp):
            raise ValueError("Evidence is missing from materialized episode: " + name)


def _frame_target(teacher, capture, journal):
    if journal is None:  # explicit read-only diagnostic
        return
    if isinstance(journal, dict) and journal.get('kind') == 'stream':
        if (capture.get('sync_clock') != 'host_monotonic'
                or teacher.get('recording_role') != 'policy_evaluation'
                or teacher.get('command_link') != 'resolve_actual_wire_events_by_capture_monotonic'
                or teacher.get('captured_monotonic') != capture['camera'].get('source_timestamp_monotonic')):
            raise ValueError('Stream frame lacks its real monotonic acquisition provenance')
        return
    basis = capture.get("sync_clock")
    if basis == "host_monotonic":
        clock, key = capture["camera"]["source_timestamp_monotonic"], "issued_monotonic"
    elif basis == "host_unix":
        clock, key = capture["camera"]["source_timestamp_unix"], "issued_unix"
    else:
        raise ValueError("Unsupported frame clock")
    eligible = [item for item in journal if item.get(key, float("inf")) <= clock]
    if not eligible or teacher.get("command") != eligible[-1]:
        raise ValueError("Frame target does not match the actual latched command")


def _stream_frame_commands(journal, captured, cursor):
    wire = journal['events']
    while cursor['index'] < len(wire) and wire[cursor['index']]['issued_monotonic'] <= captured:
        event = wire[cursor['index']]
        if event['method'] == 'ServoJ':
            cursor['servo'] = event
        elif event['method'] == 'MoveGripper':
            cursor['gripper'] = event
        cursor['index'] += 1
    return {'source_timestamp_monotonic': captured,
            'last_servoj_attempt': cursor['servo'], 'last_gripper_attempt': cursor['gripper'],
            'all_wire_callbacks_drained': journal['callbacks_drained'],
            'note': 'Attempted wire command; return code and physical feedback do not establish task success'}


def _stream_prediction_sources(journal, frames):
    """Link each saved inference request to the camera frame actually captured."""
    requests = [item['request'] for item in journal['chunks'].values()]
    requests += [item['request'] for item in journal['predictions']]
    retirements = journal.get('prefix_retirements', {})
    links = []
    for request in requests:
        frame = frames.get(request['source_time'])
        state = request.get('state')
        if (frame is None or not isinstance(state, list) or len(state) != 8
                or any(not _finite(x) for x in state)):
            raise ValueError('Prediction observation does not match a captured frame')
        robot = frame['robot']
        joints = robot.get('joints')
        if not isinstance(joints, list) or len(joints) != 6 or any(not _finite(x) for x in joints):
            raise ValueError('Prediction frame has invalid measured joints')
        grip = robot.get('gripper_position')
        valid = bool(_finite(grip) and 0 <= grip <= 100 and robot.get('gripper_active') == 1
                     and robot.get('gripper_fault') == 0 and robot.get('gripper_fault_id') == 0
                     and robot.get('gripper_motiondone') in (1, 2, 3) and robot.get('gripper_speed') == 0)
        expected = [math.radians(x) for x in joints] + [grip / 100. if valid else 0., float(valid)]
        if any(abs(a - b) > 1e-6 for a, b in zip(state, expected)):
            raise ValueError('Prediction state disagrees with captured robot feedback')
        retired = retirements.get(request['id'])
        if retired is not None:
            stamp = retired['acknowledged_monotonic']
            robot_time = robot.get('received_monotonic')
            if not _finite(robot_time) or robot_time <= stamp or not valid:
                raise ValueError('Post-gripper prediction lacks fresh settled camera and robot feedback after ACK')
        if request.get('observation_context_profile') == 'state-history-physical-cup-tcp-v1':
            base, relative, tcp = request.get('cup_base_xyz_mm'), request.get('cup_relative_state'), robot.get('tcp')
            if (any(not isinstance(v, list) or len(v) != n or not all(map(_finite, v))
                    for v,n in ((base,3),(relative,8),(tcp,6))) or robot.get('tool') != 0 or robot.get('user') != 0):
                raise ValueError('Cup-relative request lacks a measured tool0/user0 TCP')
            wanted = [(tcp[i]-base[i])/1000 for i in range(3)] + [math.radians(x) for x in tcp[3:]] + expected[6:]
            if max(abs(a-b) for a,b in zip(relative,wanted)) > 1e-6:
                raise ValueError('Cup-relative policy state differs from captured TCP feedback')
        jpeg = request.get('rgb_jpeg')
        if jpeg is not None and (jpeg.get('sha256') != frame['media']['rgb_sha256']
                                 or jpeg.get('bytes') != frame['media']['rgb_bytes']):
            raise ValueError('Prediction JPEG differs from recorded camera frame')
        link = {'request_id': request['id'], 'source_timestamp_monotonic': request['source_time'],
                'frame_index': frame['frame_index'], 'rgb_jpeg_sha256': frame['media']['rgb_sha256'],
                'request_jpeg_hash_verified': jpeg is not None}
        if retired is not None:
            link.update(previous_request_id=retired['request_id'],
                        prefix_consumed_steps=retired['prefix_consumed_steps'],
                        prefix_retired_cause=retired['prefix_retired_cause'],
                        acknowledged_monotonic=retired['acknowledged_monotonic'],
                        source_robot_timestamp_monotonic=robot_time,
                        gripper_wire_sequence=retired['gripper_wire_sequence'])
        links.append(link)
    return links


def materialize(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ValueError("An existing, non-symlink spool root is required")
    if (root / "materialized").exists():
        raise FileExistsError("Materialized episode already exists")
    if any(root.glob(".materializing-*")):
        raise RuntimeError("Previous incomplete materialization needs inspection")
    receipt = json.loads((root / "stream-closed.json").read_text())
    if (receipt.get("schema") != "fr5-stream-closed-v1"
            or receipt.get("error") is not None or receipt.get("unacknowledged_frames") != 0
            or type(receipt.get("accepted_frames")) is not int or receipt["accepted_frames"] < 1
            or receipt.get("last_durable_sequence") != receipt["accepted_frames"] - 1):
        raise ValueError("No complete server-acknowledged stream receipt")
    codec_path = root / "wire/frame_codec.py"
    protocol_path = root / "wire/stream_protocol.py"
    if _sha(codec_path.read_bytes()) != receipt.get("codec_sha256"):
        raise ValueError("Deployed codec differs from recording codec")
    if _sha(protocol_path.read_bytes()) != receipt.get("protocol_sha256"):
        raise ValueError("Deployed protocol differs from recording protocol")
    controls, scope, summary, journal = _control_snapshot(root, receipt)
    codec = _load("fr5_archived_codec", codec_path)
    protocol = _load("fr5_archived_protocol", protocol_path)
    session = receipt["session"]
    temp = root / (".materializing-" + uuid.uuid4().hex)
    temp.mkdir(mode=0o700)
    (temp / "continuous/frames").mkdir(parents=True)
    (temp / "lerobot/meta").mkdir(parents=True)
    _sync_dir(root)
    # All other evidence is small control-channel content. Copy it to the new
    # immutable materialization without rewriting source scope or provenance.
    excluded = {"frames.stream", "INCOMPLETE", "wire", "materialized", temp.name}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in excluded:
            continue
        if path.is_symlink():
            raise ValueError("Spool evidence contains a symlink")
        if path.is_file():
            if relative.as_posix() in ("continuous/capture.jsonl", "lerobot/meta/capture.jsonl"):
                continue  # rebuilt from the authoritative per-frame records
            destination = temp / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            _sync_file(destination)
    capture_path = temp / "continuous/capture.jsonl"
    teacher_path = temp / "lerobot/meta/capture.jsonl"
    stream_hash = hashlib.sha256()
    frame_count = 0
    last_camera = last_robot = basis = None
    stream_mode = isinstance(journal, dict) and journal.get('kind') == 'stream'
    stream_cursor = {'index': 0, 'servo': None, 'gripper': None}
    stream_frames = {}
    manifest_path = temp / "stream-frames.jsonl"
    with ExitStack() as stack:
        source = stack.enter_context((root / "frames.stream").open("rb"))
        captures = stack.enter_context(capture_path.open("x", encoding="utf-8"))
        teachers = stack.enter_context(teacher_path.open("x", encoding="utf-8"))
        manifest = stack.enter_context(manifest_path.open("x", encoding="utf-8"))
        command_log = (stack.enter_context((temp / 'stream-frame-commands.jsonl').open('x', encoding='utf-8'))
                       if stream_mode else None)
        optional_logs = []
        for filename in (root / "continuous/capture.jsonl", root / "lerobot/meta/capture.jsonl"):
            optional_logs.append(stack.enter_context(filename.open(encoding="utf-8"))
                                 if filename.is_file() and filename.stat().st_size else None)
        while True:
            head = source.read(protocol.HEADER.size)
            if not head:
                break
            if len(head) != protocol.HEADER.size:
                raise ValueError("Truncated stream header")
            magic, nm, nr, nd, _ = protocol.HEADER.unpack(head)
            if (magic != protocol.MAGIC or not 0 < nm <= protocol.MAX_META
                    or not 0 < nr <= protocol.MAX_MEDIA or not 0 < nd <= protocol.MAX_MEDIA):
                raise ValueError("Invalid stream record size")
            body = source.read(nm + nr + nd)
            if len(body) != nm + nr + nd:
                raise ValueError("Truncated stream record")
            packet = head + body
            stream_hash.update(packet)
            record, rgb, depth = protocol.unpack_record(packet)
            media, capture, teacher = record.get("media"), record.get("capture"), record.get("teacher")
            if (record.get("schema") != "fr5-stream-frame-v1" or record.get("session") != session
                    or record.get("sequence") != frame_count or not isinstance(capture, dict)
                    or not isinstance(teacher, dict) or not isinstance(media, dict)
                    or capture.get("frame_index") != frame_count
                    or teacher.get("frame_index") != frame_count
                    or capture.get("media_codec") != media
                    or capture.get("camera") != teacher.get("camera")
                    or capture.get("robot") != teacher.get("robot")
                    or capture.get("sync_clock") != teacher.get("sync_clock")
                    or capture.get("sync_skew_s") != teacher.get("sync_skew_s")):
                raise ValueError("Frame sequence or source provenance mismatch")
            frame_basis = capture["sync_clock"]
            if frame_basis == "host_monotonic":
                camera_time = capture["camera"].get("source_timestamp_monotonic")
                robot_time = capture["robot"].get("received_monotonic")
            elif frame_basis == "host_unix":
                camera_time = capture["camera"].get("source_timestamp_unix")
                robot_time = capture["robot"].get("timestamp")
            else:
                raise ValueError("Unsupported frame clock")
            if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    for value in (camera_time, robot_time))
                    or (basis is not None and basis != frame_basis)
                    or (last_camera is not None and camera_time <= last_camera)
                    or (last_robot is not None and robot_time <= last_robot)
                    or abs(robot_time - camera_time - capture["sync_skew_s"]) > 1e-6):
                raise ValueError("Frame clocks or synchronized state are inconsistent")
            basis, last_camera, last_robot = frame_basis, camera_time, robot_time
            _frame_target(teacher, capture, journal)
            if stream_mode:
                command = _stream_frame_commands(journal, camera_time, stream_cursor)
                command['frame_index'] = frame_count
                command_log.write(json.dumps(command, allow_nan=False, ensure_ascii=False) + '\n')
                stream_frames[camera_time] = {'frame_index': frame_count, 'robot': capture['robot'], 'media': media}
            for old_log, value in zip(optional_logs, (capture, teacher)):
                if old_log is not None:
                    line = old_log.readline()
                    if not line or json.loads(line) != value:
                        raise ValueError("Stream and persisted journal disagree")
            captures.write(json.dumps(capture, allow_nan=False, ensure_ascii=False) + "\n")
            teachers.write(json.dumps(teacher, allow_nan=False, ensure_ascii=False) + "\n")
            rgb_name, depth_name = f"frames/{frame_count:06d}.jpg", f"frames/{frame_count:06d}.npz"
            if capture.get("rgb_file") != rgb_name or capture.get("depth_file") != depth_name:
                raise ValueError("Unexpected materialized media path")
            _, depth_array = codec.decode_media(media, rgb, depth)
            rgb_path = temp / "continuous" / rgb_name
            depth_path = temp / "continuous" / depth_name
            with rgb_path.open("xb") as image:
                image.write(rgb)
                image.flush()
                os.fsync(image.fileno())
            np.savez_compressed(depth_path, depth_mm=depth_array)
            _sync_file(depth_path)
            with np.load(depth_path, allow_pickle=False) as archive:
                if (set(archive.files) != {"depth_mm"} or archive["depth_mm"].dtype != np.float32
                        or _sha(archive["depth_mm"].astype("<f4", copy=False).tobytes()) != media["depth_source_sha256"]):
                    raise ValueError("Materialized depth is not bit-identical to the source")
            if _sha(rgb_path.read_bytes()) != media["rgb_sha256"]:
                raise ValueError("Materialized JPEG hash mismatch")
            manifest.write(json.dumps({"sequence": frame_count, "packet_sha256": _sha(packet),
                "camera": capture["camera"], "sync_clock": capture["sync_clock"],
                "rgb_file": rgb_name, "rgb_jpeg_sha256": media["rgb_sha256"],
                "rgb_source_sha256": media["rgb_source_sha256"], "rgb_lossy": True,
                "depth_file": depth_name, "depth_source_sha256": media["depth_source_sha256"],
                "depth_lossless": True}, allow_nan=False, ensure_ascii=False) + "\n")
            frame_count += 1
        if (any(log is not None and log.readline() for log in optional_logs)
                or frame_count != receipt["accepted_frames"]):
            raise ValueError("Frame count or journal tail disagrees with receipt")
        for file in (manifest, captures, teachers):
            file.flush()
            os.fsync(file.fileno())
        if command_log is not None:
            command_log.flush()
            os.fsync(command_log.fileno())
    if stream_mode:
        links = _stream_prediction_sources(journal, stream_frames)
        with (temp / 'stream-prediction-sources.json').open('x', encoding='utf-8') as file:
            json.dump(links, file, allow_nan=False, ensure_ascii=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        if journal.get('execution_mode') in ('ordered_complete_chunk', 'ordered_policy_prefix'):
            with (temp / 'stream-target-timing.jsonl').open('x', encoding='utf-8') as file:
                for row in journal['ordered_timing']:
                    file.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + '\n')
                file.flush()
                os.fsync(file.fileno())
    if controls is not None:
        for name, item in controls["files"].items():
            if _sha((temp / name).read_bytes()) != item["sha256"]:
                raise ValueError("Materialized control/evidence differs from server snapshot: " + name)
        _evidence_in_snapshot(temp, receipt, scope, summary)
    result = {"schema": "fr5-stream-materialization-v1", "session": session,
              "frames": frame_count, "source_stream_sha256": stream_hash.hexdigest(),
              "codec_sha256": receipt["codec_sha256"],
              "protocol_sha256": _sha(protocol_path.read_bytes()),
              "capture_journal_sha256": _sha(capture_path.read_bytes()),
              "teacher_journal_sha256": _sha(teacher_path.read_bytes()),
              "rgb": "JPEG quality 90; lossy; source and JPEG hashes recorded per frame",
              "depth": "lossless float32 millimeters; source hash verified after materialization",
              "original_local_episode": receipt.get("local_episode"),
              "control_verification": "verified" if controls is not None else "diagnostic_only",
              "scope_sha256": controls.get("scope_sha256") if controls else None,
              "summary_sha256": controls.get("summary_sha256") if controls else None,
              "target_alignment": ("not_applicable_read_only" if stream_mode and journal['read_only_diagnostic']
                                   else "incomplete_missing_wire_callbacks" if stream_mode and not journal['callbacks_drained']
                                   else "verified_against_wire_and_frame_clocks" if journal is not None
                                   else "diagnostic_only"),
              "physical_commands_replayed": 0}
    if stream_mode:
        result['stream_command_alignment_sha256'] = _sha((temp / 'stream-frame-commands.jsonl').read_bytes())
        result['stream_prediction_sources_sha256'] = _sha((temp / 'stream-prediction-sources.json').read_bytes())
        result['dataset_role'] = 'policy_evaluation'
        result['task_success_verified'] = False
        if journal.get('execution_mode') in ('ordered_complete_chunk', 'ordered_policy_prefix'):
            result['stream_target_timing_sha256'] = _sha((temp / 'stream-target-timing.jsonl').read_bytes())
    result_path = temp / "stream-materialization.json"
    with result_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    for directory, _, _ in os.walk(temp, topdown=False):
        _sync_dir(directory)
    os.rename(temp, root / "materialized")
    _sync_dir(root)
    return result


def verify_materialized_legacy_diagnostic(root):
    """Verify a pre-manifest, explicitly read-only diagnostic without writes.

    Historical diagnostic receipts never certified control commands or task
    success.  Compare the original binary packets to both reconstructed
    journals and their RGB/depth files, and return the original receipt without
    upgrading its evidence status.
    """
    root = Path(root)
    target = root / 'materialized'
    if (not root.is_dir() or root.resolve() != root or root.is_symlink()
            or not target.is_dir() or target.is_symlink()
            or (root / 'control-manifest.json').exists() or (root / 'control-manifest.json').is_symlink()):
        raise ValueError('Expected an existing pre-manifest diagnostic')

    def regular(path):
        if path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError('Diagnostic contains a missing or unsafe file')
        return path

    def digest(path):
        value = hashlib.sha256()
        with regular(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(block)
        return value.hexdigest()

    scope = json.loads(regular(root / 'scope.json').read_text())
    closed = json.loads(regular(root / 'stream-closed.json').read_text())
    result = json.loads(regular(target / 'stream-materialization.json').read_text())
    if (scope.get('diagnostic_only') is not True or scope.get('training_ready') is not False
            or type(scope.get('physical_commands_sent')) is not int or scope['physical_commands_sent'] != 0
            or closed.get('schema') != 'fr5-stream-closed-v1' or closed.get('error') is not None
            or type(closed.get('accepted_frames')) is not int or closed['accepted_frames'] < 1
            or closed.get('last_durable_sequence') != closed['accepted_frames'] - 1
            or closed.get('unacknowledged_frames') != 0 or closed.get('unacknowledged_bytes') != 0
            or result.get('schema') != 'fr5-stream-materialization-v1'
            or result.get('session') != closed.get('session') or result.get('frames') != closed['accepted_frames']
            or result.get('original_local_episode') != closed.get('local_episode')
            or result.get('physical_commands_replayed') != 0
            or result.get('source_stream_sha256') != digest(root / 'frames.stream')
            or result.get('codec_sha256') != closed.get('codec_sha256')
            or result.get('codec_sha256') != digest(root / 'wire/frame_codec.py')
            or result.get('protocol_sha256') != closed.get('protocol_sha256')
            or result.get('protocol_sha256') != digest(root / 'wire/stream_protocol.py')
            or result.get('capture_journal_sha256') != digest(target / 'continuous/capture.jsonl')
            or result.get('teacher_journal_sha256') != digest(target / 'lerobot/meta/capture.jsonl')):
        raise ValueError('Legacy diagnostic receipt or source hash mismatch')

    codec = _load('fr5_legacy_diagnostic_codec', root / 'wire/frame_codec.py')
    protocol = _load('fr5_legacy_diagnostic_protocol', root / 'wire/stream_protocol.py')
    count, basis, last_camera, last_robot = 0, None, None, None
    with ExitStack() as stack:
        source = stack.enter_context(regular(root / 'frames.stream').open('rb'))
        captures = stack.enter_context(regular(target / 'continuous/capture.jsonl').open())
        teachers = stack.enter_context(regular(target / 'lerobot/meta/capture.jsonl').open())
        frame_manifest = stack.enter_context(regular(target / 'stream-frames.jsonl').open())
        while True:
            head = source.read(protocol.HEADER.size)
            if not head:
                break
            if len(head) != protocol.HEADER.size:
                raise ValueError('Legacy diagnostic packet header truncated')
            magic, nm, nr, nd, _ = protocol.HEADER.unpack(head)
            if (magic != protocol.MAGIC or not 0 < nm <= protocol.MAX_META
                    or not 0 < nr <= protocol.MAX_MEDIA or not 0 < nd <= protocol.MAX_MEDIA):
                raise ValueError('Legacy diagnostic packet sizes invalid')
            body = source.read(nm + nr + nd)
            if len(body) != nm + nr + nd:
                raise ValueError('Legacy diagnostic packet truncated')
            packet = head + body
            record, rgb, depth = protocol.unpack_record(packet)
            capture_line, teacher_line, manifest_line = captures.readline(), teachers.readline(), frame_manifest.readline()
            if not capture_line or not teacher_line or not manifest_line:
                raise ValueError('Legacy diagnostic reconstructed journals are incomplete')
            capture, teacher, manifest = json.loads(capture_line), json.loads(teacher_line), json.loads(manifest_line)
            media = record.get('media')
            if (not all(isinstance(item, dict) for item in (capture, teacher, manifest, media))
                    or record.get('schema') != 'fr5-stream-frame-v1' or record.get('session') != closed['session']
                    or record.get('sequence') != count or record.get('capture') != capture
                    or record.get('teacher') != teacher or capture.get('frame_index') != count
                    or teacher.get('frame_index') != count or capture.get('camera') != teacher.get('camera')
                    or capture.get('robot') != teacher.get('robot') or capture.get('sync_clock') != teacher.get('sync_clock')
                    or capture.get('sync_skew_s') != teacher.get('sync_skew_s') or capture.get('media_codec') != media
                    or teacher.get('command', {}).get('executed') is not False
                    or manifest.get('sequence') != count or manifest.get('packet_sha256') != _sha(packet)
                    or manifest.get('camera') != capture['camera'] or manifest.get('sync_clock') != capture['sync_clock']
                    or manifest.get('rgb_source_sha256') != media.get('rgb_source_sha256')
                    or manifest.get('rgb_lossy') is not True or manifest.get('depth_lossless') is not True):
                raise ValueError('Legacy diagnostic packet and reconstructed source disagree')
            frame_basis = capture.get('sync_clock')
            if frame_basis == 'host_monotonic':
                camera_time = capture['camera'].get('source_timestamp_monotonic')
                robot_time = capture['robot'].get('received_monotonic')
            elif frame_basis == 'host_unix':
                camera_time = capture['camera'].get('source_timestamp_unix')
                robot_time = capture['robot'].get('timestamp')
            else:
                raise ValueError('Legacy diagnostic frame clock unsupported')
            if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    for value in (camera_time, robot_time))
                    or (basis is not None and basis != frame_basis)
                    or (last_camera is not None and camera_time <= last_camera)
                    or (last_robot is not None and robot_time <= last_robot)
                    or abs(robot_time - camera_time - capture['sync_skew_s']) > 1e-6):
                raise ValueError('Legacy diagnostic frame timestamps are inconsistent')
            basis, last_camera, last_robot = frame_basis, camera_time, robot_time
            rgb_name, depth_name = f'frames/{count:06d}.jpg', f'frames/{count:06d}.npz'
            if (capture.get('rgb_file') != rgb_name or capture.get('depth_file') != depth_name
                    or manifest.get('rgb_file') != rgb_name or manifest.get('depth_file') != depth_name
                    or manifest.get('rgb_jpeg_sha256') != media['rgb_sha256']
                    or manifest.get('depth_source_sha256') != media['depth_source_sha256']
                    or regular(target / 'continuous' / rgb_name).read_bytes() != rgb):
                raise ValueError('Legacy diagnostic media path or JPEG mismatch')
            _, decoded_depth = codec.decode_media(media, rgb, depth)
            with np.load(regular(target / 'continuous' / depth_name), allow_pickle=False) as archive:
                if (set(archive.files) != {'depth_mm'} or archive['depth_mm'].dtype != np.float32
                        or not np.array_equal(archive['depth_mm'], decoded_depth)):
                    raise ValueError('Legacy diagnostic depth mismatch')
            count += 1
        if captures.readline() or teachers.readline() or frame_manifest.readline() or count != closed['accepted_frames']:
            raise ValueError('Legacy diagnostic frame count mismatch')
    return result


def verify_materialized(root):
    """Read back an existing reconstruction after a lost response; never rewrite it."""
    root = Path(root)
    target = root/'materialized'
    if (not root.is_dir() or root.resolve() != root or root.is_symlink()
            or not target.is_dir() or target.is_symlink()):
        raise ValueError('Existing materialized recording is not a regular directory')
    def digest(path):
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError('Materialized recording contains an invalid file')
        value = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    closed = json.loads((root/'stream-closed.json').read_text())
    controls, _, _, journal = _control_snapshot(root, closed)
    result = json.loads((target/'stream-materialization.json').read_text())
    stream_mode = isinstance(journal, dict) and journal.get('kind') == 'stream'
    expected_alignment = ('not_applicable_read_only' if stream_mode and journal['read_only_diagnostic']
                          else 'incomplete_missing_wire_callbacks' if stream_mode and not journal['callbacks_drained']
                          else 'verified_against_wire_and_frame_clocks')
    if (controls is None or result.get('schema') != 'fr5-stream-materialization-v1'
            or result.get('session') != closed.get('session')
            or result.get('frames') != closed.get('accepted_frames')
            or result.get('scope_sha256') != controls['scope_sha256']
            or result.get('summary_sha256') != controls['summary_sha256']
            or result.get('control_verification') != 'verified'
            or result.get('target_alignment') != expected_alignment
            or result.get('source_stream_sha256') != digest(root/'frames.stream')
            or result.get('codec_sha256') != digest(root/'wire/frame_codec.py')
            or result.get('protocol_sha256') != digest(root/'wire/stream_protocol.py')
            or result.get('capture_journal_sha256') != digest(target/'continuous/capture.jsonl')
            or result.get('teacher_journal_sha256') != digest(target/'lerobot/meta/capture.jsonl')):
        raise ValueError('Existing materialization no longer matches its recorded source')
    if stream_mode and (result.get('dataset_role') != 'policy_evaluation'
            or result.get('task_success_verified') is not False
            or result.get('stream_command_alignment_sha256') != digest(target/'stream-frame-commands.jsonl')
            or result.get('stream_prediction_sources_sha256') != digest(target/'stream-prediction-sources.json')):
        raise ValueError('Existing stream evaluation alignment no longer matches its recorded source')
    if stream_mode and journal.get('execution_mode') in ('ordered_complete_chunk', 'ordered_policy_prefix'):
        path = target/'stream-target-timing.jsonl'
        if (result.get('stream_target_timing_sha256') != digest(path)
                or [json.loads(line) for line in path.read_text().splitlines()] != journal['ordered_timing']):
            raise ValueError('Existing ordered target timing no longer matches its recorded source')
    for name, entry in controls['files'].items():
        if digest(target/name) != entry['sha256']:
            raise ValueError('Materialized control file changed: '+name)
    count = 0
    cursor = {'index': 0, 'servo': None, 'gripper': None}
    frames = {}
    with ExitStack() as stack:
        captures = stack.enter_context((target/'continuous/capture.jsonl').open())
        teachers = stack.enter_context((target/'lerobot/meta/capture.jsonl').open())
        command_log = stack.enter_context((target/'stream-frame-commands.jsonl').open()) if stream_mode else None
        for index, line in enumerate(captures):
            row = json.loads(line)
            teacher = json.loads(teachers.readline())
            _frame_target(teacher, row, journal)
            if (row.get('frame_index') != index or row.get('rgb_file') != f'frames/{index:06d}.jpg'
                    or row.get('depth_file') != f'frames/{index:06d}.npz'):
                raise ValueError('Materialized frame sequence changed')
            media = row['media_codec']
            if stream_mode:
                captured = row['camera']['source_timestamp_monotonic']
                expected = _stream_frame_commands(journal, captured, cursor)
                expected['frame_index'] = index
                command_line = command_log.readline()
                if not command_line or json.loads(command_line) != expected:
                    raise ValueError('Materialized stream frame command alignment changed')
                frames[captured] = {'frame_index': index, 'robot': row['robot'], 'media': media}
            if digest(target/'continuous'/row['rgb_file']) != media['rgb_sha256']:
                raise ValueError('Materialized image changed')
            path = target/'continuous'/row['depth_file']
            if path.is_symlink() or path.resolve() != path:
                raise ValueError('Materialized depth is not a regular file')
            with np.load(path, allow_pickle=False) as archive:
                depth = archive['depth_mm']
                if (set(archive.files) != {'depth_mm'} or depth.dtype != np.float32
                        or depth.shape != (media['height'], media['width'])
                        or _sha(depth.astype('<f4',copy=False).tobytes()) != media['depth_source_sha256']):
                    raise ValueError('Materialized depth changed')
            count += 1
        if teachers.readline() or (command_log is not None and command_log.readline()) or count != result['frames']:
            raise ValueError('Materialized frame count changed')
    if stream_mode and json.loads((target/'stream-prediction-sources.json').read_text()) != _stream_prediction_sources(journal, frames):
        raise ValueError('Materialized prediction source links changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    base = Path("/home/data4/hts/demo1/fr5-learning/recording-spool")
    if args.root.parent != base:
        raise ValueError("Only a fresh 4090 recording spool can be materialized")
    print(json.dumps(materialize(args.root), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
