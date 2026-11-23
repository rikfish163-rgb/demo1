#!/usr/bin/env python3
"""Analyze a recorded ordered-policy episode using local timing evidence only."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import statistics


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def read_jsonl(paths):
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    # Nearest-rank percentiles use only observed samples.
    def percentile(p):
        return values[math.ceil(p * len(values)) - 1]
    return {'count': len(values), 'min': values[0], 'mean': statistics.fmean(values),
            'p50': statistics.median(values), 'p90': percentile(.90),
            'p95': percentile(.95), 'p99': percentile(.99), 'max': values[-1],
            'sum': sum(values)}


def request_observation(request):
    valid = request['state'][7] == 1
    return {
        'observation_source_monotonic': request['source_time'],
        'gripper_position_percent': request['state'][6] * 100 if valid else None,
        'valid': valid, 'request_id': request['id'], 'observation_kind': 'request',
    }


def predictions(chunks, prediction_records):
    records_by_id = {item['request']['id']: item for item in prediction_records}
    if len(records_by_id) != len(prediction_records) or len(records_by_id) != len(chunks):
        raise ValueError('Prediction journal and chunk counts or request IDs do not match')
    rows = []
    for index, chunk in enumerate(chunks):
        request, reply = chunk['request'], chunk['reply']
        record = records_by_id[request['id']]
        if (record['reply']['inference_seconds'] != reply['inference_seconds']
                or record['request']['source_time'] != request['source_time']):
            raise ValueError(f'Prediction journal disagrees with chunk {index}')
        source = request['source_time']
        requested = chunk['requested_monotonic']
        ready = chunk['observation_ready_monotonic']
        started = chunk['predict_started_monotonic']
        received = chunk['received_monotonic']
        pre_start = chunk['prefix_precheck_started_monotonic']
        pre_end = chunk['prefix_precheck_finished_monotonic']
        inference = reply['inference_seconds']
        preflight = chunk['joint_prefix_preflight']
        fk_count = int(preflight['anchor_fk_sample'] is not None) + sum(
            len(segment['fk_samples']) for segment in preflight['segments'])
        rows.append({
            'chunk_index': index, 'request_id': request['id'],
            'source_monotonic': source, 'requested_monotonic': requested,
            'observation_after_monotonic': chunk.get('observation_after_monotonic'),
            'observation_ready_monotonic': ready, 'predict_started_monotonic': started,
            'reply_received_monotonic': received, 'precheck_started_monotonic': pre_start,
            'precheck_finished_monotonic': pre_end,
            'server_inference_s': inference,
            'observation_wait_s': ready - requested,
            'observation_ready_to_predict_s': started - ready,
            'client_predict_elapsed_s': received - started,
            'client_predict_minus_server_inference_s': received - started - inference,
            'reply_to_precheck_s': pre_start - received,
            'prefix_precheck_s': pre_end - pre_start,
            'source_age_at_reply_s': received - source,
            'source_age_at_prefix_ready_s': pre_end - source,
            'prefix_steps': chunk['execution_prefix_steps'],
            'prefix_preflight_passed': preflight['passed'], 'fk_sample_count': fk_count,
            'fresh_observation': request_observation(request),
        })
    return rows


def validate_unacknowledged_tail(submitted, acknowledgements, summary):
    snapshot = (summary or {}).get('controller_snapshot')
    if not isinstance(snapshot, dict):
        raise ValueError('Unacknowledged tail has no final controller snapshot')
    reason = summary.get('termination_source')
    clean_snapshot = {
        'status': 'closed', 'reason': reason, 'stop_result': 0, 'servo_end_result': 0,
        'pending_callbacks': 0, 'inflight_commands': 0, 'workers_alive': [],
        'undelivered_callbacks': [], 'startup_inflight': False,
        'gripper_command_inflight': False,
    }
    if (reason not in ('explicit_duration_limit', 'operator_stop')
            or summary.get('reason') != reason
            or summary.get('summary_stage') != 'final_control_snapshot'
            or summary.get('wire_callbacks_drained') is not True
            or any(summary.get(key, 'not_recorded') is not None
                   for key in ('error', 'failed', 'journal_error'))
            or any(snapshot.get(key, 'not_recorded') is not None
                   for key in ('failure', 'callback_failure'))
            or any(snapshot.get(key) != value for key, value in clean_snapshot.items())):
        raise ValueError('Unacknowledged tail lacks consistent clean termination evidence')
    tail = submitted[-1]
    token = tail['delivery_token']
    ack_token = acknowledgements[-1]['delivery_token'] if acknowledgements else 0
    if (snapshot.get('ordered_last_token') != token
            or snapshot.get('ordered_acknowledged_token') != ack_token):
        raise ValueError('Unacknowledged tail snapshot tokens disagree with logged targets')
    pending = snapshot.get('ordered_pending')
    if (not isinstance(pending, dict) or pending.get('token') != token
            or pending.get('version') != token
            or pending.get('joints') != [math.degrees(value) for value in tail['target'][:6]]
            or pending.get('gripper') != tail['effective_gripper_target_percent']
            or not isinstance(pending.get('joint_ack'), bool)
            or not isinstance(pending.get('gripper_ack'), bool)
            or (pending['joint_ack'] and pending['gripper_ack'])):
        raise ValueError('Unacknowledged tail pending target disagrees with its submission')


def target_timings(submitted, acknowledgements, prediction_rows, summary=None):
    for sequence, item in enumerate(submitted):
        if item['sequence'] != sequence:
            raise ValueError('Submitted target sequence is not a contiguous ordered sequence')
        if item['delivery_token'] != sequence + 1:
            raise ValueError('Submitted target token disagrees with its ordered sequence')
    for sequence, ack in enumerate(acknowledgements):
        if ack['sequence'] != sequence or sequence >= len(submitted):
            raise ValueError('Acknowledgement sequence is not a contiguous submitted prefix')
        if ack['delivery_token'] != submitted[sequence]['delivery_token']:
            raise ValueError('Target acknowledgement token does not match')
    if len(submitted) != len(acknowledgements):
        if len(submitted) != len(acknowledgements) + 1:
            raise ValueError('Only one stopped tail target may lack an acknowledgement')
        validate_unacknowledged_tail(submitted, acknowledgements, summary)
    by_id = {item['request_id']: item for item in prediction_rows}
    rows = []
    for item in submitted:
        sequence = item['sequence']
        ack = acknowledgements[sequence] if sequence < len(acknowledgements) else None
        consumed = item.get('prefix_consumed_steps')
        ack_consumed = ack.get('prefix_consumed_steps') if ack else None
        if consumed is not None and ack_consumed is not None and consumed != ack_consumed:
            raise ValueError('Submitted and acknowledged prefix consumption does not match')
        source = by_id[item['request_id']]['source_monotonic']
        submit = item['submitted_monotonic']
        acknowledged = ack['acknowledged_monotonic'] if ack else None
        row = {
            'sequence': item['sequence'], 'delivery_token': item['delivery_token'],
            'request_id': item['request_id'], 'action_index': item['action_index'],
            'delivery_status': 'acknowledged' if ack else 'unacknowledged_at_stop',
            'prefix_consumed_steps': consumed, 'ack_prefix_consumed_steps': ack_consumed,
            'gripper_target_percent': item['effective_gripper_target_percent'],
            'submitted_monotonic': submit, 'acknowledged_monotonic': acknowledged,
            'target_monotonic': item['target_monotonic'],
            'predicted_source_target_monotonic': item['predicted_source_target_monotonic'],
            'source_age_at_submit_s': submit - source,
            'source_age_at_ack_s': acknowledged - source if ack else None,
            'scheduled_target_lateness_s': item['target_monotonic'] - item['predicted_source_target_monotonic'],
            'submit_to_ack_s': acknowledged - submit if ack else None,
        }
        for key in ('prefix_retired_cause', 'gripper_commands_before', 'gripper_commands_after'):
            if ack is None:
                row[key] = None
            elif key in ack:
                row[key] = ack[key]
        rows.append(row)
    return rows


def retirement_timings(targets, chunks):
    by_request = {}
    for target in targets:
        by_request.setdefault(target['request_id'], []).append(target)
    rows = []
    for index, chunk in enumerate(chunks):
        request_id = chunk['request']['id']
        request_targets = by_request.get(request_id, [])
        retired = [target for target in request_targets if target.get('prefix_retired_cause') is not None]
        target = request_targets[-1] if request_targets else None
        if retired and (len(retired) != 1 or retired[0] is not target):
            raise ValueError('Request retirement must be its last acknowledged target')
        following = chunks[index + 1] if index + 1 < len(chunks) else None
        if following is None and not retired:
            continue
        if target is None or target['acknowledged_monotonic'] is None:
            raise ValueError('Next prediction has no previous request retirement acknowledgement')
        ack = target['acknowledged_monotonic']
        if following is not None:
            after = following.get('observation_after_monotonic')
            if (('prefix_retired_cause' in target and target['prefix_retired_cause'] is None)
                    or (after != ack and (after is not None or 'prefix_retired_cause' in target))):
                raise ValueError('Next prediction disagrees with its previous request retirement ACK')
            source = following['request']['source_time']
            requested = following['requested_monotonic']
            ready = following['observation_ready_monotonic']
            if source <= ack or requested < ack or ready < source or ready < requested:
                raise ValueError('Next current observation is not fresh after retirement ACK')
        row = {
            'request_id': request_id, 'target_sequence': target['sequence'],
            'delivery_token': target['delivery_token'], 'acknowledged_monotonic': ack,
            'prefix_consumed_steps': target['ack_prefix_consumed_steps'],
            'prefix_retired_cause': target.get('prefix_retired_cause'),
            'gripper_commands_before': target.get('gripper_commands_before'),
            'gripper_commands_after': target.get('gripper_commands_after'),
            'next_request_id': following['request']['id'] if following else None,
            'next_requested_monotonic': following['requested_monotonic'] if following else None,
            'next_source_monotonic': following['request']['source_time'] if following else None,
            'next_observation_after_monotonic': following.get('observation_after_monotonic') if following else None,
            'next_observation_ready_monotonic': following['observation_ready_monotonic'] if following else None,
            'fresh_prediction_observation': request_observation(following['request']) if following else None,
        }
        for name, key in (('request', 'next_requested_monotonic'), ('source', 'next_source_monotonic'),
                          ('observation_ready', 'next_observation_ready_monotonic')):
            row[f'ack_to_next_{name}_s'] = row[key] - ack if following else None
        rows.append(row)
    return rows


def gripper_timings(wire, targets, chunks, snapshot, retirements=None):
    servos = sorted((item for item in wire if item['method'] == 'ServoJ'),
                    key=lambda item: item['issued_monotonic'])
    commands = sorted((item for item in wire if item['method'] == 'MoveGripper'),
                      key=lambda item: item['issued_monotonic'])
    servo_issued = [item['issued_monotonic'] for item in servos]
    servo_returned = [item['returned_monotonic'] for item in servos]
    retirements = retirement_timings(targets, chunks) if retirements is None else retirements
    retired_by_request = {item['request_id']: item for item in retirements}
    last_feedback = snapshot.get('gripper_completion_feedback')
    rows = []
    for index, command in enumerate(commands):
        issued, returned = command['issued_monotonic'], command['returned_monotonic']
        matching = [item for item in targets
                    if item['submitted_monotonic'] <= issued
                    and (item['acknowledged_monotonic'] is None or issued <= item['acknowledged_monotonic'])]
        if len(matching) != 1:
            raise ValueError(f'Cannot uniquely match gripper command {command["sequence"]} to a target')
        target = matching[0]
        if target['gripper_target_percent'] != command['target_deg_percent']:
            raise ValueError('Gripper wire command disagrees with its ordered target')
        previous_index = bisect_right(servo_returned, issued) - 1
        next_index = bisect_left(servo_issued, returned)
        previous_return = servos[previous_index]['returned_monotonic'] if previous_index >= 0 else None
        next_issue = servos[next_index]['issued_monotonic'] if next_index < len(servos) else None
        ack = target['acknowledged_monotonic']
        upper_bounds = [value for value in (next_issue, ack) if value is not None]
        completion_upper = min(upper_bounds) if upper_bounds else None
        if completion_upper is not None and completion_upper < returned:
            raise ValueError('Gripper completion cannot precede RPC return')
        feedback = None
        if (last_feedback and last_feedback['target_percent'] == command['target_deg_percent']
                and (previous_return if previous_return is not None else target['submitted_monotonic'])
                <= last_feedback['command_began_monotonic'] <= issued
                and returned <= last_feedback['completed_monotonic']
                and (completion_upper is None or last_feedback['completed_monotonic'] <= completion_upper)):
            feedback = {key: last_feedback[key] for key in (
                'target_percent', 'position_percent', 'motiondone', 'feedback_monotonic',
                'command_began_monotonic', 'completed_monotonic')}
        retirement = retired_by_request.get(target['request_id'], {})
        observed = retirement.get('fresh_prediction_observation')
        row = {
            'number': index + 1, 'wire_sequence': command['sequence'],
            'target_sequence': target['sequence'], 'delivery_token': target['delivery_token'],
            'request_id': target['request_id'], 'action_index': target['action_index'],
            'target_percent': command['target_deg_percent'],
            'submitted_monotonic': target['submitted_monotonic'],
            'rpc_issued_monotonic': issued, 'rpc_returned_monotonic': returned,
            'rpc_callback_monotonic': command.get('callback_monotonic'),
            'rpc_return_code': command['return_code'], 'rpc_error': command['error'],
            'acknowledged_monotonic': ack,
            'submit_to_rpc_issue_s': issued - target['submitted_monotonic'],
            'rpc_elapsed_s': returned - issued,
            'rpc_return_to_ack_s': ack - returned if ack is not None else None,
            'submit_to_ack_s': ack - target['submitted_monotonic'] if ack is not None else None,
            'previous_servoj_returned_monotonic': previous_return,
            'next_servoj_issued_monotonic': next_issue,
            'surrounding_servoj_gap_s': (next_issue - previous_return
                                         if next_issue is not None and previous_return is not None else None),
            'completion_detection_monotonic_bounds': ([returned, completion_upper]
                                                      if completion_upper is not None else None),
            'completion_wait_after_rpc_s_bounds': ([0.0, completion_upper - returned]
                                                   if completion_upper is not None else None),
            'recorded_completion_feedback': feedback,
            'recorded_rpc_return_to_completion_s': (feedback['completed_monotonic'] - returned
                                                    if feedback else None),
            'recorded_completion_to_ack_s': (ack - feedback['completed_monotonic']
                                             if feedback and ack is not None else None),
            'retirement_target_sequence': retirement.get('target_sequence'),
            'fresh_prediction_observation': observed,
            'first_valid_post_ack_prediction_observation': (
                observed if observed and observed['valid'] and ack is not None
                and observed['observation_source_monotonic'] > ack else None),
        }
        for key in ('next_request_id', 'next_requested_monotonic', 'next_source_monotonic',
                    'next_observation_after_monotonic', 'next_observation_ready_monotonic'):
            row[key] = retirement.get(key)
        for name, key in (('request', 'next_requested_monotonic'), ('source', 'next_source_monotonic'),
                          ('observation_ready', 'next_observation_ready_monotonic')):
            event = row[key]
            row[f'ack_to_next_{name}_s'] = event - ack if event is not None and ack is not None else None
            row[f'recorded_completion_to_next_{name}_s'] = (
                event - feedback['completed_monotonic'] if event is not None and feedback else None)
            row[f'completion_to_next_{name}_s_bounds'] = (
                [event - completion_upper, event - returned]
                if event is not None and completion_upper is not None else None)
        rows.append(row)
    return rows


def servo_summary(wire, snapshot):
    counts = Counter(item['method'] for item in wire)
    servos = sorted((item for item in wire if item['method'] == 'ServoJ'),
                    key=lambda item: item['issued_monotonic'])
    non_servo = [{key: item.get(key) for key in ('method', 'sequence', 'issued_monotonic',
                  'returned_monotonic', 'return_code', 'error')}
                 for item in wire if item['method'] in ('ServoMoveStart', 'StopMotion', 'ServoMoveEnd')]
    return {
        'wire_method_counts': dict(counts),
        'servoj_return_codes': dict(Counter(str(item['return_code']) for item in servos)),
        'servoj_errors': [item['sequence'] for item in servos if item['error'] is not None],
        'servoj_rpc_s': distribution(item['returned_monotonic'] - item['issued_monotonic']
                                     for item in servos),
        'servoj_issue_interval_s': distribution(b['issued_monotonic'] - a['issued_monotonic']
                                                for a, b in zip(servos, servos[1:])),
        'servoj_return_to_next_issue_s': distribution(b['issued_monotonic'] - a['returned_monotonic']
                                                      for a, b in zip(servos, servos[1:])),
        'lifecycle_rpc': non_servo,
        'final_snapshot': {key: snapshot.get(key) for key in (
            'status', 'reason', 'failure', 'callback_failure', 'stop_result', 'servo_end_result',
            'servo_commands', 'gripper_commands', 'servo_overruns', 'peripheral_servo_pauses',
            'peripheral_servo_pause_seconds', 'gripper_command_inflight', 'ordered_last_token',
            'ordered_acknowledged_token', 'ordered_pending', 'pending_callbacks',
            'inflight_commands', 'undelivered_callbacks', 'startup_inflight', 'workers_alive')},
    }


def failure_chain(summary, scope, prediction_rows, target_rows, gripper_rows, wire):
    match = re.search(r'request ([0-9a-f]+), action (\d+) \(limit ([0-9.]+)s\)',
                      summary.get('error') or '')
    if not match:
        return {'recorded_error': summary.get('error'), 'timing_chain_available': False}
    request_id, action_text, limit_text = match.groups()
    action, limit = int(action_text), float(limit_text)
    prediction = next(item for item in prediction_rows if item['request_id'] == request_id)
    earlier = [item for item in target_rows if item['request_id'] == request_id
               and item['action_index'] == action - 1]
    if len(earlier) != 1:
        raise ValueError('The target preceding the recorded late action is missing')
    previous = earlier[0]
    fps = scope['policy_contract']['fps']
    expected = prediction['source_monotonic'] + (action + 1) / fps
    ack = previous['acknowledged_monotonic']
    stop = next(item for item in wire if item['method'] == 'StopMotion')
    stop_issued = stop['issued_monotonic']
    command = next((item for item in gripper_rows if item['target_sequence'] == previous['sequence']), None)
    if not ack <= stop_issued:
        raise ValueError('StopMotion precedes the previous target acknowledgement')
    return {
        'recorded_error': summary['error'], 'request_id': request_id, 'rejected_action_index': action,
        'max_target_lateness_s': limit, 'fps': fps,
        'prediction': prediction, 'previous_action': previous, 'previous_action_gripper': command,
        'rejected_action_source_target_monotonic': expected,
        'latest_passing_check_monotonic': expected + limit - 1 / fps,
        'check_monotonic_recorded_bounds': [ack, stop_issued],
        'lateness_s_recorded_bounds': [ack + 1 / fps - expected, stop_issued + 1 / fps - expected],
        'minimum_lateness_over_limit_s': ack + 1 / fps - expected - limit,
        'stopmotion_issued_monotonic': stop_issued,
        'stopmotion_returned_monotonic': stop['returned_monotonic'],
        'stopmotion_return_code': stop['return_code'],
        'explanation': ('下一次检查调用 chunk.take(now + 1/fps)；日志没有单独记录 now，'
                        '因此检查时刻及迟到量以先前确认和 StopMotion 发出时间为界。'),
    }


def analyze(episode):
    scope = read_json(episode / 'scope.json')
    summary = read_json(episode / 'summary.json')
    chunks = [read_json(path) for path in sorted(episode.glob('chunk-*.json'))]
    journal = list(read_jsonl(sorted((episode / 'stream-prediction').glob('*.jsonl'))))
    submitted = list(read_jsonl([episode / 'submitted-targets.jsonl']))
    acknowledgements = list(read_jsonl([episode / 'acknowledged-targets.jsonl']))
    wire = list(read_jsonl(sorted((episode / 'stream-wire').glob('*.jsonl'))))
    prediction_rows = predictions(chunks, journal)
    target_rows = target_timings(submitted, acknowledgements, prediction_rows, summary)
    retirement_rows = retirement_timings(target_rows, chunks)
    snapshot = summary['controller_snapshot']
    gripper_rows = gripper_timings(wire, target_rows, chunks, snapshot, retirement_rows)
    acknowledged_rows = [item for item in target_rows if item['acknowledged_monotonic'] is not None]
    servo = servo_summary(wire, snapshot)
    if (len(chunks) != summary['prediction_records'] or len(wire) != summary['wire_records']
            or servo['wire_method_counts'].get('ServoJ', 0) != snapshot['servo_commands']
            or len(gripper_rows) != snapshot['gripper_commands']):
        raise ValueError('Local records disagree with the final controller summary')
    failure = failure_chain(summary, scope, prediction_rows, target_rows, gripper_rows, wire)
    first_wire = wire[0]
    offset = first_wire['issued_unix'] - first_wire['issued_monotonic']
    local_tz = timezone(timedelta(hours=8))
    if failure.get('timing_chain_available', True):
        failure['source_time_china_approx'] = datetime.fromtimestamp(
            failure['prediction']['source_monotonic'] + offset, local_tz).isoformat(timespec='milliseconds')
        failure['stop_time_china_approx'] = datetime.fromtimestamp(
            failure['stopmotion_issued_monotonic'] + offset, local_tz).isoformat(timespec='milliseconds')
    prediction_fields = ('server_inference_s', 'observation_wait_s', 'observation_ready_to_predict_s',
                         'client_predict_elapsed_s', 'client_predict_minus_server_inference_s',
                         'reply_to_precheck_s', 'prefix_precheck_s', 'source_age_at_reply_s',
                         'source_age_at_prefix_ready_s')
    target_fields = ('source_age_at_submit_s', 'source_age_at_ack_s',
                     'scheduled_target_lateness_s', 'submit_to_ack_s')
    gripper_fields = ('submit_to_rpc_issue_s', 'rpc_elapsed_s', 'rpc_return_to_ack_s',
                      'submit_to_ack_s', 'surrounding_servoj_gap_s', 'recorded_completion_to_ack_s',
                      'ack_to_next_request_s', 'ack_to_next_source_s', 'ack_to_next_observation_ready_s',
                      'recorded_completion_to_next_request_s', 'recorded_completion_to_next_source_s',
                      'recorded_completion_to_next_observation_ready_s')
    gripper_sequences = {item['target_sequence'] for item in gripper_rows}
    gripper_distributions = {key: distribution(item[key] for item in gripper_rows if item[key] is not None)
                             for key in gripper_fields}
    gripper_distributions['recorded_rpc_return_to_completion_s'] = distribution(
        item['recorded_rpc_return_to_completion_s'] for item in gripper_rows
        if item['recorded_rpc_return_to_completion_s'] is not None)
    gripper_distributions['completion_wait_after_rpc_upper_bound_s'] = distribution(
        item['completion_wait_after_rpc_s_bounds'][1] for item in gripper_rows
        if item['completion_wait_after_rpc_s_bounds'] is not None)
    cause_names = {
        item['sequence']: (item['prefix_retired_cause'] or 'null'
                           if 'prefix_retired_cause' in item else 'not_recorded')
        for item in acknowledged_rows
    }
    cause_counts = dict(Counter(cause_names.values()))
    retirement_fields = ('ack_to_next_request_s', 'ack_to_next_source_s', 'ack_to_next_observation_ready_s')
    prefix_retirement = {
        'acknowledgement_cause_counts': cause_counts,
        'consumed_steps_by_cause': {
            cause: dict(Counter(str(item['ack_prefix_consumed_steps'])
                                if item['ack_prefix_consumed_steps'] is not None else 'not_recorded'
                                for item in acknowledged_rows if cause_names[item['sequence']] == cause))
            for cause in cause_counts
        },
        'next_observation_distributions_s': {
            cause: {key: distribution(item[key] for item in retirement_rows
                                      if cause_names[item['target_sequence']] == cause and item[key] is not None)
                    for key in retirement_fields}
            for cause in cause_counts if cause != 'null'
        },
    }
    exact_feedback_count = sum(item['recorded_completion_feedback'] is not None for item in gripper_rows)
    missing_feedback_count = len(gripper_rows) - exact_feedback_count
    limitations = []
    if missing_feedback_count:
        limitations.append(
            f'{missing_feedback_count} 次夹爪命令的精确完成反馈未写入这些本地控制日志；'
            '完成区间及预测采样不等于精确反馈时间。')
    if cause_counts.get('not_recorded'):
        limitations.append(
            f"{cause_counts['not_recorded']} 条 ACK 未记录 retirement 原因；"
            '后续请求按前一请求末次真实 ACK 与当前观测源时刻关联。')
    limitations.extend([
        '客户端预测总耗时减去服务端推理耗时仍包含传输及其他处理，不能归因于单一环节。',
        'ServoJ RPC 返回码和夹爪完成反馈不证明物体、轨迹或策略任务结果。',
    ])
    return {
        'schema': 'fr5-jev-pi05-live-timing-v1', 'episode': str(episode),
        'clock': {'durations': 'seconds', 'monotonic': 'host_monotonic',
                  'wall_time': 'UTC+08:00, estimated from first wire timestamp pair',
                  'percentiles': 'p50 median; p90/p95/p99 nearest-rank'},
        'counts': {'chunks': len(chunks), 'prediction_records': len(journal),
                   'submitted_targets': len(target_rows), 'acknowledged_targets': len(acknowledgements),
                   'unacknowledged_targets_at_stop': len(target_rows) - len(acknowledged_rows),
                   'wire_records': len(wire), 'servoj': servo['wire_method_counts'].get('ServoJ', 0),
                   'move_gripper': len(gripper_rows),
                   'gripper_completion_feedback_exact': exact_feedback_count},
        'termination': {key: summary.get(key) for key in (
            'reason', 'termination_source', 'error', 'failed', 'journal_error',
            'summary_stage', 'wire_callbacks_drained', 'outcome', 'task_success_verified')},
        'unacknowledged_tail_target': (
            {**target_rows[-1], 'termination_source': summary['termination_source'],
             'ordered_pending': snapshot['ordered_pending']}
            if len(acknowledged_rows) != len(target_rows) else None),
        'definitions': {
            'server_inference_s': 'reply.inference_seconds reported by the inference server',
            'client_predict_minus_server_inference_s': 'client predict interval minus server inference; includes transport and other work',
            'source_age': 'host monotonic event time minus request.source_time',
            'scheduled_target_lateness_s': 'target_monotonic minus predicted_source_target_monotonic',
            'submit_to_ack_s': 'logged acknowledgement minus logged submission; includes joint endpoint and any gripper completion',
            'rpc_return_to_ack_s': 'logged acknowledgement minus MoveGripper RPC return; not the exact feedback duration',
            'surrounding_servoj_gap_s': 'next ServoJ issue minus previous ServoJ return; an upper bound on the contained peripheral lock pause',
            'completion_detection_bounds': 'RPC return to the earliest available next ServoJ issue or target ACK; null when neither upper bound is recorded',
            'prediction_observation': 'current request state at request.source_time; invalid current state stays invalid',
            'observation_after_monotonic': 'previous request retirement ACK used as the lower bound for the next current observation',
            'prefix_consumed_steps': 'submitted cursor and ACK cursor are recorded separately; only real ACKs contribute retirement causes',
            'completion_to_next_s_bounds': 'next request/source/observation-ready time minus completion upper/lower bounds',
        },
        'prediction_distributions_s': {key: distribution(item[key] for item in prediction_rows)
                                       for key in prediction_fields},
        'target_distributions_s': {key: distribution(item[key] for item in target_rows if item[key] is not None)
                                   for key in target_fields},
        'target_ack_by_gripper_change_s': {
            'with_move_gripper': distribution(item['submit_to_ack_s'] for item in acknowledged_rows
                                               if item['sequence'] in gripper_sequences),
            'without_move_gripper': distribution(item['submit_to_ack_s'] for item in acknowledged_rows
                                                  if item['sequence'] not in gripper_sequences)},
        'prefix_consumed_steps': {
            'submitted': dict(Counter(str(item['prefix_consumed_steps'])
                                      if item['prefix_consumed_steps'] is not None else 'not_recorded'
                                      for item in target_rows)),
            'acknowledged': dict(Counter(str(item['ack_prefix_consumed_steps'])
                                         if item['ack_prefix_consumed_steps'] is not None else 'not_recorded'
                                         for item in acknowledged_rows)),
        },
        'prefix_retirement': prefix_retirement,
        'prefix_retirements': retirement_rows,
        'gripper_distributions_s': gripper_distributions,
        'predictions': prediction_rows, 'targets': target_rows,
        'move_gripper': gripper_rows, 'servo': servo, 'final_lateness_stop': failure,
        'limitations': limitations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.episode.resolve())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'counts': result['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
