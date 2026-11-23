#!/usr/bin/env python3
"""Audit archived cup observation, prediction, submission and TCP feedback times."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess


TRIAL = 'pi05-cup-hover-speed-roll64-trial02-20261115'
AUDIT_NAME = 'pi05-cup-trial02-action-time-audit-20261117-v1'
SERVER_ROOT = Path('/home/data4/hts/demo1/fr5-learning')
PICKUP_MM = (159.7, 71.3, 95.8)
TIMING_KEYS = ('requested_monotonic', 'observation_ready_monotonic',
               'predict_started_monotonic', 'received_monotonic')


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def vector(value, size):
    return (list(value) if isinstance(value, (list, tuple)) and len(value) == size
            and all(number(x) is not None for x in value) else None)


def index_unique(rows, key, label):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError(f'Duplicate {label}: {identity}')
        result[identity] = row
    return result


def pair_events(chunks, predictions, submissions, acknowledgements):
    """Join requests by id and delivery events by (sequence, token)."""
    by_request = index_unique(chunks, lambda r: r['request']['id'], 'chunk request')
    by_prediction = index_unique(predictions, lambda r: r['request']['id'], 'prediction request')
    if by_request.keys() != by_prediction.keys():
        raise ValueError('Chunk and prediction request ids differ')
    by_submission = index_unique(submissions, lambda r: (r['sequence'], r['delivery_token']),
                                 'submission')
    index_unique(submissions, lambda r: r['sequence'], 'submission sequence')
    index_unique(submissions, lambda r: r['delivery_token'], 'delivery token')
    by_ack = index_unique(acknowledgements, lambda r: (r['sequence'], r['delivery_token']),
                          'acknowledgement')
    if by_ack.keys() - by_submission.keys():
        raise ValueError('An acknowledgement has no matching submission sequence and token')
    ordered = sorted(submissions, key=lambda r: r['sequence'])
    events = {request_id: [] for request_id in by_request}
    for i, submission in enumerate(ordered):
        request_id = submission['request_id']
        if request_id not in by_request:
            raise ValueError(f'Submission has no prediction request: {request_id}')
        ack = by_ack.get((submission['sequence'], submission['delivery_token']))
        if ack is not None and ack['acknowledged_monotonic'] < submission['submitted_monotonic']:
            raise ValueError('Acknowledgement precedes its submission')
        events[request_id].append({'submission': submission, 'ack': ack,
                                  'next_submit_monotonic': ordered[i + 1]['submitted_monotonic']
                                  if i + 1 < len(ordered) else None})
    groups = []
    for request_id, chunk in sorted(by_request.items(), key=lambda item: item[1]['_chunk_index']):
        prediction = by_prediction[request_id]
        if chunk['reply']['id'] != request_id or prediction['reply']['id'] != request_id:
            raise ValueError('Reply does not match its request id')
        if prediction['request']['source_time'] != chunk['request']['source_time']:
            raise ValueError('Paired request camera source times differ')
        for key in TIMING_KEYS:
            recorded = prediction.get('execution_metadata', {}).get(key)
            if recorded is not None and recorded != chunk.get(key):
                raise ValueError(f'Paired request timing differs: {key}')
        prefix = chunk['execution_prefix_steps']
        group_events = events[request_id]
        indices = [event['submission']['action_index'] for event in group_events]
        if len(set(indices)) != len(indices) or any(not 0 <= i < prefix for i in indices):
            raise ValueError('Submitted action indices do not identify a unique execution prefix')
        for event in group_events:
            submission = event['submission']
            action_index = submission['action_index']
            actions = chunk['reply'].get('actions', [])
            if (action_index >= len(actions) or vector(actions[action_index], 7) is None
                    or submission.get('raw_action') != actions[action_index]):
                raise ValueError('Submitted raw_action differs from the paired model reply action')
        groups.append({'chunk_index': chunk['_chunk_index'], 'chunk': chunk,
                       'prediction': prediction, 'events': group_events,
                       'complete': indices == list(range(prefix))
                       and all(event['ack'] is not None for event in group_events)})
    return groups


class FeedbackTimeline:
    """Select recorded TCP packets using their host receipt times."""

    def __init__(self, captures):
        self.rows = sorted((row for row in captures
                            if number(row.get('robot', {}).get('received_monotonic')) is not None),
                           key=lambda row: row['robot']['received_monotonic'])
        self.times = [row['robot']['received_monotonic'] for row in self.rows]
        self.camera = index_unique(captures, lambda row: row['camera']['source_timestamp_monotonic'],
                                   'camera source timestamp')

    def latest_before(self, stamp):
        if stamp is None:
            return None
        index = bisect_left(self.times, stamp) - 1
        return self.rows[index] if index >= 0 else None

    def first_after(self, stamp, *, before=None):
        if stamp is None:
            return None
        index = bisect_right(self.times, stamp)
        if index == len(self.rows) or (before is not None and self.times[index] >= before):
            return None
        return self.rows[index]


def relative_feedback(frame, cup_origin):
    result = {'frame_index': None, 'evidence': None, 'camera_source_monotonic': None,
              'received_monotonic': None, 'tcp_base_xyz_mm': None,
              'cup_relative_xyz_mm': None, 'reason': None}
    if frame is None:
        result['reason'] = 'no_recorded_feedback_in_time_window'
        return result
    robot = frame.get('robot', {})
    result.update(frame_index=frame.get('frame_index'), evidence=frame.get('_evidence'),
                  camera_source_monotonic=frame['camera'].get('source_timestamp_monotonic'),
                  received_monotonic=number(robot.get('received_monotonic')))
    origin, tcp, flange = vector(cup_origin, 3), vector(robot.get('tcp'), 6), vector(robot.get('flange'), 6)
    if frame.get('sync_clock') != 'host_monotonic':
        result['reason'] = 'feedback_clock_is_not_host_monotonic'
    elif result['received_monotonic'] is None or tcp is None:
        result['reason'] = 'missing_recorded_tcp_or_receipt_time'
    elif origin is None:
        result['reason'] = 'missing_fixed_cup_origin'
    elif robot.get('tool') != 0 or robot.get('user') != 0:
        result['reason'] = 'tcp_coordinate_basis_is_not_tool0_user0'
    elif flange is None or math.dist(tcp[:3], flange[:3]) > 1.:
        result['reason'] = 'recorded_tcp_does_not_establish_the_flange_reference'
    else:
        result.update(tcp_base_xyz_mm=tcp[:3],
                      cup_relative_xyz_mm=[x - y for x, y in zip(tcp[:3], origin)])
    return result


def progress_mm(start, target, reference):
    if start is None or target is None:
        return None
    return math.dist(start, reference) - math.dist(target, reference)


def direction(value):
    if value is None:
        return None
    if value < 0.:
        return 'negative_under_1mm' if value > -1. else 'negative_at_least_1mm'
    return 'positive' if value > 0. else 'zero'


def compare_progress(camera_xyz, target_xyz, pre_submit_xyz, *, reference_mm=PICKUP_MM):
    camera = progress_mm(camera_xyz, target_xyz, reference_mm)
    pre_submit = progress_mm(pre_submit_xyz, target_xyz, reference_mm)
    return {'camera_anchor_progress_mm': camera, 'pre_submit_progress_mm': pre_submit,
            'camera_anchor_direction': direction(camera), 'pre_submit_direction': direction(pre_submit),
            'positive_at_camera_negative_at_submit': camera > 0. and pre_submit < 0.
            if camera is not None and pre_submit is not None else None,
            'state_change_since_camera_mm': math.dist(camera_xyz, pre_submit_xyz)
            if camera_xyz is not None and pre_submit_xyz is not None else None,
            'state_progress_since_camera_mm': progress_mm(camera_xyz, pre_submit_xyz, reference_mm)}


def timing_breakdown(group, source_frame, previous_last_ack=None):
    chunk, events = group['chunk'], group['events']
    first = events[0]['submission']['submitted_monotonic'] if events else None
    last_ack = events[-1]['ack']['acknowledged_monotonic'] if events and events[-1]['ack'] else None
    camera = source_frame['camera'] if source_frame else {}
    source = chunk['request']['source_time'] if chunk['request'].get('source_clock') == 'host_monotonic' else None
    times = {'camera_source_monotonic': source,
             'camera_packet_received_monotonic': number(camera.get('received_monotonic')),
             'source_feedback_received_monotonic': number(source_frame['robot'].get('received_monotonic'))
             if source_frame else None,
             **{key: number(chunk.get(key)) for key in TIMING_KEYS},
             'first_submit_monotonic': first, 'last_ack_monotonic': last_ack,
             'previous_last_ack_monotonic': previous_last_ack}
    durations, unavailable = {}, {}

    def elapsed(name, begin, end):
        if begin is None or end is None:
            durations[name] = None
            unavailable[name] = 'missing_recorded_timestamp'
        elif end < begin:
            durations[name] = None
            unavailable[name] = 'recorded_interval_has_reversed_time_order'
        else:
            durations[name] = end - begin

    requested, ready, started, received = (times[key] for key in TIMING_KEYS)
    elapsed('request_to_observation_ready_s', requested, ready)
    elapsed('camera_to_packet_received_s', source, times['camera_packet_received_monotonic'])
    elapsed('camera_to_observation_ready_s', source, ready)
    elapsed('observation_ready_to_predict_start_s', ready, started)
    elapsed('client_prediction_call_s', started, received)
    inference = number(chunk['reply'].get('inference_seconds'))
    durations['server_model_inference_s'] = inference
    if inference is None:
        unavailable['server_model_inference_s'] = 'missing_recorded_server_inference_duration'
    elapsed('client_prediction_nonmodel_s', inference, durations['client_prediction_call_s'])
    elapsed('result_to_first_submit_s', received, first)
    elapsed('camera_to_first_submit_s', source, first)
    elapsed('first_submit_to_last_ack_s', first, last_ack)
    elapsed('previous_last_ack_to_first_submit_s', previous_last_ack, first)
    elapsed('previous_last_ack_to_request_s', previous_last_ack, requested)
    return {'timestamps': times, 'durations_s': durations, 'unavailable': unavailable}


def wire_evidence(row):
    return ({key: row.get(key) for key in ('sequence', 'issued_monotonic', 'returned_monotonic',
                                          'return_code', 'error', '_evidence')}
            if row is not None else None)


def wire_window(event, wire_rows, wire_times):
    submission, ack = event['submission'], event['ack']
    if ack is None:
        return {'first_servoj_in_window': None, 'first_matching_endpoint': None,
                'endpoint_candidate_count': 0, 'reason': 'missing_matching_acknowledgement'}
    begin, end = submission['submitted_monotonic'], ack['acknowledged_monotonic']
    window = wire_rows[bisect_left(wire_times, begin):bisect_right(wire_times, end)]
    joints = vector(submission.get('decoded', {}).get('joint_degrees'), 6)
    candidates = []
    for row in window:
        args = row.get('raw_args', [])
        actual = vector(args[0], 6) if args else None
        returned = number(row.get('returned_monotonic'))
        if (joints is not None and actual is not None and returned is not None and returned <= end
                and row.get('return_code') == 0 and row.get('error') is None
                and max(abs(x - y) for x, y in zip(joints, actual)) <= 1e-6):
            candidates.append(row)
    return {'first_servoj_in_window': wire_evidence(window[0]) if window else None,
            'first_matching_endpoint': wire_evidence(candidates[0]) if candidates else None,
            'endpoint_candidate_count': len(candidates),
            'reason': None if candidates else 'no_successful_matching_endpoint_rpc_in_submit_ack_window'}


def hold_evidence(previous_event, until, timeline, wire_rows, wire_times, cup_origin):
    if previous_event is None or previous_event['ack'] is None or until is None:
        return None
    began = previous_event['ack']['acknowledged_monotonic']
    expected = vector(previous_event['submission'].get('decoded', {}).get('joint_degrees'), 6)
    window = wire_rows[bisect_right(wire_times, began):bisect_left(wire_times, until)]
    different = []
    for row in window:
        args = row.get('raw_args', [])
        actual = vector(args[0], 6) if args else None
        if (expected is not None and actual is not None
                and max(abs(x - y) for x, y in zip(expected, actual)) > 1e-6):
            different.append(row)
    samples = [relative_feedback(row, cup_origin) for row in
               timeline.rows[bisect_right(timeline.times, began):bisect_left(timeline.times, until)]]
    valid = [sample for sample in samples if sample['cup_relative_xyz_mm'] is not None]
    first, last = (valid[0], valid[-1]) if valid else (None, None)
    measured = len(valid) >= 2
    return {'began_monotonic': began, 'ended_monotonic': until, 'duration_s': until - began,
            'previous_submission': previous_event['submission'].get('_evidence'),
            'previous_acknowledgement': previous_event['ack'].get('_evidence'),
            'wire_servoj_count': len(window),
            'wire_not_previous_endpoint_count': len(different) if expected is not None and window else None,
            'first_different_wire_event': wire_evidence(different[0]) if different else None,
            'feedback_count': len(samples), 'usable_feedback_count': len(valid),
            'first_feedback': first, 'last_feedback': last,
            'feedback_span_s': last['received_monotonic'] - first['received_monotonic'] if measured else None,
            'measured_displacement_mm': math.dist(first['cup_relative_xyz_mm'], last['cup_relative_xyz_mm'])
            if measured else None,
            'measured_max_excursion_mm': max(math.dist(first['cup_relative_xyz_mm'], sample['cup_relative_xyz_mm'])
                                             for sample in valid) if measured else None,
            'measurement_reason': None if measured else 'fewer_than_two_usable_feedback_samples_during_hold'}


def numeric_summary(values, *, total=False):
    values = [value for value in values if value is not None]
    result = {'count': len(values), 'median': statistics.median(values) if values else None,
              'min': min(values) if values else None, 'max': max(values) if values else None}
    if total:
        result['sum'] = sum(values) if values else None
    return result


def direction_case(row):
    return {key: row[key] for key in ('chunk_index', 'request_id', 'sequence', 'action_index',
                                      'model_target_source', 'camera_anchor_xyz_mm', 'model_target_xyz_mm',
                                      'submitted_monotonic', 'observation_age_at_submit_s',
                                      'pre_submit_feedback', 'pre_submit_feedback_age_s', 'comparison',
                                      'target_at_chunk_first_submit', 'sign_change_window')}


def direction_counts(rows):
    compared = [row for row in rows if row['positive_at_camera_negative_at_submit'] is not None]
    return {'count': len(rows), 'compared_count': len(compared),
            'unassessed_count': len(rows) - len(compared),
            'positive_at_camera_negative_at_submit_count': sum(
                row['positive_at_camera_negative_at_submit'] for row in compared),
            'camera_anchor_directions': dict(Counter(row['camera_anchor_direction'] for row in rows)),
            'pre_submit_directions': dict(Counter(row['pre_submit_direction'] for row in rows)),
            'camera_anchor_progress_mm': numeric_summary(row['camera_anchor_progress_mm'] for row in rows),
            'pre_submit_progress_mm': numeric_summary(row['pre_submit_progress_mm'] for row in rows)}


def audit_records(chunks, predictions, submissions, acknowledgements, captures, wire, scope,
                  *, reference_mm=PICKUP_MM):
    groups = pair_events(chunks, predictions, submissions, acknowledgements)
    timeline = FeedbackTimeline(captures)
    servo = sorted((row for row in wire if row['method'] == 'ServoJ'),
                   key=lambda row: row['issued_monotonic'])
    servo_times = [row['issued_monotonic'] for row in servo]
    stop = min((row['issued_monotonic'] for row in wire if row['method'] == 'StopMotion'), default=None)
    cup = scope.get('initial_cup', {})
    origin = vector(cup.get('base_xyz_mm'), 3)
    chunk_rows, target_rows, previous_ack, previous_event = [], [], None, None
    for group in groups:
        chunk, request = group['chunk'], group['chunk']['request']
        source_frame = timeline.camera.get(request['source_time'])
        basis_reason = None
        if origin is None or request.get('cup_base_xyz_mm') != origin:
            basis_reason = 'request_does_not_establish_the_fixed_cup_origin'
        elif chunk['reply'].get('wire_action_representation') != 'cup-relative-tcp-v1':
            basis_reason = 'recorded_targets_do_not_establish_cup_relative_tcp_coordinates'
        elif request.get('source_clock') != 'host_monotonic':
            basis_reason = 'request_clock_is_not_host_monotonic'
        elif source_frame and source_frame['camera'].get('source_camera_serial') != cup.get('camera_serial'):
            basis_reason = 'camera_source_does_not_match_the_fixed_cup_frame'
        state = vector(request.get('cup_relative_state'), 8)
        camera_xyz = [x * 1000. for x in state[:3]] if state is not None and basis_reason is None else None
        first_time = group['events'][0]['submission']['submitted_monotonic'] if group['events'] else None
        first_pre = relative_feedback(timeline.latest_before(first_time), origin)
        if basis_reason:
            first_pre.update(cup_relative_xyz_mm=None, reason=basis_reason)
        local_targets = []
        for event in group['events']:
            submission, ack = event['submission'], event['ack']
            submit_time = submission['submitted_monotonic']
            ack_time = ack['acknowledged_monotonic'] if ack else None
            raw = vector(chunk['reply']['actions'][submission['action_index']], 7)
            target_xyz = [x * 1000. for x in raw[:3]] if raw is not None and basis_reason is None else None
            pre = relative_feedback(timeline.latest_before(submit_time), origin)
            upper = event['next_submit_monotonic'] if event['next_submit_monotonic'] is not None else stop
            after = relative_feedback(timeline.first_after(ack_time, before=upper), origin)
            if basis_reason:
                for feedback in (pre, after):
                    feedback.update(cup_relative_xyz_mm=None, reason=basis_reason)
            comparison = compare_progress(camera_xyz, target_xyz, pre['cup_relative_xyz_mm'],
                                          reference_mm=reference_mm)
            first_comparison = compare_progress(camera_xyz, target_xyz, first_pre['cup_relative_xyz_mm'],
                                                reference_mm=reference_mm)
            sign_window = None
            if comparison['positive_at_camera_negative_at_submit']:
                first_progress = first_comparison['pre_submit_progress_mm']
                if first_progress is not None:
                    sign_window = ('camera_to_chunk_first_submit' if first_progress < 0.
                                   else 'within_executed_prefix' if first_progress > 0. else 'zero_at_first_submit')
            wire_result = wire_window(event, servo, servo_times)
            endpoint = wire_result['first_matching_endpoint']
            measured = after['cup_relative_xyz_mm']
            row = {'chunk_index': group['chunk_index'], 'request_id': request['id'],
                   'sequence': submission['sequence'], 'delivery_token': submission['delivery_token'],
                   'action_index': submission['action_index'],
                   'evidence': {'submission': submission.get('_evidence'),
                                'acknowledgement': ack.get('_evidence') if ack else None},
                   'model_target_source': {'chunk_reply': chunk.get('_evidence'),
                                           'request_id': request['id'],
                                           'action_index': submission['action_index'],
                                           'field': 'reply.actions[action_index]',
                                           'submission_raw_action_matches_reply': True},
                   'camera_anchor_xyz_mm': camera_xyz, 'model_target_xyz_mm': target_xyz,
                   'coordinate_reason': basis_reason,
                   'submitted_monotonic': submit_time, 'acknowledged_monotonic': ack_time,
                   'observation_age_at_submit_s': submit_time - request['source_time']
                   if request.get('source_clock') == 'host_monotonic' else None,
                   'submit_to_ack_s': ack_time - submit_time if ack_time is not None else None,
                   'pre_submit_feedback': pre,
                   'pre_submit_feedback_age_s': submit_time - pre['received_monotonic']
                   if pre['received_monotonic'] is not None else None,
                   'comparison': comparison, 'target_at_chunk_first_submit': first_comparison,
                   'sign_change_window': sign_window, 'wire': wire_result,
                   'submit_to_endpoint_rpc_return_s': endpoint['returned_monotonic'] - submit_time
                   if endpoint else None,
                   'endpoint_rpc_return_to_ack_s': ack_time - endpoint['returned_monotonic']
                   if endpoint else None,
                   'post_ack_feedback': after,
                   'feedback_after_ack_s': after['received_monotonic'] - ack_time
                   if after['received_monotonic'] is not None and ack_time is not None else None,
                   'measured_progress_from_camera_mm': progress_mm(camera_xyz, measured, reference_mm),
                   'measured_progress_from_pre_submit_mm': progress_mm(pre['cup_relative_xyz_mm'], measured,
                                                                        reference_mm),
                   'post_ack_endpoint_error_mm': math.dist(target_xyz, measured)
                   if target_xyz is not None and measured is not None else None,
                   'pre_to_post_feedback_interval_s': after['received_monotonic'] - pre['received_monotonic']
                   if after['received_monotonic'] is not None and pre['received_monotonic'] is not None else None,
                   'measured_pre_to_post_displacement_mm': math.dist(pre['cup_relative_xyz_mm'], measured)
                   if pre['cup_relative_xyz_mm'] is not None and measured is not None else None}
            local_targets.append(row)
            target_rows.append(row)
        timing = timing_breakdown(group, source_frame, previous_ack)
        hold = hold_evidence(previous_event, first_time if first_time is not None else stop,
                             timeline, servo, servo_times, origin)
        row = {'chunk_index': group['chunk_index'], 'request_id': request['id'],
               'recorded_status': chunk.get('status'), 'complete_executed_prefix': group['complete'],
               'submitted_count': len(local_targets),
               'evidence': {'chunk': chunk.get('_evidence'),
                            'prediction': group['prediction'].get('_evidence'),
                            'source_capture': source_frame.get('_evidence') if source_frame else None},
               'source_frame_index': source_frame.get('frame_index') if source_frame else None,
               'camera_anchor_xyz_mm': camera_xyz, 'coordinate_reason': basis_reason,
               'timing': timing, 'preceding_hold': hold,
               'hold_kind': 'inter_block_hold' if first_time is not None and hold is not None
               else 'terminal_hold_until_stop' if hold is not None else None,
               'recorded_proposal_draws': chunk['reply'].get('cup_proposal_selection', {}).get('draws', 1),
               'recorded_draw_durations_s': [proposal.get('inference_seconds') for proposal in
                                            chunk['reply'].get('cup_proposal_selection', {}).get('proposals', [])],
               'endpoint_at_first_submit': None, 'endpoint_at_own_submit': None}
        if group['complete']:
            last = local_targets[-1]
            row['endpoint_at_first_submit'] = {
                'pre_submit_feedback': first_pre,
                'pre_submit_feedback_age_s': local_targets[0]['pre_submit_feedback_age_s'],
                **compare_progress(camera_xyz, last['model_target_xyz_mm'], first_pre['cup_relative_xyz_mm'],
                                   reference_mm=reference_mm)}
            row['endpoint_at_own_submit'] = last
        chunk_rows.append(row)
        if group['events']:
            last_ack = group['events'][-1]['ack']
            previous_ack = last_ack['acknowledged_monotonic'] if last_ack else None
            previous_event = group['events'][-1]
    complete = [row for row in chunk_rows if row['complete_executed_prefix']]
    first_negative = next((row for row in complete if row['endpoint_at_first_submit']
                           ['camera_anchor_progress_mm'] is not None and row['endpoint_at_first_submit']
                           ['camera_anchor_progress_mm'] < 0.), None)
    first_submit = min((row['submitted_monotonic'] for row in target_rows), default=None)
    final_ack = max((row['acknowledged_monotonic'] for row in target_rows
                     if row['acknowledged_monotonic'] is not None), default=None)
    span = final_ack - first_submit if final_ack is not None and first_submit is not None else None
    acknowledged_count = sum(row['acknowledged_monotonic'] is not None for row in target_rows)
    inter_holds = [row['preceding_hold'] for row in chunk_rows if row['hold_kind'] == 'inter_block_hold']
    flips = [row for row in target_rows if row['comparison']['positive_at_camera_negative_at_submit']]
    negative = [row for row in complete if row['endpoint_at_first_submit']['pre_submit_progress_mm'] is not None
                and row['endpoint_at_first_submit']['pre_submit_progress_mm'] < 0.]
    largest_negative = min(negative, key=lambda row: row['endpoint_at_first_submit']['pre_submit_progress_mm'],
                           default=None)
    largest_negative = ({'chunk_index': largest_negative['chunk_index'],
                         'endpoint_at_first_submit': largest_negative['endpoint_at_first_submit'],
                         'endpoint_at_own_submit': largest_negative['endpoint_at_own_submit']['comparison'],
                         'model_target_source': largest_negative['endpoint_at_own_submit']['model_target_source'],
                         'measured_progress_from_camera_mm': largest_negative['endpoint_at_own_submit']
                         ['measured_progress_from_camera_mm'],
                         'post_ack_endpoint_error_mm': largest_negative['endpoint_at_own_submit']
                         ['post_ack_endpoint_error_mm']} if negative else None)
    summary = {'recorded_prediction_chunks': len(groups), 'complete_executed_chunks': len(complete),
               'unexecuted_chunks': sum(row['submitted_count'] == 0 for row in chunk_rows),
               'partial_executed_chunks': sum(row['submitted_count'] > 0 and not row['complete_executed_prefix']
                                              for row in chunk_rows),
               'submitted_targets': len(target_rows), 'acknowledged_targets': acknowledged_count,
               'submission_raw_actions_matching_model_reply': len(target_rows),
               'recorded_feedback_frames': len(captures),
               'recorded_trial_move_gripper_commands': sum(row['method'] == 'MoveGripper' for row in wire),
               'timing_s': {key: numeric_summary((row['timing']['durations_s'][key] for row in complete), total=True)
                            for key in complete[0]['timing']['durations_s']} if complete else {},
               'ack_endpoint_throughput_hz': acknowledged_count / span if span and span > 0. else None,
               'first_submit_to_final_ack_s': span,
               'all_submitted_targets': direction_counts([row['comparison'] for row in target_rows]),
               'chunk_endpoint_at_first_submit': direction_counts([row['endpoint_at_first_submit']
                                                                  for row in complete]),
               'chunk_endpoint_at_own_submit': direction_counts([row['endpoint_at_own_submit']['comparison']
                                                                for row in complete]),
               'sign_change_cases': [direction_case(row) for row in flips],
               'sign_change_windows': dict(Counter(row['sign_change_window'] for row in flips)),
               'sign_change_negative_at_least_1mm_count': sum(row['comparison']['pre_submit_progress_mm'] <= -1.
                                                             for row in flips),
               'latency_cause_established': None,
               'latency_cause_reason': 'Archived state changes identify numerical sign changes; '
               'they do not separate latency from residual motion, within-prefix motion and feedback uncertainty.',
               'physical_wrong_direction_established': None,
               'physical_wrong_direction_reason': 'Fixed-reference distance and submillimetre sign changes '
               'do not provide independent physical task ground truth.',
               'largest_negative_chunk_endpoint': largest_negative,
               'inter_block_holds': {'count': len(inter_holds),
                                    'wire_servoj_count': sum(row['wire_servoj_count'] for row in inter_holds),
                                    'wire_not_previous_endpoint_count': sum(row['wire_not_previous_endpoint_count']
                                        for row in inter_holds) if all(row['wire_not_previous_endpoint_count'] is not None
                                        for row in inter_holds) else None,
                                    'measured_displacement_mm': numeric_summary(row['measured_displacement_mm']
                                                                                for row in inter_holds),
                                    'measured_max_excursion_mm': numeric_summary(row['measured_max_excursion_mm']
                                                                                 for row in inter_holds)},
               'terminal_hold_until_stop': next((row['preceding_hold'] for row in chunk_rows
                                                 if row['hold_kind'] == 'terminal_hold_until_stop'), None),
               'pre_submit_feedback_age_s': numeric_summary(row['pre_submit_feedback_age_s'] for row in target_rows),
               'submit_to_ack_s': numeric_summary((row['submit_to_ack_s'] for row in target_rows), total=True),
               'matching_wire_endpoints': sum(row['wire']['first_matching_endpoint'] is not None for row in target_rows),
               'chunk_endpoint_post_ack_error_mm': numeric_summary(
                   row['endpoint_at_own_submit']['post_ack_endpoint_error_mm'] for row in complete),
               'post_ack_feedback_available_targets': sum(row['post_ack_feedback']['cup_relative_xyz_mm'] is not None
                                                         for row in target_rows),
               'post_ack_execution_error_unassessed_targets': sum(row['post_ack_endpoint_error_mm'] is None
                                                                 for row in target_rows),
               'first_predicted_negative_progress': first_negative}
    return {'schema': 'fr5-cup-trial-action-time-audit-v1', 'dataset_role': 'evaluation_only',
            'robot_commands': 0, 'model_inferences': 0, 'task_success_verified': False,
            'physical_successes': 0,
            'demonstrations': {'user_confirmed_count': 8, 'train_sources': list(range(7)),
                               'holdout_sources': [7], 'failed_trials': 'evaluation_only'},
            'fixed_reference_cup_relative_xyz_mm': list(reference_mm),
            'summary': summary, 'chunks': chunk_rows, 'targets': target_rows,
            'limitations': [
                'TCP feedback is the last archived sample strictly before submission, not every native packet; '
                'receipt time and sample age define its availability. No future feedback is interpolated.',
                'submitted_monotonic is sampled immediately before submit_ordered. ServoJ windows identify '
                'endpoint candidates by time and recorded joint values; wire records have no delivery token.',
                'Client prediction time includes RPC and reply validation. Server inference_seconds sums all '
                'recorded draws. Their difference includes all work outside that timer.',
                'Submit-to-ACK and inter-block holds use host timestamps. ACK indicates endpoint RPC return; '
                'physical change is separately measured by the first archived feedback after ACK and before '
                'the next submission or StopMotion. These samples do not determine exact physical settle time.',
                'The fixed demonstration pickup is expressed in the initial cup frame, with tool0 TCP matching '
                'the flange. Submillimetre negative progress is numerical direction, not established physical '
                'error. Timing and direction comparisons alone do not establish a latency cause.']}


def read_json(path):
    row = json.loads(path.read_text())
    row['_evidence'] = {'path': str(path)}
    return row


def parse_jsonl(data, path, host=None):
    rows = []
    for line, text in enumerate(data.splitlines(), 1):
        if text.strip():
            row = json.loads(text)
            row['_evidence'] = {'path': str(path), 'line': line, 'host': host}
            rows.append(row)
    return rows


def ssh_run(host, command, *, data=None):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, command],
                          input=data, text=True, capture_output=True, check=True, timeout=60).stdout


def write_remote_json(host, path, report):
    writer = ('import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
              'p.parent.mkdir(parents=True,exist_ok=True); '
              'f=p.open("x"); f.write(sys.stdin.read()); f.close()')
    command = 'python3 -B -c ' + shlex.quote(writer) + ' ' + shlex.quote(str(path))
    ssh_run(host, command, data=json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-root', type=Path,
                        default=Path('/home/dianrobot/.local/state/fr5-recording-spool') / TRIAL)
    parser.add_argument('--ssh-host', default='4090')
    parser.add_argument('--capture-jsonl', type=Path,
                        default=SERVER_ROOT / 'recording-spool' / TRIAL / 'materialized/continuous/capture.jsonl')
    parser.add_argument('--detail-dir', type=Path, default=SERVER_ROOT / 'audits' / AUDIT_NAME)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results' /
                        (AUDIT_NAME + '.json'))
    args = parser.parse_args()
    if not args.detail_dir.is_relative_to(SERVER_ROOT / 'audits'):
        raise ValueError('Detailed audit output must be inside the server audits directory')
    if args.output.exists():
        raise FileExistsError(args.output)
    scope = read_json(args.trial_root / 'scope.json')
    original_summary = read_json(args.trial_root / 'summary.json')
    review = read_json(args.trial_root / 'user-review.json')
    chunks = []
    for path in sorted(args.trial_root.glob('chunk-*.json')):
        row = read_json(path)
        row['_chunk_index'] = int(path.stem.split('-')[1])
        chunks.append(row)
    predictions, wire = [], []
    for path in sorted((args.trial_root / 'stream-prediction').glob('*.jsonl')):
        predictions.extend(parse_jsonl(path.read_text(), path))
    for path in sorted((args.trial_root / 'stream-wire').glob('*.jsonl')):
        wire.extend(parse_jsonl(path.read_text(), path))
    submitted_path, ack_path = (args.trial_root / name for name in
                                ('submitted-targets.jsonl', 'acknowledged-targets.jsonl'))
    submissions = parse_jsonl(submitted_path.read_text(), submitted_path)
    acknowledgements = parse_jsonl(ack_path.read_text(), ack_path)
    captured = ssh_run(args.ssh_host, 'cat -- ' + shlex.quote(str(args.capture_jsonl)))
    captures = parse_jsonl(captured, args.capture_jsonl, args.ssh_host)
    report = audit_records(chunks, predictions, submissions, acknowledgements, captures, wire, scope)
    report.update(created_at_utc=datetime.now(timezone.utc).isoformat(), trial=TRIAL,
                  trial_root=str(args.trial_root), original_trial_verdict=review.get('verdict'),
                  original_trial_task_success_verified=original_summary.get('task_success_verified'),
                  sources={'trial_root': str(args.trial_root), 'capture_jsonl': str(args.capture_jsonl),
                           'capture_host': args.ssh_host,
                           'reference': {'path': str(args.trial_root / 'runtime-source/cup_timing.py'),
                                         'symbol': 'PICKUP_FLANGE_FROM_CUP_MM'},
                           'coordinate_conversion': str(args.trial_root / 'runtime-source/cup_relative_runtime.py'),
                           'request_timing': str(args.trial_root / 'runtime-source/stream_runtime.py'),
                           'model_draw_timing': str(args.trial_root / 'runtime-source/cup_action_selection.py')})
    detail_path = args.detail_dir / 'action-time-detail.json'
    report['detail_artifact'] = {'host': args.ssh_host, 'path': str(detail_path)}
    write_remote_json(args.ssh_host, detail_path, report)
    small = {key: value for key, value in report.items() if key not in ('chunks', 'targets')}
    with args.output.open('x') as output:
        output.write(json.dumps(small, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'detail_path': str(detail_path),
                      'summary': report['summary'], 'robot_commands': 0, 'model_inferences': 0},
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
