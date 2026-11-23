"""Test archived action timing and causal TCP feedback selection."""

import copy
import unittest

from fr5_lerobot.diagnostics import audit_cup_trial_action_time as audit


def capture(camera_time, feedback_time, xyz, index=0, *, tool=0, user=0):
    pose = [*xyz, 0., 0., 0.]
    return {'frame_index': index, 'sync_clock': 'host_monotonic',
            'camera': {'source_timestamp_monotonic': camera_time,
                       'received_monotonic': camera_time + .1,
                       'source_camera_serial': 'recorded-camera'},
            'robot': {'received_monotonic': feedback_time, 'tcp': pose,
                      'flange': pose[:], 'tool': tool, 'user': user}}


def records():
    chunks, predictions, submissions, acknowledgements = [], [], [], []
    for index, request_id, offset in [(0, 'a', 0.), (1, 'b', 3.)]:
        request = {'id': request_id, 'source_clock': 'host_monotonic',
                   'source_time': 10.2 + offset, 'cup_base_xyz_mm': [0., 0., 0.],
                   'cup_relative_state': [.01, 0., 0., 0., 0., 0., 1., 1.]}
        actions = [[.009 - i * .001, 0., 0., 0., 0., 0., 1.] for i in range(3)]
        chunk = {'_chunk_index': index, 'request': request,
                 'reply': {'id': request_id, 'actions': actions,
                           'wire_action_representation': 'cup-relative-tcp-v1',
                           'inference_seconds': .42},
                 'requested_monotonic': 10. + offset,
                 'observation_ready_monotonic': 10.4 + offset,
                 'predict_started_monotonic': 10.5 + offset,
                 'received_monotonic': 11.1 + offset,
                 'execution_prefix_steps': 3, 'status': 'available'}
        chunks.append(chunk)
        predictions.append({'request': copy.deepcopy(request),
                            'reply': {'id': request_id},
                            'execution_metadata': {key: chunk[key] for key in (
                                'requested_monotonic', 'observation_ready_monotonic',
                                'predict_started_monotonic', 'received_monotonic')}})
        for action_index in range(3):
            sequence = 3 * index + action_index
            submissions.append({'sequence': sequence, 'delivery_token': sequence + 1,
                                'request_id': request_id, 'action_index': action_index,
                                'submitted_monotonic': 11.4 + offset + action_index * .3,
                                'raw_action': actions[action_index],
                                'decoded': {'joint_degrees': [float(sequence)] * 6}})
            acknowledgements.append({'sequence': sequence, 'delivery_token': sequence + 1,
                                     'acknowledged_monotonic': 11.6 + offset + action_index * .3})
    return chunks, predictions, submissions, acknowledgements


class ActionTimeTests(unittest.TestCase):
    def test_pairs_requests_sequences_and_tokens_instead_of_line_order(self):
        chunks, predictions, submissions, acknowledgements = records()
        paired = audit.pair_events(chunks[::-1], predictions[::-1],
                                   submissions[::-1], acknowledgements[::-1])
        self.assertEqual([0, 1], [row['chunk_index'] for row in paired])
        self.assertEqual(['a', 'b'], [row['chunk']['request']['id'] for row in paired])
        self.assertEqual([3, 4, 5], [event['submission']['sequence']
                                   for event in paired[1]['events']])
        self.assertEqual(4, paired[1]['events'][0]['ack']['delivery_token'])
        self.assertTrue(all(row['complete'] for row in paired))

    def test_different_ack_token_cannot_attach_to_same_sequence(self):
        chunks, predictions, submissions, acknowledgements = records()
        acknowledgements[0]['delivery_token'] = 99
        with self.assertRaisesRegex(ValueError, 'acknowledgement.*submission'):
            audit.pair_events(chunks, predictions, submissions, acknowledgements)

    def test_missing_ack_is_incomplete_and_is_not_synthesized(self):
        chunks, predictions, submissions, acknowledgements = records()
        paired = audit.pair_events(chunks, predictions, submissions, acknowledgements[1:])
        self.assertFalse(paired[0]['complete'])
        self.assertIsNone(paired[0]['events'][0]['ack'])
        self.assertTrue(paired[1]['complete'])

    def test_submitted_raw_action_must_match_the_paired_model_reply(self):
        chunks, predictions, submissions, acknowledgements = records()
        submissions[0]['raw_action'] = list(submissions[0]['raw_action'])
        submissions[0]['raw_action'][0] += .001
        with self.assertRaisesRegex(ValueError, 'raw_action.*model reply action'):
            audit.pair_events(chunks, predictions, submissions, acknowledgements)

    def test_pre_submit_uses_receive_time_and_excludes_future_and_equal_time(self):
        earlier = capture(10.8, 10.7, [8., 0., 0.], 1)
        future = capture(10.9, 11.05, [7., 0., 0.], 2)
        equal = capture(10.95, 11., [6., 0., 0.], 3)
        timeline = audit.FeedbackTimeline([future, equal, earlier])
        self.assertEqual(1, timeline.latest_before(11.)['frame_index'])
        self.assertIsNone(timeline.latest_before(10.7))

    def test_after_ack_feedback_is_bounded_by_the_next_submission(self):
        timeline = audit.FeedbackTimeline([capture(12., 12.02, [7., 0., 0.], 1),
                                          capture(12.2, 12.21, [6., 0., 0.], 2)])
        self.assertEqual(1, timeline.first_after(12., before=12.1)['frame_index'])
        self.assertIsNone(timeline.first_after(12.1, before=12.2))

    def test_wire_endpoint_pairing_requires_value_and_rpc_return_in_the_ack_window(self):
        event = {'submission': {'submitted_monotonic': 11.,
                                'decoded': {'joint_degrees': [2.] * 6}},
                 'ack': {'acknowledged_monotonic': 11.2}}
        wire = [{'sequence': 1, 'issued_monotonic': 11.01, 'returned_monotonic': 11.02,
                 'raw_args': [[1.] * 6], 'return_code': 0, 'error': None},
                {'sequence': 2, 'issued_monotonic': 11.17, 'returned_monotonic': 11.18,
                 'raw_args': [[2.] * 6], 'return_code': 0, 'error': None},
                {'sequence': 3, 'issued_monotonic': 11.19, 'returned_monotonic': 11.21,
                 'raw_args': [[2.] * 6], 'return_code': 0, 'error': None}]
        result = audit.wire_window(event, wire, [row['issued_monotonic'] for row in wire])
        self.assertEqual(1, result['endpoint_candidate_count'])
        self.assertEqual(2, result['first_matching_endpoint']['sequence'])

    def test_hold_measures_recorded_samples_without_extrapolating_to_its_boundaries(self):
        timeline = audit.FeedbackTimeline([capture(9.9, 9.99, [12., 0., 0.]),
                                          capture(10.1, 10.1, [10., 0., 0.], 1),
                                          capture(10.4, 10.4, [10.2, 0., 0.], 2),
                                          capture(10.9, 10.9, [10.1, 0., 0.], 3),
                                          capture(11., 11., [15., 0., 0.], 4)])
        previous = {'submission': {'decoded': {'joint_degrees': [2.] * 6}},
                    'ack': {'acknowledged_monotonic': 10.}}
        wire = [{'sequence': 1, 'issued_monotonic': 10.2, 'returned_monotonic': 10.21,
                 'raw_args': [[2.] * 6], 'return_code': 0, 'error': None}]
        result = audit.hold_evidence(previous, 11., timeline, wire, [10.2], [0., 0., 0.])
        self.assertEqual(0, result['wire_not_previous_endpoint_count'])
        self.assertEqual(3, result['usable_feedback_count'])
        self.assertAlmostEqual(.8, result['feedback_span_s'])
        self.assertAlmostEqual(.1, result['measured_displacement_mm'])
        self.assertAlmostEqual(.2, result['measured_max_excursion_mm'])

    def test_real_delays_separate_server_duration_rpc_execution_and_hold(self):
        chunks, predictions, submissions, acknowledgements = records()
        paired = audit.pair_events(chunks, predictions, submissions, acknowledgements)
        timing = audit.timing_breakdown(paired[0], capture(10.2, 10.19, [10., 0., 0.]), 9.9)
        expected = {'request_to_observation_ready_s': .4,
                    'camera_to_observation_ready_s': .2,
                    'observation_ready_to_predict_start_s': .1,
                    'client_prediction_call_s': .6,
                    'server_model_inference_s': .42,
                    'client_prediction_nonmodel_s': .18,
                    'result_to_first_submit_s': .3,
                    'camera_to_first_submit_s': 1.2,
                    'first_submit_to_last_ack_s': .8,
                    'previous_last_ack_to_first_submit_s': 1.5}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(value, timing['durations_s'][key])
        pieces = ('previous_last_ack_to_request_s', 'request_to_observation_ready_s',
                  'observation_ready_to_predict_start_s', 'client_prediction_call_s',
                  'result_to_first_submit_s')
        self.assertAlmostEqual(1.5, sum(timing['durations_s'][key] for key in pieces))

    def test_direction_flip_and_small_negative_are_distinct_numeric_results(self):
        flipped = audit.compare_progress([10., 0., 0.], [8., 0., 0.], [7., 0., 0.],
                                         reference_mm=[0., 0., 0.])
        self.assertEqual(2., flipped['camera_anchor_progress_mm'])
        self.assertEqual(-1., flipped['pre_submit_progress_mm'])
        self.assertTrue(flipped['positive_at_camera_negative_at_submit'])
        positive = audit.compare_progress([10., 0., 0.], [8., 0., 0.], [9., 0., 0.],
                                          reference_mm=[0., 0., 0.])
        self.assertFalse(positive['positive_at_camera_negative_at_submit'])
        small = audit.compare_progress([10., 0., 0.], [10.284052, 0., 0.], [10., 0., 0.],
                                       reference_mm=[0., 0., 0.])
        self.assertEqual('negative_under_1mm', small['pre_submit_direction'])
        self.assertFalse(small['positive_at_camera_negative_at_submit'])

    def test_missing_feedback_or_coordinate_basis_is_null_with_reason(self):
        missing = audit.relative_feedback(None, [0., 0., 0.])
        self.assertIsNone(missing['cup_relative_xyz_mm'])
        self.assertEqual('no_recorded_feedback_in_time_window', missing['reason'])
        unsupported = audit.relative_feedback(capture(10., 9.99, [1., 2., 3.], tool=1),
                                              [0., 0., 0.])
        self.assertIsNone(unsupported['cup_relative_xyz_mm'])
        self.assertEqual('tcp_coordinate_basis_is_not_tool0_user0', unsupported['reason'])
        comparison = audit.compare_progress([10., 0., 0.], [8., 0., 0.], None,
                                            reference_mm=[0., 0., 0.])
        self.assertIsNone(comparison['pre_submit_progress_mm'])
        self.assertIsNone(comparison['positive_at_camera_negative_at_submit'])

    def test_audit_preserves_unexecuted_chunks_and_evaluation_scope(self):
        chunks, predictions, submissions, acknowledgements = records()
        rejected = copy.deepcopy(chunks[-1])
        rejected['_chunk_index'] = 2
        rejected['request']['id'] = rejected['reply']['id'] = 'rejected'
        rejected['request']['source_time'] = 16.2
        for key in ('requested_monotonic', 'observation_ready_monotonic',
                    'predict_started_monotonic', 'received_monotonic'):
            rejected[key] += 3.
        rejected['status'] = 'rejected_before_execution'
        chunks.append(rejected)
        predictions.append({'request': rejected['request'], 'reply': rejected['reply']})
        captures = [capture(10.2, 10.19, [10., 0., 0.]),
                    capture(11.3, 11.29, [10., 0., 0.], 1),
                    capture(13.2, 13.19, [10., 0., 0.], 2),
                    capture(14.3, 14.29, [10., 0., 0.], 3)]
        result = audit.audit_records(chunks, predictions, submissions, acknowledgements,
                                     captures, [],
                                     {'initial_cup': {'base_xyz_mm': [0., 0., 0.],
                                                      'camera_serial': 'recorded-camera'}},
                                     reference_mm=[0., 0., 0.])
        self.assertEqual(2, result['summary']['complete_executed_chunks'])
        self.assertEqual(1, result['summary']['unexecuted_chunks'])
        self.assertEqual(0, result['summary']['chunk_endpoint_at_first_submit']
                         ['positive_at_camera_negative_at_submit_count'])
        self.assertEqual('evaluation_only', result['dataset_role'])
        self.assertEqual(0, result['robot_commands'])
        self.assertEqual(0, result['model_inferences'])
        self.assertEqual(6, result['summary']['submission_raw_actions_matching_model_reply'])
        self.assertTrue(result['targets'][0]['model_target_source']['submission_raw_action_matches_reply'])
        self.assertFalse(result['task_success_verified'])
        self.assertEqual('rejected_before_execution', result['chunks'][-1]['recorded_status'])

    def test_sign_changes_distinguish_observation_age_from_within_prefix_state_changes(self):
        chunks, predictions, submissions, acknowledgements = records()
        captures = [capture(10.2, 10.19, [10., 0., 0.]),
                    capture(11.3, 11.29, [8.5, 0., 0.], 1),
                    capture(11.9, 11.89, [6.5, 0., 0.], 2),
                    capture(13.2, 13.19, [10., 0., 0.], 3),
                    capture(14.3, 14.29, [10., 0., 0.], 4)]
        result = audit.audit_records(chunks, predictions, submissions, acknowledgements,
                                     captures, [],
                                     {'initial_cup': {'base_xyz_mm': [0., 0., 0.],
                                                      'camera_serial': 'recorded-camera'}},
                                     reference_mm=[0., 0., 0.])
        self.assertEqual({'camera_to_chunk_first_submit': 1, 'within_executed_prefix': 1},
                         result['summary']['sign_change_windows'])
        self.assertIsNone(result['summary']['latency_cause_established'])
        self.assertIsNone(result['summary']['physical_wrong_direction_established'])


if __name__ == '__main__':
    unittest.main()
