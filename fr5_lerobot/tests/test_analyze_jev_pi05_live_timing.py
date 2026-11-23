"""Local timing evidence tests using small, hardware-free episodes."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from fr5_lerobot.diagnostics import analyze_jev_pi05_live_timing as timing


def episode_records(*, tail=False, reason='explicit_duration_limit'):
    chunks = []
    for index, identifier in enumerate(('aa', 'bb', 'cc')):
        source = 100.0 if index == 0 else 100.1 + index
        requested = 100.05 if index == 0 else 100.01 + index
        ready = 100.1 if index == 0 else 100.2 + index
        request = {
            'id': identifier, 'source_time': source,
            'state': [0.] * 6 + [.8 if index == 0 else .7, 1.],
            'observation_history': [],
        }
        chunk = {
            'request': request, 'reply': {'inference_seconds': .1},
            'requested_monotonic': requested, 'observation_ready_monotonic': ready,
            'predict_started_monotonic': ready + .01,
            'received_monotonic': ready + .2,
            'prefix_precheck_started_monotonic': ready + .21,
            'prefix_precheck_finished_monotonic': ready + .25,
            'execution_prefix_steps': 2,
            'joint_prefix_preflight': {
                'passed': True, 'anchor_fk_sample': None, 'segments': [],
            },
        }
        if index:
            chunk['observation_after_monotonic'] = 100. + index
        chunks.append(chunk)

    submissions = []
    acks = []
    details = (
        ('aa', 0, 100.4, 100.5, 80, None, 0, 0),
        ('aa', 1, 100.55, 101., 70, 'native_gripper_command_completed', 0, 1),
        ('bb', 0, 101.6, 102., 75, 'native_gripper_command_completed', 1, 2),
        ('cc', 0, 102.6, 102.8, 75, None, 2, 2),
        ('cc', 1, 102.85, 103., 75, 'execution_prefix_completed', 2, 2),
    )
    sources = {chunk['request']['id']: chunk['request']['source_time'] for chunk in chunks}
    for sequence, (identifier, action, submit, ack, gripper, cause, before, after) in enumerate(details):
        target = [.01 * (sequence + 1)] * 6 + [gripper / 100]
        submissions.append({
            'sequence': sequence, 'delivery_token': sequence + 1,
            'request_id': identifier, 'action_index': action,
            'prefix_consumed_steps': action + 1, 'execution_prefix_steps': 2,
            'target': target, 'effective_gripper_target_percent': gripper,
            'submitted_monotonic': submit, 'target_monotonic': submit + 1 / 6,
            'predicted_source_target_monotonic': sources[identifier] + (action + 1) / 6,
        })
        acks.append({
            'sequence': sequence, 'delivery_token': sequence + 1,
            'acknowledged_monotonic': ack, 'prefix_consumed_steps': action + 1,
            'prefix_retired_cause': cause, 'gripper_commands_before': before,
            'gripper_commands_after': after,
        })

    wire = []
    for method, issued, returned, target in (
        ('ServoMoveStart', 100.36, 100.37, None),
        ('ServoJ', 100.38, 100.39, None),
        ('ServoJ', 100.58, 100.59, None),
        ('MoveGripper', 100.6, 100.61, 70),
        ('ServoJ', 100.94, 100.95, None),
        ('ServoJ', 101.67, 101.68, None),
        ('MoveGripper', 101.7, 101.71, 75),
        ('ServoJ', 101.94, 101.95, None),
        ('ServoJ', 102.64, 102.65, None),
        ('ServoJ', 102.9, 102.91, None),
        ('StopMotion', 102.99 if tail else 103.1, 103.11, None),
        ('ServoMoveEnd', 103.12, 103.13, None),
    ):
        wire.append({
            'sequence': len(wire), 'method': method,
            'issued_monotonic': issued, 'returned_monotonic': returned,
            'issued_unix': 1790000000. + issued, 'target_deg_percent': target,
            'return_code': 0, 'error': None,
        })

    pending = None
    if tail:
        acks.pop()
        submissions[-1]['target'][6] = .74
        pending = {
            'token': 5, 'version': 5,
            'joints': [math.degrees(value) for value in submissions[-1]['target'][:6]],
            'gripper': 75, 'joint_ack': False, 'gripper_ack': True,
        }
    snapshot = {
        'status': 'closed', 'reason': reason, 'failure': None, 'callback_failure': None,
        'stop_result': 0, 'servo_end_result': 0, 'servo_commands': 7,
        'gripper_commands': 2, 'gripper_command_inflight': False,
        'ordered_last_token': 5, 'ordered_acknowledged_token': len(acks),
        'ordered_pending': pending, 'pending_callbacks': 0, 'inflight_commands': 0,
        'workers_alive': [], 'undelivered_callbacks': [], 'startup_inflight': False,
        'gripper_completion_feedback': {
            'target_percent': 75, 'position_percent': 75., 'motiondone': 1,
            'feedback_monotonic': 101.89, 'command_began_monotonic': 101.69,
            'completed_monotonic': 101.9,
        },
    }
    summary = {
        'reason': reason, 'termination_source': reason, 'error': None,
        'failed': None, 'journal_error': None, 'summary_stage': 'final_control_snapshot',
        'wire_callbacks_drained': True, 'controller_snapshot': snapshot,
        'prediction_records': len(chunks), 'wire_records': len(wire),
        'outcome': 'pending_review', 'task_success_verified': False,
    }
    return {
        'scope': {'policy_contract': {'fps': 6}}, 'summary': summary, 'chunks': chunks,
        'submitted': submissions, 'acks': acks, 'wire': wire,
    }


def write_episode(root, records):
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def write_jsonl(path, values):
        path.parent.mkdir(exist_ok=True)
        path.write_text(''.join(json.dumps(value) + '\n' for value in values), encoding='utf-8')

    write_json(root / 'scope.json', records['scope'])
    write_json(root / 'summary.json', records['summary'])
    for index, chunk in enumerate(records['chunks']):
        write_json(root / f'chunk-{index:05d}.json', chunk)
    journal = [{'request': chunk['request'], 'reply': chunk['reply']} for chunk in records['chunks']]
    write_jsonl(root / 'stream-prediction/000000.jsonl', journal)
    write_jsonl(root / 'submitted-targets.jsonl', records['submitted'])
    write_jsonl(root / 'acknowledged-targets.jsonl', records['acks'])
    write_jsonl(root / 'stream-wire/000000.jsonl', records['wire'])


class TimingTests(unittest.TestCase):
    def analyze(self, records):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, records)
            result = timing.analyze(root)
            json.dumps(result, allow_nan=False)
            return result

    def test_duration_stop_preserves_one_unacknowledged_tail(self):
        result = self.analyze(episode_records(tail=True))
        self.assertEqual(result['counts']['submitted_targets'], 5)
        self.assertEqual(result['counts']['acknowledged_targets'], 4)
        self.assertEqual(result['counts']['unacknowledged_targets_at_stop'], 1)
        target = result['targets'][-1]
        self.assertEqual(target['delivery_status'], 'unacknowledged_at_stop')
        self.assertEqual(target['prefix_consumed_steps'], 2)
        for field in ('acknowledged_monotonic', 'source_age_at_ack_s', 'submit_to_ack_s',
                      'ack_prefix_consumed_steps'):
            self.assertIsNone(target[field])
        self.assertEqual(result['target_distributions_s']['source_age_at_submit_s']['count'], 5)
        self.assertEqual(result['target_distributions_s']['submit_to_ack_s']['count'], 4)
        self.assertEqual(result['target_ack_by_gripper_change_s']['without_move_gripper']['count'], 2)
        tail = result['unacknowledged_tail_target']
        self.assertEqual(tail['delivery_token'], 5)
        self.assertEqual(tail['ordered_pending']['gripper'], 75)
        self.assertEqual(result['termination']['outcome'], 'pending_review')
        self.assertFalse(result['termination']['task_success_verified'])
        self.assertNotIn('minimal_improvement', result)

    def test_operator_stop_uses_the_same_tail_evidence(self):
        result = self.analyze(episode_records(tail=True, reason='operator_stop'))
        self.assertEqual(result['counts']['unacknowledged_targets_at_stop'], 1)
        self.assertEqual(result['termination']['termination_source'], 'operator_stop')

    def test_all_acknowledged_targets_and_prefix_statistics(self):
        result = self.analyze(episode_records())
        self.assertEqual(result['counts']['unacknowledged_targets_at_stop'], 0)
        self.assertIsNone(result['unacknowledged_tail_target'])
        self.assertEqual(result['target_distributions_s']['submit_to_ack_s']['count'], 5)
        self.assertTrue(all(row['delivery_status'] == 'acknowledged' for row in result['targets']))
        self.assertEqual(result['prefix_retirement']['acknowledgement_cause_counts'], {
            'null': 2, 'native_gripper_command_completed': 2, 'execution_prefix_completed': 1,
        })
        self.assertEqual(result['prefix_retirement']['consumed_steps_by_cause']
                         ['native_gripper_command_completed'], {'1': 1, '2': 1})
        self.assertEqual(result['prefix_consumed_steps']['submitted'], {'1': 3, '2': 2})
        self.assertEqual(result['targets'][1]['gripper_commands_before'], 0)
        self.assertEqual(result['targets'][1]['gripper_commands_after'], 1)
        self.assertEqual(result['targets'][1]['ack_prefix_consumed_steps'], 2)

    def test_missing_middle_ack_is_not_a_tail(self):
        records = episode_records()
        del records['acks'][1]
        with self.assertRaisesRegex(ValueError, 'sequence'):
            self.analyze(records)

    def test_more_than_one_missing_ack_is_rejected(self):
        records = episode_records(tail=True)
        records['acks'].pop()
        with self.assertRaises(ValueError):
            self.analyze(records)

    def test_wrong_tokens_are_rejected(self):
        for collection, index in (('acks', 1), ('submitted', 1), ('acks', -1)):
            records = episode_records(tail=True)
            records[collection][index]['delivery_token'] = 77
            with self.subTest(collection=collection, index=index), self.assertRaisesRegex(ValueError, 'token'):
                self.analyze(records)
        records = episode_records()
        records['acks'][1]['delivery_token'] = records['submitted'][1]['delivery_token'] = 77
        with self.assertRaisesRegex(ValueError, 'token'):
            self.analyze(records)

    def test_duplicate_or_unordered_sequences_are_rejected(self):
        for collection in ('submitted', 'acks'):
            for duplicate in (False, True):
                records = episode_records()
                if duplicate:
                    records[collection][1]['sequence'] = 0
                else:
                    records[collection][0], records[collection][1] = records[collection][1], records[collection][0]
                with self.subTest(collection=collection, duplicate=duplicate), self.assertRaisesRegex(ValueError, 'sequence'):
                    self.analyze(records)

    def test_tail_requires_consistent_final_termination(self):
        for key, value in (
            ('reason', 'error'), ('reason', 'unknown_stop'), ('termination_source', 'operator_stop'),
            ('summary_stage', 'provisional'), ('wire_callbacks_drained', False),
            ('error', 'late action'), ('failed', 'journal failed'), ('journal_error', 'undelivered'),
        ):
            records = episode_records(tail=True)
            records['summary'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze(records)
        for key, value in (
            ('reason', 'operator_stop'), ('status', 'stopped'), ('failure', 'servo failure'),
            ('callback_failure', 'callback failed'), ('stop_result', 1), ('servo_end_result', 1),
            ('pending_callbacks', 1), ('inflight_commands', 1), ('workers_alive', ['fr5-servo']),
            ('undelivered_callbacks', [{'sequence': 1}]), ('startup_inflight', True),
            ('gripper_command_inflight', True),
        ):
            records = episode_records(tail=True)
            records['summary']['controller_snapshot'][key] = value
            with self.subTest(snapshot_key=key), self.assertRaises(ValueError):
                self.analyze(records)

    def test_tail_tokens_and_pending_target_must_match_submission(self):
        records = episode_records(tail=True)
        snapshot = records['summary']['controller_snapshot']
        variations = []
        for key, value in (('ordered_last_token', 6), ('ordered_acknowledged_token', 5),
                           ('ordered_pending', None)):
            changed = deepcopy(snapshot)
            changed[key] = value
            variations.append(changed)
        for key, value in (('token', 4), ('version', 4), ('joints', [0.] * 6), ('gripper', 74),
                           ('joint_ack', True)):
            changed = deepcopy(snapshot)
            changed['ordered_pending'][key] = value
            variations.append(changed)
        for changed in variations:
            fixture = deepcopy(records)
            fixture['summary']['controller_snapshot'] = changed
            with self.subTest(snapshot=changed), self.assertRaises(ValueError):
                self.analyze(fixture)

    def test_submission_and_ack_prefix_consumption_must_agree(self):
        records = episode_records()
        records['acks'][1]['prefix_consumed_steps'] = 1
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.analyze(records)

    def test_retirement_links_next_current_observation_and_completion_bounds(self):
        records = episode_records()
        records['chunks'][1]['request']['observation_history'] = [
            {'age_s': .09, 'state': [0.] * 6 + [.99, 1.]},
        ]
        result = self.analyze(records)
        retirement = result['prefix_retirements'][0]
        self.assertEqual(retirement['request_id'], 'aa')
        self.assertEqual(retirement['next_request_id'], 'bb')
        self.assertEqual(retirement['next_observation_after_monotonic'], 101.)
        self.assertEqual(retirement['acknowledged_monotonic'], 101.)
        self.assertAlmostEqual(retirement['ack_to_next_source_s'], .1)
        self.assertAlmostEqual(retirement['ack_to_next_request_s'], .01)
        first, last = result['move_gripper']
        observed = first['first_valid_post_ack_prediction_observation']
        self.assertEqual(observed['observation_kind'], 'request')
        self.assertEqual(observed['request_id'], 'bb')
        self.assertEqual(observed['observation_source_monotonic'], 101.1)
        self.assertIsNone(first['recorded_completion_feedback'])
        self.assertEqual(first['completion_detection_monotonic_bounds'], [100.61, 100.94])
        bounds = first['completion_to_next_source_s_bounds']
        self.assertAlmostEqual(bounds[0], 101.1 - 100.94)
        self.assertAlmostEqual(bounds[1], 101.1 - 100.61)
        self.assertIsNone(first['recorded_completion_to_next_source_s'])
        self.assertEqual(last['recorded_completion_feedback']['completed_monotonic'], 101.9)
        self.assertAlmostEqual(last['recorded_completion_to_next_source_s'], .2)
        self.assertEqual(result['counts']['gripper_completion_feedback_exact'], 1)
        self.assertTrue(any('1 次夹爪' in limitation for limitation in result['limitations']))

    def test_valid_history_does_not_replace_invalid_current_observation(self):
        records = episode_records()
        request = records['chunks'][1]['request']
        request['state'][7] = 0.
        request['observation_history'] = [{'age_s': .09, 'state': [0.] * 6 + [.7, 1.]}]
        result = self.analyze(records)
        command = result['move_gripper'][0]
        self.assertIsNone(command['first_valid_post_ack_prediction_observation'])
        observed = command['fresh_prediction_observation']
        self.assertFalse(observed['valid'])
        self.assertEqual(observed['observation_kind'], 'request')
        self.assertEqual(observed['request_id'], 'bb')

    def test_next_chunk_must_reference_its_previous_request_retirement_ack(self):
        for index, after in ((1, 100.5), (2, 101.), (1, None)):
            records = episode_records()
            records['chunks'][index]['observation_after_monotonic'] = after
            with self.subTest(index=index, after=after), self.assertRaisesRegex(ValueError, 'retirement'):
                self.analyze(records)
        records = episode_records()
        records['acks'][1]['prefix_retired_cause'] = None
        with self.assertRaisesRegex(ValueError, 'retirement'):
            self.analyze(records)

    def test_current_source_must_be_newer_than_retirement_ack(self):
        for source in (101., 100.99):
            records = episode_records()
            records['chunks'][1]['request']['source_time'] = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'observation'):
                self.analyze(records)

    def test_legacy_all_ack_lateness_report_still_works(self):
        records = episode_records(reason='error')
        error = 'RuntimeError: request cc, action 2 (limit 0.5s)'
        records['summary'].update(error=error, failed=error, outcome='interrupted')
        for row in records['acks']:
            for field in ('prefix_retired_cause', 'gripper_commands_before', 'gripper_commands_after'):
                row.pop(field)
        result = self.analyze(records)
        self.assertEqual(result['counts']['acknowledged_targets'], 5)
        self.assertEqual(result['prefix_retirement']['acknowledgement_cause_counts'], {'not_recorded': 5})
        failure = result['final_lateness_stop']
        self.assertEqual(failure['request_id'], 'cc')
        self.assertEqual(failure['previous_action']['sequence'], 4)
        self.assertEqual(failure['check_monotonic_recorded_bounds'], [103., 103.1])
        self.assertGreater(failure['minimum_lateness_over_limit_s'], 0.)

    def test_stopped_gripper_tail_has_no_invented_completion_or_ack(self):
        records = episode_records(tail=True, reason='operator_stop')
        records['submitted'][-1]['effective_gripper_target_percent'] = 65
        records['submitted'][-1]['target'][6] = .65
        snapshot = records['summary']['controller_snapshot']
        snapshot['ordered_pending'].update(gripper=65, gripper_ack=False)
        snapshot['gripper_commands'] = 3
        records['wire'].insert(-2, {
            'sequence': 12, 'method': 'MoveGripper', 'issued_monotonic': 102.95,
            'returned_monotonic': 102.96, 'issued_unix': 1790000102.95,
            'target_deg_percent': 65, 'return_code': 0, 'error': None,
        })
        records['summary']['wire_records'] += 1
        result = self.analyze(records)
        command = result['move_gripper'][-1]
        self.assertIsNone(command['acknowledged_monotonic'])
        self.assertIsNone(command['recorded_completion_feedback'])
        self.assertIsNone(command['completion_detection_monotonic_bounds'])
        self.assertIsNone(command['submit_to_ack_s'])
        self.assertIsNone(command['fresh_prediction_observation'])
        self.assertEqual(result['gripper_distributions_s']['submit_to_ack_s']['count'], 2)


if __name__ == '__main__':
    unittest.main()
