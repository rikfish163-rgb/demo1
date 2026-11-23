"""Native gripper ACK evidence and ordered prefix retirement materialization."""

import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from .materialize_stream import (
    _stream_controls, _stream_prediction_sources, _validate_prefix_ack,
    materialize, verify_materialized,
)
from . import stream_protocol
from .stream_storage import CODEC
from .test_stream_materialization import digest
from lerobot_robot_fairino.frame_codec import encode_media


class StreamPrefixRetirementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fr5-prefix-retirement-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'spool'
        self.root.mkdir()
        self.prepare()

    def put(self, name, value):
        destination = self.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value if isinstance(value, bytes) else value.encode())

    def prepare(self, consumed=1, prefix=10, *, legacy=False, gripper=76):
        self.consumed = consumed
        self.scope = {
            'dataset_role': 'policy_evaluation',
            'policy_contract': {'name': 'fr5-joint-position-stream-v1'},
            'task_instruction': 'test', 'checkpoint_sha256': 'a' * 64,
            'execution_mode': 'ordered_complete_chunk' if prefix == 50 else 'ordered_policy_prefix',
            'execution_chunk_steps': prefix, 'prediction_chunk_steps': 50, 'prefix_skip': False,
            'observation_context_profile': 'state-history-physical-v1',
            'wire_action_representation': 'absolute-joint-position-v1',
            'target_delivery': 'ordered_native_acknowledgements',
        }
        if not legacy:
            self.scope['observation_rule'] = 'prefix_or_completed_native_gripper_command'
        self.chunks, self.predictions, self.targets, self.acks, self.wire, self.frames = [], [], [], [], [], {}
        media, self.rgb, self.depth = encode_media(
            np.full((12, 16, 3), 20, np.uint8), np.full((12, 16), 500, np.float32))
        last_ack = 100.2 + (consumed - 1) * .25 + (.5 if gripper != 100 else .18)
        self.sources = [100., last_ack + .2]
        for number, source in enumerate(self.sources):
            grip = 100 if number == 0 else gripper
            robot = {'received_monotonic': source + .004, 'joints': [0.] * 6,
                     'gripper_position': grip, 'gripper_active': 1, 'gripper_fault': 0,
                     'gripper_fault_id': 0, 'gripper_motiondone': 1, 'gripper_speed': 0}
            self.frames[source] = {'frame_index': number, 'robot': robot, 'media': media}
            request = {
                'schema': 'fr5-stream-request-v1', 'id': f'prefix-{number}',
                'policy_contract': 'fr5-joint-position-stream-v1', 'task': 'test',
                'source_time': source, 'source_clock': 'host_monotonic',
                'observation_context_profile': 'state-history-physical-v1',
                'state': [0.] * 6 + [grip / 100., 1.],
                'rgb_jpeg_base64': base64.b64encode(self.rgb).decode(),
                'rgb_jpeg': {'bytes': len(self.rgb), 'sha256': digest(self.rgb),
                             'storage': 'exact_base64_in_this_request'},
            }
            actions = [[index / 1000., 0., 0., 0., 0., 0.,
                        1. if number == 0 and index < consumed - 1 else gripper / 100.]
                       for index in range(50)]
            reply = {'schema': 'fr5-stream-reply-v1', 'id': request['id'],
                     'checkpoint_sha256': 'a' * 64, 'policy_contract': request['policy_contract'],
                     'wire_action_representation': 'absolute-joint-position-v1', 'actions': actions}
            audit_request = {key: value for key, value in request.items()
                             if key not in ('rgb_jpeg_base64', 'rgb_jpeg')}
            chunk = {'request': deepcopy(audit_request), 'reply': deepcopy(reply),
                     'execution_prefix_steps': prefix, 'received_monotonic': source + .1}
            if number:
                chunk['observation_after_monotonic'] = last_ack
            self.chunks.append(chunk)
            self.predictions.append({'request': deepcopy(request), 'reply': deepcopy(reply)})
            for index in range(consumed if number == 0 else 2):
                sequence = len(self.targets)
                submitted = source + .2 + index * .25
                raw = actions[index]
                target = {
                    'sequence': sequence, 'delivery_token': sequence + 1, 'request_id': request['id'],
                    'action_index': index, 'raw_action': raw, 'target': raw[:6] + [raw[6]],
                    'submitted_monotonic': submitted, 'target_monotonic': submitted + 1 / 6,
                    'predicted_source_target_monotonic': source + (index + 1) / 6,
                    'execution_prefix_steps': prefix, 'prediction_chunk_steps': 50,
                    'prefix_consumed_steps': index + 1,
                    'effective_gripper_target_percent': round(raw[6] * 100),
                }
                changed = number == 0 and index == consumed - 1 and gripper != 100
                before, after = (0, int(changed)) if number == 0 else (int(gripper != 100),) * 2
                ack = {'sequence': sequence, 'delivery_token': sequence + 1,
                       'acknowledged_monotonic': submitted + (.5 if changed else .18),
                       'prefix_consumed_steps': index + 1,
                       'prefix_retired_cause': ('native_gripper_command_completed' if changed else
                                                'execution_prefix_completed' if index + 1 == prefix else None),
                       'gripper_commands_before': before, 'gripper_commands_after': after}
                if legacy:
                    target.pop('prefix_consumed_steps')
                    ack = {key: ack[key] for key in
                           ('sequence', 'delivery_token', 'acknowledged_monotonic')}
                self.targets.append(target)
                if number == 0 or index == 0:
                    self.acks.append(ack)
                if changed:
                    self.add_wire('MoveGripper', submitted + .01, gripper)
                self.add_wire('ServoJ', submitted + (.45 if changed else 1 / 6),
                              [math.degrees(value) for value in raw[:6]])
        self.add_wire('ServoMoveStart', 100.15)
        self.add_wire('StopMotion', self.targets[-1]['submitted_monotonic'] + .6)
        self.add_wire('ServoMoveEnd', self.targets[-1]['submitted_monotonic'] + .61)
        self.wire.sort(key=lambda row: row['issued_monotonic'])
        for sequence, row in enumerate(self.wire, 1):
            row['sequence'] = sequence
        self.write_controls()

    def add_wire(self, method, issued, target=None):
        args = ([1, target, 20, 10, 10000, 1, 0, 0., 0, 0] if method == 'MoveGripper'
                else [target, [0.] * 4, 0., 0., .008, 0., 0., 0] if method == 'ServoJ' else [])
        self.wire.append({'method': method, 'args': args, 'raw_args': deepcopy(args),
                          'target_deg_percent': target, 'issued_monotonic': issued,
                          'returned_monotonic': issued + .002, 'issued_unix': issued + 1000.,
                          'returned_unix': issued + 1000.002, 'callback_monotonic': issued + .003,
                          'return_code': 0, 'error': None})

    def write_journal(self, kind, rows):
        records = [dict(row, callback_sequence=index) for index, row in enumerate(rows)]
        raw = ''.join(json.dumps(row) + '\n' for row in records).encode()
        self.put(f'stream-{kind}/000000.jsonl', raw)
        index = {'schema': f'fr5-evaluation-{kind}-index-v1', 'count': len(rows),
                 'chunks': [{'path': '000000.jsonl', 'bytes': len(raw),
                             'sha256': digest(raw), 'rows': len(rows)}]}
        self.put(f'stream-{kind}/index.json', json.dumps(index))

    def write_controls(self):
        self.put('scope.json', json.dumps(self.scope))
        self.put('SESSION_STARTED', '{}')
        self.put('submitted-targets.jsonl', ''.join(json.dumps(row) + '\n' for row in self.targets))
        self.put('acknowledged-targets.jsonl', ''.join(json.dumps(row) + '\n' for row in self.acks))
        for number, chunk in enumerate(self.chunks):
            self.put(f'chunk-{number:05d}.json', json.dumps(chunk))
        self.write_journal('wire', self.wire)
        self.write_journal('prediction', self.predictions)
        counts = {method: sum(row['method'] == method for row in self.wire)
                  for method in ('ServoJ', 'MoveGripper')}
        frames = len(self.frames)
        self.summary = {
            'schema': 'fr5-stream-policy-evaluation-summary-v1', 'dataset_role': 'policy_evaluation',
            'task_instruction': 'test', 'checkpoint_sha256': 'a' * 64,
            'task_success_verified': False, 'training_ready': False,
            'awaiting_whole_user_review': True, 'read_only_diagnostic': False, 'model_execution': True,
            'storage_status': 'server_stream_closed', 'summary_stage': 'final_control_snapshot',
            'termination_source': 'operator_stop', 'reason': 'operator_stop', 'failed': None, 'error': None,
            'recorded_frames': frames, 'continuous_capture': {'frames': frames},
            'wire_records': len(self.wire), 'prediction_records': len(self.predictions),
            'wire_method_counts': counts, 'commands': sum(counts.values()), 'wire_callbacks_drained': True,
            'controller_snapshot': {'servo_commands': counts['ServoJ'],
                                    'gripper_commands': counts['MoveGripper'], 'pending_callbacks': 0,
                                    'workers_alive': False},
            'scope_sha256': digest((self.root / 'scope.json').read_bytes()),
        }
        self.put('summary.json', json.dumps(self.summary))
        self.receipt = {'schema': 'fr5-stream-closed-v1', 'session': 'e' * 32,
                        'accepted_frames': frames, 'last_durable_sequence': frames - 1,
                        'unacknowledged_frames': 0, 'error': None,
                        'local_episode': str(Path(self.tmp.name) / 'original'),
                        'codec_sha256': digest(CODEC.read_bytes()),
                        'protocol_sha256': digest(Path(stream_protocol.__file__).read_bytes())}
        self.put('stream-closed.json', json.dumps(self.receipt))

    def controls(self):
        self.write_controls()
        entries = {path.relative_to(self.root).as_posix(): {} for path in self.root.rglob('*') if path.is_file()}
        return _stream_controls(self.root, self.receipt, self.scope, self.summary, entries)

    def archive(self):
        self.write_controls()
        self.put('continuous/report.json', json.dumps({'frames': len(self.frames), 'capture_complete': True}))
        self.put('continuous/events.jsonl', '')
        packets = []
        for captured, frame in sorted(self.frames.items()):
            index, robot, media = frame['frame_index'], frame['robot'], frame['media']
            camera = {'source_timestamp_monotonic': captured}
            capture = {'frame_index': index, 'camera': camera, 'robot': robot, 'sync_clock': 'host_monotonic',
                       'sync_skew_s': robot['received_monotonic'] - captured,
                       'rgb_file': f'frames/{index:06d}.jpg', 'depth_file': f'frames/{index:06d}.npz',
                       'media_codec': media}
            teacher = {key: capture[key] for key in ('frame_index', 'camera', 'robot', 'sync_clock', 'sync_skew_s')}
            teacher.update(recording_role='policy_evaluation',
                           command_link='resolve_actual_wire_events_by_capture_monotonic',
                           captured_monotonic=captured)
            packets.append(stream_protocol.encode_record(
                {'schema': 'fr5-stream-frame-v1', 'session': self.receipt['session'],
                 'sequence': index, 'capture': capture, 'teacher': teacher, 'media': media}, self.rgb, self.depth))
        self.put('frames.stream', b''.join(packets))
        (self.root / 'wire').mkdir(exist_ok=True)
        shutil.copyfile(CODEC, self.root / 'wire/frame_codec.py')
        shutil.copyfile(stream_protocol.__file__, self.root / 'wire/stream_protocol.py')
        entries = {}
        for path in self.root.rglob('*'):
            name = path.relative_to(self.root).as_posix()
            if path.is_file() and not name.startswith('wire/') and name not in ('frames.stream', 'control-manifest.json'):
                raw = path.read_bytes()
                entries[name] = {'bytes': len(raw), 'sha256': digest(raw)}
        self.put('control-manifest.json', json.dumps({
            'schema': 'fr5-final-control-manifest-v1', 'session': self.receipt['session'],
            'local_episode': self.receipt['local_episode'], 'files': entries,
            'scope_sha256': entries['scope.json']['sha256'], 'summary_sha256': entries['summary.json']['sha256']}))

    def set_next_source(self, source, robot_time=None):
        frame = self.frames.pop(self.chunks[1]['request']['source_time'])
        frame['robot']['received_monotonic'] = source + .004 if robot_time is None else robot_time
        self.frames[source] = frame
        self.chunks[1]['request']['source_time'] = source
        self.predictions[1]['request']['source_time'] = source
        for row in self.targets[self.consumed:]:
            row['predicted_source_target_monotonic'] = source + (row['action_index'] + 1) / 6

    def set_index(self, sequence, index):
        row = self.targets[sequence]
        chunk = self.chunks[0 if sequence < self.consumed else 1]
        raw = chunk['reply']['actions'][index]
        row.update(action_index=index, raw_action=raw, target=raw[:6] + [raw[6]],
                   prefix_consumed_steps=index + 1, effective_gripper_target_percent=round(raw[6] * 100),
                   predicted_source_target_monotonic=chunk['request']['source_time'] + (index + 1) / 6)
        for ack in self.acks:
            if ack['sequence'] == sequence:
                ack['prefix_consumed_steps'] = index + 1

    def test_native_gripper_retirement_consumes_one_two_or_three_actions(self):
        for consumed, prefix in ((1, 10), (2, 10), (3, 10), (3, 3)):
            with self.subTest(consumed=consumed, prefix=prefix):
                self.prepare(consumed, prefix)
                journal = self.controls()
                timing = journal['ordered_timing']
                self.assertEqual([row['action_index'] for row in timing], list(range(consumed)) + [0, 1])
                retired = timing[consumed - 1]
                self.assertEqual(retired['prefix_consumed_steps'], consumed)
                self.assertEqual(retired['prefix_retired_cause'], 'native_gripper_command_completed')
                self.assertEqual((retired['gripper_commands_before'], retired['gripper_commands_after']), (0, 1))
                self.assertEqual(journal['chunks']['prefix-0']['reply']['actions'],
                                 self.predictions[0]['reply']['actions'])
                self.assertEqual(len(journal['chunks']['prefix-0']['reply']['actions'][consumed:]), 50 - consumed)
                links = _stream_prediction_sources(journal, self.frames)
                after = next(link for link in links if link['request_id'] == 'prefix-1')
                self.assertEqual(after['prefix_consumed_steps'], consumed)
                self.assertEqual(after['prefix_retired_cause'], retired['prefix_retired_cause'])
                self.assertGreater(after['source_timestamp_monotonic'], retired['acknowledged_monotonic'])
                self.assertGreater(after['source_robot_timestamp_monotonic'], retired['acknowledged_monotonic'])
                self.assertNotIn('acknowledged_monotonic', timing[-1])

    def test_complete_chunk_mode_can_retire_after_native_gripper_ack(self):
        self.prepare(3, 50)
        self.assertEqual([row['action_index'] for row in self.controls()['ordered_timing']], [0, 1, 2, 0, 1])

    def test_joint_full_prefix_without_move_gripper_keeps_normal_boundary(self):
        self.prepare(3, 3, gripper=100)
        timing = self.controls()['ordered_timing']
        self.assertEqual([row['action_index'] for row in timing], [0, 1, 2, 0, 1])
        self.assertEqual(timing[2]['prefix_retired_cause'], 'execution_prefix_completed')
        self.assertEqual((timing[2]['gripper_commands_before'], timing[2]['gripper_commands_after']), (0, 0))

    def test_materialization_preserves_consumption_alignment_and_native_tail(self):
        self.prepare(2)
        self.archive()
        result = materialize(self.root)
        output = self.root / 'materialized'
        timing = [json.loads(line) for line in (output / 'stream-target-timing.jsonl').read_text().splitlines()]
        self.assertEqual((timing[1]['prefix_consumed_steps'], timing[1]['prefix_retired_cause']),
                         (2, 'native_gripper_command_completed'))
        links = json.loads((output / 'stream-prediction-sources.json').read_text())
        self.assertTrue(any(link.get('prefix_consumed_steps') == 2 for link in links))
        self.assertEqual(json.loads((output / 'chunk-00000.json').read_text())['reply']['actions'],
                         self.predictions[0]['reply']['actions'])
        self.assertEqual(verify_materialized(self.root), result)
        self.assertFalse(result['task_success_verified'])

    def test_retirement_requires_a_real_successful_move_gripper(self):
        for failure in ('missing', 'return_code', 'error', 'late_return', 'wrong_target'):
            with self.subTest(failure=failure):
                self.prepare()
                event = next(row for row in self.wire if row['method'] == 'MoveGripper')
                if failure == 'missing':
                    self.wire.remove(event)
                elif failure == 'return_code':
                    event['return_code'] = 7
                elif failure == 'error':
                    event['error'] = 'RPC outcome unknown'
                elif failure == 'late_return':
                    event['returned_monotonic'] = self.acks[0]['acknowledged_monotonic'] + .01
                    event['callback_monotonic'] = event['returned_monotonic'] + .001
                else:
                    event['target_deg_percent'] = event['args'][1] = event['raw_args'][1] = 75
                with self.assertRaisesRegex(ValueError, 'completed MoveGripper'):
                    self.controls()

    def test_wire_callback_can_follow_the_target_ack(self):
        event = next(row for row in self.wire if row['method'] == 'MoveGripper')
        event['callback_monotonic'] = self.acks[0]['acknowledged_monotonic'] + .1
        self.assertEqual(self.controls()['ordered_timing'][0]['prefix_retired_cause'],
                         'native_gripper_command_completed')

    def test_rpc_acceptance_without_ack_cannot_retire(self):
        self.acks = []
        with self.assertRaisesRegex(ValueError, 'prior native acknowledgement'):
            self.controls()
        (self.root / 'acknowledged-targets.jsonl').unlink()
        entries = {path.relative_to(self.root).as_posix(): {} for path in self.root.rglob('*') if path.is_file()}
        with self.assertRaisesRegex(ValueError, 'prior native acknowledgement'):
            _stream_controls(self.root, self.receipt, self.scope, self.summary, entries)

    def test_observation_rule_alone_cannot_skip_a_prefix(self):
        for row in self.targets:
            row.pop('prefix_consumed_steps')
        self.acks = [{key: ack[key] for key in ('sequence', 'delivery_token', 'acknowledged_monotonic')}
                     for ack in self.acks]
        with self.assertRaisesRegex(ValueError, 'Ordered action sequence'):
            self.controls()

    def test_ack_token_consumption_reason_and_counts_must_match(self):
        for key, value in (('delivery_token', 9), ('prefix_consumed_steps', 2),
                           ('prefix_retired_cause', 'execution_prefix_completed'),
                           ('prefix_retired_cause', None), ('gripper_commands_before', 1),
                           ('gripper_commands_after', 0)):
            with self.subTest(key=key, value=value):
                self.prepare()
                self.acks[0][key] = value
                with self.assertRaisesRegex(ValueError, 'prefix|gripper'):
                    self.controls()

    def test_camera_source_and_observation_bound_must_follow_ack(self):
        for failure in ('source', 'bound'):
            with self.subTest(failure=failure):
                self.prepare()
                stamp = self.acks[0]['acknowledged_monotonic']
                if failure == 'source':
                    self.set_next_source(stamp)
                else:
                    self.chunks[1]['observation_after_monotonic'] = stamp - .01
                with self.assertRaisesRegex(ValueError, 'observation does not follow'):
                    self.controls()

    def test_robot_source_must_follow_ack(self):
        stamp = self.acks[0]['acknowledged_monotonic']
        self.set_next_source(stamp + .02, stamp - .02)
        with self.assertRaisesRegex(ValueError, 'camera and robot feedback after ACK'):
            _stream_prediction_sources(self.controls(), self.frames)

    def test_unsubmitted_next_chunk_still_requires_post_ack_observation(self):
        self.targets = self.targets[:1]
        self.acks = self.acks[:1]
        self.assertTrue(_stream_prediction_sources(self.controls(), self.frames))
        self.set_next_source(self.acks[0]['acknowledged_monotonic'])
        with self.assertRaisesRegex(ValueError, 'observation does not follow'):
            self.controls()

    def test_post_ack_feedback_must_be_settled(self):
        for key, value in (('gripper_motiondone', 0), ('gripper_speed', 1),
                           ('gripper_fault', 1), ('gripper_active', 0)):
            with self.subTest(key=key):
                self.prepare()
                self.frames[self.sources[1]]['robot'][key] = value
                for request in (self.chunks[1]['request'], self.predictions[1]['request']):
                    request['state'][6:] = [0., 0.]
                with self.assertRaisesRegex(ValueError, 'settled camera and robot feedback'):
                    _stream_prediction_sources(self.controls(), self.frames)

    def test_post_ack_contact_feedback_is_supported(self):
        for done, position in ((2, 79), (3, 78)):
            with self.subTest(done=done):
                self.prepare()
                self.frames[self.sources[1]]['robot'].update(gripper_motiondone=done, gripper_position=position)
                for request in (self.chunks[1]['request'], self.predictions[1]['request']):
                    request['state'][6] = position / 100.
                self.assertTrue(_stream_prediction_sources(self.controls(), self.frames))

    def test_indices_start_at_zero_and_increase_within_both_chunks(self):
        for sequence, index in ((0, 1), (1, 2), (3, 1), (4, 2)):
            with self.subTest(sequence=sequence, index=index):
                self.prepare(3)
                self.set_index(sequence, index)
                with self.assertRaisesRegex(ValueError, 'Ordered action sequence'):
                    self.controls()

    def test_a_retired_chunk_cannot_consume_another_action(self):
        row = self.targets[1]
        row['request_id'] = 'prefix-0'
        self.set_index(1, 1)
        row['predicted_source_target_monotonic'] = 100. + 2 / 6
        with self.assertRaisesRegex(ValueError, 'Ordered action sequence'):
            self.controls()

    def test_native_fifty_action_tail_must_match_prediction_journal(self):
        self.chunks[0]['reply']['actions'][49][0] += .01
        with self.assertRaisesRegex(ValueError, 'native 50-action prediction'):
            self.controls()

    def test_cup_profiles_cannot_use_native_gripper_retirement(self):
        for profile in ('state-history-physical-initial-cup-v1', 'state-history-physical-cup-tcp-v1',
                        'state-history-physical-cup-tracking-v1', 'state-history-physical-cup-setdown-v1'):
            with self.subTest(profile=profile):
                scope = dict(self.scope, observation_context_profile=profile)
                chunk = deepcopy(self.chunks[0])
                chunk['request']['observation_context_profile'] = profile
                grippers = [row for row in self.wire if row['method'] == 'MoveGripper']
                with self.assertRaisesRegex(ValueError, 'joint observation rule'):
                    _validate_prefix_ack(self.targets[0], self.acks[0], chunk, scope, grippers, 10)

    def test_cup_joint_profile_keeps_full_prefix_after_gripper(self):
        self.prepare(3, 3)
        profile = 'state-history-physical-initial-cup-v1'
        self.scope.update(observation_context_profile=profile, observation_rule='prefix',
                          initial_cup={'uv': [960., 540.], 'observed_monotonic': 90.,
                                       'camera_serial': 'CP1E54200015'})
        for chunk, prediction in zip(self.chunks, self.predictions):
            for request in (chunk['request'], prediction['request']):
                request.update(observation_context_profile=profile, initial_cup_uv=[960., 540.])
        self.acks[2]['prefix_retired_cause'] = 'execution_prefix_completed'
        self.assertEqual([row['action_index'] for row in self.controls()['ordered_timing']], [0, 1, 2, 0, 1])

    def test_legacy_full_prefix_materializes_without_new_rules(self):
        self.prepare(3, 3, legacy=True)
        self.chunks[1]['observation_after_monotonic'] = 1.
        journal = self.controls()
        self.assertEqual([row['action_index'] for row in journal['ordered_timing']], [0, 1, 2, 0, 1])
        self.assertTrue(all('prefix_consumed_steps' not in row and 'prefix_retired_cause' not in row
                            for row in journal['ordered_timing']))
        self.archive()
        result = materialize(self.root)
        self.assertEqual(verify_materialized(self.root), result)


if __name__ == '__main__':
    unittest.main()
