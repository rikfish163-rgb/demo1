"""Synthetic-only tests for stream evaluation recording and reconstruction."""
import base64
import ast
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fr5_jev.async_spool import AsyncFrameSpool
from fr5_jev.materialize_stream import materialize, verify_materialized
from fr5_jev import materialize_stream
from fr5_jev.stream_storage import CODEC, StreamStorage
from fr5_jev.test_async_spool import RUNNER
from fr5_jev import stream_protocol
from fr5_lerobot.tests.test_demo_writer import synthetic_sample
from lerobot_robot_fairino.frame_codec import encode_media
from lerobot_robot_fairino.pi05_bridge import observation_request
from lerobot_robot_fairino.stream_contract import CONTRACT
from lerobot_robot_fairino.stream_recording import StreamEvaluationRecording, record_readonly_diagnostic


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SyntheticRobot:
    is_connected = True

    def __init__(self):
        self.config = SimpleNamespace(camera=SimpleNamespace(expected_serial='SYNTHETIC_ONLY',
            timeout_s=.1, max_age_s=1.), max_sync_skew_s=.1)
        self.index = 0

    def get_sample(self):
        value = synthetic_sample(self.index)
        self.index += 1
        mono = time.monotonic() - .01
        value['camera']['metadata']['source_timestamp_monotonic'] = mono
        value['robot_state']['received_monotonic'] = mono + .004
        value['sync_skew_s'] = value['robot_state']['received_monotonic'] - mono
        return value


class FakeStorage:
    def __init__(self, local, server):
        self.root = local
        self.remote = server / local.name
        with patch('fr5_jev.async_spool.ssh_receiver_command', side_effect=lambda name, session: [
                sys.executable, '-u', '-c', RUNNER, str(server), name, session, '']):
            self.frame_spool = AsyncFrameSpool(local, local.name, max_unacked_age=10).start()
        self.codec_sha256 = sha(CODEC.read_bytes())
        self.protocol_sha256 = sha(Path(stream_protocol.__file__).read_bytes())
        self.sync_gate = threading.Event()
        self.sync_gate.set()

    def sync_tree(self):
        self.sync_gate.wait(4)
        for path in self.root.rglob('*'):
            rel = path.relative_to(self.root)
            if not path.is_file() or rel.parts[0] == 'frame-outbox':
                continue
            if rel.as_posix() in ('continuous/capture.jsonl', 'lerobot/meta/capture.jsonl'):
                continue
            dest = self.remote / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        wire = self.remote / 'wire'
        wire.mkdir()
        shutil.copyfile(CODEC, wire / 'frame_codec.py')
        shutil.copyfile(stream_protocol.__file__, wire / 'stream_protocol.py')
        update_manifest(self.remote, self.root, self.frame_spool.session)
        shutil.copyfile(self.remote/'control-manifest.json', self.root/'control-manifest.json')

    def materialize(self):
        return materialize(self.remote)

    def finalize_control(self):
        manifest = json.loads((self.root/'control-manifest.json').read_text())
        if manifest['summary_sha256'] != sha((self.root/'summary.json').read_bytes()):
            raise ValueError('Summary changed after manifest')
        shutil.copyfile(self.root/'control-manifest.json', self.remote/'control-manifest.json')


def update_manifest(root, local, session):
    entries = {}
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name.startswith(('wire/', 'materialized/', '.materializing-')) or name in (
                'frames.stream', 'INCOMPLETE', 'control-manifest.json'):
            continue
        raw = path.read_bytes()
        entries[name] = {'bytes': len(raw), 'sha256': sha(raw)}
    manifest = {'schema': 'fr5-final-control-manifest-v1', 'session': session,
        'local_episode': str(local), 'scope_sha256': entries['scope.json']['sha256'],
        'summary_sha256': entries['summary.json']['sha256'], 'files': entries}
    (root / 'control-manifest.json').write_text(json.dumps(manifest))


def wire_event(method, sequence, issued):
    target = None
    args = []
    if method == 'ServoJ':
        target = [1., 2., 3., 4., 5., 6.]
        args = [target, [0., 0., 0., 0.], 0., 0., .008, 0., 0., 0]
    elif method == 'MoveGripper':
        target = 76
        args = [1, 76, 20, 10, 10000, 1, 0, 0., 0, 0]
    return {'sequence': sequence, 'method': method, 'args': args, 'raw_args': args,
        'issued_monotonic': issued, 'returned_monotonic': issued+.001,
        'issued_unix': time.time(), 'returned_unix': time.time(),
        'return_code': 0, 'error': None, 'target_deg_percent': target}


class StreamRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fr5-eval-only-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root, self.server = base / 'episode', base / 'server'
        self.root.mkdir()
        self.server.mkdir()
        self.scope = {'task_instruction': 'synthetic cup task', 'checkpoint_sha256': 'a'*64,
                      'policy_contract': CONTRACT, 'dataset_role': 'policy_evaluation'}
        (self.root / 'scope.json').write_text(json.dumps(self.scope))
        self.storage = None

    def start(self):
        self.storage = FakeStorage(self.root, self.server)
        self.addCleanup(self.storage.sync_gate.set)
        self.addCleanup(self.storage.frame_spool.abort)
        with patch('lerobot_robot_fairino.stream_recording.StreamStorage.start_readonly',
                   return_value=self.storage):
            self.recording = StreamEvaluationRecording(self.root, self.scope).start(SyntheticRobot())
        return self.recording

    def prediction(self, sample):
        robot = sample['robot_state']
        valid = (robot['gripper_active'] == 1 and robot['gripper_fault'] == 0
                 and robot['gripper_fault_id'] == 0 and robot['gripper_motiondone'] in (1, 2, 3)
                 and robot['gripper_speed'] == 0)
        state = [math.radians(x) for x in robot['joints']] + [robot['gripper_position']/100 if valid else 0., float(valid)]
        _, jpeg, _ = encode_media(sample['camera']['color_rgb'], sample['camera']['depth_mm'])
        request = {'schema': 'fr5-stream-request-v1', 'policy_contract': CONTRACT['name'],
            'id': 'synthetic-request', 'source_clock': 'host_monotonic',
            'source_time': sample['camera']['metadata']['source_timestamp_monotonic'],
            'task': self.scope['task_instruction'], 'state': state,
            'rgb_jpeg_base64': base64.b64encode(jpeg).decode()}
        reply = {'schema': 'fr5-stream-reply-v1', 'id': request['id'], 'checkpoint_sha256': 'a'*64,
                 'policy_contract': CONTRACT['name'], 'actions': [[0.]*7 for _ in range(50)]}
        return request, reply

    def snapshot(self, arm=0, grip=0):
        return {'servo_commands': arm, 'gripper_commands': grip, 'pending_callbacks': 0,
                'inflight_commands': 0, 'startup_inflight': False,
                'undelivered_callbacks': [], 'callback_failure': None, 'workers_alive': []}

    def wait_frames(self, count=3):
        until = time.monotonic()+4
        while self.recording._frames < count and time.monotonic() < until:
            time.sleep(.02)
        self.assertGreaterEqual(self.recording._frames, count)

    def local_observation_recorder(self):
        recorder = StreamEvaluationRecording(self.root, self.scope)
        recorder.capture = SimpleNamespace(assert_healthy=Mock())
        recorder.journal = SimpleNamespace(error=None)
        recorder.robot = SimpleNamespace(config=SimpleNamespace(
            camera=SimpleNamespace(timeout_s=.5), state_timeout_s=.5))
        return recorder

    @staticmethod
    def timed_sample(camera_time, robot_time, *, motiondone=1):
        return {'camera': {'metadata': {'source_timestamp_monotonic': camera_time}},
                'robot_state': {'received_monotonic': robot_time, 'gripper_motiondone': motiondone},
                'sync_clock': 'host_monotonic', 'sync_skew_s': robot_time-camera_time}

    def test_post_ack_observation_waits_for_both_clocks_and_keeps_invalid_gripper(self):
        recorder = self.local_observation_recorder()
        boundary = time.monotonic()-.1
        old = self.timed_sample(boundary-.02, boundary-.01)
        recorder._sample(old)
        self.assertEqual(recorder.observation(), old)  # Default keeps its existing behavior.
        stop, started, done = threading.Event(), threading.Event(), threading.Event()
        result = {}
        def wait_observation():
            started.set()
            try:
                result['sample'] = recorder.observation(after_monotonic=boundary, stop_event=stop)
            except BaseException as exc:
                result['error'] = exc
            finally:
                done.set()
        thread = threading.Thread(target=wait_observation)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(recorder.wake_observation_waiters)
        self.addCleanup(stop.set)
        self.assertTrue(started.wait(1))
        self.assertFalse(done.wait(.06))  # Neither clock is after the ACK.
        recorder._sample(self.timed_sample(boundary+.01, boundary-.005))
        self.assertFalse(done.wait(.06))  # New camera with old robot state is insufficient.
        fresh = self.timed_sample(boundary+.02, boundary+.03, motiondone=0)
        recorder._sample(fresh)
        self.assertTrue(done.wait(1))
        self.assertNotIn('error', result)
        self.assertEqual(result['sample'], fresh)
        self.assertEqual(result['sample']['robot_state']['gripper_motiondone'], 0)

    def test_post_ack_observation_cancels_and_propagates_device_failure(self):
        recorder = self.local_observation_recorder()
        boundary = time.monotonic()-.1
        recorder._sample(self.timed_sample(boundary-.02, boundary-.01))
        stop, done = threading.Event(), threading.Event()
        result = {}
        def wait_observation():
            try:
                recorder.observation(after_monotonic=boundary, stop_event=stop)
            except BaseException as exc:
                result['error'] = exc
            finally:
                done.set()
        thread = threading.Thread(target=wait_observation)
        thread.start()
        stop.set()
        recorder.wake_observation_waiters()
        self.assertTrue(done.wait(1))
        thread.join(1)
        self.assertIsInstance(result.get('error'), InterruptedError)
        recorder.capture.assert_healthy.side_effect = [None, RuntimeError('camera failed')]
        with self.assertRaisesRegex(RuntimeError, 'camera failed'):
            recorder.observation(after_monotonic=boundary)
        recorder.capture.assert_healthy.side_effect = None
        recorder.robot.config.camera.timeout_s = .03
        recorder.robot.config.state_timeout_s = .03
        with self.assertRaisesRegex(TimeoutError, 'after target acknowledgement'):
            recorder.observation(after_monotonic=boundary)

    def test_concurrent_callbacks_stop_without_waiting_for_remote_and_materialize(self):
        recorder = self.start()
        request, reply = self.prediction(recorder.observation())
        recorder.record_prediction(request, reply, {'synthetic': True})
        audit = {'request': {k: v for k, v in request.items() if k != 'rgb_jpeg_base64'}, 'reply': reply}
        (self.root / 'chunk-00000.json').write_text(json.dumps(audit))
        target = {'sequence': 0, 'request_id': request['id'], 'action_index': 0,
                  'raw_action': [0.]*7, 'target': [0.]*7,
                  'target_monotonic': request['source_time']+1/6,
                  'submitted_monotonic': time.monotonic()}
        (self.root / 'submitted-targets.jsonl').write_text(json.dumps(target)+'\n')
        issued = time.monotonic()-.02
        rows = [wire_event('ServoMoveStart', 1, issued), wire_event('ServoJ', 2, issued+.002),
                wire_event('MoveGripper', 3, issued+.004), wire_event('StopMotion', 4, issued+.006),
                wire_event('ServoMoveEnd', 5, issued+.008)]
        threads = [threading.Thread(target=recorder.command_callback, args=(row,)) for row in reversed(rows)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.wait_frames()
        self.storage.sync_gate.clear()
        began = time.monotonic()
        summary = recorder.finish('operator_stop', controller_snapshot=self.snapshot(1, 1))
        self.assertLess(time.monotonic()-began, 1.5)
        self.assertEqual(summary['storage_status'], 'pending_archive')
        self.assertEqual(summary['termination_source'], 'operator_stop')
        self.assertEqual(summary['commands'], 2)
        self.assertIsNone(summary['failed'])
        self.assertEqual(summary['summary_stage'], 'provisional')
        self.assertFalse(summary['task_success_verified'])
        self.assertTrue(summary['wire_callbacks_drained'])
        self.assertTrue(recorder._finalize_thread.is_alive())
        self.storage.sync_gate.set()
        recorder.wait_finalized(timeout=8)
        marker = json.loads((self.root/'frame-outbox/archive-finalized.json').read_text())
        self.assertEqual(marker['schema'], 'fr5-evaluation-archive-finalized-v1')
        self.assertEqual(marker['summary_sha256'], sha((self.root/'summary.json').read_bytes()))
        self.assertEqual(marker['control_manifest_sha256'], sha((self.root/'control-manifest.json').read_bytes()))
        result = recorder.materialize()
        self.assertEqual(result['dataset_role'], 'policy_evaluation')
        self.assertFalse(result['task_success_verified'])
        self.assertEqual(result['target_alignment'], 'verified_against_wire_and_frame_clocks')
        self.assertEqual(verify_materialized(self.storage.remote), result)
        output = self.storage.remote / 'materialized'
        final_summary = json.loads((output/'summary.json').read_text())
        self.assertEqual(final_summary['summary_stage'], 'final_control_snapshot')
        self.assertEqual(final_summary['termination_source'], 'operator_stop')
        self.assertEqual(final_summary['commands'], 2)
        self.assertEqual(result['summary_sha256'], sha((output/'summary.json').read_bytes()))
        self.assertEqual((output/'summary.json').read_bytes(), (self.root/'summary.json').read_bytes())
        stored_request = json.loads((output/'stream-prediction/000000.jsonl').read_text())['request']
        self.assertEqual(base64.b64decode(stored_request['rgb_jpeg_base64']),
                         base64.b64decode(request['rgb_jpeg_base64']))
        alignment = [json.loads(x) for x in (output/'stream-frame-commands.jsonl').read_text().splitlines()]
        self.assertTrue(any(row['last_gripper_attempt'] and row['last_gripper_attempt']['args'][1] == 76
                            for row in alignment))
        links = json.loads((output/'stream-prediction-sources.json').read_text())
        self.assertTrue(any(link['request_jpeg_hash_verified'] for link in links))
        self.assertTrue((output/'submitted-targets.jsonl').exists())
        self.assertTrue((output/'chunk-00000.json').exists())

    def test_ordered_chunks_materialize_distinct_source_and_submission_times(self):
        self.scope.update(execution_mode='ordered_complete_chunk', execution_chunk_steps=50, prefix_skip=False)
        (self.root/'scope.json').write_text(json.dumps(self.scope))
        recorder = self.start()
        request, reply = self.prediction(recorder.observation())
        submitted = []
        for chunk_number in range(2):
            request_item = dict(request, id=f'ordered-{chunk_number}')
            reply_item = dict(reply, id=request_item['id'])
            recorder.record_prediction(request_item, reply_item)
            audit_request = {k: v for k, v in request_item.items() if k != 'rgb_jpeg_base64'}
            (self.root/f'chunk-{chunk_number:05d}.json').write_text(json.dumps({
                'request': audit_request, 'reply': reply_item,
                'received_monotonic': request['source_time']+.2}))
            for index in range(50 if chunk_number == 0 else 2):
                n = len(submitted)
                actual = request['source_time']+2.+n/6
                submitted.append({'sequence': n, 'request_id': request_item['id'], 'action_index': index,
                    'raw_action': [0.]*7, 'target': [0.]*7,
                    'predicted_source_target_monotonic': request['source_time']+(index+1)/6,
                    'submitted_monotonic': actual-1/6, 'target_monotonic': actual})
        (self.root/'submitted-targets.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in submitted))
        self.wait_frames()
        recorder.finish('operator_stop', controller_snapshot=self.snapshot())
        recorder.wait_finalized(timeout=8)
        result = recorder.materialize()
        output = self.storage.remote/'materialized'
        timing_path = output/'stream-target-timing.jsonl'
        timing = [json.loads(line) for line in timing_path.read_text().splitlines()]
        self.assertEqual(len(timing), 52)
        self.assertAlmostEqual(timing[0]['target_time_offset_s'], 2.-1/6)
        self.assertEqual((timing[49]['action_index'], timing[50]['action_index']), (49, 0))
        self.assertEqual(result['stream_target_timing_sha256'], sha(timing_path.read_bytes()))
        self.assertEqual(verify_materialized(self.storage.remote), result)
        timing_path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'ordered target timing'):
            verify_materialized(self.storage.remote)

    def test_ordered_policy_prefix_materializes_and_verifies_native_evidence(self):
        self.scope.update(execution_mode='ordered_policy_prefix', execution_chunk_steps=3,
                          prediction_chunk_steps=50, prefix_skip=False)
        (self.root/'scope.json').write_text(json.dumps(self.scope))
        recorder = self.start()
        request, reply = self.prediction(recorder.observation())
        submitted = []
        for chunk_number, count in enumerate((3, 2)):
            request_item = dict(request, id=f'prefix-{chunk_number}')
            reply_item = dict(reply, id=request_item['id'])
            recorder.record_prediction(request_item, reply_item)
            audit_request = {k: v for k, v in request_item.items() if k != 'rgb_jpeg_base64'}
            (self.root/f'chunk-{chunk_number:05d}.json').write_text(json.dumps({
                'request': audit_request, 'reply': reply_item, 'execution_prefix_steps': 3,
                'received_monotonic': request['source_time']+.2}))
            for index in range(count):
                n = len(submitted)
                actual = request['source_time']+2.+n/6
                submitted.append({'sequence': n, 'request_id': request_item['id'], 'action_index': index,
                    'raw_action': [0.]*7, 'target': [0.]*7, 'execution_prefix_steps': 3,
                    'prediction_chunk_steps': 50,
                    'predicted_source_target_monotonic': request['source_time']+(index+1)/6,
                    'submitted_monotonic': actual-1/6, 'target_monotonic': actual})
        (self.root/'submitted-targets.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in submitted))
        recorder.command_callback(wire_event('ServoJ', 1, time.monotonic()-.01))
        self.wait_frames()
        recorder.finish('operator_stop', controller_snapshot=self.snapshot(1, 0))
        recorder.wait_finalized(timeout=8)
        result = recorder.materialize()
        output = self.storage.remote/'materialized'
        timing_path = output/'stream-target-timing.jsonl'
        timing = [json.loads(line) for line in timing_path.read_text().splitlines()]
        self.assertEqual([row['action_index'] for row in timing], [0, 1, 2, 0, 1])
        self.assertAlmostEqual(timing[3]['target_time_offset_s'], 2.+3/6-1/6)
        self.assertEqual(result['stream_target_timing_sha256'], sha(timing_path.read_bytes()))
        self.assertFalse(result['task_success_verified'])
        self.assertEqual(verify_materialized(self.storage.remote), result)
        timing_path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'ordered target timing'):
            verify_materialized(self.storage.remote)

    def test_reject_forged_native_wire_even_with_updated_hashes(self):
        recorder = self.start()
        row = wire_event('ServoJ', 1, time.monotonic()-.01)
        recorder.command_callback(row)
        self.wait_frames()
        recorder.finish('operator_stop', controller_snapshot=self.snapshot(1, 0))
        recorder.wait_finalized(timeout=8)
        remote = self.storage.remote
        path = remote/'stream-wire/000000.jsonl'
        event = json.loads(path.read_text())
        event['args'][0][0] = event['raw_args'][0][0] = 999
        path.write_text(json.dumps(event)+'\n')
        index_path = remote/'stream-wire/index.json'
        index = json.loads(index_path.read_text())
        raw = path.read_bytes()
        index['chunks'][0].update(bytes=len(raw), sha256=sha(raw))
        index_path.write_text(json.dumps(index))
        update_manifest(remote, self.root, self.storage.frame_spool.session)
        with self.assertRaisesRegex(ValueError, 'ServoJ target'):
            materialize(remote)
        self.assertFalse((remote/'materialized').exists())

    def test_unknown_result_is_preserved_and_not_success(self):
        recorder = self.start()
        row = wire_event('ServoJ', 1, time.monotonic()-.01)
        row.update(return_code=None, error='TimeoutError: reply unknown')
        recorder.command_callback(row)
        self.wait_frames()
        recorder.finish('error', error='response lost', controller_snapshot=self.snapshot(1, 0))
        recorder.wait_finalized(timeout=8)
        result = recorder.materialize()
        self.assertFalse(result['task_success_verified'])
        rows = [json.loads(x) for x in (self.storage.remote/'materialized/stream-wire/000000.jsonl').read_text().splitlines()]
        self.assertIsNone(rows[0]['return_code'])
        self.assertIn('unknown', rows[0]['error'])

    def test_unfinished_callback_log_is_materialized_only_as_incomplete(self):
        recorder = self.start()
        recorder.command_callback(wire_event('ServoJ', 1, time.monotonic()-.01))
        self.wait_frames()
        snapshot = self.snapshot(1, 0)
        snapshot['pending_callbacks'] = 1
        summary = recorder.finish('error', error='callback pending', controller_snapshot=snapshot)
        self.assertFalse(summary['wire_callbacks_drained'])
        recorder.wait_finalized(timeout=8)
        result = recorder.materialize()
        self.assertEqual(result['target_alignment'], 'incomplete_missing_wire_callbacks')
        self.assertFalse(result['task_success_verified'])
        self.assertEqual(verify_materialized(self.storage.remote), result)
        # The real storage client accepts this specifically labelled policy
        # evaluation receipt, but never treats the incomplete alignment as a
        # successfully demonstrated task.
        shutil.copyfile(self.storage.remote/'control-manifest.json', self.root/'control-manifest.json')
        client = StreamStorage(self.root, sha((self.root/'scope.json').read_bytes()),
                               self.storage.frame_spool, required_bytes=1)
        with patch('fr5_jev.stream_storage.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(result).encode())):
            self.assertEqual(client.materialize(), result)

    def test_prediction_source_jpeg_hash_mismatch_is_rejected(self):
        recorder = self.start()
        request, reply = self.prediction(recorder.observation())
        request['rgb_jpeg_base64'] = base64.b64encode(b'another image').decode()
        recorder.record_prediction(request, reply)
        self.wait_frames()
        recorder.finish('operator_stop', controller_snapshot=self.snapshot())
        recorder.wait_finalized(timeout=8)
        with self.assertRaisesRegex(ValueError, 'Prediction JPEG'):
            recorder.materialize()
        self.assertFalse((self.storage.remote/'materialized').exists())

    def test_many_simultaneous_callbacks_remain_complete_and_sort_by_issue_time(self):
        recorder = self.start()
        base = time.monotonic()-1.
        rows = [wire_event('ServoJ', index+1, base+(index+1)*.000001) for index in range(320)]
        errors = []
        def send(items):
            try:
                for item in reversed(items):
                    recorder.command_callback(item)
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=send, args=(rows[offset::8],)) for offset in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(errors, [])
        self.wait_frames()
        summary = recorder.finish('operator_stop', controller_snapshot=self.snapshot(320, 0))
        self.assertEqual(summary['wire_records'], 320)
        self.assertTrue(summary['wire_callbacks_drained'])
        recorder.wait_finalized(timeout=8)
        result = recorder.materialize()
        self.assertEqual(result['target_alignment'], 'verified_against_wire_and_frame_clocks')
        alignment = (self.storage.remote/'materialized/stream-frame-commands.jsonl').read_text().splitlines()
        self.assertEqual(json.loads(alignment[-1])['last_servoj_attempt']['sequence'], 320)

    def test_tampering_materialized_frame_alignment_is_rejected(self):
        recorder = self.start()
        recorder.command_callback(wire_event('ServoJ', 1, time.monotonic()-.01))
        self.wait_frames()
        recorder.finish('operator_stop', controller_snapshot=self.snapshot(1, 0))
        recorder.wait_finalized(timeout=8)
        recorder.materialize()
        alignment = self.storage.remote/'materialized/stream-frame-commands.jsonl'
        alignment.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'alignment'):
            verify_materialized(self.storage.remote)

    def test_corrupted_stored_prediction_jpeg_is_rejected_even_with_new_file_hashes(self):
        recorder = self.start()
        request, reply = self.prediction(recorder.observation())
        recorder.record_prediction(request, reply)
        self.wait_frames()
        recorder.finish('operator_stop', controller_snapshot=self.snapshot())
        recorder.wait_finalized(timeout=8)
        remote = self.storage.remote
        path = remote/'stream-prediction/000000.jsonl'
        event = json.loads(path.read_text())
        event['request']['rgb_jpeg_base64'] = base64.b64encode(b'not the model input').decode()
        path.write_text(json.dumps(event)+'\n')
        index_path = remote/'stream-prediction/index.json'
        index = json.loads(index_path.read_text())
        raw = path.read_bytes()
        index['chunks'][0].update(bytes=len(raw), sha256=sha(raw))
        index_path.write_text(json.dumps(index))
        update_manifest(remote, self.root, self.storage.frame_spool.session)
        with self.assertRaisesRegex(ValueError, 'JPEG bytes or hash'):
            materialize(remote)
        self.assertFalse((remote/'materialized').exists())

    def test_storage_start_failure_preserves_original_error_without_retry(self):
        recorder = StreamEvaluationRecording(self.root, self.scope)
        with patch('lerobot_robot_fairino.stream_recording.StreamStorage.start_readonly',
                   side_effect=RuntimeError('synthetic storage unavailable')) as start:
            with self.assertRaisesRegex(RuntimeError, 'synthetic storage unavailable'):
                recorder.start(SyntheticRobot())
        summary = recorder.finish('error', error='synthetic storage unavailable')
        self.assertEqual(start.call_count, 1)
        self.assertEqual(summary['storage_status'], 'startup_failed')
        self.assertEqual(summary['recorded_frames'], 0)
        self.assertFalse(summary['task_success_verified'])
        self.assertIn('synthetic storage unavailable', summary['error'])

    def test_readonly_diagnostic_has_frames_but_no_controller_actions_or_predictions(self):
        self.scope['metadata'] = {'model_execution': False}
        self.scope['task_instruction'] = '记录器连接诊断（不执行动作）'
        (self.root/'scope.json').write_text(json.dumps(self.scope))
        recorder = self.start()
        self.wait_frames()
        with self.assertRaises(PermissionError):
            recorder.command_callback(wire_event('ServoJ', 1, time.monotonic()-.01))
        request, reply = self.prediction(recorder.observation())
        with self.assertRaises(PermissionError):
            recorder.record_prediction(request, reply)
        summary = recorder.finish('readonly_diagnostic_complete', controller_snapshot=None)
        self.assertEqual(summary['outcome'], 'observed')
        self.assertEqual((summary['commands'], summary['wire_records'], summary['prediction_records']), (0, 0, 0))
        self.assertIsNone(summary['wire_callbacks_drained'])
        self.assertFalse(summary['awaiting_whole_user_review'])
        self.assertFalse(summary['training_ready'])
        recorder.wait_finalized(timeout=8)
        result = recorder.materialize()
        self.assertEqual(result['target_alignment'], 'not_applicable_read_only')
        self.assertGreaterEqual(result['frames'], 3)
        self.assertEqual(verify_materialized(self.storage.remote), result)
        client = StreamStorage(self.root, sha((self.root/'scope.json').read_bytes()),
                               self.storage.frame_spool, required_bytes=1)
        with patch('fr5_jev.stream_storage.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(result).encode())):
            self.assertEqual(client.materialize(), result)

    def test_readonly_helper_uses_connected_observer_without_constructing_controller(self):
        root = self.root.parent / 'readonly-helper'
        observer = SyntheticRobot()
        def make_storage(local, _scope_sha, **_kwargs):
            self.storage = FakeStorage(local, self.server)
            self.addCleanup(self.storage.sync_gate.set)
            self.addCleanup(self.storage.frame_spool.abort)
            return self.storage
        with patch('lerobot_robot_fairino.stream_recording.new_stream_recording_path', return_value=root), \
             patch('lerobot_robot_fairino.stream_recording.StreamStorage.start_readonly', side_effect=make_storage):
            recorder = record_readonly_diagnostic(observer, 'readonly-helper', 'a'*64, seconds=.4)
        summary = recorder.wait_finalized(timeout=8)
        self.assertEqual(summary['commands'], 0)
        self.assertEqual(summary['prediction_records'], 0)
        self.assertFalse(summary['model_execution'])
        self.assertEqual(recorder.materialize()['target_alignment'], 'not_applicable_read_only')

    def test_recovery_injects_missing_helpers_for_frozen_stream_module(self):
        self.scope['metadata'] = {'model_execution': False}
        (self.root/'scope.json').write_text(json.dumps(self.scope))
        recorder = self.start()
        self.wait_frames()
        recorder.finish('readonly_diagnostic_complete', controller_snapshot=None)
        recorder.wait_finalized(timeout=8)
        expected = recorder.materialize()
        remote = self.storage.remote
        # Keep the archived control rules intact while removing the helpers a
        # historical immutable module might not contain.
        source = Path(materialize_stream.__file__).read_text()
        lines = source.splitlines(keepends=True)
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                     and node.name in {'_finite', '_stream_frame_commands', '_stream_prediction_sources',
                                       '_validate_stream_targets'}]
        for node in sorted(functions, key=lambda item: item.lineno, reverse=True):
            del lines[node.lineno-1:node.end_lineno]
        (remote/'wire/materialize_stream.py').write_text(''.join(lines))
        client = StreamStorage(self.root, sha((self.root/'scope.json').read_bytes()),
                               self.storage.frame_spool, required_bytes=1)
        client.remote_root = remote
        actual_run = subprocess.run
        def local_runner(command, **kwargs):
            executable = shlex.split(command[-1])
            executable[0] = sys.executable
            return actual_run(executable, **kwargs)
        before = {p: sha(p.read_bytes()) for p in remote.rglob('*') if p.is_file()}
        with patch('fr5_jev.stream_storage.subprocess.run', side_effect=local_runner):
            self.assertEqual(client.materialize(), expected)
        self.assertEqual(before, {p: sha(p.read_bytes()) for p in remote.rglob('*') if p.is_file()})

    def test_policy_input_jpeg_matches_the_recorded_frame_encoder(self):
        sample = synthetic_sample(0)
        sample['camera']['color_rgb'] = np.tile(np.arange(640, dtype=np.uint8), (360, 1))[:, :, None].repeat(3, axis=2)
        sample['camera']['depth_mm'] = np.full((360, 640), 500, np.float32)
        captured = time.monotonic()-.01
        sample['camera']['metadata']['source_timestamp_monotonic'] = captured
        sample['robot_state']['received_monotonic'] = captured+.004
        sample['sync_skew_s'] = sample['robot_state']['received_monotonic']-captured
        request = observation_request(sample, 'synthetic cup task')
        metadata, jpeg, _ = encode_media(sample['camera']['color_rgb'], sample['camera']['depth_mm'])
        self.assertEqual(base64.b64decode(request['rgb_jpeg_base64']), jpeg)
        self.assertEqual(metadata['rgb_sha256'], sha(jpeg))

    def test_control_storage_limit_is_reported_instead_of_silently_losing_wire(self):
        (self.root/'synthetic-control.txt').write_bytes(b'x'*3000)
        with patch('lerobot_robot_fairino.stream_recording.MAX_LOCAL_CONTROL', 4096), \
             patch('lerobot_robot_fairino.stream_recording._CHUNK_BYTES', 1024):
            recorder = self.start()
            recorder.command_callback(wire_event('ServoJ', 1, time.monotonic()-.01))
            deadline = time.monotonic()+3
            while recorder.journal.error is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIn('bounded local storage limit', recorder.journal.error)
            with self.assertRaisesRegex(RuntimeError, 'journal incomplete'):
                recorder.finish('error', error='storage capacity', controller_snapshot=self.snapshot(1, 0))
            recorder._finalize_thread.join(8)
            self.assertFalse(recorder._finalize_thread.is_alive())
            summary = json.loads((self.root/'summary.json').read_text())
            self.assertEqual(summary['outcome'], 'interrupted')
            self.assertEqual(summary['storage_status'], 'journal_failed_local_only')
            self.assertFalse(summary['task_success_verified'])

    def test_lost_final_ack_retries_manifest_without_changing_reviewed_summary(self):
        recorder = self.start()
        self.wait_frames()
        original_sync = self.storage.sync_tree
        def lost_response():
            original_sync()
            raise TimeoutError('synthetic lost acknowledgement')
        with patch.object(self.storage, 'sync_tree', side_effect=lost_response):
            recorder.finish('operator_stop', controller_snapshot=self.snapshot())
            with self.assertRaisesRegex(RuntimeError, 'lost acknowledgement'):
                recorder.wait_finalized(timeout=8)
        before = (self.root/'summary.json').read_bytes()
        self.assertTrue((self.root/'frame-outbox/archive-error.json').exists())
        self.assertFalse((self.root/'archive-error.json').exists())
        self.assertFalse((self.root/'frame-outbox/archive-finalized.json').exists())
        recorder.finalize_storage()
        self.assertEqual((self.root/'summary.json').read_bytes(), before)
        self.assertTrue((self.root/'frame-outbox/archive-finalized.json').exists())
        self.assertEqual(recorder.materialize()['summary_sha256'], sha(before))


if __name__ == '__main__':
    unittest.main()
