"""Closed archive recovery tests; no network, capture or robot commands."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fr5_jev import stream_protocol, stream_storage
from fr5_jev.stream_storage import StreamStorage
from fr5_runtime import console_data, system


class ArchiveStreamTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='fr5-system-archive-')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.root = self.base / 'whole-task-interrupted'
        self.root.mkdir()
        self.outbox = self.root / 'frame-outbox'
        self.outbox.mkdir()
        self.session = 'c' * 32
        self.write('scope.json', {'whole_task': True, 'training_ready': False})
        scope_sha = hashlib.sha256((self.root / 'scope.json').read_bytes()).hexdigest()
        self.write('summary.json', {'scope_sha256': scope_sha, 'whole_task': True,
                   'outcome': 'interrupted', 'interrupted': True,
                   'task_success_verified': False, 'training_ready': False})
        self.closed = {
            'schema': 'fr5-stream-closed-v1', 'session': self.session,
            'accepted_frames': 183, 'last_durable_sequence': 182,
            'unacknowledged_frames': 0, 'unacknowledged_bytes': 0, 'error': None,
            'local_episode': str(self.root),
            'codec_sha256': hashlib.sha256(stream_storage.CODEC.read_bytes()).hexdigest(),
            'protocol_sha256': hashlib.sha256(Path(stream_protocol.__file__).read_bytes()).hexdigest()}
        self.write('stream-closed.json', self.closed)
        self.write('frame-outbox/stream-session.json', {
            'schema': 'fr5-local-outbox-v1', 'name': self.root.name, 'session': self.session})
        self.write('evidence.json', {'observed': True})
        self.original = {name: (self.root / name).read_bytes() for name in
                         ('scope.json', 'summary.json', 'stream-closed.json', 'evidence.json')}
        self.proof = {'schema': 'fr5-stream-materialization-v1', 'session': self.session,
                      'frames': 183, 'control_verification': 'verified',
                      'target_alignment': 'verified_against_wire_and_frame_clocks'}
        self.uploads = []
        self.order = []
        self.remote_state = {'materialized': False, 'control_manifest': False}
        for target, value in (('fr5_runtime.system.SPOOL', self.base),
                              ('fr5_jev.stream_storage.LOCAL_STREAM_BASE', self.base)):
            guard = patch(target, value)
            guard.start()
            self.addCleanup(guard.stop)
        for target in ('fr5_jev.stream_storage.AsyncFrameSpool.start',
                       'fr5_jev.stream_storage.subprocess.Popen',
                       'fr5_jev.stream_storage.subprocess.run',
                       'fr5_runtime.system.robot_lease', 'fr5_runtime.system.observe'):
            guard = patch(target, side_effect=AssertionError('Unexpected process, capture or robot access'))
            guard.start()
            self.addCleanup(guard.stop)
        upload = patch.object(StreamStorage, '_push', autospec=True, side_effect=self.upload)
        upload.start()
        self.addCleanup(upload.stop)
        materialize = patch.object(StreamStorage, 'materialize', autospec=True,
                                   side_effect=self.materialize)
        self.materializer = materialize.start()
        self.addCleanup(materialize.stop)
        replay = patch.object(console_data, 'render_replay', side_effect=self.render)
        self.renderer = replay.start()
        self.addCleanup(replay.stop)
        remote = patch.object(system, 'remote_json', return_value=self.remote_state)
        self.remote = remote.start()
        self.addCleanup(remote.stop)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False) + '\n')

    def upload(self, storage, payloads):
        self.order.append('upload')
        self.uploads.append(dict(payloads))
        return {'session': storage.frame_spool.session, 'files': {
            name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}}

    def materialize(self, storage, *, timeout):
        self.order.append('materialize')
        self.assertEqual(storage.frame_spool.session, self.session)
        self.assertEqual(storage.frame_spool.status()['accepted_frames'], 183)
        self.assertEqual(timeout, 600)
        return self.proof

    def render(self, name):
        self.order.append('render')
        self.assertEqual(name, self.root.name)
        self.assertEqual(json.loads((self.outbox / 'materialization-receipt.json').read_text()),
                         self.proof)
        return {'frames': 183, 'all_frames_preserved': True}

    def archive(self, *, render=False, cli=False):
        with patch.object(system.time, 'monotonic', side_effect=[0, 181]), \
                patch.object(system.time, 'sleep', side_effect=AssertionError('Closed archive waited')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            if cli:
                argv = ['fr5-system', 'archive-stream', '--episode', self.root.name]
                if render:
                    argv.append('--render')
                with patch('sys.argv', argv):
                    system.main()
                return json.loads('\n'.join(output.getvalue().splitlines()[1:]))
            return system.archive_stream(self.root.name, render=render)

    def assert_records_unchanged(self):
        for name, raw in self.original.items():
            self.assertEqual((self.root / name).read_bytes(), raw, name)

    def test_public_closed_whole_task_recovers_without_background_marker(self):
        result = self.archive(render=True, cli=True)
        self.assertEqual(result['materialization'], self.proof)
        self.assertTrue(result['replay']['all_frames_preserved'])
        self.assertEqual(self.order, ['upload', 'upload', 'materialize', 'render'])
        self.assertTrue((self.root / 'control-manifest.json').is_file())
        self.assert_records_unchanged()

    def test_existing_manifest_is_resumed_without_rewriting_control_records(self):
        StreamStorage.from_closed(self.root).sync_tree()
        manifest = (self.root / 'control-manifest.json').read_bytes()
        self.uploads.clear()
        self.archive()
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(set(self.uploads[0]), {'scope.json', 'control-manifest.json'})
        self.assertEqual(self.uploads[0]['control-manifest.json'], manifest)
        self.assertEqual((self.root / 'control-manifest.json').read_bytes(), manifest)
        self.assert_records_unchanged()

    def test_existing_materialization_is_reused_after_local_cache_release(self):
        cache = self.root / 'view-01/scene.npz'
        cache.parent.mkdir()
        cache.write_bytes(b'archived-checkpoint-cache')
        StreamStorage.from_closed(self.root).sync_tree()
        manifest = (self.root / 'control-manifest.json').read_bytes()
        cache.unlink()
        self.uploads.clear()
        self.remote_state.update(materialized=True, control_manifest=True)
        self.assertEqual(self.archive()['materialization'], self.proof)
        self.assertEqual(self.uploads, [])
        self.assertFalse(cache.exists())
        self.assertEqual((self.root / 'control-manifest.json').read_bytes(), manifest)
        self.assert_records_unchanged()

    def test_durable_remote_control_snapshot_needs_no_repeat_upload(self):
        StreamStorage.from_closed(self.root).sync_tree()
        self.uploads.clear()
        self.remote_state.update(control_manifest=True)
        self.assertEqual(self.archive()['materialization'], self.proof)
        self.assertEqual(self.uploads, [])
        self.assert_records_unchanged()

    def test_closed_recording_can_resume_after_a_background_archive_error(self):
        self.write('frame-outbox/archive-error.json', {'error': 'control upload timeout'})
        self.assertEqual(self.archive()['materialization'], self.proof)
        self.assert_records_unchanged()

    def test_incomplete_or_failed_closed_stream_is_not_uploaded_or_rendered(self):
        for change in ({'error': 'receiver failed'}, {'unacknowledged_frames': 1},
                       {'unacknowledged_bytes': 1}, {'last_durable_sequence': 181}):
            with self.subTest(change=change):
                self.write('stream-closed.json', {**self.closed, **change})
                packet = self.outbox / '000183.packet'
                packet.write_bytes(b'retained-unacknowledged-frame')
                with self.assertRaisesRegex(ValueError, 'matching durable acknowledgements'):
                    self.archive(render=True)
                self.assertEqual(self.uploads, [])
                self.materializer.assert_not_called()
                self.renderer.assert_not_called()
                self.assertEqual(packet.read_bytes(), b'retained-unacknowledged-frame')
                self.assertFalse((self.root / 'control-manifest.json').exists())
                self.assertFalse((self.outbox / 'materialization-receipt.json').exists())

    def test_materialization_failure_resumes_the_same_manifest_on_retry(self):
        self.materializer.side_effect = [RuntimeError('remote archive unavailable'), self.proof]
        with self.assertRaisesRegex(RuntimeError, 'remote archive unavailable'):
            self.archive(render=True)
        self.assertFalse((self.outbox / 'materialization-receipt.json').exists())
        self.renderer.assert_not_called()
        manifest = (self.root / 'control-manifest.json').read_bytes()
        result = self.archive(render=True)
        self.assertEqual(result['materialization'], self.proof)
        self.assertEqual((self.root / 'control-manifest.json').read_bytes(), manifest)
        self.assert_records_unchanged()

    def test_unclosed_recording_waits_or_reports_the_archive_error(self):
        (self.root / 'stream-closed.json').unlink()
        with self.assertRaises(TimeoutError):
            self.archive()
        self.write('frame-outbox/archive-error.json', {'error': 'receiver unavailable'})
        with self.assertRaisesRegex(RuntimeError, 'receiver unavailable'):
            self.archive()
        self.assertEqual(self.uploads, [])
        self.materializer.assert_not_called()

    def test_evaluation_closed_receipt_waits_for_the_final_control_snapshot(self):
        scope = {'whole_task': True, 'dataset_role': 'policy_evaluation'}
        self.write('scope.json', scope)
        scope_sha = hashlib.sha256((self.root / 'scope.json').read_bytes()).hexdigest()
        self.write('summary.json', {'scope_sha256': scope_sha, 'outcome': 'interrupted',
                                   'summary_stage': 'provisional'})
        with self.assertRaises(TimeoutError):
            self.archive()
        self.assertEqual(self.uploads, [])
        self.remote.assert_not_called()
        self.materializer.assert_not_called()

    def test_background_completion_marker_keeps_the_existing_verified_path(self):
        self.write('frame-outbox/archive-finalized.json', {'session': self.session})
        result = self.archive(render=True)
        self.assertEqual(result['materialization'], self.proof)
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.order, ['materialize', 'render'])
        self.assert_records_unchanged()


if __name__ == '__main__':
    unittest.main()
