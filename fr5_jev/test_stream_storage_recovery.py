"""Offline tests for bounded control uploads and closed-session recovery."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import stream_protocol, stream_storage
from .stream_storage import StreamStorage


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


class FileCountBatchTests(unittest.TestCase):
    def test_sync_tree_splits_at_32_including_scope_in_every_batch(self):
        with tempfile.TemporaryDirectory(prefix="fr5-control-batch-") as tmp:
            root = Path(tmp).resolve() / "episode"
            root.mkdir()
            (root / "scope.json").write_bytes(b'{"scope":"synthetic"}')
            (root / "summary.json").write_bytes(b'{"outcome":"pending_review"}')
            for index in range(63):
                (root / f"evidence-{index:03d}.json").write_bytes(b"{}")
            outbox = root / "frame-outbox"
            outbox.mkdir()
            spool = SimpleNamespace(session="b" * 32, max_bytes=8 * 1024**2, outbox=outbox)
            storage = StreamStorage(root, sha((root / "scope.json").read_bytes()), spool, required_bytes=1)
            batches = []

            def push(payloads):
                batches.append(dict(payloads))
                return {"session": spool.session, "files": {name: sha(raw) for name, raw in payloads.items()}}

            # Replacing _push keeps this test entirely offline; sync_tree and
            # sync_files themselves still select and construct real batches.
            with patch.object(storage, "_push", side_effect=push), \
                    patch("fr5_jev.stream_storage.subprocess.run",
                          side_effect=AssertionError("Unexpected process")), \
                    patch("fr5_jev.stream_storage.subprocess.Popen",
                          side_effect=AssertionError("Unexpected process")):
                manifest = storage.sync_tree()

            self.assertEqual(stream_storage.MAX_BATCH_FILES, 32)
            data_batches, manifest_batch = batches[:-1], batches[-1]
            self.assertEqual([len(batch) for batch in data_batches], [32, 32, 3])
            self.assertEqual(set(manifest_batch), {"scope.json", "control-manifest.json"})
            self.assertTrue(all("scope.json" in batch for batch in batches))
            self.assertTrue(all(len(batch) <= stream_storage.MAX_BATCH_FILES for batch in batches))
            uploaded = [name for batch in data_batches for name in batch if name != "scope.json"]
            expected = {"summary.json", *(f"evidence-{index:03d}.json" for index in range(63))}
            self.assertEqual(len(uploaded), len(expected))
            self.assertEqual(set(uploaded), expected)
            self.assertEqual(set(manifest["files"]), expected | {"scope.json"})


class ClosedRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="fr5-closed-recovery-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.base = self.tmp / "spool"
        self.root = self.base / "episode-closed"
        self.root.mkdir(parents=True)
        (self.root / "frame-outbox").mkdir()
        (self.root / "SESSION_STARTED").write_bytes(b"")
        (self.root / "scope.json").write_bytes(b'{"contract":"synthetic"}\n')
        self.scope_sha = sha((self.root / "scope.json").read_bytes())
        self.session = "c" * 32
        self.summary = {"scope_sha256": self.scope_sha, "outcome": "pending_review"}
        self.outbox_session = {
            "schema": "fr5-local-outbox-v1", "name": self.root.name, "session": self.session,
            "remote_host": "4090", "remote_parent": str(stream_protocol.REMOTE_BASE),
        }
        self.closed_receipt = {
            "schema": "fr5-stream-closed-v1", "session": self.session,
            "accepted_frames": 3, "last_durable_sequence": 2,
            "unacknowledged_frames": 0, "unacknowledged_bytes": 0, "error": None,
            "peak_unacknowledged_frames": 1, "peak_unacknowledged_bytes": 100,
            "max_ack_latency_s": 0.1, "max_local_enqueue_s": 0.01,
            "codec_sha256": sha(stream_storage.CODEC.read_bytes()),
            "protocol_sha256": sha(Path(stream_protocol.__file__).read_bytes()),
            "local_episode": str(self.root),
        }
        self.reset_documents()
        (self.root / "evidence.json").write_bytes(b'{"evidence":"original"}\n')

        # A failure here is never swallowed by the expected validation errors:
        # these tests must neither start a frame stream nor make an SSH call.
        for target in ("fr5_jev.stream_storage.AsyncFrameSpool.start",
                       "fr5_jev.stream_storage.subprocess.Popen",
                       "fr5_jev.stream_storage.subprocess.run"):
            guard = patch(target, side_effect=AssertionError("Unexpected process or frame stream"))
            guard.start()
            self.addCleanup(guard.stop)
        base_patch = patch("fr5_jev.stream_storage.LOCAL_STREAM_BASE", self.base)
        base_patch.start()
        self.addCleanup(base_patch.stop)

    def reset_documents(self):
        write_json(self.root / "summary.json", self.summary)
        write_json(self.root / "stream-closed.json", self.closed_receipt)
        write_json(self.root / "frame-outbox" / "stream-session.json", self.outbox_session)

    def from_closed(self, root=None):
        return StreamStorage.from_closed(root or self.root)

    def assert_rejected(self, root=None):
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            self.from_closed(root)

    def fake_uploads(self, storage, *, fail_first_manifest=False):
        calls = []
        failures = [int(fail_first_manifest)]

        def push(payloads):
            calls.append(dict(payloads))
            if "control-manifest.json" in payloads and failures[0]:
                failures[0] -= 1
                raise RuntimeError("synthetic manifest upload failure")
            return {"session": self.session, "files": {name: sha(raw) for name, raw in payloads.items()}}

        upload_patch = patch.object(storage, "_push", side_effect=push)
        upload_patch.start()
        self.addCleanup(upload_patch.stop)
        return calls

    def test_valid_closed_session_does_not_start_a_frame_stream(self):
        storage = self.from_closed()
        self.assertEqual(storage.root, self.root)
        self.assertEqual(storage.scope_sha, self.scope_sha)
        self.assertEqual(storage.frame_spool.session, self.session)
        self.assertEqual(storage.frame_spool.status()["accepted_frames"], 3)

    def test_legacy_readonly_diagnostic_can_reopen_for_verification_without_summary(self):
        write_json(self.root/'scope.json', {'diagnostic_only': True, 'training_ready': False,
                                            'physical_commands_sent': 0})
        (self.root/'summary.json').unlink()
        receipt = {**self.closed_receipt, 'protocol_sha256': 'a'*64}
        write_json(self.root/'stream-closed.json', receipt)
        storage = self.from_closed()
        self.assertTrue(storage._legacy_readonly_recovery)
        self.assertEqual(storage.protocol_sha256, 'a'*64)
        self.assertEqual(storage.frame_spool.session, self.session)

    def test_earliest_diagnostic_uses_closed_session_when_outbox_metadata_is_absent(self):
        write_json(self.root/'scope.json', {'diagnostic_only': True, 'training_ready': False,
                                            'physical_commands_sent': 0})
        (self.root/'summary.json').unlink()
        session_path = self.root/'frame-outbox/stream-session.json'
        session_path.unlink()
        storage = self.from_closed()
        self.assertTrue(storage._legacy_readonly_recovery)
        self.assertEqual(storage.frame_spool.session, self.closed_receipt['session'])
        session_path.symlink_to(self.tmp/'missing')
        self.assert_rejected()

    def test_missing_summary_is_only_allowed_for_explicit_readonly_diagnostics(self):
        (self.root/'summary.json').unlink()
        self.assert_rejected()
        for scope in ({'diagnostic_only': True, 'training_ready': True, 'physical_commands_sent': 0},
                      {'diagnostic_only': True, 'training_ready': False, 'physical_commands_sent': 1}):
            write_json(self.root/'scope.json', scope)
            self.assert_rejected()

    def test_legacy_readonly_diagnostic_rejects_dangling_control_links(self):
        write_json(self.root/'scope.json', {'diagnostic_only': True, 'training_ready': False,
                                            'physical_commands_sent': 0})
        (self.root/'summary.json').unlink()
        for name in ('summary.json', 'control-manifest.json'):
            with self.subTest(name=name):
                path = self.root/name
                path.symlink_to(self.tmp/'missing')
                try:
                    self.assert_rejected()
                finally:
                    path.unlink()

    def test_rejects_root_outside_local_spool_or_symlink(self):
        outside = self.tmp / "outside"
        shutil.copytree(self.root, outside)
        receipt = {**self.closed_receipt, "local_episode": str(outside)}
        outbox = {**self.outbox_session, "name": outside.name}
        write_json(outside / "stream-closed.json", receipt)
        write_json(outside / "frame-outbox" / "stream-session.json", outbox)
        self.assert_rejected(outside)
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.assert_rejected(alias)

    def test_rejects_linked_scope_receipt_or_outbox_metadata(self):
        for index, relative in enumerate(("scope.json", "stream-closed.json",
                                          "frame-outbox/stream-session.json")):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                target = self.tmp / f"linked-metadata-{index}.json"
                target.write_bytes(original)
                path.unlink()
                path.symlink_to(target)
                try:
                    self.assert_rejected()
                finally:
                    path.unlink()
                    path.write_bytes(original)

    def test_rejects_inconsistent_scope_or_session_identity(self):
        scope_path = self.root / "scope.json"
        original_scope = scope_path.read_bytes()
        scope_path.write_bytes(b'{"contract":"changed"}\n')
        try:
            self.assert_rejected()
        finally:
            scope_path.write_bytes(original_scope)
        cases = (
            ("closed session", "stream-closed.json", {"session": "d" * 32}),
            ("closed episode", "stream-closed.json", {"local_episode": str(self.base / "other")}),
            ("outbox session", "frame-outbox/stream-session.json", {"session": "d" * 32}),
            ("outbox name", "frame-outbox/stream-session.json", {"name": "other"}),
            ("summary scope", "summary.json", {"scope_sha256": "0" * 64}),
        )
        for name, relative, change in cases:
            with self.subTest(name=name):
                self.reset_documents()
                path = self.root / relative
                value = json.loads(path.read_text())
                value.update(change)
                write_json(path, value)
                self.assert_rejected()

    def test_rejects_unclosed_incomplete_or_failed_stream(self):
        cases = (
            {"accepted_frames": 0, "last_durable_sequence": -1},
            {"accepted_frames": -1, "last_durable_sequence": -2},
            {"accepted_frames": True, "last_durable_sequence": 0},
            {"unacknowledged_frames": 1},
            {"unacknowledged_bytes": 1},
            {"last_durable_sequence": 1},
            {"error": "receiver failed"},
            {"schema": "not-a-closed-stream"},
        )
        for change in cases:
            with self.subTest(change=change):
                self.reset_documents()
                receipt = {**self.closed_receipt, **change}
                write_json(self.root / "stream-closed.json", receipt)
                self.assert_rejected()

    def test_finalize_without_manifest_calls_sync_tree(self):
        storage = self.from_closed()
        with patch.object(storage, "sync_tree", return_value={"synthetic": True}) as sync_tree:
            storage.finalize_control()
        sync_tree.assert_called_once_with()

    def test_manifest_failure_retries_only_the_same_manifest(self):
        storage = self.from_closed()
        calls = self.fake_uploads(storage, fail_first_manifest=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic manifest upload failure"):
            storage.finalize_control()

        manifest_path = self.root / "control-manifest.json"
        self.assertTrue(manifest_path.is_file())
        original = manifest_path.read_bytes()
        self.assertGreaterEqual(len(calls), 2)
        first_manifest = calls[-1]
        self.assertEqual(set(first_manifest), {"scope.json", "control-manifest.json"})
        self.assertEqual(first_manifest["control-manifest.json"], original)

        count_before_retry = len(calls)
        storage.finalize_control()
        self.assertEqual(len(calls), count_before_retry + 1)
        self.assertEqual(calls[-1], first_manifest)
        self.assertEqual(manifest_path.read_bytes(), original)

    def test_existing_manifest_rejects_missing_changed_or_new_control_files(self):
        for change in ("missing", "changed", "added"):
            with self.subTest(change=change):
                storage = self.from_closed()
                calls = self.fake_uploads(storage)
                storage.sync_tree()
                calls.clear()
                evidence = self.root / "evidence.json"
                original = evidence.read_bytes()
                try:
                    if change == "missing":
                        evidence.unlink()
                    elif change == "changed":
                        # Same size also verifies that the content hash is checked.
                        evidence.write_bytes(original.replace(b"original", b"modified"))
                    else:
                        (self.root / "unexpected.json").write_bytes(b"{}")
                    with self.assertRaises((ValueError, RuntimeError, OSError)):
                        storage.finalize_control()
                    self.assertEqual(calls, [])
                finally:
                    evidence.write_bytes(original)
                    (self.root / "unexpected.json").unlink(missing_ok=True)
                    (self.root / "control-manifest.json").unlink(missing_ok=True)

    def test_existing_manifest_rejects_identity_changes_before_upload(self):
        for field, value in (("session", "d" * 32), ("scope_sha256", "0" * 64),
                             ("summary_sha256", "0" * 64),
                             ("local_episode", str(self.base / "other"))):
            with self.subTest(field=field):
                storage = self.from_closed()
                calls = self.fake_uploads(storage)
                storage.sync_tree()
                calls.clear()
                path = self.root / "control-manifest.json"
                manifest = json.loads(path.read_text())
                manifest[field] = value
                write_json(path, manifest)
                try:
                    with self.assertRaises((ValueError, RuntimeError, OSError)):
                        storage.finalize_control()
                    self.assertEqual(calls, [])
                finally:
                    path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
