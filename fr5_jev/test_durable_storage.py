"""Synthetic storage checks only; no robot, camera, SSH or network calls."""

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from . import durable_storage as storage


class DurableStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-durable-test-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "recording-spool"
        self.base.mkdir()
        self.root = self.base / "episode-1"
        self.root.mkdir()
        self.scope = {"evidence": ["SYNTHETIC"]}
        self.scope_file = self.root / "scope.json"
        self.scope_file.write_text(json.dumps(self.scope))
        self.scope_sha = hashlib.sha256(self.scope_file.read_bytes()).hexdigest()
        self.base_patch = patch.object(storage, "LOCAL_BASE", self.base)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.mount_patch = patch.object(storage, "_mounted", return_value=True)
        self.mount_patch.start()
        self.addCleanup(self.mount_patch.stop)

    def test_preflight_requires_verified_server_and_cleans_only_synthetic_probe(self):
        with patch.object(storage, "_server_check", return_value={"free_bytes": 1000000}) as remote, \
                patch.object(storage, "PROBE_LIMIT_S", 10):
            result = storage.DurableStorage.preflight(self.root, self.scope_sha, required_bytes=100)
        self.assertEqual(result.root, self.root)
        self.assertEqual(len(result.probe_frame_seconds), 3)
        self.assertEqual(sorted(p.name for p in self.base.iterdir()), ["episode-1"])
        self.assertEqual(self.scope_file.read_text(), json.dumps(self.scope))
        receipt = json.loads((self.root / "storage-preflight.json").read_text())
        self.assertEqual(receipt["scope_sha256"], self.scope_sha)
        self.assertIn("storage-preflight.json", remote.call_args.args[1])

    def test_slow_remote_recording_fails_closed_and_preserves_scope(self):
        with patch.object(storage, "_server_check", return_value={"free_bytes": 1000000}), \
                patch.object(storage, "_probe_frame", return_value=1.0):
            with self.assertRaisesRegex(RuntimeError, "too slow"):
                storage.DurableStorage.preflight(self.root, self.scope_sha, required_bytes=100)
        self.assertEqual(sorted(p.name for p in self.base.iterdir()), ["episode-1"])
        self.assertEqual(hashlib.sha256(self.scope_file.read_bytes()).hexdigest(), self.scope_sha)

    def test_server_failure_and_existing_recording_are_rejected(self):
        with patch.object(storage, "_server_check", side_effect=RuntimeError("unavailable")), \
                patch.object(storage, "PROBE_LIMIT_S", 10):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                storage.DurableStorage.preflight(self.root, self.scope_sha, required_bytes=100)
        (self.root / "SESSION_STARTED").touch()
        with self.assertRaisesRegex(RuntimeError, "already claimed"):
            storage.DurableStorage.preflight(self.root, self.scope_sha, required_bytes=100)

    def test_outside_recording_and_volatile_geometry_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "direct"):
            storage.DurableStorage.preflight(self.base.parent, self.scope_sha, required_bytes=100)
        geometry = self.base.parent / "geometry.json"
        geometry.write_text("{}")
        self.scope["geometry_path"] = str(geometry)
        self.scope_file.write_text(json.dumps(self.scope))
        digest = hashlib.sha256(self.scope_file.read_bytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "geometry_path"):
            storage.DurableStorage.preflight(self.root, digest, required_bytes=100)

    def test_mount_absence_and_changed_scope_reject_before_probe(self):
        with patch.object(storage, "_mounted", return_value=False), \
                patch.object(storage, "_bench") as benchmark:
            with self.assertRaisesRegex(RuntimeError, "mount"):
                storage.DurableStorage.preflight(self.root, self.scope_sha, required_bytes=100)
            benchmark.assert_not_called()
        self.scope_file.write_text("{}")
        with patch.object(storage, "_bench") as benchmark:
            with self.assertRaisesRegex(RuntimeError, "Scope changed"):
                storage.DurableStorage.preflight(self.root, self.scope_sha, required_bytes=100)
            benchmark.assert_not_called()

    def test_command_sync_binds_exact_scope_and_command_bytes(self):
        command = self.root / "command-0001.json"
        command.write_text('{"status":"attempted_once"}')
        s = storage.DurableStorage(self.root, self.scope_sha)
        with patch.object(storage, "_server_check", return_value={}) as remote:
            s.sync_files(command)
        args, _ = remote.call_args
        self.assertEqual(args[1], {"scope.json": self.scope_sha,
                                   command.name: hashlib.sha256(command.read_bytes()).hexdigest()})
        with self.assertRaisesRegex(ValueError, "outside"):
            s.sync_files(self.base / "other.json")

    def test_unresponsive_storage_preflight_has_a_deadline(self):
        release = threading.Event()
        try:
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                storage._bounded(lambda: release.wait(2), timeout=.02)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
