"""Final snapshot, subclass summary and bounded-cache tests; no devices."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from .stream_storage import MAX_BATCH, MAX_LOCAL_CONTROL, StreamStorage
from .test_teacher_varied import VariedTests
from .teacher_varied import STAGES
from .teacher_targets import TeacherSession


class FinalControlSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-final-control-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "episode"
        self.root.mkdir()
        (self.root / "scope.json").write_text('{"scope":"synthetic"}')
        (self.root / "summary.json").write_text('{"outcome":"pending_review"}')
        self.sha = hashlib.sha256((self.root / "scope.json").read_bytes()).hexdigest()
        (self.root / "frame-outbox").mkdir()
        spool = SimpleNamespace(session="d" * 32, max_bytes=8 * 1024**2,
                                outbox=self.root / "frame-outbox")
        self.storage = StreamStorage(self.root, self.sha, spool, required_bytes=1)
        self.received, self.batches = {}, []

        def push(files):
            self.batches.append(sum(len(value) for value in files.values()))
            self.assertLessEqual(self.batches[-1], MAX_BATCH)
            for name, value in files.items():
                self.received[name] = value
            return {"session": spool.session, "files": {
                name: hashlib.sha256(value).hexdigest() for name, value in files.items()}, "free_bytes": 1}

        self.storage._push = push

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_zero_frame_startup_preserves_failure_without_materializing_empty_data(self):
        session = TeacherSession.__new__(TeacherSession)
        session.root, session.scope_sha = self.root, self.sha
        session.journal = [{'sequence':1, 'kind':'hold', 'executed':False}]
        session.command_sequence = 1
        session.peak_height, session.released = 0., False
        session.failed = 'startup camera unavailable'
        session.writer = None
        session.capture = SimpleNamespace(finish=Mock(return_value={
            'capture_complete':False, 'frames':0, 'error':'startup camera unavailable'}))
        spool = SimpleNamespace(close=Mock(return_value={'accepted_frames':0,
                                'last_durable_sequence':-1, 'unacknowledged_frames':0}), abort=Mock())
        session.storage = SimpleNamespace(frame_spool=spool, codec_sha256='a'*64,
            protocol_sha256='b'*64, sync_tree=Mock(), materialize=Mock(),
            release_checkpoint_caches=Mock())
        summary = session.finish('interrupted')
        self.assertEqual(summary['outcome'], 'interrupted')
        self.assertEqual(summary['commands'], 0)
        self.assertEqual(summary['continuous_capture']['frames'], 0)
        self.assertEqual(summary['failed'], 'startup camera unavailable')
        session.storage.sync_tree.assert_called_once()
        session.storage.materialize.assert_not_called()
        spool.abort.assert_not_called()
        receipt = json.loads((self.root/'stream-closed.json').read_text())
        self.assertEqual(receipt['materialization_status'], 'not_attempted_no_frames')

    def test_every_view_and_workflow_file_copied_in_bounded_batches(self):
        self.put("view-close/color.jpg", b"synthetic-image")
        self.put("view-close/observation.json", b'{"state":"synthetic"}')
        self.put("view-close/scene.npz", b"synthetic-depth")
        self.put("workflow-metrics.json", b'{"outcome":"pending_user_whole_episode_review"}')
        self.put("action-outcomes.json", b"[]")
        self.put("completed-stages.json", b"[]")
        self.put("decision-0001.json", b"{}")
        self.put("frame-outbox/000000.packet", b"must-not-upload")
        self.put("continuous/capture.jsonl", b"local-placeholder")
        for index in range(3):
            path = self.root / f"view-{index}" / "scene.npz"
            path.parent.mkdir()
            with path.open("wb") as file:
                file.truncate(12 * 1024**2)
        result = self.storage.sync_tree()
        self.assertGreaterEqual(len(self.batches), 3)  # 36 MiB plus the final manifest
        for name in ("view-close/color.jpg", "view-close/observation.json", "view-close/scene.npz",
                     "workflow-metrics.json", "action-outcomes.json", "completed-stages.json", "decision-0001.json"):
            self.assertIn(name, result["files"])
            self.assertEqual(result["files"][name]["sha256"], hashlib.sha256(self.received[name]).hexdigest())
        self.assertNotIn("frame-outbox/000000.packet", self.received)
        self.assertNotIn("continuous/capture.jsonl", self.received)
        self.assertEqual(result["summary_sha256"], hashlib.sha256((self.root / "summary.json").read_bytes()).hexdigest())
        self.assertTrue((self.root / "view-close/scene.npz").exists())

    def test_local_control_cache_is_bounded_before_another_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "cache limit"):
            self.storage.ensure_local_capacity(reserve_bytes=MAX_LOCAL_CONTROL)
        self.assertTrue((self.root / "scope.json").exists())

    def test_changed_control_during_upload_refuses_manifest(self):
        original = self.storage._push
        changed = False

        def mutate(files):
            nonlocal changed
            result = original(files)
            if not changed and "summary.json" in files:
                (self.root / "summary.json").write_text('{"outcome":"changed"}')
                changed = True
            return result

        self.storage._push = mutate
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.storage.sync_tree()
        self.assertFalse((self.root / "control-manifest.json").exists())

    def test_only_verified_checkpoint_cache_is_released(self):
        scene = self.put("view-close/scene.npz", b"synthetic-depth")
        image = self.put("view-close/color.jpg", b"synthetic-image")
        state = self.put("view-close/observation.json", b"{}")
        manifest = self.storage.sync_tree()
        receipt = {"control_verification": "verified", "target_alignment": "verified_against_wire_and_frame_clocks",
                   "session": self.storage.frame_spool.session, "scope_sha256": manifest["scope_sha256"],
                   "summary_sha256": manifest["summary_sha256"]}
        with self.assertRaisesRegex(RuntimeError, "verified"):
            self.storage.release_checkpoint_caches({**receipt, "control_verification": "incomplete"})
        self.assertTrue(scene.exists())
        released = self.storage.release_checkpoint_caches(receipt)
        self.assertEqual([item["path"] for item in released], ["view-close/scene.npz"])
        self.assertFalse(scene.exists())
        self.assertTrue(image.exists())
        self.assertTrue(state.exists())
        self.assertEqual(self.received["view-close/scene.npz"], b"synthetic-depth")
        self.assertTrue((self.root / "frame-outbox/released-checkpoint-caches.json").is_file())


class VariedSummaryOrderingTests(unittest.TestCase):
    def test_final_subclass_fields_exist_before_snapshot_and_materialization(self):
        fixture = VariedTests(methodName="test_phase_skip_replay_and_success_shortcut_rejected")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        session, root = fixture.s, fixture.root
        session.completed_stages = list(STAGES)
        session.peak_height = 50.
        session.released = True
        session.failed = None
        image = root / "view-complete/color.jpg"
        image.parent.mkdir()
        image.write_bytes(b"synthetic review evidence")
        (image.parent / "observation.json").write_text('{"state":"synthetic"}')
        (root / "workflow-metrics.json").write_text('{"outcome":"pending_user_whole_episode_review"}')
        session.capture = SimpleNamespace(finish=Mock(return_value={"capture_complete": True, "frames": 1}))
        session.writer = SimpleNamespace(_started=True, finish=Mock(return_value={"recorded_frames": 1, "storage_complete": True}))
        spool = SimpleNamespace(close=Mock(return_value={"accepted_frames": 1, "last_durable_sequence": 0}),
                                abort=Mock())
        snapshots = []

        def sync_tree():
            value = json.loads((root / "summary.json").read_text())
            snapshots.append(hashlib.sha256((root / "summary.json").read_bytes()).hexdigest())
            self.assertEqual(value["completed_stages"], STAGES)
            self.assertEqual(value["split_role"], session.scope["split_role"])
            self.assertEqual(value["start_offset_mm"], 100.)
            self.assertTrue((root / "view-complete/observation.json").exists())
            self.assertEqual(json.loads((root / "workflow-metrics.json").read_text())["outcome"],
                             "pending_user_whole_episode_review")

        def materialize():
            self.assertEqual(hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(), snapshots[0])

        session.storage = SimpleNamespace(frame_spool=spool, codec_sha256="codec", protocol_sha256="protocol",
                                          sync_tree=Mock(side_effect=sync_tree), materialize=Mock(side_effect=materialize),
                                          release_checkpoint_caches=Mock())
        result = session.finish("pending_review", [str(image)])
        self.assertEqual(result["completed_stages"], STAGES)
        self.assertEqual(hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(), snapshots[0])
        session.storage.materialize.assert_called_once()
        spool.abort.assert_not_called()


if __name__ == "__main__":
    unittest.main()
