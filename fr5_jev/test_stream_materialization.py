"""Synthetic final archive checks for evidence, timestamps and actual wire targets."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .materialize_stream import _control_snapshot, _stream_controls, _validate_stream_targets, materialize, verify_materialized
from .stream_storage import CODEC
from . import stream_protocol
from lerobot_robot_fairino.frame_codec import encode_media


def digest(data):
    return hashlib.sha256(data).hexdigest()


class OrderedTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fr5-ordered-targets-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'submitted-targets.jsonl'
        actions = [[float(i), 0., 0., 0., 0., 0., .76] for i in range(50)]
        self.chunks = {
            'first': {'request': {'source_time': 100.}, 'received_monotonic': 100.3,
                      'reply': {'actions': actions}},
            'second': {'request': {'source_time': 109.}, 'received_monotonic': 109.3,
                       'reply': {'actions': actions}},
        }

    def row(self, sequence, request_id, index, submitted):
        chunk = self.chunks[request_id]
        raw = chunk['reply']['actions'][index]
        return {'sequence': sequence, 'request_id': request_id, 'action_index': index,
                'raw_action': raw, 'target': raw[:6] + [.76], 'submitted_monotonic': submitted,
                'target_monotonic': submitted + 1/6,
                'predicted_source_target_monotonic': chunk['request']['source_time']+(index+1)/6}

    def validate(self, rows, *, ordered=True, prefix_steps=None):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return _validate_stream_targets(self.path, self.chunks, ordered, prefix_steps)

    def test_complete_chunk_then_partial_next_preserves_real_timing_offset(self):
        rows = [self.row(i, 'first', i, 100.5+i/6) for i in range(50)]
        rows += [self.row(50+i, 'second', i, 109.5+i/6) for i in range(2)]
        timing = self.validate(rows)
        self.assertEqual(len(timing), 52)
        self.assertAlmostEqual(timing[0]['target_time_offset_s'], .5)
        self.assertAlmostEqual(timing[50]['target_time_offset_s'], .5)
        self.assertEqual((timing[49]['action_index'], timing[50]['action_index']), (49, 0))

    def test_rejects_skipped_prefix_duplicate_and_early_chunk_switch(self):
        cases = [
            [self.row(0, 'first', 1, 100.5)],
            [self.row(0, 'first', 0, 100.5), self.row(1, 'first', 0, 100.7)],
            [self.row(0, 'first', 0, 100.5), self.row(1, 'second', 0, 109.5)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, 'Ordered action sequence'):
                    self.validate(rows)

    def test_rejects_forged_source_timing_and_noncausal_submission(self):
        row = self.row(0, 'first', 0, 100.5)
        row['predicted_source_target_monotonic'] += 1.
        with self.assertRaisesRegex(ValueError, 'timing is inconsistent'):
            self.validate([row])
        row = self.row(0, 'first', 0, 100.2)  # before prediction arrived
        with self.assertRaisesRegex(ValueError, 'timing is inconsistent'):
            self.validate([row])

    def test_legacy_recording_keeps_original_source_clock_rule(self):
        row = self.row(0, 'first', 3, 100.5)
        row.pop('predicted_source_target_monotonic')
        row['target_monotonic'] = 100.+4/6
        self.assertEqual(self.validate([row], ordered=False), [])
        row['target_monotonic'] += .2
        with self.assertRaisesRegex(ValueError, 'predicted source time'):
            self.validate([row], ordered=False)

    def test_effective_gripper_field_preserves_submitted_target_verification(self):
        row = self.row(0, 'first', 0, 100.5)
        row['effective_gripper_target_percent'] = 75
        self.assertEqual(len(self.validate([row])), 1)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved['target'][6], .76)
        self.assertEqual(saved['effective_gripper_target_percent'], 75)

    def test_policy_prefix_then_partial_next_preserves_both_clocks(self):
        rows = [self.row(i, 'first', i, 100.5+i/6) for i in range(3)]
        rows += [self.row(3+i, 'second', i, 109.5+i/6) for i in range(2)]
        for row in rows:
            row.update(execution_prefix_steps=3, prediction_chunk_steps=50)
        timing = self.validate(rows, prefix_steps=3)
        self.assertEqual([row['action_index'] for row in timing], [0, 1, 2, 0, 1])
        self.assertAlmostEqual(timing[3]['target_time_offset_s'], .5)

    def test_policy_prefix_rejects_skips_early_switch_and_forged_dimensions(self):
        def row(sequence, request_id, index, submitted):
            value = self.row(sequence, request_id, index, submitted)
            value.update(execution_prefix_steps=3, prediction_chunk_steps=50)
            return value

        bad_prefix = row(0, 'first', 0, 100.5)
        bad_prefix['execution_prefix_steps'] = 4
        bad_prediction = row(0, 'first', 0, 100.5)
        bad_prediction['prediction_chunk_steps'] = 49
        cases = [
            ([row(0, 'first', 1, 100.5)], 'Ordered action sequence'),
            ([row(0, 'first', 0, 100.5), row(1, 'first', 2, 100.7)], 'Ordered action sequence'),
            ([row(0, 'first', 0, 100.5), row(1, 'second', 0, 109.5)], 'Ordered action sequence'),
            ([row(0, 'first', 3, 100.5)], 'configured policy prefix'),
            ([bad_prefix], 'configured policy prefix'),
            ([bad_prediction], 'configured policy prefix'),
        ]
        for rows, message in cases:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                self.validate(rows, prefix_steps=3)

    def test_policy_prefix_scope_and_chunk_metadata_must_match(self):
        root = self.path.parent
        scope = {'dataset_role': 'policy_evaluation', 'policy_contract': {'name': 'fr5-joint-position-stream-v1'},
                 'task_instruction': 'test', 'checkpoint_sha256': 'a'*64,
                 'execution_mode': 'ordered_policy_prefix', 'execution_chunk_steps': 3,
                 'prediction_chunk_steps': 50, 'prefix_skip': False}
        summary = {'schema': 'fr5-stream-policy-evaluation-summary-v1', 'dataset_role': 'policy_evaluation',
                   'task_instruction': 'test', 'checkpoint_sha256': 'a'*64, 'task_success_verified': False,
                   'training_ready': False, 'awaiting_whole_user_review': True, 'read_only_diagnostic': False,
                   'model_execution': True, 'storage_status': 'server_stream_closed',
                   'summary_stage': 'final_control_snapshot', 'termination_source': 'stop', 'reason': 'stop',
                   'failed': None, 'error': None, 'recorded_frames': 0, 'continuous_capture': {'frames': 0},
                   'wire_records': 0, 'prediction_records': 0, 'wire_method_counts': {'ServoJ': 0, 'MoveGripper': 0},
                   'commands': 0, 'wire_callbacks_drained': False}
        receipt = {'accepted_frames': 0}
        with patch('fr5_jev.materialize_stream._stream_index', return_value=[]):
            self.assertEqual(_stream_controls(root, receipt, scope, summary, {})['ordered_timing'], [])
            for key, value in [('execution_chunk_steps', 0), ('execution_chunk_steps', 50),
                               ('execution_chunk_steps', True), ('prediction_chunk_steps', 49),
                               ('prefix_skip', True)]:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'prediction prefix'):
                    _stream_controls(root, receipt, dict(scope, **{key: value}), summary, {})
            chunk = {'request': {'id': 'first'}, 'reply': {}, 'execution_prefix_steps': 4}
            (root/'chunk-00000.json').write_text(json.dumps(chunk))
            with patch('fr5_jev.materialize_stream._validate_prediction'):
                with self.assertRaisesRegex(ValueError, 'Prediction chunk disagrees'):
                    _stream_controls(root, receipt, scope, summary, {'chunk-00000.json': {}})


class MaterializationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-final-archive-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "server"
        self.root.mkdir()
        self.original = self.base / "original"
        self.session = "e" * 32
        wire = self.root / "wire"
        wire.mkdir()
        shutil.copyfile(CODEC, wire / "frame_codec.py")
        shutil.copyfile(stream_protocol.__file__, wire / "stream_protocol.py")
        self.put("INCOMPLETE", "{}")
        self.put("view-complete/color.jpg", b"SYNTHETIC")
        self.put("view-complete/observation.json", '{"synthetic":true}')
        self.put("view-complete/scene.npz", b"SYNTHETIC CACHE")
        self.put("workflow-metrics.json", '{"outcome":"pending_user_whole_episode_review"}')
        self.put("action-outcomes.json", "[]")
        self.put("completed-stages.json", '["move","release"]')
        self.put("decision-move.json", "{}")
        evidence = str(self.original / "view-complete/color.jpg")
        scope = {"phase_order": ["move", "release"], "split_role": "training_complete_trial",
                 "evidence": [evidence]}
        self.put("scope.json", json.dumps(scope))
        self.scope_sha = digest((self.root / "scope.json").read_bytes())
        self.hold = {"sequence": 1, "issued_unix": 100., "kind": "hold", "executed": False,
                     "target": [0, 0, 0, 0, 0, 0, 80]}
        self.arm = {"sequence": 2, "issued_unix": 101., "kind": "arm", "executed": True,
                    "target": [1, 2, 3, 4, 5, 6, 80]}
        self.gripper = {"sequence": 3, "issued_unix": 102., "kind": "gripper", "executed": True,
                        "target": [1, 2, 3, 4, 5, 6, 40]}
        self.put("target-command-journal.json", json.dumps([self.hold, self.arm, self.gripper]))
        for name, method, args, command in (
                ("command-0002.json", "MoveL", [[1, 2, 3, 4, 5, 6], [0] * 6], self.arm),
                ("command-0003.json", "MoveGripper", [1, 40], self.gripper)):
            self.put(name, json.dumps({"method": method, "args": args, "command": command,
                     "intended_target": command["target"], "status": "completed", "outcome": {"return_code": 0},
                     "completed_at_unix": command["issued_unix"] + .05}))
        self.put("continuous/report.json", '{"capture_complete":true,"frames":3}')
        self.put("continuous/events.jsonl", "")
        self.put("lerobot/meta/demo.json", '{"recorded_frames":3,"storage_complete":true}')
        summary = {"outcome": "pending_review", "scope_sha256": self.scope_sha,
                   "completed_stages": ["move", "release"], "split_role": "training_complete_trial",
                   "start_offset_mm": 100., "released": True, "failed": None,
                   "continuous_capture": {"capture_complete": True, "frames": 3},
                   "dataset": {"recorded_frames": 3, "storage_complete": True}, "evidence": [evidence]}
        self.put("summary.json", json.dumps(summary))
        self.receipt = {"schema": "fr5-stream-closed-v1", "session": self.session,
                        "accepted_frames": 3, "last_durable_sequence": 2, "unacknowledged_frames": 0,
                        "error": None, "local_episode": str(self.original),
                        "codec_sha256": digest(CODEC.read_bytes()),
                        "protocol_sha256": digest(Path(stream_protocol.__file__).read_bytes())}
        self.put("stream-closed.json", json.dumps(self.receipt))
        self.write_stream([self.hold, self.arm, self.gripper])
        self.write_manifest()

    def put(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
        return path

    def write_stream(self, commands):
        records = []
        for index, command in enumerate(commands):
            camera = {"source_timestamp_unix": 100.1 + index, "source_camera_serial": "SYNTHETIC"}
            robot = {"timestamp": 100.1 + index, "joints": [0] * 6, "gripper_position": 80}
            media, rgb, depth = encode_media(np.full((12, 16, 3), 20 + index, np.uint8),
                                             np.full((12, 16), 500 + index, np.float32))
            capture = {"frame_index": index, "camera": camera, "robot": robot, "sync_clock": "host_unix",
                       "sync_skew_s": 0., "rgb_file": f"frames/{index:06d}.jpg",
                       "depth_file": f"frames/{index:06d}.npz", "media_codec": media}
            teacher = {"frame_index": index, "camera": camera, "robot": robot, "sync_clock": "host_unix",
                       "sync_skew_s": 0., "command": command}
            records.append(stream_protocol.encode_record({"schema": "fr5-stream-frame-v1",
                "session": self.session, "sequence": index, "capture": capture,
                "teacher": teacher, "media": media}, rgb, depth))
        self.put("frames.stream", b"".join(records))

    def write_manifest(self):
        entries = {}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            name = path.relative_to(self.root).as_posix()
            if name.startswith("wire/") or name in {"frames.stream", "INCOMPLETE", "control-manifest.json"}:
                continue
            data = path.read_bytes()
            entries[name] = {"bytes": len(data), "sha256": digest(data)}
        manifest = {"schema": "fr5-final-control-manifest-v1", "session": self.session,
                    "local_episode": str(self.original), "scope_sha256": self.scope_sha,
                    "summary_sha256": entries["summary.json"]["sha256"], "files": entries}
        self.put("control-manifest.json", json.dumps(manifest))

    def test_complete_varied_summary_evidence_and_wire_targets_are_verified(self):
        receipt = materialize(self.root)
        self.assertEqual(receipt["frames"], 3)
        self.assertEqual(receipt["control_verification"], "verified")
        self.assertEqual(receipt["target_alignment"], "verified_against_wire_and_frame_clocks")
        output = self.root / "materialized"
        for name in ("scope.json", "summary.json", "view-complete/color.jpg", "view-complete/observation.json",
                     "view-complete/scene.npz", "workflow-metrics.json", "action-outcomes.json"):
            self.assertEqual((output / name).read_bytes(), (self.root / name).read_bytes())
        self.assertEqual(receipt["summary_sha256"], digest((output / "summary.json").read_bytes()))

    def test_lost_reply_can_verify_existing_materialization_without_rewriting(self):
        receipt = materialize(self.root)
        image = self.root/'materialized/continuous/frames/000000.jpg'
        before = image.stat().st_mtime_ns
        self.assertEqual(verify_materialized(self.root), receipt)
        self.assertEqual(image.stat().st_mtime_ns, before)

    def test_retry_refuses_changed_materialized_image(self):
        materialize(self.root)
        image = self.root/'materialized/continuous/frames/000000.jpg'
        image.write_bytes(b'corrupted synthetic image')
        with self.assertRaisesRegex(ValueError, 'image changed'):
            verify_materialized(self.root)

    def test_forged_wire_target_is_rejected_even_if_file_hashes_match(self):
        path = self.root / "command-0002.json"
        command = json.loads(path.read_text())
        command["args"][0][0] = 999
        path.write_text(json.dumps(command))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Joint target"):
            materialize(self.root)
        self.assertFalse((self.root / "materialized").exists())

    def test_frame_label_must_be_the_actual_latched_command(self):
        self.write_stream([self.hold, self.gripper, self.gripper])
        with self.assertRaisesRegex(ValueError, "latched command"):
            materialize(self.root)
        self.assertFalse((self.root / "materialized").exists())

    def test_missing_review_evidence_is_not_published(self):
        summary_path = self.root / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["evidence"] = [str(self.original / "view-missing/color.jpg")]
        summary_path.write_text(json.dumps(summary))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Evidence is missing"):
            materialize(self.root)
        self.assertFalse((self.root / "materialized").exists())

    def test_monotonic_journal_keeps_original_unix_clock_steps(self):
        journal = [self.hold, self.arm, self.gripper]
        for index, command in enumerate(journal):
            command["issued_unix"] = 100. - index
            command["issued_monotonic"] = 1000. + index
            if index:
                path = self.root / f"command-{command['sequence']:04d}.json"
                wire = json.loads(path.read_text())
                wire["command"] = command
                wire["completed_monotonic"] = command["issued_monotonic"] + .1
                path.write_text(json.dumps(wire))
        self.put("target-command-journal.json", json.dumps(journal))
        self.write_manifest()
        _, _, _, recovered = _control_snapshot(self.root, self.receipt)
        self.assertEqual([item["issued_unix"] for item in recovered], [100., 99., 98.])


if __name__ == "__main__":
    unittest.main()
