"""Readiness gate tests with wholly synthetic evidence and mocked server checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from . import stream_readiness as ready
from .stream_storage import StreamStorage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-ready-test-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.codec = self.root / "codec.py"
        self.protocol = self.root / "protocol.py"
        self.codec.write_text("synthetic codec")
        self.protocol.write_text("synthetic protocol")
        self.log = self.state / "tests.log"
        self.log.write_text("Ran 52 tests in 1s\n\nOK\n")
        self.diag = self.root / "diagnostic.json"
        self.diag.write_text(json.dumps({"status": "measurement_complete", "physical_commands_sent": 0,
            "server_root": str(ready.REMOTE_BASE / ("diagnostic-stream-readonly-" + "a" * 32)),
            "requested_fps": 6, "requested_seconds": 30, "measured_camera_fps": 6.01,
            "max_frame_gap_s": .2, "capture_frames": 181, "errors": [],
            "stream_durability": {"accepted_frames": 181, "last_durable_sequence": 180,
                "unacknowledged_frames": 0, "peak_unacknowledged_frames": 3, "error": None},
            "remote_integrity": {"frames": 181, "codec_sha256": sha(self.codec), "protocol_sha256": sha(self.protocol)}}))
        synthetic_local = self.root / ("synthetic-final-control-" + "b" * 32)
        synthetic_local.mkdir()
        (synthetic_local / "scope.json").write_text("synthetic scope")
        (synthetic_local / "summary.json").write_text("synthetic summary")
        self.proof = self.state / "stream-readiness.json"
        self.code = ("fr5_lerobot/src/lerobot_robot_fairino/frame_codec.py", "fr5_jev/stream_protocol.py")
        for name, source in zip(self.code, (self.codec, self.protocol)):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        self.value = {"schema": "fr5-stream-readiness-v1", "created_unix": time.time(),
            "code_sha256": {name: sha(self.root / name) for name in self.code},
            "tests": {"status": "passed", "count": 52, "log": "tests.log", "sha256": sha(self.log)},
            "diagnostic": {"local_result": str(self.diag), "result_sha256": sha(self.diag), "materialization_sha256": "a" * 64},
            "synthetic_finalization": {"server_name": "synthetic-final-control-" + "b" * 32,
                "local_episode": str(synthetic_local),
                "control_verification": "verified", "target_alignment": "verified_against_wire_and_frame_clocks",
                "frames": 4, "materialization_sha256": "b" * 64,
                "scope_sha256": sha(synthetic_local / "scope.json"),
                "summary_sha256": sha(synthetic_local / "summary.json")}}
        self.proof.write_text(json.dumps(self.value))
        for name, value in (("ROOT", self.root), ("PROOF", self.proof), ("CODE_FILES", self.code)):
            p = patch.object(ready, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.server = patch.object(ready, "_server_check", return_value={})
        self.mock_server = self.server.start()
        self.addCleanup(self.server.stop)

    def test_complete_evidence_is_checked_on_both_server_roots(self):
        self.assertEqual(ready.validate_readiness()["tests"]["count"], 52)
        self.assertEqual(self.mock_server.call_count, 2)

    def test_test_log_must_stay_in_proof_directory(self):
        outside = self.root / "tests.log"
        outside.write_bytes(self.log.read_bytes())
        for name in ("../tests.log", str(outside)):
            with self.subTest(log=name):
                self.value["tests"]["log"] = name
                self.proof.write_text(json.dumps(self.value))
                with self.assertRaisesRegex(RuntimeError, "integration test evidence"):
                    ready.validate_readiness()
                self.mock_server.assert_not_called()

    def test_changed_code_or_stale_proof_blocks_before_server_calls(self):
        (self.root / self.code[0]).write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            ready.validate_readiness()
        self.mock_server.assert_not_called()
        (self.root / self.code[0]).write_bytes(self.codec.read_bytes())
        self.value["created_unix"] -= 9 * 3600
        self.proof.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            ready.validate_readiness()

    def test_unconfirmed_server_or_missing_readonly_frames_blocks(self):
        self.mock_server.side_effect = RuntimeError("server unavailable")
        with self.assertRaisesRegex(RuntimeError, "server unavailable"):
            ready.validate_readiness()
        self.mock_server.reset_mock(side_effect=True)
        diagnostic = json.loads(self.diag.read_text())
        diagnostic["stream_durability"]["unacknowledged_frames"] = 1
        self.diag.write_text(json.dumps(diagnostic))
        self.value["diagnostic"]["result_sha256"] = sha(self.diag)
        self.proof.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(RuntimeError, "durable frame"):
            ready.validate_readiness()
        self.mock_server.assert_not_called()

    def test_episode_spool_is_not_started_when_gate_fails(self):
        with patch("fr5_jev.stream_readiness.validate_readiness", side_effect=RuntimeError("not verified")), \
                patch.object(StreamStorage, "start_readonly") as start:
            with self.assertRaisesRegex(RuntimeError, "not verified"):
                StreamStorage.start_for_episode(self.root, "scope", required_bytes=1)
        start.assert_not_called()

    def test_nominal_rate_tolerance_and_new_capacity_are_bounded(self):
        diagnostic = json.loads(self.diag.read_text())
        diagnostic['measured_camera_fps'] = 5.99
        diagnostic['stream_durability']['peak_unacknowledged_frames'] = 23
        def update():
            self.diag.write_text(json.dumps(diagnostic))
            self.value['diagnostic']['result_sha256'] = sha(self.diag)
            self.proof.write_text(json.dumps(self.value))
        update()
        ready.validate_readiness()
        diagnostic['measured_camera_fps'] = 5.9
        update()
        with self.assertRaisesRegex(RuntimeError, 'durable frame'):
            ready.validate_readiness()
        diagnostic['measured_camera_fps'] = 6
        diagnostic['stream_durability']['peak_unacknowledged_frames'] = 24
        update()
        with self.assertRaisesRegex(RuntimeError, 'durable frame'):
            ready.validate_readiness()


if __name__ == "__main__":
    unittest.main()
