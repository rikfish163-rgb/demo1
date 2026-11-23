"""Control-transfer deadlines; isolated subprocess mocks, no network or devices."""

import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .stream_storage import (CONTROL_TIMEOUT_BASE_S, CONTROL_TIMEOUT_MAX_S,
                             CONTROL_TIMEOUT_RATE_BPS, MAX_BATCH, StreamStorage, _control_timeout)


class ControlTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-control-timeout-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "synthetic-session"
        self.root.mkdir()
        outbox = self.root / "frame-outbox"
        outbox.mkdir()
        self.storage = StreamStorage(self.root, "a" * 64,
                                     SimpleNamespace(session="b" * 32, outbox=outbox), required_bytes=1)

    def test_control_upload_uses_the_data4_virtual_environment(self):
        raw = b'{}'
        receipt = {'session': 'b' * 32, 'files': {
            'scope.json': hashlib.sha256(raw).hexdigest()}}
        with patch('fr5_jev.stream_storage.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(receipt).encode())) as run:
            self.storage._push({'scope.json': raw})
        command = shlex.split(run.call_args.args[0][-1])
        self.assertEqual(command[:2],
                         ['/home/data4/hts/demo1/fr5-learning/.venv/bin/python', '-c'])

    def test_deadline_scales_with_actual_encoded_payload_and_is_bounded(self):
        observed = []

        def send(_cmd, *, input, capture_output, timeout, check):
            size = struct.unpack("!I", input[:4])[0]
            request = json.loads(input[4:4 + size])
            observed.append((len(input), len(request["files"]), timeout))
            return SimpleNamespace(stdout=json.dumps({"session": request["session"], "files": {
                row["path"]: row["sha256"] for row in request["files"]}}).encode())

        with patch("fr5_jev.stream_storage.subprocess.run", side_effect=send):
            self.storage._push({"command.json": b"{}"})
            self.storage._push({"view/scene.npz": b"x" * (4 * 1024**2)})
        for size, count, deadline in observed:
            self.assertEqual(deadline, _control_timeout(size, file_count=count))
        self.assertGreaterEqual(observed[0][2], CONTROL_TIMEOUT_BASE_S + 1)
        self.assertLess(observed[0][2], CONTROL_TIMEOUT_BASE_S + 2)
        self.assertGreater(observed[1][2], 15 + 4)
        self.assertGreaterEqual(_control_timeout(MAX_BATCH), 15 + 32)
        self.assertLessEqual(_control_timeout(MAX_BATCH), CONTROL_TIMEOUT_MAX_S)
        self.assertEqual(_control_timeout(10**12), CONTROL_TIMEOUT_MAX_S)
        self.assertEqual(_control_timeout(CONTROL_TIMEOUT_RATE_BPS), CONTROL_TIMEOUT_BASE_S + 1)

    def test_deadline_accounts_for_each_control_file(self):
        # The same number of bytes may require many more server fsyncs when
        # split into small files. The default keeps the byte-only calculation.
        payload_bytes = 512 * 1024
        self.assertEqual(_control_timeout(payload_bytes), CONTROL_TIMEOUT_BASE_S + 1)
        self.assertEqual(_control_timeout(payload_bytes, file_count=1), CONTROL_TIMEOUT_BASE_S + 2)
        self.assertEqual(_control_timeout(payload_bytes, file_count=32), CONTROL_TIMEOUT_BASE_S + 33)
        self.assertLessEqual(_control_timeout(payload_bytes, file_count=32), CONTROL_TIMEOUT_MAX_S)
        self.assertEqual(_control_timeout(10**12, file_count=32), CONTROL_TIMEOUT_MAX_S)

    def test_timeout_remains_failure_and_never_produces_a_receipt(self):
        with patch("fr5_jev.stream_storage.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("synthetic", 20)):
            with self.assertRaisesRegex(RuntimeError, "did not durably acknowledge") as caught:
                self.storage._push({"scope.json": b"SYNTHETIC"})
        self.assertIsInstance(caught.exception.__cause__, subprocess.TimeoutExpired)
        diagnostic_path = self.root / "frame-outbox" / "last-control-error.json"
        diagnostic_text = diagnostic_path.read_text()
        diagnostic = json.loads(diagnostic_text)
        self.assertEqual(diagnostic["stage"], "control_upload")
        self.assertEqual(diagnostic["exception_type"], "TimeoutExpired")
        self.assertEqual(diagnostic["file_count"], 1)
        self.assertEqual(diagnostic["timeout_seconds"],
                         _control_timeout(diagnostic["payload_bytes"], file_count=1))
        self.assertNotIn("SYNTHETIC", diagnostic_text)

    def test_excessive_file_count_is_rejected_before_launch(self):
        payloads = {f"file-{index:02d}.json": b"{}" for index in range(33)}
        with patch("fr5_jev.stream_storage.subprocess.run") as run:
            with self.assertRaises(ValueError):
                self.storage._push(payloads)
        run.assert_not_called()

    def test_ssh_connection_error_is_available_in_record_and_exception(self):
        error = subprocess.CalledProcessError(255, 'ssh', stderr=b'Connection reset by peer')
        with patch('fr5_jev.stream_storage.subprocess.run', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'Connection reset by peer'):
                self.storage._push({'scope.json': b'SYNTHETIC'})
        text = (self.root/'frame-outbox/last-control-error.json').read_text()
        self.assertEqual(json.loads(text)['stderr'], 'Connection reset by peer')
        self.assertNotIn('SYNTHETIC', text)

    def test_mismatched_receipt_remains_failure(self):
        fake = SimpleNamespace(stdout=json.dumps({"session": "b" * 32, "files": {"x": "0" * 64}}).encode())
        with patch("fr5_jev.stream_storage.subprocess.run", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "receipt does not match"):
                self.storage._push({"x": b"source"})


if __name__ == "__main__":
    unittest.main()
