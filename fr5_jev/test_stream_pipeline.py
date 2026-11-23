"""Offline integration of the real recorder, codec, stream and reconstruction."""

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .async_spool import AsyncFrameSpool
from .materialize_stream import materialize
from .stream_storage import CODEC, StreamStorage
from . import stream_protocol
from .test_async_spool import RUNNER
from fr5_lerobot.tests.test_demo_writer import synthetic_command, synthetic_sample
from lerobot_robot_fairino.continuous_capture import ContinuousCapture
from lerobot_robot_fairino.demo_writer import DemoWriter


class FakeCameraRobot:
    is_connected = True

    def __init__(self):
        self.config = SimpleNamespace(camera=SimpleNamespace(expected_serial="SYNTHETIC_ONLY",
                                                               max_age_s=1., timeout_s=.1),
                                      max_sync_skew_s=.1)
        self.index = 0

    def get_sample(self):
        sample = synthetic_sample(self.index)
        captured = time.time()
        sample["camera"]["metadata"]["source_timestamp_unix"] = captured
        sample["robot_state"]["timestamp"] = captured + .004321
        sample["sync_skew_s"] = sample["robot_state"]["timestamp"] - captured
        self.index += 1
        return sample


class StreamPipelineTests(unittest.TestCase):
    def test_capture_finalization_uses_configured_frame_buffer_deadline(self):
        self.spool.max_unacked_age = 30.
        with patch.object(self.spool, 'wait_for_ack', wraps=self.spool.wait_for_ack) as wait:
            self.build()
        self.assertEqual(wait.call_args.kwargs['timeout'], 35.)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-stream-pipeline-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.local = self.base / "episode"
        self.local.mkdir()
        (self.local / "scope.json").write_text(json.dumps({"diagnostic_only": True}))
        self.server = self.base / "server"
        self.server.mkdir()
        command = lambda name, session: [
            __import__("sys").executable, "-u", "-c", RUNNER, str(self.server), name, session, ""]
        with patch("fr5_jev.async_spool.ssh_receiver_command", side_effect=command):
            self.spool = AsyncFrameSpool(self.local, "synthetic-only", max_unacked_age=10).start()
        self.addCleanup(self.spool.abort)

    def build(self):
        robot = FakeCameraRobot()
        issued = robot.get_sample()
        command = synthetic_command(issued)
        writer = DemoWriter(self.local / "lerobot", fps=6, task="SYNTHETIC ONLY", journal_only=True,
                            external_capture_root=self.local / "continuous", sync_writes=True,
                            frame_spool=self.spool,
                            metadata={"image_width": 16, "image_height": 12}).start()
        capture = ContinuousCapture(robot, self.local / "continuous", fps=6, task="SYNTHETIC ONLY",
            sync_writes=True, frame_spool=self.spool, sample_callback=lambda sample: writer.add(sample, command)).start()
        deadline = time.monotonic() + 3
        while len(capture._times) < 4 and time.monotonic() < deadline:
            capture.assert_healthy()
            time.sleep(.01)
        self.assertGreaterEqual(len(capture._times), 4)
        report = capture.finish("observed")
        self.assertTrue(report["capture_complete"])
        writer.finish("interrupted")
        status = self.spool.close(timeout=4)
        remote = self.server / "synthetic-only"
        shutil.copyfile(self.local / "scope.json", remote / "scope.json")
        shutil.copytree(self.local / "continuous", remote / "continuous")
        shutil.copytree(self.local / "lerobot", remote / "lerobot")
        wire = remote / "wire"
        wire.mkdir()
        shutil.copyfile(CODEC, wire / "frame_codec.py")
        shutil.copyfile(stream_protocol.__file__, wire / "stream_protocol.py")
        status.update(schema="fr5-stream-closed-v1", local_episode=str(self.local),
                      codec_sha256=hashlib.sha256(CODEC.read_bytes()).hexdigest(),
                      protocol_sha256=hashlib.sha256(Path(stream_protocol.__file__).read_bytes()).hexdigest())
        (remote / "stream-closed.json").write_text(json.dumps(status))
        return remote, report

    def test_record_stream_decode_and_preserve_actual_journals(self):
        remote, report = self.build()
        result = materialize(remote)
        self.assertEqual(result["frames"], report["frames"])
        output = remote / "materialized"
        rows = [json.loads(line) for line in (output / "continuous/capture.jsonl").read_text().splitlines()]
        teacher = [json.loads(line) for line in (output / "lerobot/meta/capture.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(teacher))
        self.assertTrue(all(row["media_codec"]["rgb_lossy"] for row in rows))
        self.assertTrue(all(row["media_codec"]["depth_lossless"] for row in rows))
        self.assertTrue(all(not row["command"]["executed"] for row in teacher))
        for index, row in enumerate(rows):
            with np.load(output / "continuous" / row["depth_file"], allow_pickle=False) as data:
                np.testing.assert_array_equal(data["depth_mm"], np.full((12, 16), 501 + index, np.float32))
            self.assertTrue((output / "continuous" / row["rgb_file"]).is_file())
        self.assertTrue((remote / "INCOMPLETE").exists())
        with self.assertRaises(FileExistsError):
            materialize(remote)

    def test_changed_journal_is_rejected_without_completed_episode(self):
        remote, _ = self.build()
        journal = remote / "lerobot/meta/capture.jsonl"
        data = (remote / "frames.stream").read_bytes()
        header = stream_protocol.HEADER.unpack_from(data)
        size = stream_protocol.HEADER.size + sum(header[1:4])
        record, _, _ = stream_protocol.unpack_record(data[:size])
        modified = record["teacher"]
        modified["command"]["target"][0] = 999
        journal.write_text(json.dumps(modified) + "\n")
        with self.assertRaisesRegex(ValueError, "journal disagree"):
            materialize(remote)
        self.assertFalse((remote / "materialized").exists())
        self.assertTrue((remote / "frames.stream").is_file())

    def test_old_archived_module_verifies_existing_diagnostic_without_writing(self):
        (self.local / 'scope.json').write_text(json.dumps({
            'diagnostic_only': True, 'training_ready': False, 'physical_commands_sent': 0}))
        remote, _ = self.build()
        expected = materialize(remote)
        # Simulate the earliest frozen modules: their reconstruction code has
        # no control snapshot or newer stream-specific helper definitions.
        saved = remote / 'wire/materialize_stream.py'
        saved.write_text('''from contextlib import ExitStack\nimport hashlib, importlib.util, json\nfrom pathlib import Path\nimport numpy as np\ndef _load(name, path):\n spec=importlib.util.spec_from_file_location(name,path)\n module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n return module\ndef _sha(data): return hashlib.sha256(data).hexdigest()\ndef materialize(root): raise AssertionError("must verify, not reconstruct")\n''')
        storage = StreamStorage(self.local, hashlib.sha256((self.local/'scope.json').read_bytes()).hexdigest(),
                                self.spool, required_bytes=1)
        storage.remote_root = remote
        actual_run = subprocess.run
        def local_runner(command, **kwargs):
            executable = shlex.split(command[-1])
            executable[0] = sys.executable
            return actual_run(executable, **kwargs)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in remote.rglob('*') if p.is_file()}
        with patch('fr5_jev.stream_storage.subprocess.run', side_effect=local_runner):
            self.assertEqual(storage.materialize(), expected)
            after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in remote.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            image = remote/'materialized/continuous/frames/000000.jpg'
            image.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError, 'materialization did not complete'):
                storage.materialize()

    def test_slow_remote_fsync_does_not_hold_capture_or_sample_callback(self):
        local = self.base / "blocked-client"
        local.mkdir()
        gate = str(self.base / "slow-fsync")
        command = lambda name, session: [
            __import__("sys").executable, "-u", "-c", RUNNER, str(self.server), name, session, gate]
        with patch("fr5_jev.async_spool.ssh_receiver_command", side_effect=command):
            spool = AsyncFrameSpool(local, "blocked-stream", max_unacked_age=10).start()
        self.addCleanup(spool.abort)
        robot = FakeCameraRobot()
        command = synthetic_command(robot.get_sample())
        writer = DemoWriter(local / "lerobot", fps=6, task="SYNTHETIC ONLY", journal_only=True,
                            external_capture_root=local / "continuous", frame_spool=spool,
                            metadata={"image_width": 16, "image_height": 12}).start()
        capture = ContinuousCapture(robot, local / "continuous", fps=6, task="SYNTHETIC ONLY",
                   frame_spool=spool, sample_callback=lambda sample: writer.add(sample, command)).start()
        try:
            deadline = time.monotonic() + 3
            while len(capture._times) < 4 and time.monotonic() < deadline:
                capture.assert_healthy()
                time.sleep(.01)
            self.assertGreaterEqual(len(capture._times), 4)
            self.assertEqual(spool.status()["last_durable_sequence"], -1)
            self.assertGreaterEqual(spool.status()["unacknowledged_frames"], 4)
        finally:
            Path(gate + ".release").touch()
            report = capture.finish("observed")
            writer.finish("interrupted")
            spool.close(timeout=4)
        self.assertTrue(report["capture_complete"])


if __name__ == "__main__":
    unittest.main()
