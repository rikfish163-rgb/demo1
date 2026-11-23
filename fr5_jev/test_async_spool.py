"""Offline stream tests: subprocess pipes and temporary files only."""

import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from . import stream_protocol as protocol
from .async_spool import AsyncFrameSpool, SpoolError, inspect_outbox


RUNNER = r'''
import os,sys,time
from pathlib import Path
from fr5_jev.stream_protocol import receive
base,name,session,gate=sys.argv[1:]
def sync(fd):
    if gate and os.readlink('/proc/self/fd/'+str(fd)).endswith('/frames.stream'):
        Path(gate+'.entered').touch()
        while not Path(gate+'.release').exists():time.sleep(.005)
    os.fsync(fd)
ok=receive(sys.stdin.fileno(),sys.stdout.buffer,Path(base),name,session,
           batch_size=1,batch_seconds=.02,sync=sync)
sys.exit(0 if ok else 1)
'''


def media(rgb, depth):
    return {"schema": "fr5-rgbd-wire-media-v1", "rgb_codec": "jpeg", "jpeg_quality": 90,
            "rgb_lossy": True, "rgb_source_sha256": hashlib.sha256(rgb).hexdigest(),
            "depth_codec": "float32le-zlib", "depth_dtype": "float32", "depth_unit": "mm",
            "depth_lossless": True, "depth_source_sha256": hashlib.sha256(depth).hexdigest(),
            "rgb_bytes": len(rgb), "depth_bytes": len(depth),
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            "depth_sha256": hashlib.sha256(depth).hexdigest()}


class AsyncSpoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-async-test-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.server = self.base / "server"
        self.server.mkdir()
        self.client = self.base / "client"
        self.client.mkdir()
        self.gate = str(self.base / "gate")

    def command(self, name, session, gate=None):
        return [sys.executable, "-u", "-c", RUNNER, str(self.server), name, session, gate or ""]

    def wait(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("Timed out waiting for isolated stream test")

    def start(self, *, gate=None, **kwargs):
        with patch("fr5_jev.async_spool.ssh_receiver_command",
                   side_effect=lambda name, session: self.command(name, session, gate)):
            spool = AsyncFrameSpool(self.client, "new-session", **kwargs).start()
        self.addCleanup(spool.abort)
        return spool

    def enqueue(self, spool, index):
        rgb, depth = f"jpeg-{index}".encode(), f"depth-{index}".encode()
        return spool.enqueue(capture={"frame_index": index, "source_time": index / 6},
                             teacher={"actual_target": [index], "executed": False},
                             media=media(rgb, depth), rgb=rgb, depth=depth)

    def test_ack_follows_server_fsync_and_unconfirmed_outbox_survives(self):
        spool = self.start(gate=self.gate, max_frames=2, max_unacked_age=10)
        self.assertEqual(self.enqueue(spool, 0), 0)
        self.wait(lambda: Path(self.gate + ".entered").exists())
        self.assertEqual(self.enqueue(spool, 1), 1)
        self.assertEqual(spool.status()["last_durable_sequence"], -1)
        self.assertEqual(len(list(spool.outbox.glob("*.packet"))), 2)
        with self.assertRaisesRegex(SpoolError, "backlog"):
            self.enqueue(spool, 2)
        self.assertEqual(spool.status()["unacknowledged_frames"], 2)
        Path(self.gate + ".release").touch()
        spool.wait_for_ack(1, timeout=3)
        self.wait(lambda: not list(spool.outbox.glob("*.packet")))
        status = spool.close(timeout=3)
        self.assertEqual(status["last_durable_sequence"], 1)
        self.assertTrue((self.server / "new-session" / "INCOMPLETE").is_file())
        packets = (self.server / "new-session" / "frames.stream").read_bytes()
        first = protocol.HEADER.unpack_from(packets)
        size = protocol.HEADER.size + sum(first[1:4])
        first_meta, first_rgb, first_depth = protocol.unpack_record(packets[:size])
        second_meta, _, _ = protocol.unpack_record(packets[size:])
        self.assertEqual(first_meta["capture"]["frame_index"], 0)
        self.assertEqual(first_meta["teacher"]["actual_target"], [0])
        self.assertEqual((first_rgb, first_depth), (b"jpeg-0", b"depth-0"))
        self.assertEqual(second_meta["sequence"], 1)

    def test_transport_failure_retains_unacknowledged_packet(self):
        spool = self.start(gate=self.gate, max_unacked_age=10)
        self.enqueue(spool, 0)
        self.wait(lambda: Path(self.gate + ".entered").exists())
        spool.abort()
        self.wait(lambda: spool._process.poll() is not None)
        self.assertEqual(len(list(spool.outbox.glob("*.packet"))), 1)
        audit = inspect_outbox(self.client)
        self.assertEqual(audit["pending_packets"][0]["sequence"], 0)
        self.assertTrue(audit["remote_reconciliation_required"])
        self.assertEqual(audit["commands_replayed"], 0)
        with self.assertRaises(SpoolError):
            spool.assert_healthy()
        with self.assertRaises(FileExistsError):
            AsyncFrameSpool(self.client, "new-session").start()

    def test_age_limit_is_independent_of_available_capacity(self):
        spool = self.start(gate=self.gate, max_frames=12, max_unacked_age=.15)
        self.enqueue(spool, 0)
        self.wait(lambda: Path(self.gate + ".entered").exists())
        time.sleep(.2)
        with self.assertRaisesRegex(SpoolError, "acknowledged"):
            spool.assert_healthy()
        self.assertEqual(len(list(spool.outbox.glob("*.packet"))), 1)

    def test_whole_task_buffer_survives_a_short_server_pause_without_dropping_frames(self):
        spool = self.start(gate=self.gate, max_frames=120, max_bytes=32*1024**2, max_unacked_age=30.)
        for index in range(30):
            self.enqueue(spool,index)
        self.wait(lambda:Path(self.gate+'.entered').exists())
        with spool._condition:
            for pending in spool._pending.values():
                pending['queued'] = time.monotonic()-6.
        spool.assert_healthy()
        self.assertEqual(len(list(spool.outbox.glob('*.packet'))),30)
        Path(self.gate+'.release').touch()
        spool.wait_for_ack(29,timeout=3)
        receipt=spool.close(timeout=3)
        self.assertEqual(receipt['accepted_frames'],30)
        self.assertEqual(receipt['last_durable_sequence'],29)
        self.assertEqual(receipt['unacknowledged_frames'],0)


    def test_default_capacity_retains_frames_until_bounded_and_durably_acked(self):
        spool = self.start(gate=self.gate)
        self.assertEqual(spool.max_frames, 24)
        self.assertEqual(spool.max_bytes, 8 * 1024 * 1024)
        self.assertEqual(spool.max_unacked_age, 4.)
        for index in range(23):
            self.assertEqual(self.enqueue(spool, index), index)
        self.wait(lambda: Path(self.gate + ".entered").exists())
        spool.assert_healthy()
        self.assertEqual(self.enqueue(spool, 23), 23)
        with self.assertRaisesRegex(SpoolError, "backlog"):
            spool.assert_healthy()
        with self.assertRaisesRegex(SpoolError, "backlog"):
            self.enqueue(spool, 24)
        self.assertEqual(spool.status()["unacknowledged_frames"], 24)
        self.assertEqual(len(list(spool.outbox.glob("*.packet"))), 24)
        self.assertEqual(spool.status()["last_durable_sequence"], -1)
        Path(self.gate + ".release").touch()
        spool.wait_for_ack(23, timeout=3)
        self.assertEqual(spool.close(timeout=3)["last_durable_sequence"], 23)

    def test_byte_limit_rejects_frame_without_discarding_existing_packet(self):
        spool = self.start(gate=self.gate, max_bytes=2048)
        self.enqueue(spool, 0)
        self.wait(lambda: Path(self.gate + ".entered").exists())
        rgb = b"x" * 2048
        depth = b"depth"
        with self.assertRaisesRegex(SpoolError, "backlog"):
            spool.enqueue(capture={"frame_index": 1}, teacher={"executed": False},
                          media=media(rgb, depth), rgb=rgb, depth=depth)
        self.assertEqual(spool.status()["unacknowledged_frames"], 1)
        self.assertEqual(len(list(spool.outbox.glob("*.packet"))), 1)
        Path(self.gate + ".release").touch()
        spool.wait_for_ack(0, timeout=3)
        self.assertEqual(spool.close(timeout=3)["last_durable_sequence"], 0)

    def test_truncated_frame_is_never_acknowledged(self):
        session = "b" * 32
        proc = subprocess.Popen(self.command("partial", session), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.assertEqual(json.loads(proc.stdout.readline())["type"], "ready")
        rgb, depth = b"rgb", b"depth"
        meta = {"schema": "fr5-stream-frame-v1", "session": session, "sequence": 0,
                "capture": {}, "teacher": {}, "media": media(rgb, depth)}
        packet = protocol.encode_record(meta, rgb, depth)
        proc.stdin.write(packet[:-1])
        proc.stdin.close()
        response = json.loads(proc.stdout.readline())
        self.assertEqual(response["type"], "end")
        self.assertFalse(response["clean_eof"])
        self.assertEqual(response["frames"], 0)
        self.assertNotEqual(proc.wait(timeout=3), 0)
        proc.stdout.close()
        proc.stderr.close()
        self.assertEqual((self.server / "partial" / "frames.stream").read_bytes(), b"")

    def test_corrupt_packet_and_duplicate_sequence_fail_without_false_ack(self):
        session = "c" * 32
        proc = subprocess.Popen(self.command("invalid", session), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.assertEqual(json.loads(proc.stdout.readline())["type"], "ready")
        rgb, depth = b"rgb", b"depth"
        meta = {"schema": "fr5-stream-frame-v1", "session": session, "sequence": 0,
                "capture": {}, "teacher": {}, "media": media(rgb, depth)}
        packet = protocol.encode_record(meta, rgb, depth)
        proc.stdin.write(packet + packet)
        first, end = json.loads(proc.stdout.readline()), json.loads(proc.stdout.readline())
        self.assertEqual((first["type"], first["sequence"]), ("ack", 0))
        self.assertEqual(end["type"], "end")
        self.assertIn("Invalid sequence", end["error"])
        self.assertNotEqual(proc.wait(timeout=3), 0)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()
        altered = bytearray(packet)
        altered[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            protocol.unpack_record(bytes(altered))


if __name__ == "__main__":
    unittest.main()
