"""Memory-only packet/transport tests. No camera or robot connections."""

import ctypes
import queue
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

from lerobot_robot_fairino.state_stream import (
    FR5StateStream, StateStreamError, _FrameParser, _decode_packet,
)
from lerobot_robot_fairino.state_structs import RobotStatePkg


def packet(counter=0, tail_bytes=2, **fields):
    pkg = RobotStatePkg()
    pkg.frame_head = 0x5A5A
    pkg.frame_cnt = counter if counter < 128 else counter - 256
    pkg.program_state = pkg.robot_state = pkg.robot_mode = 1
    pkg.motion_done = pkg.rbtEnableState = pkg.gripper_active = 1
    pkg.gripper_motiondone = 2
    pkg.gripper_position = 42
    pkg.jt_cur_pos[:] = [1, 2, 3, 4, 5, 6]
    pkg.tl_cur_pos[:] = [100, 200, 300, 10, 20, 30]
    pkg.flange_cur_pos[:] = [90, 190, 290, 9, 19, 29]
    pkg.actual_qd[:] = [0.1] * 6
    for name, value in fields.items():
        setattr(pkg, name, value)
    raw = bytearray(bytes(pkg)[:-2] + b"\x31" * tail_bytes + b"\x00\x00")
    struct.pack_into("<H", raw, 3, len(raw) - 7)
    raw[-2:] = (sum(raw[:-2]) % 65536).to_bytes(2, "little")
    return bytes(raw)


class FakeSocket:
    """No send API is provided, deliberately enforcing receive-only use."""

    def __init__(self, *chunks):
        self.chunks = queue.Queue()
        for chunk in chunks:
            self.chunks.put(chunk)
        self.timeout = 0.1
        self.closed = False

    def settimeout(self, value):
        self.timeout = value

    def recv(self, _size):
        try:
            item = self.chunks.get(timeout=self.timeout)
        except queue.Empty:
            raise socket.timeout()
        if isinstance(item, BaseException):
            raise item
        return item

    def shutdown(self, _how):
        self.chunks.put(b"")

    def close(self):
        self.closed = True


class PacketTests(unittest.TestCase):
    def test_verified_layout_offsets_and_two_frame_versions(self):
        self.assertEqual(ctypes.sizeof(RobotStatePkg), 958)
        self.assertEqual(RobotStatePkg.jt_cur_pos.offset, 16)
        self.assertEqual(RobotStatePkg.rbtEnableState.offset, 771)
        for tail in (0, 2):
            with self.subTest(tail=tail):
                state = _decode_packet(packet(255, tail_bytes=tail), 123.0, 45.0)
                self.assertEqual(state["frame_counter"], 255)
                self.assertEqual(state["joints"], [1, 2, 3, 4, 5, 6])
                self.assertEqual(state["gripper_position"], 42)
                self.assertEqual(state["gripper_active"], 1)
                self.assertEqual(state["rbtEnableState"], 1)
                self.assertEqual(state["timestamp"], 123.0)
                self.assertEqual(state["received_monotonic"], 45.0)
                self.assertEqual(state["joint_speeds_deg_s"], [0.1] * 6)

    def test_split_header_garbage_coalesced_frames_and_counter_wrap(self):
        parser = _FrameParser()
        p1, p2 = packet(255), packet(0)
        self.assertEqual(parser.feed(b"garbage\x5a"), [])
        self.assertEqual(parser.feed(p1[1:17]), [])
        result = parser.feed(p1[17:] + p2)
        self.assertEqual(result, [p1, p2])
        self.assertEqual(len(parser.buffer), 0)

    def test_bad_checksum_bad_length_and_truncated_frame_resynchronize(self):
        parser = _FrameParser()
        bad = bytearray(packet(1))
        bad[-1] ^= 0xFF
        unsupported = packet(2, tail_bytes=4)
        good = packet(3)
        self.assertEqual(parser.feed(bytes(bad) + unsupported + good[:30]), [])
        self.assertEqual(parser.feed(good[30:]), [good])
        self.assertEqual(parser.feed(b"noise" * 10000), [])
        self.assertLessEqual(len(parser.buffer), 1)

    def test_decode_rejects_nonfinite_and_invalid_checksums(self):
        raw = bytearray(packet())
        struct.pack_into("<d", raw, RobotStatePkg.jt_cur_pos.offset, float("nan"))
        raw[-2:] = (sum(raw[:-2]) % 65536).to_bytes(2, "little")
        with self.assertRaisesRegex(StateStreamError, "Non-finite"):
            _decode_packet(bytes(raw), 1, 2)
        raw = bytearray(packet())
        raw[-1] ^= 1
        with self.assertRaisesRegex(StateStreamError, "checksum"):
            _decode_packet(bytes(raw), 1, 2)
        with self.assertRaisesRegex(StateStreamError, "layout"):
            _decode_packet(packet(tail_bytes=4), 1, 2)

    def test_fault_states_are_reported_not_silently_zeroed(self):
        state = _decode_packet(packet(main_code=42, EmergencyStop=1, gripper_fault=15), 1, 2)
        self.assertEqual((state["main_code"], state["EmergencyStop"], state["gripper_fault"]), (42, 1, 15))


class StreamTests(unittest.TestCase):
    def connect_fake(self, fake=None, **kwargs):
        fake = fake or FakeSocket(packet())
        stream = FR5StateStream(**kwargs)
        self.addCleanup(stream.disconnect)
        with patch("lerobot_robot_fairino.state_stream.socket.create_connection", return_value=fake) as create:
            stream.connect()
        create.assert_called_once_with((stream.host, stream.port), timeout=stream.timeout_s)
        return stream, fake

    def wait_failed(self, stream):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                stream.latest()
            except StateStreamError:
                return
            time.sleep(0.005)
        self.fail("Receiver did not fail closed")

    def test_connect_latest_copies_and_disconnect(self):
        stream, fake = self.connect_fake()
        state = stream.latest()
        state["joints"][0] = 999
        self.assertEqual(stream.latest()["joints"][0], 1)
        stream.connect()  # idempotent: no second connection
        stream.disconnect()
        self.assertTrue(fake.closed)
        with self.assertRaises(StateStreamError):
            stream.latest()
        with self.assertRaises(StateStreamError):
            stream.nearest(state["timestamp"])

    def test_socket_eof_and_exception_invalidate_cached_history(self):
        for ending in (b"", OSError("broken transport")):
            with self.subTest(ending=repr(ending)):
                stream, fake = self.connect_fake()
                timestamp = stream.latest()["timestamp"]
                fake.chunks.put(ending)
                self.wait_failed(stream)
                with self.assertRaises(StateStreamError):
                    stream.nearest(timestamp)
                self.assertEqual(len(stream._history), 0)

    def test_freshness_and_timeout_even_if_garbage_keeps_arriving(self):
        stream, fake = self.connect_fake(timeout_s=0.12)
        with self.assertRaises(TimeoutError):
            stream.latest(max_age_s=0)
        for _ in range(50):
            fake.chunks.put(b"invalid frames")
        self.wait_failed(stream)
        with self.assertRaises(StateStreamError):
            stream.latest(max_age_s=100)

    def test_nearest_past_exact_future_and_skew_rejection(self):
        stream, fake = self.connect_fake()
        first = stream.latest()
        self.assertEqual(stream.nearest(first["timestamp"], max_skew_s=0)["frame_counter"], 0)
        self.assertEqual(stream.nearest(first["timestamp"] - 0.05)["frame_counter"], 0)
        with self.assertRaises(TimeoutError):
            stream.nearest(first["timestamp"] - 1, max_skew_s=0.1)
        target = time.time() + 0.025
        producer = threading.Timer(0.04, lambda: fake.chunks.put(packet(1)))
        producer.start()
        self.addCleanup(producer.cancel)
        near = stream.nearest(target, max_skew_s=0.05, timeout_s=0.3)
        self.assertEqual(near["frame_counter"], 1)
        self.assertLessEqual(abs(near["timestamp"] - target), 0.05)
        with self.assertRaises(TimeoutError):
            stream.nearest(time.time() + 10, timeout_s=0.01)

    def test_history_expires_and_does_not_escape_as_stale_nearest(self):
        stream, _ = self.connect_fake(history_seconds=0.01)
        timestamp = stream.latest()["timestamp"]
        time.sleep(0.02)
        with self.assertRaises(TimeoutError):
            stream.nearest(timestamp, timeout_s=0)
        self.assertEqual(len(stream._history), 0)

    def test_connect_no_valid_frame_and_explicit_reconnect(self):
        stream = FR5StateStream(timeout_s=0.03)
        self.addCleanup(stream.disconnect)
        fake = FakeSocket(b"bad header")
        with patch("lerobot_robot_fairino.state_stream.socket.create_connection", return_value=fake):
            with self.assertRaises(StateStreamError):
                stream.connect()
        self.assertTrue(fake.closed)
        with self.assertRaises(StateStreamError):
            stream.latest()
        with patch("lerobot_robot_fairino.state_stream.socket.create_connection", return_value=FakeSocket(packet(6))):
            stream.connect()
        self.assertEqual(stream.latest()["frame_counter"], 6)

    def test_connection_failure_and_invalid_arguments(self):
        stream = FR5StateStream()
        with patch("lerobot_robot_fairino.state_stream.socket.create_connection", side_effect=OSError("no route")):
            with self.assertRaises(StateStreamError):
                stream.connect()
        for kwargs in ({"timeout_s": 0}, {"history_seconds": float("nan")}, {"port": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FR5StateStream(**kwargs)
        for kwargs in ({"max_skew_s": -1}, {"timeout_s": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                stream.nearest(0, **kwargs)

    def test_socket_setup_failure_closes_transport(self):
        stream = FR5StateStream()
        fake = FakeSocket()
        fake.settimeout = lambda _value: (_ for _ in ()).throw(OSError("setup failed"))
        with patch("lerobot_robot_fairino.state_stream.socket.create_connection", return_value=fake):
            with self.assertRaises(StateStreamError):
                stream.connect()
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
