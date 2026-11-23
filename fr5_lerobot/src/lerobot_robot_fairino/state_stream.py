"""Receive-only FR5 TCP 20004 telemetry, with bounded host-time history.

No RPC, enable, configuration or movement commands are issued. Timestamps are
host receive times, not robot hardware timestamps or hardware synchronization.
Supported layouts are the copied SDK v2.1.4 frame and the locally verified
v3.8.4.2 frame with two additional trailing bytes. Unknown layouts are rejected.
"""

from __future__ import annotations

import copy
import ctypes
import math
import socket
import struct
import threading
import time
from collections import deque

from .state_structs import RobotStatePkg


class StateStreamError(RuntimeError):
    """Disconnected stream or invalid telemetry; cached data is unavailable."""


_STRUCT_SIZE = ctypes.sizeof(RobotStatePkg)
_FRAME_SIZES = frozenset((_STRUCT_SIZE, _STRUCT_SIZE + 2))
_SCALARS = (
    "program_state", "robot_state", "main_code", "sub_code", "robot_mode",
    "tool", "user", "EmergencyStop", "motion_done", "mc_queue_len",
    "collisionState", "safety_stop0_state", "safety_stop1_state",
    "gripper_fault", "gripper_position", "rbtEnableState",
    "gripper_motiondone", "gripper_active", "gripper_speed",
    "gripper_current", "gripper_fault_id",
)


def _decode_packet(packet: bytes, timestamp: float, received_monotonic: float) -> dict:
    """Decode one already framed packet; recheck all framing invariants."""
    if len(packet) not in _FRAME_SIZES or packet[:2] != b"\x5a\x5a":
        raise StateStreamError("Unsupported FR5 state frame layout")
    if struct.unpack_from("<H", packet, 3)[0] + 7 != len(packet):
        raise StateStreamError("FR5 state length mismatch")
    if sum(packet[:-2]) % 65536 != int.from_bytes(packet[-2:], "little"):
        raise StateStreamError("FR5 state checksum mismatch")
    pkg = RobotStatePkg.from_buffer_copy(packet)
    state = {name: getattr(pkg, name) for name in _SCALARS}
    for key, field in (
        ("joints", "jt_cur_pos"), ("tcp", "tl_cur_pos"),
        ("flange", "flange_cur_pos"), ("tcp_speed", "actual_TCP_CmpSpeed"),
        ("joint_speeds_deg_s", "actual_qd"),
    ):
        values = list(getattr(pkg, field))
        if not all(math.isfinite(value) for value in values):
            raise StateStreamError(f"Non-finite FR5 state field: {key}")
        state[key] = values
    state.update(
        timestamp=timestamp,
        received_monotonic=received_monotonic,
        frame_counter=pkg.frame_cnt & 0xFF,
        timestamp_source="host_receive_unix",
        packet_size=len(packet),
    )
    return state


class _FrameParser:
    """Resynchronize after garbage, split reads, bad lengths or bad checksums."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        self.buffer.extend(chunk)
        frames = []
        while True:
            index = self.buffer.find(b"\x5a\x5a")
            if index < 0:
                # Retain one byte in case the next recv completes the header.
                self.buffer[:] = self.buffer[-1:] if self.buffer.endswith(b"\x5a") else b""
                break
            if index:
                del self.buffer[:index]
            if len(self.buffer) < 5:
                break
            size = struct.unpack_from("<H", self.buffer, 3)[0] + 7
            if size not in _FRAME_SIZES:
                del self.buffer[0]
                continue
            if len(self.buffer) < size:
                break
            packet = bytes(self.buffer[:size])
            if sum(packet[:-2]) % 65536 != int.from_bytes(packet[-2:], "little"):
                del self.buffer[0]
                continue
            frames.append(packet)
            del self.buffer[:size]
        return frames


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


class FR5StateStream:
    """A single receive-only connection; reconnection must be explicit.

    connect() waits for the first valid frame. latest() is nonblocking and raises
    TimeoutError if its freshness bound is not met. nearest() waits up to its
    timeout for frames reaching the requested host timestamp in clock_basis, then returns
    the closest retained frame only if within max_skew_s. Historical matches
    still require a live, healthy stream. Returned dictionaries are copies.

    joints and orientation angles are degrees, Cartesian positions are mm,
    joint_speeds_deg_s are degrees/s. Safety/fault values are reported without
    interpreting them as authorization to move the robot.
    """

    def __init__(self, host="192.168.58.2", port=20004, timeout_s=2, history_seconds=5):
        self.host = str(host)
        self.port = int(port)
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.timeout_s = _positive_finite(timeout_s, "timeout_s")
        self.history_seconds = _positive_finite(history_seconds, "history_seconds")
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._history = deque(maxlen=max(2, int(self.history_seconds * 1000)))
        self._socket = None
        self._thread = None
        self._stop = threading.Event()
        self._error = None
        self._last_valid_monotonic = None

    def connect(self):
        with self._lifecycle_lock:
            with self._condition:
                if self._socket is not None:
                    self._check_live_locked()
                    return self
                self._history.clear()
                self._error = None
                self._last_valid_monotonic = None
            conn = None
            try:
                conn = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                conn.settimeout(min(0.1, self.timeout_s))
            except OSError as exc:
                if conn is not None:
                    conn.close()
                with self._condition:
                    self._error = f"State connection failed: {exc}"
                raise StateStreamError(self._error) from exc
            stop = threading.Event()
            with self._condition:
                self._socket = conn
                self._stop = stop
                self._thread = threading.Thread(
                    target=self._receive, args=(conn, stop), name="fr5-state-receiver", daemon=True
                )
                self._thread.start()
                deadline = time.monotonic() + self.timeout_s
                while not self._history and self._error is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._history and self._error is None:
                    return self
                error = self._error or "No valid FR5 state frame before connection timeout"
            self._disconnect_locked()
            with self._condition:
                self._error = error
            raise StateStreamError(error)

    def disconnect(self):
        with self._lifecycle_lock:
            self._disconnect_locked()

    def _disconnect_locked(self):
        with self._condition:
            self._stop.set()
            conn, worker = self._socket, self._thread
            self._socket = None
            self._history.clear()
            self._last_valid_monotonic = None
            self._condition.notify_all()
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=min(self.timeout_s, 1.0) + 0.2)

    def _receive(self, conn, stop):
        parser = _FrameParser()
        last_valid = time.monotonic()
        failure = None
        try:
            while not stop.is_set():
                if time.monotonic() - last_valid > self.timeout_s:
                    raise StateStreamError("Valid FR5 state frames timed out")
                try:
                    chunk = conn.recv(8192)
                except socket.timeout:
                    continue
                received_monotonic, timestamp = time.monotonic(), time.time()
                if not chunk:
                    raise StateStreamError("FR5 state stream disconnected (EOF)")
                states = [
                    _decode_packet(frame, timestamp, received_monotonic)
                    for frame in parser.feed(chunk)
                ]
                if states:
                    last_valid = received_monotonic
                    with self._condition:
                        if stop.is_set() or self._socket is not conn:
                            break
                        self._history.extend(states)
                        self._last_valid_monotonic = last_valid
                        self._prune_locked(last_valid)
                        self._condition.notify_all()
        except Exception as exc:
            failure = f"FR5 state receiver failed: {exc}"
        finally:
            conn.close()
            with self._condition:
                if self._socket is conn:
                    self._socket = None
                    self._history.clear()
                    self._last_valid_monotonic = None
                    self._error = failure or "FR5 state receiver stopped"
                    self._condition.notify_all()

    def _check_live_locked(self):
        if self._error is not None:
            raise StateStreamError(self._error)
        if self._socket is None:
            raise StateStreamError("FR5 state stream is not connected")
        if self._last_valid_monotonic is None:
            raise StateStreamError("FR5 state stream has no valid frames")
        if time.monotonic() - self._last_valid_monotonic > self.timeout_s:
            raise StateStreamError("FR5 state stream is stale")

    def _prune_locked(self, now):
        cutoff = now - self.history_seconds
        while self._history and self._history[0]["received_monotonic"] < cutoff:
            self._history.popleft()

    def latest(self, max_age_s=0.5) -> dict:
        max_age_s = _positive_finite(max_age_s, "max_age_s", allow_zero=True)
        with self._condition:
            self._check_live_locked()
            now = time.monotonic()
            self._prune_locked(now)
            if not self._history or now - self._history[-1]["received_monotonic"] > max_age_s:
                raise TimeoutError("No sufficiently recent FR5 state frame")
            return copy.deepcopy(self._history[-1])

    def nearest(self, timestamp_unix, max_skew_s=0.15, timeout_s=0.5, *, clock_basis='host_unix') -> dict:
        if clock_basis not in ('host_unix','host_monotonic'):
            raise ValueError('Unknown state clock domain')
        key = 'received_monotonic' if clock_basis == 'host_monotonic' else 'timestamp'
        timestamp_unix = float(timestamp_unix)
        if not math.isfinite(timestamp_unix):
            raise ValueError("timestamp_unix must be finite")
        max_skew_s = _positive_finite(max_skew_s, "max_skew_s", allow_zero=True)
        timeout_s = _positive_finite(timeout_s, "timeout_s", allow_zero=True)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                self._check_live_locked()
                now = time.monotonic()
                self._prune_locked(now)
                closest = min(
                    self._history, key=lambda item: abs(item[key] - timestamp_unix), default=None
                )
                reached_target = any(item[key] >= timestamp_unix for item in self._history)
                if reached_target or now >= deadline:
                    if closest is not None and abs(closest[key] - timestamp_unix) <= max_skew_s:
                        return copy.deepcopy(closest)
                    raise TimeoutError("No FR5 state frame within requested timestamp skew")
                self._condition.wait(min(deadline - now, 0.1))
