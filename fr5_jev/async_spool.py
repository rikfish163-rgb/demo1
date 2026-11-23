"""Bounded, locally durable, asynchronous frame stream to the 4090 spool.

Import is inert. Only explicit start launches SSH. Unconfirmed packets stay in
the small local outbox across disconnects and reboots; no commands are replayed.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import threading
import time
import uuid

from . import stream_protocol as protocol


class SpoolError(RuntimeError):
    pass


def inspect_outbox(root):
    """Inspect retained packets after a crash; never retransmit or issue commands."""
    outbox = Path(root) / "frame-outbox"
    session = json.loads((outbox / "stream-session.json").read_text())
    if session.get("schema") != "fr5-local-outbox-v1" or not protocol.SESSION.fullmatch(session.get("session", "")):
        raise ValueError("Invalid retained outbox session")
    records, partials = [], []
    for path in sorted(outbox.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Invalid retained outbox entry")
        if path.suffix == ".partial":
            partials.append({"name": path.name, "bytes": path.stat().st_size})
        elif path.suffix == ".packet":
            if path.stat().st_size > protocol.HEADER.size + protocol.MAX_META + 2 * protocol.MAX_MEDIA:
                raise ValueError("Retained frame exceeds protocol limits")
            packet = path.read_bytes()
            meta, _, _ = protocol.unpack_record(packet)
            digest = hashlib.sha256(packet).hexdigest()
            sequence = meta.get("sequence")
            if (meta.get("session") != session["session"] or type(sequence) is not int or sequence < 0
                    or path.name != f"{sequence:09d}-{digest}.packet"):
                raise ValueError("Retained frame identity or checksum mismatch")
            records.append({"sequence": sequence, "sha256": digest, "bytes": len(packet), "name": path.name})
    return {"session": session, "pending_packets": records, "partial_files": partials,
            "remote_reconciliation_required": True, "commands_replayed": 0}


def ssh_receiver_command(name, session):
    """Ship this exact stdlib receiver source; no shell interpolation of input."""
    if not protocol.NAME.fullmatch(name) or not protocol.SESSION.fullmatch(session):
        raise ValueError("Invalid recording name or session")
    source = Path(protocol.__file__).read_text(encoding="utf-8")
    command = "python3 -u -c " + shlex.quote(source) + " " + shlex.quote(name) + " " + shlex.quote(session)
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
            "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=1", "4090", command]


class AsyncFrameSpool:
    # At the requested 6 Hz, 24 frames correspond to the existing four-second
    # unacknowledged-age limit. The independent 8 MiB bound is unchanged.
    def __init__(self, root, name, *, max_frames=24, max_bytes=8 * 1024 * 1024,
                 max_unacked_age=4., process_factory=subprocess.Popen):
        self.root = Path(root)
        if (not protocol.NAME.fullmatch(name) or type(max_frames) is not int or max_frames < 1
                or type(max_bytes) is not int or max_bytes < 1
                or not isinstance(max_unacked_age, (int, float)) or max_unacked_age <= 0):
            raise ValueError("Invalid bounded stream configuration")
        if not self.root.is_dir() or self.root.is_symlink() or self.root.resolve() != self.root:
            raise ValueError("Local session must be an existing non-symlink directory")
        self.name, self.session = name, uuid.uuid4().hex
        self.outbox = self.root / "frame-outbox"
        self.max_frames, self.max_bytes, self.max_unacked_age = max_frames, max_bytes, float(max_unacked_age)
        self._process_factory = process_factory
        self._condition = threading.Condition()
        self._producer_lock = threading.Lock()
        self._pending = OrderedDict()
        self._send_queue = deque()
        self._bytes = 0
        self._peak_frames = self._peak_bytes = 0
        self._max_ack_latency = self._max_local_enqueue = 0.
        self._next_sequence = 0
        self._last_ack = -1
        self._error = None
        self._closing = False
        self._started = False
        self._end_receipt = None
        self._process = None

    def start(self, *, timeout=8.):
        if self._started:
            raise SpoolError("Stream can only start once")
        self.outbox.mkdir(mode=0o700, exist_ok=False)
        protocol._sync_dir(self.root)
        with (self.outbox / "stream-session.json").open("x", encoding="utf-8") as file:
            json.dump({"schema": "fr5-local-outbox-v1", "name": self.name,
                       "session": self.session, "remote_host": "4090",
                       "remote_parent": str(protocol.REMOTE_BASE),
                       "buffer_policy": {'max_frames':self.max_frames,'max_bytes':self.max_bytes,
                                         'max_unacked_age_seconds':self.max_unacked_age}}, file)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        protocol._sync_dir(self.outbox)
        self._started = True
        try:
            self._process = self._process_factory(ssh_receiver_command(self.name, self.session),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            ready, _, _ = select.select([self._process.stdout], [], [], timeout)
            if not ready:
                raise TimeoutError("Remote stream did not become ready")
            response = self._process.stdout.readline(8192)
            receipt = json.loads(response)
            if receipt != {"type": "ready", "session": self.session, "schema": "fr5-stream-v1"}:
                raise SpoolError("Invalid remote stream handshake")
            self._sender = threading.Thread(target=self._send_loop, name="fr5-frame-send", daemon=True)
            self._reader = threading.Thread(target=self._ack_loop, name="fr5-frame-ack", daemon=True)
            self._sender.start()
            self._reader.start()
            return self
        except BaseException:
            self.abort()
            raise

    def _fail(self, exc):
        with self._condition:
            if self._error is None:
                self._error = f"{type(exc).__name__}: {exc}"
            self._condition.notify_all()

    def _check_locked(self):
        if not self._started or self._process is None:
            raise SpoolError("Frame stream is not started")
        if self._error is not None:
            raise SpoolError("Frame stream failed: " + self._error)
        if self._process.poll() is not None:
            raise SpoolError("Frame receiver exited")
        if self._pending:
            oldest = next(iter(self._pending.values()))
            if time.monotonic() - oldest["queued"] > self.max_unacked_age:
                raise SpoolError("Oldest frame has not been durably acknowledged in time")

    def assert_healthy(self):
        with self._condition:
            self._check_locked()
            if self._closing:
                raise SpoolError("Frame stream is closing")
            if len(self._pending) >= self.max_frames or self._bytes >= self.max_bytes:
                raise SpoolError("Frame stream backlog is full")

    def enqueue(self, *, capture, teacher, media, rgb, depth):
        """Persist one actual RGB-D sample locally, then enqueue for SSH.

        The caller never waits for network I/O. Capacity failure is explicit;
        it never discards, overwrites, or silently substitutes a source frame.
        """
        started = time.monotonic()
        with self._producer_lock:
            with self._condition:
                self._check_locked()
                if self._closing:
                    raise SpoolError("Frame stream is closing")
                sequence = self._next_sequence
            metadata = {"schema": "fr5-stream-frame-v1", "session": self.session,
                        "sequence": sequence, "capture": capture, "teacher": teacher, "media": media}
            packet = protocol.encode_record(metadata, rgb, depth)
            digest = hashlib.sha256(packet).hexdigest()
            with self._condition:
                self._check_locked()
                if len(self._pending) >= self.max_frames or self._bytes + len(packet) > self.max_bytes:
                    raise SpoolError("Frame stream backlog is full")
            path = self.outbox / f"{sequence:09d}-{digest}.packet"
            partial = self.outbox / f"{sequence:09d}-{digest}.partial"
            try:
                with partial.open("xb") as handle:
                    handle.write(packet)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.rename(partial, path)
                protocol._sync_dir(self.outbox)
            except BaseException as exc:
                self._fail(exc)
                raise
            with self._condition:
                # If the receiver failed during local fsync, keep this packet
                # for manual recovery but never report it as accepted.
                self._check_locked()
                self._pending[sequence] = {"path": path, "sha256": digest,
                                           "size": len(packet), "queued": time.monotonic(), "sent": False}
                self._bytes += len(packet)
                self._peak_frames = max(self._peak_frames, len(self._pending))
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                self._max_local_enqueue = max(self._max_local_enqueue, time.monotonic() - started)
                self._next_sequence += 1
                self._send_queue.append(sequence)
                self._condition.notify_all()
            return sequence

    def _send_loop(self):
        try:
            while True:
                with self._condition:
                    while not self._send_queue and not self._closing and self._error is None:
                        self._condition.wait()
                    if self._error is not None or (not self._send_queue and self._closing):
                        return
                    sequence = self._send_queue.popleft()
                    entry = self._pending[sequence]
                    entry["sent"] = True
                # Neither local file nor SSH write is inside the state lock.
                with entry["path"].open("rb") as handle:
                    while True:
                        chunk = handle.read(65536)
                        if not chunk:
                            break
                        view = memoryview(chunk)
                        while view:
                            written = self._process.stdin.write(view)
                            if not written:
                                raise OSError("Short SSH pipe write")
                            view = view[written:]
        except BaseException as exc:
            self._fail(exc)

    def _ack_loop(self):
        try:
            while True:
                raw = self._process.stdout.readline(8192)
                if not raw:
                    raise SpoolError("Receiver closed without a final receipt")
                item = json.loads(raw)
                if item.get("session") != self.session:
                    raise SpoolError("Receipt session mismatch")
                if item.get("type") == "end":
                    with self._condition:
                        if (not self._closing or self._pending or item.get("clean_eof") is not True
                                or item.get("error") is not None or item.get("frames") != self._next_sequence):
                            raise SpoolError("Receiver ended with an incomplete stream")
                        self._end_receipt = item
                        self._condition.notify_all()
                    return
                with self._condition:
                    sequence = item.get("sequence")
                    if (item.get("type") != "ack" or type(sequence) is not int
                            or sequence != self._last_ack + 1 or sequence not in self._pending):
                        raise SpoolError("Out-of-order or unexpected frame receipt")
                    entry = self._pending[sequence]
                    if not entry["sent"] or item.get("sha256") != entry["sha256"]:
                        raise SpoolError("Frame receipt hash mismatch")
                    del self._pending[sequence]
                    self._bytes -= entry["size"]
                    self._last_ack = sequence
                    self._max_ack_latency = max(self._max_ack_latency, time.monotonic() - entry["queued"])
                    self._condition.notify_all()
                # The packet may be removed only after the durable server ACK.
                entry["path"].unlink()
                protocol._sync_dir(self.outbox)
        except BaseException as exc:
            self._fail(exc)

    def watermark(self):
        with self._condition:
            self._check_locked()
            return self._next_sequence - 1

    def wait_for_ack(self, sequence, *, timeout=8.):
        deadline = time.monotonic() + timeout
        with self._condition:
            if type(sequence) is not int or sequence >= self._next_sequence or sequence < -1:
                raise ValueError("Unknown frame watermark")
            while self._last_ack < sequence:
                self._check_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for durable frame receipt")
                self._condition.wait(min(remaining, .1))

    def status(self):
        with self._condition:
            return {"session": self.session, "accepted_frames": self._next_sequence,
                    "last_durable_sequence": self._last_ack,
                    "unacknowledged_frames": len(self._pending), "unacknowledged_bytes": self._bytes,
                    "peak_unacknowledged_frames": self._peak_frames, "peak_unacknowledged_bytes": self._peak_bytes,
                    "max_ack_latency_s": self._max_ack_latency, "max_local_enqueue_s": self._max_local_enqueue,
                    "error": self._error}

    def close(self, *, timeout=None):
        if timeout is None:
            timeout = max(10., self.max_unacked_age + 5.)
        deadline = time.monotonic() + timeout
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            while self._pending:
                self._check_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Unacknowledged frames remain in the local outbox")
                self._condition.wait(min(remaining, .1))
        self._sender.join(max(0., deadline - time.monotonic()))
        if self._sender.is_alive():
            raise TimeoutError("Frame sender did not drain")
        self._process.stdin.close()
        self._reader.join(max(0., deadline - time.monotonic()))
        if self._reader.is_alive() or self._end_receipt is None:
            raise SpoolError("No complete receiver shutdown receipt")
        if self._process.wait(timeout=max(.01, deadline - time.monotonic())) != 0:
            raise SpoolError("Receiver failed during shutdown")
        self._process.stdout.close()
        return self.status()

    def abort(self):
        """Stop only our transport; deliberately leave unacknowledged packets."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
        for thread_name in ("_sender", "_reader"):
            thread = getattr(self, thread_name, None)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1)
        if self._process is not None:
            for pipe in (self._process.stdin, self._process.stdout):
                if pipe is not None and not pipe.closed:
                    pipe.close()

