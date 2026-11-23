"""Binary RGB-D spool protocol and the standalone remote receiver.

The receiver has no device dependencies. Its stdout is reserved for JSON
receipts; each frame receipt is emitted only after the server has fsynced the
complete record. An interrupted stream remains explicitly incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import struct
import sys
import time


MAGIC = b"F5S1"
HEADER = struct.Struct("!4sIII32s")
MAX_META = 1024 * 1024
MAX_MEDIA = 16 * 1024 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
SESSION = re.compile(r"[0-9a-f]{32}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REMOTE_BASE = Path("/home/data4/hts/demo1/fr5-learning/recording-spool")


def encode_record(metadata, rgb, depth):
    if not isinstance(metadata, dict) or not isinstance(rgb, bytes) or not isinstance(depth, bytes):
        raise ValueError("A metadata object and RGB/depth byte strings are required")
    meta = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(meta) <= MAX_META or not 0 < len(rgb) <= MAX_MEDIA or not 0 < len(depth) <= MAX_MEDIA:
        raise ValueError("Frame exceeds protocol limits or is missing media")
    payload = meta + rgb + depth
    return HEADER.pack(MAGIC, len(meta), len(rgb), len(depth), hashlib.sha256(payload).digest()) + payload


def unpack_record(packet):
    if len(packet) < HEADER.size:
        raise ValueError("Partial frame header")
    magic, nm, nr, nd, digest = HEADER.unpack_from(packet)
    if magic != MAGIC or not 0 < nm <= MAX_META or not 0 < nr <= MAX_MEDIA or not 0 < nd <= MAX_MEDIA:
        raise ValueError("Invalid frame header")
    if len(packet) != HEADER.size + nm + nr + nd:
        raise ValueError("Partial or oversized frame")
    payload = packet[HEADER.size:]
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("Frame checksum mismatch")
    metadata = json.loads(payload[:nm])
    if not isinstance(metadata, dict):
        raise ValueError("Frame metadata must be an object")
    return metadata, payload[nm:nm + nr], payload[nm + nr:]


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _receipt(out, item):
    out.write((json.dumps(item, separators=(",", ":")) + "\n").encode("ascii"))
    out.flush()


def receive(stream_fd, out, base, name, session, *, batch_size=3, batch_seconds=.25,
            sync=os.fsync):
    """Receive a fresh session; return False for a truncated or invalid tail.

    ``base`` and ``sync`` are injectable for isolated tests. CLI production
    uses the fixed server spool path and the real fsync implementation.
    """
    base = Path(base)
    if not NAME.fullmatch(name) or not SESSION.fullmatch(session):
        raise ValueError("Invalid session name or identifier")
    if not base.is_dir() or base.is_symlink() or base.resolve() != base:
        raise ValueError("Spool parent must be an existing, non-symlink directory")
    if batch_size < 1 or not 0 < batch_seconds <= 1:
        raise ValueError("Invalid fsync batching")
    root = base / name
    root.mkdir(mode=0o700, exist_ok=False)
    _sync_dir(base)
    marker = root / "INCOMPLETE"
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "fr5-stream-v1", "session": session,
                                 "state": "receiving_or_interrupted"}) + "\n")
        handle.flush()
        sync(handle.fileno())
    segment_path = root / "frames.stream"
    buffer = bytearray()
    pending = []
    next_sequence = 0
    last_sync = time.monotonic()
    clean_eof = False
    error = None
    with segment_path.open("xb", buffering=0) as segment:
        _sync_dir(root)
        _receipt(out, {"type": "ready", "session": session, "schema": "fr5-stream-v1"})

        def flush():
            nonlocal last_sync
            if pending:
                sync(segment.fileno())
                for sequence, digest in pending:
                    _receipt(out, {"type": "ack", "session": session,
                                   "sequence": sequence, "sha256": digest})
                pending.clear()
            last_sync = time.monotonic()

        try:
            while True:
                timeout = max(0.0, batch_seconds - (time.monotonic() - last_sync)) if pending else None
                ready, _, _ = select.select([stream_fd], [], [], timeout)
                if not ready:
                    flush()
                    continue
                chunk = os.read(stream_fd, 65536)
                if not chunk:
                    clean_eof = not buffer
                    if not clean_eof:
                        error = "truncated frame"
                    break
                buffer.extend(chunk)
                while len(buffer) >= HEADER.size:
                    magic, nm, nr, nd, _ = HEADER.unpack_from(buffer)
                    if magic != MAGIC or not 0 < nm <= MAX_META or not 0 < nr <= MAX_MEDIA or not 0 < nd <= MAX_MEDIA:
                        raise ValueError("Invalid frame header")
                    size = HEADER.size + nm + nr + nd
                    if len(buffer) < size:
                        break
                    packet = bytes(buffer[:size])
                    del buffer[:size]
                    meta, rgb, depth = unpack_record(packet)
                    media = meta.get("media")
                    if (meta.get("schema") != "fr5-stream-frame-v1"
                            or meta.get("session") != session
                            or type(meta.get("sequence")) is not int
                            or meta["sequence"] != next_sequence
                            or not isinstance(meta.get("capture"), dict)
                            or not isinstance(meta.get("teacher"), dict)
                            or not isinstance(media, dict)
                            or media.get("schema") != "fr5-rgbd-wire-media-v1"
                            or media.get("rgb_codec") != "jpeg" or media.get("jpeg_quality") != 90
                            or media.get("rgb_lossy") is not True
                            or media.get("depth_codec") not in ("uint16le-rowdelta-zlib", "float32le-zlib")
                            or media.get("depth_dtype") != "float32" or media.get("depth_unit") != "mm"
                            or media.get("depth_lossless") is not True
                            or not isinstance(media.get("rgb_source_sha256"), str)
                            or not SHA256.fullmatch(media["rgb_source_sha256"])
                            or not isinstance(media.get("depth_source_sha256"), str)
                            or not SHA256.fullmatch(media["depth_source_sha256"])
                            or media.get("rgb_bytes") != len(rgb)
                            or media.get("depth_bytes") != len(depth)
                            or media.get("rgb_sha256") != hashlib.sha256(rgb).hexdigest()
                            or media.get("depth_sha256") != hashlib.sha256(depth).hexdigest()):
                        raise ValueError("Invalid sequence, provenance, or media contract")
                    # FileIO.write may legally be short; complete the record.
                    view = memoryview(packet)
                    while view:
                        count = segment.write(view)
                        if not count:
                            raise OSError("Short stream write")
                        view = view[count:]
                    pending.append((next_sequence, hashlib.sha256(packet).hexdigest()))
                    next_sequence += 1
                    if len(pending) >= batch_size:
                        flush()
                if pending and time.monotonic() - last_sync >= batch_seconds:
                    flush()
        except (ValueError, UnicodeError, json.JSONDecodeError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Even preceding valid frames in a truncated stream can be confirmed.
        flush()
        _receipt(out, {"type": "end", "session": session, "frames": next_sequence,
                       "clean_eof": clean_eof, "error": error})
    return clean_eof and error is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("session")
    args = parser.parse_args()
    if not receive(sys.stdin.fileno(), sys.stdout.buffer, REMOTE_BASE, args.name, args.session):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
