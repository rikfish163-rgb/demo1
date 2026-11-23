"""Bounded RGB-D wire encoding; no device, filesystem or network operations.

RGB is explicitly JPEG-compressed. Depth is bit-exact float32 after decoding;
integer depths use a reversible uint16 row predictor, never quantization.
"""
from __future__ import annotations

import hashlib
import io
import zlib

import cv2
import numpy as np
from PIL import Image

MAX_PIXELS = 1920 * 1080
MAX_BLOB_BYTES = 16 * 1024**2
SCHEMA = "fr5-rgbd-wire-media-v1"


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _shape(metadata):
    h, w = metadata.get("height"), metadata.get("width")
    if (type(h) is not int or type(w) is not int or h < 1 or w < 1
            or h * w > MAX_PIXELS):
        raise ValueError("Invalid bounded RGB-D shape")
    return h, w


def _inflate(payload, expected):
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, expected + 1)
    except zlib.error as exc:
        raise ValueError("Invalid depth compression") from exc
    if (len(raw) != expected or not decoder.eof or decoder.unconsumed_tail
            or decoder.unused_data):
        raise ValueError("Depth byte count or compression boundary mismatch")
    return raw


def encode_media(color_rgb, depth_mm, *, jpeg_quality=90):
    """Return (metadata, JPEG bytes, lossless depth bytes) from one real sample."""
    color, depth = np.asarray(color_rgb), np.asarray(depth_mm)
    if (color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8
            or depth.ndim != 2 or depth.dtype != np.float32
            or color.shape[:2] != depth.shape):
        raise ValueError("Expected aligned-size RGB uint8 and depth float32 arrays")
    metadata = {"schema": SCHEMA, "height": depth.shape[0], "width": depth.shape[1]}
    _shape(metadata)
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Depth must be finite nonnegative millimeters")
    if type(jpeg_quality) is not int or not 90 <= jpeg_quality <= 100:
        raise ValueError("Demonstration JPEG quality must be90..100")
    color = np.ascontiguousarray(color)
    depth = np.ascontiguousarray(depth, dtype="<f4")
    depth_raw = depth.tobytes()
    ok, jpeg = cv2.imencode(".jpg", color[:, :, ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("RGB JPEG encoding failed")
    rgb_payload = jpeg.tobytes()
    reconstructed_rgb = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)[:, :, ::-1]
    depth_codec, depth_encoded_raw = "float32le-zlib", depth_raw
    if float(depth.max()) <= 65535:
        integer = depth.astype("<u2")
        if integer.astype("<f4").tobytes() == depth_raw:
            previous = np.pad(integer[:, :-1].astype(np.int32), ((0, 0), (1, 0)))
            delta = (integer.astype(np.int32) - previous).astype("<u2")
            depth_codec, depth_encoded_raw = "uint16le-rowdelta-zlib", delta.tobytes()
    depth_payload = zlib.compress(depth_encoded_raw, level=6)
    metadata.update(
        rgb_codec="jpeg", rgb_lossy=True, jpeg_quality=jpeg_quality,
        rgb_color_order="RGB", rgb_encoder="opencv-" + cv2.__version__,
        rgb_source_sha256=_sha(color.tobytes()),
        rgb_mean_abs_error_8bit=float(np.abs(color.astype(np.int16)
                                            - reconstructed_rgb.astype(np.int16)).mean()),
        rgb_bytes=len(rgb_payload), rgb_sha256=_sha(rgb_payload),
        depth_codec=depth_codec, depth_dtype="float32", depth_unit="mm",
        depth_lossless=True, depth_source_sha256=_sha(depth_raw),
        depth_bytes=len(depth_payload), depth_sha256=_sha(depth_payload),
    )
    return metadata, rgb_payload, depth_payload


def decode_media(metadata, rgb_payload, depth_payload):
    """Verify transmitted bytes, dimensions and exact depth before returning RGB-D."""
    if (not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA
            or metadata.get("rgb_codec") != "jpeg" or metadata.get("rgb_lossy") is not True
            or metadata.get("rgb_color_order") != "RGB"
            or type(metadata.get("jpeg_quality")) is not int
            or not 90 <= metadata["jpeg_quality"] <= 100
            or metadata.get("depth_dtype") != "float32" or metadata.get("depth_unit") != "mm"
            or metadata.get("depth_lossless") is not True):
        raise ValueError("Invalid media contract")
    h, w = _shape(metadata)
    for name, payload in (("rgb", rgb_payload), ("depth", depth_payload)):
        if (not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BLOB_BYTES
                or metadata.get(name + "_bytes") != len(payload)
                or metadata.get(name + "_sha256") != _sha(payload)):
            raise ValueError("Encoded media size or hash mismatch")
    # Read the JPEG header before allowing a decoder allocation.
    with Image.open(io.BytesIO(rgb_payload)) as header:
        if header.format != "JPEG" or header.size != (w, h):
            raise ValueError("JPEG shape does not match source metadata")
    bgr = cv2.imdecode(np.frombuffer(rgb_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (h, w, 3):
        raise ValueError("JPEG decode failed")
    if metadata["depth_codec"] == "uint16le-rowdelta-zlib":
        delta = np.frombuffer(_inflate(depth_payload, h * w * 2), dtype="<u2").reshape(h, w)
        depth = (np.cumsum(delta, axis=1, dtype=np.uint64) & 65535).astype("<f4")
    elif metadata["depth_codec"] == "float32le-zlib":
        depth = np.frombuffer(_inflate(depth_payload, h * w * 4), dtype="<f4").reshape(h, w).copy()
    else:
        raise ValueError("Unsupported depth codec")
    if (not np.isfinite(depth).all() or np.any(depth < 0)
            or _sha(depth.tobytes()) != metadata.get("depth_source_sha256")):
        raise ValueError("Decoded depth differs from original float32 values")
    return np.ascontiguousarray(bgr[:, :, ::-1]), depth
