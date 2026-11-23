"""Read aligned RGB-D snapshots without opening the physical camera or robot.

This adapter polls the existing bridge on demand; it does not run an acquisition
thread. ``read_latest`` only peeks at the last successfully decoded local bundle.
The source timestamp is provided by the bridge, not a sensor clock. Updated
bridges stamp the frameset when received, before alignment or image encoding.
"""

from copy import deepcopy
from io import BytesIO
import json
import math
from threading import Lock
import time
from typing import Any
import urllib.request
import zipfile

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .configuration_gemini import GeminiBridgeConfig
from .timing import camera_clock, MONOTONIC


_MAX_PACKET_BYTES = 32 * 1024 * 1024
_MAX_EXPANDED_BYTES = 64 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Camera bridge redirects are not allowed")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class GeminiBridgeCamera(Camera):
    """LeRobot Camera backed by a single synchronized NPZ snapshot response."""

    def __init__(self, config: GeminiBridgeConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._latest_bundle: dict[str, Any] | None = None
        self._last_delivered_timestamp: float | None = None
        self._last_delivered_clock = None
        self._io_lock = Lock()
        # Ignore proxy environment variables: this endpoint is strictly local.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # This plugin never scans USB devices or starts the bridge itself.
        return []

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Gemini bridge camera is already connected")
        self._latest_bundle = None
        self._last_delivered_timestamp = None
        self._last_delivered_clock = None
        if warmup:
            self._latest_bundle = self._fetch_bundle(self.config.timeout_s)
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._latest_bundle = None
        self._last_delivered_timestamp = None
        self._last_delivered_clock = None

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Gemini bridge camera is not connected")

    def decode_bundle(self, payload: bytes, *, received_unix: float | None = None,
                      received_monotonic: float | None = None) -> dict[str, Any]:
        """Decode and validate bytes; no network/device access (also useful for replay)."""
        if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_PACKET_BYTES:
            raise ValueError("Invalid or oversized camera packet")
        received_unix = time.time() if received_unix is None else received_unix
        received_monotonic = time.monotonic() if received_monotonic is None else received_monotonic
        received_unix = _finite_number(received_unix, "received_unix")
        received_monotonic = _finite_number(received_monotonic, "received_monotonic")
        try:
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                entries = archive.infolist()
                names = {e.filename for e in entries}
                required = {"color_bgr.npy", "depth_mm.npy", "metadata.npy"}
                if (len(entries) != len(names) or not required <= names
                        or names - required - {"raw_depth_mm.npy"}):
                    raise ValueError("Camera packet requires color_bgr, depth_mm, metadata and optional raw_depth_mm only")
                if sum(e.file_size for e in entries) > _MAX_EXPANDED_BYTES:
                    raise ValueError("Expanded camera packet is too large")
            with np.load(BytesIO(payload), allow_pickle=False) as packet:
                color = packet["color_bgr"]
                depth = packet["depth_mm"]
                raw_meta = packet["metadata"]
                if raw_meta.shape != () or raw_meta.dtype.kind not in ("U", "S"):
                    raise ValueError("metadata must be a scalar JSON string")
                metadata = json.loads(raw_meta.item())
                raw_depth = packet["raw_depth_mm"] if "raw_depth_mm" in packet.files else None
        except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Invalid camera packet: {exc}") from exc
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3 or min(color.shape[:2]) < 1:
            raise ValueError("color_bgr must be nonempty HWC uint8 with three channels")
        if depth.shape != color.shape[:2] or depth.dtype.kind != "f" or depth.dtype.itemsize != 4:
            raise ValueError("depth_mm must be aligned HW float32")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("depth_mm must be finite and nonnegative; zero denotes missing depth")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        if raw_depth is not None:
            profile = metadata.get("raw_depth_profile", {}).get("intrinsic", {})
            if (raw_depth.ndim != 2 or min(raw_depth.shape) < 1
                    or raw_depth.dtype != np.float32
                    or raw_depth.shape != (profile.get("height"), profile.get("width"))
                    or not np.isfinite(raw_depth).all() or np.any(raw_depth < 0)):
                raise ValueError("raw_depth_mm must match the native depth profile as finite nonnegative HW float32")
        camera, capture = metadata.get("camera"), metadata.get("capture")
        if not isinstance(camera, dict) or camera.get("serial") != self.config.expected_serial:
            raise ValueError("Camera serial does not match expected_serial")
        if not isinstance(capture, dict) or capture.get("alignment") != "depth_to_color":
            raise ValueError("Camera capture must be aligned depth_to_color")
        if capture.get("depth_unit") != "mm":
            raise ValueError("Camera depth_unit must be mm")
        captured = _finite_number(capture.get("time_unix"), "capture.time_unix")
        captured_mono = capture.get('received_monotonic')
        if captured_mono is not None:
            captured_mono = _finite_number(captured_mono,'capture.received_monotonic')
            if captured_mono <= 0 or captured_mono > received_monotonic + .1:
                raise ValueError('Camera monotonic clock is invalid or not on this host')
            age = received_monotonic-captured_mono
        else:
            age = received_unix-captured
        if captured <= 0 or (captured_mono is None and captured > received_unix + 0.1):
            raise ValueError("Camera capture timestamp is invalid or in the future")
        if age > self.config.max_age_s:
            raise TimeoutError("Camera capture is stale")
        intrinsic = metadata.get("color_intrinsic")
        if not isinstance(intrinsic, dict):
            raise ValueError("Missing color_intrinsic")
        source_h, source_w = depth.shape
        if intrinsic.get("width") != source_w or intrinsic.get("height") != source_h:
            raise ValueError("Intrinsics dimensions do not match the source images")
        values = {key: _finite_number(intrinsic.get(key), f"color_intrinsic.{key}")
                  for key in ("fx", "fy", "cx", "cy")}
        if values["fx"] <= 0 or values["fy"] <= 0:
            raise ValueError("Focal lengths must be positive")
        sx, sy = self.config.width / source_w, self.config.height / source_h
        scaled = dict(intrinsic, fx=values["fx"] * sx, fy=values["fy"] * sy,
                      cx=values["cx"] * sx, cy=values["cy"] * sy,
                      width=self.config.width, height=self.config.height)
        # Explicit nearest-neighbor decimation; never average depth across edges.
        if (source_w, source_h) == (self.config.width, self.config.height):
            color_rgb = np.ascontiguousarray(color[..., ::-1])
            depth_mm = np.ascontiguousarray(depth, dtype=np.float32)
        else:
            yy = np.arange(self.config.height, dtype=np.int64) * source_h // self.config.height
            xx = np.arange(self.config.width, dtype=np.int64) * source_w // self.config.width
            color_rgb = np.ascontiguousarray(color[yy[:, None], xx[None, :], ::-1])
            depth_mm = np.ascontiguousarray(depth[yy[:, None], xx[None, :]], dtype=np.float32)
        meta = deepcopy(metadata)
        # A fast bridge packet already carries scaled intrinsics. Preserve its
        # full-resolution original as provenance; scale only the packet matrix.
        original = metadata.get("color_intrinsic_original", intrinsic)
        timestamp_origin = ("bridge_host_frame_receive_unix"
                            if capture.get("timestamp_origin") == "host_frame_receive_unix"
                            else "bridge_host_capture_unix")
        meta.update(color_intrinsic_original=deepcopy(original), color_intrinsic_scaled=scaled,
                    color_intrinsic_packet=deepcopy(intrinsic),
                    color_intrinsic=deepcopy(scaled), source_timestamp_unix=captured,
                    source_camera_serial=camera["serial"], received_unix=received_unix,
                    received_monotonic=received_monotonic,
                    timestamp_origin=timestamp_origin, resize_method="nearest")
        if captured_mono is not None:
            meta['source_timestamp_monotonic'] = captured_mono
        # Raw RGB may be distorted while SDK-aligned depth uses a rectified
        # grid. Equal image sizes alone do not establish pixel correspondence.
        aligned = metadata.get("aligned_depth_profile", {}).get("intrinsic")
        if aligned is not None:
            if aligned.get("width") != source_w or aligned.get("height") != source_h:
                raise ValueError("Aligned depth profile dimensions differ from depth_mm")
            depth_intrinsic = {key: _finite_number(aligned.get(key), f"aligned_depth.{key}")
                               for key in ("fx", "fy", "cx", "cy")}
            if depth_intrinsic["fx"] <= 0 or depth_intrinsic["fy"] <= 0:
                raise ValueError("Aligned depth focal lengths must be positive")
            meta["aligned_depth_profile_packet"] = deepcopy(metadata["aligned_depth_profile"])
            meta["aligned_depth_profile"]["intrinsic"] = dict(
                aligned, fx=depth_intrinsic["fx"] * sx, fy=depth_intrinsic["fy"] * sy,
                cx=depth_intrinsic["cx"] * sx, cy=depth_intrinsic["cy"] * sy,
                width=self.config.width, height=self.config.height)
        meta["pixel_geometry"] = {
            "rgb": "source color pixels, resized without rectification",
            "depth": "SDK depth_to_color grid, resized independently",
            "same_pixel_correspondence_verified": False,
            "raw_depth_in_source_packet": raw_depth is not None,
            "raw_depth_retained_in_bundle": False,
        }
        return {"color_rgb": color_rgb, "depth_mm": depth_mm, "metadata": meta}

    def _fetch_bundle(self, timeout_s: float) -> dict[str, Any]:
        request = urllib.request.Request(self.config.url, headers={"Cache-Control": "no-cache"})
        with self._opener.open(request, timeout=timeout_s) as response:
            payload = response.read(_MAX_PACKET_BYTES + 1)
        return self.decode_bundle(payload, received_unix=time.time(), received_monotonic=time.monotonic())

    def read_bundle(self, *, timeout_ms: float | None = None, require_new: bool = True) -> dict[str, Any]:
        """Fetch a same-frame RGB-D bundle, requiring an increasing timestamp by default.

        ``fps`` is advisory for the caller's recording loop, not a claim that the
        bridge captures at this rate. No USB device or robot SDK is initialized.
        """
        self._ensure_connected()
        duration = self.config.timeout_s if timeout_ms is None else _finite_number(timeout_ms, "timeout_ms") / 1000
        if duration <= 0:
            raise ValueError("timeout_ms must be positive")
        deadline = time.monotonic() + duration
        if not self._io_lock.acquire(timeout=duration):
            raise TimeoutError("Camera read is busy")
        try:
            while True:
                self._ensure_connected()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No new Gemini frame before timeout")
                bundle = self._fetch_bundle(min(self.config.timeout_s, remaining))
                if time.monotonic() > deadline:
                    raise TimeoutError("Camera HTTP read exceeded deadline")
                timestamp,basis = camera_clock(bundle['metadata'])
                if self._last_delivered_clock is not None and basis != self._last_delivered_clock:
                    raise ValueError('Camera clock domain changed during connection')
                if (not require_new or self._last_delivered_timestamp is None
                        or timestamp > self._last_delivered_timestamp):
                    self._ensure_connected()
                    self._latest_bundle = bundle
                    self._last_delivered_timestamp = max(timestamp, self._last_delivered_timestamp or timestamp)
                    self._last_delivered_clock = basis
                    return deepcopy(bundle)
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        finally:
            self._io_lock.release()

    def read(self) -> np.ndarray:
        return self.read_bundle()["color_rgb"]

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self.read_bundle(timeout_ms=timeout_ms, require_new=True)["color_rgb"]

    def read_latest(self, max_age_ms: int = 1000) -> np.ndarray:
        """Nonblocking cached RGB peek; call read/read_bundle to refresh the cache."""
        self._ensure_connected()
        max_age_s = _finite_number(max_age_ms, "max_age_ms") / 1000
        if max_age_s < 0:
            raise ValueError("max_age_ms must be nonnegative")
        bundle = self._latest_bundle
        if bundle is None:
            raise RuntimeError("Camera has not captured a bundle yet")
        meta = bundle["metadata"]
        age = (time.monotonic()-meta['source_timestamp_monotonic'] if 'source_timestamp_monotonic' in meta else
               max(time.time() - meta["source_timestamp_unix"],
                  meta["received_unix"] - meta["source_timestamp_unix"]
                  + time.monotonic() - meta["received_monotonic"]))
        if age > min(max_age_s, self.config.max_age_s):
            raise TimeoutError("Cached Gemini frame is stale")
        return bundle["color_rgb"].copy()
