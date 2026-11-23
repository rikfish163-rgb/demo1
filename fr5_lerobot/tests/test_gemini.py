"""Offline camera contract tests. All HTTP is mocked; no devices are touched."""

from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot_camera_gemini import GeminiBridgeCamera, GeminiBridgeConfig


def packet(*, timestamp=None, serial="CP1E54200015", alignment="depth_to_color",
           unit="mm", color=None, depth=None, intrinsic=None, object_metadata=False):
    color = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3) if color is None else color
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) + 500 if depth is None else depth
    meta = {"camera": {"serial": serial, "name": "Gemini 335"},
            "capture": {"time_unix": time.time() if timestamp is None else timestamp,
                        "alignment": alignment, "depth_unit": unit},
            "color_intrinsic": intrinsic or {"fx": 100., "fy": 110., "cx": 3., "cy": 2., "width": 6, "height": 4}}
    raw_meta = np.array(meta, dtype=object) if object_metadata else np.array(json.dumps(meta))
    stream = BytesIO()
    np.savez_compressed(stream, color_bgr=color, depth_mm=depth, metadata=raw_meta)
    return stream.getvalue()


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.camera = GeminiBridgeCamera(GeminiBridgeConfig(width=3, height=2))

    def tearDown(self):
        self.camera.disconnect()

    def test_factory_constructs_without_network(self):
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network attempted")):
            camera = make_cameras_from_configs({"overhead": GeminiBridgeConfig()})["overhead"]
        self.assertIsInstance(camera, GeminiBridgeCamera)
        self.assertFalse(camera.is_connected)

    def test_same_frame_rgb_depth_resize_and_intrinsics(self):
        bundle = self.camera.decode_bundle(packet(timestamp=1000), received_unix=1000.2, received_monotonic=22.)
        expected = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[::2, ::2, ::-1]
        np.testing.assert_array_equal(bundle["color_rgb"], expected)
        np.testing.assert_array_equal(bundle["depth_mm"], (np.arange(24, dtype=np.float32).reshape(4, 6) + 500)[::2, ::2])
        self.assertEqual(bundle["depth_mm"].dtype, np.float32)
        self.assertTrue(bundle["color_rgb"].flags.c_contiguous)
        meta = bundle["metadata"]
        self.assertEqual(meta["color_intrinsic_scaled"], {"fx": 50., "fy": 55., "cx": 1.5, "cy": 1., "width": 3, "height": 2})
        self.assertEqual(meta["color_intrinsic_original"]["width"], 6)
        self.assertEqual(meta["capture"]["time_unix"], 1000)
        self.assertEqual(meta["source_timestamp_unix"], 1000)
        self.assertEqual(meta["source_camera_serial"], "CP1E54200015")
        self.assertEqual(meta["received_unix"], 1000.2)
        self.assertEqual(meta["received_monotonic"], 22.)

    def test_rejects_wrong_serial_alignment_units_stale_future(self):
        for kwargs in ({"serial": "wrong"}, {"alignment": "none"}, {"unit": "m"},
                       {"timestamp": 998}, {"timestamp": 1001}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TimeoutError)):
                self.camera.decode_bundle(packet(timestamp=1000, **kwargs) if "timestamp" not in kwargs else packet(**kwargs),
                                          received_unix=1000., received_monotonic=22.)

    def test_rejects_bad_packet_and_array_contracts(self):
        bad_depth = np.ones((4, 6), dtype=np.float32); bad_depth[0, 0] = np.nan
        invalid = [b"not npz", packet(object_metadata=True), packet(color=np.zeros((4, 6, 3), np.float32)),
                   packet(depth=np.ones((4, 5), np.float32)), packet(depth=np.ones((4, 6), np.uint16)),
                   packet(depth=bad_depth), packet(depth=-np.ones((4, 6), np.float32)),
                   packet(intrinsic={"fx": float("nan"), "fy": 110., "cx": 3., "cy": 2., "width": 6, "height": 4})]
        for payload in invalid:
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                self.camera.decode_bundle(payload)

    def test_current_four_array_packet_preserves_two_pixel_models(self):
        with np.load(BytesIO(packet(timestamp=1000)), allow_pickle=False) as original:
            arrays = {key: original[key] for key in original.files}
        meta = json.loads(arrays["metadata"].item())
        meta["raw_depth_profile"] = {"intrinsic": {"width": 3, "height": 2}}
        meta["aligned_depth_profile"] = {"intrinsic": dict(meta["color_intrinsic"]),
                                         "distortion": {"k1": 0.}}
        meta["color_distortion"] = {"k1": .1}
        arrays["metadata"] = np.array(json.dumps(meta))
        arrays["raw_depth_mm"] = np.full((2, 3), 600, np.float32)
        output = BytesIO(); np.savez_compressed(output, **arrays)
        bundle = self.camera.decode_bundle(output.getvalue(), received_unix=1000.2)
        self.assertEqual(bundle["metadata"]["aligned_depth_profile"]["intrinsic"]["width"], 3)
        self.assertEqual(bundle["metadata"]["aligned_depth_profile_packet"]["intrinsic"]["width"], 6)
        self.assertEqual(bundle["metadata"]["color_distortion"]["k1"], .1)
        self.assertTrue(bundle["metadata"]["pixel_geometry"]["raw_depth_in_source_packet"])
        self.assertFalse(bundle["metadata"]["pixel_geometry"]["same_pixel_correspondence_verified"])
        for key, value in (("raw_depth_mm", np.ones((4, 3), np.float32)),
                           ("unexpected", np.zeros(1))):
            changed = dict(arrays); changed[key] = value
            invalid = BytesIO(); np.savez_compressed(invalid, **changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.camera.decode_bundle(invalid.getvalue(), received_unix=1000.2)

    def test_mock_http_duplicate_frame_timeout_and_latest_cache(self):
        payload = packet()
        with patch.object(self.camera._opener, "open", side_effect=lambda *a, **k: BytesIO(payload)) as opened:
            self.camera.connect()
            first = self.camera.async_read(timeout_ms=100)
            calls = opened.call_count
            np.testing.assert_array_equal(self.camera.read_latest(), first)
            self.assertEqual(opened.call_count, calls)
            with self.assertRaises(TimeoutError):
                self.camera.async_read(timeout_ms=15)
            # Explicit peeks may use a duplicate but must preserve source time.
            duplicate = self.camera.read_bundle(require_new=False)
            self.assertEqual(duplicate["metadata"]["source_timestamp_unix"], self.camera._last_delivered_timestamp)
        first[:] = 0
        self.assertTrue(self.camera.read_latest().any())

    def test_latest_rejects_stale_cache_and_disconnected(self):
        with self.assertRaises(DeviceNotConnectedError):
            self.camera.read_latest()
        self.camera.connect(warmup=False)
        with self.assertRaises(RuntimeError):
            self.camera.read_latest()
        self.camera._latest_bundle = self.camera.decode_bundle(packet())
        self.camera._latest_bundle["metadata"]["received_monotonic"] -= 2
        with self.assertRaises(TimeoutError):
            self.camera.read_latest()

    def test_new_frame_skips_repeated_and_reversed_source_timestamps(self):
        now = time.time()
        payloads = [packet(timestamp=now - 0.1), packet(timestamp=now - 0.2),
                    packet(timestamp=now - 0.1), packet(timestamp=now)]
        self.camera.connect(warmup=False)
        with patch.object(self.camera._opener, "open",
                          side_effect=[BytesIO(payload) for payload in payloads]) as opened:
            first = self.camera.read_bundle(timeout_ms=200)
            second = self.camera.read_bundle(timeout_ms=200)
        self.assertEqual(first["metadata"]["source_timestamp_unix"], now - 0.1)
        self.assertEqual(second["metadata"]["source_timestamp_unix"], now)
        self.assertEqual(opened.call_count, 4)

    def test_config_rejects_remote_urls_and_invalid_values(self):
        for kwargs in ({"url": "http://192.168.58.2/snapshot.npz"}, {"fps": 0}, {"width": -1},
                       {"timeout_s": float("nan")}, {"max_age_s": 0}, {"expected_serial": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GeminiBridgeConfig(**kwargs)

    def test_config_accepts_both_loopback_snapshot_endpoints(self):
        for path in ("snapshot.npz", "snapshot_fast.npz"):
            config = GeminiBridgeConfig(url=f"http://127.0.0.1:8765/{path}")
            self.assertTrue(config.url.endswith(path))
        for url in ("http://127.0.0.1:8765/snapshot_fast.npz?cache=1",
                    "http://192.168.58.2/snapshot_fast.npz",
                    "http://127.0.0.1:8765/snapshot_fast.npz/other"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                GeminiBridgeConfig(url=url)


class FastBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import only definitions. SDK constructors are tripwires and the main
        # block is not run: no USB, HTTP server, robot connection or directory IO.
        sdk = types.ModuleType("pyorbbecsdk")
        sdk.__version__ = "offline-fixture"
        for name in ("Pipeline", "Config", "OBSensorType", "OBFormat", "AlignFilter",
                     "OBStreamType", "OBFrameAggregateOutputMode"):
            setattr(sdk, name, Mock(side_effect=AssertionError("SDK must not run in offline tests")))
        path = Path(__file__).resolve().parents[2] / "fr5_runtime/monitor.py"
        spec = importlib.util.spec_from_file_location("offline_bridge_contract", path)
        cls.bridge = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"pyorbbecsdk": sdk}), patch.object(Path, "mkdir"):
            spec.loader.exec_module(cls.bridge)

    def setUp(self):
        self.color = np.arange(1080 * 1920 * 3, dtype=np.uint8).reshape(1080, 1920, 3)
        self.depth = np.arange(1080 * 1920, dtype=np.float32).reshape(1080, 1920)
        self.meta = {
            "camera": {"serial": "CP1E54200015", "name": "Gemini 335"},
            "color_intrinsic": {"fx": 1035., "fy": 1032., "cx": 969., "cy": 534.,
                                "width": 1920, "height": 1080},
            "capture": {"time_unix": 1000., "received_monotonic": 25.,
                        "timestamp_origin": "host_frame_receive_unix",
                        "alignment": "depth_to_color", "depth_unit": "mm"},
        }
        self.bridge.snapshot.clear()
        self.bridge.state["camera"] = {"ok": False}

    def test_fast_packet_is_uncompressed_paired_and_scaled(self):
        payload = self.bridge.snapshot_packet(self.color, self.depth, self.meta, fast=True)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()))
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            np.testing.assert_array_equal(data["color_bgr"], self.color[::3, ::3])
            np.testing.assert_array_equal(data["depth_mm"], self.depth[::3, ::3])
            meta = json.loads(data["metadata"].item())
        self.assertEqual(meta["capture"], self.meta["capture"])
        self.assertEqual(meta["color_intrinsic"], {
            "fx": 345., "fy": 344., "cx": 323., "cy": 178., "width": 640, "height": 360})
        self.assertEqual(meta["color_intrinsic_original"], self.meta["color_intrinsic"])
        self.assertNotIn("color_intrinsic_original", self.meta)  # source not mutated
        self.assertEqual(self.meta["color_intrinsic"]["width"], 1920)

        camera = GeminiBridgeCamera(GeminiBridgeConfig(url="http://127.0.0.1:8765/snapshot_fast.npz"))
        bundle = camera.decode_bundle(payload, received_unix=1000.02, received_monotonic=25.02)
        np.testing.assert_array_equal(bundle["color_rgb"], self.color[::3, ::3, ::-1])
        np.testing.assert_array_equal(bundle["depth_mm"], self.depth[::3, ::3])
        self.assertEqual(bundle["metadata"]["color_intrinsic"]["fx"], 345.)
        self.assertEqual(bundle["metadata"]["color_intrinsic_original"]["fx"], 1035.)
        self.assertEqual(bundle["metadata"]["source_timestamp_unix"], 1000.)
        self.assertEqual(bundle["metadata"]["timestamp_origin"], "bridge_host_frame_receive_unix")

    def test_legacy_snapshot_retains_resolution_compression_and_metadata(self):
        payload = self.bridge.snapshot_packet(self.color, self.depth, self.meta)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertTrue(all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist()))
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            np.testing.assert_array_equal(data["color_bgr"], self.color)
            np.testing.assert_array_equal(data["depth_mm"], self.depth)
            self.assertEqual(json.loads(data["metadata"].item()), self.meta)

    def _handler(self, path):
        handler = object.__new__(self.bridge.Handler)
        handler.path, handler.wfile = path, BytesIO()
        for name in ("send_response", "send_header", "end_headers", "send_error"):
            setattr(handler, name, Mock())
        return handler

    def test_fast_endpoint_reads_current_frame_without_retimestamp_or_packet_cache(self):
        self.bridge.snapshot.update(color=self.color, depth=self.depth, metadata=self.meta)
        self.bridge.state["camera"] = {"ok": True}
        first = self._handler("/snapshot_fast.npz")
        first.do_GET()
        first.send_response.assert_called_once_with(200)
        first.send_header.assert_any_call("Cache-Control", "no-store, no-cache, max-age=0")
        first.send_header.assert_any_call("Content-Length", str(len(first.wfile.getvalue())))
        # Publish a different real frame; serialization must read it afresh.
        next_meta = dict(self.meta, capture=dict(self.meta["capture"], time_unix=1000.1))
        self.bridge.snapshot.update(color=self.color + np.uint8(1), depth=self.depth + 10,
                                    metadata=next_meta)
        second = self._handler("/snapshot_fast.npz")
        second.do_GET()
        with np.load(BytesIO(first.wfile.getvalue()), allow_pickle=False) as a, \
             np.load(BytesIO(second.wfile.getvalue()), allow_pickle=False) as b:
            self.assertEqual(json.loads(a["metadata"].item())["capture"]["time_unix"], 1000.)
            self.assertEqual(json.loads(b["metadata"].item())["capture"]["time_unix"], 1000.1)
            np.testing.assert_array_equal(b["depth_mm"], a["depth_mm"] + 10)

    def test_snapshot_endpoints_reject_failed_camera_even_if_old_frame_exists(self):
        self.bridge.snapshot.update(color=self.color, depth=self.depth, metadata=self.meta)
        for path in ("/snapshot.npz", "/snapshot_fast.npz"):
            handler = self._handler(path)
            handler.do_GET()
            handler.send_error.assert_called_once_with(503, "Waiting for camera")
            self.assertEqual(handler.wfile.getvalue(), b"")

    def test_acquisition_stamps_receipt_before_alignment_without_preview_or_sleep(self):
        class FinishOfflineLoop(BaseException):
            pass

        source_color = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        source_depth = np.arange(24, dtype=np.uint16).reshape(4, 6) + 1000
        color = Mock()
        color.get_data.return_value = source_color
        color.get_format.return_value = "RGB"
        color.get_width.return_value, color.get_height.return_value = 6, 4
        profile = color.get_stream_profile.return_value.as_video_stream_profile.return_value
        profile.get_intrinsic.return_value = types.SimpleNamespace(
            fx=100., fy=110., cx=3., cy=2., width=6, height=4)
        profile.get_distortion.return_value = types.SimpleNamespace(
            k1=0., k2=0., p1=0., p2=0., k3=0., k4=0., k5=0., k6=0., model="none")
        profile.get_fps.return_value = 8
        profile.get_format.return_value = "RGB"
        depth = Mock()
        depth.get_data.return_value = source_depth.tobytes()
        depth.get_width.return_value, depth.get_height.return_value = 6, 4
        depth.get_depth_scale.return_value = 1.
        depth.get_stream_profile.return_value.as_video_stream_profile.return_value = profile
        depth.get_stream_profile.return_value.get_extrinsic_to.return_value = types.SimpleNamespace(
            rot=np.eye(3).ravel(), transform=np.zeros(3))
        for frame in (color, depth):
            frame.get_index.return_value = 1
            frame.get_timestamp_us.return_value = 1000000
            frame.get_system_timestamp_us.return_value = 1000000000
        frames = Mock()
        frames.get_color_frame.return_value, frames.get_depth_frame.return_value = color, depth
        pipe = Mock()
        pipe.wait_for_frames.side_effect = [frames, FinishOfflineLoop()]
        info = pipe.get_device.return_value.get_device_info.return_value
        info.get_name.return_value, info.get_serial_number.return_value = "Gemini 335", "CP1E54200015"
        info.get_firmware_version.return_value = "offline-fixture"
        events = []

        def receive_time():
            events.append("receipt")
            return 1000.

        def align_frame(value):
            events.append("alignment")
            return value

        align = Mock()
        align.process.side_effect = align_frame
        align.get_config_value.return_value = 0.
        with patch.object(self.bridge, "Pipeline", return_value=pipe), \
             patch.object(self.bridge, "Config", return_value=Mock()), \
             patch.object(self.bridge, "OBSensorType", types.SimpleNamespace(COLOR_SENSOR=1, DEPTH_SENSOR=2)), \
             patch.object(self.bridge, "OBFormat", types.SimpleNamespace(RGB="RGB", MJPG="MJPG", YUYV="YUYV", BGR="BGR")), \
             patch.object(self.bridge, "OBStreamType", types.SimpleNamespace(COLOR_STREAM=1)), \
             patch.object(self.bridge, "OBFrameAggregateOutputMode", types.SimpleNamespace(FULL_FRAME_REQUIRE=1)), \
             patch.object(self.bridge, "AlignFilter", return_value=align), \
             patch.object(self.bridge.time, "time", side_effect=receive_time), \
             patch.object(self.bridge.time, "sleep", side_effect=AssertionError("Acquisition must not sleep")), \
             patch.object(self.bridge.cv2, "VideoCapture", side_effect=AssertionError("No real camera in offline tests")), \
             patch.object(self.bridge.cv2, "imencode", side_effect=AssertionError("Preview must be separate")):
            with self.assertRaises(FinishOfflineLoop):
                self.bridge.camera_loop()
        self.assertEqual(events, ["receipt", "alignment"])
        pipe.stop.assert_called_once_with()
        capture = self.bridge.snapshot["metadata"]["capture"]
        self.assertEqual(capture["time_unix"], 1000.)
        self.assertEqual(capture["timestamp_origin"], "host_frame_receive_unix")
        np.testing.assert_array_equal(self.bridge.snapshot["color"], source_color[..., ::-1])
        np.testing.assert_array_equal(self.bridge.snapshot["depth"], source_depth.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
