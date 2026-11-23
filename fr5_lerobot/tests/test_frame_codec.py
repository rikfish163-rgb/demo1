"""Pure media tests; no camera, robot, filesystem recording or network."""
import copy
import hashlib
import unittest
import zlib

import numpy as np
from lerobot_robot_fairino.frame_codec import encode_media, decode_media


class FrameCodecTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        self.rgb[:, :, :] = [20, 100, 200]
        self.depth = np.full((12, 16), 850, dtype=np.float32)

    def test_integer_depth_is_bit_exact_across_large_modular_deltas(self):
        self.depth[0, :5] = [0, 65535, 1, 65000, 0]
        original = self.depth.copy()
        meta, rgb, depth = encode_media(self.rgb, self.depth)
        decoded_rgb, decoded_depth = decode_media(meta, rgb, depth)
        self.assertEqual(meta['depth_codec'], 'uint16le-rowdelta-zlib')
        self.assertTrue(meta['rgb_lossy'])
        self.assertEqual(decoded_rgb.shape, self.rgb.shape)
        self.assertEqual(decoded_depth.tobytes(), original.tobytes())
        np.testing.assert_array_equal(self.depth, original)

    def test_fractional_depth_and_negative_zero_are_preserved_without_quantization(self):
        for value in (.25, -0., 80000.):
            with self.subTest(value=value):
                self.depth[0, 0] = value
                meta, rgb, depth = encode_media(self.rgb, self.depth)
                self.assertEqual(meta['depth_codec'], 'float32le-zlib')
                self.assertEqual(decode_media(meta, rgb, depth)[1].tobytes(), self.depth.tobytes())

    def test_payload_corruption_and_source_depth_hash_mismatch_are_rejected(self):
        meta, rgb, depth = encode_media(self.rgb, self.depth)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            decode_media(meta, rgb[:-1] + b'X', depth)
        changed = copy.deepcopy(meta)
        changed['depth_source_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'original float32'):
            decode_media(changed, rgb, depth)

    def test_depth_inflation_limit_and_trailing_bytes_are_enforced(self):
        meta, rgb, depth = encode_media(self.rgb, self.depth)
        for replacement in (zlib.compress(b'\0' * 20000), depth + b'trailing'):
            changed = dict(meta, depth_bytes=len(replacement),
                           depth_sha256=hashlib.sha256(replacement).hexdigest())
            with self.assertRaisesRegex(ValueError, 'boundary mismatch'):
                decode_media(changed, rgb, replacement)

    def test_false_jpeg_dimensions_are_rejected_before_decode(self):
        meta, rgb, depth = encode_media(self.rgb, self.depth)
        with self.assertRaisesRegex(ValueError, 'JPEG shape'):
            decode_media(dict(meta, width=15), rgb, depth)

    def test_invalid_arrays_or_quality_are_rejected(self):
        for value in (np.nan, np.inf, -1.):
            bad = self.depth.copy(); bad[0, 0] = value
            with self.assertRaises(ValueError):
                encode_media(self.rgb, bad)
        with self.assertRaises(ValueError):
            encode_media(self.rgb, self.depth.astype(np.float64))
        with self.assertRaises(ValueError):
            encode_media(self.rgb, self.depth, jpeg_quality=85)


if __name__ == '__main__':
    unittest.main()
