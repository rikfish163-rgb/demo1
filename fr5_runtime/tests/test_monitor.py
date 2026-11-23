"""Offline packet compatibility checks; never starts camera or robot threads."""
import copy
import io
import json
import unittest

import numpy as np

from fr5_runtime.monitor import snapshot_packet


class SnapshotIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.color=np.arange(1080*1920*3,dtype=np.uint8).reshape(1080,1920,3)
        cls.depth=(np.arange(1080*1920,dtype=np.float32).reshape(1080,1920)%6000)*.125
        cls.meta={'color_intrinsic':{'fx':1400.,'fy':1402.,'cx':959.5,'cy':539.5,
                                    'width':1920,'height':1080},
                  'color_distortion':{k:0. for k in ['k1','k2','p1','p2','k3','k4','k5','k6']},
                  'capture':{'time_unix':1234567890.,'received_monotonic':42.,
                             'alignment':'depth_to_color','depth_unit':'mm'}}
        cls.meta['aligned_depth_profile']={'intrinsic':copy.deepcopy(cls.meta['color_intrinsic']),
                                           'distortion':{k:0. for k in cls.meta['color_distortion']}}
        cls.raw=np.arange(24,dtype=np.float32).reshape(4,6)*.125

    def test_full_packet_preserves_independent_grids_and_fractional_depth(self):
        with np.load(io.BytesIO(snapshot_packet(self.color,self.depth,self.meta,raw_depth=self.raw)),allow_pickle=False) as f:
            np.testing.assert_array_equal(f['color_bgr'],self.color)
            np.testing.assert_array_equal(f['depth_mm'],self.depth)
            np.testing.assert_array_equal(f['raw_depth_mm'],self.raw)
            self.assertEqual(json.loads(str(f['metadata'].item())),self.meta)

    def test_fast_packet_matches_resized_coordinates_without_mutating_native_metadata(self):
        before=copy.deepcopy(self.meta)
        with np.load(io.BytesIO(snapshot_packet(self.color,self.depth,self.meta,fast=True,raw_depth=self.raw)),allow_pickle=False) as f:
            self.assertEqual(f['color_bgr'].shape,(360,640,3))
            self.assertEqual(f['depth_mm'].shape,(360,640))
            self.assertNotIn('raw_depth_mm',f.files)
            meta=json.loads(str(f['metadata'].item()))
            self.assertEqual(meta['capture'],before['capture'])
            self.assertAlmostEqual(meta['aligned_depth_profile']['intrinsic']['fx'],before['color_intrinsic']['fx']/3)
            self.assertEqual(meta['aligned_depth_profile_original'],before['aligned_depth_profile'])
            np.testing.assert_array_equal(f['color_bgr'],self.color[::3,::3])
        self.assertEqual(self.meta,before)


if __name__=='__main__':
    unittest.main()
