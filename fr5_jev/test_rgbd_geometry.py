"""Geometric regression tests, no devices or network."""
import copy
import unittest
import cv2
import numpy as np
from fr5_jev.rgbd_geometry import color_to_depth_pixels, sample_rgb_points, color_in_depth_grid


class PixelGeometryTests(unittest.TestCase):
    def setUp(self):
        k=dict(fx=100.,fy=100.,cx=100.,cy=100.,width=201,height=201)
        d=dict(k1=.3,k2=0.,p1=0.,p2=0.,k3=0.,k4=0.,k5=0.,k6=0.)
        self.meta={'capture':{'alignment':'depth_to_color','depth_unit':'mm'},
          'color_intrinsic':k,'color_distortion':d,'color_distortion_model':'BROWN_CONRADY',
          'aligned_depth_profile':{'intrinsic':copy.deepcopy(k),'distortion':{x:0. for x in d},'distortion_model':'BROWN_CONRADY'}}
        self.rgb_pixel=np.array([[180.29,100.]])  # 100 + 100*.7*(1 + .3*.7**2).

    def test_distorted_rgb_does_not_sample_neighboring_depth_object(self):
        depth=np.full((201,201),900.,np.float32);depth[:,167:174]=500.
        result=sample_rgb_points(self.rgb_pixel,depth,self.meta)
        np.testing.assert_allclose(result['depth_pixels'],[[170.,100.]],atol=.07)
        np.testing.assert_allclose(result['points_camera_mm'],[[350.,0.,500.]],atol=.35)
        self.assertEqual(depth[100,180],900.)

    def test_edges_and_holes_stay_invalid(self):
        depth=np.full((201,201),500.,np.float32);depth[:,171:]=900.
        self.assertFalse(sample_rgb_points(self.rgb_pixel,depth,self.meta)['valid'][0])
        depth[:]=0
        self.assertFalse(sample_rgb_points(self.rgb_pixel,depth,self.meta)['valid'][0])

    def test_missing_pixel_model_is_not_silently_assumed(self):
        self.meta.pop('aligned_depth_profile')
        with self.assertRaises(ValueError):color_to_depth_pixels(self.rgb_pixel,self.meta)

    def test_display_remap_matches_analytic_distortion(self):
        color=np.zeros((201,201,3),np.uint8);color[:,:,0]=np.arange(201,dtype=np.uint8)
        rectified=color_in_depth_grid(color,self.meta)
        self.assertEqual(rectified[100,170,0],180)


if __name__=='__main__':unittest.main()
