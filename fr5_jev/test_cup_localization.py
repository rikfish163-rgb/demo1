"""Synthetic geometry checks only; these scenes are not robot demonstrations."""
import unittest

import cv2
import numpy as np

from .cup_localization import _ellipse, detect


class LocalizationTests(unittest.TestCase):
    def scene(self, centers=(), *, raised=True, height_mm=100.):
        color = np.full((1080,1920,3),180,np.uint8)
        depth = np.full((1080,1920),850,np.float32)
        for center in centers:
            cv2.circle(color,center,52,(60,60,60),-1)
            if raised:
                cv2.circle(depth,center,52,850-height_mm,-1)
        intr = {'fx':1034.,'fy':1034.,'cx':960.,'cy':540.,'width':1920,'height':1080}
        dist = {name:0. for name in ('k1','k2','p1','p2','k3','k4','k5','k6')}
        meta = {'camera':{'serial':'CP1E54200015'},
                'capture':{'alignment':'depth_to_color','depth_unit':'mm','time_unix':1.},
                'color_intrinsic':intr,'color_distortion':dist,'color_distortion_model':'BROWN_CONRADY',
                'aligned_depth_profile':{'intrinsic':intr,'distortion':dist,'distortion_model':'BROWN_CONRADY'}}
        calibration = np.diag([1.,-1.,-1.,1.]);calibration[2,3]=850.
        return color,depth,meta,calibration

    def locate(self, *args, **kwargs):
        color,depth,meta,calibration = self.scene(*args, **kwargs)
        return detect(color,depth,meta,calibration=calibration)

    def test_changed_positions_are_not_restricted_to_old_roi(self):
        for center in ((700,500),(1400,850)):
            with self.subTest(center=center):
                result = self.locate([center])
                self.assertEqual(result['status'], 'found')
                self.assertLess(np.linalg.norm(np.asarray(result['cup_uv'])-center), 2.)

    def test_empty_image_and_flat_printed_circle_are_not_cups(self):
        self.assertEqual(self.locate()['status'], 'not_found')
        self.assertEqual(self.locate([(700,500)], raised=False)['status'], 'not_found')

    def test_multiple_matching_objects_are_reported_without_selecting_one(self):
        result = self.locate([(700,500),(1400,850)])
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIsNone(result['cup_uv'])
        self.assertEqual(len(result['candidates']), 2)

    def test_two_thirds_of_a_fitted_rim_is_sufficient_but_shorter_arcs_are_not(self):
        center = np.array([160.,160.])
        radius = 55.
        for sectors in (15,16):
            with self.subTest(sectors=sectors):
                edges = np.zeros((320,320),np.uint8)
                gx = np.zeros_like(edges,dtype=np.float32)
                gy = np.zeros_like(gx)
                angles = np.concatenate([
                    (sector+.5+np.linspace(-.3,.3,7))*2*np.pi/24-np.pi
                    for sector in range(sectors)])
                points = np.rint(center+radius*np.column_stack([np.cos(angles),np.sin(angles)])).astype(int)
                for x,y in points:
                    edges[y,x] = 255
                    gx[y,x],gy[y,x] = (np.array([x,y])-center)*2
                trace = {}
                ellipse = _ellipse(edges,edges,gx,gy,*center,radius,diagnostics=trace)
                self.assertEqual(trace['covered_sectors'],sectors)
                if sectors == 16:
                    self.assertIsNotNone(ellipse)
                    self.assertLess(np.linalg.norm(np.asarray(ellipse['cup_uv'])-center),.5)
                else:
                    self.assertIsNone(ellipse)
                    self.assertEqual(trace['rejected_by'],'edge_coverage')

    def test_raised_round_endcap_above_cup_height_is_not_a_cup(self):
        self.assertEqual(self.locate([(700,500)],height_mm=200.)['status'],'not_found')

    def test_diagnostics_identify_the_printed_circle_depth_rejection(self):
        color,depth,meta,calibration = self.scene([(700,500)],raised=False)
        trace = {}
        result = detect(color,depth,meta,calibration=calibration,diagnostics=trace)
        self.assertEqual(result['status'],'not_found')
        self.assertTrue(trace['candidate_traces'])
        self.assertEqual(trace['candidate_traces'][0]['rejected_by'],'raised_wall_depth')

    def test_incompatible_camera_model_is_rejected(self):
        color,depth,meta,calibration = self.scene([(700,500)])
        meta['capture']['depth_unit']='m'
        with self.assertRaises(ValueError):
            detect(color,depth,meta,calibration=calibration)


if __name__=='__main__':
    unittest.main()
