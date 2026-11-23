"""Observation tests for fresh RGB-D rim evidence and the π0.5 context."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from fr5_jev.cup_tracking import CupTracker
from fr5_lerobot.training.stream_observation_history import CUP_TRACKING_PROFILE, render_task


class CupTrackingTests(unittest.TestCase):
    def setUp(self):
        self.transform = np.diag([1., -1., -1., 1.])
        self.transform[2, 3] = 850.
        self.serial = 'CP1E54200015'
        self.initial = {'camera_serial': self.serial, 'observed_monotonic': 10.,
                        'table_normal': [0., 0., 1.], 'table_offset_mm': 0., 'rim_xyz_mm': [0., 0., 110.]}
        self.tracker = CupTracker(self.initial, self.transform)

    def frame(self, stamp, *, center=None, height=110, depth_height=None, jaw_x=0., contact=1, valid=True):
        intr = {'width': 640, 'height': 360, 'fx': 344., 'fy': 344., 'cx': 320., 'cy': 180.}
        distortion = dict.fromkeys(('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'), 0.)
        camera = {'source_timestamp_monotonic': stamp, 'source_camera_serial': self.serial,
                  'capture': {'alignment': 'depth_to_color', 'depth_unit': 'mm'},
                  'color_intrinsic': intr, 'color_distortion': distortion,
                  'color_distortion_model': 'BROWN_CONRADY',
                  'aligned_depth_profile': {'intrinsic': intr, 'distortion': distortion,
                                            'distortion_model': 'BROWN_CONRADY'}}
        robot = {'received_monotonic': stamp + .003, 'flange': [jaw_x, 0., -60., 0., 0., 0.],
                 'tool': 0, 'user': 0, 'gripper_motiondone': contact}
        rgb = np.full((360, 640, 3), 190, np.uint8)
        depth = np.full((360, 640), 850., np.float32)
        if center is not None:
            radius = round(344. * 37.5 / (850. - height))
            cv2.circle(rgb, center, radius, (50, 50, 50), -1, cv2.LINE_AA)
            wall_height = height if depth_height is None else depth_height
            cv2.circle(depth, center, radius + 2, 850. - wall_height, -1)
        return self.tracker.update(rgb, depth, camera, robot, gripper_valid=valid)

    def test_lifted_rim_follows_depth_and_no_rim_is_not_a_jaw_prediction(self):
        opening = self.frame(10.2, center=(320, 180))
        self.assertEqual(opening['reason'], 'observed')
        self.assertAlmostEqual(opening['rim_xyz_mm'][2], 110., delta=1.)
        lifted = self.frame(10.4, center=(326, 180), height=153, jaw_x=4., contact=2)
        self.assertEqual(lifted['reason'], 'observed')
        self.assertGreater(lifted['rim_xyz_mm'][0], 8.)
        self.assertAlmostEqual(lifted['rim_xyz_mm'][2], 153., delta=3.)
        hidden = self.frame(10.6, jaw_x=8., contact=2)
        self.assertIsNone(hidden['rim_minus_jaw_mm'])
        self.assertIsNone(hidden['rim_xyz_mm'])
        self.assertEqual(hidden['native_contact'], 1)
        with self.assertRaisesRegex(ValueError, 'ordered'):
            self.frame(10.5)

    def test_observed_and_missing_geometry_render_differently(self):
        options = {'initial_cup_uv': [1300., 700.]}
        prompt = render_task('pick and replace', [None]*6, CUP_TRACKING_PROFILE, **options,
                             cup_tracking={'rim_minus_jaw_mm': [20., -15., 0.], 'native_contact': 0})
        self.assertIn('4,-3,0; contact: 0.', prompt)
        missing = render_task('pick and replace', [None]*6, CUP_TRACKING_PROFILE, **options,
                               cup_tracking={'rim_minus_jaw_mm': None, 'native_contact': None})
        self.assertIn('bins): -; contact: -.', missing)
        self.assertEqual(prompt.split(' History')[1], missing.split(' History')[1])

    def test_open_approach_rejects_a_drifting_rim_and_recovers_fresh_evidence(self):
        first = self.frame(10.2, center=(320, 180))
        self.assertEqual(first['reason'], 'observed')
        nearby = self.frame(10.4, center=(323, 180))
        self.assertEqual(nearby['reason'], 'observed')
        displaced = self.frame(10.6, center=(327, 180))
        self.assertIsNone(displaced['rim_xyz_mm'])
        self.assertIsNone(displaced['rim_minus_jaw_mm'])
        recovered = self.frame(10.8, center=(320, 180))
        self.assertEqual(recovered['reason'], 'observed')
        self.assertAlmostEqual(recovered['rim_xyz_mm'][0], 0., delta=2.)

    def test_contact_permits_a_real_lift_after_an_obscured_pregrasp_frame(self):
        self.frame(10.2, center=(320, 180))
        raised_before_contact = self.frame(10.4, center=(320, 180), height=150.)
        self.assertIsNone(raised_before_contact['rim_xyz_mm'])
        contact_hidden = self.frame(10.6, contact=2)
        self.assertEqual(contact_hidden['native_contact'], 1)
        observed_lift = self.frame(10.8, center=(327, 180), height=150., contact=2)
        self.assertEqual(observed_lift['reason'], 'observed')
        self.assertGreater(observed_lift['rim_xyz_mm'][2], 140.)

    def test_geometry_seed_recovers_an_oversized_hough_circle(self):
        oversized = np.array([[[317.5, 181.5, 23.4]]], dtype=np.float32)
        with patch('cv2.HoughCircles', return_value=oversized):
            observed = self.frame(10.2, center=(320, 180))
        self.assertEqual(observed['reason'], 'observed')
        self.assertAlmostEqual(observed['rim_xyz_mm'][2], 110., delta=1.)
        self.assertAlmostEqual(observed['rim_xyz_mm'][0], 0., delta=2.)
        self.assertGreaterEqual(observed['raised_depth_samples'], 11)

    def test_geometry_seed_keeps_hidden_rim_missing(self):
        with patch('cv2.HoughCircles', return_value=None):
            hidden = self.frame(10.2)
        self.assertIsNone(hidden['rim_minus_jaw_mm'])
        self.assertIsNone(hidden['rim_xyz_mm'])

    def test_geometry_seed_requires_raised_depth_evidence(self):
        with patch('cv2.HoughCircles', return_value=None):
            unsupported = self.frame(10.2, center=(320, 180), depth_height=0.)
        self.assertIsNone(unsupported['rim_minus_jaw_mm'])
        self.assertIsNone(unsupported['rim_xyz_mm'])


if __name__ == '__main__':
    unittest.main()
