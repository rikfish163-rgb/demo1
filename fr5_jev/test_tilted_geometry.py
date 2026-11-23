"""Tilted routes retain measured grasp geometry and a complete pickup/place path."""
import unittest

import numpy as np

from fr5_runtime.kinematics import transform
from fr5_jev import object_geometry as geometry


class TiltedGeometryTests(unittest.TestCase):
    def measurement(self):
        normal = np.array([.05, -.02, 1.])
        normal /= np.linalg.norm(normal)
        edge = np.array([0., -1., 0.])
        edge -= normal * (edge @ normal)
        edge /= np.linalg.norm(edge)
        return {'table_normal': normal.tolist(), 'table_offset_mm': 0.,
                'top_center_base_mm': [-165., -919., 123.],
                'visible_closing_width_mm': 57., 'visible_planar_extents_mm': [57., 69.],
                'closing_axis_base': edge.tolist()}

    def test_tilted_grasp_reconstructs_measured_contact_and_normal_lifts(self):
        from fr5_jev.whole_task_collection import STAGES, endpoint_actions

        measurement = self.measurement()
        normal = np.asarray(measurement['table_normal'])
        start = [-153., -657., 350., 89.6, -.2, -4.2]
        offset = [5.208, 8.072, 162.]
        routes, report = geometry.tilted_routes(start, measurement, offset, tilt_from_down_deg=55.)
        contact = np.asarray(report['contact_tcp'])
        rotation = transform(contact)[:3, :3]
        np.testing.assert_allclose(rotation @ offset + contact[:3],
                                   np.asarray(measurement['top_center_base_mm']) - 10. * normal,
                                   atol=1e-10)
        self.assertAlmostEqual(rotation[:, 2] @ -normal, np.cos(np.deg2rad(55.)), delta=1e-12)
        self.assertAlmostEqual(rotation[:, 0] @ normal, 0., delta=1e-12)
        self.assertAlmostEqual(abs(rotation[:, 0] @ measurement['closing_axis_base']), 0., delta=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1., delta=1e-12)
        np.testing.assert_allclose(np.asarray(routes['lift_probe'][-1])[:3] - contact[:3], 10. * normal)
        np.testing.assert_allclose(np.asarray(routes['lift_full'][-1])[:3] - contact[:3], 50. * normal)
        np.testing.assert_allclose(np.asarray(routes['approach_near'][0])[:3] - contact[:3],
                                   -80. * rotation[:, 2])
        for phase, following in zip(STAGES, STAGES[1:]):
            np.testing.assert_allclose(routes[phase][-1], routes[following][0], atol=1e-10)
        self.assertEqual(routes['return'][-1], routes['close'][0])
        self.assertEqual(routes['release'][0], routes['return'][-1])
        self.assertEqual(routes['approach_clear'][1][3:], start[3:])
        self.assertGreaterEqual(routes['approach_clear'][1][2],
                                routes['approach_clear'][-1][2] + 20.)
        self.assertEqual({a['phase'] for a in endpoint_actions(routes)}, set(STAGES))
        self.assertFalse(report['geometry_or_collision_independently_validated'])

    def test_tilted_grasp_checks_long_edge_and_physical_angle_range(self):
        start = [-153., -657., 350., 89.6, -.2, -4.2]
        measurement = self.measurement()
        too_wide = {**measurement, 'visible_planar_extents_mm': [57., 75.]}
        with self.assertRaisesRegex(ValueError, 'available opening'):
            geometry.tilted_routes(start, too_wide, [5.208, 8.072, 162.])
        for angle in (0., 90., -10., float('nan'), float('inf')):
            with self.subTest(angle=angle):
                with self.assertRaisesRegex(ValueError, 'strictly between'):
                    geometry.tilted_routes(start, measurement, [5.208, 8.072, 162.],
                                           tilt_from_down_deg=angle)

    def test_short_edge_grasp_fits_a_long_object_and_preserves_the_measured_contact(self):
        measurement = {**self.measurement(), 'visible_planar_extents_mm': [62.5, 81.1]}
        start = [-117., -808., 198., 124.8, -.3, -20.6]
        offset = [5.208, 8.072, 162.]
        routes, report = geometry.tilted_routes(
            start, measurement, offset, closing_edge='short')
        contact = np.asarray(report['contact_tcp'])
        rotation = transform(contact)[:3, :3]
        normal = np.asarray(measurement['table_normal'])
        self.assertAlmostEqual(abs(rotation[:, 0] @ measurement['closing_axis_base']),
                               1., delta=1e-12)
        self.assertAlmostEqual(rotation[:, 2] @ -normal, np.cos(np.deg2rad(55.)), delta=1e-12)
        np.testing.assert_allclose(rotation @ offset + contact[:3],
                                   np.asarray(measurement['top_center_base_mm']) - 10. * normal,
                                   atol=1e-10)
        self.assertEqual(report['visible_closing_width_mm'], 62.5)
        self.assertEqual(report['closing_edge'], 'short')
        self.assertEqual(routes['release'][0], routes['return'][-1])
        with self.assertRaisesRegex(ValueError, 'available opening'):
            geometry.tilted_routes(start, measurement, offset)
        with self.assertRaisesRegex(ValueError, 'available opening'):
            geometry.tilted_routes(start, {**measurement, 'visible_planar_extents_mm': [75., 81.1]},
                                   offset, closing_edge='short')

    def test_tilted_grasp_preserves_high_start_and_rejects_low_contact(self):
        measurement = self.measurement()
        start = [-153., -657., 430., 89.6, -.2, -4.2]
        routes, _ = geometry.tilted_routes(start, measurement, [5.208, 8.072, 162.])
        self.assertEqual(routes['approach_clear'][1][2], 430.)
        self.assertEqual(routes['retreat_high'][-1][2], 430.)
        too_low = {**measurement, 'top_center_base_mm': [0., 0., 15.]}
        with self.assertRaisesRegex(ValueError, 'fingertip clearance'):
            geometry.tilted_routes(start, too_low, [5.208, 8.072, 162.])


if __name__ == '__main__':
    unittest.main()
