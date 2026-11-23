"""Object measurement keeps RGB-depth geometry and physical path constraints."""
import unittest
from unittest.mock import patch

import numpy as np

from fr5_runtime.kinematics import transform
from fr5_jev import object_geometry as geometry


class ObjectGeometryTests(unittest.TestCase):
    def test_selected_component_excludes_another_same_color_object(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[10:35, 10:35] = [150, 45, 100]
        image[45:70, 65:90] = [150, 45, 100]
        mask = geometry.selected_color_mask(image, [20, 20])
        self.assertTrue(mask[20, 20])
        self.assertFalse(mask[55, 75])
        with self.assertRaisesRegex(ValueError, 'explicit mask'):
            geometry.selected_color_mask(image, [40, 40])

    def test_depth_holes_do_not_become_an_object_surface(self):
        metadata = {'color_intrinsic': {'width': 40, 'height': 40}}
        mask = np.ones((40, 40), dtype=bool)
        sample = {'points_camera_mm': np.zeros((200, 3)), 'valid': np.zeros(200, dtype=bool)}
        with patch.object(geometry, 'sample_rgb_points', return_value=sample):
            with self.assertRaisesRegex(ValueError, 'consistent RGB-D'):
                geometry.measure_mask(mask, np.zeros((40, 40)), metadata, np.eye(4), [0, 0, 1], 0)

    def test_measurement_uses_visible_width_and_an_explicit_support_plane(self):
        metadata = {'color_intrinsic': {'width': 40, 'height': 40}}
        xx, yy = np.meshgrid(np.linspace(-20, 20, 20), np.linspace(-10, 10, 20))
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 45.)))
        sample = {'points_camera_mm': points, 'valid': np.ones(len(points), dtype=bool)}
        with patch.object(geometry, 'sample_rgb_points', return_value=sample):
            result = geometry.measure_mask(np.ones((40, 40)), np.ones((40, 40)), metadata,
                                           np.eye(4), [0, 0, 1], 5.)
        self.assertAlmostEqual(result['visible_closing_width_mm'], 20.)
        self.assertEqual(result['surface_height_p10_p90_mm'], [40., 40.])
        self.assertIn('hidden object volume', result['measurement'])
        with self.assertRaisesRegex(ValueError, 'raw RGB'):
            geometry.measure_mask(np.ones((20, 20)), np.ones((40, 40)), metadata,
                                  np.eye(4), [0, 0, 1], 5.)

    def test_rotated_near_square_uses_surface_edges_not_pca_diagonal(self):
        metadata = {'color_intrinsic': {'width': 40, 'height': 40}}
        normal = np.array([.2, -.1, 1.])
        normal /= np.linalg.norm(normal)
        tangent_x = np.array([1., 0., -normal[0] / normal[2]])
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(normal, tangent_x)
        xx, yy = np.meshgrid(np.linspace(-30., 30., 21), np.linspace(-31., 31., 21))
        diagonal = np.linspace(-1., 1., 101)
        x = np.r_[xx.ravel(), 30. * diagonal]
        y = np.r_[yy.ravel(), 31. * diagonal]
        for degrees in (0, 17, 43):
            with self.subTest(degrees=degrees):
                angle = np.deg2rad(degrees)
                edge_x = np.cos(angle) * tangent_x + np.sin(angle) * tangent_y
                edge_y = -np.sin(angle) * tangent_x + np.cos(angle) * tangent_y
                points = np.outer(x, edge_x) + np.outer(y, edge_y) + 120. * normal
                sample = {'points_camera_mm': points, 'valid': np.ones(len(points), dtype=bool)}
                with patch.object(geometry, 'sample_rgb_points', return_value=sample):
                    result = geometry.measure_mask(np.ones((40, 40)), np.ones((40, 40)), metadata,
                                                   np.eye(4), normal, 0.)
                closing = np.asarray(result['closing_axis_base'])
                self.assertAlmostEqual(result['visible_closing_width_mm'], 60., delta=.001)
                np.testing.assert_allclose(sorted(result['visible_planar_extents_mm']), [60., 62.], atol=.001)
                self.assertAlmostEqual(abs(closing @ edge_x), 1., delta=1e-6)
                self.assertAlmostEqual(closing @ normal, 0., delta=1e-12)
                self.assertAlmostEqual(np.ptp(points @ closing), result['visible_closing_width_mm'], delta=1e-9)
                self.assertEqual(result['visible_surface_points'], len(points))
                self.assertIn('hidden object volume', result['measurement'])
                _, report = geometry.top_down_routes([0., 0., 200., 0., 0., 0.], result, [0., 0., 162.])
                self.assertFalse(report['geometry_or_collision_independently_validated'])

    def test_visible_width_keeps_valid_boundary_points_for_opening_check(self):
        metadata = {'color_intrinsic': {'width': 40, 'height': 40}}
        xx, yy = np.meshgrid(np.linspace(-25., 25., 20), np.linspace(-25., 25., 20))
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 45.)))
        corners = np.array([[-38., -39., 45.], [-38., 39., 45.],
                            [38., -39., 45.], [38., 39., 45.]])
        points = np.vstack((points, corners))
        sample = {'points_camera_mm': points, 'valid': np.ones(len(points), dtype=bool)}
        with patch.object(geometry, 'sample_rgb_points', return_value=sample):
            result = geometry.measure_mask(np.ones((40, 40)), np.ones((40, 40)), metadata,
                                           np.eye(4), [0, 0, 1], 0.)
        self.assertEqual(result['visible_surface_points'], len(points))
        self.assertAlmostEqual(result['visible_closing_width_mm'], 76., delta=.001)
        self.assertAlmostEqual(np.ptp(points @ result['closing_axis_base']),
                               result['visible_closing_width_mm'], delta=1e-9)
        with self.assertRaisesRegex(ValueError, 'available opening'):
            geometry.top_down_routes([0., 0., 200., 0., 0., 0.], result, [0., 0., 162.])

    def test_high_target_retreat_produces_action_and_continuous_full_route(self):
        from fr5_jev.whole_task_collection import (
            CONTRACT, STAGES, endpoint_actions, task_instruction, validate_scope,
        )

        measurement = {
            'table_normal': [-.0026037097998817785, .00782392460023824, .9999660028716615],
            'table_offset_mm': -7.151203676945643,
            'top_center_base_mm': [-167.5691973917641, -919.9324849813719, 122.55142270958245],
            'visible_closing_width_mm': 56.83468412867421,
            'closing_axis_base': [-.07574041142885574, -.9970985733893385, .007604276529283998],
        }
        start = [-272.3535461425781, -754.3182983398438, 112.51351928710938,
                 92.76527404785156, -1.3113449811935425, -67.24824523925781]
        for start_z, transfer_height in ((start[2], 350.), (420., 350.), (start[2], 430.)):
            with self.subTest(start_z=start_z, transfer_height=transfer_height):
                start_pose = [*start[:2], start_z, *start[3:]]
                routes, report = geometry.top_down_routes(
                    start_pose, measurement, [0., 0., 162.], transfer_height_mm=transfer_height)
                actions = endpoint_actions(routes, preserve_return_waypoints=True)
                retreat = [a for a in actions if a['phase'] == 'retreat_high']
                self.assertEqual(len(retreat), 1)
                self.assertEqual(retreat[0]['kind'], 'arm')

                clear = np.asarray(routes['retreat_high'][0])
                transfer = np.asarray(routes['retreat_high'][-1])
                self.assertGreater(clear[2], 350.)
                self.assertAlmostEqual(transfer[2], max(start_z, clear[2] + 20., transfer_height))
                self.assertGreaterEqual(transfer[2] - clear[2], 20. - 1e-9)
                np.testing.assert_allclose(transfer[:2], clear[:2], atol=1e-10)
                np.testing.assert_allclose(retreat[0]['tcp_mm_deg'], transfer, atol=1e-10)
                np.testing.assert_allclose(routes['approach_clear'][0], start_pose, atol=1e-10)
                np.testing.assert_allclose(routes['approach_clear'][-2:], [transfer, clear], atol=1e-10)
                for phase, following in zip(STAGES, STAGES[1:]):
                    np.testing.assert_allclose(routes[phase][-1], routes[following][0], atol=1e-10)
                np.testing.assert_allclose(routes['return'][-1], report['contact_tcp'], atol=1e-10)
                np.testing.assert_allclose(routes['release'][0], report['contact_tcp'], atol=1e-10)

                scope = {
                    'contract': CONTRACT, 'whole_task': True, 'robot_ip': '192.168.58.2',
                    'task_instruction': task_instruction(5., 1, '紫色方块'),
                    'task_spec': {'hold_seconds': 5., 'repetitions': 1, 'finish': 'released',
                                  'object_description': '紫色方块'},
                    'phase_order': STAGES, 'hold_seconds': 5., 'retreat_mode': 'full',
                    'initial_tcp': start_pose, 'contact_tcp': report['contact_tcp'],
                    'terminal_tcp': transfer.tolist(), 'planned_routes': [routes],
                    'closing_percent': 76, 'actions': actions, 'manual_action_plan': {'actions': actions},
                    'move_speed': 10., 'gripper_speed': 20, 'force': 10,
                    'split_role': 'training_complete_trial', 'authorization_source': 'offline test',
                }
                validate_scope(scope)

    def test_top_down_grasp_returns_to_the_same_contact_pose(self):
        measurement = {'table_normal': [0, 0, 1], 'table_offset_mm': 0.,
                       'top_center_base_mm': [-200., -600., 45.],
                       'visible_closing_width_mm': 40., 'closing_axis_base': [1., 0., 0.]}
        start = [-250., -500., 120., 90., 0., -45.]
        routes, report = geometry.top_down_routes(start, measurement, [5., 8., 162.])
        contact = np.asarray(report['contact_tcp'])
        np.testing.assert_allclose(transform(contact)[:3, 2], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(transform(contact)[:3, :3] @ [5., 8., 162.] + contact[:3],
                                   [-200., -600., 35.], atol=1e-12)
        self.assertEqual(routes['return'][-1], routes['close'][0])
        self.assertEqual(routes['release'][0], routes['return'][-1])
        self.assertEqual(routes['approach_clear'][1][3:], start[3:])
        self.assertGreaterEqual(routes['approach_clear'][1][2], 350.)
        self.assertFalse(report['geometry_or_collision_independently_validated'])
        too_wide = {**measurement, 'visible_closing_width_mm': 77.}
        with self.assertRaisesRegex(ValueError, 'available opening'):
            geometry.top_down_routes(start, too_wide, [5., 8., 162.])
        too_low = {**measurement, 'top_center_base_mm': [-200., -600., 15.]}
        with self.assertRaisesRegex(ValueError, 'fingertip clearance'):
            geometry.top_down_routes(start, too_low, [5., 8., 162.])

    def test_full_lift_follows_the_ten_millimeter_probe(self):
        from fr5_jev.whole_task_collection import STAGES, endpoint_actions

        measurement = {'table_normal': [0, 0, 1], 'table_offset_mm': 0.,
                       'top_center_base_mm': [-200., -600., 120.],
                       'visible_closing_width_mm': 40., 'closing_axis_base': [1., 0., 0.]}
        start = [-250., -500., 120., 90., 0., -45.]
        with self.assertRaisesRegex(ValueError, 'grasp motion'):
            geometry.top_down_routes(start, measurement, [5., 8., 162.], lift_mm=10.)
        routes, _ = geometry.top_down_routes(start, measurement, [5., 8., 162.], lift_mm=11.)
        actions = endpoint_actions(routes)
        self.assertEqual({action['phase'] for action in actions}, set(STAGES))
        self.assertEqual([action['tcp_mm_deg'][2] for action in actions if action['phase'] == 'lift_full'],
                         [routes['lift_probe'][-1][2] + 1.])

    def test_side_grasp_uses_the_measured_other_edge_and_a_complete_route(self):
        from fr5_jev.whole_task_collection import STAGES, endpoint_actions

        measurement = {
            'table_normal': [-.0038970345458033497, .008166943828895, .9999590562369264],
            'table_offset_mm': -8.328707946924201,
            'top_center_base_mm': [-166.76911298280302, -919.6634070326995, 121.53504982373633],
            'visible_planar_extents_mm': [56.556217191960286, 68.80261076639937],
            'visible_closing_width_mm': 56.556217191960286,
            'closing_axis_base': [.04509230636057735, .998950927718784, -.007982976691971039],
        }
        start = [-272.22540283203125, -754.39306640625, 112.32270050048828,
                 92.76971435546875, -1.324289083480835, -67.22651672363281]
        offset = [5.208, 8.072, 162.]
        routes, report = geometry.side_routes(start, measurement, offset)
        explicit_routes, explicit_report = geometry.side_routes(start, measurement, offset,
                                                                 side_retraction_mm=0.)
        self.assertEqual(routes, explicit_routes)
        self.assertEqual(report, explicit_report)
        self.assertEqual(report['side_retraction_mm'], 0.)
        contact = np.asarray(report['contact_tcp'])
        np.testing.assert_allclose(contact, [-164.5963951077059, -757.7462806554161, 102.14835514006059,
                                             89.54260011653957, -.2441502369299542, -2.5826084824726365],
                                   atol=1e-8)
        frame = transform(contact)
        np.testing.assert_allclose(frame[:3, :3] @ offset + contact[:3],
                                   report['grasp_center_base_mm'], atol=1e-9)
        self.assertAlmostEqual(frame[:3, 2] @ measurement['table_normal'], 0., places=12)
        self.assertAlmostEqual(report['visible_closing_width_mm'], 68.80261076639937)
        np.testing.assert_allclose(np.asarray(routes['approach_contact'][0])[:3] - contact[:3],
                                   -20. * frame[:3, 2], atol=1e-9)
        np.testing.assert_allclose(np.asarray(routes['retreat_probe'][-1])[:3] - contact[:3],
                                   -10. * frame[:3, 2], atol=1e-9)
        np.testing.assert_allclose(np.asarray(routes['lift_probe'][-1])[:3] - contact[:3],
                                   10. * np.asarray(measurement['table_normal']), atol=1e-9)
        for phase, following in zip(STAGES, STAGES[1:]):
            np.testing.assert_allclose(routes[phase][-1], routes[following][0], atol=1e-10)
        self.assertEqual({action['phase'] for action in endpoint_actions(routes)}, set(STAGES))
        self.assertFalse(report['geometry_or_collision_independently_validated'])

        shallow_routes, shallow = geometry.side_routes(start, measurement, offset,
                                                        penetration_mm=5., side_retraction_mm=21.)
        normal = np.asarray(measurement['table_normal'])
        tool_axis = np.asarray(shallow['tool_axis_base'])
        shallow_contact = np.asarray(shallow['contact_tcp'])
        expected_center = np.asarray(measurement['top_center_base_mm']) - 5. * normal - 21. * tool_axis
        np.testing.assert_allclose(shallow['grasp_center_base_mm'], expected_center, atol=1e-9)
        np.testing.assert_allclose(shallow_contact[:3] + transform(shallow_contact)[:3, :3] @ offset,
                                   expected_center, atol=1e-9)
        np.testing.assert_allclose(shallow_contact[:3] - contact[:3],
                                   5. * normal - 21. * tool_axis, atol=1e-9)
        self.assertEqual(shallow['grasp_offset_flange_mm'], offset)
        self.assertEqual(shallow['penetration_mm'], 5.)
        self.assertEqual(shallow['side_retraction_mm'], 21.)

        def translated(direction, distance):
            pose = shallow_contact.copy()
            pose[:3] += distance * direction
            return pose

        clear = translated(tool_axis, -80.)
        near = translated(tool_axis, -20.)
        probe = translated(normal, 10.)
        lift = translated(normal, 50.)
        retreat = translated(tool_axis, -10.)
        raised = np.asarray(start).copy()
        raised[2] = max(start[2], clear[2] + 20., 350.)
        oriented = np.r_[raised[:3], shallow_contact[3:]]
        transfer = np.r_[clear[:2], raised[2], shallow_contact[3:]]
        expected_routes = {
            'approach_clear': [start, raised, oriented, transfer, clear],
            'approach_near': [clear, near], 'approach_contact': [near, shallow_contact],
            'close': [shallow_contact, shallow_contact], 'lift_probe': [shallow_contact, probe],
            'lift_full': [probe, lift], 'return': [lift, shallow_contact],
            'release': [shallow_contact, shallow_contact], 'retreat_probe': [shallow_contact, retreat],
            'retreat_clear': [retreat, clear], 'retreat_high': [clear, transfer],
        }
        for phase, expected in expected_routes.items():
            with self.subTest(phase=phase):
                np.testing.assert_allclose(shallow_routes[phase], expected, atol=1e-9)
        np.testing.assert_allclose(shallow_routes['approach_clear'][0], start, atol=1e-9)
        np.testing.assert_allclose(shallow_routes['approach_clear'][1][:2], start[:2], atol=1e-9)
        np.testing.assert_allclose(shallow_routes['approach_clear'][1][3:], start[3:], atol=1e-9)
        self.assertGreaterEqual(shallow_routes['approach_clear'][1][2], start[2])
        for phase, following in zip(STAGES, STAGES[1:]):
            np.testing.assert_allclose(shallow_routes[phase][-1], shallow_routes[following][0], atol=1e-9)

        for retraction in (-1., float('nan'), float('inf'), min(measurement['visible_planar_extents_mm']) / 2.):
            with self.subTest(retraction=retraction), self.assertRaisesRegex(ValueError, 'Side retraction'):
                geometry.side_routes(start, measurement, offset, side_retraction_mm=retraction)

    def test_shallow_side_grasp_matches_the_archived_preflight_contact(self):
        start = [-272.1752624511719, -754.4278564453125, 112.11268615722656,
                 92.77660369873047, -1.3308579921722412, -67.21270751953125]
        original_contact = np.array([-159.443471199867, -758.2979436979812, 103.78179816896865,
                                     89.5710810343794, -.15394922496554386, -4.541781330680777])
        normal = np.array([-.002085620221432401, .007675231621309602, .9999683700036972])
        offset = [5.208, 8.072, 162.]
        original_rotation = transform(original_contact)[:3, :3]
        measurement = {
            'table_normal': normal, 'table_offset_mm': -6.556139439820521,
            'top_center_base_mm': original_contact[:3] + original_rotation @ offset + 10. * normal,
            'closing_axis_base': original_rotation[:, 2], 'visible_planar_extents_mm': [58., 67.12],
        }
        routes, report = geometry.side_routes(start, measurement, offset,
                                               penetration_mm=5., side_retraction_mm=21.)
        expected = [-157.7906178027523, -737.3261306206002, 108.62443504313545,
                    89.5710810343794, -.15394922496554386, -4.541781330680777]
        np.testing.assert_allclose(report['contact_tcp'], expected, atol=1e-9, rtol=0.)
        np.testing.assert_allclose(routes['return'][-1], expected, atol=1e-9, rtol=0.)

    def test_side_grasp_checks_the_actual_closing_edge_width(self):
        measurement = {
            'table_normal': [0., 0., 1.], 'table_offset_mm': 0.,
            'top_center_base_mm': [-200., -600., 120.], 'closing_axis_base': [0., 1., 0.],
            'visible_closing_width_mm': 40., 'visible_planar_extents_mm': [40., 76.],
        }
        start = [-250., -500., 120., 90., 0., -45.]
        with self.assertRaisesRegex(ValueError, 'side-grasp width'):
            geometry.side_routes(start, measurement, [5., 8., 162.])
        routes, _ = geometry.top_down_routes(start, measurement, [5., 8., 162.])
        self.assertEqual(routes['return'][-1], routes['close'][0])


if __name__ == '__main__':
    unittest.main()
