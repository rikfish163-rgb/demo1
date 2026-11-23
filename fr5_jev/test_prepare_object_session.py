"""Offline preparation from a synthetic RGB-D observation."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from fr5_runtime.kinematics import transform
from . import prepare_object_session as preparer
from .rgbd_geometry import color_to_depth_pixels
from .whole_task_collection import STAGES, WholeTaskSession
from .stream_storage import StreamStorage


class PrepareObjectSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        color_intrinsic = {'fx': 1034., 'fy': 1034., 'cx': 960., 'cy': 540.,
                           'width': 1920, 'height': 1080}
        depth_intrinsic = {**color_intrinsic, 'fx': 950., 'fy': 950., 'cx': 955., 'cy': 542.}
        distortion = {name: 0. for name in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')}
        cls.metadata = {
            'camera': {'serial': 'CP1E54200015'},
            'capture': {'alignment': 'depth_to_color', 'depth_unit': 'mm'},
            'color_intrinsic': color_intrinsic, 'color_distortion': distortion,
            'color_distortion_model': 'BROWN_CONRADY',
            'aligned_depth_profile': {'intrinsic': depth_intrinsic, 'distortion': distortion,
                                      'distortion_model': 'BROWN_CONRADY'},
        }
        cls.pixel = [1150., 630.]
        cls.color = np.full((1080, 1920, 3), 180, dtype=np.uint8)
        corners = cv2.boxPoints((tuple(cls.pixel), (78., 86.), 32.))
        cv2.fillConvexPoly(cls.color, np.rint(corners).astype(np.int32), (180, 30, 140))
        depth_corners = color_to_depth_pixels(corners, cls.metadata)
        object_mask = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.fillConvexPoly(object_mask, np.rint(depth_corners).astype(np.int32), 1)

        cls.normal = np.array([.10, -.06, 1.])
        cls.normal /= np.linalg.norm(cls.normal)
        y, x = np.mgrid[:1080, :1920]
        k = depth_intrinsic
        denominator = (cls.normal[2] - cls.normal[0] * (x - k['cx']) / k['fx']
                       + cls.normal[1] * (y - k['cy']) / k['fy'])
        table_depth = cls.normal[2] * 850. / denominator
        object_depth = (cls.normal[2] * 850. - 50.) / denominator
        cls.depth = np.rint(np.where(object_mask, object_depth, table_depth)).astype(np.uint16)
        cls.calibration = np.diag([1., -1., -1., 1.])
        cls.calibration[2, 3] = 850.
        cls.start = [0., 0., 200., 90., 0., 0.]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spool = self.root / 'spool'
        self.spool.mkdir()
        self.observation = self.root / 'observation'
        self.observation.mkdir()
        self.destination = self.spool / 'episode'
        base = patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', self.spool)
        base.start()
        self.addCleanup(base.stop)
        self.guards = []
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'subprocess.Popen',
                       'xmlrpc.client.ServerProxy',
                       'lerobot_robot_fairino.control_transport.FR5RPC.call',
                       'lerobot_robot_fairino.action_control.FR5ActionController.execute',
                       'fr5_jev.whole_task_collection.WholeTaskSession.start'):
            guard = patch(target, side_effect=AssertionError('Preparation must not access devices'))
            self.guards.append(guard.start())
            self.addCleanup(guard.stop)
        self.write_observation()

    def write_observation(self, *, age=0., metadata_mismatch=False):
        now = time.time()
        metadata = deepcopy(self.metadata)
        metadata['capture']['time_unix'] = now
        np.savez_compressed(self.observation / 'scene.npz', color_bgr=self.color,
                            depth_mm=self.depth, metadata=json.dumps(metadata))
        report_metadata = deepcopy(metadata)
        if metadata_mismatch:
            report_metadata['capture']['time_unix'] -= 1.
        report = {'status': 'observation_completed', 'completed_at_unix': now - age,
                  'motion_commands_sent': 0, 'gripper_commands_sent': 0,
                  'camera_age_at_capture_s': .1,
                  'robot_before': {'tcp': self.start}, 'robot_after': {'tcp': self.start},
                  'camera': report_metadata,
                  'calibration': {'base_from_color_camera_mm': self.calibration.tolist()}}
        (self.observation / 'observation.json').write_text(json.dumps(report), encoding='utf-8')

    def prepare(self, **kwargs):
        options = {'closing_percent': 63, 'grasp_offset': [5., 8., 162.]}
        options.update(kwargs)
        return preparer.prepare(self.destination, self.observation, '紫色方块', self.pixel, **options)

    def fast_table(self):
        return (self.normal, 0., {'normal': self.normal.tolist(), 'offset_mm': 0.,
                                 'inliers': 2000, 'rms_mm': .2})

    def test_rotated_object_connects_measurement_routes_and_scope_without_rpc(self):
        with patch.object(preparer, 'fit_table', wraps=preparer.fit_table) as fit:
            result = self.prepare()
        self.assertEqual(result['status'], 'prepared_no_motion')
        self.assertNotIn('success', result)
        for guard in self.guards:
            guard.assert_not_called()
        scope = json.loads((self.destination / 'scope.json').read_text())
        evidence = json.loads((self.destination / 'geometry.json').read_text())
        geometry = evidence['geometry_report']
        measurement = geometry['measurement']
        self.assertEqual(scope['task_spec'], {'object_description': '紫色方块', 'hold_seconds': 5.,
                                              'repetitions': 1, 'finish': 'released'})
        self.assertEqual(scope['task_instruction'], '抓取桌面上的紫色方块，在空中维持5秒后放下')
        self.assertEqual(scope['phase_order'], STAGES)
        self.assertEqual(scope['retreat_mode'], 'full')
        self.assertEqual(scope['closing_percent'], 63)
        self.assertEqual(scope['selection_mask'], 'selection-mask.png')
        self.assertIn('selection-mask.png', scope['evidence'])
        self.assertEqual([a['gripper_percent'] for a in scope['actions'] if a['kind'] == 'gripper'],
                         [63, 100])
        self.assertEqual(geometry['source_observation'], str(self.observation))
        self.assertEqual(geometry['selection_method'], 'color_pixel')
        self.assertEqual(geometry['selected_rgb_pixel'], self.pixel)
        self.assertEqual(geometry['selection_mask'], scope['selection_mask'])
        saved_mask = cv2.imread(str(self.destination / scope['selection_mask']), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(saved_mask != 0, preparer.selected_color_mask(self.color, self.pixel))
        self.assertEqual(geometry['object_description'], scope['task_spec']['object_description'])
        self.assertEqual(geometry['closing_percent'], scope['closing_percent'])
        self.assertEqual(geometry['contact_tcp'], scope['contact_tcp'])
        self.assertEqual([geometry['routes']], evidence['routes'])
        self.assertEqual(evidence['routes'], scope['planned_routes'])
        self.assertEqual(geometry['routes']['return'][-1], scope['contact_tcp'])
        self.assertEqual(geometry['grasp_offset_flange_mm'], [5., 8., 162.])
        self.assertEqual(geometry['grasp_below_top_mm'], 10.)
        self.assertEqual(geometry['penetration_mm'], 10.)
        self.assertEqual(geometry['side_retraction_mm'], 0.)
        self.assertFalse(geometry['geometry_or_collision_independently_validated'])
        self.assertIn('hidden object volume and support height remain unmeasured', measurement['measurement'])
        self.assertGreater(measurement['visible_surface_points'], 100)
        self.assertGreater(measurement['visible_closing_width_mm'], 45.)
        self.assertLess(measurement['visible_closing_width_mm'], 74.)
        self.assertAlmostEqual(measurement['surface_height_p10_p90_mm'][1], 50., delta=1.)
        angle = np.deg2rad(32.)
        edge = np.array([np.cos(angle), -np.sin(angle), 0.])
        edge[2] = -(self.normal[:2] @ edge[:2]) / self.normal[2]
        edge /= np.linalg.norm(edge)
        self.assertGreater(abs(edge @ measurement['closing_axis_base']), .98)
        np.testing.assert_allclose(geometry['table']['normal'], self.normal, atol=.003)
        expected_depth_pixel = color_to_depth_pixels([self.pixel], self.metadata)[0]
        np.testing.assert_allclose(fit.call_args.args[4], expected_depth_pixel, atol=1e-6)
        self.assertGreater(np.linalg.norm(expected_depth_pixel - self.pixel), 5.)
        np.testing.assert_allclose(geometry['selected_depth_pixel_for_table_exclusion'], expected_depth_pixel)
        self.assertEqual((self.destination / 'source-scene.npz').read_bytes(),
                         (self.observation / 'scene.npz').read_bytes())
        self.assertEqual((self.destination / 'source-observation.json').read_bytes(),
                         (self.observation / 'observation.json').read_bytes())
        self.assertFalse((self.destination / 'summary.json').exists())
        self.assertFalse((self.destination / 'user-review.json').exists())
        self.assertFalse((self.destination / 'target-command-journal.json').exists())
        with patch.object(preparer, 'fit_table', return_value=self.fast_table()):
            with self.assertRaisesRegex(ValueError, 'fresh directory'):
                self.prepare()

    def test_cli_uses_observed_grasp_reference(self):
        reference_path = self.root / 'observed-episode'
        reference = {'source_episode': str(reference_path),
                     'effective_grasp_offset_flange_mm': [3., 4., 170.],
                     'basis': 'observed pick and place', 'not_independent_tcp_calibration': True}
        argv = ['prepare_object_session', '--directory', str(self.destination),
                '--observation', str(self.observation), '--object-description', '紫色方块',
                '--rgb-pixel', '1150', '630', '--closing-percent', '61',
                '--approach', 'side', '--split', 'held_out_complete_trial',
                '--grasp-below-top-mm', '5', '--side-retraction-mm', '21',
                '--grasp-reference', str(reference_path)]
        with patch.object(preparer, 'load_grasp_reference', return_value=reference) as load_reference, \
             patch.object(preparer, 'fit_table', return_value=self.fast_table()), \
             patch.object(preparer, 'side_routes', wraps=preparer.side_routes) as builder, \
             patch('sys.argv', argv), redirect_stdout(io.StringIO()) as output:
            preparer.main()
        load_reference.assert_called_once_with(reference_path)
        self.assertEqual(builder.call_args.kwargs, {'penetration_mm': 5., 'side_retraction_mm': 21.})
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'prepared_no_motion')
        self.assertNotIn('success', result)
        geometry = json.loads((self.destination / 'geometry.json').read_text())['geometry_report']
        self.assertEqual(geometry['grasp_reference'], reference)
        self.assertEqual(geometry['approach'], 'side')
        self.assertEqual(geometry['grasp_offset_flange_mm'], reference['effective_grasp_offset_flange_mm'])
        self.assertEqual(geometry['grasp_below_top_mm'], 5.)
        self.assertEqual(geometry['penetration_mm'], 5.)
        self.assertEqual(geometry['side_retraction_mm'], 21.)
        scope = json.loads((self.destination / 'scope.json').read_text())
        self.assertEqual(scope['closing_percent'], 61)
        self.assertEqual(scope['split_role'], 'held_out_complete_trial')
        self.assertEqual(geometry['contact_tcp'], scope['contact_tcp'])
        self.assertEqual([geometry['routes']], scope['planned_routes'])
        self.assertEqual(geometry['routes']['approach_contact'][-1], scope['contact_tcp'])
        self.assertEqual(geometry['routes']['return'][-1], scope['contact_tcp'])
        measurement = geometry['measurement']
        expected_center = (np.asarray(measurement['top_center_base_mm'])
                           - 5. * np.asarray(measurement['table_normal'])
                           - 21. * np.asarray(geometry['tool_axis_base']))
        np.testing.assert_allclose(geometry['grasp_center_base_mm'], expected_center, atol=1e-9)
        contact = np.asarray(scope['contact_tcp'])
        np.testing.assert_allclose(contact[:3] + transform(contact)[:3, :3]
                                   @ reference['effective_grasp_offset_flange_mm'], expected_center, atol=1e-9)

    def test_gray_object_mask_preserves_selection_and_enters_control_archive(self):
        self.color = self.color.copy()
        foreground = np.any(self.color != 180, axis=2)
        self.color[foreground] = [80, 80, 80]
        self.write_observation()
        with self.assertRaisesRegex(ValueError, 'explicit mask'):
            self.prepare()

        # The centroid is in a hole, so the exclusion pixel must still be foreground.
        foreground[628:633, 1148:1153] = False
        mask_bgr = np.zeros((*foreground.shape, 3), dtype=np.uint8)
        mask_bgr[foreground, 0] = 1
        mask_path = self.root / 'gray-mask.png'
        self.assertTrue(cv2.imwrite(str(mask_path), mask_bgr))
        argv = ['prepare_object_session', '--directory', str(self.destination),
                '--observation', str(self.observation), '--object-description', '灰色方块',
                '--mask', str(mask_path), '--closing-percent', '63',
                '--approach', 'side', '--split', 'held_out_complete_trial',
                '--grasp-offset', '5', '8', '162']
        with patch.object(preparer, 'fit_table', wraps=preparer.fit_table) as fit, \
             patch('sys.argv', argv), redirect_stdout(io.StringIO()) as output:
            preparer.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'prepared_no_motion')
        scope = json.loads((self.destination / 'scope.json').read_text())
        geometry = json.loads((self.destination / 'geometry.json').read_text())['geometry_report']
        self.assertEqual(scope['split_role'], 'held_out_complete_trial')
        self.assertEqual(scope['task_spec']['object_description'], '灰色方块')
        self.assertEqual(scope['closing_percent'], 63)
        self.assertEqual(geometry['approach'], 'side')
        self.assertEqual(geometry['selection_method'], 'explicit_mask')
        self.assertIsNone(geometry['selected_rgb_pixel'])
        self.assertEqual(geometry['selection_mask'], scope['selection_mask'])
        self.assertIn(scope['selection_mask'], scope['evidence'])
        saved_mask_path = self.destination / scope['selection_mask']
        saved_mask = cv2.imread(str(saved_mask_path), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(saved_mask != 0, foreground)
        self.assertLess(saved_mask_path.stat().st_size, 1024 * 1024)
        pixel = geometry['table_exclusion_rgb_pixel']
        self.assertTrue(foreground[int(pixel[1]), int(pixel[0])])
        v, u = np.nonzero(foreground)
        distance = (u - u.mean()) ** 2 + (v - v.mean()) ** 2
        self.assertAlmostEqual((pixel[0] - u.mean()) ** 2 + (pixel[1] - v.mean()) ** 2, distance.min())
        expected = color_to_depth_pixels([pixel], self.metadata)[0]
        np.testing.assert_allclose(fit.call_args.args[4], expected)
        np.testing.assert_allclose(geometry['selected_depth_pixel_for_table_exclusion'], expected)
        self.assertEqual(geometry['measurement']['selected_rgb_pixels'], int(foreground.sum()))
        self.assertGreater(geometry['measurement']['visible_surface_points'], 100)
        self.assertIn('hidden object volume', geometry['measurement']['measurement'])
        self.assertFalse(geometry['geometry_or_collision_independently_validated'])

        spool = Mock()
        spool.start.return_value = spool
        spool.outbox = self.destination / 'frame-outbox'
        spool.session = 'offline-test'
        uploaded = []

        def record_push(_storage, payloads):
            uploaded.append(dict(payloads))
            return {}

        with patch('fr5_jev.stream_storage.AsyncFrameSpool', return_value=spool), \
             patch.object(StreamStorage, '_push', autospec=True, side_effect=record_push), \
             patch.object(StreamStorage, 'ensure_local_capacity'):
            session = WholeTaskSession(self.destination, robot=Mock(), rpc=Mock())
            session.prepare_storage()
        self.assertEqual(uploaded[0]['selection-mask.png'], saved_mask_path.read_bytes())
        self.assertIn(saved_mask_path, session.storage._control_files())
        for guard in self.guards:
            guard.assert_not_called()

    def test_empty_wrong_grid_or_ambiguous_selection_is_rejected(self):
        options = {'closing_percent': 63, 'grasp_offset': [5., 8., 162.]}
        for image, message in ((np.zeros((1080, 1920), dtype=np.uint8), 'no foreground'),
                               (np.ones((100, 200), dtype=np.uint8), 'raw RGB pixel grid')):
            with self.subTest(shape=image.shape):
                path = self.root / 'invalid-mask.png'
                self.assertTrue(cv2.imwrite(str(path), image))
                with self.assertRaisesRegex(ValueError, message):
                    preparer.prepare(self.destination, self.observation, '灰色方块',
                                     mask_path=path, **options)
                self.assertFalse(self.destination.exists())
        with self.assertRaisesRegex(ValueError, 'exactly one RGB pixel or PNG mask'):
            preparer.prepare(self.destination, self.observation, '灰色方块', **options)
        with self.assertRaisesRegex(ValueError, 'exactly one RGB pixel or PNG mask'):
            preparer.prepare(self.destination, self.observation, '灰色方块', self.pixel,
                             mask_path=self.root / 'invalid-mask.png', **options)
        argv = ['prepare_object_session', '--directory', str(self.destination),
                '--observation', str(self.observation), '--object-description', '灰色方块',
                '--rgb-pixel', '1150', '630', '--mask', str(self.root / 'invalid-mask.png'),
                '--closing-percent', '63', '--grasp-offset', '5', '8', '162']
        with patch('sys.argv', argv), redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as raised:
                preparer.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('not allowed with argument', error.getvalue())
        for guard in self.guards:
            guard.assert_not_called()

    def test_stale_and_inconsistent_source_are_rejected(self):
        with patch.object(preparer, 'fit_table', return_value=self.fast_table()):
            for options, message in (({'age': 301.}, 'current observation'),
                                     ({'metadata_mismatch': True}, 'metadata and observation report')):
                with self.subTest(options=options):
                    self.write_observation(**options)
                    with self.assertRaisesRegex(ValueError, message):
                        self.prepare()
                    self.assertFalse(self.destination.exists())

    def test_side_approach_preserves_observation_and_task_without_device_calls(self):
        with patch.object(preparer, 'fit_table', return_value=self.fast_table()), \
             patch.object(preparer, 'side_routes', wraps=preparer.side_routes) as builder:
            result = self.prepare(approach='side', grasp_below_top_mm=5., side_retraction_mm=21.)
        self.assertEqual(builder.call_args.kwargs, {'penetration_mm': 5., 'side_retraction_mm': 21.})
        self.assertEqual(result['status'], 'prepared_no_motion')
        for guard in self.guards:
            guard.assert_not_called()
        scope = json.loads((self.destination / 'scope.json').read_text())
        geometry = json.loads((self.destination / 'geometry.json').read_text())['geometry_report']
        self.assertEqual(geometry['approach'], 'side')
        self.assertEqual(geometry['grasp_below_top_mm'], 5.)
        self.assertEqual(geometry['side_retraction_mm'], 21.)
        self.assertEqual(geometry['routes']['return'][-1], scope['contact_tcp'])
        frame = transform(scope['contact_tcp'])
        self.assertAlmostEqual(frame[:3, 2] @ geometry['measurement']['table_normal'], 0., places=12)
        self.assertEqual(scope['closing_percent'], 63)
        self.assertEqual(scope['task_spec']['object_description'], '紫色方块')
        self.assertFalse(geometry['geometry_or_collision_independently_validated'])

    def test_top_down_depth_and_side_retraction_validation(self):
        with patch.object(preparer, 'load_scene') as load_scene:
            with self.assertRaisesRegex(ValueError, 'only supported for a side approach'):
                self.prepare(side_retraction_mm=1.)
            argv = ['prepare_object_session', '--directory', str(self.destination),
                    '--observation', str(self.observation), '--object-description', '紫色方块',
                    '--rgb-pixel', '1150', '630', '--closing-percent', '63',
                    '--side-retraction-mm', '1', '--grasp-offset', '5', '8', '162']
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'only supported for a side approach'):
                preparer.main()
            load_scene.assert_not_called()
        self.assertFalse(self.destination.exists())

        with patch.object(preparer, 'fit_table', return_value=self.fast_table()), \
             patch.object(preparer, 'top_down_routes', wraps=preparer.top_down_routes) as builder:
            self.prepare(grasp_below_top_mm=5.)
        self.assertEqual(builder.call_args.kwargs, {'penetration_mm': 5.})
        geometry = json.loads((self.destination / 'geometry.json').read_text())['geometry_report']
        self.assertEqual(geometry['grasp_below_top_mm'], 5.)
        self.assertEqual(geometry['penetration_mm'], 5.)
        self.assertEqual(geometry['side_retraction_mm'], 0.)
        measurement = geometry['measurement']
        expected_center = (np.asarray(measurement['top_center_base_mm'])
                           - 5. * np.asarray(measurement['table_normal']))
        np.testing.assert_allclose(geometry['grasp_center_base_mm'], expected_center, atol=1e-9)

    def test_tilted_approach_keeps_measured_geometry_and_complete_task(self):
        with patch.object(preparer, 'fit_table', return_value=self.fast_table()):
            result = self.prepare(approach='tilted', tilt_from_down_deg=55.)
        self.assertEqual(result['status'], 'prepared_no_motion')
        scope = json.loads((self.destination / 'scope.json').read_text())
        geometry = json.loads((self.destination / 'geometry.json').read_text())['geometry_report']
        self.assertEqual(geometry['approach'], 'tilted')
        self.assertEqual(geometry['tilt_from_down_deg'], 55.)
        self.assertEqual(geometry['contact_tcp'], scope['contact_tcp'])
        self.assertEqual([geometry['routes']], scope['planned_routes'])
        self.assertEqual(scope['phase_order'], STAGES)
        self.assertEqual(scope['task_spec']['finish'], 'released')
        self.assertEqual(scope['task_spec']['hold_seconds'], 5.)
        rotation = transform(scope['contact_tcp'])[:3, :3]
        self.assertAlmostEqual(rotation[:, 2] @ -self.normal, np.cos(np.deg2rad(55.)), delta=1e-10)
        self.assertFalse(geometry['geometry_or_collision_independently_validated'])
        self.assertFalse((self.destination / 'summary.json').exists())
        for guard in self.guards:
            guard.assert_not_called()

    def test_tilted_cli_transmits_the_physical_angle_and_rejects_wrong_approach(self):
        argv = ['prepare_object_session', '--directory', str(self.destination),
                '--observation', str(self.observation), '--object-description', '紫色方块',
                '--rgb-pixel', '1150', '630', '--closing-percent', '0',
                '--grasp-offset', '5.208', '8.072', '162', '--approach', 'tilted',
                 '--tilt-from-down-deg', '60', '--tilted-closing-edge', 'short']
        with patch('sys.argv', argv), patch.object(preparer, 'prepare', return_value={}) as prepare:
            with redirect_stdout(io.StringIO()):
                preparer.main()
        self.assertEqual(prepare.call_args.kwargs['approach'], 'tilted')
        self.assertEqual(prepare.call_args.kwargs['tilt_from_down_deg'], 60.)
        self.assertEqual(prepare.call_args.kwargs['tilted_closing_edge'], 'short')
        with self.assertRaisesRegex(ValueError, 'Tilt angle is only supported'):
            self.prepare(approach='side', tilt_from_down_deg=60.)
        with self.assertRaisesRegex(ValueError, 'only supported for a tilted approach'):
            self.prepare(approach='side', tilted_closing_edge='short')

    def test_requires_real_offset_and_keeps_opening_and_closing_limits(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.prepare(grasp_offset=None)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.prepare(grasp_reference=self.root, grasp_offset=[0., 0., 162.])
        for closing in (63.5, 63., True, -1, 100):
            with self.subTest(closing=closing), self.assertRaisesRegex(ValueError, 'integer physical'):
                self.prepare(closing_percent=closing)
        with self.assertRaisesRegex(ValueError, 'simple filename'):
            preparer.prepare(self.spool / '方块', self.observation, '紫色方块', self.pixel,
                             closing_percent=63, grasp_offset=[5., 8., 162.])
        measurement = {'table_normal': self.normal.tolist(), 'table_offset_mm': 0.,
                       'top_center_base_mm': [0., 0., 50.], 'visible_closing_width_mm': 75.,
                       'closing_axis_base': [1., 0., 0.]}
        with patch.object(preparer, 'fit_table', return_value=self.fast_table()), \
             patch.object(preparer, 'measure_mask', return_value=measurement):
            with self.assertRaisesRegex(ValueError, 'available opening'):
                self.prepare()
            measurement['visible_closing_width_mm'] = 40.
            with self.assertRaisesRegex(ValueError, 'tool-sized grasp offset'):
                self.prepare(grasp_offset=[0., 0., 500.])
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
