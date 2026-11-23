"""Object collection keeps geometry, storage and real task labels in order."""
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from . import collect_object as collection


class VisibleToolTests(unittest.TestCase):
    def check(self, target, support=None):
        geometry = {'contact_tcp': [0.] * 6, 'routes': {'contact': [[0.] * 6, [0.] * 6]}}
        mask = np.ones((12, 12), dtype=bool)
        target_cloud = np.tile(target, (110, 1))
        clouds = [target_cloud]
        if support is not None:
            clouds.append(np.tile(support, (110, 1)))
        with patch.object(collection, '_points_for_mask', side_effect=clouds), \
             patch.object(collection, 'selected_color_mask', return_value=np.zeros_like(mask)):
            return collection.visible_tool_clearances(
                np.zeros((12, 12, 3), dtype=np.uint8), np.ones((12, 12)), {}, geometry, mask,
                [1., 1.] if support is not None else None)

    def test_selected_surface_between_open_fingers_fits_the_closing_span(self):
        report = self.check([5.208, 8.072, 140.], [110., 8., 60.])
        self.assertEqual(report['visible_target_points_in_finger_yz_span'], 110)
        self.assertEqual(report['visible_points'], {'target': 110, 'support': 110})
        self.assertGreater(report['open_padded_tool_visible_point_distance_mm']['support_housing'], 5.)

    def test_tool_or_raised_support_intersection_stops_the_candidate(self):
        for target, support in [([50., 8., 140.], None),
                                ([5.208, 8.072, 140.], [0., 0., 70.])]:
            with self.subTest(target=target, support=support):
                with self.assertRaisesRegex(ValueError, 'intersects'):
                    self.check(target, support)

    def test_clear_surface_outside_finger_height_does_not_prove_a_grasp(self):
        with self.assertRaisesRegex(ValueError, 'closing span'):
            self.check([5., 40., 150.])


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'object-trial'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.source = {'completed_at_unix': 1., 'robot_after': {
            'joints': [0.] * 6, 'gripper_active': 1, 'gripper_position': 100,
            'gripper_motiondone': 1, 'gripper_fault': 0, 'gripper_fault_id': 0, 'gripper_speed': 0}}
        self.geometry = {'routes': {'contact': [[0.] * 6, [0.] * 6]},
                         'table': {'normal': [0., 0., 1.], 'offset_mm': 0.}}

        def capture(path):
            path.mkdir()
            (path / 'observation.json').write_text('{}')
            return self.source

        def prepare(directory, observation, *args, **kwargs):
            self.assertEqual(Path(observation).parent.parent, Path('/dev/shm'))
            directory.mkdir()
            (directory / 'geometry.json').write_text(json.dumps({'geometry_report': self.geometry}))
            cv2.imwrite(str(directory / 'selection-mask.png'), np.ones((12, 12), dtype=np.uint8) * 255)
            return {'status': 'prepared_no_motion'}

        targets = {
            'new_stream_recording_path': Mock(return_value=self.directory),
            '_server_ready': Mock(return_value={'storage_available': True}),
            'capture': Mock(side_effect=capture),
            'require_ready': Mock(),
            'prepare': Mock(side_effect=prepare),
            'load_scene': Mock(return_value=({}, np.zeros((12, 12, 3), dtype=np.uint8),
                                            np.ones((12, 12)), np.eye(4))),
            'visible_tool_clearances': Mock(return_value={'visible_points': {'target': 110}}),
            'table_clearances': Mock(return_value={'housing': 30.}),
            'sample_geometry': Mock(return_value=([{'q': [0.] * 6}], {'minimum_joint_margin_deg': 30.})),
            '_upload_observation': Mock(return_value='/server/observations/object-trial'),
            'WholeTaskSession': Mock(),
        }
        for name, mock in targets.items():
            self.stack.enter_context(patch.object(collection, name, mock))
            setattr(self, name, mock)
        self.stack.enter_context(patch.object(collection.system, 'robot_lease', side_effect=nullcontext))
        self.stack.enter_context(patch.object(collection.system, 'LOCAL_STATE', Path(self.temp.name) / 'state'))
        self.archive = self.stack.enter_context(patch.object(collection.system, 'archive_stream'))
        self.options = {'name': 'object-trial', 'object_description': '紫色方块', 'rgb_pixel': [1200., 550.],
                        'support_rgb_pixel': [1100., 550.], 'closing_percent': 0,
                        'grasp_offset': [5.208, 8.072, 162.]}

    def test_unavailable_server_stops_before_scene_capture_or_hardware_session(self):
        self._server_ready.side_effect = RuntimeError('SSH unavailable')
        with self.assertRaisesRegex(RuntimeError, 'SSH unavailable'):
            collection.collect(**self.options, execute=True)
        self.capture.assert_not_called()
        self.WholeTaskSession.assert_not_called()
        self.assertFalse(self.directory.exists())

    def test_prepare_only_uploads_evidence_without_a_motion_session(self):
        result = collection.collect(**self.options)
        self.assertEqual(result['status'], 'prepared_no_motion')
        self.assertFalse(result['task_success_verified'])
        self.WholeTaskSession.assert_not_called()
        self.archive.assert_not_called()
        preflight = self._upload_observation.call_args.args[2]
        self.assertEqual(preflight['status'], 'passed')
        self.assertEqual(preflight['sample_count'], 1)
        self.assertFalse((self.directory / 'path-samples.json').exists())
        self.assertFalse((self.directory / 'summary.json').exists())

    def test_bad_kinematics_preserves_failed_evidence_and_sends_no_actions(self):
        self.sample_geometry.side_effect = RuntimeError('IK 112')
        with self.assertRaisesRegex(RuntimeError, 'IK 112'):
            collection.collect(**self.options, execute=True)
        self.WholeTaskSession.assert_not_called()
        self.archive.assert_not_called()
        preflight = json.loads((self.directory / 'collection-preflight.json').read_text())
        self.assertEqual(preflight['status'], 'failed')
        self.assertEqual(preflight['motion_commands_sent'], 0)
        self.assertEqual(self._upload_observation.call_args.args[2]['status'], 'failed')

    def test_object_preparation_failure_preserves_the_actual_source_observation(self):
        self.prepare.side_effect = ValueError('Visible closing width exceeds the opening')
        with self.assertRaisesRegex(ValueError, 'closing width'):
            collection.collect(**self.options, execute=True)
        self.WholeTaskSession.assert_not_called()
        self.sample_geometry.assert_not_called()
        preflight = json.loads((self.directory / 'collection-preflight.json').read_text())
        self.assertEqual(preflight['status'], 'failed')
        self.assertEqual(preflight['source_completed_unix'], self.source['completed_at_unix'])
        observation, name, saved, rows = self._upload_observation.call_args.args
        self.assertEqual(name, 'object-trial')
        self.assertEqual(saved, preflight)
        self.assertEqual(rows, [])
        self.assertEqual(Path(observation).parent.parent, Path('/dev/shm'))

    def test_contact_feedback_cannot_be_used_as_an_empty_gripper_start(self):
        self.source['robot_after']['gripper_motiondone'] = 2
        with self.assertRaisesRegex(RuntimeError, 'empty-gripper'):
            collection.collect(**self.options, execute=True)
        self.prepare.assert_not_called()
        self.WholeTaskSession.assert_not_called()

    def session(self, failure=None):
        session = self.WholeTaskSession.return_value
        session.robot.is_connected = True
        session.writer._finished = False

        def start():
            self._upload_observation.assert_called_once()
            self.assertEqual(self._upload_observation.call_args.args[2]['status'], 'passed')

        def finish():
            (self.directory / 'stream-closed.json').write_text('{}')
            session.writer._finished = True
            return {'outcome': 'pending_review', 'task_success_verified': False}

        session.start.side_effect = start
        session.run.side_effect = failure or finish
        session.finish.side_effect = lambda outcome: (
            (self.directory / 'stream-closed.json').write_text('{}') and {'outcome': outcome})
        self.archive.return_value = {'frames': 183, 'control_verification': 'verified'}
        return session

    def test_complete_execution_archives_the_whole_record_and_waits_for_user_review(self):
        session = self.session()
        result = collection.collect(**self.options, execute=True)
        self.assertEqual(result['status'], 'pending_review')
        self.assertFalse(result['task_success_verified'])
        self.assertTrue(result['success_requires_user_review'])
        self.assertEqual(result['archive_status'], 'completed')
        self.assertTrue(session.defer_materialization)
        self.archive.assert_called_once_with('object-trial', render=True)
        session.robot.disconnect.assert_called_once()
        session.finish.assert_not_called()

    def test_interrupted_execution_stops_and_archives_without_a_success_label(self):
        session = self.session(RuntimeError('No actual gripper contact'))
        result = collection.collect(**self.options, execute=True)
        self.assertEqual(result['status'], 'interrupted')
        self.assertFalse(result['task_success_verified'])
        session.controller.stop.assert_called_once_with('No actual gripper contact')
        session.finish.assert_called_once_with('interrupted')
        self.archive.assert_called_once_with('object-trial', render=True)

    def test_archive_failure_keeps_the_closed_task_result_available_for_archive_stream(self):
        session = self.session()
        self.archive.side_effect = RuntimeError('SSH reset')
        result = collection.collect(**self.options, execute=True)
        self.assertEqual(result['status'], 'archive_pending')
        self.assertEqual(result['summary']['outcome'], 'pending_review')
        self.assertEqual(result['archive_status'], 'pending')
        self.assertFalse(result['task_success_verified'])
        self.assertEqual(result['archive_error'], 'RuntimeError: SSH reset')
        saved = json.loads(Path(result['result_record']).read_text())
        self.assertEqual(saved, result)
        self.assertFalse((self.directory / 'collection-result.json').exists())
        session.start.assert_called_once()
        session.robot.disconnect.assert_called_once()
        self.archive.assert_called_once_with('object-trial', render=True)


if __name__ == '__main__':
    unittest.main()
