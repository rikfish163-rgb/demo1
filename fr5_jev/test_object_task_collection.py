"""Offline checks for source-backed object-task preparation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from .whole_task_collection import (DEFAULT_OBJECT, STAGES, TASK, WholeTaskSession,
                                    PHASE_DECISION, prepare_from_routes, task_instruction, validate_scope)


class ObjectTaskCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.spool = cls.root / 'spool'
        cls.spool.mkdir()
        cls.observation = cls.root / 'observation'
        cls.observation.mkdir()
        cls.color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cls.depth = np.full((1080, 1920), 800, dtype=np.uint16)
        cv2.imwrite(str(cls.observation/'color.jpg'), cls.color)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('Offline test cannot contact a device or server'))
            guard.start()
            self.addCleanup(guard.stop)
        self.base = patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', self.spool)
        self.base.start()
        self.addCleanup(self.base.stop)
        self.case = tempfile.TemporaryDirectory(dir=self.spool)
        self.addCleanup(self.case.cleanup)
        # The recorder requires an uncreated directory directly under the spool.
        self.destination = self.spool / (Path(self.case.name).name + '-episode')
        self.addCleanup(lambda: __import__('shutil').rmtree(self.destination, ignore_errors=True))
        self.start = [0., 0., 200., 90., 0., 0.]
        self.contact = [100., 0., 200., 90., 0., 0.]
        now = time.time()
        metadata = {'camera':{'serial':'CP1E54200015'}, 'capture':{'time_unix':now}}
        cv2.imwrite(str(self.observation/'color.jpg'), self.color)
        np.savez_compressed(self.observation/'scene.npz', color_bgr=self.color, depth_mm=self.depth,
                            metadata=json.dumps(metadata))
        report = {'status':'observation_completed', 'completed_at_unix':now,
                  'motion_commands_sent':0, 'gripper_commands_sent':0,
                  'camera_age_at_capture_s':.1,
                  'robot_before':{'tcp':self.start}, 'robot_after':{'tcp':self.start},
                  'camera':metadata,
                  'calibration':{'base_from_color_camera_mm':np.eye(4).tolist()}}
        (self.observation/'observation.json').write_text(json.dumps(report), encoding='utf-8')
        self.geometry = {'source':'actual RGB-D and measured tool',
                         'swept_space':{'checked':False, 'note':'requires current inspection'}}

    def routes(self):
        start = self.start
        transfer = [0., 0., 240., 90., 0., 0.]
        clear = [100., 0., 240., 90., 0., 0.]
        near = [100., 0., 220., 90., 0., 0.]
        contact = self.contact
        probe = [100., 0., 210., 90., 0., 0.]
        lift = [100., 0., 250., 90., 0., 0.]
        retreat = [90., 0., 200., 90., 0., 0.]
        retreat_clear = [80., 0., 200., 90., 0., 0.]
        return {'approach_clear':[start, transfer, clear], 'approach_near':[clear, near],
                'approach_contact':[near, contact], 'close':[contact, contact],
                'lift_probe':[contact, probe], 'lift_full':[probe, lift],
                'return':[lift, probe, contact], 'release':[contact, contact],
                'retreat_probe':[contact, retreat], 'retreat_clear':[retreat, retreat_clear],
                'retreat_high':[retreat_clear, start]}

    def prepare(self, **kwargs):
        options = {'closing_percent':63, 'hold_seconds':3., 'repetitions':2}
        options.update(kwargs)
        routes = options.pop('routes', self.routes())
        return prepare_from_routes(self.destination, self.observation, '紫色方块', routes,
                                   self.contact, self.geometry, **options)

    def test_cup_default_instruction_is_unchanged(self):
        self.assertEqual(DEFAULT_OBJECT, '透明杯子')
        self.assertEqual(task_instruction(5, 1), TASK)
        self.assertEqual(task_instruction(3, 2),
            '抓取桌面上的透明杯子，在空中维持3秒后放下并松开夹爪，再次抓起，在空中维持3秒后放下并松开夹爪')
        self.assertEqual(task_instruction(3, 1, '紫色方块'), '抓取桌面上的紫色方块，在空中维持3秒后放下')

    def test_two_cycles_preserve_real_source_plan_and_return_waypoints(self):
        with patch('fr5_jev.whole_task_collection.WholeTaskSession.start') as start, \
             patch('lerobot_robot_fairino.action_control.FR5ActionController.execute') as execute:
            result = self.prepare()
        start.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(result['status'], 'prepared_no_motion')
        self.assertFalse(result['success'])
        scope = json.loads((self.destination/'scope.json').read_text())
        geometry = json.loads((self.destination/'geometry.json').read_text())
        self.assertEqual(scope['task_spec'], {'object_description':'紫色方块', 'hold_seconds':3.,
                                             'repetitions':2, 'finish':'released'})
        self.assertEqual(scope['task_instruction'], task_instruction(3, 2, '紫色方块'))
        self.assertEqual(scope['phase_order'], STAGES * 2)
        self.assertEqual(geometry['geometry_report'], self.geometry)
        self.assertEqual(geometry['source_observation'], str(self.observation))
        self.assertEqual(geometry['routes'], scope['planned_routes'])
        self.assertEqual((self.destination/'source-observation.json').read_bytes(),
                         (self.observation/'observation.json').read_bytes())
        self.assertEqual((self.destination/'source-scene.npz').read_bytes(),
                         (self.observation/'scene.npz').read_bytes())
        self.assertEqual((self.destination/'scene.jpg').read_bytes(), (self.observation/'color.jpg').read_bytes())
        for cycle in (1, 2):
            actions = [action for action in scope['actions'] if action['cycle'] == cycle]
            self.assertEqual([a['phase'] for a in actions if a['kind'] == 'gripper'], ['close', 'release'])
            self.assertEqual([a['gripper_percent'] for a in actions if a['kind'] == 'gripper'], [63, 100])
            self.assertEqual([a['tcp_mm_deg'] for a in actions if a['phase'] == 'return'],
                             self.routes()['return'][1:])
        self.assertFalse((self.destination/'summary.json').exists())
        self.assertFalse((self.destination/'user-review.json').exists())
        self.assertFalse((self.destination/'target-command-journal.json').exists())
        session = WholeTaskSession(self.destination, robot=Mock(), rpc=Mock())
        self.assertEqual(session.task_description, scope['task_instruction'])
        validate_scope(scope)

    def test_selection_mask_is_stored_as_scope_evidence(self):
        for mask in (np.zeros((1080, 1920), dtype=np.uint8),
                     np.ones((100, 200), dtype=np.uint8)):
            with self.subTest(shape=mask.shape), self.assertRaisesRegex(ValueError, 'Selection mask'):
                self.prepare(selection_mask=mask, repetitions=1)
            self.assertFalse(self.destination.exists())
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mask[400:430, 800:840] = 1
        self.prepare(selection_mask=mask, repetitions=1)
        scope = json.loads((self.destination/'scope.json').read_text())
        self.assertEqual(scope['selection_mask'], 'selection-mask.png')
        self.assertIn(scope['selection_mask'], scope['evidence'])
        saved = cv2.imread(str(self.destination/scope['selection_mask']), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(saved != 0, mask != 0)

    def test_short_retreat_requires_and_accepts_a_matching_next_cycle_start(self):
        routes = self.routes()
        with self.assertRaisesRegex(ValueError, 'join'):
            self.prepare(routes=routes, retreat_mode='short')
        second = deepcopy(routes)
        second['approach_clear'][0] = routes['retreat_clear'][-1]
        self.prepare(routes=[routes, second], retreat_mode='short')
        scope = json.loads((self.destination/'scope.json').read_text())
        self.assertEqual(scope['phase_order'], STAGES[:-1] * 2)
        self.assertEqual(scope['planned_routes'][1]['approach_clear'][0], routes['retreat_clear'][-1])
        self.assertFalse(any(a['phase'] == 'retreat_high' for a in scope['actions']))

    def test_wrong_description_and_inconsistent_scope_actions_are_rejected(self):
        for desc in ('', ' ', '紫色方块\n放下', None, 42):
            with self.subTest(desc=desc), self.assertRaises(ValueError):
                task_instruction(3, 1, desc)
        self.prepare()
        scope = json.loads((self.destination/'scope.json').read_text())
        bad = deepcopy(scope)
        bad['task_spec']['object_description'] = '蓝色方块'
        with self.assertRaises(ValueError):
            validate_scope(bad)
        bad = deepcopy(scope)
        bad['task_instruction'] = task_instruction(5, 2, '紫色方块')
        with self.assertRaises(ValueError):
            validate_scope(bad)
        bad = deepcopy(scope)
        next(a for a in bad['actions'] if a['phase'] == 'close')['gripper_percent'] = 76
        with self.assertRaises(ValueError):
            validate_scope(bad)
        bad = deepcopy(scope)
        next(a for a in bad['actions'] if a['phase'] == 'return')['tcp_mm_deg'][0] += 1
        bad['manual_action_plan']['actions'] = deepcopy(bad['actions'])
        with self.assertRaises(ValueError):
            validate_scope(bad)
        bad = deepcopy(scope)
        bad['manual_action_plan']['actions'][0]['tcp_mm_deg'][0] += 1
        with self.assertRaises(ValueError):
            validate_scope(bad)

    def test_bad_route_or_reused_destination_is_rejected_without_motion(self):
        bad = self.routes()
        bad['return'][-1] = [100., 0., 202., 90., 0., 0.]
        bad['release'] = [bad['return'][-1], bad['return'][-1]]
        bad['retreat_probe'][0] = bad['return'][-1]
        with self.assertRaises(ValueError):
            self.prepare(routes=bad)
        bad = self.routes()
        bad['approach_near'][0] = self.start
        with self.assertRaisesRegex(ValueError, 'join'):
            self.prepare(routes=bad)
        with self.assertRaises(ValueError):
            self.prepare(closing_percent=63.5)
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'fresh directory'):
            self.prepare()

    def test_geometry_report_cannot_name_a_different_scene_or_plan(self):
        original = deepcopy(self.geometry)
        for field, value in (('source_observation', str(self.root/'other-observation')),
                             ('object_description', '透明杯子'),
                             ('closing_percent', 76),
                             ('contact_tcp', [0., 0., 200., 90., 0., 0.]),
                             ('routes', {**self.routes(), 'return': self.routes()['lift_full']})):
            self.geometry = {**original, field:value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Geometry report'):
                self.prepare()
            self.assertFalse(self.destination.exists())
        self.geometry = {**original, 'source_observation':str(self.observation),
                         'contact_tcp':self.contact, 'routes':self.routes()}
        self.prepare()

    def test_measured_top_down_routes_can_prepare_two_cycles_without_validation_claim(self):
        from .object_geometry import top_down_routes

        measurement = {'table_normal':[0., 0., 1.], 'table_offset_mm':0.,
                       'top_center_base_mm':[-200., -600., 45.],
                       'visible_closing_width_mm':40., 'closing_axis_base':[1., 0., 0.]}
        first, first_report = top_down_routes(self.start, measurement, [5., 8., 162.])
        second, second_report = top_down_routes(first['retreat_high'][-1], measurement, [5., 8., 162.])
        geometry = {**first_report, 'measurement':measurement, 'second_route_report':second_report}
        result = prepare_from_routes(self.destination, self.observation, '紫色方块', [first, second],
                                     first_report['contact_tcp'], geometry,
                                     closing_percent=63, hold_seconds=3, repetitions=2)
        self.assertEqual(result['status'], 'prepared_no_motion')
        scope = json.loads((self.destination/'scope.json').read_text())
        evidence = json.loads((self.destination/'geometry.json').read_text())
        self.assertEqual(scope['phase_order'], STAGES * 2)
        self.assertEqual(scope['planned_routes'][1]['approach_clear'][0], first['retreat_high'][-1])
        self.assertFalse(evidence['geometry_report']['geometry_or_collision_independently_validated'])
        self.assertFalse((self.destination/'target-command-journal.json').exists())

    def test_stale_observation_is_rejected_and_cannot_start(self):
        path = self.observation/'observation.json'
        report = json.loads(path.read_text())
        report['completed_at_unix'] = time.time() - 301
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'current observation'):
            self.prepare()
        report['completed_at_unix'] = time.time()
        path.write_text(json.dumps(report))
        self.prepare()
        session = WholeTaskSession(self.destination, robot=Mock(), rpc=Mock())
        session.scope['source_observation_completed_at_unix'] = time.time() - 301
        session.prepare_storage = Mock()
        with self.assertRaisesRegex(ValueError, 'current observation'):
            session.start()
        session.prepare_storage.assert_not_called()
        session.robot.connect.assert_not_called()

    def test_observation_metadata_must_match_rgbd_capture(self):
        path = self.observation/'observation.json'
        report = json.loads(path.read_text())
        report['camera']['capture']['time_unix'] -= 1
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'metadata and observation report'):
            self.prepare()
        self.assertFalse(self.destination.exists())

    def test_scene_image_comes_from_the_source_rgbd_not_an_unrelated_jpeg(self):
        unrelated = np.full_like(self.color, 255)
        cv2.imwrite(str(self.observation/'color.jpg'), unrelated)
        self.prepare(repetitions=1)
        recorded = cv2.imread(str(self.destination/'scene.jpg'))
        self.assertTrue(np.array_equal(recorded, self.color))

    def test_start_rejects_a_different_live_pose_before_any_dispatch(self):
        self.prepare()
        session = WholeTaskSession(self.destination, robot=Mock(), rpc=Mock())
        session.prepare_storage = Mock()
        session.state = Mock(return_value={'tcp':[1., 0., 200., 90., 0., 0.]})
        session.dispatch = Mock()
        session.begin_recording = Mock()
        with patch('fr5_jev.whole_task_collection.TeachingController'):
            with self.assertRaisesRegex(RuntimeError, 'Start differs'):
                session.start()
        session.dispatch.assert_not_called()
        session.begin_recording.assert_not_called()

    def test_official_decision_uses_object_task_and_only_object_cases(self):
        self.prepare()
        session = WholeTaskSession(self.destination, robot=Mock(), rpc=Mock())
        session.scope_sha, session.cycle = 'offline-scope', 1
        session.writer = Mock(task=session.scope['task_instruction'])
        session.robot.last_sample = {'camera':{'metadata':{
            'source_camera_serial':'offline', 'source_timestamp_unix':time.time()}}}
        session.state = Mock(return_value={'gripper_position':100., 'gripper_motiondone':1})
        def reply(request, *_args, **_kwargs):
            choices = request['questions'][0]['choices']
            probabilities = [{'value': item['value'], 'probability':float(item['value'] == 'transfer_empty_tool')}
                             for item in choices]
            return {'model':'gpt-6-luna', 'answers':[{'type':'choice', 'name':'next_skill',
                    'choice':'transfer_empty_tool', 'confidence':1., 'probabilities':probabilities}]}, 1.
        with patch('fr5_jev.decision.call_jev', side_effect=reply) as decision, \
             patch('fr5_jev.memory.retrieve', return_value=[]) as retrieve:
            session.fetch_phase_decision('approach_clear', 1)
        retrieve.assert_called_once_with(['紫色方块'])
        request = decision.call_args.args[0]
        state = request['input'][0]['content'][0]['data']
        self.assertEqual(state['facts']['task'], task_instruction(3, 2, '紫色方块'))
        question = request['questions'][0]
        self.assertNotIn('cup', question['instructions'].lower())
        self.assertTrue(all('cup' not in item['description'].lower() for item in question['choices']))
        self.assertEqual(state['facts']['current_scope']['task_spec']['object_description'], '紫色方块')
        self.assertEqual(session.scope['phase_decision'], PHASE_DECISION)
        self.assertEqual(session.policy_name, 'current_codex_chat+gpt-6-luna')
        self.assertEqual(decision.call_args.kwargs['provider'], 'openai')
        record = json.loads((session.root/'jev-c01-approach_clear.json').read_text())
        self.assertEqual(record['selected_by'], 'openai_decisions')
        self.assertEqual(record['response']['model'], 'gpt-6-luna')
        self.assertEqual(record['endpoint'], 'https://api.openai.com/v1/decisions')

    def test_official_refusal_stops_before_phase_dispatch(self):
        self.prepare()
        session = WholeTaskSession(self.destination, robot=Mock(), rpc=Mock())
        session.scope_sha, session.cycle = 'offline-scope', 1
        session.robot.last_sample = {'camera':{'metadata':{
            'source_camera_serial':'offline', 'source_timestamp_unix':time.time()}}}
        session.state = Mock(return_value={'gripper_position':100., 'gripper_motiondone':1})
        session.dispatch = Mock()
        reply = {'model':'gpt-6-luna', 'answers':[{'type':'refusal', 'name':'next_skill'}]}
        with patch('fr5_jev.decision.call_jev', return_value=(reply, 1.)), \
             patch('fr5_jev.memory.retrieve', return_value=[]), \
             self.assertRaisesRegex(RuntimeError, 'requests LLM intervention'):
            session.fetch_phase_decision('approach_clear', 1)
        session.dispatch.assert_not_called()
        record = json.loads((session.root/'jev-c01-approach_clear.json').read_text())
        self.assertFalse(record['automatic_continuation'])
        self.assertNotIn('selected_by', record)


if __name__ == '__main__':
    unittest.main()
