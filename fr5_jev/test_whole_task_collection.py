"""Offline command/data regressions; synthetic fixtures are not training data."""
import base64
from copy import deepcopy
from contextlib import contextmanager, redirect_stdout
import io
import json
import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from .whole_task_collection import (TASK, CONTRACT, STAGES, WholeTaskSession,
    PHASE_DECISION, PLAN_SOURCES, RecordingRPC, TeachingController, endpoint_actions, task_instruction,
    validate_scope, open_gripper_complete, record_llm_route_plan, main as collection_main)
from .continuous_workflow import PHASE_SKILLS
from .decision import DecisionError
from .teacher_targets import TeacherSession
from lerobot_robot_fairino.pi05_rollout import DirectSettings
from lerobot_robot_fairino.frame_codec import encode_media
from lerobot_camera_gemini.timing import MONOTONIC
from lerobot_robot_fairino.whole_task_source import _holds, task_specification


class WholeTaskCollectionTests(unittest.TestCase):
    def test_whole_task_uses_bounded_buffer_for_transient_server_delays(self):
        session=WholeTaskSession.__new__(WholeTaskSession)
        session.root=Path('/synthetic/no-device')
        session.scope_sha='a'*64
        session.max_session_s=1800
        with patch('fr5_jev.stream_storage.StreamStorage.start_readonly') as start:
            session.prepare_storage()
        self.assertEqual(start.call_args.kwargs['spool_settings'],
                         {'max_frames':120,'max_bytes':32*1024**2,'max_unacked_age':30.})

    def test_actual_release_99_percent_and_native_done_is_complete(self):
        state = {'gripper_position':99,'gripper_motiondone':1,'gripper_active':1,
                 'gripper_fault':0,'gripper_fault_id':0,'gripper_speed':0}
        self.assertTrue(open_gripper_complete(state))
        self.assertFalse(open_gripper_complete({**state,'gripper_motiondone':2}))
        self.assertFalse(open_gripper_complete({**state,'gripper_position':98}))
        self.assertFalse(open_gripper_complete({**state,'gripper_active':0}))

    def setUp(self):
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'subprocess.Popen',
                       'urllib.request.OpenerDirector.open', 'fr5_jev.decision.api_key'):
            guard = patch(target, side_effect=AssertionError('Offline test cannot use devices, HTTP or credentials'))
            guard.start()
            self.addCleanup(guard.stop)

    def routes(self):
        start = [0., 0., 200., 90., 0., 0.]
        end = [100., 0., 200., 90., 0., 0.]
        return {phase: [start, end] for phase in STAGES}

    def scope(self, seconds=5., repetitions=1):
        return {'contract': CONTRACT, 'robot_ip': '192.168.58.2', 'whole_task': True,
                'task_instruction': task_instruction(seconds, repetitions),
                'task_spec': {'hold_seconds':seconds, 'repetitions':repetitions, 'finish':'released',
                              'object_description':'透明杯子'},
                'phase_order': STAGES * repetitions, 'hold_seconds': seconds,
                'move_speed': 10., 'gripper_speed': 20, 'force': 10,
                'initial_tcp': [0,0,200,90,0,0], 'contact_tcp': [100,0,200,90,0,0],
                'split_role': 'training_complete_trial', 'authorization_source': 'existing user message',
                'actions': [a for cycle in range(1, repetitions+1)
                            for a in endpoint_actions(self.routes(), cycle=cycle)]}

    def decision_session(self, directory, *, source='current_codex_chat', repetitions=1,
                         seconds=3., provider='emrank'):
        session = WholeTaskSession.__new__(WholeTaskSession)
        session.root, session.scope = Path(directory), self.scope(seconds, repetitions)
        if provider == 'openai':
            session.scope['phase_decision'] = deepcopy(PHASE_DECISION)
        raw = json.dumps(session.scope).encode()
        (session.root/'scope.json').write_bytes(raw)
        (session.root/'scene.jpg').write_bytes(b'offline scene')
        (session.root/'TEST_FIXTURE_ONLY').write_text('Synthetic offline fixture; not a physical demonstration or training data.\n')
        session.scope_sha = hashlib.sha256(raw).hexdigest()
        record_llm_route_plan(session.root, 'Reviewed finite task route', source=source)
        session.phase, session.cycle, session.decision_id = 'initial_hold', 0, 'scope.json'
        session.policy_name = 'current_codex_chat+'+('gpt-6-luna' if provider == 'openai' else 'jev')
        session.origin, session.peak_height = np.array(session.scope['contact_tcp']), 0.
        session.completed_stages, session.completed_cycles, session.journal = [], 0, []
        session.hold_durations, session.hold_records, session.held_seconds = [], [], 0.
        session._last_completed_monotonic, session._pending_hold = None, None
        session.return_endpoint, session.pending_wire = None, None
        session.closed = session.returned = session.released = False
        session.holding_position = None
        session.lock, session.interrupted = threading.RLock(), threading.Event()
        session._last_recorded_sample = None
        session._prefetch_thread, session._prefetch_result = None, None
        session._prefetch_ready, session._watchdog_stop = threading.Event(), threading.Event()
        session.failed = None
        session.robot = Mock(is_connected=False, last_sample={'camera':{'metadata':{
            'source_camera_serial':'synthetic', 'source_timestamp_unix':100.}}})
        session.capture, session.controller = Mock(spec=['phase', 'assert_healthy']), Mock()
        session.writer = Mock(_finished=False)
        session.start = Mock()
        session.finish = Mock(return_value={'outcome':'pending_review'})
        def fresh(*, after_monotonic=None, image_path=None):
            if after_monotonic is not None:
                time.sleep(.02)
            captured = time.monotonic()
            captured_unix = time.time()
            robot = session.state()
            robot.update(received_monotonic=captured, timestamp=captured_unix)
            observed = {'frame_index':session.fresh_recorded_observation.call_count-1,
                'camera_time':captured, 'clock':MONOTONIC,
                'robot_time_monotonic':captured, 'robot':robot,
                'previous_completion_monotonic':session._last_completed_monotonic,
                'camera':{'source_camera_serial':'synthetic', 'source_timestamp_unix':captured_unix,
                          'source_timestamp_monotonic':captured}}
            if image_path is not None:
                _, jpeg, _ = encode_media(np.full((3,4,3), 64, np.uint8), np.zeros((3,4), np.float32))
                Path(image_path).write_bytes(jpeg)
                observed['image_file'] = Path(image_path).name
            return observed
        session.fresh_recorded_observation = Mock(side_effect=fresh)
        state = {'joints':[0.]*6, 'tcp':list(session.scope['initial_tcp']),
                 'gripper_position':100., 'gripper_active':1, 'gripper_fault':0,
                 'gripper_fault_id':0, 'gripper_motiondone':1, 'gripper_speed':0}
        session.state = Mock(side_effect=lambda **_: deepcopy(state))
        def dispatch(action):
            issued = time.monotonic()
            if action['phase'] == 'return':
                session.complete_hold_at_return(issued, time.time())
            if action['kind'] == 'arm':
                state['tcp'] = list(action['tcp_mm_deg'])
            else:
                state.update(gripper_position=action['gripper_percent'],
                             gripper_motiondone=2 if action['phase'] == 'close' else 1)
            session.journal.append({'executed':True,
                'teacher':{'phase':action['phase'], 'cycle':action['cycle']}})
            session._last_completed_monotonic = issued
            return deepcopy(state)
        session.dispatch = Mock(side_effect=dispatch)
        return session

    def jev_reply(self, request, choice=None):
        if choice is None:
            snapshot = (request['input'][0]['content'][0]['data'] if 'input' in request else request['state'])
            choice = PHASE_SKILLS[snapshot['facts']['requested_phase']]
        if 'input' in request:
            choices = [entry['value'] for entry in request['questions'][0]['choices']]
            return {'model':'gpt-6-luna', 'id':'synthetic-not-training',
                    'answers':[{'name':'next_skill', 'type':'choice', 'choice':choice, 'confidence':.37,
                        'probabilities':[{'value':name, 'probability':.53 if name == choice else .47/(len(choices)-1)}
                                         for name in choices]}],
                    'usage':{'input_tokens':123, 'output_tokens':0}}, 1.
        choices = request['questions']['next_skill']['criteria']
        return {'answers':{'next_skill':{'type':'choice', 'choice':choice, 'confidence':1.,
            'probabilities':{name:float(name == choice) for name in choices}}}}, 1.

    def execute_decision_session(self, session, reply):
        with patch('fr5_jev.whole_task_collection.WholeTaskSession', return_value=session), \
             patch('fr5_jev.whole_task_collection.signal.signal'), \
             patch('sys.argv', ['whole_task_collection', 'run', '--directory', str(session.root), '--execute']), \
             patch('fr5_jev.decision.call_jev', side_effect=reply), \
             patch('fr5_jev.memory.retrieve', return_value=[]), redirect_stdout(io.StringIO()):
            collection_main()

    def test_noncontinuing_jev_choices_prevent_following_motion_despite_route_plan(self):
        for source in PLAN_SOURCES:
            for choice in ('hold', 'observe_scene', 'request_astra'):
                with self.subTest(source=source, choice=choice), tempfile.TemporaryDirectory() as tmp:
                    session = self.decision_session(tmp, source=source)
                    def reply(request, *_args, **_kwargs):
                        phase = request['state']['facts']['requested_phase']
                        return self.jev_reply(request, choice if phase == 'approach_near' else None)
                    with self.assertRaisesRegex(RuntimeError, 'Jev requests LLM intervention'):
                        self.execute_decision_session(session, reply)
                    self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list],
                                     ['approach_clear'])
                    session.controller.stop.assert_called_once()
                    session.finish.assert_called_once_with('interrupted')
                    record = json.loads((session.root/'jev-c01-approach_near.json').read_text())
                    self.assertEqual(record['answer']['choice'], choice)
                    self.assertEqual(record['state']['facts']['existing_llm_route_plan']['source'], source)
                    self.assertFalse(record['automatic_continuation'])
                    self.assertFalse(record['reviewed_by_current_chat'])
                    self.assertNotIn('llm_plan_fallback', record)
                    if choice == 'observe_scene':
                        self.assertEqual(record['route'], 'observe_scene')
                        refreshed = json.loads((session.root/'jev-c01-approach_near-after-observe.json').read_text())
                        self.assertEqual(refreshed['answer']['choice'], 'observe_scene')
                        self.assertEqual(refreshed['route'], 'needs_llm')
                    self.assertEqual(session.fresh_recorded_observation.call_count,
                                     3 if choice == 'observe_scene' else 2)

    def test_failed_jev_decision_does_not_execute_route_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp)
            with self.assertRaisesRegex(RuntimeError, 'Jev requests LLM intervention'):
                self.execute_decision_session(session, DecisionError('Offline decision unavailable'))
            session.dispatch.assert_not_called()
            session.controller.stop.assert_called_once()
            session.finish.assert_called_once_with('interrupted')
            record = json.loads((session.root/'jev-c01-approach_clear.json').read_text())
            self.assertEqual(record['decision_error'], 'Offline decision unavailable')
            self.assertFalse(record['automatic_continuation'])

    def test_jev_stop_during_hold_interrupts_before_return_and_hold_deadline(self):
        for choice in ('hold', 'observe_scene', 'request_astra', 'decision_error'):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp:
                session = self.decision_session(tmp)
                clock = [100.]
                class InlineThread:
                    def __init__(self, target, **kwargs): self.target = target
                    def start(self): self.target()
                def reply(request, *_args, **_kwargs):
                    if request['state']['facts']['requested_phase'] == 'return':
                        if choice == 'decision_error':
                            raise DecisionError('Offline return decision unavailable')
                        return self.jev_reply(request, choice)
                    return self.jev_reply(request)
                with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                     patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                     patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread), \
                     self.assertRaisesRegex(RuntimeError, 'Jev return decision failed'):
                    self.execute_decision_session(session, reply)
                self.assertAlmostEqual(clock[0], 100.02 if choice == 'observe_scene' else 100.)
                self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list],
                                 STAGES[:STAGES.index('return')])
                session.controller.stop.assert_called_once()
                session.finish.assert_called_once_with('interrupted')
                hold = json.loads((session.root/'hold-01.json').read_text())
                self.assertFalse(hold['return_command_issued'])
                self.assertEqual(hold['measurement'], 'interrupted_before_return')
                self.assertIsNone(session._pending_hold)
                self.assertEqual(session.hold_durations, [])
                self.assertFalse((session.root/'llm-c01-return.json').exists())
                self.assertEqual(session.completed_stages, STAGES[:STAGES.index('return')])

    def test_pending_return_waits_for_real_jev_answer_with_route_plan(self):
        for choice in ('return_place', 'hold', 'observe_scene', 'request_astra'):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp:
                session = self.decision_session(tmp)
                clock, pending = [100.], []
                class DelayedThread:
                    def __init__(self, target, **kwargs): pending.append(target)
                    def start(self): pass
                def advance(seconds):
                    clock[0] += seconds
                    if clock[0] >= 106. and pending:
                        pending.pop()()
                def reply(request, *_args, **_kwargs):
                    phase = request['state']['facts']['requested_phase']
                    return self.jev_reply(request, choice if phase == 'return' else None)
                with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                     patch('fr5_jev.whole_task_collection.time.sleep', side_effect=advance), \
                     patch('fr5_jev.whole_task_collection.threading.Thread', DelayedThread):
                    if choice == 'return_place':
                        self.execute_decision_session(session, reply)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'Jev return decision failed'):
                            self.execute_decision_session(session, reply)
                self.assertGreaterEqual(clock[0], 106.)
                hold = json.loads((session.root/'hold-01.json').read_text())
                self.assertGreaterEqual(hold['actual_duration_s'], 6.)
                self.assertEqual(hold['return_command_issued'], choice == 'return_place')
                phases = [call.args[0]['phase'] for call in session.dispatch.call_args_list]
                self.assertEqual(phases, STAGES if choice == 'return_place' else STAGES[:STAGES.index('return')])
                record = json.loads((session.root/'jev-c01-return.json').read_text())
                self.assertEqual(record['answer']['choice'], choice)
                self.assertFalse((session.root/'llm-c01-return.json').exists())

    def test_matching_jev_choices_complete_two_cycles_with_real_hold_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, repetitions=2)
            clock = [100.]
            class InlineThread:
                def __init__(self, target, **kwargs): self.target = target
                def start(self): self.target()
            with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                 patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread):
                self.execute_decision_session(session, lambda request, *_a, **_kw:self.jev_reply(request))
            self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list], STAGES * 2)
            self.assertEqual(session.completed_cycles, 2)
            self.assertEqual(session.hold_durations, [3., 3.])
            self.assertEqual(session.fresh_recorded_observation.call_count, 2*len(STAGES)+1)
            session.controller.stop.assert_not_called()
            self.assertEqual(session.finish.call_args.args[0], 'pending_review')
            for cycle in (1, 2):
                for phase in STAGES:
                    record = json.loads((session.root/f'jev-c{cycle:02d}-{phase}.json').read_text())
                    self.assertEqual(record['selected_by'], 'jev')
                    self.assertEqual(record['answer']['choice'], PHASE_SKILLS[phase])
                    self.assertTrue(record['automatic_continuation'])
                    self.assertEqual(record['state']['facts']['current_cycle_completed_stages'],
                                     STAGES[:STAGES.index(phase)])
                    self.assertEqual(record['state']['facts']['completed_cycles'], cycle-1)
                    if phase == 'return':
                        self.assertEqual(record['state']['facts']['completed_stages'], STAGES*(cycle-1)+STAGES[:6])
                        self.assertIsNone(record['state']['facts']['actual_hold_duration_s'])
                        self.assertTrue(record['state']['facts']['hold_in_progress']['return_dispatch_waits_for_requested_hold'])
                    elif phase == 'release':
                        self.assertEqual(record['state']['facts']['actual_hold_duration_s'], 3.)
                        self.assertEqual(record['state']['facts']['hold_record']['repetition'], cycle)

    def test_release_snapshot_contains_measured_completion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, seconds=5., provider='openai')
            clock = [100.]
            latest = session.state()
            original_dispatch = session.dispatch.side_effect
            actual_return = [100.2, -.1, 200.3, 90.01, -.02, 359.9]
            def state(**_):
                measured = deepcopy(latest)
                if session.phase == 'release':
                    measured['tcp'][0] += .04
                return measured
            def dispatch(action):
                if action['phase'] == 'return':
                    clock[0] += .27
                measured = original_dispatch(action)
                if action['phase'] == 'lift_full':
                    measured['tcp'][2] = 250.54
                elif action['phase'] == 'return':
                    measured['tcp'] = actual_return.copy()
                latest.update(measured)
                return measured
            session.state.side_effect, session.dispatch.side_effect = state, dispatch
            class InlineThread:
                def __init__(self, target, **kwargs): self.target = target
                def start(self): self.target()
            with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                 patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread):
                self.execute_decision_session(session, lambda request, *_a, **_kw:self.jev_reply(request))
            record = json.loads((session.root/'jev-c01-release.json').read_text())
            facts = record['state']['facts']
            self.assertEqual(facts['completed_stages'], STAGES[:7])
            self.assertEqual(facts['current_cycle_completed_stages'], STAGES[:7])
            self.assertTrue(facts['return_completed'])
            self.assertFalse(facts['release_completed'])
            self.assertEqual(facts['contact_reference'],
                             {'source':'scope.json.contact_tcp', 'basis':'planned grasp and return TCP reference',
                              'tcp_mm_deg':session.scope['contact_tcp']})
            self.assertAlmostEqual(facts['current_tcp_mm_deg'][0], 100.24)
            self.assertEqual(facts['return_target_tcp_mm_deg'], session.scope['contact_tcp'])
            endpoint = facts['actual_return_endpoint']
            self.assertEqual(endpoint['tcp_mm_deg'], actual_return)
            np.testing.assert_allclose(endpoint['position_residual_mm'], [.2,-.1,.3])
            self.assertAlmostEqual(endpoint['position_residual_norm_mm'], np.linalg.norm([.2,-.1,.3]))
            np.testing.assert_allclose(endpoint['orientation_residual_deg'], [.01,-.02,-.1])
            self.assertAlmostEqual(facts['peak_tcp_height_above_contact_reference_mm'], 50.54)
            self.assertAlmostEqual(facts['actual_hold_duration_s'], 5.27)
            hold = json.loads((session.root/'hold-01.json').read_text())
            self.assertEqual(facts['hold_record'], {'path':'hold-01.json', **hold})
            self.assertFalse(hold['object_lift_visually_verified'])
            self.assertIsNone(facts.get('object_support_visually_verified'))
            self.assertEqual(record['state'], record['request']['input'][0]['content'][0]['data'])
            self.assertEqual(record['endpoint'], PHASE_DECISION['endpoint'])
            self.assertEqual(record['requested_model'], PHASE_DECISION['model'])
            self.assertEqual(record['selected_by'], 'openai_decisions')
            self.assertTrue(record['automatic_continuation'])
            self.assertEqual(record['answer']['confidence'], .37)
            session.controller.stop.assert_not_called()

    def test_initial_decision_uses_committed_pair_after_previous_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, provider='openai')
            session.cycle, session.phase = 1, 'release'
            session.closed, session.returned, session.holding_position = True, True, 76.
            session.completed_stages = STAGES[:7]
            session._last_completed_monotonic = 2.
            live = session.state()
            live.update(tcp=[900.,900.,900.,90.,0.,0.],
                        gripper_position=76., gripper_motiondone=2)
            session.state.return_value, session.state.side_effect = live, None
            old = {'camera':{'source_camera_serial':'synthetic', 'source_timestamp_unix':20.,
                             'source_timestamp_monotonic':1.9},
                   'robot':{**live, 'tcp':[0.]*6, 'received_monotonic':1.91},
                   'camera_time':1.9, 'robot_time_monotonic':1.91, 'clock':MONOTONIC}
            new = {'camera':{'source_camera_serial':'synthetic', 'source_timestamp_unix':21.,
                             'source_timestamp_monotonic':2.2},
                   'robot':{**live, 'tcp':[-117.,-809.,199.,90.,0.,0.], 'received_monotonic':2.21},
                   'camera_time':2.2, 'robot_time_monotonic':2.21, 'clock':MONOTONIC}
            session._last_recorded_sample = old
            session.robot.last_sample = {'camera':{'metadata':deepcopy(old['camera'])}, 'robot_state':old['robot']}
            session.capture._lock = threading.RLock()
            session.capture._times, session.capture._clock_basis = [1.9], MONOTONIC
            session.fresh_recorded_observation = WholeTaskSession.fresh_recorded_observation.__get__(session)
            def advance(_):
                if session._last_recorded_sample is old:
                    session._last_recorded_sample = new
                else:
                    session.capture._times.append(2.2)
            with patch('fr5_jev.whole_task_collection.time.sleep', side_effect=advance) as pause, \
                 patch('fr5_jev.whole_task_collection.time.time', side_effect=[1000.,1001.]), \
                 patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=[10.,11.]), \
                 patch('fr5_jev.decision.call_jev', side_effect=lambda request, *_a, **_kw:self.jev_reply(request)), \
                 patch('fr5_jev.memory.retrieve', return_value=[]):
                session.fetch_phase_decision('release', 1)
            record = json.loads((session.root/'jev-c01-release.json').read_text())
            snapshot = record['state']
            observation = snapshot['facts']['scene_observation']
            self.assertEqual(pause.call_count, 2)
            self.assertEqual(snapshot['captured_at_unix'], 21.)
            self.assertEqual(snapshot['facts']['current_tcp_mm_deg'], new['robot']['tcp'])
            self.assertEqual(observation['camera'], new['camera'])
            self.assertEqual(observation['robot'], new['robot'])
            self.assertEqual(observation['frame_index'], 1)
            self.assertEqual(observation['previous_completion_monotonic'], 2.)
            self.assertGreater(observation['camera_time'], 2.)
            self.assertGreater(observation['robot_time_monotonic'], 2.)
            self.assertEqual(snapshot['facts']['current_cycle_completed_stages'], STAGES[:7])
            self.assertEqual(record['request_started_monotonic'], 10.)
            self.assertEqual(record['response_received_monotonic'], 11.)
            self.assertEqual(record['request_started_unix'], 1000.)
            self.assertEqual(record['response_received_unix'], 1001.)
            self.assertEqual(record['request']['input'],
                             [{'role':'user', 'content':[{'type':'input_json', 'data':snapshot}]}])
            self.assertEqual(old['camera']['source_timestamp_unix'], 20.)
            self.assertEqual(old['robot']['tcp'], [0.]*6)
            session.dispatch.assert_not_called()

    def test_release_decision_requires_live_and_recorded_gripper_contact(self):
        for invalid_source in ('live', 'recorded'):
            with self.subTest(invalid_source=invalid_source), tempfile.TemporaryDirectory() as tmp:
                session = self.decision_session(tmp, provider='openai')
                session.cycle, session.phase, session.returned = 1, 'release', True
                session.holding_position = 76.
                valid = session.state()
                valid.update(gripper_position=76., gripper_motiondone=2)
                invalid = {**valid, 'gripper_active':0, 'gripper_motiondone':0}
                session.controller.error = None
                session.controller.ready_state.return_value = invalid if invalid_source == 'live' else valid
                session.state = WholeTaskSession.state.__get__(session)
                session.fresh_recorded_observation.return_value = {
                    'robot':invalid if invalid_source == 'recorded' else valid}
                session.fresh_recorded_observation.side_effect = None
                with patch('fr5_jev.decision.call_jev') as decision, \
                     self.assertRaisesRegex(RuntimeError, 'gripper contact feedback|Gripper contact feedback'):
                    session.fetch_phase_decision('release', 1)
                decision.assert_not_called()
                session.dispatch.assert_not_called()
                session.controller.execute.assert_not_called()
                self.assertFalse((session.root/'jev-c01-release.json').exists())

    def test_openai_observe_refreshes_once_before_model_selected_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, provider='openai')
            clock, release_calls = [100.], []
            def reply(request, *_args, **_kwargs):
                snapshot = request['input'][0]['content'][0]['data']
                if snapshot['facts']['requested_phase'] == 'release':
                    self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list], STAGES[:7])
                    response, latency = self.jev_reply(request, 'observe_scene' if not release_calls else 'release_gripper')
                    release_calls.append((deepcopy(request), deepcopy(response)))
                    return response, latency
                return self.jev_reply(request)
            class InlineThread:
                def __init__(self, target, **kwargs): self.target = target
                def start(self): self.target()
            with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                 patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread):
                self.execute_decision_session(session, reply)
            self.assertEqual(len(release_calls), 2)
            original = json.loads((session.root/'jev-c01-release.json').read_text())
            refreshed = json.loads((session.root/'jev-c01-release-after-observe.json').read_text())
            observation = json.loads((session.root/'observe-c01-release.json').read_text())
            for record, (request, response) in zip((original, refreshed), release_calls):
                self.assertEqual(record['request'], request)
                self.assertEqual(record['response'], response)
                self.assertEqual(record['selected_by'], 'openai_decisions')
                self.assertFalse(record['reviewed_by_current_chat'])
                self.assertFalse(record['physical_execution'])
                paired = record['state']['facts']['scene_observation']
                self.assertEqual(record['state']['captured_at_unix'], paired['camera']['source_timestamp_unix'])
                self.assertEqual(record['state']['camera']['capture_time_monotonic'], paired['camera_time'])
                self.assertEqual(record['state']['facts']['current_tcp_mm_deg'], paired['robot']['tcp'])
                self.assertEqual(record['state']['robot']['received_monotonic'], paired['robot_time_monotonic'])
            self.assertEqual(original['answer']['choice'], 'observe_scene')
            self.assertFalse(original['automatic_continuation'])
            self.assertEqual(original['route'], 'observe_scene')
            self.assertEqual(original['observation_record'], observation['record'])
            self.assertEqual(original['followup_decision_id'], 'jev-c01-release-after-observe.json')
            self.assertEqual(observation['requested_by_decision'], 'jev-c01-release.json')
            self.assertEqual(refreshed['answer']['choice'], 'release_gripper')
            self.assertTrue(refreshed['automatic_continuation'])
            self.assertEqual(refreshed['state']['facts']['scene_observation'], observation)
            content = refreshed['request']['input'][0]['content']
            self.assertEqual(content[0]['data'], refreshed['state'])
            self.assertEqual(content[1]['type'], 'input_image')
            self.assertEqual(base64.b64decode(content[1]['image_url'].split(',',1)[1]),
                             (session.root/observation['image_file']).read_bytes())
            self.assertEqual(len(original['request']['input'][0]['content']), 1)
            self.assertEqual(original['request']['input'],
                             [{'role':'user', 'content':[{'type':'input_json', 'data':original['state']}]}])
            self.assertEqual(len(content), 2)
            self.assertEqual(original['request']['questions'][0]['choices'],
                             refreshed['request']['questions'][0]['choices'])
            self.assertEqual(refreshed['state']['camera']['capture_time_unix'],
                             observation['camera']['source_timestamp_unix'])
            refresh = [call for call in session.fresh_recorded_observation.call_args_list
                       if call.kwargs.get('after_monotonic') is not None]
            self.assertEqual(len(refresh), 1)
            self.assertGreater(observation['camera_time'], refresh[0].kwargs['after_monotonic'])
            self.assertGreater(observation['robot_time_monotonic'], observation['requested_monotonic'])
            self.assertGreaterEqual(observation['requested_monotonic'], original['response_received_monotonic'])
            self.assertGreaterEqual(refreshed['request_started_monotonic'], observation['camera_time'])
            self.assertEqual(observation['requested_monotonic'], refresh[0].kwargs['after_monotonic'])
            self.assertEqual(session.fresh_recorded_observation.call_count, len(STAGES)+2)
            self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list], STAGES)
            phase = next(call for call in session.capture.phase.call_args_list if call.args[0] == 'release')
            self.assertEqual(phase.kwargs['teacher']['decision_id'], 'jev-c01-release-after-observe.json')
            session.controller.stop.assert_not_called()
            session.finish.assert_called_once()
            self.assertEqual(session.finish.call_args.args[0], 'pending_review')

    def test_recorded_observe_chain_matches_dataset_entry_contract(self):
        from fr5_lerobot.training.prepare_jev_pi05_dataset import _validate_openai_phase_decision
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, provider='openai')
            session.cycle, session.phase, session.returned = 1, 'release', True
            session.holding_position = 76.
            session.completed_stages = STAGES[:7]
            session._last_completed_monotonic = 2.
            state = session.state()
            state.update(gripper_position=76., gripper_motiondone=2)
            session.state.return_value, session.state.side_effect = state, None
            samples = [
                {'camera':{'source_camera_serial':'synthetic', 'source_timestamp_unix':1000.+stamp,
                           'source_timestamp_monotonic':stamp},
                 'robot':{**deepcopy(state), 'received_monotonic':stamp+.01},
                 'camera_time':stamp, 'robot_time_monotonic':stamp+.01, 'clock':MONOTONIC,
                 'color_rgb':np.full((3,4,3), color, np.uint8)}
                for stamp, color in ((2.2,16), (12.2,64))]
            session._last_recorded_sample = samples[0]
            session.capture._lock = threading.RLock()
            session.capture._times, session.capture._clock_basis = [2.2], MONOTONIC
            session.fresh_recorded_observation = WholeTaskSession.fresh_recorded_observation.__get__(session)
            def advance(_):
                session._last_recorded_sample = samples[1]
                session.capture._times.append(12.2)
            requests = []
            def reply(request, *_args, **_kwargs):
                requests.append(deepcopy(request))
                return self.jev_reply(request, 'observe_scene' if len(requests) == 1 else 'release_gripper')
            with patch('fr5_jev.whole_task_collection.time.sleep', side_effect=advance) as pause, \
                 patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=[10.,11.,12.,13.,14.]), \
                 patch('fr5_jev.whole_task_collection.time.time', side_effect=[1010.,1011.,1012.,1013.,1014.]), \
                 patch('fr5_jev.decision.call_jev', side_effect=reply), \
                 patch('fr5_jev.memory.retrieve', return_value=[]):
                name = session.fetch_phase_decision('release', 1)
            pause.assert_called_once()
            self.assertEqual(name, 'jev-c01-release-after-observe.json')
            (session.root/'continuous/frames').mkdir(parents=True)
            captures = []
            for index, sample in enumerate(samples):
                rgb_file = f'frames/{index:06d}.jpg'
                _, jpeg, _ = encode_media(sample['color_rgb'], np.zeros((3,4), np.float32))
                (session.root/'continuous'/rgb_file).write_bytes(jpeg)
                captures.append({'frame_index':index, 'camera':sample['camera'],
                                 'robot':sample['robot'], 'rgb_file':rgb_file})
            (session.root/'continuous/capture.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in captures))
            manifest = {'schema':'fr5-final-control-manifest-v1', 'session':'synthetic-not-training', 'files':{}}
            for path in session.root.glob('*.json'):
                raw = path.read_bytes()
                manifest['files'][path.name] = {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
            image = session.root/'observe-c01-release.jpg'
            raw = image.read_bytes()
            manifest['files'][image.name] = {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
            (session.root/'control-manifest.json').write_text(json.dumps(manifest))
            source = {'root':session.root, 'source_scope_sha256':session.scope_sha,
                      'segments':[[{'command_issued_monotonic':1., 'command_completed_monotonic':2.}]]}
            teacher = {'phase':'release', 'cycle':1, 'decision_id':name}
            journal = [{'executed':True, 'issued_monotonic':15., 'teacher':teacher}]
            record = json.loads((session.root/name).read_text())
            _validate_openai_phase_decision(teacher, name, record, source, manifest, captures, journal)
            self.assertEqual(len(requests), 2)
            self.assertIn('not a physical demonstration or training data',
                          (session.root/'TEST_FIXTURE_ONLY').read_text())
            session.dispatch.assert_not_called()

    def test_refresh_followup_stop_still_interrupts_release(self):
        for choice in ('hold', 'request_astra', 'observe_scene', 'decision_error', 'refusal'):
            with self.subTest(choice=choice), tempfile.TemporaryDirectory() as tmp:
                session = self.decision_session(tmp, provider='openai')
                clock, calls = [100.], []
                def reply(request, *_args, **_kwargs):
                    facts = request['input'][0]['content'][0]['data']['facts']
                    if facts['requested_phase'] != 'release':
                        return self.jev_reply(request)
                    calls.append(deepcopy(request))
                    if len(calls) == 1:
                        return self.jev_reply(request, 'observe_scene')
                    if choice == 'decision_error':
                        raise DecisionError('Synthetic refresh decision unavailable')
                    if choice == 'refusal':
                        return {'model':'gpt-6-luna', 'id':'synthetic-not-training',
                                'answers':[{'name':'next_skill', 'type':'refusal'}]}, 1.
                    return self.jev_reply(request, choice)
                class InlineThread:
                    def __init__(self, target, **kwargs): self.target = target
                    def start(self): self.target()
                reason = 'consecutive observe_scene' if choice == 'observe_scene' else 'Phase decision requests LLM intervention'
                with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                     patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                     patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread), \
                     self.assertRaisesRegex(RuntimeError, reason):
                    self.execute_decision_session(session, reply)
                self.assertEqual(len(calls), 2)
                self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list], STAGES[:7])
                self.assertEqual(session.fresh_recorded_observation.call_count, STAGES.index('release')+2)
                session.controller.stop.assert_called_once()
                session.finish.assert_called_once_with('interrupted')
                original = json.loads((session.root/'jev-c01-release.json').read_text())
                refreshed = json.loads((session.root/'jev-c01-release-after-observe.json').read_text())
                self.assertEqual(original['request'], calls[0])
                self.assertEqual(original['answer']['choice'], 'observe_scene')
                self.assertEqual(refreshed['request'], calls[1])
                self.assertEqual(refreshed['route'], 'needs_llm')
                self.assertFalse(refreshed['automatic_continuation'])
                if choice in ('decision_error', 'refusal'):
                    self.assertIn('decision_error', refreshed)
                    if choice == 'refusal':
                        self.assertEqual(refreshed['response']['answers'][0]['type'], 'refusal')
                        self.assertIn('response_received_monotonic', refreshed)
                    else:
                        self.assertNotIn('response_received_monotonic', refreshed)
                else:
                    self.assertEqual(refreshed['answer']['choice'], choice)
                    self.assertEqual(refreshed['selected_by'], 'openai_decisions')

    def test_openai_observe_during_hold_keeps_hold_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, provider='openai')
            clock, calls = [100.], []
            def reply(request, *_args, **_kwargs):
                facts = request['input'][0]['content'][0]['data']['facts']
                if facts['requested_phase'] == 'return':
                    self.assertEqual(facts['current_cycle_completed_stages'], STAGES[:6])
                    self.assertIsNone(facts['actual_hold_duration_s'])
                    self.assertTrue(facts['hold_in_progress']['return_dispatch_waits_for_requested_hold'])
                    self.assertFalse(any(call.args[0]['phase'] == 'return' for call in session.dispatch.call_args_list))
                    choice = 'observe_scene' if not calls else 'return_place'
                    calls.append(deepcopy(request))
                    return self.jev_reply(request, choice)
                return self.jev_reply(request)
            class InlineThread:
                def __init__(self, target, **kwargs): self.target = target
                def start(self): self.target()
            with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                 patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread):
                self.execute_decision_session(session, reply)
            self.assertEqual(len(calls), 2)
            self.assertAlmostEqual(clock[0], 103.)
            self.assertEqual(session.hold_durations, [3.])
            hold = json.loads((session.root/'hold-01.json').read_text())
            self.assertTrue(hold['return_command_issued'])
            record = json.loads((session.root/'jev-c01-return-after-observe.json').read_text())
            self.assertEqual(record['selected_by'], 'openai_decisions')
            self.assertTrue(record['automatic_continuation'])
            phase = next(call for call in session.capture.phase.call_args_list if call.args[0] == 'return')
            self.assertEqual(phase.kwargs['teacher']['decision_id'], 'jev-c01-return-after-observe.json')
            session.controller.stop.assert_not_called()

    def test_observation_failure_does_not_redispatch_or_redecide(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, provider='openai')
            original_fresh = session.fresh_recorded_observation.side_effect
            def fresh(**kwargs):
                if kwargs.get('after_monotonic') is not None:
                    raise RuntimeError('Synthetic capture failure')
                return original_fresh(**kwargs)
            session.fresh_recorded_observation.side_effect = fresh
            requests = []
            def reply(request, *_args, **_kwargs):
                facts = request['input'][0]['content'][0]['data']['facts']
                requests.append(deepcopy(request))
                return self.jev_reply(request, 'observe_scene' if facts['requested_phase'] == 'approach_near' else None)
            with self.assertRaisesRegex(RuntimeError, 'Synthetic capture failure'):
                self.execute_decision_session(session, reply)
            self.assertEqual(len(requests), 2)
            self.assertEqual([call.args[0]['phase'] for call in session.dispatch.call_args_list], ['approach_clear'])
            self.assertFalse((session.root/'jev-c01-approach_near-after-observe.json').exists())
            record = json.loads((session.root/'jev-c01-approach_near.json').read_text())
            self.assertEqual(record['request'], requests[-1])
            self.assertEqual(record['observation_error'], 'RuntimeError: Synthetic capture failure')
            self.assertEqual(record['route'], 'needs_llm')
            self.assertFalse(record['automatic_continuation'])
            session.controller.stop.assert_called_once()
            session.finish.assert_called_once_with('interrupted')

    def test_initial_observation_failure_prevents_model_and_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, provider='openai')
            session.fresh_recorded_observation.side_effect = RuntimeError('Synthetic capture failure')
            with patch('fr5_jev.decision.call_jev') as decision, \
                 self.assertRaisesRegex(RuntimeError, 'Synthetic capture failure'):
                self.execute_decision_session(session, decision)
            decision.assert_not_called()
            session.dispatch.assert_not_called()
            session.controller.stop.assert_called_once()
            session.finish.assert_called_once_with('interrupted')
            self.assertFalse((session.root/'jev-c01-approach_clear.json').exists())

    def test_large_endpoint_not_subdivided_or_closing_ramped(self):
        routes = self.routes()
        routes['return'] = [[0,0,250,90,0,0], [0,0,202,90,0,0], [0,0,200,90,0,0]]
        actions = endpoint_actions(routes)
        self.assertEqual(len(actions), len(STAGES))
        self.assertEqual([a['gripper_percent'] for a in actions if a['phase']=='close'], [76])
        self.assertEqual([a['tcp_mm_deg'] for a in actions if a['phase']=='return'], [[0,0,200,90,0,0]])

    def test_short_retreat_ends_at_clearance_and_never_returns_to_start(self):
        routes = self.routes()
        actions = endpoint_actions(routes, retreat_mode='short')
        self.assertEqual(actions[-1]['phase'], 'retreat_clear')
        self.assertFalse(any(a['phase'] == 'retreat_high' for a in actions))
        scope = self.scope()
        scope.update(retreat_mode='short', phase_order=STAGES[:-1], actions=actions)
        validate_scope(scope)
        scope['actions'] = scope['actions'][:-1]
        with self.assertRaises(ValueError):
            validate_scope(scope)

    def test_whole_task_requires_transitions_and_complete_release(self):
        scope = self.scope()
        validate_scope(scope)
        scope['actions'] = scope['actions'][:-1]
        with self.assertRaises(ValueError):
            validate_scope(scope)
        scope = self.scope()
        next(a for a in scope['actions'] if a['phase']=='release')['gripper_percent'] = 76
        with self.assertRaises(ValueError):
            validate_scope(scope)

    def test_literal_task_and_physical_percent_contract(self):
        scope = self.scope()
        scope['task_instruction'] = 'Close the gripper'
        with self.assertRaises(ValueError):
            validate_scope(scope)
        with self.assertRaises(ValueError):
            endpoint_actions(self.routes(), closing_percent=102)

    def test_existing_cup_scope_retains_its_literal_task(self):
        scope = self.scope()
        del scope['task_spec']['object_description']
        validate_scope(scope)
        scope['task_instruction'] = '抓取桌面上的紫色方块，在空中维持5秒后放下'
        with self.assertRaises(ValueError):
            validate_scope(scope)

    def test_variable_task_scope_and_two_complete_cycles(self):
        self.assertEqual(task_instruction(5, 1), TASK)
        self.assertEqual(task_instruction(3, 1), '抓取桌面上的透明杯子，在空中维持3秒后放下')
        scope = self.scope(3., 2)
        self.assertEqual(scope['task_spec'], {'hold_seconds':3., 'repetitions':2, 'finish':'released',
                                              'object_description':'透明杯子'})
        self.assertIn('再次抓起', scope['task_instruction'])
        self.assertEqual(scope['task_instruction'].count('维持3秒'), 2)
        self.assertEqual(scope['phase_order'], STAGES * 2)
        self.assertEqual([a['cycle'] for a in scope['actions']], [1]*len(STAGES)+[2]*len(STAGES))
        self.assertEqual(len({a['id'] for a in scope['actions']}), 2*len(STAGES))
        validate_scope(scope)
        for seconds, repetitions in ((0,1), (-1,1), (float('nan'),1), (float('inf'),1),
                                     (3,0), (3,3), (3,True)):
            with self.subTest(seconds=seconds, repetitions=repetitions), self.assertRaises(ValueError):
                task_instruction(seconds, repetitions)
        scope['task_spec']['finish'] = 'held'
        with self.assertRaises(ValueError):
            validate_scope(scope)

    def test_recorded_targets_latch_wire_commands_not_measured_drift(self):
        class Session:
            phase = 'initial_hold'
            decision_id = 'scope.json'
            cycle = 0
            policy_name = 'current_codex_chat+gpt-6-luna'
            lock = threading.RLock()
        with tempfile.TemporaryDirectory() as tmp:
            s = Session()
            s.root, s.journal = Path(tmp), []
            controller = TeachingController(s, stream=None, camera_fresh=lambda: True,
                                            limits=DirectSettings(), rpc=object())
            controller._append([1,2,3,4,5,6,100], 'hold', executed=False)
            s.phase, s.pending_wire = 'close', {'method':'MoveGripper', 'args':[1,76]}
            s.cycle = 1
            command = controller._append([1.01,2,3,4,5,6,76], 'gripper', executed=True)
            self.assertEqual(command['target'], [1,2,3,4,5,6,76])
            self.assertEqual(command['teacher']['policy'], 'current_codex_chat+gpt-6-luna')
            s.phase, s.pending_wire = 'lift_full', {'method':'MoveL', 'args':[[2,3,4,5,6,7]]}
            command = controller._append([2,3,4,5,6,7,79], 'arm', executed=True)
            self.assertEqual(command['target'], [2,3,4,5,6,7,76])
            WholeTaskSession.persist_attempted_command(s)
            saved = json.loads((s.root/'command-0003.json').read_text())
            self.assertEqual(saved['command'], s.journal[-1])
            self.assertEqual(saved['status'], 'attempted_once')
            self.assertEqual(s.journal[1]['teacher']['phase'], 'close')
            self.assertEqual(s.journal[-1]['source'], 'agent_teacher')
            self.assertEqual(controller.command_at(command['issued_unix'])['target'], command['target'])

    def test_hold_records_true_duration_without_claiming_cup_motion(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root = Path(tmp)
            session.scope = {'task_spec': {'hold_seconds':5}}
            session.policy_name, session.decision_id = 'llm+jev', 'jev-lift_full.json'
            session.cycle = 1
            session._last_completed_monotonic = 99.9
            session._pending_hold = None
            session._prefetch_ready = threading.Event()
            session.hold_durations, session.hold_records = [], []
            session.held_seconds = 0.
            session.fetch_phase_decision = lambda phase, cycle: f'jev-c{cycle:02d}-{phase}.json'
            session.capture = type('Capture', (), {'phase':lambda *a, **kw: None})()
            states = []
            session.state = lambda **kw: states.append(kw)
            clock = [100.]
            class InlineThread:
                def __init__(self, target, **kwargs): self.target = target
                def start(self): self.target()
            with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('fr5_jev.whole_task_collection.time.sleep', side_effect=lambda n:clock.__setitem__(0,clock[0]+n)), \
                 patch('fr5_jev.whole_task_collection.threading.Thread', InlineThread):
                self.assertEqual(session.hold(), 'jev-c01-return.json')
                self.assertEqual(session._last_completed_monotonic, 99.9)
                self.assertFalse((session.root/'hold-01.json').exists())
                clock[0] += .2
                session.complete_hold_at_return(clock[0], 200.)
            hold = json.loads((session.root/'hold-01.json').read_text())
            self.assertGreaterEqual(hold['actual_duration_s'],5.)
            self.assertAlmostEqual(hold['actual_duration_s'],5.2)
            self.assertAlmostEqual(hold['actual_hover_from_lift_completion_s'],5.3)
            self.assertTrue(hold['return_command_issued'])
            self.assertEqual(hold, json.loads((session.root/'hold-5s.json').read_text()))
            self.assertEqual(session.hold_records, [{'repetition':1, 'path':'hold-01.json'}])
            self.assertFalse(hold['cup_lift_visually_verified'])
            self.assertGreater(len(states),200)

    def test_late_prefetch_waits_without_treating_task_hold_as_network_deadline(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.scope = Path(tmp), {'task_spec': {'hold_seconds':3}}
            session.policy_name, session.decision_id, session.cycle = 'llm+jev', 'lift', 2
            session._last_completed_monotonic, session._pending_hold = 99.9, None
            session._prefetch_ready = threading.Event()
            session.hold_records, session.hold_durations, session.held_seconds = [], [], 0.
            session.capture = type('Capture', (), {'phase':lambda *a, **kw: None})()
            session.state = lambda **kw: None
            clock = [100.]
            class DelayedThread:
                def __init__(self, target, **kwargs): pass
                def start(self): pass
            def advance(seconds):
                clock[0] += seconds
                if clock[0] >= 106.:
                    session._prefetch_result = {'name':'jev-c02-return.json'}
                    session._prefetch_ready.set()
            with patch('fr5_jev.whole_task_collection.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('fr5_jev.whole_task_collection.time.sleep', side_effect=advance), \
                 patch('fr5_jev.whole_task_collection.threading.Thread', DelayedThread):
                self.assertEqual(session.hold(), 'jev-c02-return.json')
                self.assertEqual(session._last_completed_monotonic, 99.9)
                self.assertFalse((session.root/'hold-02.json').exists())
                session.complete_hold_at_return(clock[0]+.2, 200.)
            row = json.loads((session.root/'hold-02.json').read_text())
            self.assertTrue(row['return_command_issued'])
            self.assertGreater(row['actual_duration_s'], 6.)
            self.assertGreater(row['timing_deviation_seconds'], 3.)
            self.assertEqual(session.hold_records, [{'repetition':2, 'path':'hold-02.json'}])
            self.assertFalse((session.root/'hold-5s.json').exists())

    def test_three_second_hold_records_real_return_time_not_timer_deadline(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.cycle = Path(tmp), 1
            session.hold_durations, session.hold_records, session.held_seconds = [], [], 0.
            session._pending_hold = {'cycle': 1, 'requested_seconds':3., 'start_monotonic':100.,
                                     'lift_completed_monotonic':99.8}
            session.complete_hold_at_return(103.7, 303.7)
            row = json.loads((session.root/'hold-01.json').read_text())
            self.assertAlmostEqual(row['actual_duration_s'],3.7)
            self.assertAlmostEqual(row['actual_hover_from_lift_completion_s'],3.9)
            self.assertEqual(row['end_monotonic'],103.7)
            self.assertFalse((session.root/'hold-5s.json').exists())

    def test_two_cycle_hold_evidence_matches_exporter_and_rejects_inconsistent_clock(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root = Path(tmp)
            session.hold_durations, session.hold_records, session.held_seconds = [], [], 0.
            (session.root/'continuous').mkdir()
            events = []
            commands = {}
            for cycle, start, lift_done, returned in ((1,101.7,101.5,104.9), (2,108.7,108.5,111.9)):
                session.cycle = cycle
                session._pending_hold = {'cycle':cycle, 'requested_seconds':3., 'start_monotonic':start,
                                         'lift_completed_monotonic':lift_done}
                session.complete_hold_at_return(returned, 2000.+returned)
                events.append({'kind':'phase', 'name':'hold', 'time_monotonic':start-.05})
                for offset, phase, issued, complete, method in (
                    (0,'close',lift_done-1.5,lift_done-1.,'MoveGripper'),
                    (1,'lift_full',lift_done-.3,lift_done,'MoveL'),
                    (2,'return',returned,returned+.2,'MoveL'),
                    (3,'release',returned+.4,returned+.6,'MoveGripper'),
                ):
                    seq = cycle*10+offset
                    commands[seq] = {'phase':phase, 'method':method,
                        'command':{'sequence':seq, 'issued_monotonic':issued}, 'completed_monotonic':complete}
            (session.root/'continuous/events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            frames = [{'frame_index':j, 'camera':{'source_timestamp_monotonic':100.+j/6}}
                      for j in range(90)]
            manifest = {'files':{}}
            for name in ('hold-01.json', 'hold-02.json'):
                raw = (session.root/name).read_bytes()
                manifest['files'][name] = {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
            task = task_specification(self.scope(3., 2))
            holds = _holds(session.root, manifest, {'hold_records':session.hold_records}, commands, frames, task)
            self.assertEqual([h['repetition'] for h in holds], [1,2])
            self.assertTrue(all(3. <= h['actual_duration_s'] < 3.5 for h in holds))
            path = session.root/'hold-02.json'
            data = json.loads(path.read_text())
            data['end_monotonic'] += 1.
            data['actual_duration_s'] += 1.
            path.write_text(json.dumps(data))
            raw = path.read_bytes()
            manifest['files']['hold-02.json'] = {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'completed lift and return'):
                _holds(session.root, manifest, {'hold_records':session.hold_records}, commands, frames, task)

    def test_sample_preserves_accepted_paired_rgb_and_robot(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        session.lock = threading.RLock()
        session._last_recorded_sample = None
        sample = {'camera': {
            'metadata': {'source_camera_serial':'synthetic', 'source_timestamp_unix':21.,
                         'source_timestamp_monotonic':2.2},
            'color_rgb': np.full((3,4,3), (224,64,16), np.uint8)},
            'robot_state': {'tcp':[1.]*6, 'received_monotonic':2.21}}
        expected = deepcopy(sample)
        row = {'frame_index':1}
        with patch.object(TeacherSession, 'sample', side_effect=[row, None]) as record:
            self.assertIs(session.sample(sample), row)
            stored = session._last_recorded_sample
            sample['camera']['metadata']['source_timestamp_unix'] = 22.
            sample['camera']['color_rgb'][:] = 0
            sample['robot_state']['tcp'][0] = 9.
            self.assertIsNone(session.sample(sample))
        self.assertEqual(record.call_count, 2)
        self.assertIs(session._last_recorded_sample, stored)
        self.assertEqual(stored['camera'], expected['camera']['metadata'])
        self.assertEqual(stored['robot'], expected['robot_state'])
        np.testing.assert_array_equal(stored['color_rgb'], expected['camera']['color_rgb'])
        self.assertEqual(stored['camera_time'], 2.2)
        self.assertEqual(stored['robot_time_monotonic'], 2.21)
        self.assertEqual(stored['clock'], MONOTONIC)

    def test_fresh_observation_waits_for_committed_camera_and_robot_after_completion(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        session.lock = threading.RLock()
        session.state = Mock()
        session._last_completed_monotonic = 2.
        session._last_recorded_sample = {'camera':{'id':'old'}, 'camera_time':2.1,
                                       'robot_time_monotonic':1.99, 'clock':MONOTONIC}
        session.capture = Mock()
        session.capture._lock = threading.RLock()
        session.capture._times = [2.1]
        session.capture._clock_basis = MONOTONIC
        def advance(_):
            session._last_recorded_sample = {'camera':{'id':'fresh'}, 'camera_time':2.2,
                                           'robot_time_monotonic':2.21, 'clock':MONOTONIC}
            session.capture._times.append(2.2)
        with patch('fr5_jev.whole_task_collection.time.sleep', side_effect=advance) as pause:
            sample = session.fresh_recorded_observation()
        self.assertEqual(sample['camera']['id'], 'fresh')
        self.assertEqual(sample['frame_index'], 1)
        pause.assert_called_once()

    def test_observe_waits_for_new_committed_image_after_request(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.lock, session.state = threading.RLock(), Mock()
            session._last_completed_monotonic = 2.
            old = {'camera':{'source_timestamp_unix':20.}, 'camera_time':2.5,
                   'robot_time_monotonic':2.51, 'clock':MONOTONIC,
                   'robot':{'tcp':[0.]*6}, 'color_rgb':np.full((3,4,3), 16, np.uint8)}
            new = {'camera':{'source_timestamp_unix':21.}, 'camera_time':3.1,
                   'robot_time_monotonic':3.12, 'clock':MONOTONIC,
                   'robot':{'tcp':[1.]*6}, 'color_rgb':np.full((3,4,3), (224,64,16), np.uint8)}
            session._last_recorded_sample = old
            session.capture = Mock()
            session.capture._lock = threading.RLock()
            session.capture._times, session.capture._clock_basis = [2.5], MONOTONIC
            def advance(_):
                if session._last_recorded_sample is old:
                    session._last_recorded_sample = new
                else:
                    session.capture._times.append(3.1)
            image_path = Path(tmp)/'synthetic-observation.jpg'
            with patch('fr5_jev.whole_task_collection.time.sleep', side_effect=advance) as pause:
                observed = session.fresh_recorded_observation(after_monotonic=3., image_path=image_path)
            self.assertEqual(pause.call_count, 2)
            self.assertEqual(observed['camera'], new['camera'])
            self.assertEqual(observed['robot'], new['robot'])
            self.assertEqual(observed['camera_time'], 3.1)
            self.assertEqual(observed['robot_time_monotonic'], 3.12)
            self.assertEqual(observed['frame_index'], 1)
            self.assertEqual(observed['image_file'], image_path.name)
            self.assertNotIn('color_rgb', observed)
            self.assertEqual(old['camera_time'], 2.5)
            self.assertEqual(old['camera']['source_timestamp_unix'], 20.)
            with Image.open(image_path) as image:
                np.testing.assert_allclose(image.getpixel((0,0)), (224,64,16), atol=2)
            _, continuous_jpeg, _ = encode_media(new['color_rgb'], np.zeros((3,4), np.float32))
            self.assertEqual(image_path.read_bytes(), continuous_jpeg)

    def test_collection_watchdog_does_not_enforce_an_arbitrary_task_deadline(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.failed = Path(tmp), None
            session.interrupted = threading.Event()
            session.controller = Mock(deadline=10., error=None)
            session.capture = Mock(spec=['assert_healthy'])
            with patch('fr5_jev.whole_task_collection.time.monotonic', return_value=10.):
                self.assertTrue(session.watchdog_once())
            session.controller.stop.assert_not_called()
            self.assertIsNone(session.failed)

    def test_recording_sync_occurs_after_controller_send_gate_unlocks(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.phase, session.cycle = Path(tmp), 'approach_clear', 1
            session.decision_id, session.decision_observation = 'jev-c01-approach_clear.json', {'camera':{}}
            session.policy_name = 'current_codex_chat+jev'
            session.lock, session.journal = threading.RLock(), []
            session.state = lambda: {'joints':[0.]*6, 'gripper_position':100.}
            session._pending_hold = None
            controller = TeachingController(session, stream=None, camera_fresh=lambda:True,
                                            limits=DirectSettings(), rpc=object())
            controller._append([0.,0.,0.,0.,0.,0.,100.], 'hold', executed=False)
            @contextmanager
            def gate():
                with controller._lock:
                    controller._append([1.,0.,0.,0.,0.,0.,100.], 'arm', executed=True)
                    yield
            class Wire:
                def call(self, method, *args, **kwargs):
                    with kwargs['before_send']():
                        pass
                    return 0
            checks = []
            def check_unlocked(path, value):
                acquired = []
                def other_thread():
                    ok = controller._lock.acquire(blocking=False)
                    acquired.append(ok)
                    if ok: controller._lock.release()
                thread = threading.Thread(target=other_thread)
                thread.start(); thread.join(1.)
                checks.append(acquired == [True])
            with patch('fr5_jev.whole_task_collection.durable_json', side_effect=check_unlocked):
                result = RecordingRPC(Wire(), session).call('MoveL', [1.,0.,0.,0.,0.,0.], before_send=gate)
            self.assertEqual(result, 0)
            self.assertEqual(checks, [True, True])
            self.assertEqual(session.pending_wire['command']['target'][0], 1.)

    def test_dispatch_records_actual_arm_and_gripper_wire_targets(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.phase, session.cycle = Path(tmp), 'approach_clear', 1
            session.policy_name = 'current_codex_chat+jev'
            session.lock, session.journal = threading.RLock(), []
            session.failed, session.interrupted = None, threading.Event()
            session.capture, session._pending_hold = None, None
            session._last_completed_monotonic = None
            session.decision_id = 'jev-c01-approach_clear.json'
            session.fresh_recorded_observation = lambda: {'camera':{'serial':'offline'}, 'frame_index':1}
            state = dict(joints=[0.]*6, tcp=[0.]*6, flange=[0.]*6,
                         gripper_position=100., gripper_active=1, gripper_fault=0,
                         gripper_fault_id=0, gripper_motiondone=1, gripper_speed=0,
                         main_code=0, sub_code=0, EmergencyStop=0, collisionState=0,
                         safety_stop0_state=0, safety_stop1_state=0, rbtEnableState=1,
                         program_state=1, robot_state=1, motion_done=1, mc_queue_len=0,
                         tool=0, user=0, frame_counter=1)
            stream = Mock()
            def latest(**_):
                state['frame_counter'] += 1
                return deepcopy(state)
            stream.latest.side_effect = latest
            class Wire:
                def call(self, method, *args, **kwargs):
                    if method == 'GetInverseKinRef': return [0, *args[1]]
                    if method == 'GetForwardKin': return [0, *args[0]]
                    if method in ('MoveL', 'MoveGripper'):
                        with kwargs['before_send']():
                            if method == 'MoveL':
                                state.update(joints=list(args[0]), tcp=list(args[1]), flange=list(args[1]))
                            else:
                                state.update(gripper_position=float(args[1]), gripper_motiondone=2)
                        return 0
                    raise AssertionError(f'Unexpected synthetic RPC {method}')
            session.rpc = Wire()
            controller = TeachingController(session, stream=stream, camera_fresh=lambda:True,
                limits=DirectSettings(), allow_motion=True, rpc=RecordingRPC(session.rpc, session))
            session.controller = controller
            controller.armed, controller.deadline = True, float('inf')
            controller.joint_limits = np.array([[-90.,90.]]*6)
            controller._append([0.,0.,0.,0.,0.,0.,100.], 'hold', executed=False)
            session.dispatch({'kind':'arm', 'tcp_mm_deg':[5.,0.,0.,0.,0.,0.]})
            session.phase, session.decision_id = 'close', 'jev-c01-close.json'
            with patch.object(controller, '_gripper_complete', return_value=True):
                session.dispatch({'kind':'gripper', 'gripper_percent':76.})
            journal = json.loads((session.root/'target-command-journal.json').read_text())
            arm = json.loads((session.root/'command-0002.json').read_text())
            grip = json.loads((session.root/'command-0003.json').read_text())
            self.assertEqual([row['kind'] for row in journal], ['hold', 'arm', 'gripper'])
            self.assertEqual(arm['status'], 'completed')
            self.assertEqual(arm['args'][0], [5.,0.,0.,0.,0.,0.])
            self.assertEqual(arm['command']['target'], [5.,0.,0.,0.,0.,0.,100.])
            self.assertEqual(grip['status'], 'completed')
            self.assertEqual(grip['args'][1], 76)
            self.assertEqual(grip['command']['target'], [5.,0.,0.,0.,0.,0.,76.])
            self.assertEqual(grip['command']['teacher']['cycle'], 1)
            self.assertEqual(grip['decision_observation']['camera']['serial'], 'offline')

    def test_jev_receives_current_cycle_route_without_geometry_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self.decision_session(tmp, seconds=3., repetitions=2)
            session.cycle = 2
            def reply(request, *_args, **_kwargs):
                choices = request['questions']['next_skill']['criteria']
                probs = {name:float(name == 'transfer_empty_tool') for name in choices}
                return {'answers': {'next_skill': {'type':'choice', 'choice':'transfer_empty_tool',
                                                   'confidence':1., 'probabilities':probs}}}, 1.
            with patch('fr5_jev.decision.call_jev', side_effect=reply), \
                 patch('fr5_jev.memory.retrieve', return_value=[]):
                result = session.decide('approach_clear', 'jev-c02-approach_clear.json')
            self.assertEqual(result['answer']['choice'], 'transfer_empty_tool')
            record = json.loads((session.root/'jev-c02-approach_clear.json').read_text())
            candidate = record['state']['candidates']['transfer_empty_tool']
            self.assertEqual(candidate['precomputed_actions'], ['c02-approach_clear-000'])
            self.assertFalse(candidate['geometry_or_collision_independently_validated'])
            self.assertFalse(record['state']['facts']['geometry_or_collision_independently_validated'])
            self.assertFalse(record['physical_execution'])

    def test_two_cycles_complete_and_refresh_after_last_command(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.scope = Path(tmp), self.scope(3., 2)
            session.origin = np.array([0., 0., 200., 90., 0., 0.])
            session.completed_stages, session.completed_cycles, session.peak_height = [], 0, 0.
            session.journal = []
            session.pending_wire, session._last_completed_monotonic = None, None
            state = {'joints':[0.]*6, 'tcp':[0.,0.,200.,90.,0.,0.], 'gripper_position':100.,
                     'gripper_active':1, 'gripper_fault':0, 'gripper_fault_id':0,
                     'gripper_motiondone':1, 'gripper_speed':0}
            session.state = lambda: deepcopy(state)
            selected = []
            session.select_phase = lambda phase, cycle, prefetched=None: selected.append((phase, cycle, prefetched))
            def dispatch(action):
                session.journal.append({'executed':True, 'teacher':{'cycle':action['cycle'], 'phase':action['phase']}})
                if action['phase'] == 'close':
                    state.update(gripper_position=76., gripper_motiondone=2)
                elif action['phase'] == 'lift_full':
                    state['tcp'][2] = 250.
                elif action['phase'] == 'return':
                    state['tcp'][2] = 200.
                elif action['phase'] == 'release':
                    state.update(gripper_position=100., gripper_motiondone=1)
                return deepcopy(state)
            session.dispatch = dispatch
            session.hold = lambda: f'jev-c{selected[-1][1]:02d}-return.json'
            session.fresh_recorded_observation = Mock(return_value={'frame_index':100})
            session.finish = Mock(return_value={'outcome':'pending_review'})
            with redirect_stdout(io.StringIO()):
                result = session.run()
            self.assertEqual(result['outcome'], 'pending_review')
            self.assertEqual(session.completed_stages, STAGES * 2)
            self.assertEqual(session.completed_cycles, 2)
            self.assertEqual([(c, name) for phase,c,name in selected if phase == 'return'],
                             [(1,'jev-c01-return.json'), (2,'jev-c02-return.json')])
            session.fresh_recorded_observation.assert_called_once()
            self.assertEqual(session.finish.call_args.args[1],
                             ['scene.jpg', 'hold-01.json', 'hold-02.json', 'target-command-journal.json'])

    def test_pending_review_requires_completed_wire_records_and_real_hold(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with tempfile.TemporaryDirectory() as tmp:
            session.root, session.scope = Path(tmp), self.scope(3., 1)
            session._watchdog_stop, session.interrupted = threading.Event(), threading.Event()
            session._prefetch_thread, session._pending_hold, session.failed = None, None, None
            session.completed_stages, session.completed_cycles = STAGES.copy(), 1
            session.released = session.returned = True
            session.hold_records = [{'repetition':1, 'path':'hold-01.json'}]
            session.hold_durations = [3.1]
            session.journal = []
            for seq, phase, kind, issued, completed in (
                (2,'close','gripper',100.,101.), (3,'lift_full','arm',102.,104.5),
                (4,'return','arm',107.7,109.), (5,'release','gripper',110.,111.),
            ):
                target = [0.,0.,0.,0.,0.,0.,76. if phase != 'release' else 100.]
                command = {'sequence':seq, 'issued_monotonic':issued, 'executed':True,
                           'kind':kind, 'target':target, 'teacher':{'cycle':1, 'phase':phase}}
                session.journal.append(command)
                wire = {'command':command, 'status':'completed', 'outcome':{'return_code':0},
                        'completed_monotonic':completed,
                        'method':'MoveL' if kind == 'arm' else 'MoveGripper',
                        'args':[target[:6]] if kind == 'arm' else [1, target[6]]}
                (session.root/f'command-{seq:04d}.json').write_text(json.dumps(wire))
            hold = {'repetition':1, 'requested_seconds':3., 'start_monotonic':104.6,
                    'end_monotonic':107.7, 'actual_duration_s':3.1,
                    'lift_completed_monotonic':104.5, 'return_command_issued':True,
                    'actual_hover_from_lift_completion_s':3.2}
            (session.root/'hold-01.json').write_text(json.dumps(hold))
            with patch.object(TeacherSession, 'finish', return_value={'ok':True}) as finish:
                session.finish('pending_review', ['hold-01.json'])
                self.assertEqual(finish.call_args.args[0], 'pending_review')
                lift_path = session.root/'command-0003.json'
                lift = json.loads(lift_path.read_text()); lift['completed_monotonic'] = 104.
                lift_path.write_text(json.dumps(lift))
                hold['lift_completed_monotonic'] = 104.
                hold['actual_hover_from_lift_completion_s'] = 3.7
                (session.root/'hold-01.json').write_text(json.dumps(hold))
                session.finish('pending_review', ['hold-01.json'])
                self.assertEqual(finish.call_args.args[0], 'pending_review')
                self.assertIsNone(session.failed)
                session.failed = None
                wire_path = session.root/'command-0005.json'
                wire = json.loads(wire_path.read_text()); wire['status'] = 'attempted_once'
                wire_path.write_text(json.dumps(wire))
                session.finish('pending_review', ['hold-01.json'])
                self.assertEqual(finish.call_args.args[0], 'interrupted')
                self.assertIn('incomplete', session.failed)

    def test_finish_cannot_self_label_success(self):
        session = WholeTaskSession.__new__(WholeTaskSession)
        with self.assertRaises(ValueError):
            session.finish('success')


if __name__ == '__main__':
    unittest.main()
