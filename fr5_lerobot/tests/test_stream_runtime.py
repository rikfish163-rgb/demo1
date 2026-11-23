"""No-network tests for delayed chunks and incompatible checkpoint rejection."""
import json
import math
import signal
import time
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lerobot_robot_fairino.stream_contract import CUP_RELATIVE_WIRE, NAME
from lerobot_robot_fairino.stream_policy import StreamPolicy, validate_reply
from lerobot_robot_fairino.stream_runtime import (
    _CUP_BOUNDARY_WEIGHTS, CupProposalRejected, OrderedChunk, InferenceWorker, predict_stream_reply, require_cup_contact,
    execution_prefix, run, shutdown, validate_loaded_settings,
)
from fr5_lerobot.training.stream_observation_history import (
    CUP_HISTORY_PROFILE, CUP_TCP_HISTORY_PROFILE, CUP_SETDOWN_PROFILE, CUP_SETDOWN_CLEAN_PROFILE,
    CUP_SETDOWN_FIXED_PROFILE,
)


def example():
    request = {'id': 'observation-a', 'source_time': 100., 'state': [0.]*6+[1., 1.]}
    reply = {'schema': 'fr5-stream-reply-v1', 'id': request['id'],
             'checkpoint_sha256': 'a'*64, 'policy_contract': NAME,
             'actions': [[math.radians(i/2)]*6+[1.] for i in range(50)]}
    return request, reply


class StreamRuntimeTests(unittest.TestCase):
    def test_shorter_execution_prefix_preserves_the_sealed_service_setting(self):
        policy = Mock(n_action_steps=10)
        self.assertEqual(execution_prefix(policy), 10)
        self.assertEqual(execution_prefix(policy, 3), 3)
        self.assertEqual(policy.n_action_steps, 10)
        for value in (0, 11, True, 3.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                execution_prefix(policy, value)

    def test_worker_publishes_three_original_actions_and_keeps_complete_evidence(self):
        request, reply = example()
        policy = Mock(checkpoint_sha='a' * 64, n_action_steps=10,
                      observation_context_profile='none')
        policy.predict.return_value = reply
        recorder = Mock()
        with tempfile.TemporaryDirectory() as directory, patch(
                'lerobot_robot_fairino.stream_runtime.make_request', return_value=request):
            worker = InferenceWorker(policy, recorder, '抓取物体', directory, execution_steps=3)
            worker.start()
            try:
                deadline = time.monotonic() + 2
                while worker.latest() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                chunk = worker.latest()
                self.assertIsNotNone(chunk)
                outputs = [chunk.take(101. + i / 6) for i in range(3)]
                self.assertEqual([row[1]['action_index'] for row in outputs], [0, 1, 2])
                self.assertTrue(chunk.exhausted)
                self.assertEqual(len(chunk.actions), 50)
                audit = json.loads((Path(directory) / 'chunk-00000.json').read_text())
                self.assertEqual(audit['execution_prefix_steps'], 3)
                self.assertEqual(audit['reply']['actions'], reply['actions'])
                self.assertEqual(policy.n_action_steps, 10)
            finally:
                worker.stop()
                worker.thread.join(2)

    def test_preview_checks_the_requested_prefix_and_preserves_the_full_reply(self):
        request, reply = example()
        policy = Mock(checkpoint='/model', ready=True, checkpoint_sha='a' * 64,
                      n_action_steps=10, training_data_kind='captured',
                      observation_context_profile='none')
        robot = Mock(is_connected=True)
        scene = {'table_plane': {'normal': [0., 0., 1.], 'offset_mm': 0.}}
        with tempfile.TemporaryDirectory() as directory, \
                patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', Path(directory)), \
                patch('socket.socket.connect', side_effect=AssertionError('offline connection')), \
                patch('lerobot_robot_fairino.robot_fairino.FairinoFR5', return_value=robot), \
                patch('lerobot_robot_fairino.stream_runtime.make_request', return_value=request), \
                patch('lerobot_robot_fairino.stream_runtime.predict_stream_reply', return_value=reply), \
                patch('fr5_lerobot.diagnostics.preflight_jev_pi05_actions.preflight',
                      return_value={'passed': True}) as preflight:
            result = run('/model', None, '抓取物体', Path(directory) / 'preview',
                         loaded_policy=policy, joint_scene=scene, execution_steps=3,
                         expected_action_steps=10)
        self.assertEqual(preflight.call_args.args[3], 3)
        self.assertEqual(result['checkpoint_n_action_steps'], 10)
        self.assertEqual(result['execution_prefix_steps'], 3)
        self.assertEqual(result['reply']['actions'], reply['actions'])
        self.assertEqual(result['robot_commands_sent'], 0)

    def test_cup_profiles_use_complete_proposal_selection(self):
        cup = {'table_normal': [0., 0., 1.]}
        reply = {'cup_proposal_selection': {'matched': True}}
        for profile, digest in (
                (CUP_SETDOWN_PROFILE, _CUP_BOUNDARY_WEIGHTS[0]),
                (CUP_SETDOWN_PROFILE, _CUP_BOUNDARY_WEIGHTS[1]),
                (CUP_SETDOWN_FIXED_PROFILE, 'other')):
            request = {'observation_context_profile': profile}
            with self.subTest(profile=profile, digest=digest), patch(
                    'lerobot_robot_fairino.stream_runtime.choose_proposal', return_value=(reply, 2, True)) as choose:
                policy = Mock(n_action_steps=3, checkpoint_sha=digest)
                self.assertIs(predict_stream_reply(policy, request, cup), reply)
                choose.assert_called_once_with(policy, request, cup['table_normal'], check_boundaries=True)
                policy.predict.assert_not_called()

    def test_cup_profiles_reject_an_exhausted_proposal_call(self):
        reply = {'cup_proposal_selection': {'draws': 3, 'matched': False}}
        for profile, digest in ((CUP_SETDOWN_PROFILE, _CUP_BOUNDARY_WEIGHTS[1]),
                                (CUP_SETDOWN_FIXED_PROFILE, 'other')):
            policy = Mock(n_action_steps=3, checkpoint_sha=digest)
            with self.subTest(profile=profile), patch(
                    'lerobot_robot_fairino.stream_runtime.choose_proposal', return_value=(reply, 3, False)):
                with self.assertRaises(CupProposalRejected) as caught:
                    predict_stream_reply(policy, {'observation_context_profile': profile},
                                         {'table_normal': [0., 0., 1.]})
            self.assertIs(caught.exception.reply, reply)
            policy.predict.assert_not_called()

    def test_cup_profiles_without_three_step_prefix_keep_single_prediction(self):
        _, reply = example()
        for profile in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            for steps in (1, 2, 4, 10, 50):
                request = {'observation_context_profile': profile}
                policy = Mock(n_action_steps=steps, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                policy.predict.return_value = reply
                with self.subTest(profile=profile, steps=steps), patch(
                        'lerobot_robot_fairino.stream_runtime.choose_proposal') as choose:
                    self.assertIs(predict_stream_reply(policy, request, None), reply)
                    policy.predict.assert_called_once_with(request)
                    choose.assert_not_called()

    def test_other_profiles_keep_single_prediction(self):
        _, reply = example()
        for profile in (None, 'none', CUP_SETDOWN_CLEAN_PROFILE):
            request = {} if profile is None else {'observation_context_profile': profile}
            policy = Mock(n_action_steps=3, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
            policy.predict.return_value = reply
            with self.subTest(profile=profile), patch(
                    'lerobot_robot_fairino.stream_runtime.choose_proposal') as choose:
                self.assertIs(predict_stream_reply(policy, request, None), reply)
                policy.predict.assert_called_once_with(request)
                choose.assert_not_called()

    def test_infeasible_proposals_are_recorded_before_the_worker_stops(self):
        request, reply = example()
        reply['cup_proposal_selection'] = {'draws': 3, 'matched': False,
            'selected_index': 0, 'proposals': [{'actions': reply['actions'], 'matched': False}] * 3}
        rejection = CupProposalRejected(reply)
        policy = Mock(checkpoint_sha='a'*64, n_action_steps=3, observation_context_profile='none')
        recorder = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch('lerobot_robot_fairino.stream_runtime.make_request', return_value=request), \
                patch('lerobot_robot_fairino.stream_runtime.predict_stream_reply', side_effect=rejection):
            worker = InferenceWorker(policy, recorder, '抓杯', directory)
            worker.start()
            worker.thread.join(2.)
            self.assertFalse(worker.thread.is_alive())
            self.assertIs(worker.error, rejection)
            self.assertIsNone(worker.current)
            recorder.record_prediction.assert_called_once()
            self.assertIs(recorder.record_prediction.call_args.args[1], reply)
            audit = json.loads((Path(directory) / 'chunk-00000.json').read_text())
            self.assertEqual(audit['status'], 'rejected_before_execution')
            self.assertEqual(audit['reply']['cup_proposal_selection']['draws'], 3)

    def test_selected_bundle_prefix_and_hash_must_match_the_loaded_service(self):
        policy=Mock(checkpoint_sha='a'*64,n_action_steps=50)
        # An old bundled service has no N field and defaults to 50. A child
        # configured for 10 must not silently execute the longer prefix.
        with self.assertRaisesRegex(ValueError,'n_action_steps differs'):
            validate_loaded_settings(policy,expected_action_steps=10,expected_sha='a'*64)
        with self.assertRaisesRegex(ValueError,'SHA differs'):
            validate_loaded_settings(policy,expected_action_steps=50,expected_sha='b'*64)
        validate_loaded_settings(policy,expected_action_steps=50,expected_sha='a'*64)
        policy.n_action_steps=10
        validate_loaded_settings(policy,expected_action_steps=10,expected_sha='a'*64)

    def test_shutdown_does_not_skip_recording_or_observer_if_policy_close_fails(self):
        policy=Mock();policy.close.side_effect=RuntimeError('lost SSH')
        robot=Mock(is_connected=True)
        controller=Mock()
        controller.close.return_value={'stop_result':0,'failure':None,'workers_alive':[],
            'startup_inflight':False,'pending_callbacks':0,'undelivered_callbacks':[]}
        recorder=Mock();recorder.finish.return_value={'task_success_verified':False}
        result,errors=shutdown(policy,robot,recorder,controller,None,'operator_stop',None)
        self.assertIn('lost SSH','; '.join(errors))
        self.assertIn('lost SSH',recorder.finish.call_args.kwargs['error'])
        robot.disconnect.assert_called_once()
        self.assertFalse(result['task_success_verified'])

    def test_uncertain_native_stop_is_preserved_as_failure(self):
        policy,robot,recorder,controller=Mock(),Mock(is_connected=True),Mock(),Mock()
        controller.close.return_value={'stop_result':None,'failure':'stop response lost','workers_alive':[],
                                       'last_issued_monotonic':100.}
        _,errors=shutdown(policy,robot,recorder,controller,None,'operator_stop',None)
        self.assertTrue(errors)
        self.assertIn('not fully acknowledged',recorder.finish.call_args.kwargs['error'])

    def test_controller_fault_with_acknowledged_cleanup_preserves_the_cause(self):
        controller_failure = 'Live device state failed: Controller reports collisionState=1'
        snapshot = {'status':'closed', 'failure':controller_failure, 'stop_result':0,
                    'servo_end_result':0, 'workers_alive':[], 'pending_callbacks':0,
                    'inflight_commands':0, 'undelivered_callbacks':[],
                    'startup_inflight':False, 'last_issued_monotonic':100.}
        for failure in (None, 'RuntimeError: Servo is not running',
                        f'RuntimeError: Native controller stopped: {controller_failure}'):
            with self.subTest(failure=failure):
                controller, recorder = Mock(), Mock()
                controller.close.return_value = snapshot
                recorder.finish.side_effect = lambda reason, error, controller_snapshot: {
                    'reason':reason, 'error':error, 'controller_snapshot':controller_snapshot,
                    'task_success_verified':False}
                summary, errors = shutdown(Mock(), Mock(is_connected=True), recorder,
                                           controller, None, 'error', failure)
                self.assertIn(controller_failure, summary['error'])
                if failure:
                    self.assertIn(failure, summary['error'])
                else:
                    self.assertIn(controller_failure, errors)
                self.assertNotIn('Controller stop/cleanup was not fully acknowledged',
                                 summary['error'])
                self.assertNotIn('Controller stop/cleanup was not fully acknowledged',
                                 '; '.join(errors))
                self.assertEqual(summary['controller_snapshot'], snapshot)
                self.assertFalse(summary['task_success_verified'])

    def test_stop_before_start_needs_no_native_stop_acknowledgement(self):
        controller=Mock()
        controller.close.return_value={'stop_result':None,'failure':None,'workers_alive':[],
            'last_issued_monotonic':None,'pending_callbacks':0}
        recorder=Mock()
        _,errors=shutdown(Mock(),Mock(is_connected=True),recorder,controller,None,'operator_stop',None)
        self.assertEqual(errors,[])
        self.assertIsNone(recorder.finish.call_args.kwargs['error'])

    def test_live_inference_writer_prevents_sealing_the_recording(self):
        policy,robot,recorder,inference=Mock(),Mock(is_connected=True),Mock(),Mock()
        inference.thread.is_alive.return_value=True
        _,errors=shutdown(policy,robot,recorder,None,inference,'operator_stop',None)
        self.assertTrue(errors)
        recorder.finish.assert_not_called()
        recorder.storage.frame_spool.abort.assert_called_once()
        robot.disconnect.assert_called_once()

    def test_transport_delay_cannot_drop_unexecuted_prefix_actions(self):
        request, reply = example()
        chunk = OrderedChunk(request, reply, 'a'*64)
        target, evidence = chunk.take(100.5)
        self.assertEqual(target[:6], [0.]*6)
        self.assertEqual(evidence['action_index'], 0)
        self.assertAlmostEqual(evidence['target_monotonic'], 100.5)

    def test_late_target_stops_before_consuming_the_next_gripper_action(self):
        request, reply = example()
        reply['actions'][1][6] = .76
        chunk = OrderedChunk(request, reply, 'a'*64, execution_steps=10)
        chunk.take(100.8)
        with self.assertRaisesRegex(RuntimeError, 'late.*action 1'):
            chunk.take(103.4)
        self.assertEqual(chunk.cursor, 1)
        self.assertFalse(chunk.exhausted)

    def test_policy_joint_jump_rejects_the_action_at_start_and_inside_a_chunk(self):
        for index in (0, 1):
            request, reply = example()
            reply['actions'][index][2] = math.radians(43.)
            chunk = OrderedChunk(request, reply, 'a'*64, execution_steps=10)
            if index:
                chunk.take(100.4)
            with self.assertRaisesRegex(RuntimeError, f'joint step is 43.00°.*action {index}'):
                chunk.take(100.6)
            self.assertEqual(chunk.cursor, index)

    def test_chunk_from_unsettled_gripper_state_cannot_drive_motion(self):
        request, reply = example()
        request['state'][6:] = [0., 0.]
        chunk = OrderedChunk(request, reply, 'a'*64, execution_steps=10)
        with self.assertRaisesRegex(RuntimeError, 'No settled gripper feedback'):
            chunk.take(100.6)
        self.assertEqual(chunk.cursor, 0)

    def test_cup_close_requires_contact_from_this_command(self):
        feedback = {'target_percent': 76, 'command_began_monotonic': 10.01,
                    'position_percent': 78, 'motiondone': 2}
        require_cup_contact(feedback, 76, 10.)
        for replacement in ({'motiondone': 1, 'position_percent': 76},
                            {'command_began_monotonic': 9.9}, {'target_percent': 80}):
            with self.assertRaisesRegex(RuntimeError, 'did not report cup contact'):
                require_cup_contact(feedback | replacement, 76, 10.)

    def test_every_chunk_action_is_consumed_exactly_once(self):
        request, reply = example()
        chunk = OrderedChunk(request, reply, 'a'*64)
        for i in range(50):
            self.assertEqual(chunk.take(101.+i/6)[1]['action_index'],i)
        self.assertTrue(chunk.exhausted)
        with self.assertRaises(StopIteration):
            chunk.take(120.)

    def test_policy_prefix_starts_at_zero_then_reobserves_without_consuming_tail(self):
        request, reply = example()
        for row in reply['actions'][:8]:
            row[6] = .757
        chunk = OrderedChunk(request, reply, 'a'*64, execution_steps=10)
        outputs = [chunk.take(101.+i/6) for i in range(10)]
        self.assertEqual([item[1]['action_index'] for item in outputs], list(range(10)))
        self.assertEqual([item[0][6] for item in outputs[:8]], [.76]*8)
        self.assertTrue(chunk.exhausted)
        self.assertEqual(len(chunk.actions), 50)  # The full prediction stays in evidence.
        with self.assertRaises(StopIteration):
            chunk.take(110.)
        next_chunk = OrderedChunk({**request, 'source_time': 109.}, reply, 'a'*64, execution_steps=10)
        self.assertEqual(next_chunk.take(110.)[1]['action_index'], 0)

    def test_ready_uses_checkpoint_action_steps_and_rejects_invalid_values(self):
        for steps in (10, 50, None, 0, 51, True, 10.5):
            policy = object.__new__(StreamPolicy)
            policy.expected_sha, policy.ready = None, False
            event = {'status':'ready', 'schema':'fr5-stream-ready-v1',
                     'checkpoint_sha256':'a'*64, 'policy_contract':NAME,
                     'state_shape':[8], 'action_shape':[50,7], 'n_action_steps':steps}
            policy.receive = Mock(return_value=event)
            if type(steps) is int and 1 <= steps <= 50:
                policy.wait_ready()
                self.assertEqual(policy.n_action_steps, steps)
            else:
                with self.assertRaises(ValueError):
                    policy.wait_ready()
                self.assertFalse(policy.ready)

    def test_recorded_failure_short_closure_is_not_erased_by_latency(self):
        request, reply = example()
        # The real failed trial's chunk 12 closed for its first ~8 steps.
        # Source-time trimming began at index 11 and erased this whole action.
        for row in reply['actions'][:8]:
            row[6]=.757
        chunk=OrderedChunk(request,reply,'a'*64)
        outputs=[chunk.take(101.8+i/6)[0][6] for i in range(50)]
        self.assertEqual(outputs[:8],[.76]*8)
        self.assertEqual(outputs[8:],[1.]*42)

    def test_gripper_device_projection_keeps_raw_prediction_for_audit(self):
        request, reply = example()
        reply['actions'][0][6] = 1.08
        target, evidence = OrderedChunk(request, reply, 'a'*64).take(100.+1/6)
        self.assertEqual(target[6], 1.)
        self.assertEqual(evidence['raw_action'][6],1.08)
        self.assertTrue(evidence['decoded']['gripper_projected_to_device_range'])

    def test_run_records_submitted_and_effective_gripper_targets_separately(self):
        request, reply = example()
        request['source_time'] = time.monotonic()
        reply['actions'][0][6] = .75
        chunk = OrderedChunk(request, reply, 'a'*64, execution_steps=10)
        policy = Mock(checkpoint='/model', ready=True, checkpoint_sha='a'*64,
                      n_action_steps=10, training_data_kind='captured',
                      observation_context_profile='none')
        robot = Mock(is_connected=True)
        robot.get_sample.return_value = {'robot_state': {
            'tool': 0, 'user': 0, 'tcp': [0.] * 6, 'flange': [0.] * 6}}
        recorder = Mock(unsafe=True)
        recorder.finish.return_value = {'task_success_verified': False}
        controller = Mock()
        controller.snapshot.return_value = {'status': 'running', 'failure': None,
                                            'gripper_target_percent': 76, 'gripper_commands': 0}
        controller.submit_ordered.return_value = 1
        controller.target_acknowledged.return_value = False
        controller.close.return_value = {'status': 'closed', 'failure': None,
                                         'last_issued_monotonic': None,
                                         'pending_callbacks': 0, 'workers_alive': []}
        inference = Mock(count=1)
        inference.latest.return_value = chunk
        inference.thread.is_alive.return_value = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'episode'
            with patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', Path(directory)), \
                 patch('socket.socket.connect', side_effect=AssertionError('offline test attempted a connection')), \
                 patch('lerobot_robot_fairino.robot_fairino.FairinoFR5', return_value=robot), \
                 patch('lerobot_robot_fairino.stream_recording.StreamEvaluationRecording', return_value=recorder), \
                 patch('lerobot_robot_fairino.servo_control.StreamServoController', return_value=controller), \
                 patch('lerobot_robot_fairino.stream_runtime.InferenceWorker', return_value=inference):
                scene = {'table_plane': {'normal': [0., 0., 1.], 'offset_mm': 0.}}
                run('/model', None, 'test', root, execute=True, loaded_policy=policy, seconds=.02,
                    joint_scene=scene)
            recorded = json.loads((root / 'submitted-targets.jsonl').read_text())
            scope = json.loads((root / 'scope.json').read_text())
        self.assertEqual(controller.submit_ordered.call_args.args[0][6], .75)
        self.assertEqual(recorded['target'][6], .75)
        self.assertEqual(recorded['raw_action'][6], .75)
        self.assertEqual(recorded['effective_gripper_target_percent'], 76)
        self.assertEqual(scope['gripper_deadband_percent'], 1)
        self.assertEqual(scope['joint_prefix_preflight_scene'], scene)

    def test_wrong_request_checkpoint_or_legacy_contract_is_rejected(self):
        for key, value in [('id','old-frame'),('checkpoint_sha256','b'*64),
                           ('policy_contract','fr5-whole-task-v3'),('actions',[[0.]*10]*8)]:
            request, reply = example()
            reply[key] = value
            with self.assertRaises(ValueError):
                validate_reply(reply, request, 'a'*64)

    def test_ready_cannot_silently_accept_legacy_weights(self):
        policy = object.__new__(StreamPolicy)
        policy.expected_sha, policy.ready = None, False
        policy.receive = Mock(return_value={'status':'ready', 'schema':'fr5-stream-ready-v1',
            'checkpoint_sha256':'a'*64, 'policy_contract':NAME,
            'state_shape':[18], 'action_shape':[8,10]})
        with self.assertRaises(ValueError):
            policy.wait_ready()
        self.assertFalse(policy.ready)

    def test_worker_waits_for_consumption_before_another_prediction(self):
        request, reply=example()
        request.update(source_clock='host_monotonic',rgb_jpeg_base64='AA==')
        policy=Mock(checkpoint_sha='a'*64,n_action_steps=50)
        policy.predict.return_value=reply
        recorder=Mock()
        with tempfile.TemporaryDirectory() as directory, \
             patch('lerobot_robot_fairino.stream_runtime.make_request',return_value=request):
            worker=InferenceWorker(policy,recorder,'抓杯',directory)
            worker.start()
            try:
                deadline=time.monotonic()+2
                while worker.latest() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertIsNotNone(worker.latest())
                self.assertEqual(policy.predict.call_count,1)
                self.assertFalse(worker.requested.is_set())
                chunk=worker.latest()
                for i in range(50):chunk.take(101.+i/6)
                after=time.monotonic()
                worker.request_next(after)
                deadline=time.monotonic()+2
                while worker.latest() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(policy.predict.call_count,2)
                self.assertEqual(worker.latest().cursor,0)
                self.assertEqual(len(list(Path(directory).glob('chunk-*.json'))),2)
                self.assertEqual(recorder.observation.call_args_list[0].kwargs,{})
                self.assertEqual(recorder.observation.call_args_list[1].kwargs['after_monotonic'],after)
                self.assertIs(recorder.observation.call_args_list[1].kwargs['stop_event'],worker.stopped)
                self.assertEqual(recorder.record_prediction.call_args.kwargs['metadata']['observation_after_monotonic'],after)
            finally:
                worker.stop();worker.thread.join(2)

    def test_stop_during_inference_cannot_publish_a_late_chunk(self):
        request, reply=example()
        request.update(source_clock='host_monotonic',rgb_jpeg_base64='AA==')
        entered, release=threading.Event(),threading.Event()
        def predict(_):
            entered.set();release.wait(2);return reply
        policy=Mock(checkpoint_sha='a'*64,n_action_steps=50,predict=Mock(side_effect=predict))
        recorder=Mock()
        with tempfile.TemporaryDirectory() as directory, \
             patch('lerobot_robot_fairino.stream_runtime.make_request',return_value=request):
            worker=InferenceWorker(policy,recorder,'抓杯',directory)
            worker.start()
            self.assertTrue(entered.wait(2))
            worker.stop();release.set();worker.thread.join(2)
            self.assertIsNone(worker.latest())
            self.assertEqual(worker.count,0)
            recorder.record_prediction.assert_not_called()
            self.assertEqual(list(Path(directory).glob('chunk-*.json')),[])

    def test_observation_wait_and_prediction_have_separate_timestamps(self):
        request, reply = example()
        request.update(source_clock='host_monotonic', rgb_jpeg_base64='AA==')
        observing, release_observation = threading.Event(), threading.Event()
        predicting, release_prediction = threading.Event(), threading.Event()

        def observe():
            observing.set()
            if not release_observation.wait(2):
                raise TimeoutError('test observation not released')
            return object()

        def predict(actual_request):
            self.assertIs(actual_request, request)
            predicting.set()
            if not release_prediction.wait(2):
                raise TimeoutError('test prediction not released')
            return reply

        policy = Mock(checkpoint_sha='a'*64, n_action_steps=50,
                      predict=Mock(side_effect=predict))
        recorder = Mock()
        recorder.observation.side_effect = observe
        with tempfile.TemporaryDirectory() as directory, \
             patch('lerobot_robot_fairino.stream_runtime.make_request', return_value=request):
            worker = InferenceWorker(policy, recorder, '抓杯', directory)
            worker.start()
            try:
                self.assertTrue(observing.wait(2))
                during_observation = time.monotonic()
                self.assertFalse(predicting.is_set())
                release_observation.set()
                self.assertTrue(predicting.wait(2))
                during_prediction = time.monotonic()
                release_prediction.set()
                deadline = time.monotonic()+2
                while worker.latest() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNotNone(worker.latest())
                metadata = recorder.record_prediction.call_args.kwargs['metadata']
                audit = json.loads((Path(directory)/'chunk-00000.json').read_text())
                keys = ('requested_monotonic', 'observation_ready_monotonic',
                        'predict_started_monotonic', 'received_monotonic')
                self.assertEqual({key: audit[key] for key in keys},
                                 {key: metadata[key] for key in keys})
                requested, ready, started, received = (metadata[key] for key in keys)
                self.assertLessEqual(requested, during_observation)
                self.assertLessEqual(during_observation, ready)
                self.assertLessEqual(ready, started)
                self.assertLessEqual(started, during_prediction)
                self.assertLessEqual(during_prediction, received)
                self.assertGreaterEqual(ready-requested, ready-during_observation)
                self.assertGreaterEqual(received-started, received-during_prediction)
                recorder.observation.assert_called_once_with()
            finally:
                release_observation.set()
                release_prediction.set()
                worker.stop(); worker.thread.join(2)

    def test_stop_during_post_ack_observation_prevents_next_inference(self):
        request, reply=example()
        request.update(source_clock='host_monotonic',rgb_jpeg_base64='AA==')
        entered, release, external_stop=threading.Event(),threading.Event(),threading.Event()
        def observe(**kwargs):
            if kwargs:
                entered.set()
                release.wait(2)
            return object()
        policy=Mock(checkpoint_sha='a'*64,n_action_steps=50)
        policy.predict.return_value=reply
        recorder=Mock()
        recorder.observation.side_effect=observe
        with tempfile.TemporaryDirectory() as directory, \
             patch('lerobot_robot_fairino.stream_runtime.make_request',return_value=request):
            worker=InferenceWorker(policy,recorder,'抓杯',directory,stop_event=external_stop)
            worker.start()
            try:
                deadline=time.monotonic()+2
                while worker.latest() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertIsNotNone(worker.latest())
                worker.request_next(time.monotonic())
                self.assertTrue(entered.wait(2))
                external_stop.set()
                release.set()
                worker.thread.join(2)
                self.assertFalse(worker.thread.is_alive())
                self.assertEqual(policy.predict.call_count,1)
                self.assertEqual(recorder.record_prediction.call_count,1)
            finally:
                release.set()
                worker.stop();worker.thread.join(2)
                recorder.wake_observation_waiters.assert_called()

    def test_post_ack_device_failure_reaches_worker_caller(self):
        request, reply=example()
        request.update(source_clock='host_monotonic',rgb_jpeg_base64='AA==')
        policy=Mock(checkpoint_sha='a'*64,n_action_steps=50)
        policy.predict.return_value=reply
        recorder=Mock()
        recorder.observation.side_effect=[object(), RuntimeError('camera failed')]
        with tempfile.TemporaryDirectory() as directory, \
             patch('lerobot_robot_fairino.stream_runtime.make_request',return_value=request):
            worker=InferenceWorker(policy,recorder,'抓杯',directory)
            worker.start()
            try:
                deadline=time.monotonic()+2
                while worker.latest() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertIsNotNone(worker.latest())
                worker.request_next(time.monotonic())
                worker.thread.join(2)
                self.assertFalse(worker.thread.is_alive())
                with self.assertRaisesRegex(RuntimeError,'camera failed'):
                    worker.latest()
                self.assertEqual(policy.predict.call_count,1)
            finally:
                worker.stop();worker.thread.join(2)


class GripperReobservationTests(unittest.TestCase):
    def exercise(self, *, profile='none', gripper=.76, stop_after=2, delayed_completion=False,
                 initial_hold=0, execution_steps=3, stop_inflight=None):
        from fr5_runtime.tests.test_servo_control import FakeNetwork, StateStream, wait_until
        from lerobot_robot_fairino.servo_control import StreamServoController

        stream, network = StateStream(), FakeNetwork()
        stream.set(joints=[0.] * 6)
        network.stream, network.clock = stream, time.monotonic
        network.complete_gripper = not delayed_completion and stop_inflight is None
        if stop_inflight == 'operator':
            network.after['MoveGripper'] = lambda: signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        ack_allowed = threading.Event()
        ack_waiting = threading.Event()
        if not delayed_completion:
            ack_allowed.set()
        observations, controller_holder, checks = [], [], {}
        cup = {'uv': [100, 200], 'base_xyz_mm': [0., 0., 0.], 'table_normal': [0., 0., 1.],
               'observed_monotonic': time.monotonic() - .1, 'camera_serial': 'fake'}
        recorder = Mock(unsafe=True, scope={'initial_cup': cup})
        recorder.finish.return_value = {'task_success_verified': False}

        def observe(**kwargs):
            now = time.monotonic()
            index = len(observations)
            snapshot = controller_holder[0].snapshot()
            observations.append({'after': kwargs.get('after_monotonic'), 'at': now,
                                 'snapshot': snapshot})
            request, _ = example()
            request.update(id=f'observation-{index}', source_time=now - (2.3 if delayed_completion and not index else 0),
                           state=[0.] * 6 + [stream.latest(.25)['gripper_position'] / 100., 1.],
                           observation_context_profile=profile)
            if profile == CUP_TCP_HISTORY_PROFILE:
                request['cup_base_xyz_mm'] = cup['base_xyz_mm']
            return request

        recorder.observation.side_effect = observe
        replies = []

        def predict(request):
            index = len(replies)
            actions = [[math.radians(i / 3 + index / 10)] * 6 +
                       [1. if index == 0 and i < initial_hold else gripper] for i in range(50)]
            if profile == CUP_TCP_HISTORY_PROFILE:
                actions = [[.1597, .0713, .0958, 0., 0., 0., gripper] for _ in range(50)]
            reply = {'schema': 'fr5-stream-reply-v1', 'id': request['id'],
                     'checkpoint_sha256': 'a' * 64, 'policy_contract': NAME,
                     'observation_context_profile': profile, 'actions': actions}
            if profile == CUP_TCP_HISTORY_PROFILE:
                reply['wire_action_representation'] = CUP_RELATIVE_WIRE
            replies.append(reply)
            return reply

        policy = Mock(checkpoint='/model', ready=True, checkpoint_sha='a' * 64,
                      n_action_steps=10, training_data_kind='captured',
                      observation_context_profile=profile)
        policy.predict.side_effect = predict
        robot = Mock(is_connected=True)
        robot.get_sample.return_value = {'robot_state': {
            'tool': 0, 'user': 0, 'tcp': [0.] * 6, 'flange': [0.] * 6}}

        class Controller(StreamServoController):
            def __init__(self, *_):
                super().__init__(stream, lambda: True, recorder.command_callback,
                                 rpc_factory=network.factory)
                self.submissions = 0
                controller_holder.append(self)

            def submit_ordered(self, *args, **kwargs):
                token = super().submit_ordered(*args, **kwargs)
                self.submissions += 1
                if self.submissions == stop_after:
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                return token

            def target_acknowledged(self, token):
                acknowledged = super().target_acknowledged(token)
                if token == 1 and acknowledged and not ack_allowed.is_set():
                    ack_waiting.set()
                    return False
                return acknowledged

        def prepare(chunk):
            if chunk.initial_cup is not None:
                chunk.cup_actions = [{'target': [0.] * 6, 'detail': {}} for _ in chunk.actions]
                chunk.cup_clearance = {'passed': True}

        def complete_gripper():
            checks['sent'] = wait_until(lambda: bool(network.matching('MoveGripper')), timeout=3)
            if not checks['sent']:
                return
            checks['observations_before_completion'] = len(observations)
            checks['ack_before_completion'] = controller_holder[0].target_acknowledged(1)
            time.sleep(.85)
            stream.set(gripper_position=round(gripper * 100), gripper_motiondone=1,
                       received_monotonic=time.monotonic())
            checks['ack_waiting'] = ack_waiting.wait(2)
            checks['observations_before_ack'] = len(observations)
            ack_allowed.set()

        completion_thread = threading.Thread(target=complete_gripper, daemon=True) if delayed_completion else None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'episode'
            scene = {'table_plane': {'normal': [0., 0., 1.], 'offset_mm': 0.}}
            location = {'status': 'found', 'cup_uv': cup['uv'], 'camera_serial': 'fake',
                        'capture_timestamp_monotonic': cup['observed_monotonic'], 'source': 'fake'}
            with patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', Path(directory)), \
                 patch('socket.socket.connect', side_effect=AssertionError('offline test attempted a connection')), \
                 patch('socket.socket.connect_ex', side_effect=AssertionError('offline test attempted a connection')), \
                 patch('lerobot_robot_fairino.robot_fairino.FairinoFR5', return_value=robot), \
                 patch('lerobot_robot_fairino.stream_recording.StreamEvaluationRecording', return_value=recorder), \
                 patch('lerobot_robot_fairino.servo_control.StreamServoController', Controller), \
                 patch('lerobot_robot_fairino.stream_runtime.make_request', side_effect=lambda sample, *_args, **_kwargs: sample), \
                 patch.object(OrderedChunk, 'prepare', prepare), \
                 patch('fr5_jev.cup_localization.snapshot', return_value=location), \
                 patch('lerobot_robot_fairino.cup_relative_runtime.detect_initial_cup', return_value=cup), \
                 patch('lerobot_camera_gemini.timing.sample_clock', side_effect=lambda _sample: (time.monotonic(), None, None)):
                if completion_thread:
                    completion_thread.start()
                try:
                    run('/model', None, 'test', root, execute=True, loaded_policy=policy,
                        seconds=.5 if stop_inflight == 'duration' else 5,
                        joint_scene=scene, execution_steps=execution_steps)
                finally:
                    ack_allowed.set()
                    if completion_thread:
                        completion_thread.join(4)
            scope = json.loads((root / 'scope.json').read_text())
            submitted = [json.loads(line) for line in (root / 'submitted-targets.jsonl').read_text().splitlines()]
            acknowledged = [json.loads(line) for line in (root / 'acknowledged-targets.jsonl').read_text().splitlines()]
            audits = [json.loads(path.read_text()) for path in sorted(root.glob('chunk-*.json'))]
            checks['finish_reason'] = recorder.finish.call_args.args[0]
        return scope, submitted, acknowledged, audits, replies, observations, network, checks

    def test_completed_native_gripper_reobserves_after_ack_and_keeps_tail(self):
        scope, submitted, ack, audits, replies, observations, network, checks = self.exercise(
            delayed_completion=True)
        self.assertTrue(checks['sent'])
        self.assertFalse(checks['ack_before_completion'])
        self.assertTrue(checks['ack_waiting'])
        self.assertEqual(checks['observations_before_completion'], 1)
        self.assertEqual(checks['observations_before_ack'], 1)
        self.assertEqual([row['action_index'] for row in submitted], [0, 0])
        self.assertEqual([row['request_id'] for row in submitted], ['observation-0', 'observation-1'])
        self.assertEqual(submitted[1]['raw_action'], replies[1]['actions'][0])
        self.assertEqual(audits[0]['reply']['actions'], replies[0]['actions'])
        self.assertEqual(len(audits[0]['reply']['actions'][1:]), 49)
        self.assertEqual(ack[0]['prefix_consumed_steps'], 1)
        self.assertEqual(ack[0]['prefix_retired_cause'], 'native_gripper_command_completed')
        self.assertEqual((ack[0]['gripper_commands_before'], ack[0]['gripper_commands_after']), (0, 1))
        self.assertEqual(observations[1]['after'], ack[0]['acknowledged_monotonic'])
        self.assertGreaterEqual(observations[1]['at'], ack[0]['acknowledged_monotonic'])
        self.assertEqual(observations[1]['snapshot']['ordered_acknowledged_token'], 1)
        self.assertEqual(observations[1]['snapshot']['gripper_completion_feedback']['target_percent'], 76)
        self.assertEqual(len(network.matching('MoveGripper')), 1)
        self.assertEqual(scope['observation_rule'], 'prefix_or_completed_native_gripper_command')

    def test_deadband_keeps_full_prefix_then_reobserves(self):
        scope, submitted, ack, audits, replies, observations, network, _ = self.exercise(
            gripper=.99, stop_after=4)
        self.assertEqual([row['action_index'] for row in submitted], [0, 1, 2, 0])
        self.assertEqual([row['request_id'] for row in submitted],
                         ['observation-0'] * 3 + ['observation-1'])
        self.assertEqual([row['raw_action'] for row in submitted[:3]], replies[0]['actions'][:3])
        self.assertEqual(network.matching('MoveGripper'), [])
        self.assertEqual([row['prefix_consumed_steps'] for row in ack], [1, 2, 3])
        self.assertEqual([row['prefix_retired_cause'] for row in ack],
                         [None, None, 'execution_prefix_completed'])
        self.assertEqual(observations[1]['after'], ack[-1]['acknowledged_monotonic'])
        self.assertEqual(audits[0]['reply']['actions'], replies[0]['actions'])
        self.assertEqual(scope['observation_rule'], 'prefix_or_completed_native_gripper_command')

    def test_gripper_command_in_middle_retires_after_that_ack(self):
        _, submitted, ack, audits, replies, observations, network, _ = self.exercise(
            initial_hold=1, stop_after=3)
        self.assertEqual([row['action_index'] for row in submitted], [0, 1, 0])
        self.assertEqual([row['prefix_consumed_steps'] for row in ack], [1, 2])
        self.assertEqual([row['prefix_retired_cause'] for row in ack],
                         [None, 'native_gripper_command_completed'])
        self.assertEqual([(row['gripper_commands_before'], row['gripper_commands_after']) for row in ack],
                         [(0, 0), (0, 1)])
        self.assertEqual(audits[0]['reply']['actions'], replies[0]['actions'])
        self.assertEqual(observations[1]['after'], ack[1]['acknowledged_monotonic'])
        self.assertEqual(len(network.matching('MoveGripper')), 1)

    def test_deadband_uses_the_configured_prefix_length(self):
        _, submitted, ack, _, _, observations, network, _ = self.exercise(
            gripper=.99, stop_after=3, execution_steps=2)
        self.assertEqual([row['action_index'] for row in submitted], [0, 1, 0])
        self.assertEqual([row['prefix_consumed_steps'] for row in ack], [1, 2])
        self.assertEqual(ack[-1]['prefix_retired_cause'], 'execution_prefix_completed')
        self.assertEqual(observations[1]['after'], ack[-1]['acknowledged_monotonic'])
        self.assertEqual(network.matching('MoveGripper'), [])

    def test_stop_or_duration_during_gripper_wait_does_not_reobserve(self):
        for stop_inflight, expected in (('operator', 'operator_stop'),
                                        ('duration', 'explicit_duration_limit')):
            with self.subTest(stop_inflight=stop_inflight):
                _, submitted, ack, _, _, observations, network, checks = self.exercise(
                    stop_after=0, stop_inflight=stop_inflight)
                self.assertEqual(checks['finish_reason'], expected)
                self.assertEqual([row['action_index'] for row in submitted], [0])
                self.assertEqual(submitted[0]['prefix_consumed_steps'], 1)
                self.assertEqual(ack, [])
                self.assertEqual(len(observations), 1)
                self.assertEqual(len(network.matching('MoveGripper')), 1)

    def test_cup_and_tcp_profiles_keep_the_full_prefix_after_gripper_command(self):
        for profile in (CUP_HISTORY_PROFILE, CUP_TCP_HISTORY_PROFILE):
            with self.subTest(profile=profile):
                scope, submitted, ack, audits, replies, observations, network, _ = self.exercise(
                    profile=profile, gripper=.90, stop_after=4)
                self.assertEqual([row['action_index'] for row in submitted], [0, 1, 2, 0])
                self.assertEqual([row['request_id'] for row in submitted],
                                 ['observation-0'] * 3 + ['observation-1'])
                self.assertEqual(len(network.matching('MoveGripper')), 1)
                self.assertEqual([row['prefix_retired_cause'] for row in ack],
                                 [None, None, 'execution_prefix_completed'])
                self.assertEqual(observations[1]['after'], ack[-1]['acknowledged_monotonic'])
                self.assertEqual(audits[0]['reply']['actions'], replies[0]['actions'])
                self.assertEqual(scope['observation_rule'], 'prefix')


class JointPrefixExecutionTests(unittest.TestCase):
    scene = {'table_plane': {'normal': [0., 0., 1.], 'offset_mm': 0.}}

    def test_entire_consumed_prefix_is_checked_and_actions_are_preserved(self):
        from fr5_lerobot.tests.test_preflight_jev_pi05_actions import FakeRPC
        request, reply = example()
        reply['actions'][0][6] = 1.08
        original = json.loads(json.dumps(reply))
        policy = Mock(checkpoint_sha='a' * 64, n_action_steps=10, observation_context_profile='none')
        policy.predict.return_value = reply
        rpc = FakeRPC()
        with tempfile.TemporaryDirectory() as root, \
                patch('lerobot_robot_fairino.stream_runtime.make_request', return_value=request), \
                patch('lerobot_robot_fairino.control_transport.FR5RPC', return_value=rpc):
            worker = InferenceWorker(policy, Mock(), '抓取物体', root, joint_scene=self.scene)
            worker.start()
            try:
                deadline = time.monotonic() + 2.
                while worker.latest() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                chunk = worker.latest()
                self.assertIsNotNone(chunk)
                self.assertEqual(chunk.actions, original['actions'])
                self.assertTrue(chunk.joint_preflight['passed'])
                self.assertEqual(len(chunk.joint_preflight['segments']), 10)
                self.assertEqual({method for method, _ in rpc.calls},
                                 {'GetJointSoftLimitDeg', 'GetForwardKin'})
                audit = json.loads((Path(root) / 'chunk-00000.json').read_text())
                self.assertTrue(audit['joint_prefix_preflight']['passed'])
                self.assertEqual(audit['reply'], original)
                self.assertLessEqual(audit['received_monotonic'], audit['prefix_precheck_started_monotonic'])
                self.assertLessEqual(audit['prefix_precheck_started_monotonic'],
                                     audit['prefix_precheck_finished_monotonic'])
            finally:
                worker.stop()
                worker.thread.join(2.)

    def test_intermediate_table_violation_rejects_prefix_and_preserves_report(self):
        from fr5_lerobot.tests.test_preflight_jev_pi05_actions import FakeRPC
        request, reply = example()
        reply['actions'][0][0] = math.radians(1.)
        policy = Mock(checkpoint_sha='a' * 64, n_action_steps=10, observation_context_profile='none')
        policy.predict.return_value = reply
        recorder = Mock()
        with tempfile.TemporaryDirectory() as root, \
                patch('lerobot_robot_fairino.stream_runtime.make_request', return_value=request), \
                patch('lerobot_robot_fairino.control_transport.FR5RPC',
                      return_value=FakeRPC(middle_below_table=True)):
            worker = InferenceWorker(policy, recorder, '抓取物体', root, joint_scene=self.scene)
            worker.start()
            worker.thread.join(2.)
            self.assertFalse(worker.thread.is_alive())
            self.assertIsNone(worker.current)
            self.assertEqual(worker.count, 0)
            self.assertIn('Joint prefix preflight failed', str(worker.error))
            recorder.record_prediction.assert_called_once()
            audit = json.loads((Path(root) / 'chunk-00000.json').read_text())
            self.assertEqual(audit['status'], 'rejected_before_execution')
            self.assertFalse(audit['joint_prefix_preflight']['passed'])
            self.assertEqual(audit['joint_prefix_preflight']['failures'][0]['check'],
                             'tool_table_clearance')
            self.assertEqual(audit['reply'], reply)

    def test_unsettled_gripper_rejects_before_any_preflight_rpc(self):
        request, reply = example()
        request['state'][7] = 0.
        chunk = OrderedChunk(request, reply, 'a' * 64, execution_steps=10, joint_scene=self.scene)
        with patch('lerobot_robot_fairino.control_transport.FR5RPC') as factory:
            with self.assertRaisesRegex(RuntimeError, 'No settled gripper feedback'):
                chunk.prepare()
        factory.assert_not_called()
        self.assertEqual(chunk.cursor, 0)

    def test_missing_or_invalid_table_rejects_execution_before_robot_connection(self):
        for scene in (None, {'table_plane': {'normal': [0., 0., 0.], 'offset_mm': 0.}}):
            policy = Mock(checkpoint='/model', ready=True, checkpoint_sha='a' * 64,
                          n_action_steps=10, training_data_kind='captured',
                          observation_context_profile='state-history-physical-v1')
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as parent, \
                    patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', Path(parent)), \
                    patch('lerobot_robot_fairino.robot_fairino.FairinoFR5') as robot:
                root = Path(parent) / 'episode'
                with self.assertRaises(ValueError):
                    run('/model', None, '抓取物体', root, execute=True, loaded_policy=policy,
                        joint_scene=scene)
                robot.assert_not_called()
                self.assertFalse(root.exists())

    def test_non_flange_coordinates_reject_before_recording_or_servo_start(self):
        for tool, user, tcp in ((1, 0, [0.] * 6), (0, 1, [0.] * 6),
                               (0, 0, [2., 0., 0., 0., 0., 0.])):
            policy = Mock(checkpoint='/model', ready=True, checkpoint_sha='a' * 64,
                          n_action_steps=10, training_data_kind='captured',
                          observation_context_profile='none')
            robot = Mock(is_connected=True)
            robot.get_sample.return_value = {'robot_state': {
                'tool': tool, 'user': user, 'tcp': tcp, 'flange': [0.] * 6}}
            with self.subTest(tool=tool, user=user, tcp=tcp), tempfile.TemporaryDirectory() as parent, \
                    patch('fr5_jev.stream_storage.LOCAL_STREAM_BASE', Path(parent)), \
                    patch('lerobot_robot_fairino.robot_fairino.FairinoFR5', return_value=robot), \
                    patch('lerobot_robot_fairino.servo_control.StreamServoController') as controller:
                with self.assertRaisesRegex(ValueError, 'tool0/user0 flange coordinates'):
                    run('/model', None, '抓取物体', Path(parent) / 'episode', execute=True,
                        loaded_policy=policy, joint_scene=self.scene)
                controller.assert_not_called()


if __name__ == '__main__':
    unittest.main()
