"""Console routing/cancellation tests. No physical device connection or action."""
from collections import deque
from copy import deepcopy
from contextlib import nullcontext
from http.server import ThreadingHTTPServer
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import urllib.request
import urllib.error

from fr5_runtime import console as fr5_console
from fr5_runtime.console import Console, Handler, byte_range, start_waypoints, validate_start


class ConsoleTests(unittest.TestCase):
    def request(self):
        return {'cup_uv':[1238,688], 'hold_seconds':5, 'repetitions':1, 'start_variant':'alternating'}

    def test_invalid_selection_or_duration_cannot_schedule_work(self):
        with tempfile.TemporaryDirectory() as directory, patch('fr5_runtime.console.threading.Thread') as thread:
            console = Console(state_dir=directory, refresh=False)
            for changes in ({'cup_uv':[0,1080]}, {'cup_uv':[float('nan'),30]},
                            {'hold_seconds':-1}, {'repetitions':True}, {'start_variant':'unknown'}):
                with self.assertRaises(ValueError):
                    console.start({**self.request(), **changes})
            thread.assert_not_called()

    def test_auto_target_needs_no_operator_pixel_and_ignores_old_selection(self):
        request = validate_start({'target_mode':'auto', 'cup_uv':[1238,688]})
        self.assertEqual(request['target_mode'], 'auto')
        self.assertIsNone(request['cup_uv'])
        self.assertEqual(validate_start({})['target_mode'], 'auto')
        with self.assertRaises(ValueError):
            validate_start({'target_mode':'manual'})

    def test_start_localization_uses_fresh_observation_not_preview_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            console = Console(state_dir=directory, refresh=False)
            console.vision = {'status':'found','cup_uv':[1238,688]}
            fresh = {'status':'found','cup_uv':[800,500], 'message':'已定位'}
            with patch('fr5_jev.cup_localization.from_observation', return_value=fresh) as locate:
                point, evidence = console.resolve_target(validate_start({}), Path(directory))
            locate.assert_called_once_with(Path(directory))
            self.assertEqual(point, [800,500])
            self.assertEqual(evidence, fresh)
            self.assertEqual(json.loads((Path(directory)/'cup-localization.json').read_text()), fresh)
            ambiguous = {'status':'ambiguous','cup_uv':None, 'message':'多个候选'}
            with patch('fr5_jev.cup_localization.from_observation', return_value=ambiguous):
                with self.assertRaisesRegex(RuntimeError, '多个候选'):
                    console.resolve_target(validate_start({}), Path(directory))
            self.assertIsNone(console.controller)

    def test_only_one_active_episode_and_start_variants_really_alternate(self):
        with tempfile.TemporaryDirectory() as directory, patch('fr5_runtime.console.threading.Thread'):
            console = Console(state_dir=directory, refresh=False)
            console.start(self.request())
            first = console.job['id']
            self.assertEqual(console.job['request']['resolved_start_variant'], 'left')
            with self.assertRaises(RuntimeError):
                console.start(self.request())
            console.update(state='completed')
            (Path(directory)/(first+'-archive.json')).write_text('{}')
            (Path(directory)/(first+'-auxiliary.json')).write_text('{}')
            (Path(directory)/(first+'-archive-recovery-history.json')).write_text(
                json.dumps({'id':first, 'kind':'collection'}))
            (Path(directory)/'cup-ui-corrupt.json').write_text('{')
            console.start(self.request())
            self.assertEqual(console.job['request']['resolved_start_variant'], 'right')

    def test_stop_reaches_current_controller_without_waiting_for_worker(self):
        with tempfile.TemporaryDirectory() as directory, patch('fr5_runtime.console.threading.Thread'):
            console = Console(state_dir=directory, refresh=False)
            controller = Mock()
            console.controller = controller
            console.stop()
            controller.stop.assert_not_called()
            console.start(self.request())
            console.controller = controller
            console.stop()
            self.assertTrue(console.stop_event.is_set())
            self.assertEqual(console.job['state'], 'stopping')
            controller.stop.assert_called_once()

    def test_native_start_variation_changes_actual_targets_and_preserves_attitude(self):
        pose = [0.,-700.,90.,90.,0.,-65.]
        geometry = {'route_model':{'contact_pose':pose, 'clear_pose':[70.,-670.,90.,90.,0.,-65.],
                                  'tool_axis_base':[-.906,-.423,0.]}}
        left = start_waypoints(pose, geometry, 'left')
        right = start_waypoints(pose, geometry, 'right')
        self.assertNotEqual(left[-1][:3], right[-1][:3])
        self.assertEqual(left[0][:2], pose[:2])
        self.assertEqual(left[-1][3:], pose[3:])
        self.assertGreaterEqual(left[0][2], max(left[-1][2], right[-1][2]))

    def test_range_streams_exact_requested_bytes(self):
        self.assertEqual(byte_range(None, 100), (0,99,False))
        self.assertEqual(byte_range('bytes=10-19', 100), (10,19,True))
        self.assertEqual(byte_range('bytes=-5', 100), (95,99,True))
        self.assertEqual(byte_range('bytes=90-', 100), (90,99,True))
        for value in ('bytes=100-110', 'bytes=10-2', 'bytes=-0', 'bytes=0-1,4-5'):
            with self.assertRaises(ValueError):
                byte_range(value, 100)

    def test_review_records_a_real_console_input_method(self):
        with tempfile.TemporaryDirectory() as directory:
            console = Console(state_dir=directory, refresh=False)
            console.refresh = Mock()
            with patch('fr5_runtime.console.system.review', return_value={}) as review:
                console.review({'episode':'cup-test','verdict':'success','statement':'整轮成功'})
            review.assert_called_once_with('cup-test','success','整轮成功',input_method='web_console_click')


class StorageFinalizationTests(unittest.TestCase):
    """All examples are synthetic; no device, server, or real API is contacted."""

    def setUp(self):
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.create_connection', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('Offline test cannot contact a device or server'))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.spool = self.base/'spool'
        self.spool.mkdir()
        self.console = Console(state_dir=self.base/'state', refresh=False)
        self.console.refresh = Mock()
        self.name = 'cup-ui-synthetic'
        self.root = self.spool/self.name
        self.scope = {'whole_task':True, 'phase_order':['close','release'],
                      'actions':[{'id':'close', 'cycle':1, 'phase':'close', 'kind':'gripper'},
                                 {'id':'release', 'cycle':1, 'phase':'release', 'kind':'gripper'}],
                      'task_spec':{'repetitions':1, 'hold_seconds':3.}}
        self.session_id = 'a'*32
        self.codec = 'b'*64
        self.protocol = 'c'*64
        self.storage = Mock()
        self.storage.frame_spool = SimpleNamespace(session=self.session_id)
        self.storage.codec_sha256 = self.codec
        self.storage.protocol_sha256 = self.protocol
        self.receipt = {'schema':'fr5-stream-closed-v1', 'session':self.session_id,
                        'accepted_frames':3, 'last_durable_sequence':2,
                        'unacknowledged_frames':0, 'unacknowledged_bytes':0, 'error':None,
                        'codec_sha256':self.codec, 'protocol_sha256':self.protocol,
                        'local_episode':str(self.root)}

    def put_snapshot(self):
        self.root.mkdir(exist_ok=True)
        raw = json.dumps(self.scope).encode()
        (self.root/'scope.json').write_bytes(raw)
        summary = {'whole_task':True, 'scope_sha256':hashlib.sha256(raw).hexdigest(),
                   'outcome':'pending_review', 'failed':None, 'released':True, 'commands':2,
                   'completed_stages':['close','release'], 'completed_cycles':1,
                   'task_spec':self.scope['task_spec'],
                   'continuous_capture':{'capture_complete':True, 'frames':3},
                   'dataset':{'storage_complete':True, 'recorded_frames':3}}
        (self.root/'summary.json').write_text(json.dumps(summary))
        (self.root/'stream-closed.json').write_text(json.dumps(self.receipt))
        journal = [{'executed':False, 'kind':'hold'},
                   *[{'executed':True, 'kind':action['kind'],
                      'teacher':{'cycle':action['cycle'], 'phase':action['phase']}}
                     for action in self.scope['actions']]]
        (self.root/'target-command-journal.json').write_text(json.dumps(journal))
        return summary

    def put_job(self, name=None, state='completed'):
        name = name or self.name
        job = {'id':name, 'episode':name, 'state':state, 'phase':'completed',
               'label':'原始标签', 'message':'原始消息', 'started_at':1}
        fr5_console.write_json(self.console.state_dir/(name+'.json'), job)
        return job

    def collect(self, mode, mutate=None):
        from fr5_jev import whole_task_collection
        fixture = self
        sessions = []
        class FakeWholeTaskSession:
            def __init__(self, root):
                self.root = root
                self.scope = deepcopy(fixture.scope)
                self.scope_sha = hashlib.sha256((root/'scope.json').read_bytes()).hexdigest()
                self.storage = fixture.storage
                self.failed = None
                self.writer = SimpleNamespace(_finished=True)
                self.controller = Mock()
                self.robot = Mock(is_connected=True)
                self._watchdog_stop = threading.Event()
                sessions.append(self)
            def start(self):
                pass
            def run(self):
                if mode == 'action_error':
                    raise RuntimeError('动作反馈异常')
                return self.finish('pending_review', ['synthetic'])
            def finish(self, outcome='interrupted', evidence=None):
                if outcome == 'pending_review':
                    if mutate:
                        mutate(fixture)
                    raise RuntimeError('控制文件未确认')
                raise AssertionError('已完成的 writer 不应重启收尾')
        self.put_snapshot()
        self.console.job = self.put_job(state='running')
        request = {'resolved_start_variant':'current', 'target_mode':'manual',
                   'hold_seconds':3., 'repetitions':1}
        observation = self.spool/'observation'
        observation.mkdir(exist_ok=True)
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch.object(fr5_console.system, 'robot_lease', return_value=nullcontext()), \
             patch.object(fr5_console.system, 'observe', return_value={'directory':str(observation)}), \
             patch.object(whole_task_collection, 'WholeTaskSession', FakeWholeTaskSession), \
             patch.object(whole_task_collection, 'prepare'), \
             patch.object(whole_task_collection, 'record_llm_route_plan'), \
             patch.object(self.console, 'resolve_target', return_value=([1,2], {})), \
             patch('fr5_runtime.console.threading.Thread') as thread:
            self.console._collect(self.name, request, threading.Event())
        return sessions[0], thread

    def test_completed_stream_control_failure_keeps_physical_completion(self):
        session, thread = self.collect('finish_error')
        self.assertEqual((self.console.job['state'], self.console.job['phase']), ('completed','completed'))
        self.assertEqual(self.console.job['storage_status'], 'pending')
        self.assertIn('待保存', self.console.job['label'])
        self.assertIn('确认', self.console.job['message'])
        self.assertIsNone(session.failed)
        session.controller.stop.assert_not_called()
        session.robot.disconnect.assert_called_once()
        self.assertIsNone(self.console.controller)
        self.assertIsNone(self.console.session)
        thread.assert_called_once()
        self.assertEqual(json.loads((self.root/'summary.json').read_text())['outcome'], 'pending_review')

    def test_action_failure_cannot_be_mistaken_for_storage_failure(self):
        session, _ = self.collect('action_error')
        self.assertEqual(self.console.job['state'], 'failed')
        self.assertIn('动作反馈异常', self.console.job['message'])
        session.controller.stop.assert_called_once()
        session.robot.disconnect.assert_called_once()

    def test_finish_exception_with_unacknowledged_stream_is_action_failure(self):
        def incomplete(fixture):
            path = fixture.root/'stream-closed.json'
            receipt = json.loads(path.read_text())
            receipt['unacknowledged_frames'] = 1
            path.write_text(json.dumps(receipt))
        session, thread = self.collect('finish_error', incomplete)
        self.assertEqual(self.console.job['state'], 'failed')
        session.controller.stop.assert_called_once()
        session.robot.disconnect.assert_called_once()
        thread.assert_not_called()

    def test_inconsistent_snapshot_or_unacknowledged_stream_cannot_complete(self):
        self.put_snapshot()
        for file, key, bad in (('summary.json','scope_sha256','0'*64),
                               ('summary.json','completed_stages',['close']),
                               ('summary.json','commands',1), ('summary.json','failed','fault'),
                               ('summary.json','released',False), ('stream-closed.json','unacknowledged_frames',1),
                               ('stream-closed.json','last_durable_sequence',1),
                               ('stream-closed.json','error','fault'),
                               ('stream-closed.json','session','invalid')):
            with self.subTest(file=file, key=key):
                self.put_snapshot()
                path = self.root/file
                value = json.loads(path.read_text())
                value[key] = bad
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    fr5_console.completed_summary(self.root, expected_scope=self.scope, storage=self.storage)

    def test_completion_allows_initial_open_and_planned_noop_targets(self):
        self.put_snapshot()
        journal_path, summary_path = self.root/'target-command-journal.json', self.root/'summary.json'
        journal = json.loads(journal_path.read_text())
        summary = json.loads(summary_path.read_text())
        journal.insert(1, {'executed':True, 'kind':'gripper',
                           'teacher':{'cycle':0, 'phase':'initial_open'}})
        summary['commands'] = 3
        journal_path.write_text(json.dumps(journal))
        summary_path.write_text(json.dumps(summary))
        fr5_console.completed_summary(self.root, expected_scope=self.scope, storage=self.storage)
        journal.pop(2)  # A planned target already reached needs no command.
        summary['commands'] = 2
        journal_path.write_text(json.dumps(journal))
        summary_path.write_text(json.dumps(summary))
        fr5_console.completed_summary(self.root, expected_scope=self.scope, storage=self.storage)
        journal[-1]['teacher']['phase'] = 'unplanned'
        journal_path.write_text(json.dumps(journal))
        with self.assertRaises(ValueError):
            fr5_console.completed_summary(self.root, expected_scope=self.scope, storage=self.storage)

    def test_archive_updates_its_own_job_and_keeps_new_job_unchanged(self):
        self.put_job()
        self.console.job = self.put_job('cup-ui-next', state='running')
        before = deepcopy(self.console.job)
        order = []
        self.storage.finalize_control.side_effect = lambda: order.append('finalize')
        self.storage.materialize.side_effect = lambda: order.append('materialize') or {'verified':True}
        with patch('fr5_runtime.console.data.render_replay'):
            self.console._archive(self.name, self.storage, [], None)
        old = json.loads((self.console.state_dir/(self.name+'.json')).read_text())
        self.assertEqual(order, ['finalize','materialize'])
        self.assertEqual(old['storage_status'], 'saved')
        self.assertIn('整轮确认', old['label'])
        self.assertEqual(self.console.job, before)
        self.assertEqual(json.loads((self.console.state_dir/'cup-ui-next.json').read_text()), before)

    def test_archive_error_keeps_completed_and_blocks_success_review(self):
        self.console.job = self.put_job()
        self.storage.finalize_control.side_effect = RuntimeError('上传未确认')
        self.console._archive(self.name, self.storage, [], None)
        self.assertEqual(self.console.job['state'], 'completed')
        self.assertEqual(self.console.job['phase'], 'completed')
        self.assertEqual(self.console.job['storage_status'], 'failed')
        self.assertIn('保存未完成', self.console.job['message'])
        self.storage.materialize.assert_not_called()
        with patch('fr5_runtime.console.system.review') as review:
            with self.assertRaisesRegex(RuntimeError, '不能确认成功'):
                self.console.review({'episode':self.name, 'verdict':'success', 'statement':'成功'})
            review.assert_not_called()

    def test_old_archive_error_does_not_replace_new_running_job(self):
        self.put_job()
        self.console.job = self.put_job('cup-ui-next', state='running')
        before = deepcopy(self.console.job)
        self.storage.materialize.side_effect = RuntimeError('保存中断')
        self.console._archive(self.name, self.storage, [], None)
        old = json.loads((self.console.state_dir/(self.name+'.json')).read_text())
        self.assertEqual(old['state'], 'completed')
        self.assertEqual(old['storage_status'], 'failed')
        self.assertEqual(self.console.job, before)

    def test_next_round_can_start_while_durable_previous_archive_is_pending(self):
        self.console.job = self.put_job()
        self.console.job['storage_status'] = 'pending'
        self.console.archive_jobs.add(self.name)
        with patch('fr5_runtime.console.threading.Thread'), \
             patch.object(fr5_console.system, 'generated', return_value='cup-ui-next'):
            result = self.console.start({'target_mode':'auto', 'start_variant':'current'})
        self.assertEqual(result['episode'], 'cup-ui-next')
        self.assertEqual(self.console.job['id'], 'cup-ui-next')
        self.assertIn(self.name, self.console.archive_jobs)

    def test_success_review_requires_materialization_for_local_closed_stream(self):
        self.put_snapshot()
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch('fr5_runtime.console.system.review', return_value={}) as review:
            payload = {'episode':self.name, 'verdict':'success', 'statement':'成功'}
            with self.assertRaisesRegex(RuntimeError, '不能确认成功'):
                self.console.review(payload)
            review.assert_not_called()
            self.console.inventory['episodes'] = [{'id':self.name, 'can_review':True}]
            self.console.review(payload)
            review.assert_called_once()

    def test_retry_rebuilds_closed_storage_deduplicates_and_never_moves(self):
        from fr5_jev.stream_storage import StreamStorage
        self.put_snapshot()
        self.put_job()
        self.console.job = self.put_job('cup-ui-next', state='running')
        original = deepcopy(self.console.job)
        controller = Mock()
        self.console.controller = controller
        self.storage.materialize.return_value = {'verified':True}
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch.object(StreamStorage, 'from_closed', create=True, return_value=self.storage) as recover, \
             patch('fr5_runtime.console.threading.Thread') as thread, \
             patch('fr5_runtime.console.data.render_replay'), patch.object(fr5_console.system, 'observe') as observe, \
             patch.object(fr5_console.system, 'robot_lease') as lease:
            first = self.console.retry_save({'episode':self.name})
            second = self.console.retry_save({'episode':self.name})
            self.assertEqual(first['storage_status'], 'pending')
            self.assertIn('正在保存', second['message'])
            thread.assert_called_once()
            recover.assert_not_called()
            thread.call_args.kwargs['target']()
            recover.assert_called_once_with(self.root)
            again = self.console.retry_save({'episode':self.name})
            self.assertEqual(again['storage_status'], 'saved')
            observe.assert_not_called()
            lease.assert_not_called()
            controller.assert_not_called()
            controller.stop.assert_not_called()
            self.assertEqual(controller.mock_calls, [])
        self.assertEqual(self.console.job, original)
        self.assertEqual(json.loads((self.console.state_dir/(self.name+'.json')).read_text())['storage_status'], 'saved')

    def test_retry_after_failure_and_rejects_unsafe_or_incomplete_roots(self):
        from fr5_jev.stream_storage import StreamStorage
        self.put_snapshot()
        self.console.job = self.put_job()
        self.storage.finalize_control.side_effect = [RuntimeError('暂时不可用'), None]
        self.storage.materialize.return_value = {'verified':True}
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch.object(StreamStorage, 'from_closed', create=True, return_value=self.storage) as recover, \
             patch('fr5_runtime.console.threading.Thread') as thread, patch('fr5_runtime.console.data.render_replay'):
            with self.assertRaises(ValueError):
                self.console.retry_save({'episode':'../elsewhere'})
            (self.spool/'linked').symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.console.retry_save({'episode':'linked'})
            self.console.retry_save({'episode':self.name})
            thread.call_args.kwargs['target']()
            self.assertEqual(self.console.job['storage_status'], 'failed')
            self.console.retry_save({'episode':self.name})
            thread.call_args.kwargs['target']()
            self.assertEqual(self.console.job['storage_status'], 'saved')
            self.assertEqual(recover.call_count, 2)
            bad = {**self.receipt, 'unacknowledged_frames':1}
            (self.root/'stream-closed.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                self.console.retry_save({'episode':self.name})

    def test_retry_does_not_race_the_current_session_finalization(self):
        self.put_snapshot()
        self.console.job = self.put_job(state='saving')
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch('fr5_runtime.console.threading.Thread') as thread:
            with self.assertRaisesRegex(RuntimeError, '仍在收尾'):
                self.console.retry_save({'episode':self.name})
            thread.assert_not_called()

    def test_already_materialized_retry_updates_matching_job_without_rescheduling(self):
        self.put_snapshot()
        self.console.job = self.put_job()
        self.console.job['storage_status'] = 'pending'
        fr5_console.write_json(self.console.state_dir/(self.name+'-archive.json'), {'status':'materialized'})
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch('fr5_runtime.console.threading.Thread') as thread:
            result = self.console.retry_save({'episode':self.name})
            self.assertEqual(result['storage_status'], 'saved')
            self.assertEqual(self.console.job['storage_status'], 'saved')
            thread.assert_not_called()

    def test_status_marks_only_locally_closed_unreviewable_idle_episodes_retryable(self):
        self.put_snapshot()
        self.console.inventory['episodes'] = [
            {'id':self.name, 'can_review':False}, {'id':self.name, 'can_review':True},
            {'id':'../outside', 'can_review':False}, {'id':['invalid'], 'can_review':False}]
        with patch.object(fr5_console.system, 'SPOOL', self.spool), \
             patch('fr5_runtime.console.urllib.request.urlopen', side_effect=OSError('offline')):
            rows = self.console.status()['episodes']
            self.assertEqual([row['can_retry_save'] for row in rows], [True, False, False, False])
            self.console.archive_jobs.add(self.name)
            self.assertFalse(self.console.status()['episodes'][0]['can_retry_save'])
            self.console.archive_jobs.clear()
            self.console.job = self.put_job(state='saving')
            self.assertFalse(self.console.status()['episodes'][0]['can_retry_save'])
            self.console.job = None
            bad = {**self.receipt, 'unacknowledged_frames':1}
            (self.root/'stream-closed.json').write_text(json.dumps(bad))
            self.assertFalse(self.console.status()['episodes'][0]['can_retry_save'])

    def test_retry_endpoint_routes_without_running_an_http_server(self):
        handler = Handler.__new__(Handler)
        console = Mock(csrf='test-token')
        console.retry_save.return_value = {'ok':True}
        handler.server = SimpleNamespace(console=console, server_port=8768)
        body = json.dumps({'episode':self.name}).encode()
        handler.headers = {'Host':'127.0.0.1:8768', 'Origin':'http://127.0.0.1:8768',
                           'X-FR5-CSRF':'test-token', 'Content-Length':str(len(body))}
        handler.path, handler.rfile, handler.respond = '/api/retry_save', io.BytesIO(body), Mock()
        handler.do_POST()
        console.retry_save.assert_called_once_with({'episode':self.name})
        handler.respond.assert_called_once_with({'ok':True})
        console.start.assert_not_called()


class RepositionTests(unittest.TestCase):
    """Exercise setup reporting with fake devices and no network access."""

    def setUp(self):
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.create_connection', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('Offline test cannot contact a device or server'))
            guard.start()
            self.addCleanup(guard.stop)

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.plan = Path(temporary.name)
        (self.plan/'scope.json').write_text(json.dumps({'initial_tcp':[0,0,50,0,0,0]}))
        (self.plan/'geometry.json').write_text(json.dumps({'route_model':{
            'contact_pose':[0,0,0,0,0,0], 'clear_pose':[1,2,3,0,0,0],
            'tool_axis_base':[1,0,0]}}))
        self.console = Console(state_dir=self.plan/'state', refresh=False)
        self.stop = threading.Event()
        self.observer = Mock()
        self.observer.stream = object()
        self.observer.is_connected = False
        self.observer.connect.side_effect = lambda: setattr(self.observer, 'is_connected', True)
        self.observer.disconnect.side_effect = lambda: setattr(self.observer, 'is_connected', False)
        self.rpc = Mock()
        self.rpc.call.side_effect = [(0,1,2,3,4,5,6), (0,7,8,9,10,11,12)]
        self.state = {'joints':[0,0,0,0,0,0], 'gripper_position':100,
                      'gripper_motiondone':1, 'gripper_active':1,
                      'gripper_fault':0, 'gripper_fault_id':0, 'gripper_speed':0}
        self.controllers = []
        self.execution_error = None
        test = self

        class FakePolicyController:
            def __init__(self, *, stream, camera_fresh, limits, allow_motion, rpc):
                self.stream, self.rpc = stream, rpc
                self._journal = deque(maxlen=100000)
                self.arm = Mock(side_effect=self._arm)
                self.execute = Mock(side_effect=self._execute)
                self.stop = Mock()
                self.watchdog = Mock()
                test.controllers.append(self)

            def _append(self, target, kind, *, executed):
                row = {'sequence':len(self._journal)+1, 'target':list(target),
                       'kind':kind, 'executed':executed}
                self._journal.append(deepcopy(row))
                return row

            def _arm(self, *, operator_confirmed):
                if not operator_confirmed:
                    raise AssertionError('Setup must confirm the operator')
                self._append([*test.state['joints'], test.state['gripper_position']], 'hold', executed=False)

            def ready_state(self):
                return deepcopy(test.state)

            def _execute(self, target, *, kind):
                row = self._append(target, kind, executed=True)
                if test.execution_error is not None and self.execute.call_count == 2:
                    raise test.execution_error
                return {'status':'completed', 'command':row, 'return_code':0,
                        'final_state':deepcopy(test.state)}

        for target, replacement in (
            ('lerobot_robot_fairino.FairinoFR5', self.observer),
            ('lerobot_robot_fairino.control_transport.FR5RPC', self.rpc),
        ):
            fake = patch(target, return_value=replacement)
            fake.start()
            self.addCleanup(fake.stop)
        fake_controller = patch('lerobot_robot_fairino.pi05_rollout.PolicyController', FakePolicyController)
        fake_controller.start()
        self.addCleanup(fake_controller.stop)

    def reposition(self):
        self.console._reposition('offline-setup', self.plan, 'left', self.stop)

    def assert_disconnected_and_cleared(self):
        self.observer.connect.assert_called_once_with()
        self.observer.get_sample.assert_called_once_with()
        self.observer.disconnect.assert_called_once_with()
        self.assertFalse(self.observer.is_connected)
        self.assertIsNone(self.console.controller)

    def test_completed_reposition_saves_deque_journal_and_both_commands(self):
        with patch('fr5_runtime.console.write_json', wraps=fr5_console.write_json) as write:
            self.reposition()
        controller, = self.controllers
        report = json.loads((self.plan/'start-setup.json').read_text())
        self.assertIsInstance(controller._journal, deque)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['journal'], list(controller._journal))
        self.assertEqual(len(report['journal']), 3)  # arm hold and both moves
        self.assertEqual([row['source'] for row in report['journal']],
                         ['llm_workflow_start_setup']*3)
        self.assertEqual([row['status'] for row in report['actions']], ['completed', 'completed'])
        self.assertEqual([row['result']['status'] for row in report['actions']], ['completed', 'completed'])
        self.assertEqual([row['tcp_target'] for row in report['actions']],
                         [[0,0,110,0,0,0], [11,2,90,0,0,0]])
        targets = [[1.,2.,3.,4.,5.,6.,100], [7.,8.,9.,10.,11.,12.,100]]
        self.assertEqual([row['target'] for row in report['actions']], targets)
        self.assertEqual([row['result']['command'] for row in report['actions']], report['journal'][1:])
        self.assertTrue(all(row['executed'] and row['kind'] == 'arm' for row in report['journal'][1:]))
        self.assertEqual(controller.execute.call_args_list, [call(target, kind='arm') for target in targets])
        self.assertEqual(self.rpc.call.call_args_list,
                         [call('GetInverseKinRef', 0, row['tcp_target'], self.state['joints'])
                          for row in report['actions']])
        self.assert_disconnected_and_cleared()

        final_report = write.call_args.args[1]
        self.assertIsInstance(final_report['journal'], list)
        controller._journal[1]['target'][0] = -999
        self.assertEqual(final_report['journal'], report['journal'])

    def test_inactive_gripper_reports_the_real_cause_without_moving(self):
        self.state.update(gripper_active=0, gripper_motiondone=0, gripper_position=0)
        with self.assertRaisesRegex(RuntimeError, '夹爪尚未激活'):
            self.reposition()
        self.controllers[0].execute.assert_not_called()
        report = json.loads((self.plan/'start-setup.json').read_text())
        self.assertEqual(report['initial_state']['gripper_active'], 0)
        self.assertEqual(report['actions'], [])
        self.assert_disconnected_and_cleared()

    def test_final_write_failure_still_disconnects_and_clears_controller(self):
        actual_write = fr5_console.write_json
        write_error = OSError('injected final write failure')
        final_writes = []

        def fail_final_write(path, report):
            if 'journal' in report:
                final_writes.append(deepcopy(report))
                raise write_error
            actual_write(path, report)

        with patch('fr5_runtime.console.write_json', side_effect=fail_final_write):
            with self.assertRaises(OSError) as caught:
                self.reposition()
        self.assertIs(caught.exception, write_error)
        self.assertEqual(len(final_writes), 1)
        self.assertEqual(final_writes[0]['status'], 'completed')
        self.assertEqual([row['status'] for row in final_writes[0]['actions']], ['completed', 'completed'])
        self.assertEqual(self.controllers[0].execute.call_count, 2)
        self.assert_disconnected_and_cleared()

    def test_execution_failure_preserves_exception_and_saves_interrupted_journal(self):
        error = RuntimeError('injected command failure')
        self.execution_error = error
        with self.assertRaises(RuntimeError) as caught:
            self.reposition()
        self.assertIs(caught.exception, error)
        controller, = self.controllers
        report = json.loads((self.plan/'start-setup.json').read_text())
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['error'], str(error))
        self.assertEqual(report['journal'], list(controller._journal))
        self.assertEqual(len(report['journal']), 3)
        self.assertEqual([row['status'] for row in report['actions']], ['completed', 'prepared'])
        controller.stop.assert_called_once_with(str(error))
        self.assert_disconnected_and_cleared()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.console = Mock(csrf='a-test-token')
        self.console.status.return_value = {'csrf':'a-test-token','job':None,'camera':{'ok':False}}
        self.console.start.return_value = {'ok':True,'episode':'test-no-hardware'}
        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.console = self.console
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.thread.join();self.server.server_close()

    def test_get_status_has_no_actuator_side_effect(self):
        with urllib.request.urlopen(self.base+'/api/status') as response:
            self.assertEqual(json.load(response)['job'], None)
        self.console.start.assert_not_called();self.console.stop.assert_not_called()

    def test_foreign_page_cannot_start_and_explicit_same_page_click_can(self):
        body=json.dumps({'cup_uv':[1,2]}).encode()
        for headers in ({}, {'X-FR5-CSRF':'a-test-token','Origin':'http://example.invalid'},
                        {'X-FR5-CSRF':'a-test-token','Host':'example.invalid','Origin':'http://example.invalid'}):
            request=urllib.request.Request(self.base+'/api/start',data=body,headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code,403)
        self.console.start.assert_not_called()
        request=urllib.request.Request(self.base+'/api/start',data=body,
            headers={'X-FR5-CSRF':'a-test-token','Origin':self.base})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(json.load(response)['episode'],'test-no-hardware')
        self.console.start.assert_called_once()


if __name__=='__main__':
    unittest.main()
