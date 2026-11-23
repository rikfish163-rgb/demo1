"""Policy workbench tests with all robot and server operations mocked."""
import io
import json
import fcntl
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from fr5_runtime import policy_console as console


class PolicyConsoleTests(unittest.TestCase):
    def test_selected_contract_never_uses_another_experiments_contract(self):
        legacy='/home/data4/hts/demo1/fr5-learning/checkpoints/old-event-policy'
        stream='/home/data4/hts/demo1/fr5-learning/checkpoints/stream/step-001000'
        previous={'checkpoint':stream,'policy_contract':'fr5-joint-position-stream-v1'}
        with patch.object(console.system,'configuration',return_value={
                'last_policy_contract':'fr5-joint-position-stream-v1',
                'checkpoint_contracts':{stream:{'name':'fr5-joint-position-stream-v1'}}}):
            self.assertIsNone(console.selected_checkpoint_contract(legacy,previous))
            self.assertEqual(console.selected_checkpoint_contract(stream),
                             'fr5-joint-position-stream-v1')
        with patch.object(console.system,'configuration',return_value={}):
            self.assertEqual(console.selected_checkpoint_contract(stream,previous),
                             'fr5-joint-position-stream-v1')

    def test_stream_step_checkpoint_is_allowed_but_path_escape_is_not(self):
        root='/home/data4/hts/demo1/fr5-learning/checkpoints/'
        with patch.object(console.system,'configuration',return_value={'last_checkpoint':root+'stream-run/step-001000'}):
            self.assertEqual(console.checkpoint(),root+'stream-run/step-001000')
        with patch.object(console.system,'configuration',return_value={'last_checkpoint':root+'../other/step-001000'}):
            self.assertIsNone(console.checkpoint())

    def test_stream_chunk_preview_shows_submitted_target_not_first_unused_prediction(self):
        root=self.spool/'stream-trial';root.mkdir()
        (root/'scope.json').write_text(json.dumps({'dataset_role':'policy_evaluation',
            'policy_contract':{'name':'fr5-joint-position-stream-v1'}}))
        (root/'chunk-00000.json').write_text(json.dumps({'request':{'id':'frame-a','state':[0.]*6+[1.,1.]},
            'reply':{'actions':[[99.]*7]*50,'inference_seconds':.4},'status':'available'}))
        (root/'submitted-targets.jsonl').write_text(json.dumps({'request_id':'frame-a','action_index':3,
            'decoded':{'joint_degrees':[1.,2.,3.,4.,5.,6.],'gripper_integer_percent':76}})+'\n')
        result=console.episode_view('stream-trial')
        self.assertEqual(result['events'][0]['joint_degrees'],[1.,2.,3.,4.,5.,6.])
        self.assertEqual(result['events'][0]['action_index'],3)
        self.assertEqual(result['events'][0]['status'],'submitted')
        self.assertIsNone(result['summary'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / 'state'
        self.spool = self.root / 'spool'
        self.spool.mkdir()
        for target, value in ((patch.object(console, 'STATE', self.state), None),
                              (patch.object(console.system, 'SPOOL', self.spool), None)):
            target.start()
            self.addCleanup(target.stop)

    def test_start_runs_existing_runtime_with_selected_task_and_no_extra_limits(self):
        hub = console.PolicyConsole()
        child = Mock(pid=1234)
        child.poll.return_value = None
        model = '/home/data4/hts/demo1/fr5-learning/checkpoints/example'
        task = '抓杯子；$(this is literal text)'
        with patch.object(console, 'checkpoint', return_value=model), \
             patch.object(console, 'active_runtimes', return_value=[]), \
             patch.object(console, 'robot_busy', return_value=False), \
             patch.object(console, 'monitor_status', return_value={'camera': {'ok': True}, 'robot': {'ok': True}}), \
             patch.object(console.system, 'generated', return_value='pi05-ui-test'), \
             patch.object(console.subprocess, 'Popen', return_value=child) as start:
            self.assertEqual(hub.start({'task': task, 'speed': 10})['episode'], 'pi05-ui-test')
            argv = start.call_args.args[0]
            self.assertEqual(argv[argv.index('--worker')+1], 'pi05-ui-test')
            self.assertEqual(argv[argv.index('--task')+1], task)
            self.assertEqual(argv[argv.index('--checkpoint')+1], model)
            self.assertNotIn('--seconds', argv)
            self.assertNotIn('--max-events', argv)
            self.assertTrue(start.call_args.kwargs['start_new_session'])
            with self.assertRaisesRegex(RuntimeError, '已有任务'):
                hub.start({'task': task, 'speed': 10})
            start.assert_called_once()

    def test_existing_robot_owner_is_never_replaced(self):
        hub = console.PolicyConsole()
        with patch.object(console, 'checkpoint', return_value='/home/data4/hts/demo1/fr5-learning/checkpoints/example'), \
             patch.object(console, 'active_runtimes', return_value=[{'pid': 99, 'episode': 'other'}]), \
             patch.object(console.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(RuntimeError, '已有任务'):
                hub.start({'task': '抓杯'})
            start.assert_not_called()

    def test_latest_episode_uses_real_policy_record_not_stale_configuration(self):
        for name, role, stamp in [('old-policy', 'policy_evaluation', 100),
                                  ('new-policy', 'policy_evaluation', 200),
                                  ('newer-collection', 'teacher', 300)]:
            directory = self.spool / name
            directory.mkdir()
            scope = directory / 'scope.json'
            scope.write_text(json.dumps({'dataset_role': role}))
            os.utime(scope, (stamp, stamp))
        self.assertEqual(console.latest_policy_episode(), 'new-policy')

    def test_stop_only_signals_worker_started_by_this_workbench(self):
        hub = console.PolicyConsole()
        with patch.object(hub, 'current_worker', return_value=None), patch.object(console.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, '没有由此页面启动'):
                hub.stop()
            kill.assert_not_called()
        worker = {'worker_pid': 1234, 'episode': 'pi05-ui-test'}
        with patch.object(hub, 'current_worker', return_value=worker), \
             patch.object(console, 'worker_alive', return_value=True), patch.object(console.os, 'kill') as kill:
            self.assertEqual(hub.stop()['episode'], 'pi05-ui-test')
            kill.assert_called_once_with(1234, signal.SIGINT)

    def test_existing_robot_lease_is_detected(self):
        local_state = self.root / 'robot-state'
        local_state.mkdir()
        lease = local_state / 'j1.lock'
        lease.touch()
        with patch.object(console.system, 'LOCAL_STATE', local_state):
            with lease.open('r') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(console.robot_busy())
                fcntl.flock(owner, fcntl.LOCK_UN)
            self.assertFalse(console.robot_busy())

    def test_stop_before_worker_launch_never_starts_robot_runtime(self):
        handlers = {}
        snapshots = []
        def save(_name, state):
            snapshots.append(dict(state))
            if len(snapshots) == 1:
                handlers[signal.SIGINT](signal.SIGINT, None)
        with patch.object(console.signal, 'signal', side_effect=lambda sig, handler:handlers.update({sig:handler})), \
             patch.object(console, 'write_state', side_effect=save), \
             patch.object(console.subprocess, 'Popen') as start:
            console.worker('pi05-ui-test', '/data/model', '抓杯')
        start.assert_not_called()
        self.assertEqual(snapshots[-1]['status'], 'stopped')

    def test_worker_uses_model_runtime_and_persists_only_small_status(self):
        state_snapshots = []
        child = Mock(pid=4321, stdout=io.StringIO('{"status":"ready"}\n{"status":"policy_event_completed","event_index":0}\n'))
        child.wait.return_value = 0
        with patch.object(console.subprocess, 'Popen', return_value=child) as launch, \
             patch.object(console.signal, 'signal'), \
             patch.object(console, 'write_state', side_effect=lambda _name, v:state_snapshots.append(dict(v))):
            console.worker('pi05-ui-test', '/data/model', '抓杯')
        argv = launch.call_args.args[0]
        self.assertIn('--execute', argv)
        self.assertEqual(argv[argv.index('--task')+1], '抓杯')
        self.assertEqual(argv[argv.index('--checkpoint')+1], '/data/model')
        self.assertNotIn('--seconds', argv)
        self.assertNotIn('--max-events', argv)
        self.assertEqual(state_snapshots[-1]['status'], 'finished')
        self.assertEqual(state_snapshots[-1]['exit_code'], 0)
        self.assertEqual(state_snapshots[-1]['last_event']['event_index'], 0)

    def test_user_review_is_recorded_only_after_explicit_review_action(self):
        hub = console.PolicyConsole()
        with patch.object(console, 'episode_view', return_value={'summary': None}), \
             patch.object(console.system, 'review') as review:
            with self.assertRaisesRegex(ValueError, '保存后'):
                hub.review({'episode': 'pi05-ui-test', 'verdict': 'success'})
            review.assert_not_called()
        with patch.object(console, 'episode_view', return_value={'summary': {'outcome': 'interrupted'}}), \
             patch.object(console.system, 'review', return_value={'verdict': 'success'}) as review:
            result = hub.review({'episode': 'pi05-ui-test', 'verdict': 'success'})
            self.assertEqual(result['verdict'], 'success')
            self.assertEqual(review.call_args.kwargs['input_method'], 'web_console_click')


if __name__ == '__main__':
    unittest.main()
