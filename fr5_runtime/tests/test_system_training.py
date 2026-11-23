"""Training entry point tests; never starts a GPU job or touches the robot."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fr5_runtime import system


class StreamTrainingCommandTests(unittest.TestCase):
    def test_cup_roll_span_is_forwarded_only_when_specified(self):
        common=['fr5-system','train-stream','--dataset','existing-eight','--run','candidate','--steps','64']
        for extra, expected in (([], None), (['--cup-roll-state-min-span-rad','.010'], '0.01')):
            with self.subTest(extra=extra), patch('sys.argv',common+extra), \
                 patch.object(system,'server_source',return_value=system.REMOTE+'/src/test'), \
                 patch.object(system,'server_program') as launch, patch.object(system,'configure'), \
                 contextlib.redirect_stdout(io.StringIO()):
                system.main()
            command=launch.call_args.args[0]
            if expected is None:
                self.assertNotIn('--cup-roll-state-min-span-rad',command)
            else:
                self.assertEqual(command[command.index('--cup-roll-state-min-span-rad')+1],expected)

    def test_vision_and_history_initialization_are_forwarded(self):
        parent=system.REMOTE+'/checkpoints/parent/step-001000'
        argv=['fr5-system','train-stream','--dataset','existing-eight','--run','vision-candidate',
              '--steps','1','--gpu','9','--adaptation','full-expert-vision-language-lora',
              '--action-representation','joint-delta-from-anchor-v1','--initialize',parent,
              '--observation-context-profile','state-history-physical-v1',
              '--lr-schedule','cosine','--decay-learning-rate','0.0000025',
              '--sampling-profile','uniform-all-frames-v1']
        with patch('sys.argv',argv), patch.object(system,'server_source',return_value=system.REMOTE+'/src/test'), \
             patch.object(system,'server_program') as launch, patch.object(system,'configure') as configure, \
             contextlib.redirect_stdout(io.StringIO()):
            system.main()
        command=launch.call_args.args[0]
        self.assertEqual(command[command.index('--adaptation')+1],'full-expert-vision-language-lora')
        self.assertEqual(command[command.index('--initialize')+1],parent)
        self.assertEqual(command[command.index('--observation-context-profile')+1],'state-history-physical-v1')
        self.assertEqual(command[command.index('--lr-schedule')+1],'cosine')
        self.assertEqual(command[command.index('--decay-learning-rate')+1],'2.5e-06')
        self.assertEqual(command[command.index('--sampling-profile')+1],'uniform-all-frames-v1')
        configure.assert_called_once_with(last_stream_candidate=system.REMOTE+'/checkpoints/vision-candidate/step-000001')

    def test_workspace_only_profile_is_explicit_and_forwarded(self):
        argv=['fr5-system','train-stream','--dataset','existing-eight','--run','crop-candidate',
              '--steps','250','--image-profile','workspace-only-v1']
        with patch('sys.argv',argv), patch.object(system,'server_source',return_value=system.REMOTE+'/src/test'), \
             patch.object(system,'server_program') as launch, patch.object(system,'configure'), \
             contextlib.redirect_stdout(io.StringIO()):
            system.main()
        command=launch.call_args.args[0]
        self.assertEqual(command[command.index('--image-profile')+1],'workspace-only-v1')

    def test_approach_open_sampler_is_explicit_and_forwarded(self):
        argv=['fr5-system','train-stream','--dataset','existing-eight','--run','conditioned-candidate',
              '--steps','500','--sampling-profile','approach-open-v1']
        with patch('sys.argv',argv), patch.object(system,'server_source',return_value=system.REMOTE+'/src/test'), \
             patch.object(system,'server_program') as launch, patch.object(system,'configure'), \
             contextlib.redirect_stdout(io.StringIO()):
            system.main()
        command=launch.call_args.args[0]
        self.assertEqual(command[command.index('--sampling-profile')+1],'approach-open-v1')

    def test_uniform_switch_and_seed_are_forwarded(self):
        for profile in ('uniform-switch-v1', 'uniform-switch-hold-v1'):
            with self.subTest(profile=profile):
                argv=['fr5-system','train-stream','--dataset','existing-eight','--run','switch-candidate',
                      '--steps','1000','--sampling-profile',profile,'--seed','20261103']
                with patch('sys.argv',argv), patch.object(system,'server_source',return_value=system.REMOTE+'/src/test'), \
                     patch.object(system,'server_program') as launch, patch.object(system,'configure'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    system.main()
                command=launch.call_args.args[0]
                self.assertEqual(command[command.index('--sampling-profile')+1], profile)
                self.assertEqual(command[command.index('--seed')+1], '20261103')

    def test_selected_stream_bundle_settings_reach_the_runtime(self):
        model=system.REMOTE+'/checkpoints/example/step-000500'
        spec={'policy_contract':{'name':'fr5-joint-position-stream-v1'},
              'checkpoint_sha256':'a'*64,'n_action_steps':10}
        argv=['fr5-system','run','--checkpoint',model,'--task','抓杯子',
              '--name','readonly-config-test']
        with tempfile.TemporaryDirectory() as temp, patch('sys.argv',argv), \
             patch.object(system,'SPOOL',Path(temp)), \
             patch.object(system,'checkpoint_spec',return_value=spec), \
             patch.object(system,'robot_lease',return_value=contextlib.nullcontext()), \
             patch('lerobot_robot_fairino.stream_runtime.run',return_value={}) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            system.main()
        self.assertEqual(run.call_args.kwargs['expected_action_steps'],10)
        self.assertEqual(run.call_args.kwargs['expected_sha'],'a'*64)

    def test_measured_table_scene_reaches_runtime_without_changing_policy_settings(self):
        model = system.REMOTE + '/checkpoints/example/step-000500'
        spec = {'policy_contract': {'name': 'fr5-joint-position-stream-v1'},
                'checkpoint_sha256': 'a' * 64, 'n_action_steps': 10}
        scene = {'table_plane': {'normal': [0., 0., 1.], 'offset_mm': 12.5},
                 'source_observation': 'read-only-rgbd'}
        with tempfile.TemporaryDirectory() as temp:
            scene_path = Path(temp) / 'scene.json'
            scene_path.write_text(json.dumps(scene))
            argv = ['fr5-system', 'run', '--checkpoint', model, '--task', '抓取物体',
                    '--name', 'readonly-scene-test', '--scene-json', str(scene_path)]
            with patch('sys.argv', argv), patch.object(system, 'SPOOL', Path(temp)), \
                    patch.object(system, 'checkpoint_spec', return_value=spec), \
                    patch.object(system, 'robot_lease', return_value=contextlib.nullcontext()), \
                    patch('lerobot_robot_fairino.stream_runtime.run', return_value={}) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                system.main()
        self.assertEqual(run.call_args.kwargs['joint_scene'], scene)
        self.assertEqual(run.call_args.kwargs['expected_action_steps'], 10)
        self.assertFalse(run.call_args.kwargs['execute'])
        self.assertFalse(run.call_args.kwargs['execute'])

    def test_short_prefix_reaches_runtime_with_original_checkpoint_validation(self):
        model = system.REMOTE + '/checkpoints/example/step-005000'
        spec = {'policy_contract': {'name': 'fr5-joint-position-stream-v1'},
                'checkpoint_sha256': 'a' * 64, 'n_action_steps': 10}
        argv = ['fr5-system', 'run', '--checkpoint', model, '--task', '抓取物体',
                '--name', 'readonly-short-prefix', '--prefix-steps', '3']
        with tempfile.TemporaryDirectory() as temp, patch('sys.argv', argv), \
                patch.object(system, 'SPOOL', Path(temp)), \
                patch.object(system, 'checkpoint_spec', return_value=spec), \
                patch.object(system, 'robot_lease', return_value=contextlib.nullcontext()), \
                patch('lerobot_robot_fairino.stream_runtime.run', return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            system.main()
        self.assertEqual(run.call_args.kwargs['execution_steps'], 3)
        self.assertEqual(run.call_args.kwargs['expected_action_steps'], 10)
        self.assertEqual(run.call_args.kwargs['expected_sha'], 'a' * 64)
        self.assertFalse(run.call_args.kwargs['execute'])

    def test_full_expert_relative_configuration_reaches_server_unchanged(self):
        argv = ['fr5-system', 'train-stream', '--dataset', 'existing-eight',
                '--run', 'relative-expert', '--steps', '500', '--gpu', '9',
                '--adaptation', 'full-expert-language-lora',
                '--action-representation', 'joint-delta-from-anchor-v1',
                '--n-action-steps', '10',
                '--image-profile', 'full-and-workspace-v1',
                '--batch-size', '1', '--gradient-accumulation', '4',
                '--learning-rate', '0.00002', '--first-action-weight', '8',
                '--gripper-weight', '3']
        with patch('sys.argv', argv), \
             patch.object(system, 'server_source', return_value=system.REMOTE+'/src/test'), \
             patch.object(system, 'server_program') as launch, \
             patch.object(system, 'configure') as configure, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            system.main()
        command = launch.call_args.args[0]
        for name, value in (('--adaptation', 'full-expert-language-lora'),
                            ('--action-representation', 'joint-delta-from-anchor-v1'),
                            ('--n-action-steps', '10'),
                            ('--image-profile', 'full-and-workspace-v1'),
                            ('--learning-rate', '2e-05'), ('--first-action-weight', '8.0'),
                            ('--gripper-weight', '3.0'), ('--gpu', '9')):
            self.assertEqual(command[command.index(name)+1], value)
        configure.assert_called_once_with(last_stream_candidate=
            system.REMOTE+'/checkpoints/relative-expert/step-000500')
        self.assertIn('"task_success_verified": false', output.getvalue())

    def test_resume_does_not_overwrite_the_parent_prefix_or_image_profile(self):
        parent=system.REMOTE+'/checkpoints/relative-expert/step-000500'
        argv=['fr5-system','train-stream','--dataset','existing-eight','--run','relative-expert',
              '--steps','1000','--resume',parent,'--adaptation','full-expert-language-lora',
              '--action-representation','joint-delta-from-anchor-v1']
        with patch('sys.argv',argv), \
             patch.object(system,'server_source',return_value=system.REMOTE+'/src/test'), \
             patch.object(system,'server_program') as launch, \
             patch.object(system,'configure'), contextlib.redirect_stdout(io.StringIO()):
            system.main()
        command=launch.call_args.args[0]
        self.assertNotIn('--n-action-steps',command)
        self.assertNotIn('--image-profile',command)
        self.assertNotIn('--sampling-profile',command)
        self.assertNotIn('--observation-context-profile',command)
        self.assertEqual(command[command.index('--resume')+1],parent)

    def test_resume_and_weights_initialization_are_mutually_exclusive(self):
        argv = ['fr5-system', 'train-stream', '--dataset', 'existing-eight',
                '--run', 'example', '--steps', '500', '--resume', 'one', '--initialize', 'two']
        with patch('sys.argv', argv), patch.object(system, 'server_program') as launch, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            system.main()
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
