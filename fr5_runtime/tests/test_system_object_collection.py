"""Object collection CLI routing tests without device or server access."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fr5_runtime import system


class ObjectCollectionCommandTests(unittest.TestCase):
    def setUp(self):
        for target in ('robot_lease', 'remote_json', 'checkpoint_spec', 'observe'):
            guard = patch.object(system, target,
                                 side_effect=AssertionError('Unexpected device, lease or server access'))
            guard.start()
            self.addCleanup(guard.stop)
        for target in ('fr5_runtime.system.urllib.request.urlopen', 'fr5_runtime.system.subprocess.run'):
            guard = patch(target, side_effect=AssertionError('CLI routing tests cannot access devices or servers'))
            guard.start()
            self.addCleanup(guard.stop)
        collector = importlib.import_module('fr5_jev.collect_object')
        collect = patch.object(collector, 'collect', autospec=True, return_value={'status': 'prepared'})
        self.collect = collect.start()
        self.addCleanup(collect.stop)
        runtime = ModuleType('lerobot_robot_fairino.stream_runtime')
        self.run_policy = runtime.run = Mock(return_value={'status': 'policy_preview'})
        modules = patch.dict(sys.modules, {runtime.__name__: runtime})
        modules.start()
        self.addCleanup(modules.stop)
        self.common = ['fr5-system', 'collect-object', '--name', 'purple-object',
                       '--object-description', '紫色方块', '--closing-percent', '82',
                       '--grasp-offset', '1.5', '-2', '3.25']

    def command(self, arguments):
        with patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()) as output:
            system.main()
        return json.loads(output.getvalue())

    def test_default_request_does_not_enable_execution(self):
        result = self.command(self.common + ['--rgb-pixel', '322.5', '217.25'])
        self.collect.assert_called_once()
        options = self.collect.call_args.kwargs
        self.assertFalse(options['execute'])
        self.assertIsNone(options['support_rgb_pixel'])
        self.assertEqual(options['approach'], 'tilted')
        self.assertEqual(options['grasp_below_top_mm'], 10.)
        self.assertEqual(options['side_retraction_mm'], 0.)
        self.assertEqual(options['tilt_from_down_deg'], 55.)
        self.assertEqual(options['tilted_closing_edge'], 'long')
        self.assertEqual(options['split'], 'training_complete_trial')
        self.assertEqual(result, {'status': 'prepared'})
        self.run_policy.assert_not_called()

    def test_explicit_options_reach_object_collection(self):
        self.command(self.common + ['--rgb-pixel', '322.5', '217.25',
                     '--support-rgb-pixel', '338.75', '254.5',
                     '--approach', 'side', '--grasp-below-top-mm', '12.5',
                     '--side-retraction-mm', '7.25', '--tilt-from-down-deg', '48',
                     '--tilted-closing-edge', 'short', '--split', 'held_out_complete_trial', '--execute'])
        self.collect.assert_called_once_with(name='purple-object', object_description='紫色方块',
            rgb_pixel=[322.5, 217.25], support_rgb_pixel=[338.75, 254.5], mask_path=None, closing_percent=82,
            grasp_offset=[1.5, -2., 3.25], approach='side', grasp_below_top_mm=12.5,
            side_retraction_mm=7.25, tilt_from_down_deg=48., tilted_closing_edge='short',
            split='held_out_complete_trial', execute=True)
        self.assertIs(type(self.collect.call_args.kwargs['closing_percent']), int)
        self.run_policy.assert_not_called()

    def test_mask_and_support_pixel_are_forwarded_without_an_object_pixel_or_execution(self):
        self.command(self.common + ['--mask', 'object-mask.png', '--support-rgb-pixel', '338.75', '254.5'])
        self.collect.assert_called_once()
        options = self.collect.call_args.kwargs
        self.assertEqual(options['mask_path'], Path('object-mask.png'))
        self.assertEqual(options['support_rgb_pixel'], [338.75, 254.5])
        self.assertIsNone(options['rgb_pixel'])
        self.assertFalse(options['execute'])

    def test_mask_and_pixel_require_exactly_one_selection(self):
        for selection in ([], ['--rgb-pixel', '322', '217', '--mask', 'object-mask.png'],
                          ['--support-rgb-pixel', '338.75', '254.5']):
            with self.subTest(selection=selection), patch('sys.argv', self.common + selection), \
                    contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    system.main()
                self.assertEqual(error.exception.code, 2)
        self.collect.assert_not_called()
        self.run_policy.assert_not_called()

    def test_existing_run_reaches_the_policy_runtime(self):
        checkpoint = system.REMOTE + '/checkpoints/example/step-000500'
        selected = {'n_action_steps': 10, 'checkpoint_sha256': 'a' * 64}
        with tempfile.TemporaryDirectory(prefix='object-cli-run-') as temp, \
                patch.object(system, 'SPOOL', Path(temp)), \
                patch.object(system, 'checkpoint_spec', return_value=selected) as select, \
                patch.object(system, 'robot_lease', return_value=contextlib.nullcontext()) as lease:
            result = self.command(['fr5-system', 'run', '--checkpoint', checkpoint,
                                   '--task', '抓取物体', '--name', 'policy-routing'])
        select.assert_called_once_with(checkpoint)
        lease.assert_called_once_with()
        self.run_policy.assert_called_once()
        self.assertEqual(self.run_policy.call_args.args[:3], (checkpoint, None, '抓取物体'))
        self.assertFalse(self.run_policy.call_args.kwargs['execute'])
        self.assertEqual(result, {'status': 'policy_preview'})
        self.collect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
