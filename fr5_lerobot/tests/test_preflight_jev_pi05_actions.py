"""Offline checks for the read-only absolute-joint pi0.5 preflight."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fr5_lerobot.diagnostics import preflight_jev_pi05_actions as preflight


class FakeRPC:
    def __init__(self, *, limits=None, fk_reply=None, middle_below_table=False):
        self.limits = [0] + [-180., 180.] * 6 if limits is None else limits
        self.fk_reply = fk_reply
        self.middle_below_table = middle_below_table
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, args))
        if method == 'GetJointSoftLimitDeg':
            return self.limits
        if method == 'GetForwardKin':
            if self.fk_reply is not None:
                return self.fk_reply
            q = args[0]
            height = 160. if self.middle_below_table and .4 <= q[0] <= .6 else 250.
            return [0, 0., 0., height, 180., 0., 0.]
        raise AssertionError(f'Unexpected RPC: {method}')


class JointPreflightTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket.connect', 'socket.socket.connect_ex'):
            guard = patch(name, side_effect=AssertionError('Offline test cannot connect to a device'))
            guard.start()
            self.addCleanup(guard.stop)
        self.request = {'id': 'test-request', 'state': [0.] * 6 + [.7, 1.]}
        self.actions = [[0.] * 6 + [1.] for _ in range(50)]
        self.reply = {'id': 'test-request', 'wire_action_representation': 'absolute-joint-position-v1',
                      'actions': self.actions}
        self.scene = {'table_plane': {'normal': [0., 0., 1.], 'offset_mm': 0.}}

    def run_preflight(self, rpc, steps=1):
        return preflight.preflight(self.request, self.reply, self.scene, steps, rpc=rpc)

    def test_prefix_uses_native_actions_and_reports_every_fk_sample(self):
        self.actions[0] = [math.radians(.5), 0., 0., 0., 0., 0., 1.2]
        self.actions[1] = [math.radians(1.5), 0., 0., 0., 0., 0., -.2]
        self.actions[49][0] = math.radians(1000.)
        original_request, original_reply = deepcopy(self.request), deepcopy(self.reply)
        rpc = FakeRPC(limits=[0, -2., 2.] + [-180., 180.] * 5)
        report = self.run_preflight(rpc, steps=2)

        self.assertTrue(report['passed'])
        self.assertFalse(report['execution_authorized'])
        self.assertEqual(self.request, original_request)
        self.assertEqual(self.reply, original_reply)
        self.assertEqual([s['raw_model_action'] for s in report['segments']], self.actions[:2])
        self.assertAlmostEqual(report['minimum_joint_soft_limit_margin_deg'], .5)
        self.assertEqual(report['anchor_fk_sample']['segment_index'], None)
        self.assertEqual([s['segment_index'] for s in report['segments'][0]['fk_samples']], [0, 0])
        fractions = [0., *[s['fraction'] for s in report['segments'][1]['fk_samples']]]
        self.assertEqual(fractions[-1], 1.)
        self.assertTrue(all(0 < b - a <= .25 for a, b in zip(fractions, fractions[1:])))
        self.assertAlmostEqual(report['segments'][1]['target_joint_deg'][0], 1.5)
        clearance = report['tool_table_clearance']
        self.assertTrue(clearance['passed'])
        self.assertEqual(clearance['joint_sample_max_step_deg'], .25)
        self.assertGreater(min(clearance['clearance_mm_by_box'].values()), 5.)
        self.assertGreater(min(clearance['minimum_fk_sampled_clearance_mm_by_box'].values()), 5.)
        self.assertIn('housing', report['segments'][0]['fk_samples'][0]['tool_clearance_mm_by_box'])
        self.assertEqual(rpc.calls[0], ('GetJointSoftLimitDeg', (1,)))
        self.assertEqual({method for method, _ in rpc.calls}, {'GetJointSoftLimitDeg', 'GetForwardKin'})

    def test_table_violation_at_intermediate_controller_fk_sample_is_rejected(self):
        self.actions[0][0] = math.radians(1.)
        rpc = FakeRPC(middle_below_table=True)
        report = self.run_preflight(rpc)
        clearance = report['tool_table_clearance']
        self.assertFalse(report['passed'])
        self.assertFalse(clearance['passed'])
        self.assertEqual(clearance['status'], 'checked')
        self.assertIsNone(clearance['clearance_mm_by_box'])
        samples = [report['anchor_fk_sample'], *report['segments'][0]['fk_samples']]
        self.assertGreater(min(samples[0]['tool_clearance_mm_by_box'].values()), 5.)
        self.assertGreater(min(samples[-1]['tool_clearance_mm_by_box'].values()), 5.)
        middle = next(s for s in samples if s['fraction'] == .5)
        self.assertLess(min(middle['tool_clearance_mm_by_box'].values()), 5.)
        self.assertLess(min(clearance['minimum_fk_sampled_clearance_mm_by_box'].values()), 5.)
        self.assertEqual(report['failures'][0]['check'], 'tool_table_clearance')
        self.assertEqual({method for method, _ in rpc.calls}, {'GetJointSoftLimitDeg', 'GetForwardKin'})

    def test_limits_are_checked_before_any_fk_for_anchor_and_targets(self):
        self.actions[0][0] = math.radians(1.)
        rpc = FakeRPC(limits=[0, -1., .75] + [-180., 180.] * 5)
        report = self.run_preflight(rpc)
        self.assertFalse(report['passed'])
        self.assertAlmostEqual(report['minimum_joint_soft_limit_margin_deg'], -.25)
        self.assertEqual(report['failures'][0]['check'], 'joint_soft_limits')
        self.assertEqual(report['failures'][0]['action_index'], 0)
        self.assertEqual(rpc.calls, [('GetJointSoftLimitDeg', (1,))])

        self.request['state'][0] = math.radians(2.)
        self.actions[0][0] = math.radians(2.)
        rpc = FakeRPC(limits=[0, -1., .75] + [-180., 180.] * 5)
        report = self.run_preflight(rpc)
        self.assertEqual(report['failures'][0]['location'], 'anchor')
        self.assertEqual(rpc.calls, [('GetJointSoftLimitDeg', (1,))])

    def test_joint_step_limit_and_boundary(self):
        self.actions[0][0] = math.radians(1.501)
        rpc = FakeRPC()
        report = self.run_preflight(rpc)
        self.assertFalse(report['passed'])
        self.assertEqual(report['failures'][0]['check'], 'joint_step')
        self.assertGreater(report['segments'][0]['joint_step_max_deg'], 1.5)
        self.assertEqual(rpc.calls, [('GetJointSoftLimitDeg', (1,))])

        self.actions[0][0] = math.radians(math.nextafter(1.5, 0.))
        report = self.run_preflight(FakeRPC())
        self.assertTrue(report['passed'])
        self.assertLessEqual(report['segments'][0]['joint_step_max_deg'], 1.5)

    def test_bad_fk_and_soft_limit_replies_are_rejected(self):
        for bad_fk in ([7], [0, 1., 2.], [0, 0., 0., float('nan'), 0., 0., 0.]):
            with self.subTest(bad_fk=bad_fk):
                rpc = FakeRPC(fk_reply=bad_fk)
                report = self.run_preflight(rpc)
                self.assertFalse(report['passed'])
                self.assertEqual(report['tool_table_clearance']['status'], 'fk_error')
                self.assertEqual(report['failures'][0]['check'], 'forward_kinematics')
                self.assertEqual({method for method, _ in rpc.calls}, {'GetJointSoftLimitDeg', 'GetForwardKin'})
        for limits in ([9], [0] + [1., 1.] * 6):
            with self.subTest(limits=limits):
                rpc = FakeRPC(limits=limits)
                report = self.run_preflight(rpc)
                self.assertFalse(report['passed'])
                self.assertEqual(report['failures'][0]['check'], 'joint_soft_limits')
                self.assertEqual(rpc.calls, [('GetJointSoftLimitDeg', (1,))])

    def test_scene_requires_explicit_complete_table_plane(self):
        self.scene = {'table': {'normal': [-.0038970345458033497, .008166943828895, .9999590562369264],
                                'offset_mm': -8.328707946924201}}
        rpc = FakeRPC(fk_reply=[0, 20., -30., 250., 91., -25., 47.])
        report = self.run_preflight(rpc)
        self.assertTrue(report['passed'])
        clearance = report['tool_table_clearance']
        for name, minimum in clearance['clearance_mm_by_box'].items():
            self.assertAlmostEqual(minimum, clearance['minimum_fk_sampled_clearance_mm_by_box'][name], places=8)
        for scene in ({}, {'table_plane': {'normal': [0., 0., 1.]}},
                      {'table_plane': {'normal': [0., 0., -1.], 'offset_mm': 0.}},
                      {'table_plane': {}, 'table': {'normal': [0., 0., 1.], 'offset_mm': 0.}}):
            with self.subTest(scene=scene):
                rpc = FakeRPC()
                with self.assertRaises((ValueError, RuntimeError)):
                    preflight.preflight(self.request, self.reply, scene, 1, rpc=rpc)
                self.assertEqual(rpc.calls, [])

    def test_bad_shape_nonfinite_tail_representation_and_prefix_are_rejected(self):
        for bad_steps in (0, 51, True):
            with self.subTest(steps=bad_steps):
                rpc = FakeRPC()
                with self.assertRaises(ValueError):
                    self.run_preflight(rpc, steps=bad_steps)
                self.assertEqual(rpc.calls, [])
        for state in ([0.] * 7, [0.] * 7 + [float('nan')]):
            with self.subTest(state=state):
                rpc = FakeRPC()
                with self.assertRaises((ValueError, RuntimeError)):
                    preflight.preflight({**self.request, 'state': state}, self.reply, self.scene, 1, rpc=rpc)
                self.assertEqual(rpc.calls, [])
        for reply in ({**self.reply, 'actions': self.actions[:3]},
                      {**self.reply, 'wire_action_representation': 'cup-relative-tcp-v1'},
                      {**self.reply, 'id': 'other'}):
            with self.subTest(reply=reply.get('wire_action_representation')):
                rpc = FakeRPC()
                with self.assertRaises(ValueError):
                    preflight.preflight(self.request, reply, self.scene, 1, rpc=rpc)
                self.assertEqual(rpc.calls, [])
        self.actions[49][0] = float('nan')
        rpc = FakeRPC()
        with self.assertRaisesRegex(ValueError, 'finite 50 by 7'):
            self.run_preflight(rpc)
        self.assertEqual(rpc.calls, [])

    def test_cli_writes_new_json_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / name for name in ('request.json', 'reply.json', 'scene.json')]
            for path, value in zip(inputs, (self.request, self.reply, self.scene)):
                path.write_text(json.dumps(value), encoding='utf-8')
            output = root / 'result.json'
            args = ['--request-json', str(inputs[0]), '--reply-json', str(inputs[1]),
                    '--scene-json', str(inputs[2]), '--prefix-steps', '1', '--output', str(output)]
            with patch('builtins.print'):
                self.assertEqual(preflight.main(args, rpc=FakeRPC()), 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertTrue(result['passed'])
            self.assertFalse(result['execution_authorized'])
            original = output.read_bytes()
            rpc = FakeRPC()
            with self.assertRaises(FileExistsError):
                preflight.main(args, rpc=rpc)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(rpc.calls, [])

            self.actions[0][0] = math.radians(2.)
            inputs[1].write_text(json.dumps(self.reply), encoding='utf-8')
            rejected = root / 'rejected.json'
            args[-1] = str(rejected)
            with patch('builtins.print'):
                self.assertEqual(preflight.main(args, rpc=FakeRPC()), 1)
            self.assertFalse(json.loads(rejected.read_text(encoding='utf-8'))['passed'])

    def test_script_entrypoint_displays_help_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, preflight.__file__, '--help'], cwd=directory,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''},
                capture_output=True, text=True, check=True, timeout=15)
        self.assertIn('--request-json', result.stdout)
        self.assertIn('--prefix-steps', result.stdout)


if __name__ == '__main__':
    unittest.main()
