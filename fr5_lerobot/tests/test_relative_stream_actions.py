"""Offline checks for FR5 relative action chunks; no robot or GPU access."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import relative_stream_actions as relative


def sample_rows():
    rows, episodes = [], {}
    for ep, count in enumerate((4, 2, 3)):
        indices = []
        for frame in range(count):
            value = float(frame) + (100. if ep == 2 else ep * 10.)
            indices.append(len(rows))
            rows.append({'episode_index': ep, 'frame_index': frame,
                         '_physical_source': str(ep),
                         '_split': 'validation' if ep == 2 else 'train',
                         'observation.state': [value] * 6 + [0., 0.],
                         'action': [value + .5] * 6 + [0.2 if frame else 1.]})
        episodes[ep] = indices
    return rows, episodes


class RelativeActionTests(unittest.TestCase):
    def test_multibatch_chunk_uses_one_raw_anchor_and_preserves_gripper(self):
        state = torch.tensor([[1., 2., 3., 4., 5., 6., 0., 0.],
                              [10., 20., 30., 40., 50., 60., .7, 1.]])
        actions = torch.tensor([[[2., 4., 6., 8., 10., 12., .9],
                                 [4., 5., 6., 7., 8., 9., .2]],
                                [[11., 22., 33., 44., 55., 66., .5],
                                 [12., 24., 36., 48., 60., 72., .1]]])
        saved_state, saved_actions = state.clone(), actions.clone()
        converted = relative.to_relative_actions(actions, state, input_representation=relative.ABSOLUTE)
        expected = torch.tensor([[[1., 2., 3., 4., 5., 6., .9],
                                  [3., 3., 3., 3., 3., 3., .2]],
                                 [[1., 2., 3., 4., 5., 6., .5],
                                  [2., 4., 6., 8., 10., 12., .1]]])
        torch.testing.assert_close(converted, expected)
        restored = relative.to_absolute_actions(converted, state, input_representation=relative.RELATIVE)
        torch.testing.assert_close(restored, actions)
        self.assertTrue(torch.equal(restored[..., 6], actions[..., 6]))
        self.assertTrue(torch.equal(actions, saved_actions))
        self.assertTrue(torch.equal(state, saved_state))

    def test_request_snapshot_and_mismatched_request(self):
        state = torch.tensor([1., 2., 3., 4., 5., 6., .5, 1.])
        original = state.clone()
        anchor = relative.RequestAnchor.capture('request-1', state)
        state[:6] = 100.
        output = torch.tensor([[.1, .2, .3, .4, .5, .6, .25],
                               [.2, .3, .4, .5, .6, .7, 1.1]])
        restored = anchor.restore(output, request_id='request-1', input_representation=relative.RELATIVE)
        torch.testing.assert_close(restored[:, :6], output[:, :6] + original[:6])
        self.assertTrue(torch.equal(restored[:, 6], output[:, 6]))
        with self.assertRaisesRegex(ValueError, 'request_id'):
            anchor.restore(output, request_id='request-2', input_representation=relative.RELATIVE)

    def test_cup_delta_restores_a_shifted_measured_pose_without_changing_grip(self):
        state = torch.tensor([.300, .070, .185, 1.62, -.02, -1.18, 1., 1.])
        targets = torch.tensor([[.298, .071, .184, 1.62, -.02, -1.18, 1.],
                                [.292, .074, .180, 1.62, -.02, -1.18, .76]])
        deltas = relative.to_relative_actions(targets, state, input_representation=relative.CUP_ABSOLUTE)
        torch.testing.assert_close(deltas[:, :3], torch.tensor([[-.002, .001, -.001], [-.008, .004, -.005]]))
        new_state = state.clone()
        new_state[:3] += torch.tensor([.062, -.002, .003])
        restored = relative.to_absolute_actions(deltas, new_state, input_representation=relative.CUP_RELATIVE)
        torch.testing.assert_close(restored[:, :3], targets[:, :3] + (new_state - state)[:3])
        self.assertTrue(torch.equal(restored[:, 6], targets[:, 6]))

    def test_cup_tick_delta_round_trip_uses_each_action_horizon(self):
        state = torch.tensor([[.300, .070, .185, 1.62, -.02, -1.18, 1., 1.],
                              [.160, .072, .096, 1.62, -.02, -1.18, .78, 1.]])
        targets = state[:, None, :7].repeat(1, 50, 1)
        targets[0, :, 0] -= torch.arange(1, 51)*.001
        targets[1, :, 2] += .004
        targets[:, 3:, 6] = .76
        deltas = relative.to_relative_actions(targets, state, input_representation=relative.CUP_ABSOLUTE,
                                             per_tick=True)
        torch.testing.assert_close(deltas[0, :, 0], torch.full((50,), -.001))
        torch.testing.assert_close(deltas[1, :, 2], .004/torch.arange(1, 51))
        restored = relative.to_absolute_actions(deltas, state, input_representation=relative.CUP_TICK_RELATIVE)
        torch.testing.assert_close(restored, targets)
        self.assertTrue(torch.equal(restored[..., 6], targets[..., 6]))

    def test_representation_guard_and_pipeline_config(self):
        state = torch.zeros(8)
        absolute = relative.TaggedActions(torch.ones(2, 7), relative.ABSOLUTE)
        delta = absolute.relative(state)
        with self.assertRaisesRegex(ValueError, 'representation'):
            delta.relative(state)
        wire = delta.absolute(state)
        with self.assertRaisesRegex(ValueError, 'representation'):
            wire.absolute(state)
        with self.assertRaises(TypeError):
            relative.to_relative_actions(absolute.values, state)
        relative.assert_manual_pipeline({'use_relative_actions': False},
                                        {'steps': [{'registry_name': 'relative_actions_processor',
                                                    'config': {'enabled': False}}]})
        for config in ({}, {'use_relative_actions': True}, {'use_relative_actions': 0}):
            with self.assertRaises(ValueError):
                relative.assert_manual_pipeline(config)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            relative.assert_manual_pipeline(SimpleNamespace(use_relative_actions=False),
                                            {'steps': [{'registry_name': 'relative_actions_processor',
                                                        'config': {'enabled': True}}]})
        class AbsoluteActionsProcessorStep:
            enabled = True
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            relative.assert_manual_pipeline({'use_relative_actions': False},
                                            postprocessor=SimpleNamespace(steps=[AbsoluteActionsProcessorStep()]))
        with self.assertRaisesRegex(ValueError, 'inspect'):
            relative.assert_manual_pipeline({'use_relative_actions': False}, {})

    def test_bundle_loader_without_sys_modules_registration(self):
        # Existing stdio service loads bundled modules with this exact pattern.
        path = Path(relative.__file__)
        spec = importlib.util.spec_from_file_location('fr5_relative_bundle_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.RELATIVE, relative.RELATIVE)

    def test_shapes_nonfinite_and_unbatched_modes(self):
        state = torch.arange(8, dtype=torch.float32)
        actions = torch.ones(3, 7)
        for target, anchor in ((torch.ones(7), state), (actions, state),
                               (torch.ones(2, 7), torch.ones(2, 8)),
                               (torch.ones(2, 3, 7), torch.ones(2, 8))):
            result = relative.to_relative_actions(target, anchor, input_representation=relative.ABSOLUTE)
            self.assertEqual(result.shape, target.shape)
        for target, anchor in ((torch.ones(2, 7), torch.ones(3, 8)),
                               (torch.ones(2, 3, 7), state),
                               (torch.ones(2, 8), state),
                               (torch.ones(7), torch.ones(7)),
                               (torch.ones(0, 7), state),
                               (torch.full((7,), float('nan')), state),
                               (torch.ones(7), torch.full((8,), float('inf')))):
            with self.assertRaises(ValueError):
                relative.to_relative_actions(target, anchor, input_representation=relative.ABSOLUTE)
        with self.assertRaises(TypeError):
            relative.to_relative_actions(torch.ones(7, dtype=torch.int64), state,
                                         input_representation=relative.ABSOLUTE)

    def test_metadata_is_explicit(self):
        record = relative.metadata()
        self.assertEqual(record['model_action_representation'], relative.RELATIVE)
        self.assertEqual(record['wire_action_representation'], relative.ABSOLUTE)
        self.assertFalse(record['manual_adapter_requires_use_relative_actions'])
        self.assertEqual(record['gripper_dimension'], 6)


class RelativeStatisticsTests(unittest.TestCase):
    def test_camera_aligned_normalization_uses_per_observation_real_prefix(self):
        rows, episodes = sample_rows()
        for row in rows:
            row['_cup_relative_state'] = [0.] * 6 + [1., 1.]
            row['_camera_aligned_actions'] = [[.01, 0., 0., 0., 0., 0., 1.]]
        rows[0]['_camera_aligned_actions'].append([.06, 0., 0., 0., 0., 0., .76])
        _, report = relative.compute_relative_action_stats(
            rows, {0: episodes[0]}, chunk_size=3, representation=relative.CUP_TICK_RELATIVE, camera_aligned=True)
        self.assertEqual(report['valid_action_count'], 5)
        self.assertEqual(report['padding_count'], 7)
        self.assertAlmostEqual(report['max'][0], .03, places=6)
        self.assertEqual(rows[0]['_camera_aligned_actions'][1][0], .06)

    def test_mask_excludes_padding_and_validation(self):
        states = torch.tensor([[10.] * 6 + [1., 1.], [20.] * 6 + [1., 1.]])
        actions = torch.tensor([[[11.] * 6 + [1.], [12.] * 6 + [.5], [1e6] * 6 + [.8]],
                                [[23.] * 6 + [.2], [1e6] * 6 + [.8], [1e6] * 6 + [.8]]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        stats, report = relative.stats_from_padded_chunks(actions, states, mask, splits=['train', 'train'])
        self.assertEqual(report['anchor_count'], 2)
        self.assertEqual(report['valid_action_count'], 3)
        self.assertEqual(report['padding_count'], 3)
        self.assertAlmostEqual(report['raw_q01'][0], np.quantile([1., 2., 3.], .01))
        self.assertAlmostEqual(report['raw_q99'][6], np.quantile([1., .5, .2], .99), places=6)
        self.assertEqual(stats['action']['q01'].shape, (7,))
        self.assertLess(float(stats['action']['q99'][0]), 4.)
        with self.assertRaisesRegex(ValueError, 'training'):
            relative.stats_from_padded_chunks(actions, states, mask, splits=['train', 'validation'])
        with self.assertRaisesRegex(ValueError, 'valid_mask'):
            relative.stats_from_padded_chunks(actions, states, mask.float(), splits=['train', 'train'])
        with self.assertRaisesRegex(ValueError, 'at least one'):
            relative.stats_from_padded_chunks(actions, states, torch.zeros_like(mask), splits=['train', 'train'])

    def test_entire_valid_chunks_not_next_step_diffs_and_no_heldout(self):
        rows, episodes = sample_rows()
        original = copy.deepcopy(rows)
        stats, report = relative.compute_relative_action_stats(rows, {0: episodes[0]}, chunk_size=3)
        # Four anchors, valid lengths 3,3,2,1.  Future targets are relative to
        # each anchor, so 1.5 and 2.5 occur even though next-step delta is .5.
        expected = [.5, 1.5, 2.5, .5, 1.5, 2.5, .5, 1.5, .5]
        self.assertEqual(report['anchor_count'], 4)
        self.assertEqual(report['valid_action_count'], 9)
        self.assertEqual(report['padding_count'], 3)
        self.assertEqual(report['train_physical_sources'], ['0'])
        self.assertEqual(report['raw_q01'][0], float(np.quantile(expected, .01)))
        self.assertEqual(report['raw_q99'][0], float(np.quantile(expected, .99)))
        self.assertGreater(float(stats['action']['q99'][0]), 2.)
        self.assertEqual(rows, original)
        changed_heldout = copy.deepcopy(rows)
        for i in episodes[2]:
            changed_heldout[i]['action'] = [1e7] * 7
        _, report2 = relative.compute_relative_action_stats(changed_heldout, {0: episodes[0]}, chunk_size=3)
        self.assertEqual(report, report2)
        with self.assertRaisesRegex(ValueError, 'training'):
            relative.compute_relative_action_stats(rows, {2: episodes[2]}, chunk_size=3)
        with self.assertRaisesRegex(ValueError, 'training episode'):
            relative.compute_relative_action_stats(rows, {0: episodes[0]}, anchor_indices=[episodes[2][0]])

    def test_repeated_sampled_anchors_and_episode_boundary(self):
        rows, episodes = sample_rows()
        _, report = relative.compute_relative_action_stats(
            rows, {0: episodes[0], 1: episodes[1]}, chunk_size=3,
            anchor_indices=[episodes[0][3], episodes[0][3], episodes[1][0]])
        self.assertEqual(report['anchor_count'], 3)
        self.assertEqual(report['unique_anchor_count'], 2)
        self.assertEqual(report['valid_action_count'], 4)
        self.assertEqual(report['padding_count'], 5)
        self.assertEqual(report['max'][0], 1.5)
        self.assertEqual(report['anchor_selection'], 'explicit_anchor_sequence')
        conflicting = copy.deepcopy(rows)
        conflicting[episodes[2][0]]['_physical_source'] = '0'
        with self.assertRaisesRegex(ValueError, 'non-training split'):
            relative.compute_relative_action_stats(conflicting, {0: episodes[0]}, chunk_size=3)
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            relative.compute_relative_action_stats(rows, {0: [0, 2]}, chunk_size=3)

    def test_cup_delta_quantiles_use_tcp_targets_from_train_only(self):
        rows, episodes = sample_rows()
        for index, row in enumerate(rows):
            start = index * .010
            row['_cup_relative_state'] = [start, 0., 0., 1.62, -.02, -1.18, 1., 1.]
            x = start - .003 if row['_split'] == 'train' else 50.
            row['_cup_relative_action'] = [x, 0., 0., 1.62, -.02, -1.18, .76]
        stats, report = relative.compute_relative_action_stats(
            rows, {0: episodes[0]}, chunk_size=2, representation=relative.CUP_RELATIVE)
        self.assertEqual(report['representation'], relative.CUP_RELATIVE)
        self.assertEqual(report['valid_action_count'], 7)
        self.assertAlmostEqual(float(stats['action']['q01'][0]), -.003, places=6)
        self.assertAlmostEqual(float(stats['action']['q99'][0]), .007, places=6)
        self.assertAlmostEqual(float(stats['action']['q99'][1]-stats['action']['q01'][1]), .001, places=6)
        self.assertAlmostEqual(report['max'][6], .76, places=6)
        _, tick_report = relative.compute_relative_action_stats(
            rows, {0: episodes[0]}, chunk_size=2, representation=relative.CUP_TICK_RELATIVE)
        self.assertAlmostEqual(tick_report['max'][0], .0035, places=6)
        self.assertEqual(tick_report['valid_action_count'], report['valid_action_count'])

    def test_fallbacks_are_disclosed_and_valid(self):
        states = torch.zeros(1, 8)
        targets = torch.zeros(1, 2, 7)
        stats, report = relative.stats_from_padded_chunks(targets, states, torch.ones(1, 2, dtype=torch.bool),
                                                          splits=['train'])
        self.assertEqual(len(report['fallbacks']), 7)
        self.assertEqual(float(stats['action']['q01'][6]), 0.)
        self.assertEqual(float(stats['action']['q99'][6]), 1.)
        self.assertTrue(bool(torch.all(stats['action']['q99'] > stats['action']['q01'])))


try:
    from lerobot.processor.relative_action_processor import (
        RelativeActionsProcessorStep, AbsoluteActionsProcessorStep,
        to_relative_actions as official_relative, to_absolute_actions as official_absolute,
    )
    from lerobot.lerobot_types import TransitionKey
    from lerobot.utils.constants import OBS_STATE
    OFFICIAL_AVAILABLE = True
except ImportError:
    OFFICIAL_AVAILABLE = False


@unittest.skipUnless(OFFICIAL_AVAILABLE, 'requires installed LeRobot with relative action processor')
class OfficialProcessorTests(unittest.TestCase):
    def test_official_061_mask_roundtrip_and_function_parity(self):
        names = [f'joint_target_{i}_rad' for i in range(1, 7)] + ['gripper_target_fraction']
        step = RelativeActionsProcessorStep(enabled=True, exclude_joints=['gripper'], action_names=names)
        mask = step._build_mask(7)
        self.assertEqual(mask, [True] * 6 + [False])
        self.assertEqual(RelativeActionsProcessorStep(enabled=True, exclude_joints=['gripper'])._build_mask(7),
                         [True] * 7)
        state = torch.randn(2, 8)
        actions = torch.randn(2, 50, 7)
        ours = relative.to_relative_actions(actions, state, input_representation=relative.ABSOLUTE)
        torch.testing.assert_close(ours, official_relative(actions, state, mask))
        converted = step({TransitionKey.OBSERVATION: {OBS_STATE: state}, TransitionKey.ACTION: actions})
        torch.testing.assert_close(converted[TransitionKey.ACTION], ours)
        restored = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)(converted)[TransitionKey.ACTION]
        torch.testing.assert_close(restored, actions)
        torch.testing.assert_close(official_absolute(ours, state, mask),
                                   relative.to_absolute_actions(ours, state, input_representation=relative.RELATIVE))

    def test_official_quantile_normalizer_roundtrip(self):
        from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        stats = {'action': {'q01': torch.tensor([-2.] * 6 + [0.]),
                            'q99': torch.tensor([2.] * 6 + [1.])}}
        kwargs = {'features': {'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
                  'norm_map': {FeatureType.ACTION: NormalizationMode.QUANTILES}, 'stats': stats}
        actions = torch.tensor([[[.2, -.3, .4, -.5, .6, -.7, .3]]])
        transition = {TransitionKey.ACTION: actions}
        norm = NormalizerProcessorStep(**kwargs)(transition)
        restored = UnnormalizerProcessorStep(**kwargs)(norm)[TransitionKey.ACTION]
        torch.testing.assert_close(restored, actions, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
