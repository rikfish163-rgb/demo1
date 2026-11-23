import unittest
from copy import deepcopy
from unittest.mock import Mock

from lerobot_robot_fairino.stream_runtime import _CUP_BOUNDARY_WEIGHTS, CupProposalRejected, predict_stream_reply
from fr5_lerobot.training.stream_observation_history import CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE


def request(gap, age, profile=CUP_SETDOWN_PROFILE):
    return {'observation_context_profile': profile,
            'cup_relative_state': [.1597, .0713, .0958 + gap / 1000., 0., 0., 0., .76, 1.],
            'cup_timing': {'phase': 'returning', 'support_gap_mm': gap, 'setdown_age_s': age},
            'cup_tracking': {'native_contact': 1, 'rim_minus_jaw_mm': None}}


def reply(height, opening):
    return {'actions': [[.1597, .0713, height, 0., 0., 0., opening] for _ in range(50)],
            'inference_seconds': .35}


class CupActionSelectionTests(unittest.TestCase):
    def test_far_single_tick_close_is_replaced_by_a_complete_model_draw(self):
        cup = {'table_normal': [0., 0., 1.]}
        for profile in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            with self.subTest(profile=profile):
                policy = Mock(n_action_steps=3, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                observation = request(40.9, None, profile)
                observation['cup_relative_state'][:3] = [.202801, .086487, .136571]
                observation['cup_relative_state'][6] = 1.
                observation['cup_timing']['phase'] = 'approaching'
                observation['cup_tracking']['native_contact'] = 0
                observation['observation_history'] = [None] * 6
                unsafe = reply(.1367, 1.)
                unsafe['sampling'] = {'cuda_rng_state': list(range(16)), 'denoising_steps': 10}
                unsafe['actions'][0] = [.201577, .085674, .136674, 0., 0., 0., .739983]
                safe = reply(.1367, 1.)
                safe['actions'][0][:2] = [.201577, .085674]
                policy.predict.side_effect = [unsafe, safe]
                selected = predict_stream_reply(policy, observation, cup)
                self.assertIs(selected['actions'], safe['actions'])
                self.assertEqual(selected['cup_proposal_selection']['selected_index'], 1)
                draws = selected['cup_proposal_selection']['proposals']
                self.assertIs(draws[0]['actions'], unsafe['actions'])
                self.assertFalse(draws[0]['matched'])
                self.assertEqual(draws[0]['sampling'], unsafe['sampling'])
                self.assertIs(draws[1]['actions'], safe['actions'])
                self.assertEqual(policy.predict.call_count, 2)

    def test_far_close_with_no_feasible_draw_stops(self):
        cup = {'table_normal': [0., 0., 1.]}
        for profile in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            with self.subTest(profile=profile):
                policy = Mock(n_action_steps=3, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                observation = request(40.9, None, profile)
                observation['cup_relative_state'][:3] = [.202801, .086487, .136571]
                observation['cup_relative_state'][6] = 1.
                observation['cup_timing']['phase'] = 'approaching'
                observation['cup_tracking']['native_contact'] = 0
                observation['observation_history'] = [None] * 6
                unsafe = reply(.1367, .74)
                with self.assertRaisesRegex(CupProposalRejected, 'three draws') as rejection:
                    policy.predict.return_value = unsafe
                    predict_stream_reply(policy, observation, cup)
                self.assertEqual(policy.predict.call_count, 3)
                self.assertFalse(rejection.exception.reply['cup_proposal_selection']['matched'])
                self.assertEqual(len(rejection.exception.reply['cup_proposal_selection']['proposals']), 3)

    def test_normal_pickup_close_is_selected_without_an_extra_draw(self):
        cup = {'table_normal': [0., 0., 1.]}
        for profile in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            with self.subTest(profile=profile):
                policy = Mock(n_action_steps=3, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                observation = request(0., None, profile)
                observation['cup_relative_state'][6] = 1.
                observation['cup_timing']['phase'] = 'at_pickup'
                observation['cup_tracking'].update(native_contact=0, rim_minus_jaw_mm=[-13.5, -3.4, 13.9])
                observation['observation_history'] = [{'age_s': .5, 'state': observation['cup_relative_state']}]
                close = reply(.0958, .76)
                raw_actions = deepcopy(close['actions'])
                policy.predict.return_value = close
                selected = predict_stream_reply(policy, observation, cup)
                self.assertIs(selected['actions'], close['actions'])
                self.assertEqual(selected['actions'], raw_actions)
                self.assertEqual([len(row) for row in selected['actions']], [7] * 50)
                selection = selected['cup_proposal_selection']
                self.assertTrue(selection['matched'])
                self.assertEqual(selection['draws'], 1)
                self.assertEqual(selection['selected_index'], 0)
                self.assertIs(selection['proposals'][0]['actions'], close['actions'])
                self.assertEqual(policy.predict.call_count, 1)

    def test_unstable_hold_and_stable_open_each_consume_a_whole_draw(self):
        cup = {'table_normal': [0., 0., 1.]}
        for profile in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            with self.subTest(profile=profile):
                policy = Mock(n_action_steps=3, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                bad_high, safe_hold = reply(.1008, 1.), reply(.1007, .76)
                raw_proposals = deepcopy([bad_high, safe_hold])
                policy.predict.side_effect = [bad_high, safe_hold]
                selected = predict_stream_reply(policy, request(5., None, profile), cup)
                self.assertIs(selected['actions'], safe_hold['actions'])
                selection = selected['cup_proposal_selection']
                self.assertEqual(selection['selected_index'], 1)
                self.assertEqual(selection['draws'], 2)
                for draw, proposal in zip(selection['proposals'], (bad_high, safe_hold)):
                    self.assertIs(draw['actions'], proposal['actions'])
                    self.assertEqual([len(row) for row in draw['actions']], [7] * 50)
                self.assertFalse(selection['proposals'][0]['matched'])
                self.assertTrue(selection['proposals'][1]['matched'])
                self.assertEqual([bad_high, safe_hold], raw_proposals)
                self.assertEqual(policy.predict.call_count, 2)
                policy.reset_mock()
                late, ready = reply(.0958, .76), reply(.0958, 1.)
                policy.predict.side_effect = [late, ready]
                selected = predict_stream_reply(policy, request(0., .5, profile), cup)
                self.assertIs(selected['actions'], ready['actions'])
                self.assertEqual(selected['cup_proposal_selection']['selected_index'], 1)
                self.assertEqual(policy.predict.call_count, 2)

    def test_exhausted_draws_stop_and_preserve_every_proposal(self):
        cup = {'table_normal': [0., 0., 1.]}
        for profile in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            with self.subTest(profile=profile):
                policy = Mock(n_action_steps=3, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                unsafe = [reply(.1008 + index * .0001, 1.) for index in range(3)]
                raw_proposals = deepcopy(unsafe)
                policy.predict.side_effect = unsafe
                with self.assertRaisesRegex(CupProposalRejected, 'three draws') as rejection:
                    predict_stream_reply(policy, request(5., None, profile), cup)
                rejected = rejection.exception.reply
                self.assertIs(rejected['actions'], unsafe[0]['actions'])
                self.assertAlmostEqual(rejected['inference_seconds'], 1.05)
                selection = rejected['cup_proposal_selection']
                self.assertFalse(selection['matched'])
                self.assertEqual(selection['draws'], 3)
                self.assertEqual(selection['selected_index'], 0)
                self.assertEqual(len(selection['proposals']), 3)
                for draw, proposal in zip(selection['proposals'], unsafe):
                    self.assertIs(draw['actions'], proposal['actions'])
                    self.assertEqual([len(row) for row in draw['actions']], [7] * 50)
                    self.assertFalse(draw['matched'])
                self.assertEqual(unsafe, raw_proposals)
                self.assertEqual(policy.predict.call_count, 3)

    def test_unrelated_original_checkpoint_is_not_retried(self):
        cup = {'table_normal': [0., 0., 1.]}
        policy = Mock(n_action_steps=3, checkpoint_sha='other')
        unsafe = reply(.1008, 1.)
        policy.predict.return_value = unsafe
        observation = request(5., None)
        self.assertIs(predict_stream_reply(policy, observation, cup), unsafe)
        policy.predict.assert_called_once_with(observation)

    def test_fixed_other_prefix_lengths_predict_once(self):
        for steps in (1, 2, 4, 10, 50):
            with self.subTest(steps=steps):
                policy = Mock(n_action_steps=steps, checkpoint_sha=_CUP_BOUNDARY_WEIGHTS[0])
                unsafe = reply(.1008, 1.)
                policy.predict.return_value = unsafe
                observation = request(5., None, CUP_SETDOWN_FIXED_PROFILE)
                self.assertIs(predict_stream_reply(policy, observation, None), unsafe)
                policy.predict.assert_called_once_with(observation)
