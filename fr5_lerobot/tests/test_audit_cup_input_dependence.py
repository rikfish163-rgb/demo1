"""CPU regressions for the production inputs and saved-action comparison."""

import base64
import copy
import io
import unittest

import numpy as np
from PIL import Image

from fr5_lerobot.diagnostics import audit_cup_input_dependence as audit


def request():
    pixels = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='JPEG')
    state = [.1597, .0713, .0958, 1.62, -.02, -1.17, 1., 1.]
    return {
        'task': '抓取杯子，在空中维持5秒后放下', 'rgb_jpeg_base64': base64.b64encode(buffer.getvalue()).decode(),
        'observation_context_profile': audit.history.CUP_SETDOWN_FIXED_PROFILE,
        'cup_relative_state': state, 'cup_base_xyz_mm': [-269., -759., -4.],
        'initial_cup_uv': [1361., 761.],
        'observation_history': [{'age_s': .5, 'state': list(state)}] + [None] * 5,
        'cup_tracking': {'rim_minus_jaw_mm': None, 'native_contact': 0},
        'cup_timing': {'phase': 'at_pickup', 'hover_seconds': None, 'visual': 'missing',
                       'support_gap_mm': None, 'setdown_age_s': None},
    }


def observed(original):
    result = copy.deepcopy(original)
    result['cup_tracking']['rim_minus_jaw_mm'] = [-13.5, -3.4, 13.9]
    result['cup_timing']['visual'] = 'table_height'
    return result


def reply(original, gripper):
    action = list(original['cup_relative_state'][:6]) + [gripper]
    return {'wire_action_representation': 'cup-relative-tcp-v1', 'actions': [list(action) for _ in range(50)],
            'inference_seconds': .3}


class InputDependenceTests(unittest.TestCase):
    def test_missing_rim_changes_conditions_but_preserves_fixed_rgb_and_state(self):
        original = request()
        changed = observed(original)
        result = audit.compare_requests(original, changed)
        self.assertTrue(result['rgb_crop_equal'])
        self.assertEqual(result['rgb_crop_different_elements'], 0)
        self.assertEqual(result['rgb_crop_shape'], [1, 3, 256, 384])
        self.assertTrue(result['model_state_equal'])
        self.assertTrue(result['cup_coordinate_origin_equal'])
        self.assertFalse(result['prompt_equal'])
        self.assertEqual([row['condition_index'] for row in result['changed_conditions']], [11, 14, 16, 17, 18, 19])
        self.assertEqual(result['action_input_condition_elements_changed'], 6 * 50)
        self.assertTrue(result['seven_motion_channels_preserved_by_hook'])
        self.assertTrue(result['fixed_conditions_equal_across_denoising_steps'])
        self.assertFalse(result['model_prediction_recomputed'])

    def test_missing_rim_keeps_valid_contact_and_allows_a_saved_close_draw(self):
        original = request()
        self.assertFalse(audit.close_at_pickup(original))
        selected = audit.replay_selection(original, reply(original, .76), [0., 0., 1.])
        self.assertEqual(selected['status'], 'first_draw_selected')
        holding = copy.deepcopy(original)
        holding['cup_tracking']['native_contact'] = 1
        holding['cup_timing']['phase'] = 'grasp_contact'
        _, _, _, conditions = audit.input_features(holding)
        self.assertEqual(float(conditions[0, 19]), 0.)
        self.assertEqual(float(conditions[0, 20]), 1.)
        self.assertEqual(float(conditions[0, 21]), 1.)

    def test_restored_rim_requires_a_new_draw_for_late_close_without_fabricating_it(self):
        original = request()
        restored = observed(original)
        saved = reply(original, 1.)
        self.assertEqual(audit.replay_selection(original, saved, [0., 0., 1.])['status'], 'first_draw_selected')
        self.assertEqual(audit.replay_selection(restored, saved, [0., 0., 1.])['status'], 'additional_draws_required')
        self.assertEqual(saved['actions'][0][6], 1.)
        self.assertEqual(audit.replay_selection(restored, reply(original, .76), [0., 0., 1.])['status'], 'first_draw_selected')


if __name__ == '__main__':
    unittest.main()
