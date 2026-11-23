"""Check the gradient decision for active and tolerated translation errors."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))
import score_cup_terminal_gradients as scoring


def case(radius, nonzero):
    groups = {name: {'trainable_tensors': 2, 'gradient_tensors': 2, 'nonfinite_gradient_tensors': 0,
                     'gradient_l2_norm': 1. if nonzero else 0.,
                     'finite_nonzero_gradient_tensors': 2 if nonzero else 0} for name in scoring.GROUPS}
    excess = max(0., (radius ** 2 + 1e-8) ** .5 - .25)
    loss = 6. * (.5 * excess ** 2 if excess < 1. else excess - .5)
    actions = [[radius] + [0.] * 6] + [[0.] * 7 for _ in range(49)]
    return {'case_id': 'original-source0', 'source_index': 0,
            'terminal_actions_normalized': actions, 'native_actions_normalized': deepcopy(actions),
            'target_prefix_normalized': [[0.] * 7 for _ in range(3)], 'action_scale': [.001] * 7,
            'temporal_weights': [1.] * 3, 'loss_normalizer': 1., 'terminal_loss': loss,
            'gradients': {'groups': groups, 'required_groups_finite_nonzero': nonzero,
                         'expert_lora_frozen_without_gradients': True,
                         'unexpected_trainable': [], 'unexpected_gradients': []},
            'invariants_passed': True, 'comparison': {'dtype_equal': True},
            'terminal_requires_grad': True, 'terminal_has_grad_fn': True,
            'native_requires_grad': False, 'native_has_grad_fn': False}


class GradientAssessmentTests(unittest.TestCase):
    def test_tolerated_zero_gradient_and_active_nonzero_have_distinct_evidence(self):
        self.assertEqual(scoring.assess_case(case(.2, False))['status'], 'within_margin_zero_gradient')
        self.assertEqual(scoring.assess_case(case(2., True))['status'], 'finite_nonzero_gradient')

    def test_zero_gradient_outside_margin_is_failed(self):
        self.assertEqual(scoring.assess_case(case(2., False))['status'], 'failed')

    def test_sampler_difference_and_nonfinite_gradient_are_failed(self):
        value = case(2., True)
        value['native_actions_normalized'][-1][0] = 1.
        self.assertEqual(scoring.assess_case(value)['status'], 'failed')
        value = case(2., True)
        value['gradients']['groups']['vision_lora']['nonfinite_gradient_tensors'] = 1
        self.assertEqual(scoring.assess_case(value)['status'], 'failed')

    def test_all_tolerated_cases_cannot_prove_gradient_connectivity(self):
        report = {'status': 'failed', 'cases': [case(.2, False)],
                  'peak_allocated_bytes': 1, 'peak_reserved_bytes': 1}
        result = scoring.assess(report, {'exit_code': 1})
        self.assertEqual(result['status'], 'inconclusive_no_positive_loss')
        self.assertFalse(result['ready_for_paired_training'])
        report['cases'].append(case(2., True))
        result = scoring.assess(report, {'exit_code': 1})
        self.assertTrue(result['ready_for_paired_training'])
        self.assertEqual(result['raw_probe_exit_code'], 1)
        self.assertFalse(result['task_success_verified'])


if __name__ == '__main__':
    unittest.main()
