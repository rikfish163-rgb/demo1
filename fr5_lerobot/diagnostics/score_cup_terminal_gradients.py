"""Assess recorded terminal gradients against the configured XYZ tolerance."""

import argparse
import json
import math
from pathlib import Path


GROUPS = ('vision_lora', 'language_lora', 'full_expert', 'action_projections')
MARGIN_MM = .25


def assess_case(case):
    actions, native = case['terminal_actions_normalized'], case['native_actions_normalized']
    full = (isinstance(actions, list) and isinstance(native, list) and len(actions) == len(native) == 50
            and all(len(a) == len(b) == 7 for a, b in zip(actions, native)))
    finite = full and all(isinstance(v, (int, float)) and math.isfinite(v)
                          for points in (actions, native) for point in points for v in point)
    if not finite or actions != native:
        return {'status': 'failed', 'reason': 'native_and_terminal_actions_differ_or_are_nonfinite'}
    radii = [math.sqrt(sum(((actions[k][i] - target[i]) * case['action_scale'][i]
                           * (k + 1) * 1000.) ** 2 for i in range(3)))
             for k, target in enumerate(case['target_prefix_normalized'])]
    numerator = sum(6. * weight * (.5 * excess ** 2 if excess < 1. else excess - .5)
                    for weight, radius in zip(case['temporal_weights'], radii)
                    for excess in [max(0., math.sqrt(radius ** 2 + 1e-8) - MARGIN_MM)])
    expected_loss = numerator / case['loss_normalizer']
    loss = case['terminal_loss']
    groups = case['gradients']['groups']
    response = case['gradients']['required_groups_finite_nonzero']
    finite_groups = all(groups[name]['trainable_tensors'] > 0 and groups[name]['gradient_tensors'] > 0
                        and groups[name]['nonfinite_gradient_tensors'] == 0 for name in GROUPS)
    zero_groups = finite_groups and all(groups[name]['gradient_l2_norm'] == 0.
                                       and groups[name]['finite_nonzero_gradient_tensors'] == 0 for name in GROUPS)
    numerical_loss_equal = (isinstance(loss, (int, float)) and math.isfinite(loss)
                            and math.isclose(loss, expected_loss, rel_tol=2e-5, abs_tol=1e-8))
    invariant = (case['invariants_passed'] and case['comparison']['dtype_equal']
                 and case['terminal_requires_grad'] and case['terminal_has_grad_fn']
                 and not case['native_requires_grad'] and not case['native_has_grad_fn']
                 and case['gradients']['expert_lora_frozen_without_gradients']
                 and not case['gradients']['unexpected_trainable']
                 and not case['gradients']['unexpected_gradients'] and not case.get('error'))
    inside = expected_loss == 0.
    passed = invariant and numerical_loss_equal and finite_groups and (
        zero_groups if inside else response)
    return {'status': ('within_margin_zero_gradient' if inside else 'finite_nonzero_gradient')
                     if passed else 'failed',
            'xyz_error_mm': radii, 'expected_terminal_loss': expected_loss,
            'recorded_terminal_loss': loss, 'within_original_margin': inside,
            'native_terminal_full_actions_equal': True, 'invariants_passed': invariant,
            'all_required_gradient_groups_finite': finite_groups}


def assess(report, receipt):
    cases = [{'case_id': case['case_id'], 'source_index': case['source_index'], **assess_case(case)}
             for case in report['cases']]
    nonzero = sum(case['status'] == 'finite_nonzero_gradient' for case in cases)
    failed = sum(case['status'] == 'failed' for case in cases)
    status = ('failed' if failed or report.get('error') or not cases else
              'passed' if nonzero else 'inconclusive_no_positive_loss')
    return {'schema': 'fr5-cup-terminal-xyz-gradient-assessment-v1', 'status': status,
            'margin_mm': MARGIN_MM, 'coefficient': 6., 'cases': cases,
            'positive_loss_cases_with_all_required_gradients': nonzero,
            'within_margin_zero_gradient_cases': sum(case['status'] == 'within_margin_zero_gradient' for case in cases),
            'failed_cases': failed, 'ready_for_paired_training': status == 'passed',
            'raw_probe_status': report['status'], 'raw_probe_exit_code': receipt['exit_code'],
            'peak_allocated_bytes': report['peak_allocated_bytes'],
            'peak_reserved_bytes': report['peak_reserved_bytes'],
            'robot_commands': 0, 'optimizer_steps': 0, 'task_success_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(json.loads(args.probe.read_text()), json.loads(args.receipt.read_text()))
    result.update(probe=str(args.probe), receipt=str(args.receipt))
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ready_for_paired_training'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
