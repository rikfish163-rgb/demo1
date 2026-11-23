#!/usr/bin/env python3
"""Project archived cup approach draws from saved visual rim observations."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fr5_jev import cup_tracking


# Existing close_at_pickup references, in base axes.
RIM_REFERENCE_MM = np.array([-13.5, -3.4, 13.9])
FIXED_PICKUP_M = np.array([.1597, .0713, .0958])
# Same below-88% close threshold as score_cup_boundary_replay.
CLOSE = .88


def finite_array(values, shape):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.shape == shape and np.isfinite(result).all() else None


def jaw_tip(pose):
    """Use CupTracker's tool0/user0 tip offset with a cup-relative m/rad pose."""
    flange = np.r_[pose[:3] * 1000., np.rad2deg(pose[3:6])]
    return cup_tracking._jaw_center({'flange': flange, 'tool': 0, 'user': 0})


def observation(request):
    state = finite_array(request.get('cup_relative_state'), (8,))
    tracking = request.get('cup_tracking') or {}
    raw_rim = tracking.get('rim_minus_jaw_mm')
    rim = finite_array(raw_rim, (3,))
    contact = tracking.get('native_contact')
    phase = (request.get('cup_timing') or {}).get('phase')
    reasons = []
    if state is None:
        reasons.append('invalid_state')
    elif state[7] != 1. or not 0. <= state[6] <= 1.:
        reasons.append('invalid_gripper')
    if raw_rim is None:
        reasons.append('missing_rim')
    elif rim is None:
        reasons.append('invalid_rim')
    if contact is None:
        reasons.append('missing_contact')
    elif contact == 1:
        reasons.append('contact')
    elif contact != 0:
        reasons.append('invalid_contact')
    if phase not in ('approaching', 'at_pickup'):
        reasons.append('ineligible_phase')
    return state, rim, reasons


def score_draw(request, actions):
    """Assess prefix3 only when every tick precedes the first close target."""
    values = finite_array(actions, (50, 7))
    if values is None:
        raise ValueError('Expected a complete finite 50 x 7 wire action chunk')
    state, rim, reasons = observation(request)
    first_close_tick = next((tick for tick, action in enumerate(values, 1) if action[6] < CLOSE), None)
    tick_assessments = []
    for tick in range(1, 4):
        tick_reasons = reasons + (['at_or_after_close']
                                 if first_close_tick is not None and tick >= first_close_tick else [])
        tick_assessments.append({'tick': tick, 'assessment': 'unassessed' if tick_reasons else 'assessed',
                                 'unassessed_reasons': tick_reasons})
    assessed_ticks = sum(t['assessment'] == 'assessed' for t in tick_assessments)
    prefix_reasons = reasons + (['at_or_after_close']
                               if first_close_tick is not None and first_close_tick <= 3 else [])
    result = {'assessment': 'assessed' if assessed_ticks == 3 else 'unassessed',
              'unassessed_reasons': prefix_reasons, 'action_prefix3': values[:3].tolist(),
              'first_close_tick': first_close_tick, 'assessed_prefix_ticks': assessed_ticks,
              'tick_assessments': tick_assessments,
              'before_error_mm': None, 'projected_rim_minus_jaw_mm': None,
              'projected_error_mm': None, 'prefix3_progress_mm': None,
              'fixed_pickup_diagnostic': None}
    if state is not None:
        before = float(np.linalg.norm(state[:3] - FIXED_PICKUP_M) * 1000.)
        errors = np.linalg.norm(values[:3, :3] - FIXED_PICKUP_M, axis=1) * 1000.
        result['fixed_pickup_diagnostic'] = {
            'before_error_mm': before, 'projected_error_mm': errors.tolist(),
            'prefix3_progress_mm': before - float(errors[2])}
    if reasons:
        return result
    # The cup-frame origin cancels; the observed vector keeps the base axes.
    current_jaw = jaw_tip(state)
    before = float(np.linalg.norm(rim - RIM_REFERENCE_MM))
    projected = np.array([rim - (jaw_tip(action) - current_jaw)
                          for action in values[:assessed_ticks]]).reshape(-1, 3)
    errors = np.linalg.norm(projected - RIM_REFERENCE_MM, axis=1).tolist()
    result.update(before_error_mm=before,
                  projected_rim_minus_jaw_mm=projected.tolist() + [None] * (3 - assessed_ticks),
                  projected_error_mm=errors + [None] * (3 - assessed_ticks),
                  prefix3_progress_mm=before - errors[2] if assessed_ticks == 3 else None)
    return result


def score_row(row):
    request, reply = row.get('request'), row.get('reply')
    if not isinstance(request, dict) or not isinstance(reply, dict):
        raise ValueError('Each prediction requires its complete saved request and reply')
    if reply.get('wire_action_representation') != 'cup-relative-tcp-v1':
        raise ValueError('Expected absolute cup-relative TCP wire actions in m/rad')
    trace = reply.get('cup_proposal_selection')
    proposals = trace['proposals'] if trace else [{'actions': reply['actions']}]
    selected = trace['selected_index'] if trace else 0
    if (not 1 <= len(proposals) <= 3 or type(selected) is not int
            or not 0 <= selected < len(proposals)
            or (trace and trace['draws'] != len(proposals))
            or row.get('draw_count', len(proposals)) != len(proposals)
            or reply['actions'] != proposals[selected]['actions']):
        raise ValueError('Selected reply differs from its complete saved proposal trace')
    state, rim, _ = observation(request)
    tracking = request.get('cup_tracking') or {}
    return {'case_id': row['case_id'], 'seed': row['seed'], 'group': row.get('group'),
            'kind': row.get('kind'), 'phase': (request.get('cup_timing') or {}).get('phase'),
            'cup_relative_state': state.tolist() if state is not None else None,
            'rim_minus_jaw_mm': rim.tolist() if rim is not None else None,
            'native_contact': tracking.get('native_contact'),
            'draw_count': len(proposals), 'selected_index': selected, 'rawfirst_index': 0,
            'selection_matched': trace.get('matched') if trace else None,
            'draws': [{'proposal_index': index, 'matched': proposal.get('matched'),
                       **score_draw(request, proposal['actions'])}
                      for index, proposal in enumerate(proposals)]}


def distribution(values):
    if not values:
        return None
    return {'mean_mm': float(np.mean(values)), 'median_mm': float(np.median(values)),
            'minimum_mm': float(min(values)), 'maximum_mm': float(max(values))}


def progress_summary(values):
    return {'draws': len(values), 'positive': sum(v > 0. for v in values),
            'negative': sum(v < 0. for v in values), 'zero': sum(v == 0. for v in values),
            'progress': distribution(values)}


def tick_summary(draws, index):
    visual = [d for d in draws if d['tick_assessments'][index]['assessment'] == 'assessed']
    reasons = Counter(reason for d in draws
                      for reason in d['tick_assessments'][index]['unassessed_reasons'])
    return {'tick': index + 1, 'draws': len(draws), 'assessed': len(visual),
            'unassessed': len(draws) - len(visual), 'unassessed_reasons': dict(reasons),
            'error': distribution([d['projected_error_mm'][index] for d in visual]),
            'progress': progress_summary([d['before_error_mm'] - d['projected_error_mm'][index]
                                          for d in visual])}


def summarize(rows):
    selected = [r['draws'][r['selected_index']] for r in rows]
    reasons = Counter(reason for draw in selected for reason in draw['unassessed_reasons'])
    summary = {'draws': len(rows), 'proposal_draws': sum(r['draw_count'] for r in rows),
               'selected_changed': sum(r['selected_index'] != 0 for r in rows),
               'eligible': sum(d['assessment'] == 'assessed' for d in selected),
               'missing': reasons['missing_rim'],
               'invalid': sum(any(reason.startswith('invalid_') for reason in d['unassessed_reasons'])
                              for d in selected),
               'unassessed': sum(d['assessment'] == 'unassessed' for d in selected),
               'partial_prefix3': sum(0 < d['assessed_prefix_ticks'] < 3 for d in selected),
               'unassessed_reasons': dict(reasons)}
    for label, draws in (('selected', selected), ('rawfirst', [r['draws'][0] for r in rows])):
        visual = [d for d in draws if d['assessment'] == 'assessed']
        fixed = [d['fixed_pickup_diagnostic'] for d in draws if d['fixed_pickup_diagnostic'] is not None]
        paired = Counter(('visual_negative' if d['prefix3_progress_mm'] < 0. else 'visual_nonnegative')
                         + '/' + ('fixed_negative' if d['fixed_pickup_diagnostic']['prefix3_progress_mm'] < 0.
                                  else 'fixed_nonnegative') for d in visual)
        summary[label] = {
            'before_error': distribution([d['before_error_mm'] for d in visual]),
            'projected_error_by_tick': [tick_summary(draws, tick) for tick in range(3)],
            'partial_prefix3': sum(0 < d['assessed_prefix_ticks'] < 3 for d in draws),
            'prefix3': progress_summary([d['prefix3_progress_mm'] for d in visual]),
            'fixed_pickup_diagnostic': progress_summary([d['prefix3_progress_mm'] for d in fixed]),
            'paired_direction_on_eligible': dict(paired)}
    return summary


def score(saved):
    rows = [score_row(row) for row in saved['predictions']]
    return {'checkpoint': saved.get('checkpoint'), 'seeds': saved.get('seeds'),
            'summary': summarize(rows),
            'by_case': {case_id: summarize([r for r in rows if r['case_id'] == case_id])
                        for case_id in sorted({r['case_id'] for r in rows})},
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = {}
    for item in args.prediction:
        label, separator, path = item.partition('=')
        if not label or not separator or not path or label in reports:
            parser.error('--prediction requires a unique LABEL=PATH')
        reports[label] = {'prediction': path, **score(json.loads(Path(path).read_text(encoding='utf-8')))}
    report = {
        'schema': 'fr5-cup-visual-progress-score-v1', 'reports': reports,
        'geometry': {
            'frame': 'base axes, millimetres; tool0/user0; cup-frame origin cancels',
            'wire_pose_units': 'XYZ metres, RPY radians; absolute poses, not tick deltas',
            'jaw_helper': 'fr5_jev.cup_tracking._jaw_center',
            'jaw_helper_source': str(Path(cup_tracking.__file__).resolve()),
            'nominal_jaw_tip_center_flange_mm': cup_tracking.JAW_TIP_CENTER_FLANGE_MM.tolist(),
            'rim_reference_mm': RIM_REFERENCE_MM.tolist(),
            'reference_source': 'lerobot_robot_fairino.cup_action_selection.close_at_pickup',
            'fixed_pickup_diagnostic_reference_m': FIXED_PICKUP_M.tolist(),
            'projection': 'rim_observed - (nominal_jaw_at_action - nominal_jaw_at_observation)',
            'close_threshold_fraction': CLOSE,
            'close_threshold_source': 'diagnostics.score_cup_boundary_replay.CLOSE',
            'first_close_tick': 'First of all 50 wire targets below the close threshold; '
                                'one-based, null when absent.',
            'projection_eligibility': 'Valid no-contact approach/pickup observation and tick strictly '
                                      'before first_close_tick.',
            'prefix3_progress': 'before_error - projected_error_at_tick3; positive means closer; '
                                'requires all three ticks assessed',
            'assessment': 'Draw assessment requires all three prefix ticks; partial_prefix3 is a '
                          'subset of unassessed draws.',
            'branches': 'selected = saved selected_index; rawfirst = saved proposal index 0',
        },
        'scope': 'Original saved Gemini visual observations plus rigid projections of already generated '
                 'actions, conditional on the cup remaining stationary before closure.',
        'limitations': [
            'Projected poses were not executed or measured; projections do not establish task success.',
            'The PGI-140 tip is nominal, and the existing close_at_pickup reference has calibration limits.',
            'The first close tick and all later ticks are unassessed; partial prefixes have no '
            'prefix3 progress.',
            'Missing or invalid observations and contact/other phases stay unassessed; '
            'fixed demonstration pickup distance is a separate diagnostic.',
        ],
        'robot_commands': 0, 'model_inferences': 0,
        'execution_verified': False, 'task_success_verified': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output),
                      'reports': {label: entry['summary'] for label, entry in reports.items()}},
                     ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
