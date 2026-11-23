#!/usr/bin/env python3
"""Compare original and released CupTiming phases on paired PGI observations."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'fr5_lerobot' / 'src'))

from lerobot_robot_fairino.cup_action_selection import choose_proposal
from evaluate_cup_timing_policy import (
    INPUT_SCHEMA, SETDOWN_FIXED_PROFILE, model_request, read_json, validate_actions, write_new,
)


SERVER_ROOT = Path('/home/data4/hts/demo1/fr5-learning')
PROFILE = SETDOWN_FIXED_PROFILE
PHASE_PATH = ('timing_setdown', 'state', 'phase')
CLOCK = 'observation.cup_action_clock'
FIXED = 'observation.fixed_cup_action_clock'
TOKENS = 'observation.language.tokens'
MASK = 'observation.language.attention_mask'
PREFIX = 3


def differences(original, corrected, path=()):
    """Return exact JSON field paths, including changes to missing values."""
    if isinstance(original, dict) and isinstance(corrected, dict):
        for key in sorted(original.keys() | corrected.keys()):
            if key not in original or key not in corrected:
                yield path + (key,)
            else:
                yield from differences(original[key], corrected[key], path + (key,))
    elif isinstance(original, list) and isinstance(corrected, list):
        if len(original) != len(corrected):
            yield path
        else:
            for i, (a, b) in enumerate(zip(original, corrected)):
                yield from differences(a, b, path + (str(i),))
    elif type(original) is not type(corrected) or original != corrected:
        yield path


def recovery_role(request):
    state = request['cup_relative_state']
    if state[7] != 1. or request['state'][6:] != state[6:]:
        raise ValueError('Expected consistent, valid recovered gripper feedback')
    contact = request['cup_tracking']['native_contact']
    if state[6] >= .98 and contact == 0:
        return 'open_release_recovery'
    if state[6] < .88 and contact == 1:
        return 'contact_recovery_control'
    raise ValueError('Expected recovered full-open or grasp-contact observations')


def pair_inputs(original, corrected):
    indexed = []
    for inputs in (original, corrected):
        if inputs.get('schema') != INPUT_SCHEMA or inputs.get('timing_ready') is not True:
            raise ValueError('Expected prepared CupTiming replay inputs')
        cases = inputs['cases']
        by_id = {case['case_id']: (i, case) for i, case in enumerate(cases)}
        if not cases or len(by_id) != len(cases):
            raise ValueError('Expected nonempty cases with unique case_id values')
        indexed.append(by_id)
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError('Original and corrected case_id sets differ')
    pairs = []
    for case_id, (i, before) in indexed[0].items():
        j, after = indexed[1][case_id]
        changed = list(differences(before, after))
        outside_phase = [path for path in changed if path != PHASE_PATH]
        if outside_phase:
            raise ValueError(f'{case_id}: paired inputs differ at /' + '/'.join(outside_phase[0]))
        if before.get('group') != 'heldout_source_7' or before.get('kind') != 'feedback_recovered':
            raise ValueError(f'{case_id}: expected held-out PGI recovery cases')
        requests = [model_request(case, 'candidate', 'native', profile=PROFILE) for case in (before, after)]
        if any(request['task'] != case['request']['task'] for request, case in zip(requests, (before, after))):
            raise ValueError(f'{case_id}: expected the original recorded task')
        request_changes = list(differences(*requests))
        if any(path != ('cup_timing', 'phase') for path in request_changes):
            raise ValueError(f'{case_id}: model requests differ outside cup_timing.phase')
        role = recovery_role(requests[0])
        phases = [request['cup_timing']['phase'] for request in requests]
        if role == 'open_release_recovery':
            if phases[0] != 'at_pickup' or phases[1] != 'released':
                raise ValueError(f'{case_id}: expected at_pickup to released after full-open recovery')
        elif changed:
            raise ValueError(f'{case_id}: contact recovery controls must remain unchanged')
        pairs.append({'case_id': case_id, 'original_case_index': i, 'corrected_case_index': j,
                      'case': before, 'original_request': requests[0], 'corrected_request': requests[1],
                      'recovery_role': role, 'phase_changed': bool(changed)})
    if not any(pair['phase_changed'] for pair in pairs):
        raise ValueError('Expected at least one released-phase intervention')
    return pairs


def selector_classification(trace):
    if trace is None:
        return {'status': 'not_applied', 'applied': False, 'matched': None, 'raw_first_matched': None}
    return {'status': 'matched' if trace['matched'] else 'unmatched', 'applied': True,
            'matched': trace['matched'], 'raw_first_matched': trace['proposals'][0]['matched']}


def predict_condition(model, request, normal):
    candidates, batches = [], []
    sample = model.training.predict_action_chunk

    def capture_batch(policy, batch):
        if batch.get(FIXED) is not True or tuple(batch[CLOCK].shape) != (1, 25):
            raise ValueError('Expected the fixed 25-value cup action clock')
        for key in (TOKENS, MASK, 'observation.state'):
            if key not in batch:
                raise ValueError(f'Missing preprocessed input: {key}')
        batches.append({key: value.detach().cpu().clone() for key, value in batch.items()
                        if model.torch.is_tensor(value)})
        return sample(policy, batch)

    def capture_predict(observation):
        prior = len(batches)
        reply = model.predict(observation)
        validate_actions(reply['actions'])
        if len(batches) != prior + 1:
            raise ValueError('Expected one preprocessed action chunk per complete draw')
        candidates.append(reply)
        return reply

    model.training.predict_action_chunk = capture_batch
    started = time.perf_counter()
    try:
        with redirect_stdout(sys.stderr):
            selected, count, changed = choose_proposal(
                SimpleNamespace(predict=capture_predict), deepcopy(request), normal, check_boundaries=True)
    finally:
        model.training.predict_action_chunk = sample
    trace = selected.get('cup_proposal_selection')
    index = trace['selected_index'] if trace is not None else 0
    if not 1 <= count <= 3 or count != len(candidates) or not 0 <= index < count:
        raise ValueError('Expected at most three complete candidates and a valid selected index')
    if selected['actions'] != candidates[index]['actions'] or changed != (index > 0):
        raise ValueError('Selected actions differ from the complete candidate')
    if trace is not None:
        if trace['draws'] != count or len(trace['proposals']) != count or any(
                draw['actions'] != candidate['actions'] for draw, candidate in zip(trace['proposals'], candidates)):
            raise ValueError('Selector trace differs from the complete candidate draws')
    elif count != 1:
        raise ValueError('Multiple draws require a selector trace')
    return {'raw_first_reply': candidates[0], 'selected_reply': selected, 'candidate_replies': candidates,
            'selector': selector_classification(trace), 'selector_trace': trace, 'draw_count': count,
            'new_draw_selected': changed, 'raw_first_selected_actions_equal': candidates[0]['actions'] == selected['actions'],
            'predict_elapsed_seconds': time.perf_counter() - started}, batches


def tensor_difference(torch, before, after):
    same_shape = before.shape == after.shape
    return {'original': before.tolist(), 'corrected': after.tolist(),
            'original_shape': list(before.shape), 'corrected_shape': list(after.shape),
            'elementwise_equal': same_shape and before.dtype == after.dtype and torch.equal(before, after),
            'changed_indices': (before != after).nonzero().tolist() if same_shape else None}


def input_channel_difference(torch, before, after):
    channels = (CLOCK, TOKENS, MASK)
    other = sorted(key for key in before if key not in channels)
    if before.keys() != after.keys() or any(
            before[key].dtype != after[key].dtype or not torch.equal(before[key], after[key]) for key in other):
        raise ValueError('Paired preprocessed state or image tensors differ outside phase channels')
    return {'channels': {key: tensor_difference(torch, before[key], after[key]) for key in channels},
            'fixed_cup_action_clock': True, 'other_tensor_keys': other,
            'other_tensors_elementwise_equal': True}


def late_open_score(case, request, actions):
    first = next((tick for tick, action in enumerate(actions[:PREFIX], 1) if action[6] >= .98), None)
    result = {'assessed': False, 'late_or_missing': None, 'first_full_open_tick': first}
    if request['cup_relative_state'][6] >= .98:
        return dict(result, status='already_open_at_observation')
    labels = case.get('targets_camera_aligned', case.get('targets', []))[:PREFIX]
    if not labels:
        return dict(result, status='missing_release_labels')
    expected = next((tick for tick, row in enumerate(labels, 1) if row['cup_relative_action'][6] >= .98), None)
    if expected is None or expected + 1 > len(labels):
        return dict(result, status='release_deadline_outside_prefix')
    deadline = expected + 1
    if any(row.get('cut_after', False) for row in labels[:deadline - 1]):
        return dict(result, status='compaction_cut_before_release_deadline')
    return dict(result, assessed=True, expected_open_tick=expected, deadline_tick=deadline,
                late_or_missing=first is None or first > deadline, status='assessed_against_recorded_labels')


def score_prefix(case, request, actions):
    prefix = actions[:PREFIX]
    anchor = request['cup_relative_state'][:3]
    role = recovery_role(request)
    released = role == 'open_release_recovery'
    early = [tick for tick, action in enumerate(prefix, 1) if action[6] < .88]
    full_open = [tick for tick, action in enumerate(prefix, 1) if action[6] >= .98]
    displacement = [1000. * math.dist(anchor, action[:3]) for action in prefix]
    gap = request['cup_timing']['support_gap_mm']
    contact = request['cup_tracking']['native_contact']
    support = {'assessed': False, 'within_1_5mm': None, 'observed_gap_mm': gap}
    if gap is None:
        support['status'] = 'missing_support'
    elif contact != 1:
        support['status'] = 'no_current_contact_reference'
    else:
        normal = case['initial_cup']['table_normal']
        gaps = [gap + 1000. * sum((x - y) * n for x, y, n in zip(action[:3], anchor, normal)) for action in prefix]
        maximum = max(abs(value) for value in [gap, *gaps])
        support.update(assessed=True, status='observed_contact_reference', max_abs_prefix_gap_mm=maximum,
                       within_1_5mm=maximum <= 1.5)
    return {'prefix_points': PREFIX, 'recovery_role': role,
            'predicted_gripper_percent': [action[6] * 100. for action in prefix],
            'below_88_ticks': early, 'full_open_ge_98_ticks': full_open,
            'release_open_assessed': released, 'early_close': bool(early) if released else None,
            'all_full_open': len(full_open) == PREFIX if released else None,
            'displacement_mm_by_tick': displacement, 'max_first_three_displacement_mm': max(displacement),
            'support': support, 'late_open': late_open_score(case, request, actions)}


def action_difference(before, after):
    return {'complete_actions_equal': before == after,
            'max_first_three_xyz_difference_mm': 1000. * max(math.dist(a[:3], b[:3])
                                                            for a, b in zip(before[:PREFIX], after[:PREFIX])),
            'max_first_three_gripper_difference_percent': 100. * max(abs(a[6] - b[6])
                                                                     for a, b in zip(before[:PREFIX], after[:PREFIX]))}


def predict_pairs(model, pairs, seeds, *, task_override=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and distinct')
    if (model.info['observation_context_profile'] != PROFILE or model.info['n_action_steps'] != PREFIX
            or model.info['action_representation'] != 'cup-relative-tcp-tick-delta-v1'):
        raise ValueError('Expected the fixed setdown profile, 50 x 7 cup tick-delta actions and prefix3')
    results = []
    for number, pair in enumerate(pairs, 1):
        case = pair['case']
        requests = {name: deepcopy(pair[name + '_request']) for name in ('original', 'corrected')}
        if task_override is not None:
            for request in requests.values():
                request['task'] = task_override
        result = {key: pair[key] for key in ('case_id', 'original_case_index', 'corrected_case_index',
                                             'recovery_role', 'phase_changed')}
        result.update(group=case['group'], kind=case['kind'], source_frame_index=case['source_frame_index'],
                      phase_path='/' + '/'.join(PHASE_PATH), seeds=[],
                      original_recorded_task=case['request']['task'], task_override=task_override,
                      original_request=requests['original'], corrected_request=requests['corrected'])
        for seed in seeds:
            row, snapshots = {'seed': seed}, {}
            for condition in ('original', 'corrected'):
                request = requests[condition]
                model.reset()
                model.torch.manual_seed(seed)
                prediction, snapshots[condition] = predict_condition(model, request, case['initial_cup']['table_normal'])
                prediction['scores'] = {view: score_prefix(case, request, prediction[view + '_reply']['actions'])
                                        for view in ('raw_first', 'selected')}
                row[condition] = prediction
            audits = []
            for index, (a, b) in enumerate(zip(row['original']['candidate_replies'], row['corrected']['candidate_replies'])):
                sampling = [reply.get('sampling', {}) for reply in (a, b)]
                states = [value.get('cuda_rng_state') for value in sampling]
                if any(state is None for state in states) or states[0] != states[1]:
                    raise ValueError('Paired complete draws have different or missing pre-sampling CUDA RNG states')
                if sampling[0]['denoising_steps'] != sampling[1]['denoising_steps']:
                    raise ValueError('Paired denoising step counts differ')
                audits.append({'draw_index': index, 'pre_sampling_cuda_rng_equal': True,
                               'preprocess': input_channel_difference(model.torch, snapshots['original'][index],
                                                                      snapshots['corrected'][index])})
            row['paired_draw_audits'] = audits
            row['unpaired_draw_counts'] = {name: row[name]['draw_count'] - len(audits) for name in ('original', 'corrected')}
            row['differences'] = {view: action_difference(row['original'][view + '_reply']['actions'],
                                                         row['corrected'][view + '_reply']['actions'])
                                  for view in ('raw_first', 'selected')}
            result['seeds'].append(row)
        results.append(result)
        print(f'release phase probe: {number}/{len(pairs)} cases', file=sys.stderr, flush=True)
    return results


def summarize(results):
    rows = [row for case in results for row in case['seeds']]
    summary = {}
    for condition in ('original', 'corrected'):
        summary[condition] = {'selector_status': dict(Counter(row[condition]['selector']['status'] for row in rows))}
        for view in ('raw_first', 'selected'):
            scores = [row[condition]['scores'][view] for row in rows]
            release = [score for score in scores if score['release_open_assessed']]
            summary[condition][view] = {
                'predictions': len(scores), 'release_open_assessed': len(release),
                'contact_recovery_controls': len(scores) - len(release),
                'early_close': sum(score['early_close'] is True for score in release),
                'all_full_open': sum(score['all_full_open'] is True for score in release),
                'max_first_three_displacement_mm': max(score['max_first_three_displacement_mm'] for score in scores),
                'release_max_first_three_displacement_mm': max((score['max_first_three_displacement_mm'] for score in release), default=None),
                'support_assessed': sum(score['support']['assessed'] for score in scores),
                'support_within_1_5mm': sum(score['support']['within_1_5mm'] is True for score in scores),
                'support_status': dict(Counter(score['support']['status'] for score in scores)),
                'late_open_assessed': sum(score['late_open']['assessed'] for score in scores),
                'late_or_missing_open': sum(score['late_open']['late_or_missing'] is True for score in scores),
                'late_open_status': dict(Counter(score['late_open']['status'] for score in scores)),
            }
    return summary


def output_path(path, checkpoint):
    path = path.resolve()
    if not path.is_relative_to(SERVER_ROOT.resolve()):
        raise ValueError('Output must be a new file under the server learning directory')
    if path.exists():
        raise FileExistsError(path)
    if checkpoint.resolve() in path.parents:
        raise ValueError('Output must be outside the checkpoint')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-inputs', type=Path, required=True)
    parser.add_argument('--corrected-inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, action='append', required=True)
    parser.add_argument('--task-override', help='Use the same task condition for both phases')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = output_path(args.output, args.checkpoint)
    if len(set(args.seed)) != len(args.seed):
        raise ValueError('Seeds must be distinct')
    pairs = pair_inputs(read_json(args.original_inputs), read_json(args.corrected_inputs))
    sys.path.insert(0, str(args.checkpoint))
    with redirect_stdout(sys.stderr):
        from serve_stream_pi05 import Inference
        model = Inference(args.checkpoint, gpu=args.gpu)
    model.torch.set_num_threads(4)
    results = predict_pairs(model, pairs, args.seed, task_override=args.task_override)
    report = {'schema': 'fr5-cup-release-phase-probe-v1',
              'original_inputs': str(args.original_inputs.resolve()), 'corrected_inputs': str(args.corrected_inputs.resolve()),
              'checkpoint': str(model.root), 'gpu': args.gpu, 'seeds': args.seed,
              'observation_context_profile': PROFILE, 'action_shape': [50, 7], 'prefix_points': PREFIX,
              'max_complete_candidates': 3, 'denoising_steps': model.policy.config.num_inference_steps,
              'task_override': args.task_override, 'timing_mode': 'native',
              'selector': 'bounded_3_complete_proposals_at_current_cup_boundaries_v2',
              'contrast': 'Only /timing_setdown/state/phase changes; original task, observations and paired manual_seed.',
              'thresholds': {'early_close_below_fraction': .88, 'full_open_at_least_fraction': .98},
              'robot_connections': 0, 'robot_commands': 0, 'task_success': False,
              'results': results, 'summary': summarize(results),
              'limitations': ['Offline predictions establish no physical motion or task success.',
                              'Missing support and release deadlines outside prefix3 remain unassessed.']}
    write_new(output, report)
    print(json.dumps({'output': str(output), 'summary': report['summary'], 'robot_commands': 0,
                      'task_success': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
