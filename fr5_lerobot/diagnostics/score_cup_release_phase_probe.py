#!/usr/bin/env python3
"""Score saved paired CupTiming phase probes on CPU against recorded jaw targets."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'fr5_lerobot' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_cup_timing_policy import PREDICTION_SCHEMA, model_request, read_json, validate_actions, write_new
from probe_cup_release_phase import CLOCK, MASK, PROFILE, TOKENS, pair_inputs, selector_classification
from score_cup_feedback_recovery import HORIZONS, evaluate


PHASES = ('original', 'corrected')
RUNS = {(model, prompt) for model in ('fixed-v7', 'coverage-v8') for prompt in ('original', 'five-second')}
CHANNEL_WIDTHS = {CLOCK: 25, TOKENS: 512, MASK: 512}


def validate_prediction(prediction):
    candidates = prediction['candidate_replies']
    count = prediction['draw_count']
    if not 1 <= count <= 3 or count != len(candidates):
        raise ValueError('Expected at most three saved complete candidates')
    for candidate in candidates:
        validate_actions(candidate['actions'])
    first, selected = prediction['raw_first_reply'], prediction['selected_reply']
    validate_actions(first['actions'])
    validate_actions(selected['actions'])
    if first != candidates[0]:
        raise ValueError('Raw-first reply differs from the first complete candidate')
    trace = selected.get('cup_proposal_selection')
    if trace != prediction['selector_trace'] or prediction['selector'] != selector_classification(trace):
        raise ValueError('Saved selector classification differs from the selected reply trace')
    index = trace['selected_index'] if trace is not None else 0
    if not 0 <= index < count or selected['actions'] != candidates[index]['actions']:
        raise ValueError('Selected actions differ from the indexed complete candidate')
    if selected.get('sampling') != candidates[index].get('sampling'):
        raise ValueError('Selected sampling evidence differs from the complete candidate')
    if trace is None:
        if count != 1:
            raise ValueError('Multiple complete candidates require a selector trace')
    elif (trace['draws'] != count or len(trace['proposals']) != count
          or any(draw['actions'] != candidate['actions'] or draw.get('sampling') != candidate.get('sampling')
                 for draw, candidate in zip(trace['proposals'], candidates))):
        raise ValueError('Selector trace differs from the saved complete candidates')
    if prediction['new_draw_selected'] != (index > 0):
        raise ValueError('Selected draw index differs from new_draw_selected')
    if prediction['raw_first_selected_actions_equal'] != (first['actions'] == selected['actions']):
        raise ValueError('Raw-first/selected full-action equality flag differs')


def adapt_predictions(inputs, saved, phase):
    """Retain complete replies and use matched only when a selector actually ran."""
    cases = {case['case_id']: case for case in inputs['cases']}
    rows = []
    for result in saved['results']:
        case = cases[result['case_id']]
        probe = dict(case, task_override=saved['task_override']) if saved['task_override'] is not None else case
        request = model_request(probe, 'candidate', 'native', profile=saved['observation_context_profile'])
        if result[phase + '_request'] != request:
            raise ValueError('Phase prediction differs from the saved causal request')
        if any(result[key] != case[key] for key in ('group', 'kind', 'source_frame_index')):
            raise ValueError('Phase prediction differs from the held-out case identity')
        for seed in result['seeds']:
            prediction = seed[phase]
            validate_prediction(prediction)
            rows.append({'case_id': case['case_id'], 'group': case['group'], 'kind': case['kind'],
                         'seed': seed['seed'], 'request': request, 'reply': prediction['selected_reply'],
                         'draw_count': prediction['draw_count'],
                         'selection_matched': prediction['selector']['matched'] is True,
                         'new_draw_selected': prediction['new_draw_selected'],
                         'predict_elapsed_seconds': prediction['predict_elapsed_seconds']})
    return {'schema': PREDICTION_SCHEMA, 'model': 'candidate', 'timing_mode': 'native',
            'checkpoint': saved['checkpoint'], 'seeds': saved['seeds'],
            'task_override': saved['task_override'],
            'observation_context_profile': saved['observation_context_profile'], 'predictions': rows}


def score_condition(inputs, saved, phase, journal):
    adapted = adapt_predictions(inputs, saved, phase)
    report = evaluate(inputs, adapted, journal)
    selectors = {(case['case_id'], row['seed']): row[phase]['selector']
                 for case in saved['results'] for row in case['seeds']}
    results = {case['case_id']: case for case in saved['results']}
    summary = report['summary']
    summary['selector_status'] = dict(Counter(selector['status'] for selector in selectors.values()))
    summary['unknown_feedback_predictions'] = summary['unassessed']
    summary['raw_first'] = summary.pop('first')
    for view in ('raw_first', 'selected'):
        for horizon in HORIZONS:
            timing = summary[view][f'prefix_{horizon}']
            timing['late_open_unassessed'] = summary['total'] - timing['late_open_assessed']
    # Selector status comes from the saved trace; geometry is assessed separately.
    report.pop('proposal_trace_summary')
    for row in report['rows']:
        selector = selectors[row['case_id'], row['seed']]
        row['selector'] = selector
        row['reported_matched'] = selector['matched'] if row['status'] == 'assessed' else None
        row['matched_source'] = 'proposal_trace' if selector['applied'] else 'not_applied'
        row['recovery_role'] = results[row['case_id']]['recovery_role']
        row['phase'] = results[row['case_id']][phase + '_request']['cup_timing']['phase']
        row['raw_first'] = row.pop('first')
        row.pop('selected_first_three')
    report['geometry_summary'] = {
        view: {'assessed': summary['assessed'],
               'rejected': sum(row[view] is not None and not row[view]['geometry']['geometry_passed']
                               for row in report['rows']),
               'close_then_reopen': sum(row[view] is not None
                                       and row[view]['geometry']['close_then_reopen_in_three_ticks']
                                       for row in report['rows'])}
        for view in ('raw_first', 'selected')}
    return report


def channel_evidence(channel, width):
    before, after = channel['original'], channel['corrected']
    if (len(before) != 1 or len(after) != 1 or len(before[0]) != width or len(after[0]) != width):
        raise ValueError('Saved preprocessed channel has an unexpected tensor shape')
    changed = [[0, i] for i, (a, b) in enumerate(zip(before[0], after[0])) if a != b]
    consistent = (channel['original_shape'] == channel['corrected_shape'] == [1, width]
                  and channel['elementwise_equal'] == (not changed)
                  and channel['changed_indices'] == changed)
    return changed, consistent


def audit_pairs(saved):
    """Compare archived RNG values, channel values and every element of each action chunk."""
    rows = [(case, row) for case in saved['results'] for row in case['seeds']]
    counts = Counter()
    unpaired = Counter({phase: 0 for phase in PHASES})
    channels = {key: Counter() for key in CHANNEL_WIDTHS}
    changed_sizes = {key: [] for key in CHANNEL_WIDTHS}
    other_keys, issues = set(), []
    control = Counter()
    interventions = Counter()
    for case, row in rows:
        label = f"{case['case_id']}/{row['seed']}"
        contact = case['recovery_role'] == 'contact_recovery_control'
        bucket = control if contact else interventions
        bucket['case_seed_pairs'] += 1
        for view in ('raw_first', 'selected'):
            before = row['original'][view + '_reply']['actions']
            after = row['corrected'][view + '_reply']['actions']
            equal = before == after
            bucket[view + '_complete_actions_equal'] += equal
            bucket[view + '_complete_actions_changed'] += not equal
            if row['differences'][view]['complete_actions_equal'] != equal:
                issues.append({'pair': label, 'field': f'differences/{view}/complete_actions_equal'})
            if contact and not equal:
                mismatches = [(i, j) for i in range(50) for j in range(7) if before[i][j] != after[i][j]]
                issues.append({'pair': label, 'field': view + '/contact_control_actions',
                               'mismatched_elements': len(mismatches), 'first_changed_index': list(mismatches[0])})
        candidates = [row[phase]['candidate_replies'] for phase in PHASES]
        paired = min(map(len, candidates))
        audits = row['paired_draw_audits']
        if len(audits) != paired:
            raise ValueError('Saved paired-draw audits do not cover the paired complete candidates')
        for phase, replies in zip(PHASES, candidates):
            value = len(replies) - paired
            unpaired[phase] += value
            if row['unpaired_draw_counts'][phase] != value:
                issues.append({'pair': label, 'field': 'unpaired_draw_counts/' + phase})
        for index, (a, b, audit) in enumerate(zip(*candidates, audits)):
            draw_label = f'{label}/draw-{index}'
            counts['paired_complete_draws'] += 1
            if audit['draw_index'] != index:
                issues.append({'pair': draw_label, 'field': 'draw_index'})
            rng_a, rng_b = a['sampling'].get('cuda_rng_state'), b['sampling'].get('cuda_rng_state')
            rng_equal = rng_a is not None and rng_b is not None and rng_a == rng_b
            counts['pre_sampling_cuda_rng_equal'] += rng_equal
            if not rng_equal or audit['pre_sampling_cuda_rng_equal'] is not True:
                issues.append({'pair': draw_label, 'field': 'pre_sampling_cuda_rng_equal'})
            steps_equal = a['sampling']['denoising_steps'] == b['sampling']['denoising_steps'] == saved['denoising_steps']
            counts['denoising_steps_equal'] += steps_equal
            if not steps_equal:
                issues.append({'pair': draw_label, 'field': 'denoising_steps'})
            if contact:
                control['paired_candidates'] += 1
                equal = a['actions'] == b['actions']
                control['paired_candidates_complete_actions_equal'] += equal
                if not equal:
                    issues.append({'pair': draw_label, 'field': 'contact_control_candidate_actions'})
            preprocess = audit['preprocess']
            if preprocess['fixed_cup_action_clock'] is not True or set(preprocess['channels']) != set(CHANNEL_WIDTHS):
                raise ValueError('Expected the fixed clock, language tokens and attention-mask evidence')
            all_equal = True
            for key, width in CHANNEL_WIDTHS.items():
                changed, consistent = channel_evidence(preprocess['channels'][key], width)
                channels[key]['equal_draws'] += not changed
                channels[key]['changed_draws'] += bool(changed)
                channels[key]['saved_evidence_consistent_draws'] += consistent
                all_equal = all_equal and not changed
                if changed:
                    changed_sizes[key].append(len(changed))
                if not consistent:
                    issues.append({'pair': draw_label, 'field': 'preprocess/channels/' + key})
                if key == CLOCK and not contact:
                    before, after = preprocess['channels'][key]['original'][0], preprocess['channels'][key]['corrected'][0]
                    phase_only = (changed == [[0, 3], [0, 9]] and before[3] == after[9] == 1.
                                  and before[9] == after[3] == 0.)
                    interventions['clock_phase_slots_only_draws'] += phase_only
                    if not phase_only:
                        issues.append({'pair': draw_label, 'field': 'phase_only_clock_slots'})
            if contact:
                control['saved_input_channels_equal_draws'] += all_equal
                if not all_equal:
                    issues.append({'pair': draw_label, 'field': 'contact_control_input_channels'})
            other_keys.update(preprocess['other_tensor_keys'])
            counts['other_tensors_claimed_equal_draws'] += preprocess['other_tensors_elementwise_equal'] is True
            if preprocess['other_tensors_elementwise_equal'] is not True:
                issues.append({'pair': draw_label, 'field': 'other_tensors_elementwise_equal'})
    control['all_50x7_actions_elementwise_equal'] = all(
        control[view + '_complete_actions_equal'] == control['case_seed_pairs'] for view in ('raw_first', 'selected')) and (
        control['paired_candidates_complete_actions_equal'] == control['paired_candidates'])
    return {'case_seed_pairs': len(rows), **dict(counts), 'unpaired_complete_draws': dict(unpaired),
            'all_draws_paired': not any(unpaired.values()),
            'channels': {key: {**dict(value), 'shape': [1, CHANNEL_WIDTHS[key]],
                               'changed_element_count_range': ([min(changed_sizes[key]), max(changed_sizes[key])]
                                                               if changed_sizes[key] else None)}
                         for key, value in channels.items()},
            'released_clock_changed_indices': [[0, 3], [0, 9]],
            'other_tensor_keys': sorted(other_keys),
            'other_tensor_equality_evidence': 'saved probe assertion; these tensor values were not archived',
            'contact_controls': dict(control), 'released_interventions': dict(interventions),
            'consistent_with_saved_probe_evidence': not issues, 'issues': issues}


def score_probe(original, corrected, saved, journal):
    if (saved.get('schema') != 'fr5-cup-release-phase-probe-v1' or saved['action_shape'] != [50, 7]
            or saved['prefix_points'] != 3 or saved['max_complete_candidates'] != 3
            or saved['timing_mode'] != 'native' or saved['observation_context_profile'] != PROFILE
            or saved['robot_connections'] != 0 or saved['robot_commands'] != 0):
        raise ValueError('Expected a complete offline fixed-clock phase probe')
    pairs = {pair['case_id']: pair for pair in pair_inputs(original, corrected)}
    results = saved['results']
    if len(results) != len(pairs) or {case['case_id'] for case in results} != pairs.keys():
        raise ValueError('Phase probe does not cover the paired held-out cases')
    for case in results:
        pair = pairs[case['case_id']]
        for key in ('original_case_index', 'corrected_case_index', 'recovery_role', 'phase_changed'):
            if case[key] != pair[key]:
                raise ValueError('Saved phase pairing differs from the input evidence')
        if case['original_recorded_task'] != pair['case']['request']['task'] or case['task_override'] != saved['task_override']:
            raise ValueError('Saved task condition differs from the input evidence')
        if [row['seed'] for row in case['seeds']] != saved['seeds']:
            raise ValueError('Case does not cover the declared paired seeds')
    conditions = {phase: score_condition(inputs, saved, phase, journal)
                  for phase, inputs in zip(PHASES, (original, corrected))}
    return {'checkpoint': saved['checkpoint'], 'task_override': saved['task_override'], 'seeds': saved['seeds'],
            'conditions': conditions, 'paired_evidence': audit_pairs(saved),
            'probe_float_thresholds': saved['thresholds'],
            'probe_prefix3_float_summary': saved['summary']}


def completed_runs(receipt):
    if (receipt.get('schema') != 'fr5-cup-release-phase-paired-inference-execution-v1'
            or receipt['status'] != 'completed' or receipt['robot_commands'] != 0):
        raise ValueError('Expected the completed offline inference execution receipt')
    indexed = []
    for field in ('commands', 'stages'):
        rows = receipt[field]
        by_key = {(row['model'], row['prompt']): row for row in rows}
        if len(by_key) != len(rows) or by_key.keys() != RUNS:
            raise ValueError('Execution receipt must cover exactly the four model/prompt runs')
        indexed.append(by_key)
    commands, stages = indexed
    if any(stage.get('exit_code') != 0 or 'finished_at_unix' not in stage for stage in stages.values()):
        raise ValueError('All four phase probes must have completed with exit_code 0')
    return commands, stages


def command_value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-inputs', type=Path, required=True)
    parser.add_argument('--corrected-inputs', type=Path, required=True)
    parser.add_argument('--execution-receipt', type=Path, required=True)
    parser.add_argument('--probe', type=Path, action='append', required=True)
    parser.add_argument('--output', default='-', help='New score JSON path, or - for stdout')
    args = parser.parse_args()
    receipt = read_json(args.execution_receipt)
    commands, stages = completed_runs(receipt)
    paths = {path.resolve() for path in args.probe}
    expected_paths = {Path(row['output']).resolve() for row in commands.values()}
    if len(args.probe) != 4 or paths != expected_paths:
        raise ValueError('Probe paths must be the four outputs recorded in the execution receipt')
    original, corrected = read_json(args.original_inputs), read_json(args.corrected_inputs)
    if original['source_root'] != corrected['source_root'] or any(
            case['source_root'] != original['source_root'] for case in original['cases']):
        raise ValueError('Expected the single original held-out source 7 recording')
    journal_path = Path(original['source_root']) / 'target-command-journal.json'
    journal = read_json(journal_path)
    scored = []
    for key, execution in commands.items():
        path = Path(execution['output'])
        saved = read_json(path)
        command = execution['command']
        for flag, field, input_path in (('--original-inputs', 'original_inputs', args.original_inputs),
                                        ('--corrected-inputs', 'corrected_inputs', args.corrected_inputs)):
            if (Path(command_value(command, flag)).resolve() != input_path.resolve()
                    or Path(saved[field]).resolve() != input_path.resolve()):
                raise ValueError('Execution or prediction inputs differ from the paired scoring inputs')
        if (saved['checkpoint'] != command_value(command, '--checkpoint') or saved['seeds'] != receipt['seeds']
                or saved['task_override'] != command_value(command, '--task-override')
                or command_value(command, '--output') != execution['output']):
            raise ValueError('Prediction checkpoint, seeds, task or output differ from the execution receipt')
        score = score_probe(original, corrected, saved, journal)
        score.update(model=key[0], prompt=key[1], prediction_source=str(path), execution=stages[key])
        scored.append(score)
    report = {'schema': 'fr5-cup-release-phase-paired-score-v1',
              'original_inputs': str(args.original_inputs.resolve()),
              'corrected_inputs': str(args.corrected_inputs.resolve()),
              'execution_receipt': str(args.execution_receipt.resolve()), 'execution_status': receipt['status'],
              'source_root': original['source_root'], 'journal': str(journal_path),
              'scope': {'training_source_indices': list(range(7)), 'heldout_source_indices': [7],
                        'evaluation_group': 'heldout_source_7', 'case_count': len(original['cases']),
                        'kind': 'feedback_recovered'},
              'feedback': {'current_valid_cases': sum(case['request']['cup_relative_state'][7] == 1.
                                                     for case in original['cases']),
                           'current_unknown_cases': sum(case['request']['cup_relative_state'][7] != 1.
                                                       for case in original['cases']),
                           'unknown_history_observations': sum(h is not None and h['state'][7] != 1.
                                                               for case in original['cases']
                                                               for h in case['request']['observation_history'])},
              'runs': scored, 'action_shape': [50, 7], 'execution_prefix_points': 3,
              'scored_horizons': list(HORIZONS),
              'clock': 'recorded camera monotonic plus tick/6 seconds; original executed jaw targets',
              'timing': 'clamped integer percent; one-point deadband; one tick allowed after recorded opening',
              'selector_matched_source': 'applied saved proposal traces; not_applied is unassessed',
              'scoring_device': 'cpu', 'inference_reruns': 0, 'training_runs': 0,
              'robot_connections': 0, 'robot_commands': 0, 'task_success_verified': False,
              'limitations': ['Offline scores establish no physical motion or task success.',
                              'Prefix10 is diagnostic; the saved policy execution prefix is three points.',
                              'Both prompts are scored against the original recorded jaw commands.',
                              'Late opening remains unassessed when already open or the deadline lies outside the prefix.',
                              'Other input tensors and tensor dtypes retain the saved probe assertions; '
                              'CPU checks compare archived channel values, shapes, changed indices and RNG states.']}
    if args.output == '-':
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    else:
        write_new(Path(args.output), report)


if __name__ == '__main__':
    main()
