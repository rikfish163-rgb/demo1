#!/usr/bin/env python3
"""Trace saved RGB, state and cup conditions without policy inference or I/O."""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))

from fr5_jev.cup_timing import CupTiming
from lerobot_robot_fairino.cup_action_selection import choose_proposal, close_at_pickup
import stream_image_profile as images
import stream_observation_history as history
import train_stream_pi05 as training

if __package__:
    from . import evaluate_cup_setdown_gate as gate
    from . import replay_true_five_second_pi05 as fit
    from . import score_cup_training_timing_fit as timing_fit
    from .replay_cup_first_proposal import AdditionalDrawRequired, SavedFirstDraw
else:
    import evaluate_cup_setdown_gate as gate
    import replay_true_five_second_pi05 as fit
    import score_cup_training_timing_fit as timing_fit
    from replay_cup_first_proposal import AdditionalDrawRequired, SavedFirstDraw


CASES = ('source3-cycle1-close_at-1468', 'source5-cycle1-close_at-2524')
CLOCK_FIELDS = (
    'remaining_hover_seconds', 'hover_clock_known',
    *(f'phase_{phase}' for phase in history.CUP_TIMING_PHASES),
    *(f'visual_{visual}' for visual in history.CUP_TIMING_VISION),
    'rim_from_jaw_x_over_60mm', 'rim_from_jaw_y_over_60mm', 'rim_from_jaw_z_over_60mm',
    'rim_measurement_known', 'native_contact', 'contact_feedback_known',
    'support_gap_over_12mm', 'support_gap_known', 'setdown_age_condition',
)


def input_features(request):
    """Evaluate the production image, prompt and timing functions on the CPU."""
    profile = request['observation_context_profile']
    with Image.open(io.BytesIO(base64.b64decode(request['rgb_jpeg_base64']))) as image:
        rgb = np.array(image.convert('RGB'))
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.
    state = torch.tensor([request['cup_relative_state']], dtype=torch.float32)
    prompt = history.render_task(
        request['task'], request['observation_history'], profile,
        initial_cup_uv=request['initial_cup_uv'], cup_tracking=request['cup_tracking'],
        cup_timing=request['cup_timing'])
    crop = images.apply_image_profile(
        {images.SOURCE_KEY: image_tensor, 'observation.state': state, 'task': [prompt]},
        images.CUP_CENTERED_IMAGE_PROFILE, initial_cup_uv=[request['initial_cup_uv']])
    clock = torch.tensor([history.action_clock(
        request['task'], request['cup_timing'], cup_tracking=request['cup_tracking'],
        profile=profile)], dtype=torch.float32)
    return crop[images.SOURCE_KEY], state, prompt, clock


def suffix_inputs(clock):
    """Capture tensors passed through the real condition hook, without weights."""
    noisy = torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32) / 1000.
    model = SimpleNamespace(embed_suffix=lambda actions, _time: actions)
    policy = SimpleNamespace(model=model)
    batch = {'observation.cup_action_clock': clock, 'observation.fixed_cup_action_clock': True}
    with training.action_clock_condition(policy, batch):
        traced = [model.embed_suffix(noisy + step, torch.tensor([step])) for step in (.9, .5, .1)]
    motion_unchanged = all(torch.equal(value[..., :7], (noisy + step)[..., :7])
                           for step, value in zip((.9, .5, .1), traced))
    fixed_each_step = all(torch.equal(traced[0][..., 7:], value[..., 7:]) for value in traced[1:])
    return traced[0][..., 7:], motion_unchanged, fixed_each_step


def compare_requests(before, after):
    for key in before.keys() | after.keys():
        if key not in ('cup_tracking', 'cup_timing') and before.get(key) != after.get(key):
            raise ValueError(f'The paired request changed {key}')
    old_crop, old_state, old_prompt, old_clock = input_features(before)
    new_crop, new_state, new_prompt, new_clock = input_features(after)
    old_suffix, old_motion, old_fixed = suffix_inputs(old_clock)
    new_suffix, new_motion, new_fixed = suffix_inputs(new_clock)
    changed = [i for i in range(len(CLOCK_FIELDS)) if old_clock[0, i] != new_clock[0, i]]
    return {
        'rgb_crop_shape': list(old_crop.shape),
        'rgb_crop_equal': torch.equal(old_crop, new_crop),
        'rgb_crop_different_elements': int(torch.count_nonzero(old_crop != new_crop)),
        'model_state_equal': torch.equal(old_state, new_state),
        'initial_cup_uv_equal': before['initial_cup_uv'] == after['initial_cup_uv'],
        'cup_coordinate_origin_equal': before['cup_base_xyz_mm'] == after['cup_base_xyz_mm'],
        'observation_history_equal': before['observation_history'] == after['observation_history'],
        'prompt_equal': old_prompt == new_prompt,
        'rendered_prompt_before': old_prompt,
        'rendered_prompt_after': new_prompt,
        'condition_count': len(CLOCK_FIELDS),
        'conditions_before': old_clock[0].tolist(),
        'conditions_after': new_clock[0].tolist(),
        'changed_conditions': [
            {'condition_index': i, 'field': CLOCK_FIELDS[i], 'action_input_index': i + 7,
             'before': float(old_clock[0, i]), 'after': float(new_clock[0, i])}
            for i in changed],
        'action_input_condition_elements_changed': int(torch.count_nonzero(old_suffix != new_suffix)),
        'action_input_conditions_before_first3': old_suffix[0, :3].tolist(),
        'action_input_conditions_after_first3': new_suffix[0, :3].tolist(),
        'seven_motion_channels_preserved_by_hook': old_motion and new_motion,
        'fixed_conditions_equal_across_denoising_steps': old_fixed and new_fixed,
        'model_prediction_recomputed': False,
    }


def replay_selection(request, reply, normal):
    """Use one saved complete draw; later draws have no recorded answers."""
    fit.validate_actions(reply)
    model = SavedFirstDraw(request, reply)
    try:
        selected, draws, changed = choose_proposal(model, request, normal, check_boundaries=True)
    except AdditionalDrawRequired:
        return {'status': 'additional_draws_required', 'selected_actions_unchanged': None}
    if draws != 1 or changed or selected['actions'] != reply['actions']:
        raise ValueError('Selection changed the saved complete action block')
    return {'status': 'first_draw_selected', 'selected_actions_unchanged': True}


def timer_pairs(case, tracker_rows):
    """Replay only prior paired captures through the existing stage observer."""
    start, stop = case['tracker_start_frame_index'], case['source_frame_index']
    captures = fit.read_jsonl(Path(case['source_root']) / 'continuous/capture.jsonl')
    timers = {name: CupTiming(case['initial_cup'], setdown=True) for name in ('before', 'after')}
    result = None
    for capture in captures:
        frame = capture['frame_index']
        if frame < start:
            continue
        if frame > stop:
            break
        row = tracker_rows[case['source_index'], frame]
        if row['camera_monotonic'] != capture['camera']['source_timestamp_monotonic']:
            raise ValueError('Saved tracker evidence uses a different camera timestamp')
        result = {name: timers[name].update(capture['camera'], capture['robot'], row[name])
                  for name in timers}
        if frame == stop:
            saved = case['observers']['setdown']
            if row['before'] != saved['tracking'] or result['before'] != saved['timing']:
                raise ValueError('The original observation cannot be reproduced from paired captures')
            return row, result
    raise ValueError(f"Missing original capture for {case['case_id']}")


def audit(inputs, predictions, tracker_replay):
    cases = timing_fit.cases_index(inputs)
    predictions_by_case = timing_fit.prediction_index(cases, predictions)
    if predictions['observation_context_profile'] != history.CUP_SETDOWN_FIXED_PROFILE:
        raise ValueError('This comparison requires the current fixed-condition model profile')
    tracker_rows = {(row['source_index'], row['frame_index']): row for row in tracker_replay}
    results = []
    for case_id in CASES:
        case = cases[case_id]
        row, timers = timer_pairs(case, tracker_rows)
        before = fit.model_request(case, 'setdown', predictions['observation_context_profile'])
        if before['cup_timing'] != timers['before']['state']:
            raise ValueError('The saved request does not use the reproduced stage observer')
        after = copy.deepcopy(before)
        after['cup_tracking'] = {k: row['after'][k] for k in ('rim_minus_jaw_mm', 'native_contact')}
        after['cup_timing'] = timers['after']['state']
        selections = []
        for seed in predictions['seeds']:
            saved = predictions_by_case[case_id, seed]['reply']
            if saved.get('cup_proposal_selection'):
                raise ValueError('Expected an unselected first draw')
            selections.append({
                'seed': seed, 'saved_first3_gripper_percent': [gate.percent(a) for a in saved['actions'][:3]],
                'original_observation': replay_selection(before, saved, gate.normal_of(case)),
                'updated_observation_same_saved_actions': replay_selection(after, saved, gate.normal_of(case)),
                'model_prediction_recomputed': False,
            })
        results.append({
            'case_id': case_id, 'source_index': case['source_index'],
            'source_frame_index': case['source_frame_index'], 'source_time': before['source_time'],
            'tracking_before': row['before'], 'tracking_after': row['after'],
            'timing_before': timers['before'], 'timing_after': timers['after'],
            'close_branch_ready_before': close_at_pickup(before),
            'close_branch_ready_after': close_at_pickup(after),
            'input_comparison': compare_requests(before, after), 'saved_first_draw_selection': selections,
        })
    return {
        'schema': 'fr5-cup-input-dependence-audit-v1', 'device': 'cpu',
        'model_inference_calls': 0, 'model_weights_loaded': False, 'robot_commands': 0,
        'task_success_verified': False, 'new_demonstrations': 0, 'failed_trials_training': 0,
        'checkpoint_of_saved_predictions': predictions['checkpoint'], 'cases': results,
        'production_input_sources': {
            'image_profile': images.__file__, 'observation_history': history.__file__,
            'action_condition_hook': training.__file__, 'timing_observer': CupTiming.update.__code__.co_filename,
            'candidate_selector': choose_proposal.__code__.co_filename,
        },
        'limitations': [
            'The CPU trace captures production inputs before the saved image/text processor and the '
            'action projection. It does not measure the network response to changed conditions.',
            'The updated-observation selector comparison holds original predicted actions fixed. '
            'Requests for later draws remain unassessed.',
            'Observed cup rims and gripper contact do not establish physical cup support or task success.',
        ],
    }


def summary(report):
    result = {key: value for key, value in report.items() if key != 'cases'}
    result['cases'] = []
    for row in report['cases']:
        compare = row['input_comparison']
        result['cases'].append({
            'case_id': row['case_id'], 'source_frame_index': row['source_frame_index'],
            'rgb_crop_equal': compare['rgb_crop_equal'], 'model_state_equal': compare['model_state_equal'],
            'cup_coordinate_origin_equal': compare['cup_coordinate_origin_equal'],
            'prompt_equal': compare['prompt_equal'], 'changed_conditions': compare['changed_conditions'],
            'action_input_condition_elements_changed': compare['action_input_condition_elements_changed'],
            'seven_motion_channels_preserved_by_hook': compare['seven_motion_channels_preserved_by_hook'],
            'fixed_conditions_equal_across_denoising_steps': compare['fixed_conditions_equal_across_denoising_steps'],
            'close_branch_ready_before': row['close_branch_ready_before'],
            'close_branch_ready_after': row['close_branch_ready_after'],
            'saved_first_draw_selection': row['saved_first_draw_selection'],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--tracker-replay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(fit.read_json(args.inputs), fit.read_json(args.predictions), fit.read_jsonl(args.tracker_replay))
    report.update(inputs=str(args.inputs), predictions=str(args.predictions), tracker_replay=str(args.tracker_replay))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    compact = summary(report)
    compact['details'] = str(args.output)
    summary_path = args.output.with_name('summary.json')
    with summary_path.open('x', encoding='utf-8') as stream:
        json.dump(compact, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'details': str(args.output), 'cases': compact['cases']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
