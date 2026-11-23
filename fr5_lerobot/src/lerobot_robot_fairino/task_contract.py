"""Shared physical units and history for whole-task pi0.5 command events.

This is an FR5 adaptation choice, not a pi0.5 architectural requirement.
No phase labels or language-model decisions enter inference state.
"""
from __future__ import annotations

from copy import deepcopy
import math

NAME = 'fr5-command-event-history-v3'
CHUNK = 8
STATE_NAMES = ([f'joint_{i}_rad' for i in range(1, 7)]
               + ['gripper_fraction', 'gripper_measurement_valid']
               + [f'previous_joint_target_{i}_rad' for i in range(1, 7)]
               + ['previous_gripper_target_fraction', 'completed_event_count',
                  'previous_dwell_seconds', 'elapsed_task_seconds'])
ACTION_NAMES = ([f'joint_target_{i}_rad' for i in range(1, 7)]
                + ['gripper_target_fraction', 'dwell_seconds', 'task_complete', 'event_kind_gripper'])
STATE_DIM, ACTION_DIM = len(STATE_NAMES), len(ACTION_NAMES)
# This describes the already trained v3 bundle. Historical fields remain intact
# for checkpoint/data integrity, not as runtime rules. In particular the legacy
# completion channel and its old threshold have no authority to stop a rollout.
CONTRACT = {
    'name': NAME, 'action_timebase': 'command_events',
    'task_conditioning': 'one_task_instruction_per_episode',
    'execution': 'one_event_then_reobserve', 'chunk_size': CHUNK,
    'state_dim': STATE_DIM, 'action_dim': ACTION_DIM,
    'state_names': STATE_NAMES, 'action_names': ACTION_NAMES,
    'hold_timing': 'model_dwell_after_endpoint_completion',
    'termination': 'model_task_complete_after_event_and_dwell',
    'event_kind': '0=arm endpoint; 1=gripper target; one native command per predicted event',
    'done_threshold': 0.5, 'phase_input': False, 'runtime_llm': False,
    'action_units': ['rad']*6 + ['fraction', 'seconds', 'probability', 'probability'],
}


def contract():
    return deepcopy(CONTRACT)


def finite(values, size, name):
    if len(values) != size or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError(f'Expected {size} finite numbers for {name}')
    return [float(v) for v in values]


def state_vector(measured, previous_target, completed_events, previous_dwell_s, elapsed_s):
    measured = finite(measured, 8, 'measured state')
    previous_target = finite(previous_target, 7, 'previous physical target')
    if (measured[7] not in (0., 1.) or not 0 <= measured[6] <= 1
            or (measured[7] == 0 and measured[6] != 0) or not 0 <= previous_target[6] <= 1
            or type(completed_events) is not int or completed_events < 0
            or type(previous_dwell_s) not in (int, float) or not math.isfinite(previous_dwell_s) or previous_dwell_s < 0
            or type(elapsed_s) not in (int, float) or not math.isfinite(elapsed_s) or elapsed_s < 0):
        raise ValueError('Invalid measured state or actual execution history')
    return measured + previous_target + [float(completed_events), float(previous_dwell_s), float(elapsed_s)]


def decode_event(values):
    """Physical projection is logged; joint endpoints are never interpolated."""
    raw = finite(values, ACTION_DIM, 'predicted event')
    grip = min(1., max(0., raw[6]))
    dwell = max(0., raw[7])
    done = min(1., max(0., raw[8]))
    kind_score = min(1., max(0., raw[9]))
    return {'raw_action': raw, 'joint_degrees': [math.degrees(q) for q in raw[:6]],
            'gripper_fraction': grip, 'gripper_integer_percent': int(round(grip*100.)),
            'dwell_seconds': dwell, 'completion_score': done,
            'event_kind': 'gripper' if kind_score >= .5 else 'arm',
            'event_kind_score': kind_score,
            'projected_fields': {'gripper': grip != raw[6], 'dwell': dwell != raw[7], 'complete': done != raw[8]}}
