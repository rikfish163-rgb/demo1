"""Position-trajectory interface for FR5 pi0.5 training and deployment."""
from __future__ import annotations

from copy import deepcopy
import math

NAME = 'fr5-joint-position-stream-v1'
CUP_RELATIVE_WIRE = 'cup-relative-tcp-v1'
ABSOLUTE_JOINT_WIRE = 'absolute-joint-position-v1'
FPS = 6
CHUNK = 50
STATE_NAMES = [f'joint_{i}_rad' for i in range(1, 7)] + ['gripper_fraction', 'gripper_measurement_valid']
ACTION_NAMES = [f'joint_target_{i}_rad' for i in range(1, 7)] + ['gripper_target_fraction']
STATE_DIM, ACTION_DIM = 8, 7
CONTRACT = {
    'name': NAME, 'action_timebase': 'fixed_rate', 'fps': FPS,
    'chunk_size': CHUNK, 'state_dim': STATE_DIM, 'action_dim': ACTION_DIM,
    'state_names': STATE_NAMES, 'action_names': ACTION_NAMES,
    'action_units': ['rad']*6 + ['fraction'],
    'gripper_convention': '0=closed, 1=open',
    'joint_representation': 'absolute_position',
    'action_timestamp': 'observation_grid_time+(action_index+1)/fps',
    'task_conditioning': 'one_full_instruction_per_episode',
    'phase_input': False, 'runtime_llm': False,
    'termination': 'operator_request_or_explicit_run_budget',
}


def contract():
    return deepcopy(CONTRACT)


def finite_vector(values, dimension, label):
    if (not isinstance(values, (list, tuple)) or len(values) != dimension
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in values)):
        raise ValueError(f'Expected {dimension} finite numbers for {label}')
    return [float(x) for x in values]


def decode_action(values):
    raw = finite_vector(values, ACTION_DIM, 'stream action')
    grip = min(1., max(0., raw[6]))
    return {'raw_action': raw, 'joint_degrees': [math.degrees(x) for x in raw[:6]],
            'gripper_integer_percent': int(round(grip*100.)),
            'gripper_projected_to_device_range': grip != raw[6]}
