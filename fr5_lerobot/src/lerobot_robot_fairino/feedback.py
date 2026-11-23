"""Validity of this FR5/DH gripper's measured feedback, not grasp success."""

import math


def gripper_feedback_valid(state):
    position = state.get("gripper_position")
    return bool(
        isinstance(position, (int, float)) and not isinstance(position, bool)
        and math.isfinite(position) and 0 <= position <= 100
        and state.get("gripper_active") == 1
        and state.get("gripper_fault") == 0
        and state.get("gripper_fault_id") == 0
        and state.get("gripper_motiondone") in (1, 2, 3)
        and state.get("gripper_speed") == 0
    )
