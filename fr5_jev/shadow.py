"""Usage: python -m fr5_jev.shadow --live [--call-jev]. No physical execution option."""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

from .decision import CONTRACT_VERSION, PROVIDERS, evaluate_shadow
from .calibration import load_readonly_calibration

ROOT = Path(__file__).resolve().parents[1]


def gripper_position_feedback_valid(robot):
    """Do not present this firmware's busy all-zero tuple as actual closure."""
    return (bool(robot['gripper_active'] & 1) and robot['gripper_fault']==0
            and robot['gripper_fault_id']==0 and robot['gripper_motiondone'] in (1,2,3)
            and robot['gripper_speed']==0)


def collect_live_state() -> dict:
    # Imports the verified state-stream reader only. Do not import an executor or use its rpc().
    import numpy as np
    from fr5_runtime.robot import read_state

    robot = read_state()
    faults = ["main_code", "sub_code", "EmergencyStop", "collisionState",
              "safety_stop0_state", "safety_stop1_state", "gripper_fault", "gripper_fault_id"]
    camera = {"available": False}
    try:
        raw = urllib.request.urlopen("http://127.0.0.1:8765/snapshot_fast.npz", timeout=3).read()
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
        camera = {
            "available": True,
            "serial": meta["camera"]["serial"],
            "capture_time_unix": meta["capture"]["time_unix"],
            "age_s": time.time() - meta["capture"]["time_unix"],
            "source_image_size": meta.get("source_image_size"),
        }
        camera["expected_device_and_fresh"] = (
            camera["serial"] == "CP1E54200015" and -0.5 <= camera["age_s"] < 2
        )
    except Exception as exc:
        camera["error_type"] = type(exc).__name__
    calibration = load_readonly_calibration(camera_serial=camera.get("serial"))
    return {
        "schema": CONTRACT_VERSION,
        "source": "live_readonly",
        "robot_id": "J1",
        "robot_ip": "192.168.58.2",
        "captured_at_unix": robot["timestamp"],
        "robot": {
            "enabled": robot["rbtEnableState"] == 1,
            "stationary": robot["motion_done"] == 1 and robot["mc_queue_len"] == 0
                          and robot["robot_state"] == 1 and robot["program_state"] == 1,
            "fault_free": all(robot[k] == 0 for k in faults),
            "tcp_mm_deg": robot["tcp"],
            "gripper_position_percent": robot["gripper_position"],
            "gripper_active": bool(robot["gripper_active"] & 1),
            "gripper_position_feedback_valid": gripper_position_feedback_valid(robot),
            "gripper_object_detected_signal": robot["gripper_motiondone"] == 2,
            "gripper_motiondone": robot["gripper_motiondone"],
            "gripper_speed_percent": robot["gripper_speed"],
            "tool_number": robot["tool"],
            "user_frame_number": robot["user"],
        },
        "camera": camera,
        "calibration": calibration,
        "facts": {
            "task": "Grasp the upright gray cup with J1 using reusable skills selected by the phase decision model",
            "grasp_completed": False,
            "lift_completed": False,
            "random_cup_localizer_validated": False,
            "grasp_tcp_validated": False,
            "cup_exterior_geometry_verified": False,
            "finger_swept_volume_outside_cup_verified": False,
            "tool_housing_clearance_verified": False,
            "previous_extrinsic_preserved": True,
            "extrinsic_status_in_source_file": calibration["source_status"],
            "handeye_geometry_validated_in_sampled_region": True,
            "handeye_heldout_rms_mm": calibration["heldout_rms_mm"],
            "handeye_heldout_max_point_mm": calibration["heldout_max_point_mm"],
            # Held-out checkerboard accuracy does not establish the complete grasp geometry.
            "extrinsic_independently_validated": False,
            "depth_accuracy_validated": False,
            "pregrasp_scope": {"status": "not_recorded_as_complete_for_current_episode"},
        },
        "recent_outcomes": [],
        "retrieved_examples": [],
        "candidates": {
            "align_xy": {"validated": False, "preconditions_met": False,
                         "evidence": "Only a fixed cup placement and bounded paths were inspected; random-placement primitive not validated"},
            "descend_pregrasp": {"validated": False, "preconditions_met": False,
                                "evidence": "A supervised descent triggered collision after fingers entered the cup. Exterior contact geometry, grasp TCP and complete tool/finger swept-volume clearance remain unverified; this action is blocked"},
            "close_gripper": {"validated": False, "preconditions_met": False,
                              "evidence": "Empty full-open operation verified; cup contact closure not tested"},
            "lift_verify": {"validated": False, "preconditions_met": False,
                            "evidence": "No cup grasp or lift success evidence available"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="Read J1 feedback and camera metadata only")
    source.add_argument("--state", type=Path, help="Read an explicitly sourced state JSON")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai", help="Decision provider; defaults to OpenAI gpt-6-luna")
    parser.add_argument("--call-jev", action="store_true", help="Send structured state to the selected provider for a shadow decision")
    parser.add_argument("--key-file", type=Path, help="Optional file containing only the selected provider's API key; never logged")
    parser.add_argument("--output", type=Path, required=True, help="New evidence file; overwriting is refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve the existing evidence and choose a new path")
    state = collect_live_state() if args.live else json.loads(args.state.read_text())
    key_file = args.key_file
    if key_file is None and args.provider == "openai":
        key_file = Path.home() / ".config/openai/robot-decisions-api-key"
    elif key_file is None and args.provider == "emrank":
        default_key = Path.home() / ".config/emrank/api-key"
        if default_key.is_file():
            key_file = default_key
    result = evaluate_shadow(state, invoke=args.call_jev, key_file=key_file, provider=args.provider)
    record = {"state": state, "result": result, "recorded_at_unix": time.time()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(record, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "request"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
