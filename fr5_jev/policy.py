"""Provisional risk-dependent gates, frozen before each Jev request.

Thresholds are engineering trial settings, not measured probabilities of safety.
No rule here can grant authorization or generate robot commands.
"""
from .decision import MOTION_SKILLS, number

VERSION = "risk-evidence-v4-bounded-empty-sensing"
CONTACT = frozenset({"close_gripper", "lift_verify", "return_place", "release_gripper"})


def thresholds(skill, *, evidence_complete=True, matched_successes=0, recent_failures=0):
    if type(evidence_complete) is not bool:
        raise ValueError("Evidence completeness must be an explicit boolean")
    if type(matched_successes) is not int or type(recent_failures) is not int:
        raise ValueError("Outcome counts must be explicit integers")
    if min(matched_successes, recent_failures) < 0:
        raise ValueError("Outcome counts cannot be negative")
    if skill in CONTACT:
        risk, confidence, probability, margin = "contact", .95, .97, .8
    elif skill == "prepare_empty_gripper":
        # Explicit user-requested experiment, 2026-10-26. Stationary empty
        # activation/open only; all physical checks and replay guards remain.
        risk, confidence, probability, margin = "noncontact_motion", .70, .80, .5
    elif skill in ("orient_empty_tool", "transfer_empty_tool", "probe_empty_gripper"):
        # User-requested experiment (2026-10-27): Jev routes an independently
        # reviewed, bounded empty-tool orientation, free-space transfer, or
        # small empty-jaw sensing probe with a low decision gate.
        # This is NOT a safety probability and does not relax any physical
        # precondition, path check, contact gate, or executor interlock.
        risk, confidence, probability, margin = "noncontact_motion", .10, .35, .05
    elif skill in MOTION_SKILLS:
        risk, confidence, probability, margin = "noncontact_motion", .85, .90, .5
    else:
        risk, confidence, probability, margin = "readonly", .50, .65, .15
    # Outcomes can only relax READONLY settings. Motion thresholds never relax
    # automatically; a different motion policy needs separate replay evaluation.
    if risk == "readonly" and matched_successes >= 5 and recent_failures == 0:
        confidence, probability = .45, .60
    if recent_failures:
        confidence = min(.99, confidence + .05)
        probability = min(.995, probability + .05)
    if not evidence_complete:
        confidence = min(.99, confidence + .1)
        probability = min(.995, probability + .1)
    return {"policy_version": VERSION, "risk": risk,
            "min_confidence": confidence, "min_probability": probability,
            "min_margin": margin, "evidence_complete": evidence_complete,
            "matched_successes": matched_successes, "recent_failures": recent_failures,
            "empirically_calibrated": False}


def assess(answer, policy, *, hard_checks=None):
    choice = answer["choice"]
    probabilities = answer["probabilities"]
    probability = probabilities[choice]
    runner_up = max((v for k, v in probabilities.items() if k != choice), default=0.)
    confidence = answer["confidence"]
    result = {"choice": choice, "probability": probability, "confidence": confidence,
              "margin": probability-runner_up, "policy": policy,
              "physical_authorization_granted_by_policy": False}
    if not all(number(v) and 0 <= v <= 1 for v in (probability, confidence, runner_up)):
        return {**result, "route": "needs_llm", "reason": "Malformed answer values"}
    if choice in MOTION_SKILLS:
        minimum = thresholds(choice)
        if (policy.get("policy_version") != VERSION or policy.get("risk") != minimum["risk"]
                or any(policy.get(k, -1) < minimum[k]
                       for k in ("min_confidence", "min_probability", "min_margin"))):
            return {**result, "route": "blocked", "reason": "Motion gate was relaxed or assigned a readonly policy"}
        required = ("specific_authorization", "preflight_passed", "scene_reviewed",
                    "bounded_executor_verified")
        missing = [k for k in required if (hard_checks or {}).get(k) is not True]
        # An explicitly user-provided default is an assumption, not a measured
        # onsite confirmation. Restrict it to bounded empty-tool motion and
        # preserve the distinction in the audit record.
        workspace = hard_checks or {}
        if workspace.get("onsite_confirmation") is True:
            result["workspace_basis"] = "onsite_confirmation"
        elif (choice in ("orient_empty_tool", "transfer_empty_tool", "probe_empty_gripper")
              and workspace.get("user_default_normal") is True):
            result["workspace_basis"] = "user_default_normal_not_independently_verified"
        else:
            missing.append("onsite_confirmation")
        if missing or not policy["evidence_complete"]:
            return {**result, "route": "blocked", "reason": "Missing physical preconditions",
                    "missing_checks": missing}
    if choice == "request_astra":
        return {**result, "route": "needs_llm", "reason": "Jev requested LLM review"}
    if (confidence < policy["min_confidence"] or probability < policy["min_probability"]
            or probability-runner_up < policy["min_margin"]):
        return {**result, "route": "needs_llm", "reason": "Decision below the predeclared risk gate"}
    return {**result, "route": "selected", "reason": "Passed the predeclared risk gate"}


def gate_table(choices, **context):
    return {name: thresholds(name, **context) for name in choices}
