"""Compact model context; keep full state in the local audit record."""


def compact_state(state):
    result = {k: state[k] for k in ("schema", "source", "robot_id", "robot_ip", "captured_at_unix", "robot", "candidates")}
    camera = state.get("camera", {})
    result["camera"] = {k: camera[k] for k in ("available", "serial", "expected_device_and_fresh", "age_s") if k in camera}
    facts = state.get("facts", {})
    # Retain physical/calibration/authorization fields. Drop obsolete narrative,
    # not unknowns or failed checks, from the model request.
    result["facts"] = {k: v for k, v in facts.items()
                       if k not in ("previous_astra_planned_descent", "why_more_observation_is_not_needed_now")}
    result["recent_outcomes"] = state.get("recent_outcomes", [])[-2:]
    result["retrieved_examples"] = [{
        "case_id": case["case_id"],
        "observed_outcome": case["observed_outcome"],
        "limitations": case["limitations"],
        "grants_execution_permission": False,
    } for case in state.get("retrieved_examples", [])[:2]]
    if "llm_annotations" in state:
        result["llm_annotations"] = state["llm_annotations"]
    if "llm_review" in state:
        result["llm_review"] = state["llm_review"]
    return result
