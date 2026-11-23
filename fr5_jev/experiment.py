"""Persist model requests, responses and routing gates frozen before inference."""
import json
import time
from pathlib import Path

from .decision import call_jev, digest, make_request, validate_answer
from .policy import assess, gate_table
from .fast import compact_state

DIRECTORY = Path.home() / ".local/state/fr5-system/jev"
KEY_FILE = Path.home()/".config/openai/robot-decisions-api-key"


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def decide(state, name, *, hard_checks=None, context=None, directory=None, compact=False):
    """Exactly one actual model request. Freeze its gate table before dispatch."""
    path = Path(directory or DIRECTORY)/(name+".json")
    if path.exists():
        raise RuntimeError("Decision record already exists")
    model_state = compact_state(state) if compact else state
    request = make_request(model_state, provider="openai")
    choices = [choice["value"] for choice in request["questions"][0]["choices"]]
    policies = gate_table(choices, **(context or {}))
    record = {"provider": "openai", "state": state, "request": request,
              "request_sha256": digest(request),
              "compact_request": compact,
              "gate_table_before_inference": policies, "hard_checks": hard_checks or {},
              "physical_commands_sent_by_decision": 0, "started_at_unix": time.time()}
    save_new(path, record)  # Reserve once, before sending anything.
    try:
        response, elapsed = call_jev(request, KEY_FILE, provider="openai")
        record.update(response=response, latency_ms=elapsed)
        answer = validate_answer(request, response, provider="openai")
        gate = assess(answer, policies[answer["choice"]], hard_checks=hard_checks)
        record.update(status="decision_received", gate=gate, finished_at_unix=time.time())
    except Exception as exc:
        # Upstream bodies and credential data never enter exception logs.
        record.update(status="decision_failed", error_type=type(exc).__name__,
                      finished_at_unix=time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)
    return record, path
