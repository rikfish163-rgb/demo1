"""Typed model decision boundary; no robot commands or automatic Astra calls."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROVIDERS = {
    "typesafe": {
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "model": "jev-1.13.0",
        "key_env": "TYPESAFE_API_KEY",
    },
    "emrank": {
        "endpoint": "https://emrank.tech/v1/decisions",
        "model": "typesafe/jev-1.13.0[slow]",
        "key_env": "EMRANK_API_KEY",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1/decisions",
        "model": "gpt-6-luna",
        "key_env": "OPENAI_API_KEY",
    },
}
ENDPOINT = PROVIDERS["typesafe"]["endpoint"]
MODEL = PROVIDERS["typesafe"]["model"]
CONTRACT_VERSION = "fr5-cup-shadow-v1"
SKILLS = {
    "observe_scene": "Refresh cup/tool localization and evidence before any physical action.",
    "prepare_empty_gripper": "Activate a verified empty gripper and open fully at a stationary reviewed pose, without moving the arm; then verify actual feedback.",
    "probe_empty_gripper": "Select one specifically reviewed, small empty-jaw movement to identify visible finger edges, with the arm stationary and the cup outside the swept volume; stop on any possible contact.",
    "orient_empty_tool": "Choose an already reviewed, bounded empty-tool orientation segment in free space, then stop and observe; this does not authorize cup approach, contact, or grasp.",
    "transfer_empty_tool": "Choose a specifically reviewed, bounded empty-tool transfer through free space, then stop and observe; this does not authorize cup contact or grasp.",
    "align_xy": "Choose an already verified horizontal alignment primitive for the localized cup.",
    "descend_pregrasp": "Choose an already verified descent primitive to a measured pregrasp pose.",
    "close_gripper": "Choose a verified bounded closing primitive after grasp alignment is confirmed.",
    "lift_verify": "Choose a verified short lift primitive and then verify that the cup follows.",
    "return_place": "Choose a verified bounded descent that returns the held cup to its original support pose.",
    "release_gripper": "Open the verified gripper only after the cup is visibly supported at the original support pose.",
    "retreat_tool": "Choose a reviewed bounded withdrawal after the cup is visually confirmed released and supported.",
    "verify_success": "Check independent visual and gripper evidence after a completed lift.",
    "request_astra": "Escalate an unsupported, novel, ambiguous or repeatedly failing situation to Astra.",
    "hold": "Wait with no new action while a prerequisite or observation is unavailable.",
}
MOTION_SKILLS = frozenset({
    "prepare_empty_gripper", "probe_empty_gripper", "orient_empty_tool",
    "transfer_empty_tool", "align_xy", "descend_pregrasp", "close_gripper",
    "lift_verify", "return_place", "release_gripper", "retreat_tool",
})


class DecisionError(ValueError):
    pass


def provider_config(provider: str) -> dict:
    if provider not in PROVIDERS:
        raise DecisionError("Unsupported decision provider")
    return PROVIDERS[provider]


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_state(state: dict) -> None:
    if state.get("schema") != CONTRACT_VERSION:
        raise DecisionError("Unsupported state schema")
    if state.get("robot_id") != "J1" or state.get("robot_ip") != "192.168.58.2":
        raise DecisionError("This contract is only for J1 at 192.168.58.2")
    if state.get("source") not in ("live_readonly", "synthetic_test"):
        raise DecisionError("State must declare live or synthetic provenance")
    if not number(state.get("captured_at_unix")):
        raise DecisionError("State needs a finite capture time")
    robot = state.get("robot", {})
    if type(robot.get("stationary")) is not bool or type(robot.get("fault_free")) is not bool:
        raise DecisionError("Robot state is missing; unknown is not fault-free")
    candidates = state.get("candidates")
    if not isinstance(candidates, dict):
        raise DecisionError("Missing candidate skill records")
    for name, record in candidates.items():
        if name not in MOTION_SKILLS or not isinstance(record, dict):
            raise DecisionError("Unknown motion candidate")
        for flag in ("validated", "preconditions_met"):
            if type(record.get(flag)) is not bool:
                raise DecisionError("Candidate validation must be explicit")
        if not isinstance(record.get("evidence"), str) or not record["evidence"]:
            raise DecisionError("Candidate needs provenance for its eligibility")


def eligible_skills(state: dict) -> dict[str, str]:
    validate_state(state)
    result = {k: SKILLS[k] for k in ("observe_scene", "request_astra", "hold")}
    for name, candidate in state["candidates"].items():
        if candidate["validated"] and candidate["preconditions_met"]:
            result[name] = SKILLS[name]
    if state.get("facts", {}).get("lift_completed") is True:
        result["verify_success"] = SKILLS["verify_success"]
    return result


def make_request(state: dict, *, provider: str = "typesafe") -> dict:
    choices = eligible_skills(state)
    model = provider_config(provider)["model"]
    instructions = (
        "Choose the next task-level skill for grasping the upright cup. "
        "Use only the recorded facts, recent outcomes and retrieved examples. "
        "Missing evidence is unknown. Choose observe_scene when sensing can resolve it; "
        "request_astra when the skill library or known strategy cannot resolve it. "
        "Do not calculate coordinates, issue joint commands, or assume task success. "
        "The listed motion candidates have been filtered by deterministic code."
    )
    if provider == "openai":
        return {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_json", "data": state}]}],
            "questions": [{
                "type": "choice",
                "name": "next_skill",
                "instructions": instructions,
                "choices": [{"value": name, "description": description}
                            for name, description in choices.items()],
            }],
        }
    return {
        "model": model,
        "state": state,
        "questions": {
            "next_skill": {
                "type": "choice",
                "instructions": instructions,
                "criteria": choices,
            },
        },
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DecisionError("Refusing to forward model credentials to a redirect")


def api_key(key_file: Path | None = None, *, provider: str = "typesafe") -> str:
    config = provider_config(provider)
    service = "OpenAI" if provider == "openai" else "Jev"
    # A provider's credential is never implicitly reused on a different host.
    key = os.environ.get(config["key_env"], "").strip()
    if not key and key_file is not None:
        try:
            key = key_file.read_text().strip()
        except OSError:
            raise DecisionError(f"Cannot read the configured {service} credential file") from None
    if not key:
        raise DecisionError(f"{config['key_env']} is missing; no {service} call was made")
    if "\n" in key or "\r" in key:
        raise DecisionError("Invalid credential format")
    return key


def call_jev(request: dict, key_file: Path | None = None, *, provider: str = "typesafe") -> tuple[dict, float]:
    # One request, no retry. Inference is observational and never triggers a robot command.
    config = provider_config(provider)
    service = "OpenAI" if provider == "openai" else "Jev"
    if request.get("model") != config["model"]:
        raise DecisionError("Request model differs from the configured provider model")
    key = api_key(key_file, provider=provider)
    if provider == "emrank":
        return call_emrank_curl(request, key)
    req = urllib.request.Request(
        config["endpoint"],
        data=json.dumps(request, allow_nan=False).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=10) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise DecisionError(f"{service} response exceeded size limit")
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise DecisionError(f"{service} response is not an object")
        return decoded, (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as exc:
        # Never include auth headers or an upstream body in an error log.
        raise DecisionError(f"{service} HTTP {exc.code}; no action dispatched") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise DecisionError(f"{service} inference failed ({type(exc).__name__}); no action dispatched") from None


def call_emrank_curl(request: dict, key: str) -> tuple[dict, float]:
    """Emrank accepts curl here, while the urllib request received HTTP 403.

    Credentials go through stdin configuration, never a process argument or a log.
    Disable user curlrc, redirects and retries; retain normal TLS certificate validation.
    """
    transport = os.environ.get('FR5_JEV_TRANSPORT', 'local')
    if transport == 'ssh-4090':
        return call_emrank_ssh(request, key)
    if transport != 'local':
        raise DecisionError('Unknown Jev transport')
    executable = shutil.which("curl")
    if executable is None:
        raise DecisionError("Emrank transport requires curl")
    if "\n" in key or "\r" in key:
        raise DecisionError("Invalid credential format")
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="fr5-jev-http-") as directory:
            payload = Path(directory) / "request.json"
            body = Path(directory) / "response.json"
            payload.write_text(json.dumps(request, allow_nan=False))
            config = (
                'url = "https://emrank.tech/v1/decisions"\n'
                'request = "POST"\n'
                'header = ' + json.dumps("Authorization: Bearer " + key) + '\n'
                'header = "Content-Type: application/json"\n'
                'data-binary = ' + json.dumps("@" + str(payload)) + '\n'
            )
            run = subprocess.run(
                [executable, "--disable", "--config", "-", "--silent", "--show-error",
                 "--proto", "=https", "--connect-timeout", "5", "--max-time", "10",
                 "--max-filesize", "1000000", "--output", str(body), "--write-out", "%{http_code}"],
                input=config, text=True, capture_output=True, timeout=12,
            )
            if run.returncode:
                raise DecisionError(f"Emrank curl transport failed (code {run.returncode}); no action dispatched")
            status = run.stdout.strip()
            if status != "200":
                if not re.fullmatch(r"\d{3}", status):
                    raise DecisionError("Invalid HTTP status from Emrank transport")
                raise DecisionError(f"Jev HTTP {status}; no action dispatched")
            if body.stat().st_size > 1_000_000:
                raise DecisionError("Jev response exceeded size limit")
            decoded = json.loads(body.read_text())
            if not isinstance(decoded, dict):
                raise DecisionError("Jev response is not an object")
            return decoded, (time.monotonic() - started) * 1000
    except DecisionError:
        raise
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise DecisionError(f"Emrank inference failed ({type(exc).__name__}); no action dispatched") from None


def call_emrank_ssh(request: dict, key: str) -> tuple[dict, float]:
    """Use the user's existing server connection when local TLS is unavailable.

    Single request, no files or keys in argv/logs, TLS validation retained.
    The exact same authorized Jev endpoint receives the structured robot state.
    """
    if '\n' in key or '\r' in key:
        raise DecisionError('Invalid credential format')
    ssh=shutil.which('ssh')
    if ssh is None:
        raise DecisionError('Remote Jev transport requires SSH')
    marker='\n__FR5_HTTP_STATUS__:'
    config=('url = "https://emrank.tech/v1/decisions"\nrequest = "POST"\n'
            'header = '+json.dumps('Authorization: Bearer '+key)+'\n'
            'header = "Content-Type: application/json"\n'
            'data-binary = '+json.dumps(json.dumps(request,allow_nan=False))+'\n')
    remote=("curl --disable --config - --silent --show-error --proto =https "
            "--connect-timeout 5 --max-time 10 --max-filesize 1000000 "
            "--write-out '\\n__FR5_HTTP_STATUS__:%{http_code}'")
    started=time.monotonic()
    try:
        result=subprocess.run([ssh,'-o','BatchMode=yes','-o','ConnectTimeout=5','4090',remote],
                              input=config,text=True,capture_output=True,timeout=18)
        if result.returncode:
            raise DecisionError(f'Jev SSH transport failed (code {result.returncode}); no action dispatched')
        body,separator,status=result.stdout.rpartition(marker)
        if not separator or not re.fullmatch(r'\d{3}',status.strip()):
            raise DecisionError('Invalid remote Jev HTTP status')
        if status.strip()!='200':
            raise DecisionError(f'Jev HTTP {status.strip()}; no action dispatched')
        if len(body.encode())>1_000_000:
            raise DecisionError('Jev response exceeded size limit')
        decoded=json.loads(body)
        if not isinstance(decoded,dict):
            raise DecisionError('Jev response is not an object')
        return decoded,(time.monotonic()-started)*1000
    except DecisionError:
        raise
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        raise DecisionError(f'Jev SSH inference failed ({type(exc).__name__}); no action dispatched') from None


def validate_answer(request: dict, response: dict, *, provider: str = "typesafe") -> dict:
    """Validate the wire answer and return the internal choice representation."""
    config = provider_config(provider)
    service = "OpenAI" if provider == "openai" else "Jev"
    if not isinstance(response, dict):
        raise DecisionError(f"{service} response is not an object")
    returned_model = response.get("model")
    if provider == "openai":
        if request.get("model") != config["model"] or returned_model != config["model"]:
            raise DecisionError("OpenAI response or request model differs from the configured model")
    elif provider == "typesafe":
        if returned_model != request["model"]:
            raise DecisionError("Jev response model differs from the pinned model")
    elif returned_model is not None:
        # The user's working Emrank example omits model metadata. Do not invent it.
        # An alias may legitimately resolve to a Jev version, but not to another family.
        if (not isinstance(returned_model, str)
                or not re.fullmatch(r"(?:typesafe/)?jev(?:-(?:latest|preview|\d+\.\d+(?:\.\d+)?))?(?:\[slow\])?", returned_model)):
            raise DecisionError("Proxy response reports a non-Jev model")
    answers = response.get("answers")
    if provider == "openai":
        if response.get("refusal"):
            raise DecisionError("OpenAI refused the decision; no action dispatched")
        if not isinstance(answers, list) or len(answers) != 1:
            raise DecisionError("Missing unique next_skill answer")
        answer = answers[0]
        if isinstance(answer, dict) and (answer.get("type") == "refusal" or answer.get("refusal")):
            raise DecisionError("OpenAI refused the decision; no action dispatched")
        if (not isinstance(answer, dict) or answer.get("type") != "choice"
                or answer.get("name") != "next_skill"):
            raise DecisionError("Missing next_skill Choice answer")
        questions = request.get("questions")
        if (not isinstance(questions, list) or len(questions) != 1
                or not isinstance(questions[0], dict) or questions[0].get("type") != "choice"
                or questions[0].get("name") != "next_skill"):
            raise DecisionError("Missing next_skill Choice question")
        entries = questions[0].get("choices")
        if not isinstance(entries, list) or not entries:
            raise DecisionError("Missing OpenAI choice labels")
        choices = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str) or not entry["value"]:
                raise DecisionError("Invalid OpenAI choice label")
            choices.append(entry["value"])
        if len(set(choices)) != len(choices):
            raise DecisionError("Duplicate OpenAI choice labels")
        entries = answer.get("probabilities")
        if not isinstance(entries, list):
            raise DecisionError("Missing OpenAI probability list")
        probabilities = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
                raise DecisionError("Invalid OpenAI probability label")
            label = entry["value"]
            if label in probabilities:
                raise DecisionError("Duplicate OpenAI probability labels")
            probabilities[label] = entry.get("probability")
        answer = {"type": answer["type"], "choice": answer.get("choice"),
                  "confidence": answer.get("confidence"), "probabilities": probabilities}
    else:
        if not isinstance(answers, dict):
            raise DecisionError("Missing answer map")
        answer = answers.get("next_skill", {})
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise DecisionError("Missing Choice answer")
        choices = request["questions"]["next_skill"]["criteria"]
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(choices):
        raise DecisionError(f"{service} probability labels differ from the eligible skills")
    if any(not number(v) or not 0 <= v <= 1 for v in probabilities.values()):
        raise DecisionError(f"Invalid {service} probabilities")
    if abs(sum(probabilities.values()) - 1) > 0.001:
        raise DecisionError(f"{service} probabilities do not sum to one")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in choices:
        raise DecisionError(f"{service} selected an ineligible skill")
    if probabilities[choice] < max(probabilities.values()) - 0.000001:
        raise DecisionError(f"{service} choice contradicts its distribution")
    confidence = answer.get("confidence")
    if not number(confidence) or not 0 <= confidence <= 1:
        raise DecisionError(f"Missing or invalid {service} confidence")
    return answer


def evaluate_shadow(state: dict, *, invoke: bool = False, key_file: Path | None = None,
                    provider: str = "typesafe") -> dict:
    """Produce an auditable decision. Return values cannot execute any skill."""
    validate_state(state)
    config = provider_config(provider)
    service = "OpenAI" if provider == "openai" else "Jev"
    result = {
        "schema": CONTRACT_VERSION,
        "mode": "shadow",
        "provider": provider,
        "endpoint": config["endpoint"],
        "requested_model": config["model"],
        "model_version_pinned": provider == "typesafe",
        "state_source": state["source"],
        "state_sha256": digest(state),
        "physical_execution": False,
        "motion_commands_sent": 0,
        "gripper_commands_sent": 0,
        "model_called": False,
        "openai_called": False,
        "jev_called": False,
        "astra_called": False,
        "thresholds_calibrated_on_robot": False,
    }
    if not state["robot"]["stationary"] or not state["robot"]["fault_free"]:
        return {**result, "route": "blocked", "reason": "Robot is moving, faulted, or unavailable"}
    if state["source"] == "live_readonly" and not -0.5 <= time.time() - state["captured_at_unix"] <= 2:
        return {**result, "route": "reobserve", "reason": "Live state is stale"}
    request = make_request(state, provider=provider)
    result.update(request=request, request_sha256=digest(request))
    if not invoke:
        return {**result, "route": "request_prepared", "reason": "No model inference requested"}
    try:
        # A call attempt is separate from a validated model answer.
        api_key(key_file, provider=provider)
        result["model_called"] = True
        result["openai_called"] = provider == "openai"
        result["jev_called"] = provider in ("typesafe", "emrank")
        response, latency_ms = call_jev(request, key_file, provider=provider)
        result.update(response=response, latency_ms=latency_ms, response_model=response.get("model"),
                      response_model_supplied=response.get("model") is not None,
                      upstream_model_identity_independently_verified=False,
                      usage=response.get("usage"))
        answer = validate_answer(request, response, provider=provider)
        result["model_answer"] = answer
        if provider in ("typesafe", "emrank"):
            result["jev_answer"] = answer
        if state["source"] == "live_readonly" and time.time() - state["captured_at_unix"] > 2:
            return {**result, "route": "reobserve", "reason": "State expired during Jev inference"}
        # These are provisional SHADOW thresholds, not a physical safety assurance.
        if answer["confidence"] < 0.8 or answer["probabilities"][answer["choice"]] < 0.85:
            return {**result, "route": "needs_astra", "reason": f"Uncertain {service} selection"}
        if answer["choice"] == "request_astra":
            return {**result, "route": "needs_astra", "reason": f"{service} requested a new strategy or review"}
        route = "openai_selected" if provider == "openai" else "jev_selected"
        return {**result, "route": route, "selected_skill": answer["choice"]}
    except DecisionError as exc:
        return {**result, "route": "needs_astra", "reason": str(exc)}
