"""Control-boundary tests with simulated model replies. These are NOT Jev accuracy evaluations."""
import copy
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import decision as d


def state():
    return {
        "schema": d.CONTRACT_VERSION,
        "source": "synthetic_test",
        "robot_id": "J1",
        "robot_ip": "192.168.58.2",
        "captured_at_unix": time.time(),
        "robot": {"stationary": True, "fault_free": True},
        "facts": {"lift_completed": False},
        "candidates": {
            "align_xy": {"validated": True, "preconditions_met": True,
                         "evidence": "Synthetic fixture only; not a real validated primitive"},
            "close_gripper": {"validated": False, "preconditions_met": True,
                              "evidence": "Synthetic fixture: contact has not been verified"},
        },
    }


def response(s, choice="align_xy", confidence=0.95):
    options = d.make_request(s)["questions"]["next_skill"]["criteria"]
    p = {k: 0.01 / (len(options) - 1) for k in options}
    p[choice] = 0.99
    return {"model": d.MODEL, "answers": {"next_skill": {
        "type": "choice", "choice": choice, "confidence": confidence, "probabilities": p,
    }}, "usage": {"input_tokens": 123, "output_tokens": 0}}


class BoundaryTests(unittest.TestCase):
    def infer(self, s, reply):
        with patch.object(d, "api_key", return_value="synthetic-test-key"), \
             patch.object(d, "call_jev", return_value=(reply, 123.0)):
            return d.evaluate_shadow(s, invoke=True)

    def test_default_does_not_call_model(self):
        with patch.object(d, "call_jev") as call:
            result = d.evaluate_shadow(state())
        call.assert_not_called()
        self.assertFalse(result["jev_called"])
        self.assertEqual(result["route"], "request_prepared")

    def test_unverified_motion_cannot_enter_the_closed_set(self):
        self.assertNotIn("close_gripper", d.eligible_skills(state()))

    def test_fault_and_motion_bypass_model(self):
        for field in ("fault_free", "stationary"):
            s = state()
            s["robot"][field] = False
            with patch.object(d, "call_jev") as call:
                result = d.evaluate_shadow(s, invoke=True)
            call.assert_not_called()
            self.assertEqual(result["route"], "blocked")

    def test_selection_is_still_advisory_only(self):
        s = state()
        result = self.infer(s, response(s))
        self.assertEqual(result["route"], "jev_selected")
        self.assertEqual(result["selected_skill"], "align_xy")
        self.assertFalse(result["physical_execution"])
        self.assertFalse(result["astra_called"])
        self.assertEqual(result["motion_commands_sent"], 0)

    def test_uncertainty_routes_to_astra_without_calling_astra(self):
        s = state()
        result = self.infer(s, response(s, confidence=0.1))
        self.assertEqual(result["route"], "needs_astra")
        self.assertFalse(result["astra_called"])

    def test_explicit_astra_selection(self):
        s = state()
        self.assertEqual(self.infer(s, response(s, "request_astra"))["route"], "needs_astra")

    def test_malformed_replies_are_not_selections(self):
        s = state()
        base = response(s)
        mutations = []
        r = copy.deepcopy(base); r["answers"] = []; mutations.append(r)
        r = copy.deepcopy(base); r["model"] = "other-model"; mutations.append(r)
        r = copy.deepcopy(base); r["answers"]["next_skill"]["choice"] = "close_gripper"; mutations.append(r)
        r = copy.deepcopy(base); r["answers"]["next_skill"]["confidence"] = None; mutations.append(r)
        r = copy.deepcopy(base); r["answers"]["next_skill"]["probabilities"]["align_xy"] = float("nan"); mutations.append(r)
        r = copy.deepcopy(base); r["answers"]["next_skill"]["probabilities"]["align_xy"] = 0.2; mutations.append(r)
        r = copy.deepcopy(base); del r["answers"]["next_skill"]["probabilities"]["hold"]; mutations.append(r)
        for reply in mutations:
            with self.subTest(reply=reply):
                self.assertEqual(self.infer(s, reply)["route"], "needs_astra")

    def test_missing_key_does_not_claim_a_jev_call(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(d, "call_jev") as call:
            result = d.evaluate_shadow(state(), invoke=True)
        call.assert_not_called()
        self.assertFalse(result["jev_called"])
        self.assertEqual(result["route"], "needs_astra")

    def test_transport_failure_is_not_a_decision(self):
        with patch.object(d, "api_key", return_value="synthetic-test-key"), \
             patch.object(d, "call_jev", side_effect=d.DecisionError("inference timeout")):
            result = d.evaluate_shadow(state(), invoke=True)
        self.assertEqual(result["route"], "needs_astra")
        self.assertNotIn("selected_skill", result)

    def test_stale_state_is_not_used(self):
        s = state()
        s.update(source="live_readonly", captured_at_unix=time.time() - 30)
        with patch.object(d, "call_jev") as call:
            result = d.evaluate_shadow(s, invoke=True)
        call.assert_not_called()
        self.assertEqual(result["route"], "reobserve")

    def test_state_that_expires_during_inference_is_not_accepted(self):
        s = state()
        s.update(source="live_readonly", captured_at_unix=100)
        with patch.object(d.time, "time", side_effect=[100.1, 103]):
            result = self.infer(s, response(s))
        self.assertEqual(result["route"], "reobserve")

    def test_j2_cannot_be_used_with_the_j1_contract(self):
        s = state()
        s["robot_id"] = "J2"
        with self.assertRaises(d.DecisionError):
            d.evaluate_shadow(s)

    def test_emrank_request_uses_the_users_endpoint_and_model(self):
        result = d.evaluate_shadow(state(), provider="emrank")
        self.assertEqual(result["endpoint"], "https://emrank.tech/v1/decisions")
        self.assertEqual(result["request"]["model"], "typesafe/jev-1.13.0[slow]")
        self.assertFalse(result["model_version_pinned"])

    def test_emrank_does_not_reuse_typesafe_credentials(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "only-for-official-host"}, clear=True):
            with self.assertRaises(d.DecisionError):
                d.api_key(provider="emrank")

    def test_emrank_can_omit_model_metadata_without_fabricating_a_version(self):
        s = state()
        reply = response(s)
        del reply["model"]
        with patch.object(d, "api_key", return_value="synthetic-test-key"), \
             patch.object(d, "call_jev", return_value=(reply, 1440.0)):
            result = d.evaluate_shadow(s, invoke=True, provider="emrank")
        self.assertEqual(result["route"], "jev_selected")
        self.assertIsNone(result["response_model"])
        self.assertFalse(result["response_model_supplied"])
        self.assertFalse(result["upstream_model_identity_independently_verified"])

    def test_emrank_alias_may_resolve_to_a_reported_jev_version(self):
        s = state()
        request = d.make_request(s, provider="emrank")
        reply = response(s)
        reply["model"] = "typesafe/jev-1.14.0"
        self.assertEqual(d.validate_answer(request, reply, provider="emrank")["choice"], "align_xy")
        reply["model"] = "typesafe/jev-1.13.0[slow]"
        self.assertEqual(d.validate_answer(request, reply, provider="emrank")["choice"], "align_xy")
        reply["model"] = "some-chat-model"
        with self.assertRaises(d.DecisionError):
            d.validate_answer(request, reply, provider="emrank")

    def test_successful_noul_is_not_a_valid_next_skill_choice(self):
        reply = {"answers": {"damaged": {"type": "noul", "noul": 0.05}}}
        with self.assertRaises(d.DecisionError):
            d.validate_answer(d.make_request(state(), provider="emrank"), reply, provider="emrank")

    def test_curl_transport_keeps_key_out_of_process_arguments(self):
        s = state()
        request = d.make_request(s, provider="emrank")
        reply = response(s)
        key = "synthetic-test-key"

        def fake_run(command, **kwargs):
            self.assertEqual(command[1:4], ["--disable", "--config", "-"])
            self.assertNotIn(key, " ".join(command))
            self.assertIn("Authorization: Bearer " + key, kwargs["input"])
            self.assertIn("https://emrank.tech/v1/decisions", kwargs["input"])
            self.assertNotIn("--location", command)
            self.assertNotIn("--insecure", command)
            Path(command[command.index("--output") + 1]).write_text(__import__("json").dumps(reply))
            return SimpleNamespace(returncode=0, stdout="200", stderr="")

        with patch.object(d.subprocess, "run", side_effect=fake_run):
            decoded, _ = d.call_emrank_curl(request, key)
        self.assertEqual(decoded, reply)

    def test_curl_http_failure_has_no_model_decision_or_secret_output(self):
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="403", stderr="synthetic-test-key")):
            with self.assertRaises(d.DecisionError) as caught:
                d.call_emrank_curl(d.make_request(state(), provider="emrank"), "synthetic-test-key")
        self.assertNotIn("synthetic-test-key", str(caught.exception))
        self.assertIn("HTTP 403", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
