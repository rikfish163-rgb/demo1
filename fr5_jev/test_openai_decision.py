"""OpenAI Decisions contract tests with synthetic replies and mocked transport."""
import copy
import io
import json
import os
from pathlib import Path
import unittest
import urllib.error
from unittest.mock import patch

from . import decision as d
from .test_decision import response as legacy_response, state


LABELS = ("observe_scene", "request_astra", "hold", "align_xy")


def response(choice="align_xy", confidence=0.95):
    return {
        "model": "gpt-6-luna",
        "answers": [{
            "type": "choice", "name": "next_skill", "choice": choice,
            "confidence": confidence,
            "probabilities": [{"value": label, "probability": 0.99 if label == choice else 0.01 / 3}
                              for label in LABELS],
        }],
        "usage": {"input_tokens": 123, "output_tokens": 10},
    }


class OpenAIDecisionTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = state()
        self.request = d.make_request(self.snapshot, provider="openai")

    def infer(self, reply):
        with patch.object(d, "api_key", return_value="synthetic-openai-key"), \
             patch.object(d, "call_jev", return_value=(reply, 123.0)) as call:
            result = d.evaluate_shadow(self.snapshot, invoke=True, provider="openai")
        call.assert_called_once_with(self.request, None, provider="openai")
        return result

    def test_request_uses_official_schema_and_preserves_snapshot_and_instructions(self):
        self.snapshot["facts"]["object_description"] = "Synthetic object fixture"
        self.snapshot["retrieved_examples"] = [{"outcome": "synthetic_test", "detail": [1, 2]}]
        before = copy.deepcopy(self.snapshot)
        legacy = d.make_request(self.snapshot)
        request = d.make_request(self.snapshot, provider="openai")
        self.assertEqual(request, {
            "model": "gpt-6-luna",
            "input": [{"role": "user", "content": [{"type": "input_json", "data": before}]}],
            "questions": [{
                "type": "choice", "name": "next_skill",
                "instructions": legacy["questions"]["next_skill"]["instructions"],
                "choices": [{"value": label, "description": d.SKILLS[label]} for label in LABELS],
            }],
        })
        self.assertEqual(self.snapshot, before)
        self.assertNotIn("close_gripper", [entry["value"] for entry in request["questions"][0]["choices"]])

    def test_legacy_requests_and_model_validation_remain_separate(self):
        for provider, model in (("typesafe", "jev-1.13.0"), ("emrank", "typesafe/jev-1.13.0[slow]")):
            with self.subTest(provider=provider):
                request = d.make_request(self.snapshot, provider=provider)
                self.assertEqual(request["model"], model)
                self.assertEqual(request["state"], self.snapshot)
                self.assertIsInstance(request["questions"], dict)
                self.assertNotIn("input", request)
                reply = legacy_response(self.snapshot)
                self.assertEqual(d.validate_answer(request, reply, provider=provider)["choice"], "align_xy")
                reply["model"] = "gpt-6-luna"
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(request, reply, provider=provider)

    def test_answer_converts_probability_list_without_mutating_raw_response(self):
        reply = response()
        reply["id"] = "synthetic-decision-id"
        before = copy.deepcopy(reply)
        answer = d.validate_answer(self.request, reply, provider="openai")
        self.assertEqual(answer, {
            "type": "choice", "choice": "align_xy", "confidence": 0.95,
            "probabilities": {label: 0.99 if label == "align_xy" else 0.01 / 3 for label in LABELS},
        })
        self.assertEqual(reply, before)
        answer["probabilities"]["align_xy"] = 0
        self.assertEqual(reply, before)

    def test_probability_order_is_irrelevant_and_values_are_not_normalized(self):
        reply = response()
        reply["answers"][0]["probabilities"][0]["probability"] += 0.0005
        reply["answers"][0]["probabilities"].reverse()
        expected = {entry["value"]: entry["probability"] for entry in reply["answers"][0]["probabilities"]}
        self.assertEqual(d.validate_answer(self.request, reply, provider="openai")["probabilities"], expected)

    def test_refusal_is_never_an_executable_choice(self):
        replies = []
        reply = response(); reply["answers"] = [{"type": "refusal", "name": "next_skill"}]; replies.append(reply)
        reply = response(); reply["refusal"] = "Synthetic refusal"; replies.append(reply)
        reply = response(); reply["answers"][0]["refusal"] = "Synthetic refusal"; replies.append(reply)
        reply = response(); reply["answers"].append({"type": "refusal", "name": "next_skill"}); replies.append(reply)
        for reply in replies:
            with self.subTest(reply=reply), self.assertRaises(d.DecisionError):
                d.validate_answer(self.request, reply, provider="openai")

    def test_response_requires_the_exact_reported_model(self):
        for model in (None, "", "jev-1.13.0", "gpt-6-luna-latest", "other-model", True):
            with self.subTest(model=model):
                reply = response()
                reply["model"] = model
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")
        reply = response()
        del reply["model"]
        with self.assertRaises(d.DecisionError):
            d.validate_answer(self.request, reply, provider="openai")

    def test_request_model_cannot_be_changed_or_omitted(self):
        for model in (None, "jev-1.13.0", "other-model"):
            with self.subTest(model=model):
                request = copy.deepcopy(self.request)
                request["model"] = model
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(request, response(), provider="openai")
        request = copy.deepcopy(self.request)
        del request["model"]
        with self.assertRaises(d.DecisionError):
            d.validate_answer(request, response(), provider="openai")

    def test_answer_requires_a_unique_named_choice(self):
        for answers in (None, {}, [], [None], [response()["answers"][0]] * 2,
                        [{"type": "choice", "name": "other"}],
                        [{"type": "noul", "name": "next_skill"}]):
            with self.subTest(answers=answers):
                reply = response()
                reply["answers"] = answers
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")

    def test_wrong_probability_labels_are_rejected(self):
        for label in ("close_gripper", "", None):
            with self.subTest(label=label):
                reply = response()
                reply["answers"][0]["probabilities"][0]["value"] = label
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")
        reply = response()
        reply["answers"][0]["probabilities"].pop()
        with self.assertRaises(d.DecisionError):
            d.validate_answer(self.request, reply, provider="openai")
        reply = response()
        reply["answers"][0]["probabilities"].append({"value": "close_gripper", "probability": 0})
        with self.assertRaises(d.DecisionError):
            d.validate_answer(self.request, reply, provider="openai")

    def test_duplicate_probability_labels_are_rejected_before_dict_conversion(self):
        reply = response()
        reply["answers"][0]["probabilities"].append(copy.deepcopy(reply["answers"][0]["probabilities"][0]))
        with self.assertRaisesRegex(d.DecisionError, "Duplicate"):
            d.validate_answer(self.request, reply, provider="openai")

    def test_duplicate_request_labels_are_rejected(self):
        request = copy.deepcopy(self.request)
        request["questions"][0]["choices"].append(copy.deepcopy(request["questions"][0]["choices"][0]))
        with self.assertRaisesRegex(d.DecisionError, "Duplicate"):
            d.validate_answer(request, response(), provider="openai")

    def test_missing_or_legacy_probability_shapes_are_rejected(self):
        for probabilities in (None, {}, {"align_xy": 1}, [], [None], [{"value": "align_xy"}]):
            with self.subTest(probabilities=probabilities):
                reply = response()
                reply["answers"][0]["probabilities"] = probabilities
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")

    def test_nonfinite_out_of_range_or_nonnumeric_probabilities_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), -0.1, 1.1, None, True, "0.99"):
            with self.subTest(value=value):
                reply = response()
                reply["answers"][0]["probabilities"][-1]["probability"] = value
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")

    def test_probability_totals_and_contradictory_choices_are_rejected(self):
        for value in (0.5, 1.0):
            with self.subTest(value=value):
                reply = response()
                reply["answers"][0]["probabilities"][-1]["probability"] = value
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")
        reply = response()
        reply["answers"][0]["choice"] = "hold"
        with self.assertRaises(d.DecisionError):
            d.validate_answer(self.request, reply, provider="openai")

    def test_choice_must_be_an_eligible_string(self):
        for choice in ("close_gripper", None, True, ["align_xy"]):
            with self.subTest(choice=choice):
                reply = response()
                reply["answers"][0]["choice"] = choice
                with self.assertRaises(d.DecisionError):
                    d.validate_answer(self.request, reply, provider="openai")

    def test_confidence_must_be_finite_numeric_and_in_range(self):
        for value in (float("nan"), float("inf"), -float("inf"), -0.1, 1.1, None, True, "0.95"):
            with self.subTest(value=value), self.assertRaises(d.DecisionError):
                d.validate_answer(self.request, response(confidence=value), provider="openai")

    def test_valid_probability_boundaries_and_ties_are_accepted(self):
        reply = response(confidence=1)
        for entry in reply["answers"][0]["probabilities"]:
            entry["probability"] = 1 if entry["value"] == "align_xy" else 0
        self.assertEqual(d.validate_answer(self.request, reply, provider="openai")["confidence"], 1)
        for entry in reply["answers"][0]["probabilities"]:
            entry["probability"] = 0.25
        self.assertEqual(d.validate_answer(self.request, reply, provider="openai")["choice"], "align_xy")

    def test_credentials_are_isolated_in_both_directions(self):
        environments = (
            ("openai", {"TYPESAFE_API_KEY": "synthetic-typesafe-key", "EMRANK_API_KEY": "synthetic-emrank-key"}),
            ("typesafe", {"OPENAI_API_KEY": "synthetic-openai-key"}),
            ("emrank", {"OPENAI_API_KEY": "synthetic-openai-key"}),
        )
        for provider, environment in environments:
            with self.subTest(provider=provider), patch.dict(os.environ, environment, clear=True), \
                 patch.object(Path, "read_text") as read, self.assertRaises(d.DecisionError):
                d.api_key(provider=provider)
            read.assert_not_called()

    def test_each_provider_uses_only_its_own_environment_key(self):
        environment = {"OPENAI_API_KEY": " synthetic-openai-key ",
                       "TYPESAFE_API_KEY": "synthetic-typesafe-key", "EMRANK_API_KEY": "synthetic-emrank-key"}
        with patch.dict(os.environ, environment, clear=True), patch.object(Path, "read_text") as read:
            for provider, expected in (("openai", "synthetic-openai-key"), ("typesafe", "synthetic-typesafe-key"),
                                       ("emrank", "synthetic-emrank-key")):
                self.assertEqual(d.api_key(Path("synthetic-unused-key-file"), provider=provider), expected)
        read.assert_not_called()

    def test_explicit_key_file_is_supported_without_implicit_credential_reuse(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": " ", "EMRANK_API_KEY": "synthetic-emrank-key"}, clear=True), \
             patch.object(Path, "read_text", return_value=" synthetic-file-key\n") as read:
            self.assertEqual(d.api_key(Path("synthetic-key-file"), provider="openai"), "synthetic-file-key")
        read.assert_called_once_with()

    def test_credential_errors_do_not_expose_file_content_or_key(self):
        for key in ("synthetic-key\ninjected", "synthetic-key\rinjected"):
            with self.subTest(key=key), patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True):
                with self.assertRaises(d.DecisionError) as caught:
                    d.api_key(provider="openai")
                self.assertNotIn("synthetic-key", str(caught.exception))
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(Path, "read_text", side_effect=OSError("synthetic-private-content")):
            with self.assertRaises(d.DecisionError) as caught:
                d.api_key(Path("synthetic-key-file"), provider="openai")
        self.assertNotIn("synthetic-private-content", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_openai_transport_posts_once_to_the_fixed_host_with_no_redirect_handler(self):
        reply = response()
        environment = {"OPENAI_API_KEY": "synthetic-openai-key", "FR5_JEV_TRANSPORT": "ssh-4090",
                       "OPENAI_BASE_URL": "https://synthetic-other-host.invalid"}
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(d.urllib.request, "build_opener") as build, \
             patch.object(d.subprocess, "run") as process:
            wire = build.return_value.open.return_value.__enter__.return_value
            wire.read.return_value = json.dumps(reply).encode()
            decoded, latency = d.call_jev(self.request, provider="openai")
        build.assert_called_once_with(d.NoRedirect)
        build.return_value.open.assert_called_once()
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/decisions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data), self.request)
        self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-openai-key")
        self.assertNotIn(b"synthetic-openai-key", request.data)
        self.assertEqual(build.return_value.open.call_args.kwargs, {"timeout": 10})
        wire.read.assert_called_once_with(1_000_001)
        process.assert_not_called()
        self.assertEqual(decoded, reply)
        self.assertGreaterEqual(latency, 0)

    def test_transport_rejects_wrong_model_before_reading_credentials(self):
        request = copy.deepcopy(self.request)
        request["model"] = "jev-1.13.0"
        with patch.object(d, "api_key") as key, patch.object(d.urllib.request, "build_opener") as build:
            with self.assertRaises(d.DecisionError):
                d.call_jev(request, provider="openai")
        key.assert_not_called()
        build.assert_not_called()

    def test_http_errors_have_no_retry_or_secret_output(self):
        for code in (401, 402, 403):
            with self.subTest(code=code), patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-openai-key"}, clear=True), \
                 patch.object(d.urllib.request, "build_opener") as build, patch.object(d.subprocess, "run") as process:
                build.return_value.open.side_effect = urllib.error.HTTPError(
                    "https://api.openai.com/v1/decisions", code, "synthetic-private-message",
                    {"Authorization": "Bearer synthetic-openai-key"}, io.BytesIO(b"synthetic-private-body"))
                with self.assertRaises(d.DecisionError) as caught:
                    d.call_jev(self.request, provider="openai")
                self.assertIn(f"OpenAI HTTP {code}", str(caught.exception))
                for secret in ("synthetic-openai-key", "synthetic-private-message", "synthetic-private-body"):
                    self.assertNotIn(secret, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                build.return_value.open.assert_called_once()
                process.assert_not_called()

    def test_transport_errors_and_invalid_json_do_not_expose_private_messages(self):
        errors = (urllib.error.URLError("synthetic-private-message"), TimeoutError("synthetic-private-message"),
                  OSError("synthetic-private-message"), d.DecisionError("synthetic-private-message"))
        for error in errors:
            with self.subTest(error=type(error).__name__), patch.object(d, "api_key", return_value="synthetic-openai-key"), \
                 patch.object(d.urllib.request, "build_opener") as build:
                build.return_value.open.side_effect = error
                with self.assertRaises(d.DecisionError) as caught:
                    d.call_jev(self.request, provider="openai")
                self.assertNotIn("synthetic-private-message", str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                build.return_value.open.assert_called_once()
        for raw in (b"synthetic-private-json", b'["synthetic-private-json"]'):
            with self.subTest(raw=raw), patch.object(d, "api_key", return_value="synthetic-openai-key"), \
                 patch.object(d.urllib.request, "build_opener") as build:
                build.return_value.open.return_value.__enter__.return_value.read.return_value = raw
                with self.assertRaises(d.DecisionError) as caught:
                    d.call_jev(self.request, provider="openai")
                self.assertNotIn("synthetic-private-json", str(caught.exception))
                build.return_value.open.assert_called_once()

    def test_redirect_is_rejected_without_forwarding_credentials(self):
        request = d.urllib.request.Request("https://api.openai.com/v1/decisions",
                                          headers={"Authorization": "Bearer synthetic-openai-key"})
        with self.assertRaises(d.DecisionError) as caught:
            d.NoRedirect().redirect_request(request, None, 302, "synthetic-private-message", {},
                                            "https://synthetic-other-host.invalid")
        self.assertNotIn("synthetic-openai-key", str(caught.exception))
        self.assertNotIn("synthetic-private-message", str(caught.exception))

    def test_shadow_dry_run_has_accurate_model_fields_and_no_credential_access(self):
        with patch.object(d, "api_key") as key, patch.object(d, "call_jev") as call:
            result = d.evaluate_shadow(self.snapshot, provider="openai")
        key.assert_not_called()
        call.assert_not_called()
        self.assertEqual(result["endpoint"], "https://api.openai.com/v1/decisions")
        self.assertEqual(result["requested_model"], "gpt-6-luna")
        self.assertEqual(result["request"], self.request)
        self.assertFalse(result["model_version_pinned"])
        for field in ("model_called", "openai_called", "jev_called", "physical_execution", "astra_called"):
            self.assertFalse(result[field])
        self.assertEqual(result["route"], "request_prepared")

    def test_shadow_records_openai_and_raw_response_without_claiming_jev(self):
        reply = response()
        result = self.infer(reply)
        self.assertEqual(result["route"], "openai_selected")
        self.assertEqual(result["selected_skill"], "align_xy")
        self.assertTrue(result["model_called"])
        self.assertTrue(result["openai_called"])
        self.assertFalse(result["jev_called"])
        self.assertNotIn("jev_answer", result)
        self.assertEqual(result["model_answer"], d.validate_answer(self.request, reply, provider="openai"))
        self.assertEqual(result["response"], reply)
        self.assertEqual(result["response_model"], "gpt-6-luna")
        self.assertTrue(result["response_model_supplied"])
        self.assertFalse(result["model_version_pinned"])
        self.assertFalse(result["upstream_model_identity_independently_verified"])
        self.assertEqual(result["usage"], reply["usage"])
        self.assertFalse(result["physical_execution"])
        self.assertFalse(result["astra_called"])
        self.assertEqual(result["motion_commands_sent"], 0)
        self.assertEqual(result["gripper_commands_sent"], 0)

    def test_legacy_shadow_keeps_jev_fields_and_route(self):
        reply = legacy_response(self.snapshot)
        with patch.object(d, "api_key", return_value="synthetic-typesafe-key"), \
             patch.object(d, "call_jev", return_value=(reply, 123.0)):
            result = d.evaluate_shadow(self.snapshot, invoke=True)
        self.assertEqual(result["route"], "jev_selected")
        self.assertTrue(result["jev_called"])
        self.assertTrue(result["model_called"])
        self.assertFalse(result["openai_called"])
        self.assertEqual(result["jev_answer"], reply["answers"]["next_skill"])
        self.assertEqual(result["model_answer"], result["jev_answer"])

    def test_missing_openai_key_does_not_claim_any_call(self):
        with patch.dict(os.environ, {"EMRANK_API_KEY": "synthetic-emrank-key"}, clear=True), \
             patch.object(d, "call_jev") as call, patch.object(Path, "read_text") as read:
            result = d.evaluate_shadow(self.snapshot, invoke=True, provider="openai")
        call.assert_not_called()
        read.assert_not_called()
        for field in ("model_called", "openai_called", "jev_called"):
            self.assertFalse(result[field])
        self.assertEqual(result["route"], "needs_astra")
        self.assertNotIn("synthetic-emrank-key", result["reason"])

    def test_shadow_transport_failure_is_an_attempt_without_a_decision(self):
        with patch.object(d, "api_key", return_value="synthetic-openai-key"), \
             patch.object(d, "call_jev", side_effect=d.DecisionError("OpenAI HTTP 402; no action dispatched")) as call:
            result = d.evaluate_shadow(self.snapshot, invoke=True, provider="openai")
        call.assert_called_once()
        self.assertTrue(result["model_called"])
        self.assertTrue(result["openai_called"])
        self.assertFalse(result["jev_called"])
        self.assertEqual(result["route"], "needs_astra")
        self.assertNotIn("model_answer", result)
        self.assertNotIn("selected_skill", result)

    def test_shadow_refusal_preserves_response_without_selecting_a_skill(self):
        reply = response()
        reply["answers"] = [{"type": "refusal", "name": "next_skill", "refusal": "Synthetic refusal"}]
        result = self.infer(reply)
        self.assertEqual(result["response"], reply)
        self.assertEqual(result["response_model"], "gpt-6-luna")
        self.assertTrue(result["openai_called"])
        self.assertFalse(result["jev_called"])
        self.assertEqual(result["route"], "needs_astra")
        self.assertFalse(result["astra_called"])
        self.assertNotIn("model_answer", result)
        self.assertNotIn("selected_skill", result)

    def test_shadow_does_not_fabricate_missing_response_model(self):
        reply = response()
        del reply["model"]
        result = self.infer(reply)
        self.assertIsNone(result["response_model"])
        self.assertFalse(result["response_model_supplied"])
        self.assertTrue(result["openai_called"])
        self.assertEqual(result["route"], "needs_astra")
        self.assertNotIn("model_answer", result)
        self.assertNotIn("selected_skill", result)

    def test_openai_uncertainty_and_explicit_astra_choice_do_not_call_astra(self):
        for reply in (response(confidence=0.1), response(choice="request_astra")):
            with self.subTest(reply=reply):
                result = self.infer(reply)
                self.assertEqual(result["route"], "needs_astra")
                self.assertIn("OpenAI", result["reason"])
                self.assertFalse(result["astra_called"])
                self.assertNotIn("selected_skill", result)


if __name__ == "__main__":
    unittest.main()
