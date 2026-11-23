"""Read-only CLI and replay tests without model, camera or robot access."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import experiment, workflow
from .decision import DecisionError, make_request
from .test_experiment import response, state


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = state()
        self.reply = response()
        for target, options in (
            ("socket.create_connection", {"side_effect": AssertionError("No network")}),
            ("fr5_jev.decision.api_key", {"side_effect": AssertionError("No credential access")}),
            ("fr5_jev.workflow.collect_live_state", {"return_value": self.snapshot}),
            ("fr5_jev.workflow.retrieve", {"return_value": []}),
        ):
            mock = patch(target, **options)
            mock.start()
            self.addCleanup(mock.stop)
        transport = patch.object(experiment, "call_jev", return_value=(self.reply, 12.))
        self.call = transport.start()
        self.addCleanup(transport.stop)
        capture = patch.object(workflow, "capture")
        self.capture = capture.start()
        self.addCleanup(capture.stop)

    def run_cli(self, mode, output, *options):
        with patch.object(sys, "argv", ["workflow", mode, "--output", str(output), *options]), \
             contextlib.redirect_stdout(io.StringIO()):
            workflow.main()

    def test_observe_calls_openai_and_only_dispatches_readonly_capture(self):
        self.snapshot["candidates"]["align_xy"] = {
            "validated": True, "preconditions_met": True, "evidence": "synthetic"}
        output = self.root / "observe"
        self.run_cli("observe", output, "--call-jev")
        record = json.loads((output / "decision.json").read_text())
        self.call.assert_called_once_with(
            record["request"], Path.home() / ".config/openai/robot-decisions-api-key", provider="openai")
        self.assertEqual(record["provider"], "openai")
        self.assertEqual(record["request"]["model"], "gpt-6-luna")
        self.assertEqual(record["state"]["candidates"], {})
        self.assertEqual(record["response"], self.reply)
        self.capture.assert_called_once_with(output / "observation",
                                             decision_record=str(output / "decision.json"))
        outcome = json.loads((output / "outcome.json").read_text())
        self.assertTrue(outcome["readonly_observation_completed"])
        self.assertEqual(outcome["physical_commands_sent"], 0)

    def test_status_and_noninvoking_modes_prepare_official_requests(self):
        for mode, options in (("status", ("--call-jev",)), ("observe", ()), ("review", ())):
            with self.subTest(mode=mode):
                output = self.root / mode
                self.run_cli(mode, output, *options)
                request = json.loads((output / "prepared-request.json").read_text())
                self.assertEqual(request["model"], "gpt-6-luna")
                self.assertIsInstance(request["questions"], list)
                self.assertIn("input", request)
                self.assertNotIn("state", request)
        self.call.assert_not_called()
        self.capture.assert_not_called()

    def test_review_calls_openai_with_recorded_annotations(self):
        annotations = {"object_description": "synthetic cup"}
        context = self.root / "context.json"
        context.write_text(json.dumps(annotations))
        self.reply.update(response("hold"))
        output = self.root / "review"
        self.run_cli("review", output, "--call-jev", "--context-file", str(context))
        record = json.loads((output / "decision.json").read_text())
        self.assertEqual(record["state"]["llm_annotations"], annotations)
        self.assertEqual(record["provider"], "openai")
        self.assertEqual(record["gate"]["choice"], "hold")
        self.call.assert_called_once()
        self.capture.assert_not_called()

    def replay(self, record, name):
        source = self.root / (name + ".json")
        source.write_text(json.dumps(record))
        before = source.read_text()
        output = self.root / name
        self.run_cli("replay", output, "--record", str(source))
        self.assertEqual(source.read_text(), before)
        return json.loads((output / "replay.json").read_text())

    def test_openai_replay_uses_raw_experiment_and_shadow_responses(self):
        detail = {"provider": "openai", "request": make_request(state(), provider="openai"),
                  "response": response()}
        for name, record in (("experiment", detail), ("shadow", {"result": detail})):
            with self.subTest(format=name):
                result = self.replay(record, name)
                self.assertEqual(result["provider"], "openai")
                self.assertEqual(result["dynamic_gate"]["choice"], "observe_scene")
                self.assertFalse(result["model_called"])
        self.call.assert_not_called()
        self.capture.assert_not_called()

    def test_legacy_emrank_replay_preserves_raw_and_answer_only_records(self):
        request = make_request(state(), provider="emrank")
        request["model"] = "typesafe/jev-latest"
        answer = {"type": "choice", "choice": "observe_scene", "confidence": .99,
                  "probabilities": {"observe_scene": .99, "request_astra": .005, "hold": .005}}
        records = {
            "legacy-experiment": {"request": request, "response": {
                "model": "typesafe/jev-latest", "answers": {"next_skill": answer}}},
            "legacy-shadow": {"result": {"provider": "emrank", "request": request,
                "response_model": "typesafe/jev-latest", "jev_answer": answer}},
            "unreported-model": {"result": {"provider": "emrank", "request": request,
                "jev_answer": answer}},
        }
        for name, record in records.items():
            with self.subTest(format=name):
                result = self.replay(record, name)
                self.assertEqual(result["provider"], "emrank")
                self.assertEqual(result["dynamic_gate"]["route"], "selected")
        self.call.assert_not_called()
        self.capture.assert_not_called()

    def test_openai_replay_cannot_reconstruct_a_raw_response_from_jev_answer(self):
        record = {"provider": "openai", "request": make_request(state(), provider="openai"),
                  "response_model": "gpt-6-luna", "jev_answer": {"choice": "observe_scene"}}
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.replay(record, "missing-response")
        self.assertEqual(error.exception.code, 2)
        self.call.assert_not_called()

    def test_replay_validates_the_actual_reported_model(self):
        reply = response()
        reply["model"] = "typesafe/jev-latest"
        record = {"provider": "openai", "request": make_request(state(), provider="openai"),
                  "response": reply, "response_model": "gpt-6-luna"}
        with self.assertRaises(DecisionError):
            self.replay(record, "wrong-model")
        self.call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
