"""Decision persistence tests with synthetic state and mocked transport."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from . import experiment
from .decision import make_request
from .fast import compact_state


def state():
    return {
        "schema": "fr5-cup-shadow-v1", "source": "synthetic_test",
        "robot_id": "J1", "robot_ip": "192.168.58.2", "captured_at_unix": time.time(),
        "robot": {"stationary": True, "fault_free": True},
        "camera": {}, "facts": {}, "candidates": {},
        "recent_outcomes": [], "retrieved_examples": [],
    }


def response(choice="observe_scene"):
    return {
        "model": "gpt-6-luna",
        "answers": [{
            "type": "choice", "name": "next_skill", "choice": choice, "confidence": .99,
            "probabilities": [
                {"value": label, "probability": .99 if label == choice else .005}
                for label in ("observe_scene", "request_astra", "hold")
            ],
        }],
    }


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = state()
        self.reply = response()
        self.transport = patch.object(experiment, "call_jev", return_value=(self.reply, 12.))
        self.call = self.transport.start()
        self.addCleanup(self.transport.stop)
        credentials = patch("fr5_jev.decision.api_key", side_effect=AssertionError("No credential access"))
        credentials.start()
        self.addCleanup(credentials.stop)

    def test_openai_request_and_frozen_gates_are_reserved_before_dispatch(self):
        def dispatch(request, key_file, *, provider):
            reserved = json.loads((self.root / "decision.json").read_text())
            self.assertEqual(reserved["provider"], "openai")
            self.assertEqual(reserved["request"], request)
            self.assertEqual(set(reserved["gate_table_before_inference"]),
                             {"observe_scene", "request_astra", "hold"})
            self.assertNotIn("response", reserved)
            return self.reply, 12.
        self.call.side_effect = dispatch
        record, path = experiment.decide(self.snapshot, "decision", directory=self.root)
        self.call.assert_called_once_with(
            make_request(self.snapshot, provider="openai"),
            Path.home() / ".config/openai/robot-decisions-api-key", provider="openai")
        self.assertEqual(record["status"], "decision_received")
        self.assertEqual(record["response"], self.reply)
        self.assertEqual(record["gate"]["route"], "selected")
        self.assertEqual(record["physical_commands_sent_by_decision"], 0)
        self.assertEqual(json.loads(path.read_text()), record)

    def test_compact_request_keeps_full_state_in_record(self):
        self.snapshot["facts"]["previous_astra_planned_descent"] = "synthetic old narrative"
        record, _ = experiment.decide(self.snapshot, "compact", directory=self.root, compact=True)
        self.assertEqual(record["request"], make_request(compact_state(self.snapshot), provider="openai"))
        self.assertEqual(record["state"], self.snapshot)
        self.assertEqual(record["status"], "decision_received")

    def test_wrong_reported_model_is_retained_as_failed_response(self):
        self.reply["model"] = "typesafe/jev-latest"
        record, _ = experiment.decide(self.snapshot, "wrong-model", directory=self.root)
        self.assertEqual(record["provider"], "openai")
        self.assertEqual(record["status"], "decision_failed")
        self.assertEqual(record["response"], self.reply)
        self.assertEqual(record["error_type"], "DecisionError")
        self.assertNotIn("gate", record)
        self.call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
