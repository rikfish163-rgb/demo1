"""Shadow CLI provider tests with synthetic replies and mocked credentials."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import decision, shadow
from .test_experiment import response, state


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = state()
        self.source = self.root / "state.json"
        self.source.write_text(json.dumps(self.snapshot))
        network = patch("socket.create_connection", side_effect=AssertionError("No network"))
        network.start()
        self.addCleanup(network.stop)

    def run_cli(self, name, *options):
        output = self.root / (name + ".json")
        with patch.object(sys, "argv", ["shadow", "--state", str(self.source),
                                       "--output", str(output), *options]), \
             contextlib.redirect_stdout(io.StringIO()):
            shadow.main()
        return json.loads(output.read_text())["result"]

    def test_default_prepares_openai_without_credentials_or_transport(self):
        with patch.object(decision, "api_key", side_effect=AssertionError("No credential access")), \
             patch.object(decision, "call_jev", side_effect=AssertionError("No model call")):
            result = self.run_cli("prepared")
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["endpoint"], "https://api.openai.com/v1/decisions")
        self.assertEqual(result["requested_model"], "gpt-6-luna")
        self.assertIsInstance(result["request"]["questions"], list)
        self.assertFalse(result["model_called"])
        self.assertFalse(result["jev_called"])

    def test_default_call_passes_only_the_openai_credential_path(self):
        reply = response()
        with patch.object(decision, "api_key", return_value="synthetic-key"), \
             patch.object(decision, "call_jev", return_value=(reply, 12.)) as call:
            result = self.run_cli("called", "--call-jev")
        call.assert_called_once_with(
            result["request"], Path.home() / ".config/openai/robot-decisions-api-key", provider="openai")
        self.assertEqual(result["route"], "openai_selected")
        self.assertEqual(result["response"], reply)
        self.assertTrue(result["openai_called"])
        self.assertFalse(result["jev_called"])
        self.assertNotIn("jev_answer", result)
        self.assertEqual(result["motion_commands_sent"], 0)
        self.assertEqual(result["gripper_commands_sent"], 0)

    def test_explicit_key_file_is_passed_unchanged(self):
        key_file = self.root / "configured-key-path"
        with patch.object(decision, "api_key", return_value="synthetic-key"), \
             patch.object(decision, "call_jev", return_value=(response(), 12.)) as call:
            result = self.run_cli("explicit-key", "--call-jev", "--key-file", str(key_file))
        call.assert_called_once_with(result["request"], key_file, provider="openai")
        self.assertFalse(key_file.exists())

    def test_explicit_legacy_providers_keep_their_model_and_credentials(self):
        answer = {"type": "choice", "choice": "observe_scene", "confidence": .99,
                  "probabilities": {"observe_scene": .99, "request_astra": .005, "hold": .005}}
        for provider in ("emrank", "typesafe"):
            with self.subTest(provider=provider):
                reply = {"model": decision.PROVIDERS[provider]["model"],
                         "answers": {"next_skill": answer}}
                with patch.object(Path, "is_file", return_value=True), \
                     patch.object(decision, "api_key", return_value="synthetic-key"), \
                     patch.object(decision, "call_jev", return_value=(reply, 12.)) as call:
                    result = self.run_cli(provider, "--call-jev", "--provider", provider)
                key_file = Path.home() / ".config/emrank/api-key" if provider == "emrank" else None
                call.assert_called_once_with(result["request"], key_file, provider=provider)
                self.assertEqual(result["provider"], provider)
                self.assertEqual(result["route"], "jev_selected")
                self.assertFalse(result["openai_called"])
                self.assertTrue(result["jev_called"])


if __name__ == "__main__":
    unittest.main()
