"""Shared FR5 state reads across worker threads."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fr5_runtime import robot
from fr5_runtime.robot import read_state  # Imported before installing a shared scope.


class SharedStateTests(unittest.TestCase):
    def stream(self):
        state = {"joints": [1.] * 6, "main_code": 0}
        return SimpleNamespace(host=robot.HOST, port=20004,
                               latest=Mock(side_effect=lambda **kw: deepcopy(state)))

    def test_worker_consumers_share_one_reader_and_never_open_a_socket(self):
        stream = self.stream()
        with patch.object(robot.socket, "create_connection", side_effect=AssertionError("extra socket")):
            with robot.use_shared_state_stream(stream):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    states = list(pool.map(lambda _: read_state(), range(12)))
                states[0]["joints"][0] = 999
                self.assertEqual(read_state()["joints"][0], 1)
        self.assertTrue(all(call.kwargs == {"max_age_s": .25} for call in stream.latest.call_args_list))

    def test_stale_stream_raises_without_fallback_and_scope_restores(self):
        stream = self.stream()
        with patch.object(robot, "_read_state_once", return_value={"direct": True}) as direct:
            with self.assertRaisesRegex(TimeoutError, "stale"):
                with robot.use_shared_state_stream(stream):
                    stream.latest.side_effect = TimeoutError("stale")
                    read_state()
            direct.assert_not_called()
            self.assertEqual(read_state(), {"direct": True})
            direct.assert_called_once()

    def test_wrong_endpoint_and_overlapping_scope_rejected(self):
        stream = self.stream()
        stream.host = "192.168.58.3"
        with self.assertRaises(ValueError):
            with robot.use_shared_state_stream(stream):
                pass
        stream.host = robot.HOST
        with robot.use_shared_state_stream(stream):
            with self.assertRaises(RuntimeError):
                with robot.use_shared_state_stream(stream):
                    pass
            self.assertEqual(read_state()["main_code"], 0)


if __name__ == "__main__":
    unittest.main()
