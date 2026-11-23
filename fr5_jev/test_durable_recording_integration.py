"""Recording failure gates with fake RPC and synthetic files only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .teacher_targets import TeacherSession, main
from . import teacher_targets
from .test_teacher_targets import FakeRobot, FakeRPC, FakeStorage, scope
from . import durable_recording_probe
from fr5_lerobot.tests.test_continuous_capture import FakeRobot as CaptureRobot
from lerobot_robot_fairino.continuous_capture import ContinuousCapture


class DurableRecordingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-spool-integration-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "scope.json").write_text(json.dumps(scope()))
        self.robot = FakeRobot()
        self.robot.connect = Mock()
        self.rpc = FakeRPC(self.robot)
        self.session = TeacherSession(self.root, robot=self.robot, rpc=self.rpc)
        self.session.capture = SimpleNamespace(assert_healthy=Mock())
        self.session.holding_position = 76
        self.session.append([*self.robot.s["joints"], 76], "hold", False)
        self.target = [0., 0., .01, 0., 90., 0., 76.]
        self.destination = [0., 0., 101., 0., 0., 0.]

    def send(self):
        return self.session.send("MoveL", (self.target[:6], self.destination), self.target,
                                 kind="arm", destination_pose=self.destination)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("Timed out waiting for the synthetic test event")

    def start_synthetic_stream(self):
        self.session.writer = SimpleNamespace(add=Mock())
        capture = ContinuousCapture(CaptureRobot(), self.root / "synthetic-stream", fps=10,
                                    task="SYNTHETIC ONLY", sample_callback=self.session.sample).start()
        self.session.capture = capture

        def finish():
            if capture._started and capture._report is None:
                capture.finish("interrupted")

        self.addCleanup(finish)
        return capture

    def test_preflight_failure_does_not_connect_or_claim_session(self):
        self.session.storage = FakeStorage()
        with patch("fr5_jev.teacher_targets.DurableStorage.preflight",
                   side_effect=RuntimeError("synthetic server unavailable")):
            with self.assertRaisesRegex(RuntimeError, "server unavailable"):
                self.session.start()
        self.robot.connect.assert_not_called()
        self.assertIsNone(self.session.storage)
        self.assertFalse((self.root / "SESSION_STARTED").exists())
        self.assertEqual(self.rpc.calls, [])

    def test_initial_camera_cache_is_not_paired_with_a_future_hold(self):
        clock = [100.]
        virtual_time = SimpleNamespace(monotonic=lambda: clock[0],
                                       sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds))
        command = {'issued_unix':1000., 'issued_monotonic':100.}
        self.robot.get_sample = Mock(side_effect=[
            {'camera':{'metadata':{'source_timestamp_unix':999.99, 'source_timestamp_monotonic':99.99}}},
            {'camera':{'metadata':{'source_timestamp_unix':1000.01, 'source_timestamp_monotonic':100.01}}},
        ])
        with patch.object(teacher_targets, 'time', virtual_time):
            self.session.wait_for_initial_observation(command)
        saved = json.loads((self.root/'recording-start.json').read_text())
        self.assertEqual(saved['unrecorded_pre_hold_observations'], 1)
        self.assertGreaterEqual(saved['first_eligible_source_time'], saved['initial_hold_issue_time'])
        self.assertEqual(self.robot.get_sample.call_count, 2)
        self.assertEqual(self.rpc.calls, [])

    def test_camera_that_never_advances_stops_before_recording_or_actuation(self):
        clock = [100.]
        virtual_time = SimpleNamespace(monotonic=lambda: clock[0],
                                       sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds))
        self.robot.get_sample = Mock(return_value={
            'camera':{'metadata':{'source_timestamp_unix':999., 'source_timestamp_monotonic':99.}}})
        with patch.object(teacher_targets, 'time', virtual_time):
            with self.assertRaisesRegex(TimeoutError, 'No camera observation newer'):
                self.session.wait_for_initial_observation(
                    {'issued_unix':1000.,'issued_monotonic':100.}, timeout_s=.1)
        self.assertFalse((self.root/'recording-start.json').exists())
        self.assertEqual(self.rpc.calls, [])

    def test_unverified_stream_root_is_blocked_before_connection(self):
        with patch("fr5_jev.stream_storage.LOCAL_STREAM_BASE", self.root.parent), \
                patch("fr5_jev.stream_storage.StreamStorage.start_for_episode",
                      side_effect=RuntimeError("Verified stream readiness is unavailable")):
            with self.assertRaisesRegex(RuntimeError, "readiness"):
                self.session.start()
        self.robot.connect.assert_not_called()
        self.assertIsNone(self.session.storage)
        self.assertFalse((self.root / "SESSION_STARTED").exists())
        self.assertEqual(self.rpc.calls, [])

    def test_known_slow_gvfs_recording_is_blocked_before_connection(self):
        with patch("fr5_jev.durable_storage.LOCAL_BASE", self.root.parent):
            with self.assertRaisesRegex(RuntimeError, "GVFS recording failed"):
                self.session.start()
        self.robot.connect.assert_not_called()
        self.assertIsNone(self.session.storage)
        self.assertFalse((self.root / "SESSION_STARTED").exists())
        self.assertEqual(self.rpc.calls, [])

    def test_checkpoint_cache_limit_is_checked_before_observation(self):
        check = Mock(side_effect=RuntimeError("synthetic cache full"))
        observer = Mock()
        self.session.storage = SimpleNamespace(ensure_local_capacity=check)
        with self.assertRaisesRegex(RuntimeError, "cache full"):
            self.session.observe_bounded(self.root / "view-close", observer=observer)
        observer.assert_not_called()
        check.assert_called_once_with(reserve_bytes=24 * 1024**2)

    def test_missing_storage_and_unconfirmed_remote_write_refuse_dispatch(self):
        with self.assertRaisesRegex(RuntimeError, "durable recording"):
            self.send()
        self.assertEqual(self.rpc.calls, [])
        self.session.storage = SimpleNamespace(sync_files=Mock(side_effect=OSError("synthetic failure")))
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            self.send()
        self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
        self.assertFalse(any(method == "StopMotion" for method, _ in self.rpc.calls))

    def test_cli_preflight_failure_does_not_write_stop_or_disconnect(self):
        robot = SimpleNamespace(is_connected=False, disconnect=Mock())
        session = SimpleNamespace(storage=None, active_command=False, writer=None,
                                  interrupted=threading.Event(), robot=robot,
                                  start=Mock(side_effect=RuntimeError("synthetic preflight failure")),
                                  stop_once=Mock())
        with patch("fr5_jev.teacher_targets.TeacherSession", return_value=session), \
                patch("fr5_jev.teacher_targets.signal.signal"), \
                patch("sys.argv", ["teacher_targets", "--directory", str(self.root), "--execute"]), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "preflight failure"):
                main()
        session.stop_once.assert_not_called()
        robot.disconnect.assert_not_called()
        self.assertFalse((self.root / "STOP").exists())

    def test_readonly_probe_label_is_explicitly_nonexecuted_and_cli_is_inert(self):
        hold = durable_recording_probe.diagnostic_hold(self.robot.s)
        self.assertFalse(hold["executed"])
        self.assertEqual(hold["target"], [*self.robot.s["joints"], self.robot.s["gripper_position"]])
        self.assertIn("synthetic_readonly", hold["teacher"]["policy"])
        with patch("sys.argv", ["durable_recording_probe"]), redirect_stdout(io.StringIO()) as out:
            durable_recording_probe.main()
        self.assertEqual(json.loads(out.getvalue())["status"], "inert")
        self.assertEqual(self.rpc.calls, [])

    def test_prepared_target_is_synced_before_send_then_attempt_and_completion(self):
        storage = FakeStorage()
        self.session.storage = storage
        self.send()
        prepared = json.loads(storage.syncs[0][0])
        attempted = json.loads(storage.syncs[1][0])
        completed = json.loads(storage.syncs[2][0])
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(prepared["intended_target"], self.target)
        self.assertNotIn("command", prepared)
        self.assertEqual(attempted["status"], "attempted_once")
        self.assertEqual(attempted["command"]["target"], self.target)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual([method for method, _ in self.rpc.calls], ["MoveL"])

    def test_stalled_pre_send_storage_cannot_send_a_delayed_command(self):
        entered, release = threading.Event(), threading.Event()

        def block(*paths):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("synthetic blocked storage")

        def healthy():
            if entered.is_set():
                raise TimeoutError("synthetic stale capture")

        self.session.storage = SimpleNamespace(sync_files=block)
        self.session.capture.assert_healthy.side_effect = healthy
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                self.wait_for(lambda: (self.root / "command-0002.json").exists()
                              and json.loads((self.root / "command-0002.json").read_text())
                              .get("status") == "failed_or_uncertain")
            finally:
                release.set()
            with self.assertRaisesRegex(TimeoutError, "stale capture"):
                future.result(timeout=3)
        self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
        self.assertFalse(any(method == "StopMotion" for method, _ in self.rpc.calls))

    def test_slow_pre_send_storage_keeps_stream_and_hold_callback_running(self):
        capture = self.start_synthetic_stream()
        entered, release = threading.Event(), threading.Event()
        calls = []

        def sync(*paths):
            calls.append(paths)
            if len(calls) == 1:
                entered.set()
                if not release.wait(6):
                    raise TimeoutError("synthetic blocked storage")

        self.session.storage = SimpleNamespace(sync_files=sync)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                before = self.session.writer.add.call_count
                self.wait_for(lambda: self.session.writer.add.call_count >= before + 3)
                capture.assert_healthy()
                self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
                for call in self.session.writer.add.call_args_list[before:]:
                    self.assertFalse(call.args[1]["executed"])
            finally:
                release.set()
            future.result(timeout=4)
        self.assertEqual(sum(method == "MoveL" for method, _ in self.rpc.calls), 1)
        self.assertFalse(any(method == "StopMotion" for method, _ in self.rpc.calls))

    def test_slow_post_send_storage_keeps_stream_and_issued_callback_running(self):
        capture = self.start_synthetic_stream()
        entered, release = threading.Event(), threading.Event()
        calls = []

        def sync(*paths):
            calls.append(paths)
            if len(calls) == 2:
                entered.set()
                if not release.wait(6):
                    raise TimeoutError("synthetic blocked storage")

        self.session.storage = SimpleNamespace(sync_files=sync)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(sum(method == "MoveL" for method, _ in self.rpc.calls), 1)
                before = self.session.writer.add.call_count
                self.wait_for(lambda: self.session.writer.add.call_count >= before + 3)
                capture.assert_healthy()
                for call in self.session.writer.add.call_args_list[before:]:
                    self.assertTrue(call.args[1]["executed"])
                    self.assertEqual(call.args[1]["target"], self.target)
                journal = json.loads((self.root / "target-command-journal.json").read_text())
                self.assertEqual(journal[-1]["target"], self.target)
            finally:
                release.set()
            future.result(timeout=4)
        self.assertEqual(sum(method == "MoveL" for method, _ in self.rpc.calls), 1)
        self.assertFalse(any(method == "StopMotion" for method, _ in self.rpc.calls))

    def test_timeout_cancels_stalled_pre_send_without_late_dispatch(self):
        capture = self.start_synthetic_stream()
        entered, release = threading.Event(), threading.Event()

        def sync(*paths):
            entered.set()
            if not release.wait(6):
                raise TimeoutError("synthetic blocked storage")

        self.session.storage = SimpleNamespace(sync_files=sync)
        self.session.command_timeouts = Mock(return_value=(1., .75))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                before = self.session.writer.add.call_count
                self.wait_for(lambda: self.session.writer.add.call_count >= before + 3)
                capture.assert_healthy()
                self.wait_for(lambda: json.loads((self.root / "command-0002.json").read_text())
                              .get("status") == "failed_or_uncertain")
                self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
            finally:
                release.set()
            with self.assertRaisesRegex(TimeoutError, "Completion feedback timed out"):
                future.result(timeout=4)
        self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
        self.assertFalse(any(method == "StopMotion" for method, _ in self.rpc.calls))
        self.assertEqual(len(self.session.journal), 1)

    def test_timeout_during_stream_ack_wait_never_dispatches_late(self):
        capture = self.start_synthetic_stream()
        entered, release = threading.Event(), threading.Event()

        def wait_for_ack(sequence):
            entered.set()
            if not release.wait(6):
                raise TimeoutError("synthetic blocked frame receipt")

        spool = SimpleNamespace(watermark=lambda: 0, wait_for_ack=wait_for_ack)
        self.session.storage = SimpleNamespace(sync_files=Mock(), frame_spool=spool)
        self.session.command_timeouts = Mock(return_value=(1., .75))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                before = self.session.writer.add.call_count
                self.wait_for(lambda: self.session.writer.add.call_count >= before + 3)
                capture.assert_healthy()
                self.wait_for(lambda: json.loads((self.root / "command-0002.json").read_text())
                              .get("status") == "failed_or_uncertain")
                self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
            finally:
                release.set()
            with self.assertRaisesRegex(TimeoutError, "Completion feedback timed out"):
                future.result(timeout=4)
        self.assertFalse(any(method == "MoveL" for method, _ in self.rpc.calls))
        self.assertEqual(len(self.session.journal), 1)

    def test_journal_snapshot_is_stable_while_persistence_is_delayed(self):
        self.start_synthetic_stream()
        self.session.storage = FakeStorage()
        entered, release = threading.Event(), threading.Event()
        original_save = teacher_targets.save

        def delayed_save(path, value):
            if path.name == "target-command-journal.json":
                entered.set()
                if not release.wait(6):
                    raise TimeoutError("synthetic delayed journal write")
            return original_save(path, value)

        with patch("fr5_jev.teacher_targets.save", side_effect=delayed_save), \
                ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                # A later in-memory label cannot alter the already-snapshotted
                # attempt. This append is synthetic and sends no command.
                later = pool.submit(self.session.append, self.target, "hold", False)
                later.result(timeout=.5)
                self.assertEqual(len(self.session.journal), 3)
            finally:
                release.set()
            future.result(timeout=4)
        journal = json.loads((self.root / "target-command-journal.json").read_text())
        self.assertEqual(len(journal), 2)
        self.assertEqual(journal[-1]["target"], self.target)
        self.assertTrue(journal[-1]["executed"])
        self.assertEqual(sum(method == "MoveL" for method, _ in self.rpc.calls), 1)

    def test_stalled_post_send_storage_stops_once_without_replay(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def block_after_send(*paths):
            calls.append(paths)
            if len(calls) == 2:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("synthetic blocked storage")

        def healthy():
            if entered.is_set():
                raise TimeoutError("synthetic stale capture")

        self.session.storage = SimpleNamespace(sync_files=block_after_send)
        self.session.capture.assert_healthy.side_effect = healthy
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.send)
            try:
                self.assertTrue(entered.wait(2))
                self.wait_for(lambda: (self.root / "STOP").exists())
            finally:
                release.set()
            with self.assertRaisesRegex(TimeoutError, "stale capture"):
                future.result(timeout=3)
        self.assertEqual(sum(method == "MoveL" for method, _ in self.rpc.calls), 1)
        self.assertEqual(sum(method == "StopMotion" for method, _ in self.rpc.calls), 1)
        record = json.loads((self.root / "command-0002.json").read_text())
        self.assertEqual(record["status"], "failed_or_uncertain")
        self.assertEqual(record["command"]["target"], self.target)


if __name__ == "__main__":
    unittest.main()
