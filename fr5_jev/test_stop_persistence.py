"""Stop handling with mocked RPC and local /tmp files only; no device calls."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from fr5_jev.teacher_targets import TeacherSession


class StopPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fr5-stop-test-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Bypass construction: the tests exercise only the stop method.
        self.session = TeacherSession.__new__(TeacherSession)
        self.session.root = self.root
        self.session.failed = None
        self.session.rpc = Mock()
        self.session.rpc.call.return_value = 0

    @staticmethod
    def disk_full():
        return OSError(errno.ENOSPC, 'synthetic disk full')

    def test_stop_is_sent_before_stop_marker_failure(self):
        events = []
        original_write = Path.write_text

        def write(path, data, *args, **kwargs):
            events.append(path.name)
            if path.name == 'STOP':
                raise self.disk_full()
            return original_write(path, data, *args, **kwargs)

        self.session.rpc.call.side_effect = lambda *args, **kwargs: events.append('rpc') or 0
        with patch.object(Path, 'write_text', autospec=True, side_effect=write):
            self.session.stop_once('synthetic contact lost', arm_in_flight=True)

        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(events[:2], ['rpc', 'STOP'])
        self.assertEqual(self.session.failed, 'synthetic contact lost')
        self.assertEqual(self.session.stop_result, {'return_code': 0, 'reason': 'synthetic contact lost'})
        self.assertEqual(json.loads((self.root/'stop-result.json').read_text()), self.session.stop_result)
        self.assertEqual([item['file'] for item in self.session.stop_persistence_errors], ['STOP'])
        self.assertIn('synthetic disk full', self.session.stop_persistence_errors[0]['error'])

    def test_both_disk_writes_fail_but_stop_result_stays_in_memory(self):
        with patch.object(Path, 'write_text', autospec=True, side_effect=self.disk_full()) as write:
            self.session.stop_once('synthetic monitor error', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(write.call_count, 2)
        self.assertEqual(self.session.failed, 'synthetic monitor error')
        self.assertEqual(self.session.stop_result, {'return_code': 0, 'reason': 'synthetic monitor error'})
        self.assertEqual([item['file'] for item in self.session.stop_persistence_errors],
                         ['STOP', 'stop-result.json'])

    def test_log_failure_does_not_relabel_a_successful_rpc_as_unconfirmed(self):
        with patch('fr5_jev.teacher_targets.save', side_effect=self.disk_full()) as save:
            self.session.stop_once('synthetic failure', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        save.assert_called_once_with(self.root/'stop-result.json',
                                     {'return_code': 0, 'reason': 'synthetic failure'})
        self.assertEqual(self.session.stop_result['return_code'], 0)
        self.assertNotIn('unconfirmed', self.session.stop_result)
        self.assertEqual((self.root/'STOP').read_text(), 'synthetic failure\n')

    def test_unconfirmed_rpc_and_disk_errors_are_retained_without_retry(self):
        self.session.rpc.call.side_effect = TimeoutError('synthetic missing stop response')
        with patch.object(Path, 'write_text', autospec=True, side_effect=self.disk_full()):
            self.session.stop_once('synthetic original failure', arm_in_flight=True)
            self.session.stop_once('later error', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(self.session.failed, 'synthetic original failure')
        self.assertIn('TimeoutError: synthetic missing stop response', self.session.stop_result['unconfirmed'])
        self.assertEqual(self.session.stop_result['reason'], 'synthetic original failure')
        self.assertEqual([item['file'] for item in self.session.stop_persistence_errors],
                         ['STOP', 'stop-result.json', 'STOP'])

    def test_rpc_interrupt_is_preserved_after_best_effort_logging(self):
        self.session.rpc.call.side_effect = KeyboardInterrupt('synthetic interruption')
        with patch.object(Path, 'write_text', autospec=True, side_effect=self.disk_full()):
            with self.assertRaises(KeyboardInterrupt):
                self.session.stop_once('synthetic original failure', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(self.session.failed, 'synthetic original failure')
        self.assertIn('KeyboardInterrupt', self.session.stop_result['unconfirmed'])
        self.assertEqual([item['file'] for item in self.session.stop_persistence_errors],
                         ['STOP', 'stop-result.json'])

    def test_nonzero_stop_return_is_not_treated_as_a_confirmed_stop(self):
        self.session.rpc.call.return_value = 4
        self.session.stop_once('synthetic failure', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(self.session.stop_result['return_code'], 4)
        self.assertEqual(json.loads((self.root/'stop-result.json').read_text())['return_code'], 4)

    def test_inactive_call_does_not_consume_the_one_stop_attempt(self):
        self.session.stop_once('first failure', arm_in_flight=False)
        self.session.rpc.call.assert_not_called()
        self.session.stop_once('later failure', arm_in_flight=True)
        self.session.stop_once('third failure', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(self.session.failed, 'first failure')
        self.assertEqual(self.session.stop_result['reason'], 'first failure')
        self.assertEqual((self.root/'STOP').read_text(), 'first failure\n')

    def test_disk_full_during_inactive_call_does_not_block_a_later_stop(self):
        with patch.object(Path, 'write_text', autospec=True, side_effect=self.disk_full()):
            self.session.stop_once('first failure', arm_in_flight=False)
            self.session.stop_once('later failure', arm_in_flight=True)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        self.assertEqual(self.session.failed, 'first failure')
        self.assertEqual(self.session.stop_result['return_code'], 0)

    def test_concurrent_stop_calls_issue_only_one_rpc(self):
        entered, release = threading.Event(), threading.Event()

        def blocked_rpc(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('synthetic stop RPC was not released')
            return 0

        self.session.rpc.call.side_effect = blocked_rpc
        with patch.object(Path, 'write_text', autospec=True, return_value=0), \
             patch('fr5_jev.teacher_targets.save') as save, \
             ThreadPoolExecutor(max_workers=5) as pool:
            first = pool.submit(self.session.stop_once, 'first failure', arm_in_flight=True)
            try:
                self.assertTrue(entered.wait(3))
                rest = [pool.submit(self.session.stop_once, 'later failure', arm_in_flight=True)
                        for _ in range(4)]
                for future in rest:
                    future.result(timeout=3)
            finally:
                release.set()
            first.result(timeout=3)
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)
        save.assert_called_once()
        self.assertEqual(self.session.failed, 'first failure')
        self.assertEqual(self.session.stop_result['return_code'], 0)

    def test_stop_does_not_wait_for_guarded_dispatch_lock(self):
        self.session.lock = threading.Lock()
        self.session.lock.acquire()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.session.stop_once, 'synthetic failure', arm_in_flight=True)
                try:
                    future.result(timeout=2)
                finally:
                    self.session.lock.release()
        finally:
            if self.session.lock.locked():
                self.session.lock.release()
        self.session.rpc.call.assert_called_once_with('StopMotion', timeout_s=2)

    def test_idle_failure_retains_first_rejected_telemetry(self):
        self.session.last_read_state={'joint_speeds_deg_s':[.109,0,0,0,0,0], 'motion_done':1}
        self.session.target=[1,2,3,4,5,6,76]
        self.session.phase='lift_full'
        self.session.command_sequence=11
        self.session.stop_once('unexpected speed', arm_in_flight=False)
        self.session.last_read_state['joint_speeds_deg_s'][0]=0
        self.session.stop_once('later failure', arm_in_flight=False)
        saved=json.loads((self.root/'failure-state.json').read_text())
        self.assertEqual(saved['state']['joint_speeds_deg_s'][0],.109)
        self.assertEqual(saved['reason'],'unexpected speed')
        self.assertEqual(saved['phase'],'lift_full')
        self.session.rpc.call.assert_not_called()

    def test_empty_exception_still_sets_a_stop_reason(self):
        self.session.stop_once(RuntimeError(), arm_in_flight=False)
        self.assertEqual(self.session.failed, 'RuntimeError')
        self.session.rpc.call.assert_not_called()
        self.assertEqual((self.root/'STOP').read_text(), 'RuntimeError\n')


if __name__ == '__main__':
    unittest.main()
