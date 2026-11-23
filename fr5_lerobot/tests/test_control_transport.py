"""Offline transport tests. All HTTP operations are replaced with mocks."""

import errno
from contextlib import contextmanager
import http.client
import threading
import unittest
from unittest.mock import Mock, patch

from lerobot_robot_fairino.control_transport import ALLOWED_METHODS, FR5RPC, NoRetryTransport


MODULE = "lerobot_robot_fairino.control_transport"


class NoRetryTransportTests(unittest.TestCase):
    def test_remote_disconnect_is_not_retried(self):
        transport = NoRetryTransport(3)
        error = http.client.RemoteDisconnected("response lost")
        with patch.object(transport, "single_request", side_effect=error) as request:
            with self.assertRaises(http.client.RemoteDisconnected) as caught:
                transport.request("robot", "/RPC2", b"motion")
        self.assertIs(caught.exception, error)
        request.assert_called_once_with("robot", "/RPC2", b"motion", False)

    def test_connection_reset_is_not_retried(self):
        transport = NoRetryTransport(3)
        error = OSError(errno.ECONNRESET, "connection reset")
        with patch.object(transport, "single_request", side_effect=error) as request:
            with self.assertRaises(OSError) as caught:
                transport.request("robot", "/RPC2", b"motion", True)
        self.assertIs(caught.exception, error)
        request.assert_called_once_with("robot", "/RPC2", b"motion", True)

    def test_request_returns_raw_result(self):
        transport = NoRetryTransport(3)
        result = (37,)
        with patch.object(transport, "single_request", return_value=result) as request:
            self.assertIs(transport.request("robot", "/RPC2", b"read"), result)
        request.assert_called_once()

    def test_timeout_applied_without_opening_connection(self):
        connection = Mock()
        with patch("xmlrpc.client.Transport.make_connection", return_value=connection):
            transport = NoRetryTransport(1.25)
            self.assertIs(transport.make_connection("robot"), connection)
        self.assertEqual(connection.timeout, .25)  # Short connection/write phase.

    def test_guard_covers_actual_write_and_releases_before_response_timeout(self):
        lock = threading.Lock()
        connection = Mock()
        events = []

        @contextmanager
        def guard():
            with lock:
                events.append("gate")
                yield
                events.append("write finished")

        def write(body):
            self.assertTrue(lock.locked())
            self.assertEqual(connection.timeout, .2)
            self.assertEqual(body, b"synthetic")
            events.append("wire")

        connection.endheaders.side_effect = write
        transport = NoRetryTransport(3, before_send=guard, write_timeout_s=.2)
        transport.send_content(connection, b"synthetic")
        self.assertEqual(events, ["gate", "wire", "write finished"])
        self.assertFalse(lock.locked())
        self.assertEqual(connection.timeout, 3)
        self.assertEqual(connection.sock.settimeout.call_args_list[-1].args, (3,))

    def test_cancelled_guard_never_writes_headers_or_body(self):
        @contextmanager
        def cancelled():
            raise RuntimeError("stopped before write")
            yield  # pragma: no cover -- defines the context-manager protocol.

        connection = Mock()
        transport = NoRetryTransport(3, before_send=cancelled)
        with self.assertRaisesRegex(RuntimeError, "stopped before write"):
            transport.send_content(connection, b"synthetic")
        connection.putheader.assert_not_called()
        connection.endheaders.assert_not_called()

    def test_stop_during_connection_setup_cancels_before_http_write(self):
        connecting, release_connection = threading.Event(), threading.Event()
        cancelled = threading.Event()
        connection = Mock()
        connection.sock = None
        errors = []

        def connect():
            connecting.set()
            if not release_connection.wait(2):
                raise AssertionError("Test failed to release fake connection")
            connection.sock = Mock()

        @contextmanager
        def guard():
            if cancelled.is_set():
                raise RuntimeError("cancelled during connection")
            yield

        connection.connect.side_effect = connect
        transport = NoRetryTransport(3, before_send=guard)

        def worker():
            try:
                transport.send_content(connection, b"synthetic")
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(connecting.wait(1))
            cancelled.set()
        finally:
            release_connection.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        connection.endheaders.assert_not_called()
        self.assertEqual(len(errors), 1)
        self.assertIn("cancelled during connection", str(errors[0]))

    def test_blocking_response_does_not_hold_send_guard_and_stop_can_proceed(self):
        lock = threading.Lock()
        in_response, release_response, stop_finished = threading.Event(), threading.Event(), threading.Event()
        connection, response = Mock(), Mock()
        response.status = 200
        errors, results = [], []

        @contextmanager
        def guard():
            with lock:
                yield

        def getresponse():
            in_response.set()
            if not release_response.wait(2):
                raise AssertionError("Test failed to release fake response")
            return response

        connection.getresponse.side_effect = getresponse
        transport = NoRetryTransport(2, before_send=guard)

        def send_request(*args, **kwargs):
            transport.send_content(connection, b"synthetic")
            return connection

        def worker():
            try:
                results.append(transport.request("offline", "/RPC2", b"synthetic"))
            except BaseException as exc:
                errors.append(exc)

        with patch.object(transport, "send_request", side_effect=send_request), \
             patch.object(transport, "parse_response", return_value=0):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                self.assertTrue(in_response.wait(1))
                with lock:
                    stop_finished.set()
                self.assertTrue(stop_finished.is_set())
                self.assertFalse(release_response.is_set())
            finally:
                release_response.set()
                thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [0])
        connection.endheaders.assert_called_once()

    def test_write_exception_releases_guard_and_is_never_retried(self):
        lock = threading.Lock()

        @contextmanager
        def guard():
            with lock:
                yield

        transport = NoRetryTransport(3, before_send=guard)
        connection = Mock()
        connection.endheaders.side_effect = TimeoutError("partial write")

        def send_request(*args, **kwargs):
            transport.send_content(connection, b"synthetic")
            return connection

        with patch.object(transport, "send_request", side_effect=send_request) as send:
            with self.assertRaisesRegex(TimeoutError, "partial write"):
                transport.request("offline", "/RPC2", b"synthetic")
        self.assertFalse(lock.locked())
        send.assert_called_once()
        connection.endheaders.assert_called_once()
        connection.getresponse.assert_not_called()


class FR5RPCTests(unittest.TestCase):
    def test_construction_does_not_create_proxy_or_transport(self):
        with patch(f"{MODULE}.NoRetryTransport") as transport, patch(
            f"{MODULE}.xmlrpc.client.ServerProxy"
        ) as proxy:
            rpc = FR5RPC()
        self.assertEqual(rpc.url, "http://192.168.58.2:20003")
        transport.assert_not_called()
        proxy.assert_not_called()

    def test_method_whitelist_is_exact(self):
        expected = {
            "GetInverseKinRef", "GetForwardKin", "GetJointSoftLimitDeg",
            "GetGripperConfig", "GetGripperMotionDone", "MoveJ", "MoveL", "MoveGripper", "StopMotion",
        }
        self.assertEqual(ALLOWED_METHODS, expected)
        with patch(f"{MODULE}.NoRetryTransport") as transport:
            for method in ["RobotEnable", "SetSpeed", "ServoJ", "moveL", "system.listMethods", "MoveL ", None, []]:
                with self.subTest(method=method), self.assertRaises(ValueError):
                    FR5RPC().call(method)
        transport.assert_not_called()

    def test_each_call_has_new_proxy_and_closed_transport(self):
        transport_a, transport_b = Mock(), Mock()
        proxy_a, proxy_b = Mock(), Mock()
        result_a, result_b = [0, 1, 2, 3, 4, 5, 6], 17
        proxy_a.GetForwardKin.return_value = result_a
        proxy_b.MoveGripper.return_value = result_b
        with patch(f"{MODULE}.NoRetryTransport", side_effect=[transport_a, transport_b]) as factory, patch(
            f"{MODULE}.xmlrpc.client.ServerProxy", side_effect=[proxy_a, proxy_b]
        ) as proxies:
            rpc = FR5RPC()
            self.assertIs(rpc.call("GetForwardKin", [0.0] * 6, timeout_s=1.5), result_a)
            self.assertIs(rpc.call("MoveGripper", 1, 15, 20, 10, 10000, 1, 0, 0.0, 0, 0), result_b)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(proxies.call_count, 2)
        self.assertIs(proxies.call_args_list[0].kwargs["transport"], transport_a)
        self.assertIs(proxies.call_args_list[1].kwargs["transport"], transport_b)
        factory.assert_any_call(1.5, before_send=None, write_timeout_s=.25)
        factory.assert_any_call(3.0, before_send=None, write_timeout_s=.25)
        proxy_a.GetForwardKin.assert_called_once_with([0.0] * 6)
        proxy_b.MoveGripper.assert_called_once_with(1, 15, 20, 10, 10000, 1, 0, 0.0, 0, 0)
        transport_a.close.assert_called_once_with()
        transport_b.close.assert_called_once_with()

    def test_exception_closes_transport_and_propagates_unchanged(self):
        for error in [http.client.RemoteDisconnected("lost"), OSError(errno.ECONNRESET, "reset")]:
            with self.subTest(error=error):
                transport, proxy = Mock(), Mock()
                proxy.MoveL.side_effect = error
                with patch(f"{MODULE}.NoRetryTransport", return_value=transport), patch(
                    f"{MODULE}.xmlrpc.client.ServerProxy", return_value=proxy
                ) as factory:
                    with self.assertRaises(type(error)) as caught:
                        FR5RPC().call("MoveL", [1.0] * 6)
                self.assertIs(caught.exception, error)
                factory.assert_called_once()
                proxy.MoveL.assert_called_once_with([1.0] * 6)
                transport.close.assert_called_once_with()

    def test_proxy_initialization_failure_still_closes_transport(self):
        transport = Mock()
        with patch(f"{MODULE}.NoRetryTransport", return_value=transport), patch(
            f"{MODULE}.xmlrpc.client.ServerProxy", side_effect=RuntimeError("construction failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "construction failed"):
                FR5RPC().call("StopMotion")
        transport.close.assert_called_once_with()

    def test_invalid_timeout_prevents_network_setup(self):
        with patch(f"{MODULE}.NoRetryTransport") as transport:
            for value in [0, -1, float("nan"), float("inf"), True, "3", None]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    FR5RPC().call("StopMotion", timeout_s=value)
        transport.assert_not_called()

    def test_guard_and_write_timeout_are_forwarded_and_validated(self):
        guard = Mock()
        transport, proxy = Mock(), Mock()
        proxy.MoveL.return_value = 0
        with patch(f"{MODULE}.NoRetryTransport", return_value=transport) as factory, \
             patch(f"{MODULE}.xmlrpc.client.ServerProxy", return_value=proxy):
            FR5RPC().call("MoveL", [0.] * 6, timeout_s=10, before_send=guard, write_timeout_s=.1)
        factory.assert_called_once_with(10., before_send=guard, write_timeout_s=.1)
        with patch(f"{MODULE}.NoRetryTransport") as factory:
            for value in (0, -1, True, float("nan"), 2):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    FR5RPC().call("MoveL", write_timeout_s=value)
            with self.assertRaises(ValueError):
                FR5RPC().call("MoveL", before_send=123)
        factory.assert_not_called()

    def test_invalid_endpoint_is_rejected(self):
        for host in ["", " robot", "robot/path", "http://robot", "user@robot", "robot?query", None]:
            with self.subTest(host=host), self.assertRaises(ValueError):
                FR5RPC(host=host)
        for port in [0, 65536, -1, 3.5, True, "20003"]:
            with self.subTest(port=port), self.assertRaises(ValueError):
                FR5RPC(port=port)
        self.assertEqual(FR5RPC(host="::1").url, "http://[::1]:20003")


if __name__ == "__main__":
    unittest.main()
