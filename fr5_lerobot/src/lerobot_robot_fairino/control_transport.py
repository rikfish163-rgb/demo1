"""Explicit, single-attempt FR5 XML-RPC calls; construction never connects.

This module is only a transport. The caller must validate motion limits,
controller readiness, command ownership, and completion. A timeout or dropped
response means the outcome is unknown; it does not authorize repeating motion.

Raw wire signatures verified against ``hardware/sdk/Robot.py``:
  MoveL(joints, pose, tool, user, vel, acc, ovl, blendR, blendMode,
        exaxis_pos, search, offset_flag, offset_pos)  # 13 arguments
  MoveGripper(index, pos, vel, force, maxtime, block, type,
              rotNum, rotVel, rotTorque)             # 10 arguments
These differ from the public Python SDK MoveL wrapper's argument order.
The protocol source is ``hardware/sdk/Robot.py``; this transport sends the
raw XML-RPC arguments directly without importing the SDK.
Official parameter references (3.8.4 path displays manual 3.8.4.1):
https://fairino-doc-zhs.readthedocs.io/3.8.4/SDKManual/PythonRobotMovement.html
https://fairino-doc-zhs.readthedocs.io/3.8.4/SDKManual/PythonRobotPeripherals.html
"""

from __future__ import annotations

from contextlib import nullcontext
import ipaddress
import math
import numbers
import re
import xmlrpc.client


ALLOWED_METHODS = frozenset(
    {
        "GetInverseKinRef",
        "GetForwardKin",
        "GetJointSoftLimitDeg",
        "GetGripperConfig",
        "GetGripperMotionDone",
        "MoveJ",
        "MoveL",
        "MoveGripper",
        "StopMotion",
    }
)


class NoRetryTransport(xmlrpc.client.Transport):
    """Disable the standard library's implicit retry after connection loss."""

    def __init__(self, timeout_s: float, *, before_send=None, write_timeout_s: float = 0.25):
        super().__init__(use_builtin_types=True)
        self.timeout_s = timeout_s
        self.before_send = before_send
        self.write_timeout_s = min(timeout_s, write_timeout_s)

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.write_timeout_s
        return connection

    def send_content(self, connection, request_body):
        """Guard the first actual HTTP write, never the blocking response read.

        HTTPConnection buffers putrequest/putheader calls. Establish TCP before
        entering the guard, so a stop during connection setup cancels the later
        HTTP write. endheaders(body) is the first application-data send. A short
        write timeout bounds the time the cancellation lock may be held. The
        response timeout is restored only after releasing it.
        """
        connection.timeout = self.write_timeout_s
        if connection.sock is None:
            connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(self.write_timeout_s)
        guard = self.before_send() if self.before_send is not None else nullcontext()
        with guard:
            super().send_content(connection, request_body)
        connection.timeout = self.timeout_s
        if connection.sock is not None:
            connection.sock.settimeout(self.timeout_s)

    def request(self, host, handler, request_body, verbose=False):
        # Transport.request normally retries RemoteDisconnected and some
        # OSErrors. A response can be lost after the controller accepted motion.
        return self.single_request(host, handler, request_body, verbose)


class FR5RPC:
    """Make one whitelisted call over a new connection and always close it."""

    def __init__(self, host: str = "192.168.58.2", port: int = 20003):
        if not isinstance(host, str) or not host or host != host.strip():
            raise ValueError("host must be a hostname or IP address")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host):
                raise ValueError("host must not contain a URL, credentials, or path") from None
            authority_host = host
        else:
            authority_host = f"[{host}]" if address.version == 6 else host
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer from 1 to 65535")
        self.host = host
        self.port = port
        self.url = f"http://{authority_host}:{port}"

    def call(self, method: str, *args, timeout_s: float = 3, before_send=None,
             write_timeout_s: float = 0.25):
        if not isinstance(method, str) or method not in ALLOWED_METHODS:
            raise ValueError(f"XML-RPC method is not permitted: {method!r}")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, numbers.Real)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive finite number")
        if (isinstance(write_timeout_s, bool) or not isinstance(write_timeout_s, numbers.Real)
                or not math.isfinite(write_timeout_s) or not 0 < write_timeout_s <= 1):
            raise ValueError("write_timeout_s must be positive, finite and <= 1 second")
        if before_send is not None and not callable(before_send):
            raise ValueError("before_send must be a context-manager factory")

        # Neither transport nor proxy is reused. A stop call can therefore use
        # its own HTTP connection while a blocking MoveL is still outstanding.
        transport = NoRetryTransport(float(timeout_s), before_send=before_send,
                                     write_timeout_s=float(write_timeout_s))
        try:
            proxy = xmlrpc.client.ServerProxy(
                self.url,
                transport=transport,
                allow_none=False,
                use_builtin_types=True,
            )
            return getattr(proxy, method)(*args)
        finally:
            # Do not call proxy.close(): arbitrary attribute access on an XML-RPC
            # proxy can become a remote method. We own the supplied transport.
            transport.close()
