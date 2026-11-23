"""LeRobot adapter for explicitly armed, supervised FR5 position commands.

Connecting only starts observations and a local watchdog. It never enables,
arms, homes or moves a robot. The operator interface must arm and maintain the
deadman heartbeat explicitly, while an observation loop refreshes last_sample.
"""

from copy import deepcopy
import math
from numbers import Real
import threading
import time

from .action_control import ControlError, FR5ActionController
from .configuration_control import FairinoFR5ControlConfig
from .robot_fairino import FairinoFR5, STATE_NAMES
from lerobot_camera_gemini.timing import camera_clock, MONOTONIC


class FairinoFR5Control(FairinoFR5):
    config_class = FairinoFR5ControlConfig
    name = "fairino_fr5_control"
    WATCHDOG_INTERVAL_S = 0.05

    def __init__(self, config: FairinoFR5ControlConfig):
        super().__init__(config)
        self.controller = FR5ActionController(
            self.stream, self.camera_fresh, deepcopy(config.limits),
            host=config.host, allow_motion=config.allow_motion,
        )
        self.last_action_result = None
        self._lifecycle_lock = threading.RLock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None
        self._watchdog_error = None
        self._disconnecting = False
        self._epoch = 0

    @property
    def action_features(self):
        return dict.fromkeys(STATE_NAMES, float)

    @property
    def watchdog_error(self):
        return self._watchdog_error

    def camera_fresh(self):
        """Check the latest observed capture; never acquire or fabricate a frame."""
        if not self.is_connected or self.last_sample is None:
            return False
        try:
            meta = self.last_sample["camera"]["metadata"]
            if meta["source_camera_serial"] != self.config.camera.expected_serial:
                return False
            captured, basis = camera_clock(meta)
            age = (time.monotonic() if basis == MONOTONIC else time.time()) - captured
            if age < -0.1 or age > self.config.camera.max_age_s:
                return False
            # Monotonic receipt age also rejects old frames after a wall-clock rollback.
            received = meta["received_unix"]
            received_mono = meta["received_monotonic"]
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                   for v in (received, received_mono)):
                return False
            elapsed = time.monotonic() - float(received_mono)
            source_age_at_receipt = (float(received_mono) if basis == MONOTONIC else float(received)) - captured
            return (elapsed >= 0 and source_age_at_receipt >= -0.1
                    and source_age_at_receipt + elapsed <= self.config.camera.max_age_s)
        except (KeyError, TypeError, ValueError):
            return False

    def connect(self, calibrate: bool = False):
        with self._lifecycle_lock:
            if self._disconnecting:
                raise RuntimeError("FR5 is disconnecting")
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                raise RuntimeError("FR5 watchdog is already running")
            self.last_sample = None
            self.last_action_result = None
            super().connect(calibrate=calibrate)
            self._watchdog_error = None
            self._watchdog_stop.clear()
            self._epoch += 1
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="fr5-control-watchdog", daemon=True,
            )
            try:
                self._watchdog_thread.start()
            except BaseException:
                self.disconnect()
                raise

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(self.WATCHDOG_INTERVAL_S):
            try:
                self.controller.watchdog()
            except BaseException as exc:
                self._watchdog_error = f"{type(exc).__name__}: {exc}"
                try:
                    self.controller.stop(f"Independent watchdog failed: {self._watchdog_error}")
                except BaseException as stop_exc:
                    self._watchdog_error += f"; stop failed: {type(stop_exc).__name__}: {stop_exc}"
                self._watchdog_stop.set()

    def _require_control_session(self):
        if not self.is_connected or self._disconnecting:
            raise ControlError("FR5 control adapter is not connected")
        if not self.config.allow_motion or not self.controller.allow_motion:
            raise ControlError("allow_motion=False: motion is disabled")
        if (self._watchdog_error or self._watchdog_stop.is_set()
                or self._watchdog_thread is None or not self._watchdog_thread.is_alive()):
            raise ControlError(f"Independent watchdog is unavailable: {self._watchdog_error or 'not running'}")

    def arm(self, *, operator_confirmed=False):
        with self._lifecycle_lock:
            self._require_control_session()
            epoch = self._epoch
        # No lifecycle lock is held during RPC, so disconnect/stop remain available.
        self.controller.arm(operator_confirmed=operator_confirmed)
        with self._lifecycle_lock:
            cancelled = epoch != self._epoch or self._disconnecting or not self.is_connected
            try:
                self._require_control_session()
            except ControlError:
                cancelled = True
        if cancelled:
            self.controller.stop("Arming cancelled by stop, disconnect or watchdog failure")
            raise ControlError("Arming cancelled by stop, disconnect or watchdog failure")

    def heartbeat(self, held: bool):
        with self._lifecycle_lock:
            self._require_control_session()
            self.controller.heartbeat(held)

    def stop(self, reason="Operator stop"):
        # Explicit stop also cancels pending arming; an unarmed controller sends no RPC.
        with self._lifecycle_lock:
            self._epoch += 1
        self.controller.stop(reason)

    def send_action(self, action):
        if not isinstance(action, dict) or set(action) != set(STATE_NAMES):
            raise ValueError("action must contain exactly the seven STATE_NAMES targets")
        target = []
        for name in STATE_NAMES:
            value = action[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Action {name} must be a finite numeric target")
            target.append(float(value))
        with self._lifecycle_lock:
            self._require_control_session()
        result = self.controller.execute(target)
        # Only FR5ActionController owns the command journal. A hold/no-op result
        # does not add a fabricated executed action, nor does get_observation().
        self.last_action_result = deepcopy(result)
        return dict(zip(STATE_NAMES, target, strict=True))

    def disconnect(self):
        with self._lifecycle_lock:
            if self._disconnecting:
                return
            self._disconnecting = True
            self._epoch += 1
            self._watchdog_stop.set()
            thread = self._watchdog_thread
        error = None
        try:
            if self.controller.armed or self.controller.busy:
                try:
                    self.controller.stop("Control adapter disconnected")
                except BaseException as exc:
                    error = exc
            try:
                super().disconnect()
            except BaseException as exc:
                error = error or exc
        finally:
            if thread is not None and thread is not threading.current_thread() and thread.ident is not None:
                thread.join(timeout=2.5)
                if thread.is_alive():
                    error = error or RuntimeError("FR5 watchdog did not exit after disconnect")
            with self._lifecycle_lock:
                self.last_sample = None
                self._disconnecting = False
        if error is not None:
            raise error
