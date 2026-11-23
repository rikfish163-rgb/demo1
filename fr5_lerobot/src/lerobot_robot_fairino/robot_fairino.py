"""FR5 telemetry adapter. No command transport is constructed anywhere here."""

import time

import numpy as np

from lerobot.robots.robot import Robot
from lerobot_camera_gemini import GeminiBridgeCamera
from .configuration_fairino import FairinoFR5Config
from .state_stream import FR5StateStream

STATE_NAMES = [f"joint_{i}.deg" for i in range(1, 7)] + ["gripper.percent"]
POSE_NAMES = ["x.mm", "y.mm", "z.mm", "rx.deg", "ry.deg", "rz.deg"]
STATUS_NAMES = [
    "tool", "user", "program_state", "robot_state", "main_code", "sub_code",
    "EmergencyStop", "collisionState", "safety_stop0_state", "safety_stop1_state",
    "rbtEnableState", "motion_done", "mc_queue_len", "gripper_fault",
    "gripper_active", "gripper_motiondone", "gripper_fault_id",
    "gripper_speed", "gripper_current",
]


class FairinoFR5(Robot):
    config_class = FairinoFR5Config
    name = "fairino_fr5_readonly"

    def __init__(self, config: FairinoFR5Config):
        super().__init__(config)
        self.config = config
        self.cameras = {"front": GeminiBridgeCamera(config.camera)}
        self.stream = FR5StateStream(
            host=config.host, port=config.state_port, timeout_s=config.state_timeout_s,
        )
        self._connected = False
        self.last_sample = None
        self._pairing_start = None

    @property
    def observation_features(self):
        return {
            **dict.fromkeys(STATE_NAMES, float),
            "front": (self.config.camera.height, self.config.camera.width, 3),
        }

    @property
    def action_features(self):
        # An observation-only dataset must not advertise or fabricate action labels.
        return {}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        # Factory joint readings need no LeRobot motor calibration. This says
        # nothing about hand-eye calibration, TCP geometry or grasp readiness.
        return True

    def calibrate(self):
        """No motor writes; hand-eye and tool calibration are separate tasks."""

    def configure(self):
        """No robot configuration changes are needed to receive telemetry."""

    def connect(self, calibrate: bool = False):
        if self._connected:
            raise RuntimeError("FR5 observer is already connected")
        # A new stream has no robot history from the previous connection.
        self.last_sample = None
        try:
            self.stream.connect()
            self.cameras["front"].connect()
            state = self.stream.latest(max_age_s=self.config.state_max_age_s)
            # Both sources use timestamps from this host. At startup, ignore
            # camera frames older than the state history we just connected to.
            self._pairing_start = {'host_unix': state.get('timestamp'),
                                   'host_monotonic': state.get('received_monotonic')}
            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def get_sample(self):
        from lerobot_camera_gemini.timing import camera_clock, MONOTONIC
        if not self._connected:
            raise RuntimeError("FR5 observer is not connected")
        # At startup the bridge can serve a frame captured before this fresh
        # robot-state connection has any history. Discard only unmatched
        # startup frames and wait for a new one; never relax the skew bound or
        # hide a synchronization failure once recording has begun.
        first_sample = self.last_sample is None
        deadline = time.monotonic() + self.config.state_timeout_s if first_sample else None
        while True:
            if first_sample:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No synchronized camera/FR5 frame before startup timeout")
                bundle = self.cameras["front"].read_bundle(timeout_ms=remaining * 1000, require_new=True)
            else:
                bundle = self.cameras["front"].read_bundle(require_new=True)
            capture_time,basis = camera_clock(bundle['metadata'])
            # Check that the stream is still healthy before using its history.
            self.stream.latest(max_age_s=self.config.state_max_age_s)
            started_at = self._pairing_start.get(basis) if first_sample and self._pairing_start else None
            if started_at is not None and capture_time < started_at:
                continue
            remaining = deadline - time.monotonic() if first_sample else self.config.state_timeout_s
            if first_sample and remaining <= 0:
                raise TimeoutError("No synchronized camera/FR5 frame before startup timeout")
            try:
                state = self.stream.nearest(
                    capture_time, max_skew_s=self.config.max_sync_skew_s,
                    timeout_s=remaining,
                    **({'clock_basis':basis} if basis == MONOTONIC else {}),
                )
                break
            except TimeoutError:
                if not first_sample or time.monotonic() >= deadline:
                    raise
        values = np.asarray([*state["joints"], state["gripper_position"]], dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Invalid FR5 joint/gripper observation")
        if not 0 <= values[-1] <= 100:
            raise ValueError("Gripper percentage is outside 0..100")
        skew = float(state['received_monotonic' if basis == MONOTONIC else 'timestamp']) - capture_time
        if abs(skew) > self.config.max_sync_skew_s:
            raise TimeoutError("Camera/robot host timestamp skew exceeds limit")
        observation = dict(zip(STATE_NAMES, values.tolist(), strict=True))
        observation["front"] = bundle["color_rgb"]
        result = {
            "observation": observation, "robot_state": state,
            "camera": bundle, "sync_skew_s": skew, "sync_clock":basis,
            "read_only": True, "action_available": False,
        }
        self.last_sample = result
        return result

    def get_observation(self):
        return self.get_sample()["observation"]

    def send_action(self, action):
        raise PermissionError(
            "This adapter only records observations; robot and gripper commands are disabled."
        )

    def disconnect(self):
        try:
            self.cameras["front"].disconnect()
        finally:
            self.stream.disconnect()
            self._connected = False
            self._pairing_start = None

