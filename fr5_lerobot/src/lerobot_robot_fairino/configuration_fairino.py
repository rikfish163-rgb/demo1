from dataclasses import dataclass, field
import math

from lerobot.robots.config import RobotConfig
from lerobot_camera_gemini import GeminiBridgeConfig


@RobotConfig.register_subclass("fairino_fr5_readonly")
@dataclass(kw_only=True)
class FairinoFR5Config(RobotConfig):
    host: str = "192.168.58.2"
    state_port: int = 20004
    state_timeout_s: float = 2.0
    state_max_age_s: float = 0.5
    max_sync_skew_s: float = 0.1
    camera: GeminiBridgeConfig = field(default_factory=GeminiBridgeConfig)

    def __post_init__(self):
        super().__post_init__()
        if not self.host or not 1 <= self.state_port <= 65535:
            raise ValueError("A robot host and valid state port are required")
        for key in ("state_timeout_s", "state_max_age_s", "max_sync_skew_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")

