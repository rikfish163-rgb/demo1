"""Configuration for the existing local Gemini RGB-D HTTP bridge."""

from dataclasses import dataclass
import math
from urllib.parse import urlsplit

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("gemini_bridge")
@dataclass(kw_only=True)
class GeminiBridgeConfig(CameraConfig):
    width: int = 640
    height: int = 360
    fps: int = 5
    url: str = "http://127.0.0.1:8765/snapshot.npz"
    expected_serial: str = "CP1E54200015"
    timeout_s: float = 2.0
    max_age_s: float = 1.0

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("timeout_s", "max_age_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.expected_serial, str) or not self.expected_serial:
            raise ValueError("expected_serial must be a nonempty string")
        parsed = urlsplit(self.url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("/snapshot.npz", "/snapshot_fast.npz")):
            raise ValueError("url must be a loopback HTTP /snapshot.npz or /snapshot_fast.npz endpoint")
