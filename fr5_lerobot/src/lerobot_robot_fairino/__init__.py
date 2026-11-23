"""Register FR5 and Gemini devices; importing never connects hardware."""

from lerobot_camera_gemini import GeminiBridgeConfig, GeminiBridgeCamera
from .configuration_fairino import FairinoFR5Config
from .robot_fairino import FairinoFR5
from .configuration_control import FairinoFR5ControlConfig
from .robot_control import FairinoFR5Control
from .action_control import MotionLimits

__all__ = ["FairinoFR5Config", "FairinoFR5", "FairinoFR5ControlConfig", "FairinoFR5Control",
           "MotionLimits", "GeminiBridgeConfig", "GeminiBridgeCamera"]

