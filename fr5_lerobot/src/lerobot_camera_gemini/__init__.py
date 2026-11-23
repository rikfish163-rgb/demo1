"""Gemini bridge plugin for LeRobot; import registers configuration only."""

from .configuration_gemini import GeminiBridgeConfig
from .camera_gemini import GeminiBridgeCamera

# LeRobot 0.4.4's factory strips "Config" to find the implementation class.
GeminiBridge = GeminiBridgeCamera

__all__ = ["GeminiBridgeConfig", "GeminiBridgeCamera", "GeminiBridge"]
