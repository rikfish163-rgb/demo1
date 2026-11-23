"""Explicit workspace configuration for opt-in, supervised FR5 control."""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig

from .action_control import MotionLimits
from .configuration_fairino import FairinoFR5Config


@RobotConfig.register_subclass("fairino_fr5_control")
@dataclass(kw_only=True)
class FairinoFR5ControlConfig(FairinoFR5Config):
    # A workspace is never inferred from an observation or a historical preset.
    limits: MotionLimits
    allow_motion: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.allow_motion) is not bool:
            raise ValueError("allow_motion must be an explicit boolean")
        if not isinstance(self.limits, MotionLimits):
            raise ValueError("An explicit MotionLimits instance is required")
        # Recheck mutable limits supplied by the caller before copying them.
        self.limits.__post_init__()
