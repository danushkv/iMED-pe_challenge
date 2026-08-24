from __future__ import annotations

from dataclasses import dataclass

from imcpe.methods.cross_stereo_pnp.config import (
    Method2Config,
    SessionCalibrationConfig,
)


@dataclass(frozen=True)
class ReverseCalibrationConfig(SessionCalibrationConfig):
    """E2 calibration settings, intentionally identical to Method 2A."""


@dataclass(frozen=True)
class ReverseMethod2Config(Method2Config):
    """Method 2B settings, intentionally identical to Method 2A."""
