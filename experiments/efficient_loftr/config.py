from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientLoFTRConfig:
    """Inference-only configuration for the official full outdoor model."""

    model_type: str = "full"
    precision: str = "fp32"
    resize_divisor: int = 32

    def __post_init__(self) -> None:
        if self.model_type not in {"full", "opt"}:
            raise ValueError("model_type must be 'full' or 'opt'")
        if self.precision not in {"fp32", "mp", "fp16"}:
            raise ValueError("precision must be fp32, mp, or fp16")
        if self.resize_divisor <= 0:
            raise ValueError("resize_divisor must be positive")


@dataclass(frozen=True)
class AssociationConfig:
    tolerance_px: float = 2.0
    confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.tolerance_px <= 0.0:
            raise ValueError("tolerance_px must be positive")
        if self.confidence_threshold is not None and not (
            0.0 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be in [0,1]")

