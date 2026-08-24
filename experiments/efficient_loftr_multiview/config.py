from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientLoFTRConfig:
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
class MultiViewConfig:
    confidence_threshold: float = 0.8
    association_tolerance_px: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        if not 0.0 < self.association_tolerance_px <= 3.0:
            raise ValueError("association_tolerance_px must be in (0,3]")

