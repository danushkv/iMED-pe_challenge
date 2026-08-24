from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method4Config:
    mode: str = "4a1"
    confidence_version: str = "v1"
    lambda_vo: float = 1.0
    absolute_weight_scale: float = 1.0
    vo_weight_scale: float = 1.0
    huber_delta_steps: float = 2.5
    alignment_iterations: int = 8
    alignment_huber_mad: float = 2.5
    optimizer_max_nfev: int = 200

    def __post_init__(self) -> None:
        if self.mode not in {"4a0", "4a1"}:
            raise ValueError("mode must be 4a0 or 4a1")
        if self.confidence_version not in {"v0", "v1", "v2"}:
            raise ValueError("confidence_version must be v0, v1, or v2")
        if self.lambda_vo < 0 or self.absolute_weight_scale < 0:
            raise ValueError("weights must be non-negative")
        if self.vo_weight_scale < 0 or self.huber_delta_steps <= 0:
            raise ValueError("invalid VO/Huber configuration")
