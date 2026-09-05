from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method4CConfig:
    mode: str = "4c1"
    weighting: str = "v1"
    lambda_vo: float = 5.0
    absolute_weight_scale: float = 1.0
    vo_weight_scale: float = 1.0
    huber_delta_steps: float = 2.5
    alignment_iterations: int = 8
    alignment_huber_mad: float = 2.5
    optimizer_max_nfev: int = 200
    weiszfeld_iterations: int = 128
    weiszfeld_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.mode not in {"4c0", "4c1"}:
            raise ValueError("mode must be 4c0 or 4c1")
        if self.weighting not in {"v0", "v1", "method4a-v1"}:
            raise ValueError("weighting must be v0, v1, or method4a-v1")
        if self.lambda_vo < 0 or self.absolute_weight_scale < 0:
            raise ValueError("absolute/VO weights must be non-negative")
        if self.vo_weight_scale < 0 or self.huber_delta_steps <= 0:
            raise ValueError("invalid VO/Huber configuration")
        if self.alignment_iterations <= 0 or self.alignment_huber_mad <= 0:
            raise ValueError("invalid alignment configuration")
        if self.optimizer_max_nfev <= 0:
            raise ValueError("optimizer_max_nfev must be positive")
        if self.weiszfeld_iterations <= 0 or self.weiszfeld_tolerance <= 0:
            raise ValueError("invalid Weiszfeld configuration")

