from __future__ import annotations

from dataclasses import asdict, dataclass


EXPERT_NAMES = ("2a", "2b", "loftr")


@dataclass(frozen=True)
class RouterConfig:
    model: str = "logistic"
    feature_set: str = "r3"
    smoothing_alpha: float = 1.0
    soft_target: bool = True
    random_state: int = 2026

    def __post_init__(self) -> None:
        if self.model not in {"logistic", "histgb"}:
            raise ValueError("model must be logistic or histgb")
        if self.feature_set not in {"r1", "r2", "r3"}:
            raise ValueError("feature_set must be r1, r2, or r3")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FusionConfig:
    lambda_vo: float = 5.0
    absolute_weight_scale: float = 1.0
    vo_weight_scale: float = 1.0
    huber_delta_steps: float = 2.5
    alignment_iterations: int = 8
    alignment_huber_mad: float = 2.5
    optimizer_max_nfev: int = 200

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

