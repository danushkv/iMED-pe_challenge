from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RoMa2AConfig:
    """Inference-only settings for the Point-6 experiment.

    The certainty and cycle thresholds are deliberately optional.  Their
    numerical values must be selected from TRAIN diagnostics, never TEST/GT.
    """

    max_source_points: int = 2048
    certainty_threshold: float | None = None
    cycle_threshold_px: float | None = None
    device: str = "cuda"
    model_name: str = "roma_outdoor"

    def validate(self) -> None:
        if self.max_source_points <= 0:
            raise ValueError("max_source_points must be positive")
        if self.certainty_threshold is not None and not (
            0.0 <= self.certainty_threshold <= 1.0
        ):
            raise ValueError("certainty_threshold must lie in [0, 1]")
        if self.cycle_threshold_px is not None and self.cycle_threshold_px <= 0:
            raise ValueError("cycle_threshold_px must be positive")
        if self.model_name != "roma_outdoor":
            raise ValueError("Point 6 initially supports full roma_outdoor only")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
