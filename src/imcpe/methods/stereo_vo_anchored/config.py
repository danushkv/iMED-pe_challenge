from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnchorEstimationConfig:
    ransac_threshold_normalized: float = 1e-3
    ransac_probability: float = 0.999
    prefer_usac_magsac: bool = True


@dataclass(frozen=True)
class AnchorThresholds:
    """Training-derived confidence gate for cross-camera anchors."""

    min_matches: int
    min_essential_inliers: int
    min_inlier_ratio: float
    min_cheirality_fraction: float
    max_median_sampson_error_normalized: float
    lower_quantile: float
    upper_quantile: float
    num_training_anchors: int

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load_json(cls, path: Path) -> "AnchorThresholds":
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class FusionConfig:
    anchor_interval: int
    alpha: float
    max_correction_deg: float = 10.0
    max_anchor_disagreement_deg: float | None = None

    def __post_init__(self) -> None:
        if self.anchor_interval <= 0:
            raise ValueError("anchor_interval must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.max_correction_deg <= 0.0:
            raise ValueError("max_correction_deg must be positive")

    def save_yaml(self, path: Path, thresholds_path: Path | None = None) -> None:
        """Write simple YAML without adding a PyYAML dependency."""

        lines = [
            f"anchor_interval: {self.anchor_interval}",
            f"alpha: {self.alpha}",
            f"max_correction_deg: {self.max_correction_deg}",
            "max_anchor_disagreement_deg: "
            + ("null" if self.max_anchor_disagreement_deg is None else str(self.max_anchor_disagreement_deg)),
        ]
        if thresholds_path is not None:
            lines.append(f"anchor_thresholds_json: {thresholds_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")

