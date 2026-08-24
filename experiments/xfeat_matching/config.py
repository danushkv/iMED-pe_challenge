"""XFeat front-end configuration; downstream geometry remains unchanged."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class XFeatConfig:
    max_features: int = 2048
    # Official sparse XFeat MNN matching uses min_cossim=-1 by default: no
    # extra cosine rejection. This avoids introducing a tuned threshold.
    min_cosine_similarity: float = -1.0
    detection_threshold: float | None = None

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if not -1.0 <= self.min_cosine_similarity <= 1.0:
            raise ValueError("min_cosine_similarity must lie in [-1, 1]")
        if self.detection_threshold is not None and self.detection_threshold < 0.0:
            raise ValueError("detection_threshold must be non-negative")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
