from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SourcePoints:
    points_px: np.ndarray
    scores: np.ndarray
    image_shape_hw: tuple[int, int]
    extraction_seconds: float


class ALIKEDSourceDetector:
    """ALIKED detector only; LightGlue is intentionally never constructed."""

    def __init__(self, max_num_keypoints: int = 2048, device: str = "cuda") -> None:
        try:
            from lightglue import ALIKED
            from lightglue.utils import load_image
        except ImportError as error:
            raise RuntimeError(
                "ALIKED/LightGlue package is required for source-point detection"
            ) from error
        self._load_image = load_image
        self.device = torch.device(device)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)

    @torch.inference_mode()
    def extract(self, image_path: str | Path) -> SourcePoints:
        image = self._load_image(str(image_path)).to(self.device)
        started = time.perf_counter()
        features = self.extractor.extract(image)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        points = features["keypoints"]
        scores = features.get("keypoint_scores")
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if scores is None:
            scores = torch.ones(points.shape[0], device=points.device)
        elif scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        return SourcePoints(
            points_px=points.detach().cpu().numpy().astype(np.float64),
            scores=scores.detach().cpu().numpy().astype(np.float64),
            image_shape_hw=(int(image.shape[-2]), int(image.shape[-1])),
            extraction_seconds=elapsed,
        )
