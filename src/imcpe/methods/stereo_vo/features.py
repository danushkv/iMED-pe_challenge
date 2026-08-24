from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd


FeatureDict = dict[str, Any]


class SharedFeatureMatcher:
    """ALIKED + LightGlue with explicit, reusable keypoint indices.

    Unlike the baseline's path-to-path matcher, this interface lets the caller
    extract a left image once and use the same feature tensor for both its
    stereo and temporal matches. Consequently, a shared left keypoint is an
    exact integer-index intersection rather than a coordinate approximation.
    """

    def __init__(self, device: str = "cuda", max_num_keypoints: int = 2048) -> None:
        self.device = torch.device(device)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)

    @torch.inference_mode()
    def extract(self, image_path: Path) -> FeatureDict:
        image = load_image(image_path).to(self.device)
        return self.extractor.extract(image)

    @torch.inference_mode()
    def match_indices(self, features0: FeatureDict, features1: FeatureDict) -> np.ndarray:
        result = self.matcher({"image0": features0, "image1": features1})
        matches = rbd(result)["matches"]
        return matches.detach().cpu().numpy().astype(np.int64, copy=False)

    @staticmethod
    def keypoints(features: FeatureDict) -> np.ndarray:
        keypoints = rbd(features)["keypoints"]
        return keypoints.detach().cpu().numpy().astype(np.float64, copy=False)


def intersect_left_keypoint_tracks(
    stereo_matches: np.ndarray,
    temporal_matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for L(t-1), R(t-1), and L(t) sharing one left keypoint."""

    temporal_by_left = {
        int(left_index): int(current_index)
        for left_index, current_index in np.asarray(temporal_matches).reshape(-1, 2)
    }
    previous_left: list[int] = []
    previous_right: list[int] = []
    current_left: list[int] = []
    for left_index, right_index in np.asarray(stereo_matches).reshape(-1, 2):
        current_index = temporal_by_left.get(int(left_index))
        if current_index is None:
            continue
        previous_left.append(int(left_index))
        previous_right.append(int(right_index))
        current_left.append(current_index)
    return (
        np.asarray(previous_left, dtype=np.int64),
        np.asarray(previous_right, dtype=np.int64),
        np.asarray(current_left, dtype=np.int64),
    )

