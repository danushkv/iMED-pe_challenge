"""Adapter around the official verlab/accelerated_features XFeat model."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import XFeatConfig


@dataclass(frozen=True)
class MatchResult:
    indices: np.ndarray
    confidence: np.ndarray


@dataclass
class RuntimeStatistics:
    extraction_seconds: float = 0.0
    matching_seconds: float = 0.0
    extracted_images: int = 0
    matched_pairs: int = 0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "extraction_seconds": self.extraction_seconds,
            "matching_seconds": self.matching_seconds,
            "extracted_images": self.extracted_images,
            "matched_pairs": self.matched_pairs,
            "extraction_ms_per_image": (
                1000.0 * self.extraction_seconds / max(self.extracted_images, 1)
            ),
            "matching_ms_per_pair": (
                1000.0 * self.matching_seconds / max(self.matched_pairs, 1)
            ),
        }


def official_checkout_info(xfeat_root: Path) -> dict[str, Any]:
    root = xfeat_root.resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "implementation": "official verlab/accelerated_features",
        "repository": "https://github.com/verlab/accelerated_features",
        "checkout": str(root),
        "commit": commit,
    }


class XFeatSparseMatcher:
    """Sparse XFeat extraction with exact-index mutual descriptor matching.

    The official pairwise convenience API returns pixel coordinates. Methods
    2A/2B require one source feature to retain the same integer ID in two
    independent matches, so this adapter extracts once and performs the same
    mutual-nearest cosine matching directly on official XFeat descriptors.
    """

    def __init__(
        self,
        *,
        xfeat_root: Path,
        weights: Path | None = None,
        device: str = "cuda",
        config: XFeatConfig = XFeatConfig(),
    ) -> None:
        config.validate()
        self.root = xfeat_root.resolve()
        if not (self.root / "modules" / "xfeat.py").is_file():
            raise FileNotFoundError(
                f"Official XFeat module not found at {self.root / 'modules/xfeat.py'}"
            )
        self.weights = (
            weights.resolve()
            if weights is not None
            else (self.root / "weights" / "xfeat.pt").resolve()
        )
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Local XFeat weights not found: {self.weights}. Runtime downloads "
                "are intentionally disabled for Docker compatibility."
            )
        self.device = torch.device(device)
        self.config = config
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        module = importlib.import_module("modules.xfeat")
        official_class = getattr(module, "XFeat")
        parameters = inspect.signature(official_class.__init__).parameters
        kwargs: dict[str, Any] = {}
        if "weights" in parameters:
            kwargs["weights"] = str(self.weights)
        if "top_k" in parameters:
            kwargs["top_k"] = config.max_features
        if "device" in parameters:
            kwargs["device"] = str(self.device)
        if (
            config.detection_threshold is not None
            and "detection_threshold" in parameters
        ):
            kwargs["detection_threshold"] = config.detection_threshold
        self.model = official_class(**kwargs)
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.runtime = RuntimeStatistics()

    @torch.inference_mode()
    def extract(self, image_path: Path) -> dict[str, Any]:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        tensor = torch.from_numpy(image).to(
            device=self.device, dtype=torch.float32
        )[None, None] / 255.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        kwargs: dict[str, Any] = {"top_k": self.config.max_features}
        if self.config.detection_threshold is not None:
            kwargs["detection_threshold"] = self.config.detection_threshold
        output = self.model.detectAndCompute(tensor, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.runtime.extraction_seconds += time.perf_counter() - start
        self.runtime.extracted_images += 1
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise RuntimeError(f"Expected one XFeat result, got {len(output)}")
            output = output[0]
        if not isinstance(output, dict):
            raise RuntimeError(f"Unexpected XFeat output type: {type(output).__name__}")
        required = ("keypoints", "descriptors", "scores")
        missing = [key for key in required if key not in output]
        if missing:
            raise RuntimeError(f"XFeat result missing keys: {missing}")
        keypoints = torch.as_tensor(output["keypoints"], device=self.device)
        descriptors = torch.as_tensor(output["descriptors"], device=self.device)
        scores = torch.as_tensor(output["scores"], device=self.device)
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if descriptors.ndim == 3 and descriptors.shape[0] == 1:
            descriptors = descriptors[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        keypoints = keypoints.reshape(-1, 2).to(torch.float32)
        descriptors = descriptors.reshape(keypoints.shape[0], -1).to(torch.float32)
        scores = scores.reshape(-1).to(torch.float32)
        if scores.shape[0] != keypoints.shape[0]:
            raise RuntimeError("XFeat score/keypoint count mismatch")
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "scores": scores,
            "image_size": np.asarray([image.shape[1], image.shape[0]], dtype=np.int64),
            "image_path": str(image_path),
        }

    @torch.inference_mode()
    def match_indices_with_scores(
        self, features0: dict[str, Any], features1: dict[str, Any]
    ) -> MatchResult:
        descriptors0 = features0["descriptors"]
        descriptors1 = features1["descriptors"]
        if descriptors0.shape[0] == 0 or descriptors1.shape[0] == 0:
            return MatchResult(
                indices=np.empty((0, 2), dtype=np.int64),
                confidence=np.empty(0, dtype=np.float64),
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        similarity = descriptors0 @ descriptors1.T
        confidence01, nearest01 = torch.max(similarity, dim=1)
        nearest10 = torch.argmax(similarity, dim=0)
        source = torch.arange(descriptors0.shape[0], device=self.device)
        mutual = nearest10[nearest01] == source
        if self.config.min_cosine_similarity > -1.0:
            mutual &= confidence01 >= self.config.min_cosine_similarity
        selected0 = source[mutual]
        selected1 = nearest01[mutual]
        selected_confidence = confidence01[mutual]
        order = torch.argsort(selected_confidence, descending=True, stable=True)
        matches = torch.stack([selected0[order], selected1[order]], dim=1)
        selected_confidence = selected_confidence[order]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.runtime.matching_seconds += time.perf_counter() - start
        self.runtime.matched_pairs += 1
        return MatchResult(
            indices=matches.detach().cpu().numpy().astype(np.int64, copy=False),
            confidence=selected_confidence.detach().cpu().numpy().astype(
                np.float64, copy=False
            ),
        )

    def match_indices(
        self, features0: dict[str, Any], features1: dict[str, Any]
    ) -> np.ndarray:
        return self.match_indices_with_scores(features0, features1).indices

    @staticmethod
    def keypoints(features: dict[str, Any]) -> np.ndarray:
        return features["keypoints"].detach().cpu().numpy().astype(
            np.float64, copy=False
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **official_checkout_info(self.root),
            "weights": str(self.weights),
            "weights_size_bytes": self.weights.stat().st_size,
            "mode": "sparse",
            "matching": "mutual-nearest cosine on official sparse descriptors",
            "config": self.config.to_json(),
            "runtime": self.runtime.snapshot(),
            "runtime_network_required": False,
        }
