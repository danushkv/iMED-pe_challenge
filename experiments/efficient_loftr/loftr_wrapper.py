"""Thin offline wrapper around the official zju3dv EfficientLoFTR code."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import time
import types
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from config import EfficientLoFTRConfig


@dataclass(frozen=True)
class LoFTRMatches:
    points0: np.ndarray
    points1: np.ndarray
    confidence: np.ndarray
    original_shape0: tuple[int, int]
    original_shape1: tuple[int, int]
    inference_shape0: tuple[int, int]
    inference_shape1: tuple[int, int]
    inference_seconds: float

    def filtered(self, threshold: float | None) -> "LoFTRMatches":
        if threshold is None:
            return self
        keep = self.confidence >= float(threshold)
        return LoFTRMatches(
            points0=self.points0[keep],
            points1=self.points1[keep],
            confidence=self.confidence[keep],
            original_shape0=self.original_shape0,
            original_shape1=self.original_shape1,
            inference_shape0=self.inference_shape0,
            inference_shape1=self.inference_shape1,
            inference_seconds=self.inference_seconds,
        )


@dataclass
class RuntimeCounter:
    inference_seconds: float = 0.0
    matched_pairs: int = 0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "inference_seconds": self.inference_seconds,
            "matched_pairs": self.matched_pairs,
            "inference_ms_per_pair": (
                1000.0 * self.inference_seconds / max(self.matched_pairs, 1)
            ),
        }


def _install_lightning_inference_stub() -> bool:
    """Provide only symbols imported by upstream inference/checkpoint pickle.

    The official repository imports legacy PyTorch Lightning from a utility
    module even though inference only needs ``detect_NaN``.  Installing the
    pinned training stack would conflict with the challenge environment.  This
    shim does not change model math and exists only when Lightning is absent.
    """

    if importlib.util.find_spec("pytorch_lightning") is not None:
        return False

    pl = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    checkpoint_module = types.ModuleType(
        "pytorch_lightning.callbacks.model_checkpoint"
    )

    def rank_zero_only(function):
        return function

    rank_zero_only.rank = 0
    utilities.rank_zero_only = rank_zero_only
    model_checkpoint = type("ModelCheckpoint", (), {})
    model_checkpoint.__module__ = checkpoint_module.__name__
    checkpoint_module.ModelCheckpoint = model_checkpoint
    callbacks.model_checkpoint = checkpoint_module
    callbacks.ModelCheckpoint = model_checkpoint
    pl.utilities = utilities
    pl.callbacks = callbacks
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.utilities"] = utilities
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules[checkpoint_module.__name__] = checkpoint_module
    return True


class EfficientLoFTRMatcher:
    def __init__(
        self,
        *,
        loftr_root: Path,
        checkpoint: Path,
        device: str = "cuda",
        config: EfficientLoFTRConfig = EfficientLoFTRConfig(),
    ) -> None:
        self.root = Path(loftr_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not (self.root / "src" / "loftr" / "__init__.py").is_file():
            raise FileNotFoundError(f"Not an EfficientLoFTR checkout: {self.root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Missing EfficientLoFTR checkpoint: {self.checkpoint}")
        self.device = torch.device(device)
        self.config = config
        self.runtime = RuntimeCounter()
        self.used_lightning_stub = _install_lightning_inference_stub()

        # Upstream imports are rooted at the checkout (``from src...``).
        root_string = str(self.root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        from src.loftr import (  # type: ignore[import-not-found]
            LoFTR,
            full_default_cfg,
            opt_default_cfg,
            reparameter,
        )

        model_config = deepcopy(
            full_default_cfg if config.model_type == "full" else opt_default_cfg
        )
        model_config["mp"] = config.precision == "mp"
        model_config["half"] = config.precision == "fp16"
        matcher = LoFTR(config=model_config)

        # This is a user-downloaded checkpoint from the official project.  It
        # contains training callback metadata, so weights_only=False is needed;
        # only the state_dict is retained.
        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        checkpoint_payload = torch.load(self.checkpoint, **load_kwargs)
        if not isinstance(checkpoint_payload, dict) or "state_dict" not in checkpoint_payload:
            raise RuntimeError("EfficientLoFTR checkpoint has no state_dict")
        matcher.load_state_dict(checkpoint_payload["state_dict"])
        del checkpoint_payload
        matcher = reparameter(matcher)
        if config.precision == "fp16":
            matcher = matcher.half()
        self.matcher = matcher.eval().to(self.device)

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _read(self, path: Path) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        original = (int(image.shape[0]), int(image.shape[1]))
        divisor = self.config.resize_divisor
        inference_width = max(divisor, image.shape[1] // divisor * divisor)
        inference_height = max(divisor, image.shape[0] // divisor * divisor)
        resized = cv2.resize(
            image,
            (inference_width, inference_height),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(resized)[None, None].to(self.device)
        tensor = tensor.half() if self.config.precision == "fp16" else tensor.float()
        tensor = tensor / 255.0
        return tensor, original, (inference_height, inference_width)

    @staticmethod
    def _restore(points: np.ndarray, original: tuple[int, int], inference: tuple[int, int]) -> np.ndarray:
        scale = np.array(
            [original[1] / inference[1], original[0] / inference[0]],
            dtype=np.float64,
        )
        return np.asarray(points, dtype=np.float64) * scale.reshape(1, 2)

    @torch.inference_mode()
    def match(self, image0_path: Path, image1_path: Path) -> LoFTRMatches:
        image0, original0, inference0 = self._read(Path(image0_path))
        image1, original1, inference1 = self._read(Path(image1_path))
        batch = {"image0": image0, "image1": image1}
        self._sync()
        start = time.perf_counter()
        if self.config.precision == "mp":
            with torch.autocast(device_type=self.device.type, enabled=True):
                self.matcher(batch)
        else:
            self.matcher(batch)
        self._sync()
        elapsed = time.perf_counter() - start
        self.runtime.inference_seconds += elapsed
        self.runtime.matched_pairs += 1
        points0 = batch["mkpts0_f"].detach().float().cpu().numpy()
        points1 = batch["mkpts1_f"].detach().float().cpu().numpy()
        confidence = batch["mconf"].detach().float().cpu().numpy().reshape(-1)
        if not (len(points0) == len(points1) == len(confidence)):
            raise RuntimeError("EfficientLoFTR returned inconsistent match arrays")
        return LoFTRMatches(
            points0=self._restore(points0, original0, inference0),
            points1=self._restore(points1, original1, inference1),
            confidence=np.asarray(confidence, dtype=np.float64),
            original_shape0=original0,
            original_shape1=original1,
            inference_shape0=inference0,
            inference_shape1=inference1,
            inference_seconds=elapsed,
        )

    def metadata(self) -> dict[str, Any]:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        return {
            "implementation": "official zju3dv/EfficientLoFTR",
            "repository": "https://github.com/zju3dv/EfficientLoFTR",
            "commit": commit,
            "checkpoint": str(self.checkpoint),
            "checkpoint_size_bytes": self.checkpoint.stat().st_size,
            "config": asdict(self.config),
            "preprocessing": "grayscale; resize each dimension down to a multiple of 32; map matches back to original pixels",
            "output": ["mkpts0_f", "mkpts1_f", "mconf"],
            "reparameterized": True,
            "legacy_lightning_stub": self.used_lightning_stub,
            "runtime_network_required": False,
        }

