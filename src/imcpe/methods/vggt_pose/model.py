from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VGGTBatchPrediction:
    extrinsics_cam_from_world: np.ndarray
    intrinsics: np.ndarray
    input_shape: tuple[int, int]
    inference_seconds: float
    peak_gpu_memory_bytes: int


class VGGTCameraPredictor:
    """Official VGGT aggregator + camera-head inference only."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        model_id: str = "facebook/VGGT-1B",
        vggt_root: Path | None = None,
    ) -> None:
        if vggt_root is not None:
            root = str(vggt_root.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
        import torch
        from vggt.models.vggt import VGGT

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = self._resolve_dtype(dtype)
        load_start = time.perf_counter()
        # Use the official Hugging Face loader.  Remove unused prediction heads
        # before transfer to GPU; only aggregator + camera_head are executed.
        model = VGGT.from_pretrained(model_id)
        model.depth_head = None
        model.point_head = None
        model.track_head = None
        self.model = model.eval().to(self.device)
        self.model_load_seconds = time.perf_counter() - load_start
        self.model_id = model_id

    def _resolve_dtype(self, requested: str):
        torch = self.torch
        if requested == "float32":
            return torch.float32
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8:
            return torch.bfloat16
        return torch.float16 if self.device.type == "cuda" else torch.float32

    def predict(self, image_paths: list[Path], preprocess_mode: str) -> VGGTBatchPrediction:
        torch = self.torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = load_and_preprocess_images(
            [str(path) for path in image_paths], mode=preprocess_mode
        ).to(self.device)
        images = images[None]
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                enabled=(
                    self.device.type == "cuda" and self.dtype != torch.float32
                ),
                dtype=self.dtype,
            ):
                aggregated_tokens_list, _ = self.model.aggregator(images)
            # This follows the official detailed camera-only example.
            pose_encoding = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_encoding, images.shape[-2:]
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
        else:
            peak = 0
        elapsed = time.perf_counter() - start
        return VGGTBatchPrediction(
            extrinsics_cam_from_world=extrinsic[0].detach().float().cpu().numpy(),
            intrinsics=intrinsic[0].detach().float().cpu().numpy(),
            input_shape=(int(images.shape[-2]), int(images.shape[-1])),
            inference_seconds=elapsed,
            peak_gpu_memory_bytes=peak,
        )
