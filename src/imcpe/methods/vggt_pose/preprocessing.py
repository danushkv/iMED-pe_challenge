"""Metadata for the official VGGT preprocessing transform.

Image tensors themselves are always produced by VGGT's official
``load_and_preprocess_images`` function.  This module mirrors its size
calculation only so known challenge intrinsics can be transformed into the
processed pixel coordinate system for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessMetadata:
    path: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    scale_x: float
    scale_y: float
    crop_top: int
    pad_left: int
    pad_top: int


def describe_official_preprocessing(
    image_path: Path, mode: str = "crop", target_size: int = 518
) -> PreprocessMetadata:
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be crop or pad")
    with Image.open(image_path) as image:
        width, height = image.size
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        crop_top = 0
        processed_width = target_size
        processed_height = target_size
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        crop_top = max((new_height - target_size) // 2, 0)
        pad_left = 0
        pad_top = 0
        processed_width = new_width
        processed_height = min(new_height, target_size)
    return PreprocessMetadata(
        path=str(image_path),
        original_width=width,
        original_height=height,
        processed_width=processed_width,
        processed_height=processed_height,
        scale_x=new_width / width,
        scale_y=new_height / height,
        crop_top=crop_top,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def transform_intrinsics_for_diagnostics(
    K: np.ndarray, metadata: PreprocessMetadata
) -> np.ndarray:
    """Map supplied intrinsics to VGGT's processed image coordinates."""

    result = np.asarray(K, dtype=np.float64).copy()
    result[0, 0] *= metadata.scale_x
    result[1, 1] *= metadata.scale_y
    result[0, 2] = result[0, 2] * metadata.scale_x + metadata.pad_left
    result[1, 2] = (
        result[1, 2] * metadata.scale_y - metadata.crop_top + metadata.pad_top
    )
    return result
