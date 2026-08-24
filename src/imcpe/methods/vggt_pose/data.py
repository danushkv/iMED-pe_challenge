"""Inference-only synchronized image and intrinsic loading for Method 3.

This module intentionally does not import any Method 1/2 package and has no
ground-truth pose reader.  In particular, using Method 3 does not require
LightGlue to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VGGTSequenceInputs:
    sequence_name: str
    frame_ids: list[int]
    k1_l: np.ndarray
    k1_r: np.ndarray
    k2_l: np.ndarray
    k2_r: np.ndarray
    e1_l_images: list[Path]
    e1_r_images: list[Path]
    e2_l_images: list[Path]
    e2_r_images: list[Path]


_REQUIRED_K_KEYS = ("K1_L", "K1_R", "K2_L", "K2_R")


def read_intrinsics_only(path: Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    matrices: dict[str, np.ndarray] = {}
    current: str | None = None
    rows: list[list[float]] = []
    for line in lines:
        if line.startswith("#"):
            payload = line.lstrip("#").strip()
            current = payload.split()[0] if payload else None
            rows = []
            continue
        if current is None:
            continue
        rows.append([float(value) for value in line.split()])
        if len(rows) == 3:
            matrix = np.asarray(rows, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise RuntimeError(f"Malformed intrinsic matrix {current} in {path}")
            matrices[current] = matrix
            current = None
            rows = []
    missing = [key for key in _REQUIRED_K_KEYS if key not in matrices]
    if missing:
        raise RuntimeError(f"{path} is missing intrinsic entries: {missing}")
    return matrices


def _image_path_map(path: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for image_path in sorted(path.glob("frame_*.png")):
        try:
            frame_id = int(image_path.stem.split("_")[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Cannot parse frame ID from {image_path}") from error
        mapping[frame_id] = image_path
    return mapping


def load_vggt_inputs(sequence_dir: Path) -> VGGTSequenceInputs:
    """Load synchronized four-camera inputs without opening ``pose.txt``."""

    intrinsics = read_intrinsics_only(sequence_dir / "K.txt")
    streams = {
        "e1_l": _image_path_map(sequence_dir / "endoscope1" / "L"),
        "e1_r": _image_path_map(sequence_dir / "endoscope1" / "R"),
        "e2_l": _image_path_map(sequence_dir / "endoscope2" / "L"),
        "e2_r": _image_path_map(sequence_dir / "endoscope2" / "R"),
    }
    if not streams["e1_l"]:
        raise RuntimeError(f"No Endoscope1-left images in {sequence_dir}")
    frame_ids = sorted(streams["e1_l"])
    for name, mapping in streams.items():
        if sorted(mapping) != frame_ids:
            missing = sorted(set(frame_ids) - set(mapping))
            extra = sorted(set(mapping) - set(frame_ids))
            raise RuntimeError(
                f"Synchronized-frame mismatch for {name} in {sequence_dir.name}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
    return VGGTSequenceInputs(
        sequence_name=sequence_dir.name,
        frame_ids=frame_ids,
        k1_l=intrinsics["K1_L"],
        k1_r=intrinsics["K1_R"],
        k2_l=intrinsics["K2_L"],
        k2_r=intrinsics["K2_R"],
        e1_l_images=[streams["e1_l"][frame_id] for frame_id in frame_ids],
        e1_r_images=[streams["e1_r"][frame_id] for frame_id in frame_ids],
        e2_l_images=[streams["e2_l"][frame_id] for frame_id in frame_ids],
        e2_r_images=[streams["e2_r"][frame_id] for frame_id in frame_ids],
    )
