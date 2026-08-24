"""Inference-only dataset loading for Method 2.

This module intentionally has no pose reader and never opens ``pose.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Method2SequenceInputs:
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
        raise RuntimeError(f"{path} is missing entries: {missing}")
    return matrices


def image_path_map(path: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for image_path in sorted(path.glob("frame_*.png")):
        try:
            frame_id = int(image_path.stem.split("_")[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Cannot parse frame ID from {image_path}") from error
        mapping[frame_id] = image_path
    return mapping


def load_method2_inputs(sequence_dir: Path) -> Method2SequenceInputs:
    """Load synchronized images and intrinsics without reading ground truth."""

    intrinsics = read_intrinsics_only(sequence_dir / "K.txt")
    streams = {
        "e1_l": image_path_map(sequence_dir / "endoscope1" / "L"),
        "e1_r": image_path_map(sequence_dir / "endoscope1" / "R"),
        "e2_l": image_path_map(sequence_dir / "endoscope2" / "L"),
        "e2_r": image_path_map(sequence_dir / "endoscope2" / "R"),
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
    return Method2SequenceInputs(
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


def session_id_from_sequence_name(sequence_name: str) -> str:
    parts = sequence_name.split("_")
    if len(parts) < 2 or parts[0] != "session" or not parts[1].isdigit():
        raise ValueError(f"Cannot parse session ID from {sequence_name!r}")
    return parts[1].zfill(3)

