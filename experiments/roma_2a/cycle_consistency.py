from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .warp_points import sample_dense_warp


@dataclass(frozen=True)
class CycleResult:
    returned_source_points_px: np.ndarray
    error_px: np.ndarray
    valid: np.ndarray
    reverse_certainty: np.ndarray


def compute_cycle_consistency(
    source_points_px: np.ndarray,
    forward_target_points_px: np.ndarray,
    forward_valid: np.ndarray,
    reverse_warp: torch.Tensor,
    reverse_certainty: torch.Tensor,
    source_shape_hw: tuple[int, int],
    target_shape_hw: tuple[int, int],
) -> CycleResult:
    """Evaluate source -> target -> source consistency in original pixels."""

    reverse = sample_dense_warp(
        reverse_warp,
        reverse_certainty,
        forward_target_points_px,
        source_shape_hw=target_shape_hw,
        target_shape_hw=source_shape_hw,
    )
    errors = np.linalg.norm(
        reverse.target_points_px - np.asarray(source_points_px, dtype=np.float64),
        axis=1,
    )
    valid = np.asarray(forward_valid, dtype=bool) & reverse.valid & np.isfinite(errors)
    errors = np.where(valid, errors, np.inf)
    return CycleResult(
        returned_source_points_px=reverse.target_points_px,
        error_px=errors,
        valid=valid,
        reverse_certainty=reverse.certainty,
    )
