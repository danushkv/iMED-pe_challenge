from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class SampledWarp:
    target_points_px: np.ndarray
    certainty: np.ndarray
    valid: np.ndarray
    target_points_normalized: np.ndarray


def pixels_to_normalized(
    points_px: np.ndarray | torch.Tensor,
    image_shape_hw: tuple[int, int],
) -> torch.Tensor:
    """Convert original-image pixels to RoMa's normalized coordinates.

    RoMa uses the grid_sample/align_corners=False convention documented by its
    official matcher implementation:

        x_norm = 2 * x_px / width  - 1
        y_norm = 2 * y_px / height - 1
    """

    height, width = image_shape_hw
    points = torch.as_tensor(points_px, dtype=torch.float32)
    scale = points.new_tensor([float(width), float(height)])
    return 2.0 * points / scale - 1.0


def normalized_to_pixels(
    points_normalized: np.ndarray | torch.Tensor,
    image_shape_hw: tuple[int, int],
) -> torch.Tensor:
    """Convert RoMa normalized coordinates to original-image pixels."""

    height, width = image_shape_hw
    points = torch.as_tensor(points_normalized, dtype=torch.float32)
    scale = points.new_tensor([float(width), float(height)])
    return 0.5 * scale * (points + 1.0)


def _unbatch_warp(warp: torch.Tensor) -> torch.Tensor:
    warp = torch.as_tensor(warp)
    while warp.ndim > 3 and warp.shape[0] == 1:
        warp = warp[0]
    if warp.ndim != 3 or warp.shape[-1] < 4:
        raise ValueError(f"Expected RoMa warp shaped HxWx4, got {tuple(warp.shape)}")
    return warp


def _unbatch_certainty(certainty: torch.Tensor) -> torch.Tensor:
    certainty = torch.as_tensor(certainty)
    while certainty.ndim > 2 and certainty.shape[0] == 1:
        certainty = certainty[0]
    if certainty.ndim != 2:
        raise ValueError(
            f"Expected RoMa certainty shaped HxW, got {tuple(certainty.shape)}"
        )
    return certainty


def sample_dense_warp(
    warp: torch.Tensor,
    certainty: torch.Tensor,
    source_points_px: np.ndarray | torch.Tensor,
    source_shape_hw: tuple[int, int],
    target_shape_hw: tuple[int, int],
) -> SampledWarp:
    """Sample a single dense RoMa warp at exact source pixel coordinates.

    The first two warp channels encode its source grid and the last two encode
    target normalized coordinates.  Sampling only the target channels retains
    exact ALIKED source identity; no nearest-neighbour feature merge occurs.
    """

    dense_warp = _unbatch_warp(warp)
    dense_certainty = _unbatch_certainty(certainty)
    if dense_warp.shape[:2] != dense_certainty.shape:
        raise ValueError(
            "Warp/certainty resolution mismatch: "
            f"{tuple(dense_warp.shape)} vs {tuple(dense_certainty.shape)}"
        )

    points_px = torch.as_tensor(source_points_px, dtype=torch.float32)
    if points_px.ndim != 2 or points_px.shape[1] != 2:
        raise ValueError(f"Expected Nx2 source points, got {tuple(points_px.shape)}")

    device = dense_warp.device
    grid = pixels_to_normalized(points_px, source_shape_hw).to(device=device)
    # grid_sample expects N x H_out x W_out x 2.  Treat all queried points as
    # one output row so the dense field is evaluated once for all ALIKED IDs.
    query_grid = grid.reshape(1, 1, -1, 2)

    target_field = dense_warp[..., -2:].permute(2, 0, 1).unsqueeze(0)
    target_normalized = torch_functional.grid_sample(
        target_field,
        query_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, :, 0, :].transpose(0, 1)

    certainty_field = dense_certainty.unsqueeze(0).unsqueeze(0)
    sampled_certainty = torch_functional.grid_sample(
        certainty_field,
        query_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0, 0, :]

    target_px = normalized_to_pixels(target_normalized, target_shape_hw).to(device)
    finite = torch.isfinite(target_normalized).all(dim=1) & torch.isfinite(
        sampled_certainty
    )
    inside_source = (grid.abs() <= 1.0).all(dim=1)
    inside_target = (target_normalized.abs() <= 1.0).all(dim=1)
    valid = finite & inside_source & inside_target

    return SampledWarp(
        target_points_px=target_px.detach().cpu().numpy().astype(np.float64),
        certainty=sampled_certainty.detach().cpu().numpy().astype(np.float64),
        valid=valid.detach().cpu().numpy().astype(bool),
        target_points_normalized=target_normalized.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
    )
