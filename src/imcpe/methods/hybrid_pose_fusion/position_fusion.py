from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .config import Method4Config


@dataclass(frozen=True)
class PositionOptimizationResult:
    centers: np.ndarray
    consensus: np.ndarray
    optimization_seconds: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_cost: float
    optimizer_nfev: int
    anchor_a_residuals: np.ndarray
    anchor_b_residuals: np.ndarray
    vo_residuals: np.ndarray
    no_absolute_max_error: float
    lambda_zero_max_error: float


def absolute_consensus(
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
) -> np.ndarray:
    denominator = weight_a + weight_b
    consensus = np.empty_like(centers_b)
    valid = denominator > 1e-12
    consensus[valid] = (
        weight_a[valid, None] * centers_a[valid]
        + weight_b[valid, None] * centers_b[valid]
    ) / denominator[valid, None]
    consensus[~valid] = centers_b[~valid]
    consensus -= consensus[0]
    return consensus


def _jacobian_sparsity(num_frames: int) -> lil_matrix:
    # Residual blocks: A anchors, B anchors (frames 1..N-1), then N-1 VO edges.
    variables = 3 * (num_frames - 1)
    residuals = 9 * (num_frames - 1)
    sparsity = lil_matrix((residuals, variables), dtype=np.int8)
    row = 0
    for _source in range(2):
        for frame in range(1, num_frames):
            column = 3 * (frame - 1)
            sparsity[row : row + 3, column : column + 3] = 1
            row += 3
    for edge in range(num_frames - 1):
        if edge > 0:
            previous_column = 3 * (edge - 1)
            sparsity[row : row + 3, previous_column : previous_column + 3] = 1
        current_column = 3 * edge
        sparsity[row : row + 3, current_column : current_column + 3] = 1
        row += 3
    return sparsity


def optimize_positions(
    *,
    centers_a: np.ndarray,
    centers_b: np.ndarray,
    delta_vo: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    weight_vo: np.ndarray,
    trajectory_step_scale: float,
    config: Method4Config,
) -> PositionOptimizationResult:
    num_frames = len(centers_b)
    consensus = absolute_consensus(centers_a, centers_b, weight_a, weight_b)
    integrated_vo = np.vstack(
        [np.zeros((1, 3), dtype=np.float64), np.cumsum(delta_vo, axis=0)]
    )
    if config.mode == "4a0" or config.lambda_vo == 0.0:
        centers = consensus.copy()
        elapsed = 0.0
        success = True
        message = "absolute consensus; optimizer not required"
        cost = 0.0
        nfev = 0
    elif config.absolute_weight_scale == 0.0:
        # Exact relative-only solution with C0 fixed; this is sanity Check A.
        centers = integrated_vo
        elapsed = 0.0
        success = True
        message = "VO-only exact integration; optimizer not required"
        cost = 0.0
        nfev = 0
    else:
        sqrt_a = np.sqrt(config.absolute_weight_scale * weight_a[1:])
        sqrt_b = np.sqrt(config.absolute_weight_scale * weight_b[1:])
        sqrt_vo = np.sqrt(config.lambda_vo * config.vo_weight_scale * weight_vo)

        def unpack(parameters: np.ndarray) -> np.ndarray:
            return np.vstack([np.zeros((1, 3)), parameters.reshape(-1, 3)])

        def residual(parameters: np.ndarray) -> np.ndarray:
            centers_current = unpack(parameters)
            anchor_a = sqrt_a[:, None] * (centers_current[1:] - centers_a[1:])
            anchor_b = sqrt_b[:, None] * (centers_current[1:] - centers_b[1:])
            relative = sqrt_vo[:, None] * (
                centers_current[1:] - centers_current[:-1] - delta_vo
            )
            return np.concatenate(
                [anchor_a.reshape(-1), anchor_b.reshape(-1), relative.reshape(-1)]
            )

        start = time.perf_counter()
        solution = least_squares(
            residual,
            consensus[1:].reshape(-1),
            loss="huber",
            f_scale=max(
                config.huber_delta_steps * trajectory_step_scale,
                np.finfo(np.float64).eps,
            ),
            jac_sparsity=_jacobian_sparsity(num_frames),
            max_nfev=config.optimizer_max_nfev,
            method="trf",
        )
        elapsed = time.perf_counter() - start
        centers = unpack(solution.x)
        success = bool(solution.success)
        message = str(solution.message)
        cost = float(solution.cost)
        nfev = int(solution.nfev)

    centers[0] = 0.0
    anchor_a_residuals = np.linalg.norm(centers - centers_a, axis=1)
    anchor_b_residuals = np.linalg.norm(centers - centers_b, axis=1)
    vo_residuals = np.linalg.norm(
        centers[1:] - centers[:-1] - delta_vo, axis=1
    )
    no_absolute_error = (
        float(np.max(np.abs(centers - integrated_vo)))
        if config.absolute_weight_scale == 0.0
        else float("nan")
    )
    lambda_zero_error = (
        float(np.max(np.abs(centers - consensus)))
        if config.lambda_vo == 0.0 or config.mode == "4a0"
        else float("nan")
    )
    return PositionOptimizationResult(
        centers=centers,
        consensus=consensus,
        optimization_seconds=elapsed,
        optimizer_success=success,
        optimizer_message=message,
        optimizer_cost=cost,
        optimizer_nfev=nfev,
        anchor_a_residuals=anchor_a_residuals,
        anchor_b_residuals=anchor_b_residuals,
        vo_residuals=vo_residuals,
        no_absolute_max_error=no_absolute_error,
        lambda_zero_max_error=lambda_zero_error,
    )


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"median": float("nan"), "p90": float("nan"), "maximum": float("nan")}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }
