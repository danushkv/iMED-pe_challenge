from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConsensusResult:
    centers: np.ndarray
    observer_residuals: np.ndarray
    spread: np.ndarray
    maximum_disagreement: np.ndarray
    pairwise_median_disagreement: np.ndarray


@dataclass(frozen=True)
class ConsensusWeights:
    weights: np.ndarray
    observer_robust_z: np.ndarray
    frame_robust_z: np.ndarray
    residual_median: float
    residual_mad_sigma: float
    spread_median: float
    spread_mad_sigma: float


def geometric_median(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    max_iterations: int = 128,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("points must be non-empty Nx3")
    if len(points) == 1:
        return points[0].copy()
    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive = np.isfinite(weights) & (weights > 0)
    points = points[positive]
    weights = weights[positive]
    if not len(points):
        raise ValueError("geometric median has no positive-weight points")
    estimate = np.average(points, axis=0, weights=weights)
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - estimate, axis=1)
        coincident = distances <= tolerance
        if np.any(coincident):
            return np.average(points[coincident], axis=0, weights=weights[coincident])
        effective = weights / np.maximum(distances, tolerance)
        updated = np.sum(effective[:, None] * points, axis=0) / np.sum(effective)
        if np.linalg.norm(updated - estimate) <= tolerance:
            estimate = updated
            break
        estimate = updated
    return estimate


def geometric_median_trajectory(
    centers: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    num_frames = centers.shape[1]
    result = np.zeros((num_frames, 3), dtype=np.float64)
    for frame in range(num_frames):
        mask = valid[:, frame]
        frame_weights = None if weights is None else weights[mask, frame]
        if weights is not None:
            mask_indices = np.flatnonzero(mask)
            positive = frame_weights > 0
            mask = np.zeros_like(mask)
            mask[mask_indices[positive]] = True
            frame_weights = frame_weights[positive]
        if not np.any(mask):
            result[frame] = result[frame - 1] if frame else 0.0
            continue
        result[frame] = geometric_median(
            centers[mask, frame],
            frame_weights,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    result -= result[0]
    return result


def build_consensus(
    centers: np.ndarray,
    valid: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> ConsensusResult:
    trajectory = geometric_median_trajectory(
        centers,
        valid,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    residuals = np.linalg.norm(centers - trajectory[None], axis=2)
    residuals[~valid] = np.nan
    spread = np.asarray(
        [np.nanmedian(residuals[:, frame]) for frame in range(centers.shape[1])]
    )
    maximum = np.asarray(
        [np.nanmax(residuals[:, frame]) for frame in range(centers.shape[1])]
    )
    pairwise = np.zeros(centers.shape[1], dtype=np.float64)
    for frame in range(centers.shape[1]):
        points = centers[valid[:, frame], frame]
        distances = [
            np.linalg.norm(points[i] - points[j])
            for i in range(len(points))
            for j in range(i + 1, len(points))
        ]
        pairwise[frame] = float(np.median(distances)) if distances else 0.0
    return ConsensusResult(trajectory, residuals, spread, maximum, pairwise)


def _robust_positive_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(finite)
    median = float(np.median(finite[mask])) if np.any(mask) else 0.0
    mad = float(np.median(np.abs(finite[mask] - median))) if np.any(mask) else 0.0
    sigma = max(1.4826 * mad, 1.0e-9)
    z = np.zeros_like(finite)
    z[mask] = np.maximum((finite[mask] - median) / sigma, 0.0)
    z[~mask] = np.inf
    return z, median, sigma


def build_consensus_weights(
    consensus: ConsensusResult,
    valid: np.ndarray,
    *,
    version: str,
) -> ConsensusWeights:
    if version == "v0":
        weights = valid.astype(np.float64)
        observer_z = np.zeros_like(weights)
        frame_z = np.zeros(valid.shape[1], dtype=np.float64)
        return ConsensusWeights(weights, observer_z, frame_z, 0.0, 1.0, 0.0, 1.0)
    if version != "v1":
        raise ValueError(f"Unsupported consensus weighting {version}")
    observer_z, residual_median, residual_sigma = _robust_positive_z(
        consensus.observer_residuals[valid]
    )
    observer_z_full = np.full_like(consensus.observer_residuals, np.inf)
    observer_z_full[valid] = observer_z
    observer_weight = 1.0 / (1.0 + observer_z_full**2)
    observer_weight[~valid] = 0.0
    frame_z, spread_median, spread_sigma = _robust_positive_z(consensus.spread)
    frame_weight = 1.0 / (1.0 + frame_z**2)
    weights = observer_weight * frame_weight[None]
    weights[:, 0] = valid[:, 0].astype(np.float64)
    return ConsensusWeights(
        weights,
        observer_z_full,
        frame_z,
        residual_median,
        residual_sigma,
        spread_median,
        spread_sigma,
    )
