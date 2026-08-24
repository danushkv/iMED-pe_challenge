from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustSimilarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (self.scale * self.rotation @ np.asarray(points).T).T + self.translation

    def apply_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return (self.scale * self.rotation @ np.asarray(vectors).T).T


def _weighted_umeyama(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    source_mean = np.sum(weights[:, None] * source, axis=0)
    target_mean = np.sum(weights[:, None] * target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (weights[:, None] * target_centered).T @ source_centered
    U, singular_values, Vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1.0
    rotation = U @ np.diag(sign) @ Vt
    variance = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    if variance <= np.finfo(np.float64).eps:
        raise RuntimeError("Degenerate trajectory for Sim(3) alignment")
    scale = float(np.sum(singular_values * sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid Sim(3) scale")
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 8,
    huber_mad: float = 2.5,
) -> RobustSimilarity:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source/target must both be Nx3")
    if len(source) < 3:
        raise ValueError("at least three points are required")
    weights = np.ones(len(source), dtype=np.float64)
    for _ in range(iterations):
        scale, rotation, translation = _weighted_umeyama(source, target, weights)
        aligned = (scale * rotation @ source.T).T + translation
        residuals = np.linalg.norm(aligned - target, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = max(1.4826 * mad, np.finfo(np.float64).eps)
        threshold = median + huber_mad * robust_sigma
        new_weights = np.ones_like(residuals)
        tail = residuals > threshold
        new_weights[tail] = threshold / np.maximum(residuals[tail], threshold)
        if np.max(np.abs(new_weights - weights)) < 1e-5:
            weights = new_weights
            break
        weights = new_weights
    scale, rotation, translation = _weighted_umeyama(source, target, weights)
    aligned = (scale * rotation @ source.T).T + translation
    residuals = np.linalg.norm(aligned - target, axis=1)
    return RobustSimilarity(scale, rotation, translation, weights, residuals)


def apply_and_rebase(similarity: RobustSimilarity, points: np.ndarray) -> np.ndarray:
    aligned = similarity.apply(points)
    return aligned - aligned[0]
