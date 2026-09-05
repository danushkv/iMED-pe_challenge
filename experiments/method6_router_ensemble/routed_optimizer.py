"""Method 6A/6B/6C fusion using the frozen Method-4A robust optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.method4c_consensus.config import Method4CConfig
from experiments.method4c_consensus.optimize import optimize_multi_observer_positions

from .config import FusionConfig
from .router import entropy


@dataclass(frozen=True)
class RoutedResult:
    centers_6a: np.ndarray
    centers_6b: np.ndarray
    centers_6c: np.ndarray
    entropy: np.ndarray
    entropy_strength: np.ndarray
    optimization_b_seconds: float
    optimization_c_seconds: float


def _normalized_weights(probabilities: np.ndarray, valid: np.ndarray) -> np.ndarray:
    weights = probabilities.T * valid
    denominator = np.sum(weights, axis=0, keepdims=True)
    fallback = valid.astype(np.float64)
    fallback /= np.maximum(np.sum(fallback, axis=0, keepdims=True), 1.0)
    return np.where(denominator > 1.0e-12, weights / np.maximum(denominator, 1e-12), fallback)


def route_and_optimize(
    *,
    expert_centers: np.ndarray,
    expert_valid: np.ndarray,
    probabilities: np.ndarray,
    delta_vo: np.ndarray,
    vo_weights: np.ndarray,
    trajectory_step_scale: float,
    fusion: FusionConfig,
) -> RoutedResult:
    weights = _normalized_weights(probabilities, expert_valid)
    safe = np.where(expert_valid[..., None], expert_centers, 0.0)
    centers_6a = np.sum(weights[..., None] * safe, axis=0)
    centers_6a -= centers_6a[0]
    config = Method4CConfig(
        mode="4c1",
        weighting="v0",
        lambda_vo=fusion.lambda_vo,
        absolute_weight_scale=fusion.absolute_weight_scale,
        vo_weight_scale=fusion.vo_weight_scale,
        huber_delta_steps=fusion.huber_delta_steps,
        alignment_iterations=fusion.alignment_iterations,
        alignment_huber_mad=fusion.alignment_huber_mad,
        optimizer_max_nfev=fusion.optimizer_max_nfev,
    )
    result_b = optimize_multi_observer_positions(
        observer_centers=expert_centers,
        observer_valid=expert_valid,
        observer_weights=weights,
        delta_vo=delta_vo,
        weight_vo=vo_weights,
        trajectory_step_scale=trajectory_step_scale,
        config=config,
    )
    values_entropy = entropy(probabilities)
    strength = np.clip(1.0 - values_entropy / np.log(probabilities.shape[1]), 0.0, 1.0)
    result_c = optimize_multi_observer_positions(
        observer_centers=expert_centers,
        observer_valid=expert_valid,
        observer_weights=weights * strength[None, :],
        delta_vo=delta_vo,
        weight_vo=vo_weights,
        trajectory_step_scale=trajectory_step_scale,
        config=config,
    )
    return RoutedResult(
        centers_6a=centers_6a,
        centers_6b=result_b.centers,
        centers_6c=result_c.centers,
        entropy=values_entropy,
        entropy_strength=strength,
        optimization_b_seconds=result_b.optimization_seconds,
        optimization_c_seconds=result_c.optimization_seconds,
    )

