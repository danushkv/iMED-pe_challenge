from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imcpe.methods.hybrid_pose_fusion.alignment import RobustSimilarity

from .config import Method4CConfig
from .consensus import geometric_median_trajectory
from .load_observers import ObserverBundle


@dataclass(frozen=True)
class MultiPositionOptimizationResult:
    centers: np.ndarray
    initial_consensus: np.ndarray
    observer_residuals: np.ndarray
    vo_residuals: np.ndarray
    optimization_seconds: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_cost: float
    optimizer_nfev: int
    no_absolute_max_error: float
    lambda_zero_max_error: float


def method1_local_displacements(bundle: ObserverBundle) -> np.ndarray:
    centers = bundle.method1.centers
    rotations = bundle.method1.rotations_initial_from_camera
    displacements: list[np.ndarray] = []
    for index in range(1, len(centers)):
        previous = np.eye(4, dtype=np.float64)
        previous[:3, :3] = rotations[index - 1]
        previous[:3, 3] = centers[index - 1]
        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = rotations[index]
        current[:3, 3] = centers[index]
        local_previous_from_current = np.linalg.inv(previous) @ current
        displacements.append(
            rotations[index - 1] @ local_previous_from_current[:3, 3]
        )
    return np.asarray(displacements)


def vo_weights(bundle: ObserverBundle) -> np.ndarray:
    weights = np.ones(len(bundle.frame_ids) - 1, dtype=np.float64)
    for index, frame_id in enumerate(bundle.frame_ids[1:]):
        diagnostic = bundle.method1.frames.get(int(frame_id), {})
        if diagnostic and not bool(diagnostic.get("pnp_success", True)):
            weights[index] = 0.0
    return weights


def align_vo_displacements(
    bundle: ObserverBundle, similarity: RobustSimilarity
) -> tuple[np.ndarray, np.ndarray]:
    centers = similarity.apply(bundle.method1.centers)
    centers -= centers[0]
    delta = similarity.apply_vectors(method1_local_displacements(bundle))
    return centers, delta


def _jacobian_sparsity(num_observers: int, num_frames: int) -> lil_matrix:
    variables = 3 * (num_frames - 1)
    residuals = 3 * (num_observers + 1) * (num_frames - 1)
    sparsity = lil_matrix((residuals, variables), dtype=np.int8)
    row = 0
    for _observer in range(num_observers):
        for frame in range(1, num_frames):
            column = 3 * (frame - 1)
            sparsity[row : row + 3, column : column + 3] = 1
            row += 3
    for edge in range(num_frames - 1):
        if edge > 0:
            previous = 3 * (edge - 1)
            sparsity[row : row + 3, previous : previous + 3] = 1
        current = 3 * edge
        sparsity[row : row + 3, current : current + 3] = 1
        row += 3
    return sparsity


def optimize_multi_observer_positions(
    *,
    observer_centers: np.ndarray,
    observer_valid: np.ndarray,
    observer_weights: np.ndarray,
    delta_vo: np.ndarray,
    weight_vo: np.ndarray,
    trajectory_step_scale: float,
    config: Method4CConfig,
) -> MultiPositionOptimizationResult:
    num_observers, num_frames, _ = observer_centers.shape
    safe_observer_centers = np.where(
        observer_valid[..., None], observer_centers, 0.0
    )
    if config.weighting == "method4a-v1":
        # Exact Method-4A regression: its two-observer optimizer starts from
        # the confidence-weighted arithmetic consensus, not a geometric median.
        denominator = np.sum(observer_weights, axis=0)
        safe = np.where(observer_valid[..., None], observer_centers, 0.0)
        initial = np.zeros((num_frames, 3), dtype=np.float64)
        usable = denominator > 1.0e-12
        initial[usable] = (
            np.sum(observer_weights[..., None] * safe, axis=0)[usable]
            / denominator[usable, None]
        )
        initial[~usable] = observer_centers[0, ~usable]
        initial -= initial[0]
    else:
        initial = geometric_median_trajectory(
            observer_centers,
            observer_valid,
            observer_weights,
            max_iterations=config.weiszfeld_iterations,
            tolerance=config.weiszfeld_tolerance,
        )
    integrated_vo = np.vstack(
        [np.zeros((1, 3), dtype=np.float64), np.cumsum(delta_vo, axis=0)]
    )

    if config.mode == "4c0" or config.lambda_vo == 0.0:
        centers = initial.copy()
        elapsed = 0.0
        success = True
        message = "absolute geometric-median consensus; optimizer not required"
        cost = 0.0
        nfev = 0
    elif config.absolute_weight_scale == 0.0:
        centers = integrated_vo.copy()
        elapsed = 0.0
        success = True
        message = "VO-only exact integration; optimizer not required"
        cost = 0.0
        nfev = 0
    else:
        sqrt_observer = np.sqrt(
            config.absolute_weight_scale * observer_weights[:, 1:]
        )
        sqrt_vo = np.sqrt(config.lambda_vo * config.vo_weight_scale * weight_vo)

        def unpack(parameters: np.ndarray) -> np.ndarray:
            return np.vstack((np.zeros((1, 3)), parameters.reshape(-1, 3)))

        def residual(parameters: np.ndarray) -> np.ndarray:
            current = unpack(parameters)
            blocks = [
                sqrt_observer[index, :, None]
                * (current[1:] - safe_observer_centers[index, 1:])
                for index in range(num_observers)
            ]
            relative = sqrt_vo[:, None] * (
                current[1:] - current[:-1] - delta_vo
            )
            return np.concatenate(
                [block.reshape(-1) for block in blocks] + [relative.reshape(-1)]
            )

        started = time.perf_counter()
        solution = least_squares(
            residual,
            initial[1:].reshape(-1),
            loss="huber",
            f_scale=max(
                config.huber_delta_steps * trajectory_step_scale,
                np.finfo(np.float64).eps,
            ),
            jac_sparsity=_jacobian_sparsity(num_observers, num_frames),
            max_nfev=config.optimizer_max_nfev,
            method="trf",
        )
        elapsed = time.perf_counter() - started
        centers = unpack(solution.x)
        success = bool(solution.success)
        message = str(solution.message)
        cost = float(solution.cost)
        nfev = int(solution.nfev)

    centers[0] = 0.0
    observer_residuals = np.linalg.norm(
        centers[None] - observer_centers, axis=2
    )
    observer_residuals[~observer_valid] = np.nan
    relative_residuals = np.linalg.norm(
        centers[1:] - centers[:-1] - delta_vo, axis=1
    )
    no_absolute = (
        float(np.max(np.abs(centers - integrated_vo)))
        if config.absolute_weight_scale == 0.0
        else float("nan")
    )
    lambda_zero = (
        float(np.max(np.abs(centers - initial)))
        if config.mode == "4c0" or config.lambda_vo == 0.0
        else float("nan")
    )
    return MultiPositionOptimizationResult(
        centers=centers,
        initial_consensus=initial,
        observer_residuals=observer_residuals,
        vo_residuals=relative_residuals,
        optimization_seconds=elapsed,
        optimizer_success=success,
        optimizer_message=message,
        optimizer_cost=cost,
        optimizer_nfev=nfev,
        no_absolute_max_error=no_absolute,
        lambda_zero_max_error=lambda_zero,
    )
