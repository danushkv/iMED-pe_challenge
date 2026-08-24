from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow

from .alignment import RobustSimilarity, apply_and_rebase, robust_sim3
from .confidence import AnchorConfidence, build_anchor_confidence
from .config import Method4Config
from .load_predictions import PredictionBundle
from .position_fusion import (
    PositionOptimizationResult,
    distribution,
    optimize_positions,
)


@dataclass(frozen=True)
class Method4Result:
    rows: list[PoseRow]
    centers_a_aligned: np.ndarray
    centers_b: np.ndarray
    centers_vo_aligned: np.ndarray
    delta_vo_aligned: np.ndarray
    centers_consensus: np.ndarray
    centers_final: np.ndarray
    confidence: AnchorConfidence
    vo_weights: np.ndarray
    optimization: PositionOptimizationResult
    alignment_a_to_b: RobustSimilarity
    alignment_vo_to_b: RobustSimilarity
    summary: dict[str, Any]


def _method1_local_displacements(bundle: PredictionBundle) -> np.ndarray:
    centers = bundle.method1.centers
    rotations = bundle.method1.rotations_initial_from_camera
    # Recover the exact stored increment T_previous_from_current. Its
    # translation is the current camera origin in the previous camera frame.
    # Rotate that local vector into Method-1 frame zero. This deliberately uses
    # local transforms rather than subtracting an independently estimated pose.
    displacements: list[np.ndarray] = []
    for index in range(1, len(centers)):
        T_previous = np.eye(4, dtype=np.float64)
        T_previous[:3, :3] = rotations[index - 1]
        T_previous[:3, 3] = centers[index - 1]
        T_current = np.eye(4, dtype=np.float64)
        T_current[:3, :3] = rotations[index]
        T_current[:3, 3] = centers[index]
        local_previous_from_current = np.linalg.inv(T_previous) @ T_current
        displacements.append(
            rotations[index - 1] @ local_previous_from_current[:3, 3]
        )
    return np.asarray(displacements)


def _vo_weights(bundle: PredictionBundle) -> np.ndarray:
    weights = np.ones(len(bundle.method1.frame_ids) - 1, dtype=np.float64)
    for index, frame_id in enumerate(bundle.method1.frame_ids[1:]):
        diagnostic = bundle.method1_frames.get(int(frame_id), {})
        if diagnostic and not bool(diagnostic.get("pnp_success", True)):
            weights[index] = 0.0
    return weights


def fuse_method4(bundle: PredictionBundle, config: Method4Config) -> Method4Result:
    centers_b = bundle.method2b.centers - bundle.method2b.centers[0]
    similarity_a = robust_sim3(
        bundle.method2a.centers,
        centers_b,
        iterations=config.alignment_iterations,
        huber_mad=config.alignment_huber_mad,
    )
    similarity_vo = robust_sim3(
        bundle.method1.centers,
        centers_b,
        iterations=config.alignment_iterations,
        huber_mad=config.alignment_huber_mad,
    )
    centers_a = apply_and_rebase(similarity_a, bundle.method2a.centers)
    centers_vo = apply_and_rebase(similarity_vo, bundle.method1.centers)
    delta_vo_local = _method1_local_displacements(bundle)
    delta_vo = similarity_vo.apply_vectors(delta_vo_local)
    confidence = build_anchor_confidence(
        frame_ids=bundle.method2b.frame_ids,
        centers_a_aligned=centers_a,
        centers_b=centers_b,
        rotations_a=bundle.method2a.rotations_initial_from_camera,
        rotations_b=bundle.method2b.rotations_initial_from_camera,
        frames_a=bundle.method2a_frames,
        frames_b=bundle.method2b_frames,
        summary_a=bundle.method2a_summary,
        summary_b=bundle.method2b_summary,
        version=config.confidence_version,
    )
    vo_weights = _vo_weights(bundle)
    optimization = optimize_positions(
        centers_a=centers_a,
        centers_b=centers_b,
        delta_vo=delta_vo,
        weight_a=confidence.weight_a,
        weight_b=confidence.weight_b,
        weight_vo=vo_weights,
        trajectory_step_scale=confidence.trajectory_step_scale,
        config=config,
    )
    rotations = bundle.rotation_source.rotations_initial_from_camera
    determinant_errors = np.abs(np.linalg.det(rotations) - 1.0)
    orthogonality_errors = np.linalg.norm(
        np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3),
        axis=(1, 2),
    )
    if np.max(determinant_errors) > 1e-7 or np.max(orthogonality_errors) > 1e-7:
        raise RuntimeError("Rotation source contains invalid SO(3) matrices")
    rows = [
        PoseRow(
            frame_idx=int(frame_id),
            # Challenge pose is T_0_from_t, hence translation equals C_t.
            t=optimization.centers[index].copy(),
            q_xyzw=Rotation.from_matrix(rotations[index]).as_quat(),
        )
        for index, frame_id in enumerate(bundle.method2b.frame_ids)
    ]
    if not np.allclose(rows[0].t, 0.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("Method 4 frame-zero translation is not zero")
    if not np.allclose(rows[0].q_xyzw, [0, 0, 0, 1], atol=1e-7, rtol=0.0):
        raise RuntimeError("Method 4 frame-zero rotation is not identity")
    downweighted = (confidence.weight_a < 0.5) | (confidence.weight_b < 0.5)
    summary: dict[str, Any] = {
        "num_frames": len(rows),
        "config": config.__dict__,
        "alignment_a_to_b": {
            "scale": similarity_a.scale,
            "residual": distribution(similarity_a.residuals),
            "num_robustly_downweighted": int(np.count_nonzero(similarity_a.weights < 0.999)),
        },
        "alignment_vo_to_b": {
            "scale": similarity_vo.scale,
            "residual": distribution(similarity_vo.residuals),
            "num_robustly_downweighted": int(np.count_nonzero(similarity_vo.weights < 0.999)),
        },
        "position_disagreement": distribution(confidence.position_disagreement),
        "position_disagreement_normalized": distribution(
            confidence.position_disagreement_normalized
        ),
        "rotation_disagreement_deg": distribution(confidence.rotation_disagreement_deg),
        "num_anchor_frames_downweighted_below_0_5": int(np.count_nonzero(downweighted)),
        "downweighted_frame_ids": bundle.method2b.frame_ids[downweighted].tolist(),
        "num_zero_weight_vo_edges": int(np.count_nonzero(vo_weights == 0.0)),
        "optimization_seconds": optimization.optimization_seconds,
        "optimizer_success": optimization.optimizer_success,
        "optimizer_message": optimization.optimizer_message,
        "optimizer_cost": optimization.optimizer_cost,
        "optimizer_nfev": optimization.optimizer_nfev,
        "anchor_a_residual": distribution(optimization.anchor_a_residuals),
        "anchor_b_residual": distribution(optimization.anchor_b_residuals),
        "vo_residual": distribution(optimization.vo_residuals),
        "no_absolute_max_error": optimization.no_absolute_max_error,
        "lambda_zero_max_error": optimization.lambda_zero_max_error,
        "maximum_rotation_determinant_error": float(np.max(determinant_errors)),
        "maximum_rotation_orthogonality_error": float(np.max(orthogonality_errors)),
    }
    return Method4Result(
        rows=rows,
        centers_a_aligned=centers_a,
        centers_b=centers_b,
        centers_vo_aligned=centers_vo,
        delta_vo_aligned=delta_vo,
        centers_consensus=optimization.consensus,
        centers_final=optimization.centers,
        confidence=confidence,
        vo_weights=vo_weights,
        optimization=optimization,
        alignment_a_to_b=similarity_a,
        alignment_vo_to_b=similarity_vo,
        summary=summary,
    )
