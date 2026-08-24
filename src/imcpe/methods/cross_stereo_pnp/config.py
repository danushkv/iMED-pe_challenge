from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionCalibrationConfig:
    """Configuration for one fixed, scale-free E1-left to E1-right transform."""

    num_samples: int = 30
    essential_ransac_threshold_normalized: float = 1e-3
    essential_ransac_probability: float = 0.999
    min_matches: int = 50
    min_essential_inliers: int = 30
    min_inlier_ratio: float = 0.25
    min_cheirality_fraction: float = 0.50
    max_rotation_deviation_deg: float = 5.0
    max_translation_deviation_deg: float = 20.0
    min_stable_pairs: int = 3
    min_stable_fraction: float = 0.50
    max_pooled_points_per_pair: int = 400
    max_epipolar_p95_px: float = 2.0
    max_reprojection_median_px: float = 2.0


@dataclass(frozen=True)
class Method2Config:
    """Initial geometry-first thresholds; all are exposed by the runner CLI."""

    min_shared_tracks: int = 8
    min_depth: float = 0.05
    max_depth: float = 200.0
    use_calibration_depth_bounds: bool = True
    max_stereo_reprojection_error_px: float = 2.0
    min_triangulation_angle_deg: float = 0.0
    pnp_min_points: int = 8
    pnp_min_inliers: int = 8
    pnp_iterations: int = 200
    pnp_reprojection_error_px: float = 3.0
    pnp_confidence: float = 0.999
    refine_lm: bool = True
    fallback_min_inlier_ratio: float = 0.20
    stop_min_success_rate: float = 0.80
    stop_min_median_shared_tracks: float = 15.0
    stop_max_median_reprojection_error_px: float = 5.0
    stop_max_translation_jump_ratio: float = 10.0

