"""Frozen, GT-independent settings for Method 5 stereo refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StereoRefinementConfig:
    sampled_frames_per_session: int = 30
    max_tracks_per_frame: int = 150
    max_rotation_correction_deg: float = 3.0
    max_translation_direction_correction_deg: float = 5.0
    rotation_prior_weight: float = 100.0
    translation_prior_weight: float = 100.0
    huber_delta_px: float = 2.0
    invalid_projection_penalty_px: float = 10.0
    optimizer_max_nfev: int = 80

    # Geometry-only validity gates. These are frozen safety checks, not fitted
    # from ATE or hidden scores.
    split_rotation_stability_deg: float = 1.5
    split_translation_stability_deg: float = 2.5
    sequence_rotation_stability_p90_deg: float = 2.0
    sequence_translation_stability_p90_deg: float = 3.0
    minimum_relative_median_improvement: float = 0.01
    maximum_per_view_p90_regression: float = 0.05
    bound_margin_fraction: float = 0.05
    # A sequence-specific fit may have only two uniformly sampled frames in
    # sessions with many sequences. Each frame still contributes tens to
    # hundreds of fixed PnP-inlier tracks, so this is sufficient for the
    # six-variable identifiability diagnostic. Split-half fits use many more.
    minimum_subset_frames: int = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
