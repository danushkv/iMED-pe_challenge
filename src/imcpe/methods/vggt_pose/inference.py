from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imcpe.io_pose import PoseRow

from .config import Method3Config
from .data import VGGTSequenceInputs
from .diagnostics import (
    Method3FrameDiagnostic,
    stereo_rotation_variation,
    summarize_diagnostics,
)
from .model import VGGTCameraPredictor
from .pose_utils import (
    camera_center_world,
    challenge_relative_from_cross,
    homogeneous,
    invert_se3,
    relative_camera_transform,
    rotation_angle_degrees,
    rotation_quality,
    scaled_camera_extrinsics,
    transform_to_pose_row,
)
from .preprocessing import (
    describe_official_preprocessing,
    transform_intrinsics_for_diagnostics,
)


@dataclass
class Method3SequenceResult:
    rows: list[PoseRow]
    frame_ids: np.ndarray
    raw_extrinsics_cam_from_world: np.ndarray
    raw_intrinsics_processed: np.ndarray
    scale_factors: np.ndarray
    scale_reference_e1_baselines_raw: np.ndarray
    e1_stereo_baselines_raw: np.ndarray
    e2_stereo_baselines_raw: np.ndarray
    T_cross_current_scaled: np.ndarray
    T_cross_reference_scaled: np.ndarray
    T_initial_from_current: np.ndarray
    T_direct_e20_from_e2t: np.ndarray
    prediction_valid: np.ndarray
    diagnostics: list[Method3FrameDiagnostic]
    summary: dict[str, object]


def select_frame_indices(
    num_frames: int,
    *,
    max_frames: int | None = None,
    sample_count: int | None = None,
) -> list[int]:
    if max_frames is not None and sample_count is not None:
        raise ValueError("use either max_frames or sample_count, not both")
    if num_frames <= 0:
        return []
    if max_frames is not None:
        return list(range(min(num_frames, max_frames)))
    if sample_count is None or sample_count >= num_frames:
        return list(range(num_frames))
    if sample_count < 2:
        return [0]
    # Uniform by index, independent of pose.txt or any ground-truth error.
    return sorted(set(np.rint(np.linspace(0, num_frames - 1, sample_count)).astype(int)))


def _current_paths(inputs: VGGTSequenceInputs, index: int) -> list[Path]:
    return [
        inputs.e1_l_images[index],
        inputs.e1_r_images[index],
        inputs.e2_l_images[index],
        inputs.e2_r_images[index],
    ]


def _known_intrinsics(inputs: VGGTSequenceInputs) -> list[np.ndarray]:
    return [inputs.k1_l, inputs.k1_r, inputs.k2_l, inputs.k2_r]


def _focal_errors(
    predicted: np.ndarray,
    known: list[np.ndarray],
    paths: list[Path],
    mode: str,
) -> tuple[list[float], list[object]]:
    metadata = [describe_official_preprocessing(path, mode=mode) for path in paths]
    errors: list[float] = []
    for pred, K, meta in zip(predicted, known, metadata):
        transformed = transform_intrinsics_for_diagnostics(K, meta)
        errors.extend(
            [
                abs(float(pred[0, 0]) - transformed[0, 0]) / transformed[0, 0],
                abs(float(pred[1, 1]) - transformed[1, 1]) / transformed[1, 1],
            ]
        )
    return errors, metadata


def _validate_geometry(
    transforms: np.ndarray, baseline: float, config: Method3Config
) -> tuple[bool, str, float, float]:
    if not np.isfinite(transforms).all():
        return False, "non-finite camera prediction", float("inf"), float("inf")
    if not np.isfinite(baseline) or baseline < config.minimum_stereo_baseline:
        return False, "degenerate predicted E1 stereo baseline", float("inf"), float("inf")
    qualities = [rotation_quality(T[:3, :3]) for T in transforms]
    max_orthogonality = max(value[0] for value in qualities)
    max_det_error = max(abs(value[1] - 1.0) for value in qualities)
    determinants = [value[1] for value in qualities]
    valid = (
        max_orthogonality <= config.maximum_rotation_orthogonality_error
        and min(determinants) >= config.minimum_rotation_determinant
        and max(determinants) <= config.maximum_rotation_determinant
    )
    reason = "" if valid else "invalid predicted rotation matrix"
    return valid, reason, max_orthogonality, max_det_error


def run_method3_sequence(
    *,
    inputs: VGGTSequenceInputs,
    predictor: VGGTCameraPredictor,
    config: Method3Config,
    frame_indices: list[int],
) -> Method3SequenceResult:
    if not frame_indices:
        raise ValueError("no frames selected")
    reference_paths = _current_paths(inputs, 0)
    known_intrinsics = _known_intrinsics(inputs)
    diagnostics: list[Method3FrameDiagnostic] = []
    raw_extrinsics: list[np.ndarray] = []
    raw_intrinsics: list[np.ndarray] = []
    scale_factors: list[float] = []
    scale_reference_e1_baselines: list[float] = []
    e1_baselines: list[float] = []
    e2_baselines: list[float] = []
    cross_current: list[np.ndarray] = []
    cross_reference: list[np.ndarray] = []
    direct_relative: list[np.ndarray] = []
    valid_flags: list[bool] = []
    e1_stereo_rotations: list[np.ndarray] = []
    e2_stereo_rotations: list[np.ndarray] = []
    previous_cross: np.ndarray | None = None

    for index in frame_indices:
        current_paths = _current_paths(inputs, index)
        image_paths = (
            current_paths
            if config.variant == "3a"
            else reference_paths + current_paths
        )
        prediction = predictor.predict(image_paths, config.preprocess_mode)
        transforms = np.stack(
            [homogeneous(item) for item in prediction.extrinsics_cam_from_world]
        )
        if config.variant == "3a":
            ref_e1_l, ref_e1_r, ref_e2_l, ref_e2_r = 0, 1, 2, 3
            cur_e1_l, cur_e1_r, cur_e2_l, cur_e2_r = 0, 1, 2, 3
            predicted_current_intrinsics = prediction.intrinsics
        else:
            ref_e1_l, ref_e1_r, ref_e2_l, ref_e2_r = 0, 1, 2, 3
            cur_e1_l, cur_e1_r, cur_e2_l, cur_e2_r = 4, 5, 6, 7
            predicted_current_intrinsics = prediction.intrinsics[4:8]

        scale_reference_e1_baseline = float(
            np.linalg.norm(
                camera_center_world(transforms[ref_e1_l])
                - camera_center_world(transforms[ref_e1_r])
            )
        )
        e1_baseline = float(
            np.linalg.norm(
                camera_center_world(transforms[cur_e1_l])
                - camera_center_world(transforms[cur_e1_r])
            )
        )
        e2_baseline = float(
            np.linalg.norm(
                camera_center_world(transforms[cur_e2_l])
                - camera_center_world(transforms[cur_e2_r])
            )
        )
        geometry_valid, failure_reason, max_orth, max_det_error = _validate_geometry(
            transforms, scale_reference_e1_baseline, config
        )
        scale = 1.0 / scale_reference_e1_baseline if geometry_valid else float("nan")
        scaled = (
            scaled_camera_extrinsics(transforms, scale)
            if geometry_valid
            else transforms.copy()
        )
        T_cross_ref = relative_camera_transform(
            scaled[ref_e2_l], scaled[ref_e1_l]
        )
        T_cross_cur_predicted = relative_camera_transform(
            scaled[cur_e2_l], scaled[cur_e1_l]
        )
        T_cross_cur = T_cross_cur_predicted
        used_previous = False
        if not geometry_valid:
            if config.failure_policy == "error" or previous_cross is None:
                raise RuntimeError(
                    f"{inputs.sequence_name} frame {inputs.frame_ids[index]}: {failure_reason}"
                )
            T_cross_cur = previous_cross.copy()
            used_previous = True
        else:
            previous_cross = T_cross_cur.copy()

        # For the duplicated t=0 Method-3B call, retain the independently
        # predicted current cross pose in raw outputs/diagnostics but normalize
        # the challenge pose with the normally estimated reference cross pose
        # itself.  This guarantees the mandatory frame-zero identity without
        # inserting identity before camera estimation.
        current_for_relative = T_cross_cur
        if config.variant == "3b" and index == 0:
            current_for_relative = T_cross_ref

        if config.variant == "3b":
            T_relative = challenge_relative_from_cross(
                T_cross_ref, current_for_relative
            )
            T_direct = scaled[ref_e2_l] @ invert_se3(scaled[cur_e2_l])
        else:
            T_relative = np.full((4, 4), np.nan, dtype=np.float64)
            T_direct = np.full((4, 4), np.nan, dtype=np.float64)

        T_e1_stereo = relative_camera_transform(
            scaled[cur_e1_r], scaled[cur_e1_l]
        )
        T_e2_stereo = relative_camera_transform(
            scaled[cur_e2_r], scaled[cur_e2_l]
        )
        focal_errors, metadata = _focal_errors(
            predicted_current_intrinsics,
            known_intrinsics,
            current_paths,
            config.preprocess_mode,
        )
        duplicate_rotation = float("nan")
        duplicate_translation = float("nan")
        if config.variant == "3b" and index == 0:
            disagreement = T_cross_ref @ invert_se3(T_cross_cur_predicted)
            duplicate_rotation = rotation_angle_degrees(disagreement[:3, :3])
            duplicate_translation = float(np.linalg.norm(disagreement[:3, 3]))

        diagnostics.append(
            Method3FrameDiagnostic(
                frame_id=inputs.frame_ids[index],
                valid=geometry_valid,
                failure_reason=failure_reason,
                inference_seconds=prediction.inference_seconds,
                peak_gpu_memory_bytes=prediction.peak_gpu_memory_bytes,
                original_resolution=[
                    metadata[0].original_width,
                    metadata[0].original_height,
                ],
                input_resolution=[prediction.input_shape[1], prediction.input_shape[0]],
                scale_reference_e1_baseline_raw=scale_reference_e1_baseline,
                e1_stereo_baseline_raw=e1_baseline,
                e2_stereo_baseline_raw=e2_baseline,
                scale_factor=scale,
                cross_translation_magnitude_scaled=float(
                    np.linalg.norm(T_cross_cur[:3, 3])
                ),
                e1_stereo_rotation_deg=rotation_angle_degrees(T_e1_stereo[:3, :3]),
                e2_stereo_rotation_deg=rotation_angle_degrees(T_e2_stereo[:3, :3]),
                focal_relative_errors=focal_errors,
                rotation_orthogonality_error_max=max_orth,
                rotation_determinant_error_max=max_det_error,
                used_previous_fallback=used_previous,
                duplicate_t0_cross_rotation_disagreement_deg=duplicate_rotation,
                duplicate_t0_cross_translation_disagreement=duplicate_translation,
            )
        )
        raw_extrinsics.append(transforms)
        raw_intrinsics.append(prediction.intrinsics)
        scale_factors.append(scale)
        scale_reference_e1_baselines.append(scale_reference_e1_baseline)
        e1_baselines.append(e1_baseline)
        e2_baselines.append(e2_baseline)
        cross_current.append(T_cross_cur)
        cross_reference.append(T_cross_ref)
        direct_relative.append(T_direct)
        valid_flags.append(geometry_valid)
        e1_stereo_rotations.append(T_e1_stereo[:3, :3])
        e2_stereo_rotations.append(T_e2_stereo[:3, :3])

    cross_current_array = np.stack(cross_current)
    if config.variant == "3a":
        cross_zero = cross_current_array[0]
        relative_array = np.stack(
            [challenge_relative_from_cross(cross_zero, pose) for pose in cross_current]
        )
        cross_reference_array = np.repeat(cross_zero[None], len(cross_current), axis=0)
    else:
        relative_array = np.stack(
            [
                challenge_relative_from_cross(reference, reference if frame_indices[i] == 0 else current)
                for i, (reference, current) in enumerate(zip(cross_reference, cross_current))
            ]
        )
        cross_reference_array = np.stack(cross_reference)

    frame_zero_error = float(np.max(np.abs(relative_array[0] - np.eye(4))))
    # VGGT returns float32 camera matrices.  R @ R.T therefore commonly leaves
    # a residual around 1e-7 even after conversion to float64.  Check that this
    # is only numerical round-off, retain the measured pre-correction error in
    # diagnostics, and write the mandatory frame-zero pose as exact identity.
    if frame_zero_error > 1.0e-5:
        raise RuntimeError(f"frame-zero normalization error: {frame_zero_error:.3e}")
    relative_array[0] = np.eye(4, dtype=np.float64)
    rows = [
        transform_to_pose_row(inputs.frame_ids[index], relative_array[position])
        for position, index in enumerate(frame_indices)
    ]
    summary = summarize_diagnostics(diagnostics)
    summary.update(
        {
            "sequence": inputs.sequence_name,
            "variant": config.variant,
            "frame_zero_identity_max_error": frame_zero_error,
            "e1_stereo_rotation_variation": stereo_rotation_variation(
                np.stack(e1_stereo_rotations)
            ),
            "e2_stereo_rotation_variation": stereo_rotation_variation(
                np.stack(e2_stereo_rotations)
            ),
        }
    )
    return Method3SequenceResult(
        rows=rows,
        frame_ids=np.asarray([inputs.frame_ids[index] for index in frame_indices]),
        raw_extrinsics_cam_from_world=np.stack(raw_extrinsics),
        raw_intrinsics_processed=np.stack(raw_intrinsics),
        scale_factors=np.asarray(scale_factors),
        scale_reference_e1_baselines_raw=np.asarray(scale_reference_e1_baselines),
        e1_stereo_baselines_raw=np.asarray(e1_baselines),
        e2_stereo_baselines_raw=np.asarray(e2_baselines),
        T_cross_current_scaled=cross_current_array,
        T_cross_reference_scaled=cross_reference_array,
        T_initial_from_current=relative_array,
        T_direct_e20_from_e2t=np.stack(direct_relative),
        prediction_valid=np.asarray(valid_flags, dtype=bool),
        diagnostics=diagnostics,
        summary=summary,
    )
