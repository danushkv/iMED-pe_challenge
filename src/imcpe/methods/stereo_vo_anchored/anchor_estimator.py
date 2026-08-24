from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from imcpe.matcher import ALikeLightGlueMatcher

from .config import AnchorEstimationConfig, AnchorThresholds


@dataclass(frozen=True)
class AnchorInputs:
    sequence_name: str
    frame_ids: list[int]
    K_e1_left: np.ndarray
    K_e2_left: np.ndarray
    e1_left_images: list[Path]
    e2_left_images: list[Path]


@dataclass(frozen=True)
class CrossCameraAnchorCache:
    sequence_name: str
    frame_ids: np.ndarray
    R_cross: np.ndarray
    t_cross_direction: np.ndarray
    R_anchor_rel: np.ndarray
    success: np.ndarray
    num_matches: np.ndarray
    num_essential_inliers: np.ndarray
    essential_inlier_ratio: np.ndarray
    cheirality_count: np.ndarray
    cheirality_fraction: np.ndarray
    median_sampson_error_normalized: np.ndarray
    essential_method: str

    def confidence_valid(self, thresholds: AnchorThresholds) -> np.ndarray:
        return (
            self.success
            & (self.num_matches >= thresholds.min_matches)
            & (self.num_essential_inliers >= thresholds.min_essential_inliers)
            & (self.essential_inlier_ratio >= thresholds.min_inlier_ratio)
            & (self.cheirality_fraction >= thresholds.min_cheirality_fraction)
            & np.isfinite(self.median_sampson_error_normalized)
            & (
                self.median_sampson_error_normalized
                <= thresholds.max_median_sampson_error_normalized
            )
            & np.isfinite(self.R_anchor_rel).all(axis=(1, 2))
        )


@dataclass(frozen=True)
class _PoseEstimate:
    R: np.ndarray
    t_direction: np.ndarray
    essential_mask: np.ndarray
    cheirality_mask: np.ndarray
    median_sampson_error: float


def _read_k_txt(path: Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    matrices: dict[str, np.ndarray] = {}
    key: str | None = None
    rows: list[list[float]] = []
    for line in lines:
        if line.startswith("#"):
            payload = line.lstrip("#").strip()
            key = payload.split()[0] if payload else None
            rows = []
            continue
        if key is None:
            continue
        rows.append([float(value) for value in line.split()])
        if len(rows) == 3:
            matrices[key] = np.asarray(rows, dtype=np.float64)
            key = None
            rows = []
    for required in ("K1_L", "K2_L"):
        if required not in matrices:
            raise RuntimeError(f"{path} is missing {required}")
    return matrices


def _image_map(path: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for image_path in sorted(path.glob("frame_*.png")):
        mapping[int(image_path.stem.split("_")[1])] = image_path
    return mapping


def load_anchor_inputs(sequence_dir: Path) -> AnchorInputs:
    """Load only images and intrinsics; never read pose.txt."""

    intrinsics = _read_k_txt(sequence_dir / "K.txt")
    e1 = _image_map(sequence_dir / "endoscope1" / "L")
    e2 = _image_map(sequence_dir / "endoscope2" / "L")
    if not e1 or not e2:
        raise RuntimeError(f"Missing cross-camera image stream in {sequence_dir}")
    if set(e1) != set(e2):
        raise RuntimeError(f"E1-L/E2-L frame mismatch in {sequence_dir}")
    frame_ids = sorted(e1)
    return AnchorInputs(
        sequence_name=sequence_dir.name,
        frame_ids=frame_ids,
        K_e1_left=intrinsics["K1_L"],
        K_e2_left=intrinsics["K2_L"],
        e1_left_images=[e1[frame_id] for frame_id in frame_ids],
        e2_left_images=[e2[frame_id] for frame_id in frame_ids],
    )


def _normalise(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)


def _essential_candidates(E: np.ndarray) -> list[np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return [E[index:index + 3] for index in range(0, E.shape[0], 3)]
    raise RuntimeError(f"Unexpected essential matrix shape: {E.shape}")


def _sampson_errors(E: np.ndarray, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    ones = np.ones((points0.shape[0], 1), dtype=np.float64)
    x0 = np.concatenate([points0, ones], axis=1)
    x1 = np.concatenate([points1, ones], axis=1)
    E_x0 = (E @ x0.T).T
    Et_x1 = (E.T @ x1.T).T
    numerator = np.abs(np.sum(x1 * E_x0, axis=1))
    denominator = np.sqrt(
        E_x0[:, 0] ** 2
        + E_x0[:, 1] ** 2
        + Et_x1[:, 0] ** 2
        + Et_x1[:, 1] ** 2
    )
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)


def _estimate_pose(
    points_e1: np.ndarray,
    points_e2: np.ndarray,
    K_e1: np.ndarray,
    K_e2: np.ndarray,
    config: AnchorEstimationConfig,
) -> tuple[_PoseEstimate, str]:
    points0 = _normalise(points_e1, K_e1)
    points1 = _normalise(points_e2, K_e2)
    if config.prefer_usac_magsac and hasattr(cv2, "USAC_MAGSAC"):
        method = cv2.USAC_MAGSAC
        method_name = "USAC_MAGSAC"
    else:
        method = cv2.RANSAC
        method_name = "RANSAC"
    E_all, mask = cv2.findEssentialMat(
        points0,
        points1,
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=method,
        prob=config.ransac_probability,
        threshold=config.ransac_threshold_normalized,
    )
    if E_all is None or mask is None:
        raise RuntimeError("findEssentialMat failed")
    essential_mask = mask.reshape(-1).astype(bool)

    best: _PoseEstimate | None = None
    best_cheirality = -1
    for E in _essential_candidates(E_all):
        recover_mask = essential_mask.astype(np.uint8).reshape(-1, 1)
        count, R, t, pose_mask = cv2.recoverPose(
            E,
            points0,
            points1,
            np.eye(3, dtype=np.float64),
            mask=recover_mask,
        )
        if int(count) <= best_cheirality:
            continue
        t_direction = np.asarray(t, dtype=np.float64).reshape(3)
        t_norm = float(np.linalg.norm(t_direction))
        if t_norm == 0.0:
            continue
        sampson = _sampson_errors(E, points0, points1)
        inlier_sampson = sampson[essential_mask]
        best = _PoseEstimate(
            R=np.asarray(R, dtype=np.float64),
            t_direction=t_direction / t_norm,
            essential_mask=essential_mask,
            cheirality_mask=pose_mask.reshape(-1).astype(bool),
            median_sampson_error=(
                float(np.median(inlier_sampson)) if inlier_sampson.size else float("nan")
            ),
        )
        best_cheirality = int(count)
    if best is None:
        raise RuntimeError("recoverPose failed")
    return best, method_name


def generate_sequence_anchor_cache(
    sequence_dir: Path,
    matcher: ALikeLightGlueMatcher,
    config: AnchorEstimationConfig = AnchorEstimationConfig(),
) -> CrossCameraAnchorCache:
    inputs = load_anchor_inputs(sequence_dir)
    count = len(inputs.frame_ids)
    R_cross = np.full((count, 3, 3), np.nan, dtype=np.float64)
    t_cross = np.full((count, 3), np.nan, dtype=np.float64)
    success = np.zeros(count, dtype=bool)
    matches = np.zeros(count, dtype=np.int64)
    inliers = np.zeros(count, dtype=np.int64)
    ratios = np.zeros(count, dtype=np.float64)
    cheirality_count = np.zeros(count, dtype=np.int64)
    cheirality_fraction = np.zeros(count, dtype=np.float64)
    sampson = np.full(count, np.nan, dtype=np.float64)
    method_name = "unknown"

    for index, frame_id in enumerate(inputs.frame_ids):
        points_e1, points_e2 = matcher.match(
            inputs.e1_left_images[index], inputs.e2_left_images[index]
        )
        matches[index] = points_e1.shape[0]
        if matches[index] < 8:
            print(
                f"frame {frame_id:06d}: matches={matches[index]} rejected (<8)",
                flush=True,
            )
            continue
        try:
            estimate, method_name = _estimate_pose(
                points_e1,
                points_e2,
                inputs.K_e1_left,
                inputs.K_e2_left,
                config,
            )
        except (cv2.error, RuntimeError) as error:
            print(f"frame {frame_id:06d}: anchor failed: {error}", flush=True)
            continue
        R_cross[index] = estimate.R
        t_cross[index] = estimate.t_direction
        inliers[index] = int(np.count_nonzero(estimate.essential_mask))
        ratios[index] = inliers[index] / max(matches[index], 1)
        cheirality_count[index] = int(np.count_nonzero(estimate.cheirality_mask))
        cheirality_fraction[index] = cheirality_count[index] / max(inliers[index], 1)
        sampson[index] = estimate.median_sampson_error
        success[index] = True
        print(
            f"frame {frame_id:06d}: matches={matches[index]} E_inliers={inliers[index]} "
            f"ratio={ratios[index]:.3f} cheirality={cheirality_fraction[index]:.3f} "
            f"sampson={sampson[index]:.6g}",
            flush=True,
        )

    if not success[0]:
        raise RuntimeError(
            "Frame-zero cross-camera anchor failed; relative rotation cannot be constructed"
        )
    R_anchor_rel = np.full_like(R_cross, np.nan)
    for index in np.flatnonzero(success):
        R_anchor_rel[index] = R_cross[0] @ R_cross[index].T
    if not np.allclose(R_anchor_rel[0], np.eye(3), atol=1e-9, rtol=0.0):
        raise RuntimeError("Frame-zero relative anchor rotation is not identity")

    return CrossCameraAnchorCache(
        sequence_name=inputs.sequence_name,
        frame_ids=np.asarray(inputs.frame_ids, dtype=np.int64),
        R_cross=R_cross,
        t_cross_direction=t_cross,
        R_anchor_rel=R_anchor_rel,
        success=success,
        num_matches=matches,
        num_essential_inliers=inliers,
        essential_inlier_ratio=ratios,
        cheirality_count=cheirality_count,
        cheirality_fraction=cheirality_fraction,
        median_sampson_error_normalized=sampson,
        essential_method=method_name,
    )


def save_anchor_cache(
    cache: CrossCameraAnchorCache,
    npz_path: Path,
    config: AnchorEstimationConfig,
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        sequence_name=np.array(cache.sequence_name),
        frame_ids=cache.frame_ids,
        R_cross=cache.R_cross,
        t_cross_direction=cache.t_cross_direction,
        R_anchor_rel=cache.R_anchor_rel,
        success=cache.success,
        num_matches=cache.num_matches,
        num_essential_inliers=cache.num_essential_inliers,
        essential_inlier_ratio=cache.essential_inlier_ratio,
        cheirality_count=cache.cheirality_count,
        cheirality_fraction=cache.cheirality_fraction,
        median_sampson_error_normalized=cache.median_sampson_error_normalized,
        essential_method=np.array(cache.essential_method),
        estimation_config_json=np.array(json.dumps(asdict(config), sort_keys=True)),
    )
    diagnostics_path = npz_path.with_name(npz_path.stem + "_diagnostics.json")
    diagnostics = {
        "sequence": cache.sequence_name,
        "num_frames": int(cache.frame_ids.size),
        "num_successful": int(np.count_nonzero(cache.success)),
        "success_rate": float(np.mean(cache.success)),
        "essential_method": cache.essential_method,
        "frames": [
            {
                "frame_idx": int(cache.frame_ids[index]),
                "success": bool(cache.success[index]),
                "num_matches": int(cache.num_matches[index]),
                "num_essential_inliers": int(cache.num_essential_inliers[index]),
                "essential_inlier_ratio": float(cache.essential_inlier_ratio[index]),
                "cheirality_count": int(cache.cheirality_count[index]),
                "cheirality_fraction": float(cache.cheirality_fraction[index]),
                "median_sampson_error_normalized": float(
                    cache.median_sampson_error_normalized[index]
                ),
            }
            for index in range(cache.frame_ids.size)
        ],
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")


def load_anchor_cache(path: Path) -> CrossCameraAnchorCache:
    with np.load(path, allow_pickle=False) as archive:
        return CrossCameraAnchorCache(
            sequence_name=str(archive["sequence_name"].item()),
            frame_ids=np.asarray(archive["frame_ids"], dtype=np.int64),
            R_cross=np.asarray(archive["R_cross"], dtype=np.float64),
            t_cross_direction=np.asarray(archive["t_cross_direction"], dtype=np.float64),
            R_anchor_rel=np.asarray(archive["R_anchor_rel"], dtype=np.float64),
            success=np.asarray(archive["success"], dtype=bool),
            num_matches=np.asarray(archive["num_matches"], dtype=np.int64),
            num_essential_inliers=np.asarray(
                archive["num_essential_inliers"], dtype=np.int64
            ),
            essential_inlier_ratio=np.asarray(
                archive["essential_inlier_ratio"], dtype=np.float64
            ),
            cheirality_count=np.asarray(archive["cheirality_count"], dtype=np.int64),
            cheirality_fraction=np.asarray(
                archive["cheirality_fraction"], dtype=np.float64
            ),
            median_sampson_error_normalized=np.asarray(
                archive["median_sampson_error_normalized"], dtype=np.float64
            ),
            essential_method=str(archive["essential_method"].item()),
        )


def derive_anchor_thresholds(
    cache_paths: list[Path],
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> AnchorThresholds:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Invalid confidence quantiles")
    caches = [load_anchor_cache(path) for path in cache_paths]
    if not caches:
        raise RuntimeError("No training anchor caches supplied")

    def collect(name: str) -> np.ndarray:
        values = [getattr(cache, name)[cache.success] for cache in caches]
        return np.concatenate(values) if values else np.empty(0)

    matches = collect("num_matches")
    inliers = collect("num_essential_inliers")
    ratios = collect("essential_inlier_ratio")
    cheirality = collect("cheirality_fraction")
    sampson = collect("median_sampson_error_normalized")
    finite = np.isfinite(sampson)
    if matches.size == 0 or not np.any(finite):
        raise RuntimeError("No successful finite training anchors")
    return AnchorThresholds(
        min_matches=int(np.floor(np.quantile(matches, lower_quantile))),
        min_essential_inliers=int(np.floor(np.quantile(inliers, lower_quantile))),
        min_inlier_ratio=float(np.quantile(ratios, lower_quantile)),
        min_cheirality_fraction=float(np.quantile(cheirality, lower_quantile)),
        max_median_sampson_error_normalized=float(
            np.quantile(sampson[finite], upper_quantile)
        ),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        num_training_anchors=int(matches.size),
    )

