#!/usr/bin/env python3
"""Sparse XFeat versus ALIKED+LightGlue correspondence-quality benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import XFeatConfig
from xfeat_wrapper import XFeatSparseMatcher
from imcpe.methods.cross_stereo_pnp.data import load_method2_inputs
from imcpe.methods.stereo_vo.features import SharedFeatureMatcher


def _sync(device: str) -> None:
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()


def _normalize(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)


def _essential_candidates(matrix: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 3):
        return [matrix]
    if matrix.ndim == 2 and matrix.shape[1] == 3 and matrix.shape[0] % 3 == 0:
        return [matrix[index:index + 3] for index in range(0, matrix.shape[0], 3)]
    return []


def _sampson(E: np.ndarray, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    ones = np.ones((points0.shape[0], 1), dtype=np.float64)
    x0 = np.concatenate([points0, ones], axis=1)
    x1 = np.concatenate([points1, ones], axis=1)
    Ex0 = (E @ x0.T).T
    Etx1 = (E.T @ x1.T).T
    numerator = np.sum(x1 * Ex0, axis=1) ** 2
    denominator = (
        Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2
        + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    )
    return numerator / np.maximum(denominator, np.finfo(float).eps)


def _geometry(
    points0: np.ndarray,
    points1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if points0.shape[0] < 8:
        return {
            "success": False,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "cheirality_count": 0,
            "sampson_median_normalized": None,
        }
    normalized0 = _normalize(points0, K0)
    normalized1 = _normalize(points1, K1)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    method_name = "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "RANSAC_fallback"
    try:
        matrix, mask = cv2.findEssentialMat(
            normalized0,
            normalized1,
            cameraMatrix=np.eye(3, dtype=np.float64),
            method=method,
            prob=0.999,
            threshold=threshold,
        )
    except cv2.error:
        matrix, mask = None, None
    if matrix is None or mask is None:
        return {
            "success": False,
            "estimator": method_name,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "cheirality_count": 0,
            "sampson_median_normalized": None,
        }
    original_mask = mask.reshape(-1).astype(bool)
    best: tuple[int, np.ndarray] | None = None
    for candidate in _essential_candidates(matrix):
        recover_mask = original_mask.astype(np.uint8).reshape(-1, 1)
        try:
            count, _, _, _ = cv2.recoverPose(
                candidate,
                normalized0,
                normalized1,
                np.eye(3, dtype=np.float64),
                mask=recover_mask,
            )
        except cv2.error:
            continue
        if best is None or int(count) > best[0]:
            best = int(count), candidate
    if best is None:
        return {
            "success": False,
            "estimator": method_name,
            "inlier_count": int(np.count_nonzero(original_mask)),
            "inlier_ratio": float(np.mean(original_mask)),
            "cheirality_count": 0,
            "sampson_median_normalized": None,
        }
    errors = _sampson(best[1], normalized0, normalized1)
    inlier_errors = errors[original_mask]
    return {
        "success": True,
        "estimator": method_name,
        "inlier_count": int(np.count_nonzero(original_mask)),
        "inlier_ratio": float(np.mean(original_mask)),
        "cheirality_count": best[0],
        "sampson_median_normalized": (
            float(np.median(inlier_errors)) if inlier_errors.size else None
        ),
    }


def _coverage(points: np.ndarray, image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    height, width = image.shape
    if points.shape[0] == 0:
        return {"occupied_cells": 0, "occupied_fraction": 0.0, "covariance_ratio": 0.0}
    x = np.clip(points[:, 0], 0.0, np.nextafter(float(width), 0.0))
    y = np.clip(points[:, 1], 0.0, np.nextafter(float(height), 0.0))
    cell = (y * 6 / height).astype(int) * 6 + (x * 6 / width).astype(int)
    occupied = int(np.unique(cell).size)
    normalized = points / np.array([width, height], dtype=np.float64)
    eigenvalues = (
        np.maximum(np.linalg.eigvalsh(np.cov(normalized, rowvar=False)), 0.0)
        if points.shape[0] >= 2 else np.zeros(2)
    )
    return {
        "occupied_cells": occupied,
        "occupied_fraction": float(occupied / 36.0),
        "covariance_ratio": float(
            eigenvalues[0] / max(eigenvalues[-1], np.finfo(float).eps)
        ),
    }


def _overlap(
    aliked0: np.ndarray,
    aliked1: np.ndarray,
    xfeat0: np.ndarray,
    xfeat1: np.ndarray,
    radius_px: float = 3.0,
) -> dict[str, float]:
    def fraction(reference0, reference1, query0, query1) -> float:
        if not len(reference0) or not len(query0):
            return 0.0
        tree = cKDTree(reference0)
        candidates = tree.query_ball_point(query0, r=radius_px)
        matched = 0
        for target, indices in zip(query1, candidates, strict=True):
            if indices and np.any(
                np.linalg.norm(reference1[np.asarray(indices)] - target, axis=1)
                <= radius_px
            ):
                matched += 1
        return float(matched / len(query0))
    return {
        "radius_px": radius_px,
        "xfeat_fraction_overlapping_aliked": fraction(
            aliked0, aliked1, xfeat0, xfeat1
        ),
        "aliked_fraction_overlapping_xfeat": fraction(
            xfeat0, xfeat1, aliked0, aliked1
        ),
    }


def _draw(
    image0_path: Path,
    image1_path: Path,
    points0: np.ndarray,
    points1: np.ndarray,
    label: str,
    output: Path,
) -> None:
    image0 = cv2.imread(str(image0_path), cv2.IMREAD_COLOR)
    image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    if image0 is None or image1 is None:
        return
    height, width = max(image0.shape[0], image1.shape[0]), image0.shape[1]
    canvas = np.zeros((height, width + image1.shape[1], 3), dtype=np.uint8)
    canvas[: image0.shape[0], :width] = image0
    canvas[: image1.shape[0], width:] = image1
    indices = (
        np.linspace(0, len(points0) - 1, 160, dtype=int)
        if len(points0) > 160 else np.arange(len(points0))
    )
    for index in indices:
        p0 = np.rint(points0[index]).astype(int)
        p1 = np.rint(points1[index]).astype(int)
        left = int(p0[0]), int(p0[1])
        right = int(p1[0] + width), int(p1[1])
        cv2.line(canvas, left, right, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, f"{label}: {len(points0)} matches", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _sample_frames(data_root: Path, split: str, samples_per_session: int):
    sequence_dirs = sorted(path for path in (data_root / split).iterdir() if path.is_dir())
    grouped: dict[str, list[tuple[Path, int]]] = {}
    loaded: dict[Path, Any] = {}
    for sequence_dir in sequence_dirs:
        inputs = load_method2_inputs(sequence_dir)
        loaded[sequence_dir] = inputs
        session = "_".join(sequence_dir.name.split("_")[:2])
        grouped.setdefault(session, []).extend(
            (sequence_dir, index) for index in range(len(inputs.frame_ids))
        )
    selected = []
    for session, frames in sorted(grouped.items()):
        count = min(samples_per_session, len(frames))
        indices = np.unique(np.linspace(0, len(frames) - 1, count, dtype=int))
        selected.extend((session, *frames[int(index)]) for index in indices)
    return selected, loaded


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in sorted({record["pair_type"] for record in records}):
        result[pair] = {}
        for frontend in ("aliked_lightglue", "xfeat_sparse"):
            subset = [
                record["frontends"][frontend]
                for record in records if record["pair_type"] == pair
            ]
            def values(key: str) -> np.ndarray:
                array = np.asarray([item[key] for item in subset], dtype=np.float64)
                return array[np.isfinite(array)]
            result[pair][frontend] = {
                "num_pairs": len(subset),
                "matches_median": float(np.median(values("num_matches"))),
                "essential_inliers_median": float(np.median(values("inlier_count"))),
                "essential_inlier_ratio_median": float(np.median(values("inlier_ratio"))),
                "sampson_median_normalized": float(np.nanmedian(values("sampson_error"))),
                "occupied_6x6_cells_median": float(np.median(values("occupied_cells"))),
                "covariance_ratio_median": float(np.median(values("covariance_ratio"))),
                "extract_ms_per_image_median": float(np.median(values("extract_ms_per_image"))),
                "match_ms_per_pair_median": float(np.median(values("match_ms_per_pair"))),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--samples-per-session", type=int, default=8)
    parser.add_argument("--xfeat-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--min-cosine", type=float, default=-1.0)
    parser.add_argument("--essential-threshold", type=float, default=1e-3)
    parser.add_argument("--visualizations-per-pair", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    aliked = SharedFeatureMatcher(device=args.device, max_num_keypoints=args.max_features)
    xfeat = XFeatSparseMatcher(
        xfeat_root=args.xfeat_root,
        weights=args.weights,
        device=args.device,
        config=XFeatConfig(args.max_features, args.min_cosine),
    )
    selected, loaded = _sample_frames(args.data_root, args.split, args.samples_per_session)
    records: list[dict[str, Any]] = []
    visual_counts: dict[str, int] = {}
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for session, sequence_dir, index in selected:
        inputs = loaded[sequence_dir]
        next_index = index + 1 if index + 1 < len(inputs.frame_ids) else index - 1
        pairs = {
            "e1_stereo": (inputs.e1_l_images[index], inputs.e1_r_images[index], inputs.k1_l, inputs.k1_r),
            "e2_stereo": (inputs.e2_l_images[index], inputs.e2_r_images[index], inputs.k2_l, inputs.k2_r),
            "cross_e1l_e2l": (inputs.e1_l_images[index], inputs.e2_l_images[index], inputs.k1_l, inputs.k2_l),
            "temporal_e2l": (inputs.e2_l_images[index], inputs.e2_l_images[next_index], inputs.k2_l, inputs.k2_l),
        }
        matchers = {
            "aliked_lightglue": aliked,
            "xfeat_sparse": xfeat,
        }
        unique_paths = sorted(
            {path for path0, path1, _, _ in pairs.values() for path in (path0, path1)}
        )
        feature_sets: dict[str, dict[Path, dict[str, Any]]] = {}
        extraction_ms_per_image: dict[str, float] = {}
        for name, matcher in matchers.items():
            _sync(args.device)
            extract_start = time.perf_counter()
            feature_sets[name] = {path: matcher.extract(path) for path in unique_paths}
            _sync(args.device)
            extraction_ms_per_image[name] = (
                1000.0 * (time.perf_counter() - extract_start) / len(unique_paths)
            )
        for pair_type, (path0, path1, K0, K1) in pairs.items():
            frontend_results: dict[str, Any] = {}
            matched_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name, matcher in matchers.items():
                features0 = feature_sets[name][path0]
                features1 = feature_sets[name][path1]
                _sync(args.device)
                match_start = time.perf_counter()
                if isinstance(matcher, XFeatSparseMatcher):
                    matched = matcher.match_indices_with_scores(features0, features1)
                    matches, confidence = matched.indices, matched.confidence
                else:
                    matches = matcher.match_indices(features0, features1)
                    confidence = np.empty(0, dtype=np.float64)
                _sync(args.device)
                match_seconds = time.perf_counter() - match_start
                keypoints0, keypoints1 = matcher.keypoints(features0), matcher.keypoints(features1)
                points0, points1 = keypoints0[matches[:, 0]], keypoints1[matches[:, 1]]
                geometry = _geometry(points0, points1, K0, K1, args.essential_threshold)
                coverage = _coverage(points0, path0)
                frontend_results[name] = {
                    "num_features0": int(keypoints0.shape[0]),
                    "num_features1": int(keypoints1.shape[0]),
                    "num_matches": int(matches.shape[0]),
                    "inlier_count": geometry["inlier_count"],
                    "inlier_ratio": geometry["inlier_ratio"],
                    "cheirality_count": geometry["cheirality_count"],
                    "sampson_error": geometry["sampson_median_normalized"] or float("nan"),
                    "essential_estimator": geometry.get("estimator"),
                    "occupied_cells": coverage["occupied_cells"],
                    "occupied_fraction": coverage["occupied_fraction"],
                    "covariance_ratio": coverage["covariance_ratio"],
                    "extract_ms_per_image": extraction_ms_per_image[name],
                    "match_ms_per_pair": 1000.0 * match_seconds,
                    "confidence_median": float(np.median(confidence)) if confidence.size else None,
                    "confidence_p10": float(np.percentile(confidence, 10)) if confidence.size else None,
                }
                matched_points[name] = points0, points1
                if visual_counts.get(pair_type, 0) < args.visualizations_per_pair:
                    _draw(
                        path0, path1, points0, points1,
                        f"{name} {pair_type}",
                        args.output_root / "visualizations" / pair_type
                        / f"{sequence_dir.name}_{inputs.frame_ids[index]:06d}_{name}.png",
                    )
            overlap = _overlap(
                *matched_points["aliked_lightglue"],
                *matched_points["xfeat_sparse"],
            )
            records.append({
                "session": session,
                "sequence": sequence_dir.name,
                "frame_idx": int(inputs.frame_ids[index]),
                "pair_type": pair_type,
                "source": str(path0),
                "target": str(path1),
                "frontends": frontend_results,
                "overlap": overlap,
            })
            visual_counts[pair_type] = visual_counts.get(pair_type, 0) + 1
    result = {
        "split": args.split,
        "samples_per_session": args.samples_per_session,
        "num_sampled_timestamps": len(selected),
        "xfeat": xfeat.metadata(),
        "geometry_unchanged": True,
        "essential_threshold_normalized": args.essential_threshold,
        "aggregate": _aggregate(records),
        "pairs": records,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / (1024.0**3)
            if torch.cuda.is_available() and args.device.startswith("cuda") else 0.0
        ),
        "no_ground_truth_used": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "correspondence_benchmark.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Pair | Frontend | matches | E-inliers | ratio | Sampson | cells | covariance | extract ms | match ms")
    for pair, values in result["aggregate"].items():
        for frontend, row in values.items():
            print(
                f"{pair:18s} | {frontend:18s} | {row['matches_median']:.1f} | "
                f"{row['essential_inliers_median']:.1f} | {row['essential_inlier_ratio_median']:.3f} | "
                f"{row['sampson_median_normalized']:.3e} | {row['occupied_6x6_cells_median']:.1f} | "
                f"{row['covariance_ratio_median']:.3f} | {row['extract_ms_per_image_median']:.2f} | "
                f"{row['match_ms_per_pair_median']:.2f}"
            )
    print(f"Saved benchmark: {output}")


if __name__ == "__main__":
    main()
