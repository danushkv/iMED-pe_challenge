#!/usr/bin/env python3
"""TRAIN-only correspondence benchmark for ALIKED+LightGlue and EfficientLoFTR.

Frames are sampled deterministically and pose.txt is never opened.  EfficientLoFTR
is executed once per pair; confidence thresholds are applied to the cached result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# Permit running this file directly from the repository without requiring an
# editable package install in the separate EfficientLoFTR environment.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for local_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))

from imcpe.methods.stereo_vo.features import SharedFeatureMatcher

from config import EfficientLoFTRConfig
from loftr_wrapper import EfficientLoFTRMatcher


PAIR_TYPES = ("cross_e1l_e2l", "e1_stereo", "e2_stereo", "temporal_e2l")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic TRAIN correspondence benchmark: ALIKED+LG vs EfficientLoFTR."
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--samples-per-session", type=int, default=5)
    p.add_argument("--loftr-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-type", choices=("full", "opt"), default="full")
    p.add_argument("--precision", choices=("fp32", "mp", "fp16"), default="fp32")
    p.add_argument("--max-keypoints", type=int, default=2048)
    p.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=(0.2, 0.5, 0.8),
        help="Thresholds applied after one LoFTR inference per pair.",
    )
    p.add_argument("--essential-threshold", type=float, default=1.0e-3)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def _session_id(sequence_name: str) -> str:
    parts = sequence_name.split("_")
    if len(parts) < 2 or parts[0] != "session":
        raise ValueError(f"Cannot extract session ID from {sequence_name!r}")
    return "_".join(parts[:2])


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(count, length))
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def _discover_pairs(split_root: Path, samples_per_session: int) -> list[dict[str, Any]]:
    sequences = sorted(p for p in split_root.glob("session_*") if p.is_dir())
    by_session: dict[str, list[Path]] = defaultdict(list)
    for sequence in sequences:
        by_session[_session_id(sequence.name)].append(sequence)

    selected: list[dict[str, Any]] = []
    for session, session_sequences in sorted(by_session.items()):
        candidates: list[tuple[Path, int, list[Path]]] = []
        for sequence in session_sequences:
            frames = sorted((sequence / "endoscope1" / "L").glob("*.png"))
            for index, frame in enumerate(frames):
                candidates.append((sequence, index, frames))
        for chosen in _uniform_indices(len(candidates), samples_per_session):
            sequence, index, frames = candidates[chosen]
            frame = frames[index]
            name = frame.name
            paths = {
                "e1l": frame,
                "e1r": sequence / "endoscope1" / "R" / name,
                "e2l": sequence / "endoscope2" / "L" / name,
                "e2r": sequence / "endoscope2" / "R" / name,
            }
            if not all(path.is_file() for path in paths.values()):
                continue
            next_e2l = None
            if index + 1 < len(frames):
                candidate = sequence / "endoscope2" / "L" / frames[index + 1].name
                if candidate.is_file():
                    next_e2l = candidate
            selected.append(
                {
                    "session": session,
                    "sequence": sequence.name,
                    "frame": name,
                    "paths": paths,
                    "next_e2l": next_e2l,
                }
            )
    return selected


def _find_intrinsics(image: Path) -> np.ndarray:
    camera_dir = image.parent
    candidates = (
        camera_dir / "K.txt",
        camera_dir.parent / "K.txt",
        # iMED-PE stores the camera intrinsics at sequence level.
        camera_dir.parent.parent / "K.txt",
        camera_dir / "intrinsics.txt",
        camera_dir.parent / "intrinsics.txt",
        camera_dir.parent.parent / "intrinsics.txt",
    )
    for path in candidates:
        if not path.is_file():
            continue
        values = np.loadtxt(path, dtype=np.float64)
        if values.size == 9:
            return values.reshape(3, 3)
        # Sequence-level K.txt contains one flattened 3x3 matrix per camera,
        # ordered E1-L, E1-R, E2-L, E2-R.  Also accept vertically stacked 3x3
        # blocks, which is an equivalent common serialization.
        endoscope = camera_dir.parent.name.lower()
        side = camera_dir.name.lower()
        camera_index = {
            ("endoscope1", "l"): 0,
            ("endoscope1", "r"): 1,
            ("endoscope2", "l"): 2,
            ("endoscope2", "r"): 3,
        }.get((endoscope, side))
        if camera_index is None:
            raise ValueError(f"Cannot identify camera from image path: {image}")
        if values.ndim == 2 and values.shape[1] == 9 and camera_index < values.shape[0]:
            return values[camera_index].reshape(3, 3)
        if values.ndim == 2 and values.shape[1] == 3 and values.shape[0] % 3 == 0:
            matrices = values.reshape(-1, 3, 3)
            if camera_index < len(matrices):
                return matrices[camera_index]
        if values.size % 9 == 0:
            matrices = values.reshape(-1, 3, 3)
            if camera_index < len(matrices):
                return matrices[camera_index]
    raise FileNotFoundError(
        f"Could not locate a 3x3 K.txt for {image}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _normalize(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points, np.float64).reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)


def _sampson(E: np.ndarray, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    x0 = np.column_stack((points0, np.ones(len(points0))))
    x1 = np.column_stack((points1, np.ones(len(points1))))
    Ex0 = (E @ x0.T).T
    Etx1 = (E.T @ x1.T).T
    numerator = np.sum(x1 * Ex0, axis=1) ** 2
    denominator = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return numerator / np.maximum(denominator, 1.0e-15)


def _coverage(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    if len(points) == 0:
        return {"occupied_fraction_6x6": 0.0, "covariance_ratio": 0.0}
    cols = np.clip((points[:, 0] / max(width, 1) * 6).astype(int), 0, 5)
    rows = np.clip((points[:, 1] / max(height, 1) * 6).astype(int), 0, 5)
    occupied = len(set(zip(rows.tolist(), cols.tolist()))) / 36.0
    covariance = np.cov(points.T) if len(points) >= 2 else np.zeros((2, 2))
    eigenvalues = np.linalg.eigvalsh(covariance)
    ratio = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
    return {"occupied_fraction_6x6": float(occupied), "covariance_ratio": ratio}


def _geometry(
    points0: np.ndarray,
    points1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "matches": int(len(points0)),
        "essential_inliers": 0,
        "essential_inlier_ratio": 0.0,
        "sampson_median": math.nan,
        "sampson_p90": math.nan,
    }
    if len(points0) < 8:
        return result
    n0, n1 = _normalize(points0, K0), _normalize(points1, K1)
    E, mask = cv2.findEssentialMat(
        n0,
        n1,
        np.eye(3),
        method=cv2.USAC_MAGSAC,
        prob=0.999,
        threshold=threshold,
    )
    if E is None or mask is None:
        return result
    E = np.asarray(E, np.float64).reshape(-1, 3, 3)[0]
    inliers = mask.reshape(-1).astype(bool)
    errors = _sampson(E, n0, n1)
    result.update(
        essential_inliers=int(inliers.sum()),
        essential_inlier_ratio=float(inliers.mean()),
        sampson_median=float(np.median(errors[inliers])) if inliers.any() else math.nan,
        sampson_p90=float(np.percentile(errors[inliers], 90)) if inliers.any() else math.nan,
    )
    return result


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["pair_type"], record["frontend"], record["configuration"])].append(record)
    output: dict[str, Any] = {}
    metrics = (
        "matches",
        "essential_inliers",
        "essential_inlier_ratio",
        "sampson_median",
        "sampson_p90",
        "occupied_fraction_6x6",
        "covariance_ratio",
        "inference_ms",
    )
    for (pair_type, frontend, configuration), values in sorted(grouped.items()):
        key = f"{pair_type}/{frontend}/{configuration}"
        output[key] = {"num_pairs": len(values)}
        for metric in metrics:
            array = np.asarray([item[metric] for item in values], dtype=np.float64)
            finite = array[np.isfinite(array)]
            output[key][f"{metric}_median"] = float(np.median(finite)) if len(finite) else math.nan
            output[key][f"{metric}_p90"] = float(np.percentile(finite, 90)) if len(finite) else math.nan
    return output


def main() -> None:
    args = parse_args()
    if args.split != "train":
        print("WARNING: confidence selection must be performed on TRAIN only.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples = _discover_pairs(args.data_root / args.split, args.samples_per_session)
    if not samples:
        raise RuntimeError(f"No image pairs discovered beneath {args.data_root / args.split}")

    aliked = SharedFeatureMatcher(max_num_keypoints=args.max_keypoints, device=args.device)
    loftr = EfficientLoFTRMatcher(
        loftr_root=args.loftr_root,
        checkpoint=args.checkpoint,
        device=args.device,
        config=EfficientLoFTRConfig(model_type=args.model_type, precision=args.precision),
    )
    records: list[dict[str, Any]] = []
    pair_map = {
        "cross_e1l_e2l": ("e1l", "e2l"),
        "e1_stereo": ("e1l", "e1r"),
        "e2_stereo": ("e2l", "e2r"),
    }

    for sample in samples:
        pairs = [(name, sample["paths"][a], sample["paths"][b]) for name, (a, b) in pair_map.items()]
        if sample["next_e2l"] is not None:
            pairs.append(("temporal_e2l", sample["paths"]["e2l"], sample["next_e2l"]))
        for pair_type, image0, image1 in pairs:
            K0, K1 = _find_intrinsics(image0), _find_intrinsics(image1)
            shape0 = cv2.imread(str(image0), cv2.IMREAD_GRAYSCALE).shape

            start = time.perf_counter()
            features0, features1 = aliked.extract(str(image0)), aliked.extract(str(image1))
            matches = np.asarray(aliked.match_indices(features0, features1), dtype=np.int64)
            aliked_ms = (time.perf_counter() - start) * 1000.0
            keypoints0, keypoints1 = aliked.keypoints(features0), aliked.keypoints(features1)
            points0 = np.asarray(keypoints0[matches[:, 0]], np.float64)
            points1 = np.asarray(keypoints1[matches[:, 1]], np.float64)
            record = {
                "session": sample["session"], "sequence": sample["sequence"], "frame": sample["frame"],
                "pair_type": pair_type, "frontend": "aliked_lightglue", "configuration": "default",
                "inference_ms": aliked_ms,
            }
            record.update(_geometry(points0, points1, K0, K1, args.essential_threshold))
            record.update(_coverage(points0, shape0[1], shape0[0]))
            records.append(record)

            prediction = loftr.match(image0, image1)
            for confidence_threshold in args.confidence_thresholds:
                keep = prediction.confidence >= confidence_threshold
                points0 = prediction.points0[keep]
                points1 = prediction.points1[keep]
                record = {
                    "session": sample["session"], "sequence": sample["sequence"], "frame": sample["frame"],
                    "pair_type": pair_type, "frontend": "efficient_loftr",
                    "configuration": f"confidence_ge_{confidence_threshold:g}",
                    "confidence_threshold": confidence_threshold,
                    "confidence_median": float(np.median(prediction.confidence[keep])) if keep.any() else math.nan,
                    "inference_ms": prediction.inference_seconds * 1000.0,
                }
                record.update(_geometry(points0, points1, K0, K1, args.essential_threshold))
                record.update(_coverage(points0, shape0[1], shape0[0]))
                records.append(record)

    payload = {
        "protocol": {
            "split": args.split,
            "samples_per_session": args.samples_per_session,
            "sampling": "deterministic uniform by physical session; no pose.txt access",
            "confidence_thresholds": args.confidence_thresholds,
            "essential_estimator": "cv2.USAC_MAGSAC on normalized coordinates",
            "essential_threshold": args.essential_threshold,
        },
        "loftr": loftr.metadata(),
        "aggregate": _aggregate(records),
        "records": records,
    }
    output = args.output_root / "correspondence_benchmark.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=True))

    print("Pair | Frontend/config | matches | E-inliers | ratio | Sampson | cells | covariance | ms/pair")
    for key, value in payload["aggregate"].items():
        print(
            f"{key:58s} | {value['matches_median']:.0f} | {value['essential_inliers_median']:.0f} | "
            f"{value['essential_inlier_ratio_median']:.3f} | {value['sampson_median_median']:.3e} | "
            f"{value['occupied_fraction_6x6_median']:.2f} | {value['covariance_ratio_median']:.3f} | "
            f"{value['inference_ms_median']:.1f}"
        )
    print(f"Saved benchmark: {output}")


if __name__ == "__main__":
    main()
