#!/usr/bin/env python3
"""Cache deterministic sampled RoMa-2A three-view tracks without GT."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from experiments.roma_2a.roma_matcher import ExactSourceRoMaMatcher
from experiments.roma_2a.roma_wrapper import OfficialRoMa
from experiments.roma_2a.source_points import ALIKEDSourceDetector
from imcpe.methods.cross_stereo_pnp.data import (
    load_method2_inputs,
    session_id_from_sequence_name,
)
from imcpe.methods.cross_stereo_pnp.feature_matching import intersect_e1_left_tracks
from imcpe.methods.cross_stereo_pnp.triangulation import load_e1_stereo_calibration


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _session_sequences(split_root: Path, requested: set[str] | None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(item for item in split_root.iterdir() if item.is_dir()):
        session = session_id_from_sequence_name(path.name)
        if requested is not None and session not in requested:
            continue
        grouped.setdefault(session, []).append(path)
    if not grouped:
        raise RuntimeError("No sequences selected")
    return grouped


def _raw_pose_map(raw_root: Path, split: str, sequence: str) -> dict[int, np.ndarray]:
    path = raw_root / split / sequence / "raw_poses.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved RoMa raw poses: {path}")
    with np.load(path, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        transforms = np.asarray(archive["T_E2_from_E1"], dtype=np.float64)
        success = np.asarray(archive["pnp_success"], dtype=bool)
    return {
        int(frame): transform
        for frame, transform, valid in zip(frame_ids, transforms, success, strict=True)
        if valid and np.isfinite(transform).all()
    }


def _uniform_indices(count: int, target: int) -> np.ndarray:
    if count <= target:
        return np.arange(count, dtype=np.int64)
    raw = np.linspace(0, count - 1, target)
    return np.unique(np.rint(raw).astype(np.int64))


def _sample_frames(
    sequences: list[Path], raw_root: Path, split: str, target: int
) -> list[tuple[Path, int, int, float, np.ndarray]]:
    per_sequence = max(2, int(math.ceil(target / max(len(sequences), 1))))
    candidates: list[tuple[Path, int, int, float, np.ndarray]] = []
    for sequence in sequences:
        inputs = load_method2_inputs(sequence)
        raw = _raw_pose_map(raw_root, split, sequence.name)
        available = [
            (index, frame)
            for index, frame in enumerate(inputs.frame_ids)
            if frame in raw
        ]
        for local in _uniform_indices(len(available), per_sequence):
            index, frame = available[int(local)]
            fraction = index / max(len(inputs.frame_ids) - 1, 1)
            candidates.append((sequence, index, frame, fraction, raw[frame]))
    if len(candidates) > target:
        keep = _uniform_indices(len(candidates), target)
        candidates = [candidates[int(index)] for index in keep]
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--roma-prediction-root", type=Path, required=True)
    parser.add_argument("--roma-root", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-source-points", type=int, default=2048)
    parser.add_argument("--certainty-threshold", type=float, default=0.20)
    parser.add_argument("--cycle-threshold-px", type=float, default=2.0)
    parser.add_argument("--frames-per-session", type=int, default=30)
    parser.add_argument("--tracks-per-frame", type=int, default=150)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = (
        {str(value).removeprefix("session_").zfill(3) for value in args.session}
        if args.session
        else None
    )
    grouped = _session_sequences(args.data_root / args.split, requested)
    roma = OfficialRoMa(args.roma_root, args.device, args.torch_home)
    detector = ALIKEDSourceDetector(args.max_source_points, args.device)
    matcher = ExactSourceRoMaMatcher(
        roma=roma,
        detector=detector,
        certainty_threshold=args.certainty_threshold,
        cycle_threshold_px=args.cycle_threshold_px,
        reuse_symmetric_reverse=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    global_summary: dict[str, Any] = {
        "split": args.split,
        "no_ground_truth_used": True,
        "sampling": "uniform over each sequence, then deterministic global trim",
        "frames_per_session": args.frames_per_session,
        "tracks_per_frame": args.tracks_per_frame,
        "certainty_threshold": args.certainty_threshold,
        "cycle_threshold_px": args.cycle_threshold_px,
        "frontend": roma.metadata(),
        "sessions": {},
    }

    for session, sequence_dirs in tqdm(grouped.items(), desc="Cache Method5 observations"):
        sampled = _sample_frames(
            sequence_dirs,
            args.roma_prediction_root,
            args.split,
            args.frames_per_session,
        )
        calibration = load_e1_stereo_calibration(args.calibration_root, session)
        sequences: list[str] = []
        frame_ids: list[int] = []
        fractions: list[float] = []
        offsets = [0]
        all_left: list[np.ndarray] = []
        all_right: list[np.ndarray] = []
        all_e2: list[np.ndarray] = []
        all_confidence: list[np.ndarray] = []
        all_k1l: list[np.ndarray] = []
        all_k1r: list[np.ndarray] = []
        all_k2l: list[np.ndarray] = []
        all_initial: list[np.ndarray] = []
        frame_records: list[dict[str, Any]] = []
        started_session = time.perf_counter()

        for sequence_dir, frame_index, frame_id, fraction, initial in sampled:
            inputs = load_method2_inputs(sequence_dir)
            left = matcher.extract(inputs.e1_l_images[frame_index])
            right = matcher.extract(inputs.e1_r_images[frame_index])
            e2 = matcher.extract(inputs.e2_l_images[frame_index])
            stereo = matcher.match_indices(left, right)
            cross = matcher.match_indices(left, e2)
            shared_left, shared_right, shared_e2 = intersect_e1_left_tracks(stereo, cross)
            confidence = np.minimum(
                right.detector_scores[shared_right], e2.detector_scores[shared_e2]
            )
            order = np.lexsort((shared_left, -confidence))
            order = order[: args.tracks_per_frame]
            shared_left = shared_left[order]
            shared_right = shared_right[order]
            shared_e2 = shared_e2[order]
            confidence = confidence[order]
            x_left = left.keypoints_px[shared_left]
            x_right = right.keypoints_px[shared_right]
            x_e2 = e2.keypoints_px[shared_e2]
            if len(x_left) < 8:
                frame_records.append({
                    "sequence": sequence_dir.name,
                    "frame_id": frame_id,
                    "cached": False,
                    "reason": f"only {len(x_left)} accepted tracks",
                })
                continue
            sequences.append(sequence_dir.name)
            frame_ids.append(frame_id)
            fractions.append(fraction)
            all_left.append(x_left)
            all_right.append(x_right)
            all_e2.append(x_e2)
            all_confidence.append(confidence)
            all_k1l.append(inputs.k1_l)
            all_k1r.append(inputs.k1_r)
            all_k2l.append(inputs.k2_l)
            all_initial.append(initial)
            offsets.append(offsets[-1] + len(x_left))
            frame_records.append({
                "sequence": sequence_dir.name,
                "frame_id": frame_id,
                "cached": True,
                "tracks": len(x_left),
                "confidence_median": float(np.median(confidence)),
            })

        if not all_left:
            raise RuntimeError(f"No usable cached observations for session {session}")
        cache_path = args.output_root / f"session_{session}.npz"
        np.savez_compressed(
            cache_path,
            session_id=np.array(session),
            sequences=np.asarray(sequences),
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            temporal_fraction=np.asarray(fractions, dtype=np.float64),
            offsets=np.asarray(offsets, dtype=np.int64),
            x_e1l=np.concatenate(all_left),
            x_e1r=np.concatenate(all_right),
            x_e2l=np.concatenate(all_e2),
            confidence=np.concatenate(all_confidence),
            K1_L=np.stack(all_k1l),
            K1_R=np.stack(all_k1r),
            K2_L=np.stack(all_k2l),
            T_e2_from_e1_initial=np.stack(all_initial),
            original_R_LR=calibration.R_LR,
            original_t_LR=calibration.t_LR_unit,
            suggested_min_depth=np.array(calibration.suggested_min_depth),
            suggested_max_depth=np.array(calibration.suggested_max_depth),
        )
        session_summary = {
            "cache": str(cache_path),
            "num_frames": len(frame_ids),
            "num_sequences": len(set(sequences)),
            "sequences": sorted(set(sequences)),
            "total_tracks": int(offsets[-1]),
            "median_tracks_per_frame": float(np.median(np.diff(offsets))),
            "runtime_seconds": time.perf_counter() - started_session,
            "frames": frame_records,
        }
        global_summary["sessions"][f"session_{session}"] = session_summary
        (args.output_root / f"session_{session}_summary.json").write_text(
            json.dumps(session_summary, indent=2, default=_json_default) + "\n"
        )
        print(
            f"session_{session}: cached {len(frame_ids)} frames / {offsets[-1]} "
            f"tracks -> {cache_path}", flush=True
        )

    (args.output_root / "cache_summary.json").write_text(
        json.dumps(global_summary, indent=2, default=_json_default) + "\n"
    )
    print(f"Saved cache summary: {args.output_root / 'cache_summary.json'}")


if __name__ == "__main__":
    main()
