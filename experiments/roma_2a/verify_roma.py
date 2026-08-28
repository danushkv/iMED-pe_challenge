from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.roma_2a.cycle_consistency import compute_cycle_consistency
from experiments.roma_2a.roma_wrapper import OfficialRoMa
from experiments.roma_2a.source_points import ALIKEDSourceDetector
from experiments.roma_2a.warp_points import (
    normalized_to_pixels,
    pixels_to_normalized,
    sample_dense_warp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify full RoMa dense-warp coordinates on one image pair."
    )
    parser.add_argument("--roma-root", required=True)
    parser.add_argument("--torch-home")
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-source-points", type=int, default=2048)
    parser.add_argument("--draw-points", type=int, default=100)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def image_shape(path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return int(image.shape[0]), int(image.shape[1])


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: float("nan") for name in ("minimum", "p10", "p25", "p50", "p75", "p90", "maximum")}
    return {
        "minimum": float(np.min(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p25": float(np.quantile(finite, 0.25)),
        "p50": float(np.quantile(finite, 0.50)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
        "maximum": float(np.max(finite)),
    }


def draw_matches(
    image0_path: str,
    image1_path: str,
    source_points: np.ndarray,
    target_points: np.ndarray,
    certainty: np.ndarray,
    valid: np.ndarray,
    maximum: int,
) -> np.ndarray:
    image0 = cv2.imread(image0_path, cv2.IMREAD_COLOR)
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    height = max(image0.shape[0], image1.shape[0])
    canvas = np.zeros((height, image0.shape[1] + image1.shape[1], 3), dtype=np.uint8)
    canvas[: image0.shape[0], : image0.shape[1]] = image0
    canvas[: image1.shape[0], image0.shape[1] :] = image1
    indices = np.flatnonzero(valid)
    if len(indices) > maximum:
        order = np.argsort(certainty[indices])[::-1][:maximum]
        indices = indices[order]
    for rank, index in enumerate(indices):
        hue = int(179 * rank / max(1, len(indices) - 1))
        color = tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        point0 = tuple(np.rint(source_points[index]).astype(int))
        point1_arr = np.rint(target_points[index]).astype(int)
        point1 = (int(point1_arr[0] + image0.shape[1]), int(point1_arr[1]))
        cv2.circle(canvas, point0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point1, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, point0, point1, color, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shape0 = image_shape(args.image0)
    shape1 = image_shape(args.image1)
    detector = ALIKEDSourceDetector(args.max_source_points, args.device)
    source = detector.extract(args.image0)
    if source.image_shape_hw != shape0:
        raise RuntimeError(
            f"ALIKED image shape mismatch: {source.image_shape_hw} vs {shape0}"
        )

    model = OfficialRoMa(args.roma_root, args.device, args.torch_home)
    forward = model.match(args.image0, args.image1)
    sampled = sample_dense_warp(
        forward.warp,
        forward.certainty,
        source.points_px,
        shape0,
        shape1,
    )
    if forward.reverse_warp is None or forward.reverse_certainty is None:
        raise RuntimeError(
            "Official RoMa model did not return a reusable symmetric reverse field"
        )
    optimized_cycle = compute_cycle_consistency(
        source.points_px,
        sampled.target_points_px,
        sampled.valid,
        forward.reverse_warp,
        forward.reverse_certainty,
        source_shape_hw=shape0,
        target_shape_hw=shape1,
    )

    # Regression reference only: reproduce the original implementation's
    # separate target-to-source invocation and compare it with the reverse
    # field reused from the symmetric forward call.
    reverse = model.match(args.image1, args.image0)
    legacy_cycle = compute_cycle_consistency(
        source.points_px,
        sampled.target_points_px,
        sampled.valid,
        reverse.warp,
        reverse.certainty,
        source_shape_hw=shape0,
        target_shape_hw=shape1,
    )
    reverse_warp_max_error = float(
        torch.max(torch.abs(forward.reverse_warp - reverse.warp)).item()
    )
    reverse_certainty_max_error = float(
        torch.max(
            torch.abs(forward.reverse_certainty - reverse.certainty)
        ).item()
    )
    optimized_cycle2 = optimized_cycle.valid & (optimized_cycle.error_px <= 2.0)
    legacy_cycle2 = legacy_cycle.valid & (legacy_cycle.error_px <= 2.0)

    round_trip = normalized_to_pixels(
        pixels_to_normalized(source.points_px, shape0), shape0
    ).numpy()
    conversion_error = np.linalg.norm(round_trip - source.points_px, axis=1)
    visualization = draw_matches(
        args.image0,
        args.image1,
        source.points_px,
        sampled.target_points_px,
        sampled.certainty,
        sampled.valid,
        args.draw_points,
    )
    cv2.imwrite(str(output_root / "forward_matches.png"), visualization)

    report = {
        "model": model.metadata(),
        "image0": {"path": args.image0, "shape_hw": list(shape0)},
        "image1": {"path": args.image1, "shape_hw": list(shape1)},
        "source_points": int(len(source.points_px)),
        "valid_forward_points": int(np.count_nonzero(sampled.valid)),
        "certainty_all": quantiles(sampled.certainty),
        "certainty_valid": quantiles(sampled.certainty[sampled.valid]),
        "cycle_error_px": quantiles(
            optimized_cycle.error_px[optimized_cycle.valid]
        ),
        "valid_cycle_points": int(np.count_nonzero(optimized_cycle.valid)),
        "coordinate_round_trip_max_error_px": float(np.max(conversion_error)),
        "forward_seconds": forward.seconds,
        "reverse_seconds": reverse.seconds,
        "symmetric_reuse_verification": {
            "reverse_warp_max_abs_error": reverse_warp_max_error,
            "reverse_certainty_max_abs_error": reverse_certainty_max_error,
            "optimized_cycle_error_px": quantiles(
                optimized_cycle.error_px[optimized_cycle.valid]
            ),
            "legacy_cycle_error_px": quantiles(
                legacy_cycle.error_px[legacy_cycle.valid]
            ),
            "cycle_le_2px_mask_disagreement_count": int(
                np.count_nonzero(optimized_cycle2 != legacy_cycle2)
            ),
            "optimized_pair_seconds": forward.seconds,
            "legacy_pair_seconds": forward.seconds + reverse.seconds,
            "measured_speedup": (
                (forward.seconds + reverse.seconds) / forward.seconds
                if forward.seconds > 0
                else float("nan")
            ),
        },
        "peak_gpu_memory_gib": max(
            forward.peak_gpu_memory_gib, reverse.peak_gpu_memory_gib
        ),
        "visualization": str(output_root / "forward_matches.png"),
    }
    (output_root / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"Saved RoMa verification: {output_root / 'verification.json'}")


if __name__ == "__main__":
    main()
