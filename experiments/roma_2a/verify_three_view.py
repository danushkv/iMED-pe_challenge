from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.roma_2a.cycle_consistency import compute_cycle_consistency
from experiments.roma_2a.roma_wrapper import OfficialRoMa
from experiments.roma_2a.source_points import ALIKEDSourceDetector
from experiments.roma_2a.warp_points import sample_dense_warp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact-index E1-L/E1-R/E2-L RoMa tracks at one TRAIN frame."
        )
    )
    parser.add_argument("--roma-root", required=True)
    parser.add_argument("--torch-home")
    parser.add_argument("--e1-left", required=True)
    parser.add_argument("--e1-right", required=True)
    parser.add_argument("--e2-left", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-source-points", type=int, default=2048)
    parser.add_argument("--draw-points", type=int, default=200)
    parser.add_argument(
        "--legacy-separate-reverse",
        action="store_true",
        help="Use separate reverse RoMa calls for regression timing.",
    )
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    names = ("minimum", "p10", "p25", "p50", "p75", "p90", "maximum")
    if values.size == 0:
        return {name: float("nan") for name in names}
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def threshold_table(
    base_valid: np.ndarray,
    track_certainty: np.ndarray,
    stereo_cycle_error: np.ndarray,
    cross_cycle_error: np.ndarray,
) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    # These are diagnostic probes, not selected parameters.  Final R1/R2 values
    # are chosen only after aggregating TRAIN distributions.
    certainty_options: tuple[float | None, ...] = (None, 0.05, 0.20, 0.50)
    cycle_options: tuple[float | None, ...] = (None, 1.0, 2.0, 3.0)
    for certainty_threshold in certainty_options:
        certainty_name = (
            "certainty_none"
            if certainty_threshold is None
            else f"certainty_ge_{certainty_threshold:g}"
        )
        certainty_mask = base_valid.copy()
        if certainty_threshold is not None:
            certainty_mask &= track_certainty >= certainty_threshold
        table[certainty_name] = {}
        for cycle_threshold in cycle_options:
            cycle_name = (
                "cycle_none"
                if cycle_threshold is None
                else f"cycle_le_{cycle_threshold:g}px"
            )
            mask = certainty_mask.copy()
            if cycle_threshold is not None:
                mask &= stereo_cycle_error <= cycle_threshold
                mask &= cross_cycle_error <= cycle_threshold
            table[certainty_name][cycle_name] = int(np.count_nonzero(mask))
    return table


def draw_three_view(
    paths: tuple[str, str, str],
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    confidence: np.ndarray,
    maximum: int,
) -> np.ndarray:
    images = [load_image(path) for path in paths]
    height = max(image.shape[0] for image in images)
    offsets = np.cumsum([0] + [image.shape[1] for image in images[:-1]])
    canvas = np.zeros((height, sum(image.shape[1] for image in images), 3), np.uint8)
    for image, offset in zip(images, offsets):
        canvas[: image.shape[0], offset : offset + image.shape[1]] = image

    indices = np.flatnonzero(mask)
    if len(indices) > maximum:
        indices = indices[np.argsort(confidence[indices])[::-1][:maximum]]
    for rank, index in enumerate(indices):
        hue = int(179 * rank / max(1, len(indices) - 1))
        hsv = np.uint8([[[hue, 230, 255]]])
        color = tuple(
            int(value)
            for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        )
        plotted = []
        for view_points, offset in zip(points, offsets):
            xy = np.rint(view_points[index]).astype(int)
            point = (int(xy[0] + offset), int(xy[1]))
            plotted.append(point)
            cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, plotted[0], plotted[1], color, 1, cv2.LINE_AA)
        cv2.line(canvas, plotted[0], plotted[2], color, 1, cv2.LINE_AA)
    labels = ("E1-L source", "E1-R stereo", "E2-L cross")
    for label, offset in zip(labels, offsets):
        cv2.putText(
            canvas,
            label,
            (int(offset) + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    e1_left_image = load_image(args.e1_left)
    e1_right_image = load_image(args.e1_right)
    e2_left_image = load_image(args.e2_left)
    e1_left_shape = e1_left_image.shape[:2]
    e1_right_shape = e1_right_image.shape[:2]
    e2_left_shape = e2_left_image.shape[:2]

    detector = ALIKEDSourceDetector(args.max_source_points, args.device)
    source = detector.extract(args.e1_left)
    if tuple(source.image_shape_hw) != tuple(e1_left_shape):
        raise RuntimeError("ALIKED source image shape differs from OpenCV image shape")

    model = OfficialRoMa(args.roma_root, args.device, args.torch_home)
    stereo_forward = model.match(args.e1_left, args.e1_right)
    cross_forward = model.match(args.e1_left, args.e2_left)
    stereo = sample_dense_warp(
        stereo_forward.warp,
        stereo_forward.certainty,
        source.points_px,
        e1_left_shape,
        e1_right_shape,
    )
    cross = sample_dense_warp(
        cross_forward.warp,
        cross_forward.certainty,
        source.points_px,
        e1_left_shape,
        e2_left_shape,
    )

    reverse_predictions = []
    if args.legacy_separate_reverse:
        stereo_reverse = model.match(args.e1_right, args.e1_left)
        cross_reverse = model.match(args.e2_left, args.e1_left)
        stereo_reverse_warp = stereo_reverse.warp
        stereo_reverse_certainty = stereo_reverse.certainty
        cross_reverse_warp = cross_reverse.warp
        cross_reverse_certainty = cross_reverse.certainty
        stereo_reverse_seconds = stereo_reverse.seconds
        cross_reverse_seconds = cross_reverse.seconds
        reverse_predictions = [stereo_reverse, cross_reverse]
        reverse_source = "separate_explicit_inference"
    else:
        if (
            stereo_forward.reverse_warp is None
            or stereo_forward.reverse_certainty is None
            or cross_forward.reverse_warp is None
            or cross_forward.reverse_certainty is None
        ):
            raise RuntimeError("RoMa symmetric reverse fields are unavailable")
        stereo_reverse_warp = stereo_forward.reverse_warp
        stereo_reverse_certainty = stereo_forward.reverse_certainty
        cross_reverse_warp = cross_forward.reverse_warp
        cross_reverse_certainty = cross_forward.reverse_certainty
        stereo_reverse_seconds = 0.0
        cross_reverse_seconds = 0.0
        reverse_source = "reused_from_forward_symmetric_inference"
    stereo_cycle = compute_cycle_consistency(
        source.points_px,
        stereo.target_points_px,
        stereo.valid,
        stereo_reverse_warp,
        stereo_reverse_certainty,
        e1_left_shape,
        e1_right_shape,
    )
    cross_cycle = compute_cycle_consistency(
        source.points_px,
        cross.target_points_px,
        cross.valid,
        cross_reverse_warp,
        cross_reverse_certainty,
        e1_left_shape,
        e2_left_shape,
    )

    base_valid = stereo.valid & cross.valid
    track_certainty = np.minimum(stereo.certainty, cross.certainty)
    track_cycle_max = np.maximum(stereo_cycle.error_px, cross_cycle.error_px)
    cycle3_mask = base_valid & (track_cycle_max <= 3.0)

    all_visualization = draw_three_view(
        (args.e1_left, args.e1_right, args.e2_left),
        (source.points_px, stereo.target_points_px, cross.target_points_px),
        base_valid,
        track_certainty,
        args.draw_points,
    )
    cycle_visualization = draw_three_view(
        (args.e1_left, args.e1_right, args.e2_left),
        (source.points_px, stereo.target_points_px, cross.target_points_px),
        cycle3_mask,
        track_certainty,
        args.draw_points,
    )
    cv2.imwrite(str(output_root / "three_view_all.png"), all_visualization)
    cv2.imwrite(str(output_root / "three_view_cycle3.png"), cycle_visualization)

    report = {
        "model": model.metadata(),
        "paths": {
            "e1_left": args.e1_left,
            "e1_right": args.e1_right,
            "e2_left": args.e2_left,
        },
        "source_points": int(len(source.points_px)),
        "valid_stereo_warps": int(np.count_nonzero(stereo.valid)),
        "valid_cross_warps": int(np.count_nonzero(cross.valid)),
        "valid_joint_warps": int(np.count_nonzero(base_valid)),
        "stereo_certainty": quantiles(stereo.certainty[stereo.valid]),
        "cross_certainty": quantiles(cross.certainty[cross.valid]),
        "track_certainty_min": quantiles(track_certainty[base_valid]),
        "stereo_cycle_error_px": quantiles(
            stereo_cycle.error_px[stereo_cycle.valid]
        ),
        "cross_cycle_error_px": quantiles(cross_cycle.error_px[cross_cycle.valid]),
        "joint_cycle_max_error_px": quantiles(track_cycle_max[base_valid]),
        "track_counts": threshold_table(
            base_valid,
            track_certainty,
            stereo_cycle.error_px,
            cross_cycle.error_px,
        ),
        "runtime": {
            "aliked_extraction_seconds": source.extraction_seconds,
            "stereo_forward_seconds": stereo_forward.seconds,
            "cross_forward_seconds": cross_forward.seconds,
            "stereo_reverse_seconds": stereo_reverse_seconds,
            "cross_reverse_seconds": cross_reverse_seconds,
            "reverse_warp_source": reverse_source,
            "forward_only_seconds": stereo_forward.seconds + cross_forward.seconds,
            "forward_and_cycle_seconds": (
                stereo_forward.seconds
                + cross_forward.seconds
                + stereo_reverse_seconds
                + cross_reverse_seconds
            ),
        },
        "peak_gpu_memory_gib": max(
            prediction.peak_gpu_memory_gib
            for prediction in [
                stereo_forward,
                cross_forward,
                *reverse_predictions,
            ]
        ),
        "visualizations": {
            "all": str(output_root / "three_view_all.png"),
            "cycle_le_3px": str(output_root / "three_view_cycle3.png"),
        },
    }
    output_path = output_root / "three_view_verification.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"Saved three-view RoMa verification: {output_path}")


if __name__ == "__main__":
    main()
