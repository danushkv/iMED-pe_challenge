from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt, write_pose_txt
from imcpe.methods.stereo_vo_anchored.anchor_estimator import load_anchor_cache
from imcpe.methods.stereo_vo_anchored.anchored_vo import fuse_anchored_trajectory
from imcpe.methods.stereo_vo_anchored.config import AnchorThresholds, FusionConfig
from imcpe.pose_metrics import RPE_DELTAS, evaluate_sequence


def _tag(interval: int, alpha: float) -> str:
    return f"K{interval:03d}_alpha{alpha:.2f}".replace(".", "p")


def _aggregate(per_sequence: dict[str, dict[str, float | int]]) -> dict[str, float]:
    fields = ["mean_ate", "median_ate", "std_ate"]
    for delta in RPE_DELTAS:
        fields.extend([f"trans_rpe_{delta}frame", f"rot_rpe_deg_{delta}frame"])
    return {
        field: float(np.mean([float(metrics[field]) for metrics in per_sequence.values()]))
        for field in fields
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRAIN-only Method 1.5A grid on selected representative sequences."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "method_1_5" / "tuning")
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--anchor-intervals", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-correction-deg", type=float, default=10.0)
    parser.add_argument("--max-anchor-disagreement-deg", type=float, default=None)
    args = parser.parse_args()

    split = "train"  # Intentionally not configurable: never tune on released test.
    thresholds = AnchorThresholds.load_json(args.thresholds)
    method1_rows: dict[str, list] = {}
    ground_truth_rows: dict[str, list] = {}
    anchors = {}
    method1_metrics: dict[str, dict[str, float | int]] = {}
    for sequence_name in args.sequence:
        method1_rows[sequence_name] = read_pose_txt(
            args.method1_root / split / sequence_name / "pose.txt"
        )
        # Ground truth is read only in this evaluation/tuning script, never by
        # anchor estimation or fusion.
        ground_truth_rows[sequence_name] = read_pose_txt(
            args.data_root / split / sequence_name / "pose.txt"
        )
        anchors[sequence_name] = load_anchor_cache(
            args.anchor_root / split / f"{sequence_name}.npz"
        )
        method1_metrics[sequence_name] = evaluate_sequence(
            ground_truth_rows[sequence_name], method1_rows[sequence_name]
        )

    grid_results: list[dict[str, object]] = []
    for interval in args.anchor_intervals:
        for alpha in args.alphas:
            config = FusionConfig(
                anchor_interval=interval,
                alpha=alpha,
                max_correction_deg=args.max_correction_deg,
                max_anchor_disagreement_deg=args.max_anchor_disagreement_deg,
            )
            tag = _tag(interval, alpha)
            per_sequence: dict[str, dict[str, float | int]] = {}
            alpha_zero_errors: dict[str, float] = {}
            for sequence_name in args.sequence:
                result = fuse_anchored_trajectory(
                    method1_rows[sequence_name],
                    anchors[sequence_name],
                    thresholds,
                    config,
                )
                output_dir = args.output_root / tag / split / sequence_name
                write_pose_txt(output_dir / "pose.txt", result.rows)
                (output_dir / "fusion_diagnostics.json").write_text(
                    json.dumps(
                        {
                            "summary": result.summary,
                            "frames": [asdict(frame) for frame in result.diagnostics],
                        },
                        indent=2,
                    )
                    + "\n"
                )
                per_sequence[sequence_name] = evaluate_sequence(
                    ground_truth_rows[sequence_name], result.rows
                )
                alpha_zero_errors[sequence_name] = float(
                    result.summary["alpha_zero_max_matrix_error"]
                )
            aggregate = _aggregate(per_sequence)
            improvements = [
                float(per_sequence[name]["mean_ate"])
                < float(method1_metrics[name]["mean_ate"])
                for name in args.sequence
            ]
            record: dict[str, object] = {
                "tag": tag,
                "config": asdict(config),
                "aggregate": aggregate,
                "num_improved_vs_method1": int(sum(improvements)),
                "num_worsened_vs_method1": int(len(improvements) - sum(improvements)),
                "per_sequence": per_sequence,
                "alpha_zero_max_matrix_errors": alpha_zero_errors,
            }
            grid_results.append(record)
            print(
                f"{tag}: ATE={aggregate['mean_ate']:.4f} "
                f"RPE10={aggregate['trans_rpe_10frame']:.4f}mm/"
                f"{aggregate['rot_rpe_deg_10frame']:.4f}deg "
                f"RPE40={aggregate['trans_rpe_40frame']:.4f}mm/"
                f"{aggregate['rot_rpe_deg_40frame']:.4f}deg "
                f"improved={sum(improvements)}/{len(improvements)}"
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "sequences": args.sequence,
        "thresholds": asdict(thresholds),
        "method1_per_sequence": method1_metrics,
        "method1_aggregate": _aggregate(method1_metrics),
        "grid": grid_results,
    }
    output_path = args.output_root / "grid_results.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved grid: {output_path}")
    print("Select K/alpha from TRAIN only, then freeze it before running test.")


if __name__ == "__main__":
    main()

