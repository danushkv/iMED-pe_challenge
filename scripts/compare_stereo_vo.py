from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def _diagnostic_summary(
    stereo_root: Path,
    split: str,
    sequence_name: str,
) -> tuple[float, float]:
    path = stereo_root / split / sequence_name / "diagnostics.json"
    if not path.exists():
        return float("nan"), float("nan")
    summary = json.loads(path.read_text())["summary"]
    return float(summary["pnp_success_rate"]), float(summary["median_pnp_inliers"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and Stereo VO using the repository's pose metrics."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--stereo-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    baseline_per_sequence, baseline_global = evaluate_split(
        args.data_root, args.baseline_root, args.split
    )
    stereo_per_sequence, stereo_global = evaluate_split(
        args.data_root, args.stereo_root, args.split
    )

    rows: dict[str, dict[str, float | int]] = {}
    print(
        "Sequence | Baseline ATE | Stereo ATE | Baseline RPE d=1 "
        "(mm/deg) | Stereo RPE d=1 (mm/deg) | PnP success | Median inliers"
    )
    print("-" * 150)
    for sequence_name in sorted(set(baseline_per_sequence) | set(stereo_per_sequence)):
        baseline = baseline_per_sequence[sequence_name]
        stereo = stereo_per_sequence[sequence_name]
        success_rate, median_inliers = _diagnostic_summary(
            args.stereo_root, args.split, sequence_name
        )
        row: dict[str, float | int] = {
            "baseline_mean_ate_mm": baseline["mean_ate"],
            "stereo_mean_ate_mm": stereo["mean_ate"],
            "pnp_success_rate": success_rate,
            "median_pnp_inliers": median_inliers,
        }
        for delta in RPE_DELTAS:
            row[f"baseline_trans_rpe_{delta}frame_mm"] = baseline[
                f"trans_rpe_{delta}frame"
            ]
            row[f"baseline_rot_rpe_{delta}frame_deg"] = baseline[
                f"rot_rpe_deg_{delta}frame"
            ]
            row[f"stereo_trans_rpe_{delta}frame_mm"] = stereo[
                f"trans_rpe_{delta}frame"
            ]
            row[f"stereo_rot_rpe_{delta}frame_deg"] = stereo[
                f"rot_rpe_deg_{delta}frame"
            ]
        rows[sequence_name] = row
        print(
            f"{sequence_name} | {baseline['mean_ate']:.4f} | {stereo['mean_ate']:.4f} | "
            f"{baseline['trans_rpe_1frame']:.4f}/{baseline['rot_rpe_deg_1frame']:.3f} | "
            f"{stereo['trans_rpe_1frame']:.4f}/{stereo['rot_rpe_deg_1frame']:.3f} | "
            f"{100.0 * success_rate:.1f}% | {median_inliers:.1f}"
        )

    valid_success = [
        float(row["pnp_success_rate"])
        for row in rows.values()
        if np.isfinite(row["pnp_success_rate"])
    ]
    valid_inliers = [
        float(row["median_pnp_inliers"])
        for row in rows.values()
        if np.isfinite(row["median_pnp_inliers"])
    ]
    comparison_global = {
        "baseline": baseline_global,
        "stereo_vo": stereo_global,
        "mean_sequence_pnp_success_rate": (
            float(np.mean(valid_success)) if valid_success else float("nan")
        ),
        "median_of_sequence_median_pnp_inliers": (
            float(np.median(valid_inliers)) if valid_inliers else float("nan")
        ),
    }
    print("\nGLOBAL MEAN")
    print(
        f"Baseline: ATE={baseline_global['mean_ate']:.4f}mm, "
        f"RPE d=1={baseline_global['trans_rpe_1frame']:.4f}mm/"
        f"{baseline_global['rot_rpe_deg_1frame']:.3f}deg"
    )
    print(
        f"Stereo VO: ATE={stereo_global['mean_ate']:.4f}mm, "
        f"RPE d=1={stereo_global['trans_rpe_1frame']:.4f}mm/"
        f"{stereo_global['rot_rpe_deg_1frame']:.3f}deg"
    )
    print(
        f"PnP: mean sequence success={100.0 * comparison_global['mean_sequence_pnp_success_rate']:.1f}%, "
        f"median inliers={comparison_global['median_of_sequence_median_pnp_inliers']:.1f}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "per_sequence": rows,
                    "global": comparison_global,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote comparison JSON: {args.json_out}")


if __name__ == "__main__":
    main()
