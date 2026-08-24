from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt, write_pose_txt
from imcpe.methods.stereo_vo_anchored.anchor_estimator import load_anchor_cache
from imcpe.methods.stereo_vo_anchored.anchored_vo import fuse_anchored_trajectory
from imcpe.methods.stereo_vo_anchored.config import AnchorThresholds, FusionConfig


def _config_tag(interval: int, alpha: float) -> str:
    return f"K{interval:03d}_alpha{alpha:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-integrate cached Method 1 poses with cross-camera rotation anchors."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "method_1_5" / "runs")
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--anchor-interval", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--max-correction-deg", type=float, default=10.0)
    parser.add_argument("--max-anchor-disagreement-deg", type=float, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    method1_split = args.method1_root / args.split
    if args.sequence:
        sequence_names = args.sequence
    else:
        sequence_names = sorted(path.name for path in method1_split.iterdir() if path.is_dir())
    thresholds = AnchorThresholds.load_json(args.thresholds)
    config = FusionConfig(
        anchor_interval=args.anchor_interval,
        alpha=args.alpha,
        max_correction_deg=args.max_correction_deg,
        max_anchor_disagreement_deg=args.max_anchor_disagreement_deg,
    )
    tag = args.tag or _config_tag(config.anchor_interval, config.alpha)
    run_root = args.output_root / tag
    output_split = run_root / args.split
    summaries: dict[str, object] = {}

    for sequence_name in tqdm(sequence_names, desc="Anchored fusion"):
        method1_path = method1_split / sequence_name / "pose.txt"
        anchor_path = args.anchor_root / args.split / f"{sequence_name}.npz"
        if not method1_path.exists():
            raise FileNotFoundError(method1_path)
        if not anchor_path.exists():
            raise FileNotFoundError(anchor_path)
        method1_rows = read_pose_txt(method1_path)
        anchors = load_anchor_cache(anchor_path)
        result = fuse_anchored_trajectory(method1_rows, anchors, thresholds, config)
        sequence_output = output_split / sequence_name
        write_pose_txt(sequence_output / "pose.txt", result.rows)
        payload = {
            "summary": result.summary,
            "frames": [asdict(diagnostic) for diagnostic in result.diagnostics],
        }
        (sequence_output / "fusion_diagnostics.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        summaries[sequence_name] = result.summary

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / f"run_summary_{args.split}.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "method1_root": str(args.method1_root),
                "anchor_root": str(args.anchor_root),
                "thresholds": str(args.thresholds),
                "fusion_config": asdict(config),
                "sequences": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    config.save_yaml(run_root / "fusion_config.yaml", args.thresholds)
    print(f"Saved Method 1.5 run: {run_root}")


if __name__ == "__main__":
    main()

