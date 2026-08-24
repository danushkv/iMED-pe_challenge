from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.matcher import ALikeLightGlueMatcher
from imcpe.methods.stereo_vo_anchored.anchor_estimator import (
    generate_sequence_anchor_cache,
    save_anchor_cache,
)
from imcpe.methods.stereo_vo_anchored.config import AnchorEstimationConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache independent E1-L to E2-L cross-camera pose anchors."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "method_1_5" / "anchors")
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-threshold", type=float, default=1e-3)
    parser.add_argument("--ransac-probability", type=float, default=0.999)
    parser.add_argument("--disable-usac-magsac", action="store_true")
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    if args.sequence:
        sequence_dirs = [split_dir / name for name in args.sequence]
    else:
        sequence_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    missing = [path for path in sequence_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing sequence directories: {missing}")

    matcher = ALikeLightGlueMatcher(device=args.device)
    # Keep the baseline extractor/matcher implementation unchanged. Its ALIKED
    # keypoint cap is 2048; reject misleading alternative values here.
    if args.max_keypoints != 2048:
        raise ValueError(
            "Cross-camera anchors intentionally reuse the unchanged baseline matcher, "
            "which uses 2048 ALIKED keypoints"
        )
    config = AnchorEstimationConfig(
        ransac_threshold_normalized=args.ransac_threshold,
        ransac_probability=args.ransac_probability,
        prefer_usac_magsac=not args.disable_usac_magsac,
    )
    output_split = args.output_root / args.split
    for sequence_dir in tqdm(sequence_dirs, desc="Anchor sequences"):
        print(f"\nGenerating anchors: {args.split}/{sequence_dir.name}", flush=True)
        cache = generate_sequence_anchor_cache(sequence_dir, matcher, config)
        output_path = output_split / f"{sequence_dir.name}.npz"
        save_anchor_cache(cache, output_path, config)
        print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()

