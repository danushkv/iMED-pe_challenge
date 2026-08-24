from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.stereo_vo_anchored.anchor_estimator import derive_anchor_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive cross-camera anchor confidence thresholds from TRAIN caches."
    )
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-quantile", type=float, default=0.10)
    parser.add_argument("--upper-quantile", type=float, default=0.90)
    args = parser.parse_args()

    train_dir = args.anchor_root / "train"
    cache_paths = sorted(
        path for path in train_dir.glob("*.npz") if not path.name.endswith("_diagnostics.npz")
    )
    thresholds = derive_anchor_thresholds(
        cache_paths,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    thresholds.save_json(args.output)
    print(f"Derived thresholds from {thresholds.num_training_anchors} successful anchors")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

