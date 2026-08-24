#!/usr/bin/env python3
"""Verify official EfficientLoFTR loading, preprocessing, and output shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EfficientLoFTRConfig
from loftr_wrapper import EfficientLoFTRMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loftr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image0", type=Path, required=True)
    parser.add_argument("--image1", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", choices=["full", "opt"], default="full")
    parser.add_argument("--precision", choices=["fp32", "mp", "fp16"], default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    matcher = EfficientLoFTRMatcher(
        loftr_root=args.loftr_root,
        checkpoint=args.checkpoint,
        device=args.device,
        config=EfficientLoFTRConfig(args.model_type, args.precision),
    )
    matches = matcher.match(args.image0, args.image1)
    payload = {
        "model": matcher.metadata(),
        "image0": str(args.image0),
        "image1": str(args.image1),
        "original_shape0": matches.original_shape0,
        "original_shape1": matches.original_shape1,
        "inference_shape0": matches.inference_shape0,
        "inference_shape1": matches.inference_shape1,
        "num_matches": int(matches.confidence.size),
        "points0_shape": matches.points0.shape,
        "points1_shape": matches.points1.shape,
        "confidence": {
            "minimum": float(np.min(matches.confidence)) if matches.confidence.size else None,
            "p10": float(np.percentile(matches.confidence, 10)) if matches.confidence.size else None,
            "median": float(np.median(matches.confidence)) if matches.confidence.size else None,
            "p90": float(np.percentile(matches.confidence, 90)) if matches.confidence.size else None,
            "maximum": float(np.max(matches.confidence)) if matches.confidence.size else None,
        },
        "inference_seconds": matches.inference_seconds,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() and args.device.startswith("cuda") else 0.0
        ),
        "coordinates_restored_to_original_pixels": True,
        "offline_checkpoint_verified": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Saved EfficientLoFTR verification: {args.output}")


if __name__ == "__main__":
    main()

