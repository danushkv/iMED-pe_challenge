#!/usr/bin/env python3
"""Verify the local official XFeat checkout and sparse output API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import XFeatConfig
from xfeat_wrapper import XFeatSparseMatcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xfeat-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--image0", type=Path, required=True)
    parser.add_argument("--image1", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--min-cosine", type=float, default=-1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matcher = XFeatSparseMatcher(
        xfeat_root=args.xfeat_root,
        weights=args.weights,
        device=args.device,
        config=XFeatConfig(
            max_features=args.max_features,
            min_cosine_similarity=args.min_cosine,
        ),
    )
    features0 = matcher.extract(args.image0)
    features1 = matcher.extract(args.image1)
    matches = matcher.match_indices_with_scores(features0, features1)
    keypoints0 = matcher.keypoints(features0)
    keypoints1 = matcher.keypoints(features1)
    image0 = cv2.imread(str(args.image0), cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(str(args.image1), cv2.IMREAD_GRAYSCALE)
    result = {
        "model": matcher.metadata(),
        "image0": {
            "path": str(args.image0),
            "original_shape": list(image0.shape) if image0 is not None else None,
            "num_features": int(keypoints0.shape[0]),
            "descriptor_shape": list(features0["descriptors"].shape),
        },
        "image1": {
            "path": str(args.image1),
            "original_shape": list(image1.shape) if image1 is not None else None,
            "num_features": int(keypoints1.shape[0]),
            "descriptor_shape": list(features1["descriptors"].shape),
        },
        "matches": {
            "count": int(matches.indices.shape[0]),
            "indices_shape": list(matches.indices.shape),
            "confidence_median": (
                float(np.median(matches.confidence))
                if matches.confidence.size else None
            ),
            "exact_source_indices": True,
        },
        "offline_weights_verified": matcher.weights.is_file(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved XFeat verification: {args.output}")


if __name__ == "__main__":
    main()
