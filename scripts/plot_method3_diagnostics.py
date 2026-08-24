#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot inference-only Method 3 diagnostics.")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = json.loads(args.diagnostics.read_text())["frames"]
    ids = np.asarray([item["frame_id"] for item in frames])
    e1 = np.asarray([item["e1_stereo_baseline_raw"] for item in frames])
    e2 = np.asarray([item["e2_stereo_baseline_raw"] for item in frames])
    cross = np.asarray([item["cross_translation_magnitude_scaled"] for item in frames])
    runtime = np.asarray([item["inference_seconds"] for item in frames])
    focal = np.asarray(
        [np.nanmedian(item["focal_relative_errors"]) for item in frames]
    )
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(ids, e1, label="predicted E1 baseline")
    axes[0].plot(ids, e2, label="predicted E2 baseline")
    axes[0].legend()
    axes[0].set_ylabel("raw VGGT units")
    axes[1].plot(ids, cross)
    axes[1].set_ylabel("scaled cross |t|")
    axes[2].plot(ids, focal)
    axes[2].set_ylabel("median focal rel. error")
    axes[3].plot(ids, runtime)
    axes[3].set_ylabel("seconds")
    axes[3].set_xlabel("frame ID")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
