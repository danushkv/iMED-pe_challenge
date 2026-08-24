#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_diagnostics(diagnostics: Path, output: Path) -> None:
    frames = json.loads(diagnostics.read_text())["frames"]
    ids = np.asarray([item["frame_idx"] for item in frames])
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(ids, [item["position_disagreement_normalized"] for item in frames])
    axes[0].set_ylabel("A/B position\ndisagreement / step")
    axes[1].plot(ids, [item["rotation_disagreement_deg"] for item in frames])
    axes[1].set_ylabel("A/B rotation (deg)")
    axes[2].plot(ids, [item["a_inlier_ratio"] for item in frames], label="A")
    axes[2].plot(ids, [item["b_inlier_ratio"] for item in frames], label="B")
    axes[2].set_ylabel("PnP inlier ratio")
    axes[2].legend()
    axes[3].plot(ids, [item["weight_a"] for item in frames], label="wA")
    axes[3].plot(ids, [item["weight_b"] for item in frames], label="wB")
    axes[3].set_ylabel("anchor weight")
    axes[3].legend()
    axes[4].plot(ids, [item["vo_increment_magnitude"] for item in frames], label="increment")
    axes[4].plot(ids, [item["vo_residual"] for item in frames], label="residual")
    axes[4].set_ylabel("VO")
    axes[4].set_xlabel("frame ID")
    axes[4].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one or every saved Method 4A diagnostics file."
    )
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-root", type=Path,
        help="Method 4 run root containing SPLIT/SEQUENCE/diagnostics.json",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    single = args.diagnostics is not None or args.output is not None
    bulk = args.run_root is not None or args.output_root is not None
    if single and bulk:
        parser.error("use either --diagnostics/--output or --run-root/--output-root")
    if single:
        if args.diagnostics is None or args.output is None:
            parser.error("single mode requires both --diagnostics and --output")
        plot_diagnostics(args.diagnostics, args.output)
        return
    if args.run_root is None or args.output_root is None:
        parser.error("bulk mode requires --run-root and --output-root")

    paths = sorted((args.run_root / args.split).glob("*/diagnostics.json"))
    if not paths:
        raise RuntimeError(f"no diagnostics under {args.run_root / args.split}")
    for path in paths:
        plot_diagnostics(path, args.output_root / f"{path.parent.name}.png")
    print(f"Saved {len(paths)} diagnostic plots under: {args.output_root}")


if __name__ == "__main__":
    main()
