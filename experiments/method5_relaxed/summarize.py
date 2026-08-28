#!/usr/bin/env python3
"""Print strict-versus-relaxed Method-5 validity decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    print(
        "Session | conv | bounds | split | cross | objective | strict | relaxed "
        "| dR deg | dT deg | objective before -> after"
    )
    for session, item in sorted(data["sessions"].items()):
        print(
            f"{session:11s} | {str(item['optimizer_converged']):5s} | "
            f"{str(item['within_bounds']):6s} | "
            f"{str(item['split_half_stable']):5s} | "
            f"{str(item['cross_sequence_stable']):5s} | "
            f"{str(item['total_objective_improved']):9s} | "
            f"{str(item['strict_valid']):6s} | {str(item['relaxed_valid']):7s} | "
            f"{item['delta_rotation_deg']:6.3f} | "
            f"{item['delta_translation_direction_deg']:6.3f} | "
            f"{item['objective_before']:.6f} -> {item['objective_after']:.6f}"
        )
    print(data["counts"])


if __name__ == "__main__":
    main()

