#!/usr/bin/env python3
"""Print the compact geometry-only Method-5 calibration report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    print("Session | valid | dR deg | dT-dir deg | split dR | split dT | reproj gain | failures")
    for name, item in sorted(data["sessions"].items()):
        failures = "; ".join(item["validity_failures"]) or "none"
        print(
            f"{name:11s} | {str(item['correction_valid']):5s} | "
            f"{item['delta_rotation_deg']:7.3f} | "
            f"{item['delta_translation_direction_deg']:10.3f} | "
            f"{item['split_rotation_disagreement_deg']:8.3f} | "
            f"{item['split_translation_disagreement_deg']:8.3f} | "
            f"{100.0 * item['relative_median_reprojection_improvement']:10.2f}% | "
            f"{failures}"
        )


if __name__ == "__main__":
    main()
