#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.cross_stereo_pnp_reverse.diagnostics import (
    create_method2ab_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-inference Method 2A-vs-2B session diagnostics."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--e1-calibration-root", type=Path, required=True)
    parser.add_argument("--e2-calibration-root", type=Path, required=True)
    parser.add_argument("--method2a-root", type=Path, required=True)
    parser.add_argument("--method2b-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = create_method2ab_diagnostics(
        data_root=args.data_root,
        split=args.split,
        comparison_json=args.comparison_json,
        e1_calibration_root=args.e1_calibration_root,
        e2_calibration_root=args.e2_calibration_root,
        method2a_root=args.method2a_root,
        method2b_root=args.method2b_root,
        output_root=args.output_root,
    )
    print(json.dumps(result["session_table"], indent=2))
    print("SESSION 005")
    print(json.dumps(result["session_005_diagnostics"], indent=2))
    print(
        "Saved: "
        f"{args.output_root / f'method2ab_session_analysis_{args.split}.json'}"
    )


if __name__ == "__main__":
    main()
