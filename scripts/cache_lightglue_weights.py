#!/usr/bin/env python3
"""Explicitly cache official ALIKED and LightGlue weights for offline use."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    args = parser.parse_args()

    from lightglue import ALIKED, LightGlue

    ALIKED(max_num_keypoints=args.max_keypoints).eval()
    LightGlue(features="aliked").eval()
    print("Official ALIKED and LightGlue weights are available in the local cache.")


if __name__ == "__main__":
    main()

