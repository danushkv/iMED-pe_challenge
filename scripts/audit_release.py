#!/usr/bin/env python3
"""Fail when a release contains local paths, generated data, or large files."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".cff", ".txt", ".sh"}
BANNED_TEXT = (
    "/mnt" + "/cluster/",
    "venkateda" + "@",
    "/home/" + "venkateda/",
)
BANNED_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".mp4"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-file-mib", type=float, default=5.0)
    args = parser.parse_args()

    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BANNED_SUFFIXES:
            failures.append(f"generated/model artifact: {relative}")
        if path.stat().st_size > args.max_file_mib * 1024 * 1024:
            failures.append(f"large file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(errors="replace")
            for banned in BANNED_TEXT:
                if banned in text:
                    failures.append(f"local path/token {banned!r}: {relative}")
    if failures:
        raise SystemExit("Release audit failed:\n" + "\n".join(failures))
    print("Release audit passed")


if __name__ == "__main__":
    main()
