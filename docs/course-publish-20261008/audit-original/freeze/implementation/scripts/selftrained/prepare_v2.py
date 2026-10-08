#!/usr/bin/env python3
"""Rebuild V2 data using only the reviewed fixed producer and augmentation chain.

Run from a pinned source checkout. package.py verifies all seven source hashes
against the committed recipe before invoking this no-argument orchestrator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_PRODUCERS = (
    "scripts/selftrained/prepare_vision_ocr.py",
    "scripts/selftrained/prepare_voice.py",
    "scripts/selftrained/prepare_text_tools.py",
)
AUGMENTATIONS = (
    "scripts/selftrained/augment_text_tools_v2.py",
    "scripts/selftrained/augment_ocr_v2.py",
    "scripts/selftrained/augment_voice_v2.py",
)
SOURCE_DATA = Path("outputs/selftrained/data")
OUTPUT_DATA = Path("outputs/selftrained-v2/data")


def rebuild(root: Path = ROOT) -> dict:
    root = root.resolve()
    # Check every fixed program before any producer or filesystem mutation.
    for relative in (*BASE_PRODUCERS, *AUGMENTATIONS):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing reviewed reconstruction source: {relative}")
    source, output = root / SOURCE_DATA, root / OUTPUT_DATA
    for relative in (SOURCE_DATA, OUTPUT_DATA):
        path = root
        for component in relative.parts:
            path /= component
            if path.is_symlink():
                raise ValueError("Reconstruction data directories must not be symbolic links")
    for relative in BASE_PRODUCERS:
        subprocess.run([sys.executable, str(root / relative)], cwd=root, check=True)
    if not source.is_dir():
        raise ValueError("Original producers did not produce their fixed data directory")
    # Fresh copy prevents stale V2 rows/assets from surviving a reconstruction.
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Original generated data must not contain symbolic links")
    if output.exists():
        if not output.is_dir():
            raise ValueError("V2 output must be a directory")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    for relative in AUGMENTATIONS:
        subprocess.run(
            [
                sys.executable,
                str(root / relative),
                "--source-data",
                SOURCE_DATA.as_posix(),
                "--output-data",
                OUTPUT_DATA.as_posix(),
            ],
            cwd=root,
            check=True,
        )
    return {
        "reconstruction_pipeline": "selftrained-v2",
        "base_producers": list(BASE_PRODUCERS),
        "augmentations": list(AUGMENTATIONS),
        "source_data": SOURCE_DATA.as_posix(),
        "output_data": OUTPUT_DATA.as_posix(),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args()
    print(json.dumps(rebuild(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
