#!/usr/bin/env python3
"""Download the WANDS dataset from Wayfair's GitHub repository.

Usage:
    python scripts/download_wands.py [--output-dir data/raw]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


WANDS_REPO = "https://github.com/wayfair/WANDS.git"


def download_wands(output_dir: str | Path = "data/raw") -> Path:
    """Download WANDS dataset via git clone.

    Args:
        output_dir: Directory to clone into.

    Returns:
        Path to the dataset directory.
    """
    output_dir = Path(output_dir)
    wands_dir = output_dir / "WANDS"

    if wands_dir.exists():
        print(f"WANDS already exists at {wands_dir}")
        print("Delete it and re-run to re-download.")
        return wands_dir / "dataset"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning WANDS repository to {wands_dir}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", WANDS_REPO, str(wands_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error cloning repository: {result.stderr}")
        print("\nAlternative: Download manually from https://github.com/wayfaireng/WANDS")
        sys.exit(1)

    # Find the dataset directory
    dataset_dir = wands_dir / "dataset"
    if not dataset_dir.exists():
        # Try alternative paths
        for candidate in wands_dir.rglob("product.csv"):
            dataset_dir = candidate.parent
            break

    # Verify files
    expected_files = ["product.csv", "query.csv", "label.csv"]
    for fname in expected_files:
        candidates = list(dataset_dir.rglob(fname))
        if not candidates:
            # Try tab-separated variants
            candidates = list(dataset_dir.rglob(f"*{fname.split('.')[0]}*"))
        if candidates:
            print(f"  ✓ Found: {candidates[0].name}")
        else:
            print(f"  ✗ Missing: {fname}")

    print(f"\nWANDS dataset downloaded to: {dataset_dir}")
    print(f"Files: {list(dataset_dir.glob('*'))}")

    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Download WANDS dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to download to (default: data/raw)",
    )
    args = parser.parse_args()

    download_wands(args.output_dir)


if __name__ == "__main__":
    main()
