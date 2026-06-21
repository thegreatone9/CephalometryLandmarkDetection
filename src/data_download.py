#!/usr/bin/env python3
"""Download and extract the Aariz cephalometric dataset.

This script fetches the Aariz dataset from Figshare and extracts it into the
data/ directory with the structure expected by the training pipeline.

Source:
    Khalid, M. A., et al. "A Benchmark Dataset for Automatic Cephalometric
    Landmark Detection and CVM Stage Classification."
    Scientific Data 12, 1336 (2025).
    https://doi.org/10.1038/s41597-025-05542-3

Figshare DOI: https://doi.org/10.6084/m9.figshare.27986417.v1
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGSHARE_URL = (
    "https://figshare.com/ndownloader/articles/27986417/versions/1"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_FILENAME = "aariz_dataset.zip"

# Expected top-level directories after extraction
EXPECTED_SPLITS = ["train", "valid", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    print(f"Downloading dataset from Figshare...")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req)

    total = response.headers.get("Content-Length")
    total = int(total) if total else None
    downloaded = 0
    block_size = 1024 * 1024  # 1 MB

    if total:
        total_mb = total / (1024 * 1024)
        print(f"  Size: {total_mb:.0f} MB")

    with open(dest, "wb") as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                mb = downloaded / (1024 * 1024)
                print(
                    f"\r  Progress: {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r  Downloaded: {mb:.0f} MB", end="", flush=True)

    print()  # newline after progress
    print(f"  Download complete.")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive with progress."""
    print(f"Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            if i % 200 == 0 or i == total:
                print(f"\r  Extracted: {i}/{total} files", end="", flush=True)
    print()
    print(f"  Extraction complete.")


def _flatten_if_nested(data_dir: Path) -> None:
    """If the zip extracts into a single subdirectory, flatten it.

    Some Figshare downloads wrap everything inside an extra directory level.
    This function detects that and moves contents up.
    """
    children = [c for c in data_dir.iterdir() if c.name != ZIP_FILENAME]

    # Check if extraction created a single subdirectory containing the splits
    if len(children) == 1 and children[0].is_dir():
        nested = children[0]
        nested_children = list(nested.iterdir())

        # Check if the nested dir contains the expected splits
        nested_names = {c.name for c in nested_children}
        if any(s in nested_names for s in EXPECTED_SPLITS):
            print(f"  Flattening nested directory: {nested.name}/")
            for item in nested_children:
                target = data_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            nested.rmdir()


def _validate_structure(data_dir: Path) -> bool:
    """Verify the expected directory structure exists."""
    ok = True
    for split in EXPECTED_SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  WARNING: Expected directory not found: {split_dir}")
            ok = False
            continue

        ceph_dir = split_dir / "Cephalograms"
        annot_dir = split_dir / "Annotations"

        if not ceph_dir.exists():
            print(f"  WARNING: Missing Cephalograms/ in {split}/")
            ok = False
        else:
            n_images = len(list(ceph_dir.glob("*")))
            print(f"  {split}/Cephalograms/: {n_images} files")

        if not annot_dir.exists():
            print(f"  WARNING: Missing Annotations/ in {split}/")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Aariz Cephalometric Dataset Downloader")
    print("=" * 60)
    print()

    # Check if data already exists
    has_splits = all((DATA_DIR / s).exists() for s in EXPECTED_SPLITS)
    if has_splits:
        print(f"Dataset already exists at: {DATA_DIR}")
        print(f"  Found splits: {', '.join(EXPECTED_SPLITS)}")
        resp = input("  Re-download? [y/N]: ").strip().lower()
        if resp != "y":
            print("Skipping download.")
            return

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    zip_path = DATA_DIR / ZIP_FILENAME
    if zip_path.exists():
        print(f"Zip file already exists: {zip_path}")
        print(f"  Skipping download, using existing file.")
    else:
        _download_with_progress(FIGSHARE_URL, zip_path)

    # Extract
    _extract_zip(zip_path, DATA_DIR)

    # Flatten if nested
    _flatten_if_nested(DATA_DIR)

    # Validate
    print()
    print("Validating directory structure...")
    valid = _validate_structure(DATA_DIR)

    # Clean up zip
    if zip_path.exists():
        print()
        resp = input(f"Delete zip file ({zip_path.stat().st_size / 1e9:.1f} GB)? [Y/n]: ").strip().lower()
        if resp != "n":
            zip_path.unlink()
            print("  Zip file deleted.")

    print()
    if valid:
        print("✅ Dataset is ready at: {DATA_DIR}")
        print("   Next step: ./run.sh train --epochs 50")
    else:
        print("⚠️  Dataset structure may need manual adjustment.")
        print(f"   Check the contents of: {DATA_DIR}")

    print()


if __name__ == "__main__":
    main()
