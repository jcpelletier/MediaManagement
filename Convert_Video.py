#!/usr/bin/env python3
"""
mkv_to_mp4.py

Recursively converts all .mkv files under a given folder to .mp4 using ffmpeg:

Video: H.264 (CRF 18, preset slow)
Audio: AAC-LC, stereo, 48 kHz, 256 kbps (safer for Android & web)
MP4:   +faststart for better streaming

Windows-friendly, handles spaces/unicode paths, and skips files that already have a matching .mp4
unless --overwrite is provided.

If --overwrite is used, existing .mp4 files are replaced and original .mkv files are deleted after successful conversion.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import datetime

def find_ffmpeg() -> str:
    """Return path to ffmpeg or raise if not found."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError(
            "ffmpeg not found in PATH. Install it (e.g., from https://www.gyan.dev/ffmpeg/builds/) "
            "and ensure ffmpeg.exe is in your PATH."
        )
    return ffmpeg

def convert_file(ffmpeg_exe: str, mkv_path: Path, overwrite: bool, dry_run: bool) -> int:
    """
    Convert a single MKV to MP4 with the same base name in the same folder.
    Returns ffmpeg exit code (0 on success). If skipped, returns 0.
    """
    mp4_path = mkv_path.with_suffix(".mp4")

    if mp4_path.exists() and not overwrite:
        print(f"SKIP: {mp4_path} already exists (use --overwrite to replace).")
        return 0

    cmd = [
        ffmpeg_exe,
        "-y" if overwrite else "-n",
        "-i", str(mkv_path),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac",
        "-profile:a", "aac_low",
        "-ac", "2",
        "-ar", "48000",
        "-b:a", "256k",
        "-movflags", "+faststart",
        str(mp4_path),
    ]

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Converting:")
    print(f"  In : {mkv_path}")
    print(f"  Out: {mp4_path}")
    if dry_run:
        print("  (dry-run) Command:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print("  ffmpeg output:\n", proc.stdout)
        print(f"ERROR: ffmpeg failed for {mkv_path} (exit {proc.returncode})")
    else:
        print("  Done.")
        if overwrite:
            try:
                mkv_path.unlink()
                print(f"  Deleted original MKV: {mkv_path}")
            except Exception as e:
                print(f"  WARNING: Could not delete {mkv_path}: {e}")

    return proc.returncode

def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert .mkv files to .mp4 with safer AAC audio for Android/web."
    )
    parser.add_argument("folder", type=Path, help="Root folder to scan recursively for .mkv files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .mp4 files and delete MKVs after successful conversion.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without running ffmpeg.")
    args = parser.parse_args()

    root = args.folder
    if not root.exists() or not root.is_dir():
        print(f"ERROR: '{root}' is not a directory.")
        sys.exit(2)

    try:
        ffmpeg_exe = find_ffmpeg()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(3)

    mkv_files = sorted(p for p in root.rglob("*.mkv") if p.is_file())
    if not mkv_files:
        print("No .mkv files found.")
        return

    print(f"Found {len(mkv_files)} .mkv file(s) under {root}.")

    failures = 0
    for mkv in mkv_files:
        rc = convert_file(ffmpeg_exe, mkv, args.overwrite, args.dry_run)
        if rc != 0:
            failures += 1

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        sys.exit(1)
    else:
        print("\nAll conversions completed successfully.")

if __name__ == "__main__":
    main()
