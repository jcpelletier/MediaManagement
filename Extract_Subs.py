#!/usr/bin/env python3
"""
Extract_Subs.py

Recursively scans a folder for .mkv files and extracts all embedded subtitle
tracks as sidecar files alongside each MKV. Jellyfin picks up sidecar files
automatically.

Output naming:
  MovieName.en.srt      — first English text track
  MovieName.en.2.srt    — second English text track
  MovieName.en.sup      — first English PGS (Blu-ray) track
  MovieName.und.srt     — track with no language tag
  etc.

Supported codecs:
  Text  (subrip, srt, ass, ssa, webvtt, mov_text)  → .{lang}.srt
  PGS   (hdmv_pgs_subtitle, pgssub)                → .{lang}.sup
  DVD   (dvdsub)                                    → .{lang}.sub  (+ .idx)

Already-existing sidecar files are skipped. Safe to re-run.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil

TEXT_CODECS  = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
IMAGE_CODECS = {"hdmv_pgs_subtitle", "pgssub", "dvdsub", "dvb_subtitle"}


def find_ffmpeg() -> tuple[str, str]:
    ffmpeg  = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found in PATH.")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found in PATH.")
    return ffmpeg, ffprobe


def probe_subtitle_streams(ffprobe_exe: str, mkv_path: Path) -> list[dict]:
    """Return subtitle stream list from ffprobe, or [] on error."""
    cmd = [
        ffprobe_exe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        str(mkv_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []


def extract_subs_from_file(
    ffmpeg_exe: str,
    ffprobe_exe: str,
    mkv_path: Path,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Extract all subtitle tracks from a single MKV.
    Returns (extracted_count, skipped_count).
    """
    streams = probe_subtitle_streams(ffprobe_exe, mkv_path)
    if not streams:
        print(f"  No subtitle tracks found.")
        return 0, 0

    extracted = skipped = 0
    lang_ext_counts: dict[str, int] = {}

    for stream in streams:
        stream_idx = stream.get("index")
        codec      = stream.get("codec_name", "").lower()
        tags       = stream.get("tags", {})
        lang       = (tags.get("language") or "und").lower()
        title      = tags.get("title", "")

        if codec in TEXT_CODECS:
            out_ext  = ".srt"
            copy_arg = ["-c:s", "srt"]
        elif codec in {"hdmv_pgs_subtitle", "pgssub"}:
            out_ext  = ".sup"
            copy_arg = ["-c:s", "copy"]
        elif codec == "dvdsub":
            out_ext  = ".sub"
            copy_arg = ["-c:s", "copy"]
        else:
            print(f"  [stream {stream_idx}] Unsupported codec '{codec}' — skipping.")
            continue

        key = f"{lang}{out_ext}"
        lang_ext_counts[key] = lang_ext_counts.get(key, 0) + 1
        count = lang_ext_counts[key]

        stem = mkv_path.stem
        suffix = f".{lang}{out_ext}" if count == 1 else f".{lang}.{count}{out_ext}"
        sub_path = mkv_path.parent / f"{stem}{suffix}"

        label = f"stream {stream_idx} ({codec}, {lang})"
        if title:
            label += f" [{title}]"

        if sub_path.exists():
            print(f"  SKIP {label} — {sub_path.name} already exists.")
            skipped += 1
            continue

        print(f"  Extracting {label} → {sub_path.name}")
        if dry_run:
            continue

        result = subprocess.run(
            [ffmpeg_exe, "-i", str(mkv_path), "-map", f"0:{stream_idx}", *copy_arg, str(sub_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            extracted += 1
        else:
            print(f"  ERROR extracting {label}:")
            print(f"  {result.stderr[-300:]}")
            sub_path.unlink(missing_ok=True)

    return extracted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract embedded subtitle tracks from MKV files as sidecar files."
    )
    parser.add_argument(
        "folder", type=Path,
        help="Root folder to scan recursively for .mkv files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be extracted without writing any files.",
    )
    args = parser.parse_args()

    root = args.folder
    if not root.exists() or not root.is_dir():
        print(f"ERROR: '{root}' is not a directory.")
        sys.exit(2)

    try:
        ffmpeg_exe, ffprobe_exe = find_ffmpeg()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(3)

    mkv_files = sorted(p for p in root.rglob("*.mkv") if p.is_file())
    if not mkv_files:
        print("No .mkv files found.")
        return

    print(f"Found {len(mkv_files)} .mkv file(s) under {root}.")
    if args.dry_run:
        print("Dry-run mode — no files will be written.\n")

    total_extracted = total_skipped = 0
    for mkv in mkv_files:
        print(f"\n{mkv.relative_to(root)}")
        extracted, skipped = extract_subs_from_file(ffmpeg_exe, ffprobe_exe, mkv, args.dry_run)
        total_extracted += extracted
        total_skipped   += skipped

    print(f"\nDone. {total_extracted} extracted, {total_skipped} already present.")


if __name__ == "__main__":
    main()
