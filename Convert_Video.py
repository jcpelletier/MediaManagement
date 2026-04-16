#!/usr/bin/env python3
"""
Convert_Video.py

Recursively converts all .mkv files under a given folder to .mp4 using ffmpeg:

Video: H.264 (NVENC hardware encoding, CQ 19, preset p4)
Audio: AAC-LC, stereo, 48 kHz, 256 kbps (safer for Android & web)
MP4:   +faststart for better streaming

Before converting each MKV, embedded subtitle tracks are extracted as sidecar
files (e.g. MovieName.en.srt, MovieName.en.sup) so they are not lost during
conversion. Jellyfin picks these up automatically.

  Text subtitles  (subrip, ass, ssa, webvtt, mov_text) → .{lang}.srt
  PGS / Blu-ray   (hdmv_pgs_subtitle)                  → .{lang}.sup
  DVD image subs  (dvdsub)                              → .{lang}.sub + .{lang}.idx

Windows-friendly, handles spaces/unicode paths, and skips files that already
have a matching .mp4 unless --overwrite is provided.

If --overwrite is used, existing .mp4 files are replaced and original .mkv
files are deleted after successful conversion.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil
import datetime

# ---------------------------------------------------------------------------
# Subtitle codec classification
# ---------------------------------------------------------------------------

TEXT_CODECS  = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
IMAGE_CODECS = {"hdmv_pgs_subtitle", "pgssub", "dvdsub", "dvb_subtitle"}


def find_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_path, ffprobe_path) or raise if not found."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found in PATH.")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found in PATH.")
    return ffmpeg, ffprobe


# ---------------------------------------------------------------------------
# Subtitle extraction
# ---------------------------------------------------------------------------

def probe_subtitle_streams(ffprobe_exe: str, mkv_path: Path) -> list[dict]:
    """Return list of subtitle stream dicts from ffprobe, or [] on error."""
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


def extract_subtitles(ffmpeg_exe: str, ffprobe_exe: str, mkv_path: Path, dry_run: bool) -> list[Path]:
    """
    Extract all subtitle tracks from an MKV as sidecar files.

    Output files are placed alongside the MKV:
      MovieName.en.srt   — first English text track
      MovieName.en.2.srt — second English text track
      MovieName.en.sup   — first English PGS track
      etc.

    Returns list of paths to extracted (or already-existing) subtitle files.
    """
    streams = probe_subtitle_streams(ffprobe_exe, mkv_path)
    if not streams:
        return []

    extracted: list[Path] = []
    lang_ext_counts: dict[str, int] = {}

    for stream in streams:
        stream_idx = stream.get("index")
        codec      = stream.get("codec_name", "").lower()
        tags       = stream.get("tags", {})
        lang       = (tags.get("language") or "und").lower()
        title      = tags.get("title", "")

        # Determine output format
        if codec in TEXT_CODECS:
            out_ext  = ".srt"
            copy_arg = ["-c:s", "srt"]
        elif codec in {"hdmv_pgs_subtitle", "pgssub"}:
            out_ext  = ".sup"
            copy_arg = ["-c:s", "copy"]
        elif codec == "dvdsub":
            out_ext  = ".sub"     # ffmpeg also writes a paired .idx automatically
            copy_arg = ["-c:s", "copy"]
        else:
            print(f"  [SUBS] Skipping unsupported codec '{codec}' (stream {stream_idx})")
            continue

        # Build a unique output filename
        key = f"{lang}{out_ext}"
        lang_ext_counts[key] = lang_ext_counts.get(key, 0) + 1
        count = lang_ext_counts[key]

        stem = mkv_path.stem   # e.g. "Forrest Gump (1994)"
        if count == 1:
            sub_name = f"{stem}.{lang}{out_ext}"
        else:
            sub_name = f"{stem}.{lang}.{count}{out_ext}"
        sub_path = mkv_path.parent / sub_name

        label = f"stream {stream_idx} ({codec}, {lang})"
        if title:
            label += f" [{title}]"

        if sub_path.exists():
            print(f"  [SUBS] Already extracted: {sub_path.name}")
            extracted.append(sub_path)
            continue

        print(f"  [SUBS] Extracting {label} → {sub_path.name}")
        if dry_run:
            print(f"  [SUBS] (dry-run) skipping actual extraction")
            continue

        cmd = [
            ffmpeg_exe,
            "-i", str(mkv_path),
            "-map", f"0:{stream_idx}",
            *copy_arg,
            str(sub_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            extracted.append(sub_path)
        else:
            print(f"  [SUBS] Failed to extract {label}:")
            print(f"  {result.stderr[-300:]}")
            sub_path.unlink(missing_ok=True)

    if extracted:
        print(f"  [SUBS] {len(extracted)} subtitle file(s) extracted.")
    else:
        print(f"  [SUBS] No subtitle tracks found in {mkv_path.name}.")

    return extracted


# ---------------------------------------------------------------------------
# Video conversion
# ---------------------------------------------------------------------------

def convert_file(
    ffmpeg_exe: str,
    ffprobe_exe: str,
    mkv_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """
    Extract subtitles then convert a single MKV to MP4.
    Returns ffmpeg exit code (0 on success or skip).
    """
    mp4_path = mkv_path.with_suffix(".mp4")

    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Processing: {mkv_path.name}")

    # Always extract subs first — before any potential MKV deletion
    extract_subtitles(ffmpeg_exe, ffprobe_exe, mkv_path, dry_run)

    if mp4_path.exists() and not overwrite:
        print(f"  SKIP: {mp4_path.name} already exists (use --overwrite to replace).")
        return 0

    cmd = [
        ffmpeg_exe,
        "-y" if overwrite else "-n",
        "-i", str(mkv_path),
        "-c:v", "h264_nvenc",
        "-rc:v", "vbr",
        "-cq", "19",
        "-preset", "p4",
        "-b:v", "0",
        "-c:a", "aac",
        "-profile:a", "aac_low",
        "-ac", "2",
        "-ar", "48000",
        "-b:a", "256k",
        "-movflags", "+faststart",
        str(mp4_path),
    ]

    print(f"  Converting → {mp4_path.name}")
    if dry_run:
        print("  (dry-run) Command:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print("  ffmpeg output:\n", proc.stdout[-1000:])
        print(f"  ERROR: ffmpeg failed (exit {proc.returncode})")
    else:
        print("  Done.")
        if overwrite:
            try:
                mkv_path.unlink()
                print(f"  Deleted original MKV: {mkv_path.name}")
            except Exception as e:
                print(f"  WARNING: Could not delete {mkv_path}: {e}")

    return proc.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert .mkv files to .mp4 (h264_nvenc), "
            "extracting embedded subtitles as sidecar files first."
        )
    )
    parser.add_argument(
        "folder", type=Path,
        help="Root folder to scan recursively for .mkv files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .mp4 files and delete MKVs after successful conversion.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without running ffmpeg.",
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

    failures = 0
    for mkv in mkv_files:
        rc = convert_file(ffmpeg_exe, ffprobe_exe, mkv, args.overwrite, args.dry_run)
        if rc != 0:
            failures += 1

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        sys.exit(1)
    else:
        print("\nAll conversions completed successfully.")


if __name__ == "__main__":
    main()
