#!/usr/bin/env python3
"""
Segment a concatenated "Play All" disc rip into individual episodes.

Some DVD rips come out as a single multi-hour title (the whole disc back to
back) instead of one title per episode, so Sort_TV skips them (over
--max-minutes). This splits that file into episode-length pieces along the
disc's chapter marks, losslessly (ffmpeg stream copy), so the normal pipeline
can then identify each episode.

How it picks boundaries: DVD authoring places a chapter mark at each episode
start (plus act/eyecatch marks within), so episode boundaries are a subset of
chapter boundaries. We estimate the episode count from the total runtime and a
target episode length, then snap each evenly-spaced cut to the nearest chapter
boundary. No content analysis needed. The boundary planner is pure and is
covered by tests/test_segmentation.py against cached Pokemon chapter data.

Usage:
  python3 Segment_Disc.py --input disc.mkv --out ./episodes/ [--episode-minutes 22]
  python3 Segment_Disc.py --input disc.mkv --dry-run        # just print the plan
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

Chapter = Tuple[float, float]   # (start_seconds, end_seconds)
Segment = Tuple[float, float]   # (start_seconds, end_seconds)

DEFAULT_EPISODE_MINUTES = 22.0
# Only segment files longer than this; shorter ones are already episode-sized.
DEFAULT_MIN_SPLIT_MINUTES = 60.0


def plan_episode_segments(chapters: List[Chapter], target_seconds: float) -> List[Segment]:
    """Group chapters into episode-length segments, cutting only on chapter
    boundaries. Pure and deterministic.

    Estimates the episode count K = round(total / target), spaces K cuts evenly
    across the runtime, and snaps each internal cut to the nearest chapter
    boundary. Returns a list of (start, end) spans covering the whole file.
    """
    if not chapters:
        return []
    start0 = chapters[0][0]
    end_n = chapters[-1][1]
    total = end_n - start0
    if total <= 0:
        return [(start0, end_n)]

    k = max(1, round(total / target_seconds))
    if k <= 1:
        return [(start0, end_n)]

    # Candidate cut points: every chapter boundary strictly inside the file.
    internal = sorted({c[0] for c in chapters} | {c[1] for c in chapters})
    internal = [b for b in internal if start0 < b < end_n]
    if not internal:
        return [(start0, end_n)]

    seg_len = total / k
    cuts: List[float] = []
    used = set()
    for i in range(1, k):
        target_t = start0 + i * seg_len
        remaining = [b for b in internal if b not in used]
        if not remaining:
            break
        nearest = min(remaining, key=lambda b: abs(b - target_t))
        cuts.append(nearest)
        used.add(nearest)

    points = [start0] + sorted(cuts) + [end_n]
    # Dedupe any coincident points to keep spans strictly increasing.
    uniq = [points[0]]
    for p in points[1:]:
        if p > uniq[-1]:
            uniq.append(p)
    return [(uniq[i], uniq[i + 1]) for i in range(len(uniq) - 1)]


def ffprobe_chapters(path: Path) -> List[Chapter]:
    cmd = ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    chapters = []
    for c in data.get("chapters", []):
        try:
            chapters.append((float(c["start_time"]), float(c["end_time"])))
        except (KeyError, ValueError):
            continue
    return chapters


def ffprobe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nw=1:nk=1", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def split_file(src: Path, segments: List[Segment], out_dir: Path,
               prefix: str, dry_run: bool = False) -> List[Path]:
    """Cut src into one file per segment with ffmpeg stream copy (lossless).
    Output names are neutral (``<prefix>_seg01.mkv``) so Sort_TV does not treat
    them as already-named episodes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for i, (start, end) in enumerate(segments, start=1):
        dst = out_dir / f"{prefix}_seg{i:02d}.mkv"
        dur = end - start
        if dry_run:
            print(f"  seg {i:02d}: {start/60:7.2f} -> {end/60:7.2f} min "
                  f"({dur/60:5.2f} min)  {dst.name}")
            written.append(dst)
            continue
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{dur:.3f}", "-map", "0", "-c", "copy",
            "-avoid_negative_ts", "make_zero", str(dst),
        ]
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        if rc != 0:
            print(f"  ERROR: ffmpeg failed on segment {i} ({dst.name})", file=sys.stderr)
            continue
        written.append(dst)
        print(f"  wrote {dst.name} ({dur/60:.2f} min)")
    return written


def segment_one(src: Path, out_dir: Path, episode_minutes: float,
                min_split_minutes: float, replace: bool, dry_run: bool) -> bool:
    """Segment a single file if it is an oversized, chaptered title. Returns
    True if it was segmented. With ``replace`` the original is deleted after a
    verified-complete split (never in dry-run, never on partial output)."""
    duration = ffprobe_duration(src)
    if duration and duration < min_split_minutes * 60:
        return False  # already episode-sized
    chapters = ffprobe_chapters(src)
    if not chapters:
        print(f"{src.name}: oversized ({duration/60:.1f} min) but no chapters; cannot segment.",
              file=sys.stderr)
        return False
    segments = plan_episode_segments(chapters, episode_minutes * 60.0)
    if len(segments) < 2:
        return False
    print(f"{src.name}: {duration/60:.1f} min, {len(chapters)} chapters -> {len(segments)} segments")
    written = split_file(src, segments, out_dir, src.stem, dry_run=dry_run)
    complete = len(written) == len(segments) and all(p.exists() for p in written)
    if replace and not dry_run and complete:
        src.unlink()
        print(f"  removed original {src.name}")
    elif replace and not dry_run and not complete:
        print(f"  WARNING: incomplete split, keeping original {src.name}", file=sys.stderr)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Concatenated disc .mkv to split")
    src.add_argument("--folder", type=Path,
                     help="Folder to scan: segment each oversized chaptered .mkv in place. "
                          "Safe no-op on folders with only episode-sized files (for the pipeline pre-step).")
    ap.add_argument("--out", type=Path, default=None, help="Output dir for --input (default: alongside input)")
    ap.add_argument("--episode-minutes", type=float, default=DEFAULT_EPISODE_MINUTES,
                    help=f"Target episode length (default: {DEFAULT_EPISODE_MINUTES:.0f})")
    ap.add_argument("--min-split-minutes", type=float, default=DEFAULT_MIN_SPLIT_MINUTES,
                    help=f"Only split files longer than this (default: {DEFAULT_MIN_SPLIT_MINUTES:.0f})")
    ap.add_argument("--replace", action="store_true",
                    help="Delete the original after a verified-complete split (--folder mode).")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, do not write files")
    args = ap.parse_args()

    if args.input:
        if not args.input.is_file():
            ap.error(f"input not found: {args.input}")
        segmented = segment_one(args.input, args.out or args.input.parent,
                                args.episode_minutes, args.min_split_minutes,
                                replace=args.replace, dry_run=args.dry_run)
        if not segmented:
            print(f"{args.input.name}: nothing to segment (under threshold or no chapters).")
        return

    # --folder mode: segment every oversized chaptered .mkv in the folder in place.
    if not args.folder.is_dir():
        ap.error(f"folder not found: {args.folder}")
    acted = 0
    for mkv in sorted(p for p in args.folder.glob("*.mkv") if p.is_file()):
        if segment_one(mkv, args.folder, args.episode_minutes, args.min_split_minutes,
                       replace=args.replace, dry_run=args.dry_run):
            acted += 1
    if not acted:
        print(f"{args.folder}: no oversized chaptered titles to segment.")


if __name__ == "__main__":
    main()
