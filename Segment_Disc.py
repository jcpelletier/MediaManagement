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

When to split at all: a long title is only a Play-All if it is longer than one
episode of *that show*. In --folder mode the show/season is resolved from
sort_hints.json or the folder name and the season's episode runtimes come from
TMDB. A title that fits the season's runtime band (same band Sort_TV uses) is a
single episode and is left whole; a longer one is split at the season's median
episode length. Without TMDB data, only titles over --min-split-minutes (the
longest title Sort_TV accepts) are split, at --episode-minutes. Before this,
hour-long drama episodes (Outlander S6, 57-80 min) were cut into ~22 min pieces
that Sort_TV then routed to Extras.

Usage:
  python3 Segment_Disc.py --input disc.mkv --out ./episodes/ [--episode-minutes 22]
  python3 Segment_Disc.py --input disc.mkv --dry-run        # just print the plan
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Chapter = Tuple[float, float]   # (start_seconds, end_seconds)
Segment = Tuple[float, float]   # (start_seconds, end_seconds)

# Fallback episode length when the show's runtimes are unknown.
DEFAULT_EPISODE_MINUTES = 22.0
# Fallback split threshold when the show's runtimes are unknown. Matches the
# Sort_Shows job's Sort_TV --max-minutes, so we only split titles Sort_TV would
# reject anyway: an unknown-show split can never destroy a usable episode.
DEFAULT_MIN_SPLIT_MINUTES = 100.0
# Sort_TV treats [min*0.8, max*1.2] of the season's TMDB runtimes as episode
# length; anything within max*1.2 is a plausible single episode.
RUNTIME_BAND_MAX_FACTOR = 1.2
# Titles shorter than this are never Play-All concatenations (two of even the
# shortest episodes run longer), so no TMDB lookup is made for them.
LOOKUP_FLOOR_MINUTES = 35.0


def decide_split_target(duration_s: float, runtimes_min: Sequence[float],
                        min_split_minutes: float = DEFAULT_MIN_SPLIT_MINUTES,
                        episode_minutes: Optional[float] = None) -> Tuple[Optional[float], str]:
    """Decide whether a title of ``duration_s`` should be split, and at what
    episode length. Pure and deterministic.

    ``runtimes_min`` are the season's known episode runtimes (TMDB); empty when
    unknown. ``episode_minutes`` is an explicit override for the split length.
    Returns (target_seconds or None to leave whole, human-readable reason).
    """
    dur_m = duration_s / 60.0
    runtimes = [float(r) for r in runtimes_min if r and r > 0]
    if runtimes:
        band_max = max(runtimes) * RUNTIME_BAND_MAX_FACTOR
        if dur_m <= band_max:
            return None, (f"{dur_m:.1f} min fits a single episode "
                          f"(season runtimes up to {max(runtimes):.0f} min, band max {band_max:.1f})")
        target_m = episode_minutes or statistics.median(runtimes)
        return target_m * 60.0, (f"{dur_m:.1f} min exceeds single-episode band max "
                                 f"{band_max:.1f}; splitting at {target_m:.1f} min episodes")
    if dur_m < min_split_minutes:
        return None, (f"{dur_m:.1f} min, show runtimes unknown and under "
                      f"{min_split_minutes:.0f} min threshold")
    target_m = episode_minutes or DEFAULT_EPISODE_MINUTES
    return target_m * 60.0, (f"{dur_m:.1f} min, show runtimes unknown; "
                             f"splitting at {target_m:.1f} min episodes")


_YEAR_SUFFIX_RE = re.compile(r"\s+(?:19|20)\d{2}$")


def lookup_season_runtimes(folder: Path) -> Tuple[str, List[int]]:
    """Resolve the folder's show/season and return (label, TMDB episode
    runtimes in minutes). Returns ("", []) when anything is unavailable (no
    API key, unparseable folder, no TMDB match, network error), which makes the
    caller fall back to the conservative unknown-show threshold."""
    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if not api_key:
        return "", []
    try:
        # Imported lazily: Sort_TV pulls in the LLM client and other deps that
        # the pure planner (and its tests) do not need.
        from Sort_TV import (TmdbClient, find_sibling_disc_hints,
                             parse_show_and_season_from_folder)
    except Exception as e:
        print(f"  (runtime lookup unavailable: {e})", file=sys.stderr)
        return "", []

    show, season = None, None
    hints_path = folder / "sort_hints.json"
    for hints in (_read_json(hints_path), find_sibling_disc_hints(folder)):
        if hints and str(hints.get("show", "")).strip() and isinstance(hints.get("season"), int):
            show, season = str(hints["show"]).strip(), hints["season"]
            break
    if not show:
        show, season = parse_show_and_season_from_folder(folder.name)
    if not show or not isinstance(season, int) or season < 1:
        return "", []

    client = TmdbClient(api_key)
    try:
        # Folder names often carry the premiere year ("OUTLANDER_2014_SEASON_6"),
        # which TMDB's free-text search does not match; retry without it.
        candidates = [show]
        stripped = _YEAR_SUFFIX_RE.sub("", show)
        if stripped and stripped != show:
            candidates.append(stripped)
        for name in candidates:
            tmdb_show = client.find_tv(name)
            if not tmdb_show:
                continue
            eps = client.get_season_episodes(tmdb_show.tv_id, season)
            runtimes = [e.runtime for e in eps if e.runtime]
            if runtimes:
                return f"{tmdb_show.name} S{season:02d}", runtimes
    except Exception as e:
        print(f"  (TMDB runtime lookup failed: {e})", file=sys.stderr)
    return "", []


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


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


class RuntimeLookup:
    """Lazily resolves a folder's season runtimes once, on first need, so
    folders holding only short titles never hit TMDB."""

    def __init__(self, folder: Path, enabled: bool = True):
        self.folder = folder
        self.enabled = enabled
        self._result: Optional[Tuple[str, List[int]]] = None

    def get(self) -> Tuple[str, List[int]]:
        if self._result is None:
            self._result = lookup_season_runtimes(self.folder) if self.enabled else ("", [])
        return self._result


def segment_one(src: Path, out_dir: Path, episode_minutes: Optional[float],
                min_split_minutes: float, replace: bool, dry_run: bool,
                runtimes: Optional[RuntimeLookup] = None) -> bool:
    """Segment a single file if it is longer than one episode of its show and
    has chapters. Returns True if it was segmented. With ``replace`` the
    original is deleted after a verified-complete split (never in dry-run,
    never on partial output)."""
    duration = ffprobe_duration(src)
    if duration and duration < LOOKUP_FLOOR_MINUTES * 60:
        return False  # far too short to be a concatenation
    label, season_runtimes = runtimes.get() if runtimes else ("", [])
    target_s, reason = decide_split_target(duration, season_runtimes,
                                           min_split_minutes, episode_minutes)
    if target_s is None:
        if season_runtimes or duration >= 60 * 60:
            # Log the long-title decisions; quiet for ordinary short ones.
            print(f"{src.name}: not splitting: {reason}" + (f" [{label}]" if label else ""))
        return False
    chapters = ffprobe_chapters(src)
    if not chapters:
        print(f"{src.name}: oversized ({duration/60:.1f} min) but no chapters; cannot segment.",
              file=sys.stderr)
        return False
    segments = plan_episode_segments(chapters, target_s)
    if len(segments) < 2:
        return False
    print(f"{src.name}: {reason}" + (f" [{label}]" if label else ""))
    print(f"{src.name}: {duration/60:.1f} min, {len(chapters)} chapters -> {len(segments)} segments")
    # Prefix with the disc folder name so segments are globally unique: different
    # discs both rip to "A1_t00.mkv", and downstream per-file tracking keys on
    # the filename, so identical basenames across folders would collide.
    prefix = f"{src.parent.name} {src.stem}"
    written = split_file(src, segments, out_dir, prefix, dry_run=dry_run)
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
    ap.add_argument("--episode-minutes", type=float, default=None,
                    help="Split length override. Default: the season's median TMDB runtime, "
                         f"or {DEFAULT_EPISODE_MINUTES:.0f} when unknown.")
    ap.add_argument("--min-split-minutes", type=float, default=DEFAULT_MIN_SPLIT_MINUTES,
                    help="When the show's runtimes are unknown, only split files longer than this "
                         f"(default: {DEFAULT_MIN_SPLIT_MINUTES:.0f})")
    ap.add_argument("--no-tmdb", action="store_true",
                    help="Skip the TMDB runtime lookup; use only the unknown-show thresholds.")
    ap.add_argument("--replace", action="store_true",
                    help="Delete the original after a verified-complete split (--folder mode).")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, do not write files")
    args = ap.parse_args()

    if args.input:
        if not args.input.is_file():
            ap.error(f"input not found: {args.input}")
        lookup = RuntimeLookup(args.input.parent, enabled=not args.no_tmdb)
        segmented = segment_one(args.input, args.out or args.input.parent,
                                args.episode_minutes, args.min_split_minutes,
                                replace=args.replace, dry_run=args.dry_run, runtimes=lookup)
        if not segmented:
            print(f"{args.input.name}: nothing to segment (episode-sized or no chapters).")
        return

    # --folder mode: segment every oversized chaptered .mkv in the folder in place.
    if not args.folder.is_dir():
        ap.error(f"folder not found: {args.folder}")
    lookup = RuntimeLookup(args.folder, enabled=not args.no_tmdb)
    acted = 0
    for mkv in sorted(p for p in args.folder.glob("*.mkv") if p.is_file()):
        if segment_one(mkv, args.folder, args.episode_minutes, args.min_split_minutes,
                       replace=args.replace, dry_run=args.dry_run, runtimes=lookup):
            acted += 1
    if not acted:
        print(f"{args.folder}: no oversized chaptered titles to segment.")


if __name__ == "__main__":
    main()
