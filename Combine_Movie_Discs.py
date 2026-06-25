#!/usr/bin/env python3
"""
Combine a movie that was ripped across multiple discs into one Jellyfin library entry.

This is a deliberate, human-triggered step — NOT part of the automatic
udev → rip → Process_Movies pipeline. A human (or Pandabot acting for one)
looks at the staging folders, decides which discs are the two halves of the
feature and which (if any) is a bonus-features disc, and names the movie. This
script then does the mechanical work safely.

Why human-triggered: two sibling folders that both fuzzy-match one title could
be "two halves to join", "feature + extras disc", or "a duplicate rip". The
order of the halves also matters — joining them backwards silently corrupts the
film. A heuristic can't tell these apart reliably, so the caller asserts the
disc order on the command line and this script trusts it.

Two output modes:
  - stack  (default): Jellyfin/Plex multi-part stacking. Each half is moved
            into the movie folder as "Title (Year) - partN.ext". No re-encode,
            no concat risk, and the players stitch them into one continuous
            playback. Codecs of the halves don't even have to match.
  - concat (--concat): stream-copy the halves into a single "Title (Year).ext".
            Only safe when the halves share codec/resolution/pixel-format/audio;
            this is checked via ffprobe and refused on mismatch unless --force.

Disc folders are passed in playback order (disc 1 first). Each --disc may be a
folder (the longest-runtime title in it is taken as that half's feature — see
--prefer) or a direct path to a video file. An optional --extras folder's video
files are moved into the movie's Extras/ subfolder.

Feature selection for a folder defaults to longest runtime rather than largest
byte size: on discs that expose multiple cuts via seamless branching (theatrical
/ special / extended), the halves are near-identical in size but differ in
duration, and "longest" reliably lands on the most complete cut.

Requires:
- ffmpeg / ffprobe on PATH
- requests + TMDB_API_KEY (optional — only for canonicalizing the title/year;
  verification is non-fatal and never blocks a human-directed combine)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_EXTENSIONS = [
    ".mkv", ".mp4", ".avi", ".mov", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".flv",
]

DEFAULT_DEST = Path("/mnt/media/Media/Movies")
DEFAULT_PROCESSED = Path("/mnt/media/Video/Processed")


# ─── small utilities ─────────────────────────────────────────────────────────

def run_cmd(cmd: list) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def sanitize_title(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', " ", title).strip()


def collect_video_files(folder: Path, extensions: List[str]) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]


def ffprobe_duration_seconds(video: Path) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video),
    ]
    rc, out, _ = run_cmd(cmd)
    if rc != 0:
        return None
    try:
        return float(out.strip())
    except ValueError:
        return None


def resolve_disc_feature(disc: Path, extensions: List[str], prefer: str = "longest") -> Optional[Path]:
    """A --disc arg is either a video file (used as-is) or a folder. For a
    folder, pick the feature half by longest runtime (default) — duration is a
    more reliable signal than byte size on seamless-branching discs that expose
    multiple cuts at near-identical sizes. Falls back to largest byte size when
    no durations are readable, or when prefer='largest'."""
    if disc.is_file():
        if disc.suffix.lower() not in extensions:
            print(f"  WARN: {disc.name} is not a recognized video extension; using it anyway")
        return disc
    if disc.is_dir():
        vids = collect_video_files(disc, extensions)
        if not vids:
            print(f"  ERROR: no video files in disc folder {disc}")
            return None
        if prefer == "largest":
            return max(vids, key=lambda p: p.stat().st_size)
        # prefer == "longest": choose by runtime, fall back to byte size
        timed = [(v, ffprobe_duration_seconds(v)) for v in vids]
        timed = [(v, d) for v, d in timed if d is not None and d > 0]
        if not timed:
            print(f"  WARN: no readable durations in {disc.name}; falling back to largest file size")
            return max(vids, key=lambda p: p.stat().st_size)
        return max(timed, key=lambda vd: vd[1])[0]
    print(f"  ERROR: disc path does not exist: {disc}")
    return None


# ─── ffprobe stream fingerprint (for concat compatibility) ────────────────────

@dataclass
class StreamFingerprint:
    vcodec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    pix_fmt: Optional[str] = None
    acodec: Optional[str] = None
    channels: Optional[int] = None

    def key(self) -> tuple:
        return (self.vcodec, self.width, self.height, self.pix_fmt, self.acodec, self.channels)

    def describe(self) -> str:
        return (f"v={self.vcodec} {self.width}x{self.height} {self.pix_fmt} "
                f"a={self.acodec} {self.channels}ch")


def probe_fingerprint(video: Path) -> Optional[StreamFingerprint]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,channels",
        "-of", "json",
        str(video),
    ]
    rc, out, _ = run_cmd(cmd)
    if rc != 0 or not out.strip():
        return None
    try:
        streams = json.loads(out).get("streams", []) or []
    except Exception:
        return None
    fp = StreamFingerprint()
    for s in streams:
        if s.get("codec_type") == "video" and fp.vcodec is None:
            fp.vcodec = s.get("codec_name")
            fp.width = s.get("width")
            fp.height = s.get("height")
            fp.pix_fmt = s.get("pix_fmt")
        elif s.get("codec_type") == "audio" and fp.acodec is None:
            fp.acodec = s.get("codec_name")
            fp.channels = s.get("channels")
    return fp


def check_concat_compatible(parts: List[Path]) -> Tuple[bool, str]:
    """Stream-copy concat needs identical video+audio formats across parts."""
    fps = []
    for p in parts:
        fp = probe_fingerprint(p)
        if fp is None:
            return False, f"could not ffprobe {p.name}"
        fps.append((p, fp))
    first = fps[0][1].key()
    for p, fp in fps[1:]:
        if fp.key() != first:
            return False, (
                f"format mismatch: {fps[0][0].name} [{fps[0][1].describe()}] "
                f"vs {p.name} [{fp.describe()}]"
            )
    return True, fps[0][1].describe()


# ─── TMDB canonicalization (optional, non-fatal) ──────────────────────────────

def canonicalize_with_tmdb(title: str, year: Optional[int]) -> Tuple[str, Optional[int]]:
    """Best-effort: return TMDB's canonical (title, year). On any failure the
    human-supplied values are returned unchanged — this never blocks."""
    key = os.environ.get("TMDB_API_KEY") or os.environ.get("TMDB_KEY")
    if not key:
        print("  [TMDB ] no TMDB_API_KEY set — using supplied title/year as-is")
        return title, year
    try:
        import requests
    except ImportError:
        print("  [TMDB ] requests not installed — using supplied title/year as-is")
        return title, year
    params = {"api_key": key, "query": title, "include_adult": "false"}
    if year:
        params["primary_release_year"] = str(year)
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception as e:
        print(f"  [TMDB ] lookup failed ({e}) — using supplied title/year as-is")
        return title, year
    if not results:
        print(f"  [TMDB ] no match for '{title}' — using supplied title/year as-is")
        return title, year
    top = results[0]
    canon_title = top.get("title") or top.get("original_title") or title
    release_date = top.get("release_date") or ""
    canon_year = year
    if release_date[:4].isdigit():
        canon_year = int(release_date[:4])
    if (canon_title, canon_year) != (title, year):
        print(f"  [TMDB ] canonicalized to '{canon_title}' ({canon_year})")
    else:
        print(f"  [TMDB ] confirmed '{canon_title}' ({canon_year})")
    return canon_title, canon_year


# ─── results ──────────────────────────────────────────────────────────────────

@dataclass
class Result:
    title: str
    year: Optional[int]
    mode: str
    parts: List[str] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)
    dry_run: bool = False
    error: Optional[str] = None


# ─── combine modes ─────────────────────────────────────────────────────────────

def do_stack(feature_files: List[Path], movie_dir: Path, base_name: str,
             overwrite: bool, dry_run: bool) -> List[str]:
    """Move each half into movie_dir as 'base_name - partN.ext'."""
    placed = []
    for i, src in enumerate(feature_files, start=1):
        dest_name = f"{base_name} - part{i}{src.suffix.lower()}"
        dest = movie_dir / dest_name
        print(f"  part{i}: {src.name}  ->  {dest}")
        if dest.exists() and not overwrite:
            print(f"  SKIP: destination exists (use --overwrite): {dest}")
            placed.append(dest_name)
            continue
        if not dry_run:
            movie_dir.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
        placed.append(dest_name)
    return placed


def do_concat(feature_files: List[Path], movie_dir: Path, base_name: str,
              overwrite: bool, dry_run: bool, force: bool) -> Tuple[List[str], Optional[str]]:
    """Stream-copy concat the halves into a single 'base_name.ext'."""
    ok, detail = check_concat_compatible(feature_files)
    if not ok:
        msg = f"concat incompatible: {detail}"
        if not force:
            print(f"  ERROR: {msg}")
            print("         Halves don't share a format — re-encode first, or use stacking (drop --concat).")
            return [], msg
        print(f"  WARN: {msg} — proceeding anyway because --force was given")
    else:
        print(f"  concat compatible: {detail}")

    suffix = feature_files[0].suffix.lower()
    dest_name = f"{base_name}{suffix}"
    dest = movie_dir / dest_name
    print(f"  concat -> {dest}")
    for i, f in enumerate(feature_files, start=1):
        print(f"    + part{i}: {f.name}")

    if dest.exists() and not overwrite:
        return [], f"destination exists (use --overwrite): {dest}"

    if dry_run:
        return [dest_name], None

    movie_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as lf:
        list_path = Path(lf.name)
        for f in feature_files:
            # ffmpeg concat demuxer: single-quote the path, escape embedded quotes
            esc = str(f.resolve()).replace("'", "'\\''")
            lf.write(f"file '{esc}'\n")
    try:
        # -map 0 keeps ALL streams (every audio track, subtitles); without it
        # ffmpeg's default selection keeps only one audio track per output.
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", str(list_path), "-map", "0", "-c", "copy", str(dest)]
        rc, _, err = run_cmd(cmd)
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass
    if rc != 0 or not dest.exists():
        return [], f"ffmpeg concat failed: {err.strip().splitlines()[-1] if err.strip() else 'unknown error'}"
    return [dest_name], None


def move_extras(extras_dir_src: Optional[Path], movie_dir: Path, extensions: List[str],
                overwrite: bool, dry_run: bool) -> List[str]:
    if not extras_dir_src:
        return []
    if not extras_dir_src.is_dir():
        print(f"  WARN: extras path is not a folder, skipping: {extras_dir_src}")
        return []
    vids = collect_video_files(extras_dir_src, extensions)
    if not vids:
        print(f"  WARN: no video files in extras folder {extras_dir_src}")
        return []
    extras_dir = movie_dir / "Extras"
    moved = []
    for ef in sorted(vids):
        dest = extras_dir / ef.name
        print(f"  extra: {ef.name} -> {dest}")
        if dest.exists() and not overwrite:
            print(f"  SKIP extra: already exists (use --overwrite): {dest}")
            moved.append(ef.name)
            continue
        if not dry_run:
            extras_dir.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            shutil.move(str(ef), str(dest))
        moved.append(ef.name)
    return moved


def move_source_to_processed(disc_paths: List[Path], extras_src: Optional[Path],
                             processed_root: Path, dry_run: bool) -> None:
    """Move the *source disc folders* (not individual files) to Processed so the
    automatic Process_Movies run won't re-grab them. Files already pulled into
    the library are simply gone from these folders."""
    folders = set()
    for d in disc_paths:
        folders.add(d if d.is_dir() else d.parent)
    if extras_src and extras_src.is_dir():
        folders.add(extras_src)
    for folder in sorted(folders):
        if not folder.exists():
            continue
        target = processed_root / folder.name
        if folder.resolve() == target.resolve():
            continue
        if target.exists():
            print(f"  SKIP processed move: already exists: {target}")
            continue
        print(f"  processed: {folder} -> {target}")
        if not dry_run:
            try:
                processed_root.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(target))
            except Exception as exc:
                print(f"  ERROR: could not move source folder '{folder}': {exc}")


# ─── orchestration ─────────────────────────────────────────────────────────────

def combine(args: argparse.Namespace, extensions: List[str]) -> Result:
    title, year = args.title, args.year
    if not args.no_verify:
        title, year = canonicalize_with_tmdb(title, year)

    safe_title = sanitize_title(title)
    if not safe_title:
        return Result(title=title, year=year, mode=args.mode, dry_run=args.dry_run,
                      error="empty title after sanitization")
    base_name = f"{safe_title} ({year})" if year else safe_title
    movie_dir = args.dest / base_name

    # Resolve each disc to its feature file, preserving the caller's order.
    feature_files: List[Path] = []
    for disc in args.disc:
        feat = resolve_disc_feature(disc, extensions, args.prefer)
        if feat is None:
            return Result(title=title, year=year, mode=args.mode, dry_run=args.dry_run,
                          error=f"could not resolve a feature file for disc: {disc}")
        size_gb = feat.stat().st_size / (1024 ** 3)
        dur = ffprobe_duration_seconds(feat)
        dur_str = f"{dur / 60:.1f} min" if dur else "unknown runtime"
        print(f"  disc {len(feature_files) + 1}: {feat}  ({size_gb:.2f} GB, {dur_str})")
        feature_files.append(feat)

    print(f"\nMovie    : {base_name}")
    print(f"Mode     : {args.mode}")
    print(f"Dest     : {movie_dir}")
    print(f"Dry run  : {args.dry_run}\n")

    if movie_dir.exists() and not args.overwrite and not args.dry_run:
        # Don't clobber an existing library entry by surprise.
        existing = list(movie_dir.iterdir())
        if existing:
            return Result(title=title, year=year, mode=args.mode, dry_run=args.dry_run,
                          error=f"movie folder already exists and is non-empty (use --overwrite): {movie_dir}")

    if args.mode == "concat":
        parts, err = do_concat(feature_files, movie_dir, base_name,
                               args.overwrite, args.dry_run, args.force)
        if err:
            return Result(title=title, year=year, mode=args.mode, dry_run=args.dry_run, error=err)
    else:
        parts = do_stack(feature_files, movie_dir, base_name, args.overwrite, args.dry_run)

    extras = move_extras(args.extras, movie_dir, extensions, args.overwrite, args.dry_run)

    if not args.keep_source:
        move_source_to_processed(args.disc, args.extras, args.processed, args.dry_run)

    return Result(title=title, year=year, mode=args.mode, parts=parts,
                  extras=extras, dry_run=args.dry_run)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine a multi-disc movie rip into one Jellyfin library entry "
                    "(multi-part stacking by default, or --concat for a single file)."
    )
    p.add_argument("--title", required=True, help="Movie title, e.g. \"Avatar\"")
    p.add_argument("--year", type=int, default=None, help="Release year, e.g. 2009")
    p.add_argument("--disc", action="append", required=True, type=Path, metavar="PATH",
                   help="Disc folder or video file for one half, IN PLAYBACK ORDER. "
                        "Repeat for each half (--disc D1 --disc D2 ...).")
    p.add_argument("--extras", type=Path, default=None,
                   help="Optional bonus-features disc folder; its videos go to Extras/.")
    p.add_argument("--prefer", choices=["longest", "largest"], default="longest",
                   help="When a --disc is a folder, pick the feature half by longest "
                        "runtime (default) or largest byte size. Ignored when --disc "
                        "is a direct file path.")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help=f"Movie library root (default: {DEFAULT_DEST})")
    p.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED,
                   help=f"Where source disc folders are moved after combining "
                        f"(default: {DEFAULT_PROCESSED})")
    p.add_argument("--concat", dest="mode", action="store_const", const="concat", default="stack",
                   help="Stream-copy the halves into one file instead of multi-part stacking. "
                        "Requires matching codecs/resolution across halves.")
    p.add_argument("--force", action="store_true",
                   help="With --concat, proceed even if the halves' formats don't match (risky).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing destination files.")
    p.add_argument("--keep-source", action="store_true",
                   help="Don't move the source disc folders to Processed afterward.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip TMDB title/year canonicalization; use --title/--year verbatim.")
    p.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTENSIONS),
                   help="Comma-separated video extensions to consider.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned actions without moving or writing anything.")
    p.add_argument("--summary-json", type=str, default=None,
                   help="Write a JSON summary of the result to this path (for notifications).")
    args = p.parse_args()
    if len(args.disc) < 2:
        p.error("provide at least two --disc paths (the halves to combine)")
    return args


def main() -> None:
    args = parse_args()
    extensions = [e.lower() if e.startswith(".") else f".{e.lower()}"
                  for e in args.extensions.split(",") if e.strip()]

    result = combine(args, extensions)

    if args.summary_json:
        try:
            with open(args.summary_json, "w") as f:
                json.dump({
                    "title": result.title,
                    "year": result.year,
                    "mode": result.mode,
                    "parts": result.parts,
                    "extras": result.extras,
                    "dry_run": result.dry_run,
                    "error": result.error,
                }, f, indent=2)
        except Exception as exc:
            print(f"WARN: could not write summary JSON: {exc}")

    if result.error:
        print(f"\nFAILED: {result.error}")
        sys.exit(1)

    verb = "Would combine" if result.dry_run else "Combined"
    name = f"{result.title} ({result.year})" if result.year else result.title
    print(f"\n{verb}: {name}  [{result.mode}]")
    for part in result.parts:
        print(f"  • {part}")
    if result.extras:
        print(f"  + {len(result.extras)} extra(s) -> Extras/")


if __name__ == "__main__":
    main()
