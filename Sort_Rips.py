#!/usr/bin/env python3
"""
Sort ripped movie folders by asking an LLM to guess the movie title.

Evidence tiers (cheapest/fastest first):
1. Folder name + video filenames/sizes  -> Claude first pass
2. + subtitle text (.srt in folder or embedded subtitle track)
3. + local Whisper audio transcription (up to 3 attempts, doubling duration each retry,
     starting at --whisper-start-seconds into the file)
4. TMDB verification to confirm/correct title and year

Requires:
- ffprobe / ffmpeg on PATH
- ANTHROPIC_API_KEY env var
- pip install anthropic pydantic requests
- pip install faster-whisper   (only when Whisper fallback is enabled)
- TMDB_KEY env var or --tmdb-api-key  (for TMDB verification)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import anthropic
import requests
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _WhisperModel = None
    _FASTER_WHISPER_AVAILABLE = False


DEFAULT_EXTENSIONS = [
    ".mkv", ".mp4", ".avi", ".mov", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".flv",
]

WHISPER_INTERVAL_SECONDS_DEFAULT = 300.0  # sample at 5, 10, 15 min (1×, 2×, 3× this value)
WHISPER_BASE_SECONDS_DEFAULT     = 60.0   # initial clip duration; doubles each retry (60, 120, 240s)
WHISPER_MAX_ATTEMPTS             = 3      # attempts at 5 min, 10 min, 15 min


# ─── data classes ────────────────────────────────────────────────────────────

@dataclass
class FolderGuess:
    title: str
    year: Optional[int]
    confidence: float


# ─── small utilities ─────────────────────────────────────────────────────────

def run_cmd(cmd: list) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def norm_title(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def similarity(a: str, b: str) -> float:
    a_n, b_n = norm_title(a), norm_title(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def parse_extensions(ext_list: Iterable[str]) -> List[str]:
    return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext_list]


def collect_video_files(folder: Path, extensions: List[str]) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]


def format_files_for_prompt(files: List[Path]) -> str:
    lines = []
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines.append(f"- {f.name} ({size_mb:.1f} MB)")
    return "\n".join(lines)


def sanitize_title(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', " ", title).strip()


# ─── subtitle extraction ─────────────────────────────────────────────────────

def ffprobe_subtitle_streams(video_path: Path) -> list:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name,codec_type:stream_tags=language,title",
        "-of", "json",
        str(video_path),
    ]
    rc, out, _ = run_cmd(cmd)
    if rc != 0 or not out.strip():
        return []
    try:
        return json.loads(out).get("streams", []) or []
    except Exception:
        return []


def pick_best_subtitle_stream(streams: list) -> Optional[int]:
    if not streams:
        return None
    preferred = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
    for s in streams:
        if (s.get("codec_name") or "").lower() in preferred:
            return s.get("index")
    return streams[0].get("index")


def subtitles_are_bitmap_only(streams: list) -> bool:
    if not streams:
        return False
    bitmap = {"hdmv_pgs_subtitle", "dvd_subtitle", "pgssub", "vobsub"}
    codecs = {(s.get("codec_name") or "").lower() for s in streams}
    return bool(codecs) and all(c in bitmap for c in codecs)


def _parse_srt_text(raw: str, max_lines: int = 80) -> Optional[str]:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+", line) or "-->" in line or not line:
            continue
        line = re.sub(r"</?[ibu]>", "", line)
        line = re.sub(r"\{\\.*?\}", "", line)
        line = re.sub(r"^\[.*?\]\s*|^\(.*?\)\s*", "", line)
        if line:
            lines.append(line)
            if len(lines) >= max_lines:
                break
    result = "\n".join(lines).strip()
    return result if result else None


def extract_srt_file_excerpt(srt_path: Path, max_lines: int = 80) -> Optional[str]:
    try:
        txt = srt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            txt = srt_path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return None
    return _parse_srt_text(txt, max_lines)


def extract_embedded_subtitle_excerpt(video_path: Path, max_lines: int = 80) -> Optional[str]:
    streams = ffprobe_subtitle_streams(video_path)
    if not streams or subtitles_are_bitmap_only(streams):
        return None
    idx = pick_best_subtitle_stream(streams)
    if idx is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        srt_path = Path(td) / "subs.srt"
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-map", f"0:{idx}", str(srt_path)]
        rc, _, _ = run_cmd(cmd)
        if rc != 0 or not srt_path.exists():
            return None
        return extract_srt_file_excerpt(srt_path, max_lines)


def get_subtitle_evidence(folder: Path, video_file: Path, max_lines: int = 80) -> Optional[str]:
    """Check .srt files in the folder first, then embedded subtitle tracks."""
    stem = video_file.stem.lower()
    # Prefer .srt files whose stem matches the video filename
    for srt in sorted(folder.glob("*.srt")):
        if srt.stem.lower().startswith(stem) or srt.stem.lower() == stem:
            excerpt = extract_srt_file_excerpt(srt, max_lines)
            if excerpt:
                print(f"  [SUBS ] Using .srt file: {srt.name}")
                return excerpt
    # Any .srt in the folder
    for srt in sorted(folder.glob("*.srt")):
        excerpt = extract_srt_file_excerpt(srt, max_lines)
        if excerpt:
            print(f"  [SUBS ] Using .srt file: {srt.name}")
            return excerpt
    # Embedded subtitle track
    excerpt = extract_embedded_subtitle_excerpt(video_file, max_lines)
    if excerpt:
        print(f"  [SUBS ] Using embedded subtitle track from: {video_file.name}")
    return excerpt


# ─── audio / Whisper ─────────────────────────────────────────────────────────

def ffprobe_duration_seconds(video_path: Path) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    rc, out, _ = run_cmd(cmd)
    if rc != 0:
        return None
    try:
        return float(out.strip())
    except ValueError:
        return None


def extract_audio_clip_wav(
    video_path: Path,
    out_wav: Path,
    start_seconds: float,
    duration_seconds: float,
) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.3f}",
        "-t", f"{duration_seconds:.3f}",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        str(out_wav),
    ]
    rc, _, _ = run_cmd(cmd)
    return rc == 0 and out_wav.exists() and out_wav.stat().st_size > 0


def transcribe_with_faster_whisper(model: "_WhisperModel", wav_path: Path) -> Optional[str]:
    segments, _ = model.transcribe(str(wav_path), beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    return text if text else None


# ─── TMDB movie verification ─────────────────────────────────────────────────

@dataclass
class TmdbMovie:
    movie_id: int
    title: str
    year: Optional[int]


class TmdbMovieClient:
    def __init__(self, api_key: str, timeout_s: float = 10.0):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.base = "https://api.themoviedb.org/3"

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base}{path}"
        params = dict(params)
        params["api_key"] = self.api_key
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[TmdbMovie]:
        params: dict = {"query": title, "include_adult": "false"}
        if year:
            params["primary_release_year"] = str(year)
        try:
            data = self._get("/search/movie", params)
        except Exception as e:
            print(f"  [TMDB ] Search failed: {e}")
            return None
        results = data.get("results") or []
        if not results:
            return None
        top = results[0]
        movie_id = top.get("id")
        canon_title = top.get("title") or top.get("original_title") or ""
        release_date = top.get("release_date") or ""
        release_year: Optional[int] = None
        if release_date:
            try:
                release_year = int(release_date[:4])
            except (ValueError, IndexError):
                pass
        if not movie_id or not canon_title:
            return None
        return TmdbMovie(movie_id=int(movie_id), title=str(canon_title), year=release_year)


def verify_movie_with_tmdb(
    tmdb: TmdbMovieClient,
    guess: FolderGuess,
    min_match: float,
) -> Optional[FolderGuess]:
    """Return a (possibly corrected) FolderGuess if TMDB confirms, else None."""
    result = tmdb.search_movie(guess.title, guess.year)
    if result is None:
        result = tmdb.search_movie(guess.title)  # retry without year constraint
    if result is None:
        print(f"  [TMDB ] No results for '{guess.title}'")
        return None

    score = similarity(guess.title, result.title)
    if score < min_match:
        print(
            f"  [TMDB ] Title mismatch: '{guess.title}' vs '{result.title}' "
            f"(score={score:.2f} < {min_match:.2f})"
        )
        return None

    corrected = (
        norm_title(guess.title) != norm_title(result.title)
        or guess.year != result.year
    )
    if corrected:
        print(
            f"  [TMDB ] Corrected: '{guess.title}' ({guess.year}) "
            f"-> '{result.title}' ({result.year})"
        )
    else:
        print(f"  [TMDB ] Confirmed: '{result.title}' ({result.year}) (score={score:.2f})")

    return FolderGuess(title=result.title, year=result.year, confidence=guess.confidence)


# ─── Claude ──────────────────────────────────────────────────────────────────

def call_claude(
    client: anthropic.Anthropic,
    folder_name: str,
    files_summary: str,
    evidence_text: Optional[str] = None,
) -> Optional[FolderGuess]:
    class _MovieGuess(BaseModel):
        title: str
        year: Optional[int] = None
        confidence: float

    user_content = f"Folder: {folder_name}\nFiles:\n{files_summary}"
    if evidence_text:
        user_content += f"\n\nAdditional evidence (subtitles / audio transcript):\n{evidence_text}"

    try:
        response = client.messages.parse(
            model="claude-sonnet-4-5",
            max_tokens=256,
            system=(
                "You are a movie identifier. Given a folder name, video files, and optionally "
                "subtitle or audio transcript evidence, return the most likely movie title and "
                "release year. If unsure, return an empty title and confidence 0."
            ),
            messages=[{"role": "user", "content": user_content}],
            output_format=_MovieGuess,
        )
    except anthropic.APIError as exc:
        print(f"  ERROR: Anthropic request failed for '{folder_name}': {exc}")
        return None

    data = response.parsed_output
    title = (data.title or "").strip()
    year = data.year
    confidence = float(data.confidence or 0)

    if year is not None:
        try:
            y = int(year)
            year = y if 1800 <= y <= 2100 else None
        except (TypeError, ValueError):
            year = None

    return FolderGuess(title=title, year=year, confidence=confidence)


# ─── rename / move ───────────────────────────────────────────────────────────

def rename_and_move(
    largest_file: Path,
    guess: FolderGuess,
    dest_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> Optional[str]:
    """Return the destination filename on success (including dry-run), None if skipped."""
    title = sanitize_title(guess.title)
    if not title:
        print(f"SKIP: Empty title after sanitization for {largest_file.parent.name}")
        return None

    new_name = f"{title}{f' ({guess.year})' if guess.year else ''}{largest_file.suffix}"
    renamed_path = largest_file.with_name(new_name)
    dest_path = dest_root / new_name

    if dest_path.exists() and not overwrite:
        print(f"SKIP: Destination exists, use --overwrite to replace: {dest_path}")
        return None

    print(f"  Rename: {largest_file.name} -> {renamed_path.name}")
    print(f"  Move  : {renamed_path} -> {dest_path}")

    if dry_run:
        return new_name

    if not dest_root.exists():
        dest_root.mkdir(parents=True, exist_ok=True)

    try:
        if largest_file != renamed_path:
            largest_file.rename(renamed_path)
        if dest_path.exists() and overwrite:
            dest_path.unlink()
        shutil.move(str(renamed_path), str(dest_path))
        return new_name
    except Exception as exc:
        print(f"  ERROR: Could not move '{largest_file}': {exc}")
        return None


def move_folder_to_processed(folder: Path, processed_root: Path, dry_run: bool) -> None:
    if not any(folder.iterdir()):
        print(f"  Delete empty folder: {folder}")
        if not dry_run:
            try:
                folder.rmdir()
            except Exception as exc:
                print(f"  ERROR: Could not delete empty folder '{folder}': {exc}")
        return

    target = processed_root / folder.name
    if folder.resolve() == target.resolve():
        return
    if target.exists():
        print(f"SKIP: Processed destination already exists: {target}")
        return

    print(f"  Processed move: {folder} -> {target}")
    if dry_run:
        return

    try:
        processed_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(target))
    except Exception as exc:
        print(f"  ERROR: Could not move folder '{folder}': {exc}")


# ─── finalize (TMDB verify then rename) ──────────────────────────────────────

def _finalize(
    largest_file: Path,
    guess: FolderGuess,
    dest_root: Path,
    overwrite: bool,
    dry_run: bool,
    tmdb_client: Optional[TmdbMovieClient],
    tmdb_min_title_match: float,
) -> Optional[str]:
    """Return the destination filename on success, None if skipped."""
    if tmdb_client is not None:
        verified = verify_movie_with_tmdb(tmdb_client, guess, tmdb_min_title_match)
        if verified is None:
            print(f"SKIP: TMDB could not confirm '{guess.title}' — preventing bad rename")
            return None
        guess = verified
    return rename_and_move(largest_file, guess, dest_root, overwrite, dry_run)


# ─── main processing ─────────────────────────────────────────────────────────

def process_folder(
    folder: Path,
    extensions: List[str],
    anthropic_client: anthropic.Anthropic,
    min_confidence: float,
    dest_root: Path,
    overwrite: bool,
    dry_run: bool,
    srt_max_lines: int = 80,
    whisper_model: Optional["_WhisperModel"] = None,
    whisper_interval_seconds: float = WHISPER_INTERVAL_SECONDS_DEFAULT,
    whisper_base_seconds: float = WHISPER_BASE_SECONDS_DEFAULT,
    tmdb_client: Optional[TmdbMovieClient] = None,
    tmdb_min_title_match: float = 0.78,
) -> Tuple[Optional[str], Optional[str]]:
    """Process one folder. Returns (moved_filename, skip_reason). Exactly one is non-None."""
    video_files = collect_video_files(folder, extensions)
    if not video_files:
        return None, "no video files"

    largest_file = max(video_files, key=lambda p: p.stat().st_size)
    files_summary = format_files_for_prompt(video_files)

    # ── tier 1 + 2: folder/file metadata + subtitles ─────────────────────────
    subtitle_text = get_subtitle_evidence(folder, largest_file, srt_max_lines)
    initial_evidence = subtitle_text  # may be None

    guess = call_claude(anthropic_client, folder.name, files_summary, evidence_text=initial_evidence)
    if not guess:
        return None, "LLM call failed"

    src = "subtitles" if subtitle_text else "filenames only"
    print(f"  [LLM  ] First pass ({src}): '{guess.title}' ({guess.year}) conf={guess.confidence:.2f}")

    if guess.confidence >= min_confidence:
        result = _finalize(largest_file, guess, dest_root, overwrite, dry_run, tmdb_client, tmdb_min_title_match)
        if result:
            return result, None
        return None, f"TMDB could not confirm '{guess.title}'"

    # ── tier 3: local Whisper audio fallback ─────────────────────────────────
    if whisper_model is None:
        reason = f"low confidence ({guess.confidence:.2f}) and Whisper not available"
        print(f"SKIP: Low confidence ({guess.confidence:.2f}) and Whisper not available")
        return None, reason

    dur_s = ffprobe_duration_seconds(largest_file)
    if dur_s is None:
        reason = f"low confidence ({guess.confidence:.2f}) — cannot read video duration"
        print(f"SKIP: {reason}")
        return None, reason

    audio_clips: List[str] = []

    for attempt in range(1, WHISPER_MAX_ATTEMPTS + 1):
        start = attempt * whisper_interval_seconds          # 5 min, 10 min, 15 min
        duration = whisper_base_seconds * (2 ** (attempt - 1))  # 60s, 120s, 240s

        if start + 1.0 >= dur_s:
            print(f"  [WHISPER] Attempt {attempt} skipped: start {start:.0f}s >= file duration {dur_s:.0f}s")
            break

        actual_duration = min(duration, dur_s - start - 1.0)
        print(f"  [WHISPER] Attempt {attempt}/{WHISPER_MAX_ATTEMPTS}: {actual_duration:.0f}s @ {start:.0f}s")

        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "clip.wav"
            ok = extract_audio_clip_wav(largest_file, wav_path, start, actual_duration)
            if not ok:
                print(f"  [WHISPER] Audio extraction failed")
            else:
                transcript = transcribe_with_faster_whisper(whisper_model, wav_path)
                if transcript:
                    preview = transcript[:120] + ("..." if len(transcript) > 120 else "")
                    print(f"  [WHISPER] Transcript: \"{preview}\"")
                    audio_clips.append(
                        f"AUDIO CLIP {attempt} ({actual_duration:.0f}s @ {start:.0f}s):\n{transcript}"
                    )
                else:
                    print(f"  [WHISPER] Empty transcript")

        # Build combined evidence: subtitles (if any) + all audio clips so far
        evidence_parts: List[str] = []
        if subtitle_text:
            evidence_parts.append(f"SUBTITLES:\n{subtitle_text}")
        if audio_clips:
            evidence_parts.append("\n\n".join(audio_clips))
        combined_evidence = "\n\n".join(evidence_parts) or None

        guess = call_claude(anthropic_client, folder.name, files_summary, evidence_text=combined_evidence)
        if not guess:
            break

        print(f"  [LLM  ] Attempt {attempt}: '{guess.title}' ({guess.year}) conf={guess.confidence:.2f}")

        if guess.confidence >= min_confidence:
            result = _finalize(largest_file, guess, dest_root, overwrite, dry_run, tmdb_client, tmdb_min_title_match)
            if result:
                return result, None
            return None, f"TMDB could not confirm '{guess.title}'"

    reason = f"low confidence ({guess.confidence:.2f}) after all Whisper attempts"
    print(f"SKIP: {reason} for '{folder.name}'")
    return None, reason


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename and move ripped movies using Claude, subtitle evidence, local Whisper, and TMDB."
    )
    parser.add_argument("--source", type=Path, default=Path(r"D:\\Video"),
                        help="Source root containing ripped folders (default: D:\\Video)")
    parser.add_argument("--dest", type=Path, default=Path(r"D:\\Media\\Movies"),
                        help="Destination root for renamed movies (default: D:\\Media\\Movies)")
    parser.add_argument("--processed", type=Path, default=None,
                        help="Folder to move processed source subfolders into (default: <source>/Processed).")
    parser.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTENSIONS),
                        help="Comma-separated video extensions to consider.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show actions without renaming or moving files.")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Minimum LLM confidence (0-1) required to rename (default: 0.6)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing files in the destination.")
    parser.add_argument("--srt-max-lines", type=int, default=80,
                        help="Max subtitle lines to include as evidence (default: 80)")

    # Whisper
    w = parser.add_argument_group("Whisper audio fallback (local faster-whisper)")
    w.add_argument("--no-whisper-fallback", action="store_true",
                   help="Disable local Whisper audio transcription fallback.")
    w.add_argument("--whisper-model", default="base",
                   help="faster-whisper model size (default: base). Options: tiny, base, small, medium, large-v3")
    w.add_argument("--whisper-device", default="cuda",
                   help="Device for faster-whisper (default: cuda). Use 'cpu' if no GPU.")
    w.add_argument("--whisper-interval-seconds", type=float, default=WHISPER_INTERVAL_SECONDS_DEFAULT,
                   help=f"Interval between sample points in seconds (default: {WHISPER_INTERVAL_SECONDS_DEFAULT:.0f}). Samples at 1×, 2×, 3× this value (5, 10, 15 min).")
    w.add_argument("--whisper-base-seconds", type=float, default=WHISPER_BASE_SECONDS_DEFAULT,
                   help=f"Initial clip duration in seconds; doubles each retry (default: {WHISPER_BASE_SECONDS_DEFAULT:.0f})")

    # TMDB
    t = parser.add_argument_group("TMDB verification")
    t.add_argument("--no-verify-api", action="store_true",
                   help="Disable TMDB title/year verification.")
    t.add_argument("--tmdb-api-key", default=None,
                   help="TMDB API key (or set TMDB_KEY env var).")
    t.add_argument("--tmdb-min-title-match", type=float, default=0.78,
                   help="Minimum title similarity to confirm via TMDB (default: 0.78)")

    parser.add_argument("--summary-json", type=str, default=None,
                        help="Write a JSON summary of moved/skipped folders to this path.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        api_key = os.environ["ANTHROPIC_API_KEY"]
    except KeyError:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(2)

    source_root: Path = args.source
    dest_root: Path = args.dest
    processed_root: Path = args.processed or (source_root / "Processed")
    extensions = parse_extensions([e.strip() for e in args.extensions.split(",") if e.strip()])

    if not source_root.exists() or not source_root.is_dir():
        print(f"ERROR: Source root is not a directory: {source_root}")
        sys.exit(2)

    anthropic_client = anthropic.Anthropic(api_key=api_key)

    # ── Whisper setup ─────────────────────────────────────────────────────────
    whisper_model = None
    whisper_enabled = not args.no_whisper_fallback
    if whisper_enabled:
        if not _FASTER_WHISPER_AVAILABLE:
            print("WARN: faster-whisper not installed; audio fallback disabled.")
            print("      Run: pip install faster-whisper")
            whisper_enabled = False
        else:
            # Try CUDA compute types first; if all fail (e.g. GPU unavailable in container)
            # fall back to CPU int8 which always works.
            if args.whisper_device == "cuda":
                candidates = [("cuda", "float16"), ("cuda", "int8"), ("cpu", "int8")]
            else:
                candidates = [("cpu", "int8")]
            print(f"Loading Whisper model '{args.whisper_model}' on {args.whisper_device}...")
            for device, compute_type in candidates:
                try:
                    whisper_model = _WhisperModel(
                        args.whisper_model,
                        device=device,
                        compute_type=compute_type,
                    )
                    print(f"Whisper model loaded (device={device}, compute_type={compute_type}).")
                    break
                except Exception as e:
                    next_idx = candidates.index((device, compute_type)) + 1
                    if next_idx < len(candidates):
                        nd, nc = candidates[next_idx]
                        print(f"  {device}/{compute_type} not available, retrying with {nd}/{nc}...")
                    else:
                        print(f"WARN: Failed to load Whisper model: {e}")
                        print("      Audio fallback disabled.")
                        whisper_enabled = False

    # ── TMDB setup ────────────────────────────────────────────────────────────
    tmdb_client = None
    verify_enabled = not args.no_verify_api
    if verify_enabled:
        tmdb_key = args.tmdb_api_key or os.environ.get("TMDB_KEY") or ""
        if not tmdb_key:
            print("WARN: TMDB verification enabled but TMDB_KEY not set; verification disabled.")
            verify_enabled = False
        else:
            tmdb_client = TmdbMovieClient(api_key=tmdb_key)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"Source   : {source_root}")
    print(f"Dest     : {dest_root}")
    print(f"Processed: {processed_root}")
    print(f"Dry run  : {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Min confidence: {args.min_confidence}")
    if whisper_enabled:
        base = args.whisper_base_seconds
        iv = args.whisper_interval_seconds
        print(
            f"Whisper  : ON  model={args.whisper_model} device={args.whisper_device}  "
            f"samples @ {iv:.0f}s/{iv*2:.0f}s/{iv*3:.0f}s  "
            f"durations {base:.0f}s/{base*2:.0f}s/{base*4:.0f}s"
        )
    else:
        print("Whisper  : OFF")
    print(f"TMDB     : {'ON  min_match=' + str(args.tmdb_min_title_match) if verify_enabled else 'OFF'}")

    processed_root_resolved = processed_root.resolve()
    subfolders = [
        p for p in source_root.iterdir()
        if p.is_dir() and p.resolve() != processed_root_resolved
    ]
    if not subfolders:
        print("No subdirectories found to process.")
        return

    moved_movies: List[str] = []
    skipped_folders: List[dict] = []

    for folder in sorted(subfolders):
        print(f"\nFolder: {folder.name}")
        moved, reason = process_folder(
            folder=folder,
            extensions=extensions,
            anthropic_client=anthropic_client,
            min_confidence=args.min_confidence,
            dest_root=dest_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            srt_max_lines=args.srt_max_lines,
            whisper_model=whisper_model,
            whisper_interval_seconds=args.whisper_interval_seconds,
            whisper_base_seconds=args.whisper_base_seconds,
            tmdb_client=tmdb_client,
            tmdb_min_title_match=args.tmdb_min_title_match,
        )
        if moved:
            moved_movies.append(moved)
        elif reason and reason != "no video files":
            skipped_folders.append({"folder": folder.name, "reason": reason})
        move_folder_to_processed(folder, processed_root, args.dry_run)

    if args.summary_json:
        summary = {
            "moved_movies": moved_movies,
            "skipped_folders": skipped_folders,
            "dry_run": args.dry_run,
            "total": len(moved_movies) + len(skipped_folders),
        }
        try:
            with open(args.summary_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary written to {args.summary_json}")
        except Exception as exc:
            print(f"WARN: Could not write summary JSON: {exc}")


if __name__ == "__main__":
    main()
