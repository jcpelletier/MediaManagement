#!/usr/bin/env python3
"""
Sort_TV.py

Path 2 (LLM-assisted) + API verification:
- LLM proposes episode_number/title from subtitles or audio transcript
- Then VERIFY using TMDB API:
    - search TV series
    - fetch season episode list
    - confirm or correct episode number/title
- Rename in-place to: "{Show} - SxxEyy - {Title}.mkv"

Requires:
- ffprobe / ffmpeg on PATH (FFmpeg)
- openai python package
- requests
- OPENAI_API_KEY env var set
- TMDB_API_KEY env var set (or pass --tmdb-api-key)

TMDB endpoints used:
- /3/search/tv
- /3/tv/{series_id}/season/{season_number}
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import requests
from openai import OpenAI


# ----------------------------
# Constants (fallback strategy)
# ----------------------------

PRIMARY_AUDIO_SECONDS = 120.0

# Shift audio sampling to 10 minutes in (600 seconds).
AUDIO_START_SECONDS_HARDCODED = 600.0

DEFAULT_FALLBACK_AUDIO_SECONDS = 300.0

INVALID_FILENAME_CHARS = r'<>:"/\\|?*'


# ----------------------------
# Small utilities
# ----------------------------

def sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(rf"[{re.escape(INVALID_FILENAME_CHARS)}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def run_cmd(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def one_line(s: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def hr():
    # ASCII-only separator (avoids UnicodeEncodeError / mojibake on Windows Jenkins consoles)
    print("-" * 60)


def short_path(p: Path, max_len: int = 90) -> str:
    s = str(p)
    return s if len(s) <= max_len else ("..." + s[-(max_len - 3):])


def norm_title(s: str) -> str:
    # Normalization for matching titles
    s = (s or "").lower().strip()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    a_n = norm_title(a)
    b_n = norm_title(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


# ----------------------------
# ffprobe helpers
# ----------------------------

def ffprobe_duration_seconds(mkv_path: Path) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(mkv_path)
    ]
    rc, out, _ = run_cmd(cmd)
    if rc != 0:
        return None
    out = out.strip()
    if not out:
        return None
    try:
        return float(out)
    except ValueError:
        return None


def ffprobe_subtitle_streams(mkv_path: Path) -> list[dict]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name,codec_type:stream_tags=language,title",
        "-of", "json",
        str(mkv_path)
    ]
    rc, out, _ = run_cmd(cmd)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
        return data.get("streams", []) or []
    except Exception:
        return []


def pick_best_subtitle_stream(streams: list[dict]) -> Optional[int]:
    if not streams:
        return None
    preferred = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
    for s in streams:
        codec = (s.get("codec_name") or "").lower()
        if codec in preferred:
            return s.get("index")
    return streams[0].get("index")


def subtitles_are_bitmap_only(streams: list[dict]) -> bool:
    if not streams:
        return False
    bitmap_codecs = {
        "hdmv_pgs_subtitle",
        "dvd_subtitle",
        "pgssub",
        "vobsub",
    }
    codecs = {(s.get("codec_name") or "").lower() for s in streams}
    return len(codecs) > 0 and all(c in bitmap_codecs for c in codecs)


def summarize_subtitle_streams(streams: list[dict]) -> str:
    if not streams:
        return "none"
    codecs = [(s.get("codec_name") or "?").lower() for s in streams]
    langs = [((s.get("tags") or {}) or {}).get("language") for s in streams]
    codec_set = sorted(set(codecs))
    lang_set = sorted(set(l for l in langs if l))
    if len(lang_set) > 1:
        lang_summary = f"{lang_set[0]} + {len(lang_set)-1} more"
    elif len(lang_set) == 1:
        lang_summary = lang_set[0]
    else:
        lang_summary = "unknown"
    return f"{len(streams)} stream(s) → {', '.join(codec_set)} ({lang_summary})"


def extract_subtitle_excerpt(
    mkv_path: Path,
    max_lines: int = 80,
    max_chars: int = 4000,
) -> Optional[str]:
    streams = ffprobe_subtitle_streams(mkv_path)
    if streams and subtitles_are_bitmap_only(streams):
        return None

    sub_stream_index = pick_best_subtitle_stream(streams)
    if sub_stream_index is None:
        return None

    with tempfile.TemporaryDirectory() as td:
        srt_path = Path(td) / "subs.srt"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(mkv_path),
            "-map", f"0:{sub_stream_index}",
            str(srt_path)
        ]
        rc, _, _ = run_cmd(cmd)
        if rc != 0 or not srt_path.exists():
            return None

        try:
            txt = srt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = srt_path.read_text(encoding="latin-1", errors="ignore")

        lines = []
        for raw_line in txt.splitlines():
            line = raw_line.strip()
            if re.fullmatch(r"\d+", line):
                continue
            if "-->" in line:
                continue
            if not line:
                continue
            line = re.sub(r"</?i>", "", line)
            line = re.sub(r"</?b>", "", line)
            line = re.sub(r"</?u>", "", line)
            line = re.sub(r"{\\.*?}", "", line)
            line = re.sub(r"^\[.*?\]\s*", "", line)
            line = re.sub(r"^\(.*?\)\s*", "", line)
            if line:
                lines.append(line)
            if len(lines) >= max_lines:
                break

        if not lines:
            return None

        excerpt = "\n".join(lines)[:max_chars].strip()
        return excerpt if excerpt else None


# ----------------------------
# Audio extraction + transcription
# ----------------------------

def extract_audio_clip_wav(
    mkv_path: Path,
    out_wav: Path,
    start_seconds: float,
    duration_seconds: float,
    verbose: bool = False
) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.3f}",
        "-t", f"{duration_seconds:.3f}",
        "-i", str(mkv_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_wav)
    ]
    rc, _, err = run_cmd(cmd)

    if verbose:
        if rc == 0:
            print(f"[AUDIO] extracted OK ({duration_seconds:.0f}s @ {start_seconds:.0f}s)")
        else:
            print(f"[AUDIO] extract FAILED rc={rc} ({duration_seconds:.0f}s @ {start_seconds:.0f}s)")
            tail = (err or "").splitlines()[-8:]
            if tail:
                print("[AUDIO] ffmpeg stderr (tail):")
                for line in tail:
                    print(f"        {line}")

    return rc == 0 and out_wav.exists() and out_wav.stat().st_size > 0


def transcribe_audio_whisper(client: OpenAI, wav_path: Path) -> Optional[str]:
    with wav_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    text = getattr(resp, "text", None)
    if not text:
        return None
    text = text.strip()
    return text if text else None


# ----------------------------
# Parsing show/season from folder
# ----------------------------

def parse_show_and_season_with_llm(client: OpenAI, folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    json_schema_object = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "show": {"type": ["string", "null"]},
            "season": {"type": ["integer", "null"], "minimum": 1},
        },
        "required": ["show", "season"],
    }

    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": "Return only the structured JSON object that matches the schema.",
            },
            {
                "role": "user",
                "content": (
                    "Extract the show name and season number from this folder name. "
                    "Return only the JSON object.\n\n"
                    f"Folder name: {folder_name}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "show_season_parse",
                "schema": json_schema_object,
                "strict": True,
            }
        },
    )

    raw = resp.output_text
    result = json.loads(raw)
    show = result.get("show")
    season = result.get("season")

    if isinstance(show, str):
        show = show.strip() or None
    else:
        show = None

    if not isinstance(season, int) or season < 1:
        season = None

    return show, season


def parse_show_and_season_from_folder(folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    name = folder_name.strip()
    name = re.sub(r"[_]+", " ", name)
    name = re.sub(r"\s*-\s*", " - ", name)
    name = re.sub(r"\s+", " ", name)

    m = re.search(r"\bSeason\s*(\d{1,2})\b", name, flags=re.IGNORECASE)
    season = int(m.group(1)) if m else None

    show = name
    if m:
        show = name[:m.start()].strip()
    show = show.rstrip("- ").strip()

    if not show:
        show = None

    if show:
        show = show.replace("-", " ")
        show = re.sub(r"\s+", " ", show).strip()

    return show, season


# ----------------------------
# LLM: identify episode
# ----------------------------

def build_prompt(show: str, season: int, evidence_text: str, duration_minutes: float, evidence_kind: str) -> str:
    return f"""You are identifying TV episodes for file renaming.

HINTS:
- Show: {show}
- Season: {season}
- File duration: {duration_minutes:.1f} minutes
- The file may be a NON-EPISODE extra (featurette, recap, deleted scenes). If so, mark is_episode=false.

EVIDENCE TYPE:
- {evidence_kind}

TASK:
Using ONLY the evidence text below, decide:
- is_episode (true/false)
- If is_episode=true: episode_number (1-based integer within the season) and episode_title
- confidence from 0.0 to 1.0

Evidence text:
\"\"\"
{evidence_text}
\"\"\"
"""


def call_llm_identify(
    client: OpenAI,
    model: str,
    show: str,
    season: int,
    evidence_text: str,
    duration_minutes: float,
    evidence_kind: str
) -> dict:
    json_schema_object = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_episode": {"type": "boolean"},
            "show": {"type": "string"},
            "season": {"type": "integer", "minimum": 1},
            "episode_number": {"type": ["integer", "null"], "minimum": 1},
            "episode_title": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string"},
        },
        "required": [
            "is_episode",
            "show",
            "season",
            "episode_number",
            "episode_title",
            "confidence",
            "notes",
        ],
    }

    prompt = build_prompt(show, season, evidence_text, duration_minutes, evidence_kind)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "Return only the structured JSON result matching the schema."},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "episode_identification",
                "schema": json_schema_object,
                "strict": True,
            }
        },
    )

    raw = resp.output_text
    return json.loads(raw)


def format_llm_compact(result: dict, season: int, min_conf: float) -> Tuple[str, bool]:
    is_episode = bool(result.get("is_episode"))
    conf = float(result.get("confidence", 0.0))
    ep_num = result.get("episode_number", None)
    ep_title = result.get("episode_title", None)

    passes = False
    if is_episode and isinstance(ep_num, int) and ep_title and conf >= min_conf:
        passes = True

    if is_episode and isinstance(ep_num, int) and ep_title:
        mark = "✓" if passes else "✗"
        line = f"S{season:02d}E{ep_num:02d} \"{ep_title}\" (conf={conf:.2f}) {mark}"
    else:
        line = f"non-episode/unknown (conf={conf:.2f}) ✗"

    return line, passes


# ----------------------------
# TMDB verification (confirm/correct)
# ----------------------------

@dataclass
class TmdbEpisode:
    episode_number: int
    name: str


class TmdbClient:
    """
    Minimal TMDB v3 client for TV show season episode lists.

    Endpoints:
    - GET https://api.themoviedb.org/3/search/tv
    - GET https://api.themoviedb.org/3/tv/{series_id}/season/{season_number}
    """
    def __init__(self, api_key: str, timeout_s: float = 10.0):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.base = "https://api.themoviedb.org/3"
        self._tv_id_cache: Dict[str, Optional[int]] = {}
        self._season_cache: Dict[Tuple[int, int], List[TmdbEpisode]] = {}

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        params = dict(params or {})
        params["api_key"] = self.api_key
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def find_tv_id(self, show_name: str) -> Optional[int]:
        key = norm_title(show_name)
        if key in self._tv_id_cache:
            return self._tv_id_cache[key]

        data = self._get("/search/tv", {"query": show_name, "include_adult": "false"})
        results = data.get("results") or []
        if not results:
            self._tv_id_cache[key] = None
            return None

        # Pick the first result; optionally you could improve with year matching.
        tv_id = results[0].get("id")
        tv_id = int(tv_id) if tv_id else None
        self._tv_id_cache[key] = tv_id
        return tv_id

    def get_season_episodes(self, tv_id: int, season: int) -> List[TmdbEpisode]:
        ck = (tv_id, season)
        if ck in self._season_cache:
            return self._season_cache[ck]

        data = self._get(f"/tv/{tv_id}/season/{season}", {})
        eps = []
        for e in (data.get("episodes") or []):
            num = e.get("episode_number")
            name = e.get("name")
            if isinstance(num, int) and name:
                eps.append(TmdbEpisode(episode_number=num, name=str(name)))
        self._season_cache[ck] = eps
        return eps


@dataclass
class VerificationResult:
    ok: bool
    corrected: bool
    episode_number: Optional[int]
    episode_title: Optional[str]
    match_score: float
    reason: str


def verify_or_correct_with_tmdb(
    tmdb: TmdbClient,
    show: str,
    season: int,
    proposed_ep_num: int,
    proposed_title: str,
    min_title_match: float,
) -> VerificationResult:
    tv_id = tmdb.find_tv_id(show)
    if not tv_id:
        return VerificationResult(
            ok=False,
            corrected=False,
            episode_number=None,
            episode_title=None,
            match_score=0.0,
            reason="tmdb: show not found"
        )

    episodes = tmdb.get_season_episodes(tv_id, season)
    if not episodes:
        return VerificationResult(
            ok=False,
            corrected=False,
            episode_number=None,
            episode_title=None,
            match_score=0.0,
            reason="tmdb: season not found or has no episodes"
        )

    # 1) If episode number exists, verify title similarity
    ep_by_num = {e.episode_number: e for e in episodes}
    if proposed_ep_num in ep_by_num:
        canon = ep_by_num[proposed_ep_num]
        score = similarity(proposed_title, canon.name)
        if score >= min_title_match:
            # Confirm (and optionally replace title with canonical)
            return VerificationResult(
                ok=True,
                corrected=(norm_title(proposed_title) != norm_title(canon.name)),
                episode_number=canon.episode_number,
                episode_title=canon.name,
                match_score=score,
                reason="tmdb: confirmed by episode number + title match"
            )

    # 2) Otherwise match title across all episodes and pick best
    best = None
    best_score = 0.0
    for e in episodes:
        sc = similarity(proposed_title, e.name)
        if sc > best_score:
            best_score = sc
            best = e

    if best and best_score >= min_title_match:
        corrected = (best.episode_number != proposed_ep_num) or (norm_title(best.name) != norm_title(proposed_title))
        return VerificationResult(
            ok=True,
            corrected=corrected,
            episode_number=best.episode_number,
            episode_title=best.name,
            match_score=best_score,
            reason="tmdb: corrected by title-to-season match"
        )

    # 3) Fail verification
    return VerificationResult(
        ok=False,
        corrected=False,
        episode_number=None,
        episode_title=None,
        match_score=best_score,
        reason=f"tmdb: could not confirm title (best_score={best_score:.2f} < {min_title_match:.2f})"
    )


# ----------------------------
# Rename
# ----------------------------

def rename_in_place(src: Path, show: str, season: int, ep: int, title: str, dry_run: bool) -> bool:
    show_s = sanitize_filename(show)
    title_s = sanitize_filename(title)
    new_name = f"{show_s} - S{season:02d}E{ep:02d} - {title_s}{src.suffix}"
    dst = src.with_name(new_name)

    if dst.exists():
        print(f"[SKIP ] target exists: {src.name} -> {dst.name}")
        return False

    if dry_run:
        print(f"[RENAME] DRYRUN {src.name} -> {dst.name}")
        return True

    src.rename(dst)
    print(f"[RENAME] {src.name} -> {dst.name}")
    return True


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing season/disc folders (or a single folder).")
    ap.add_argument("--model", default="gpt-5.2", help="OpenAI model name (default: gpt-5.2)")
    ap.add_argument("--min-minutes", type=float, default=20.0, help="Skip files shorter than this (default: 20)")
    ap.add_argument("--max-minutes", type=float, default=60.0, help="Skip files longer than this (default: 60)")
    ap.add_argument("--min-confidence", type=float, default=0.85, help="Only consider LLM result when confidence >= this (default: 0.85)")
    ap.add_argument("--max-sub-lines", type=int, default=80, help="Subtitle lines to include (default: 80)")
    ap.add_argument("--dry-run", action="store_true", help="Print planned renames, do not rename.")

    # Audio fallback strategy:
    ap.add_argument("--audio-fallback", action="store_true", default=True,
                    help="Enable audio transcription when subtitles fail (default: on).")
    ap.add_argument("--no-audio-fallback", action="store_true",
                    help="Disable audio transcription fallback.")
    ap.add_argument("--audio-seconds", type=float, default=DEFAULT_FALLBACK_AUDIO_SECONDS,
                    help="Fallback audio clip length in seconds (default: 300). Primary is always 120s.")

    # Logging
    ap.add_argument("--quiet", action="store_true", help="Reduce logging (only renames/skips/errors). Default is verbose.")

    # TMDB verification
    ap.add_argument("--no-verify-api", action="store_true",
                    help="Disable TMDB verification (not recommended if you see wrong episode numbers).")
    ap.add_argument("--tmdb-api-key", default=None,
                    help="TMDB API key (or set TMDB_API_KEY env var).")
    ap.add_argument("--tmdb-min-title-match", type=float, default=0.78,
                    help="Minimum title similarity (0-1) to confirm/correct using TMDB (default: 0.78).")
    args = ap.parse_args()

    verbose = not args.quiet
    audio_fallback_enabled = (not args.no_audio_fallback) and bool(args.audio_fallback)

    def vlog(msg: str) -> None:
        if verbose:
            print(msg)

    if verbose and not os.environ.get("OPENAI_API_KEY"):
        vlog("[WARN ] OPENAI_API_KEY is not set. LLM + transcription calls will fail.\n")

    verify_api_enabled = (not args.no_verify_api)
    tmdb_key = args.tmdb_api_key or os.environ.get("TMDB_API_KEY") or ""
    tmdb_client = None

    if verify_api_enabled:
        if not tmdb_key:
            vlog("[WARN ] TMDB verification is ON but TMDB_API_KEY is not set; verification will be skipped.\n")
            verify_api_enabled = False
        else:
            tmdb_client = TmdbClient(api_key=tmdb_key)

    client = OpenAI()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    vlog(f"Scanning for .mkv under: {root}")
    mkvs = list(root.rglob("*.mkv"))
    if not mkvs:
        print("No .mkv files found.")
        return

    vlog(f"Found {len(mkvs)} .mkv file(s).")
    vlog(f"Mode: {'DRY-RUN' if args.dry_run else 'RENAME'} | Model: {args.model} | LLM conf >= {args.min_confidence}")
    vlog(
        f"Audio fallback: {'ON' if audio_fallback_enabled else 'OFF'} "
        f"(primary {PRIMARY_AUDIO_SECONDS:.0f}s @ {AUDIO_START_SECONDS_HARDCODED:.0f}s, "
        f"fallback {args.audio_seconds:.0f}s @ {AUDIO_START_SECONDS_HARDCODED:.0f}s)"
    )
    vlog(f"TMDB verify: {'ON' if verify_api_enabled else 'OFF'} (min title match {args.tmdb_min_title_match:.2f})\n")

    total = 0
    renamed = 0
    dryrun = 0
    errors = 0

    skipped_already_named = 0
    skipped_parse = 0
    skipped_no_duration = 0
    skipped_duration_range = 0
    skipped_no_evidence = 0
    skipped_non_episode = 0
    skipped_low_conf = 0
    skipped_missing_fields = 0
    skipped_target_exists = 0
    skipped_verify_failed = 0

    used_subtitles = 0
    used_audio_primary = 0
    used_audio_fallback = 0

    for mkv in mkvs:
        total += 1

        hr()
        print(f"FILE : {short_path(mkv, 110)}")

        if re.search(r"\bS\d{2}E\d{2}\b", mkv.stem, flags=re.IGNORECASE):
            skipped_already_named += 1
            print(f"[SKIP ] already looks named (SxxEyy) -> {mkv.name}")
            continue

        folder = mkv.parent.name
        vlog(f"[PARSE] GPT-5-mini parsing folder name: \"{folder}\"")
        show, season = parse_show_and_season_with_llm(client, folder)
        if not show or not season:
            vlog("[PARSE] GPT-5-mini parse missing fields; falling back to regex parsing.")
            show, season = parse_show_and_season_from_folder(folder)
        if not show or not season:
            skipped_parse += 1
            print(f"[SKIP ] can't parse show/season from folder: \"{folder}\"")
            continue

        dur_s = ffprobe_duration_seconds(mkv)
        if dur_s is None:
            skipped_no_duration += 1
            print("[SKIP ] ffprobe couldn't read duration (is ffprobe on PATH? file readable?)")
            continue

        dur_m = dur_s / 60.0
        print(f"HINT : {show} | S{season:02d} | {dur_m:.1f}m")

        if dur_m < args.min_minutes or dur_m > args.max_minutes:
            skipped_duration_range += 1
            print(f"[SKIP ] duration {dur_m:.1f}m out of range [{args.min_minutes:.1f}, {args.max_minutes:.1f}]")
            continue

        streams = ffprobe_subtitle_streams(mkv)
        if verbose:
            print(f"[SUBS ] {summarize_subtitle_streams(streams)}")
            if streams and subtitles_are_bitmap_only(streams):
                print("[SUBS ] bitmap-only (PGS/VobSub) -> audio fallback needed")

        evidence_text = None
        evidence_kind = None

        if not (streams and subtitles_are_bitmap_only(streams)):
            subtitle_excerpt = extract_subtitle_excerpt(mkv, max_lines=args.max_sub_lines)
            if subtitle_excerpt:
                evidence_text = subtitle_excerpt
                evidence_kind = "subtitle excerpt"
                used_subtitles += 1
                if verbose:
                    print(f"[EVID ] subtitles -> \"{one_line(subtitle_excerpt)}\"")

        def attempt_audio_transcribe(seconds: float, stage: str) -> Optional[str]:
            with tempfile.TemporaryDirectory() as td:
                wav_path = Path(td) / "clip.wav"
                ok = extract_audio_clip_wav(
                    mkv,
                    wav_path,
                    AUDIO_START_SECONDS_HARDCODED,
                    seconds,
                    verbose=verbose
                )
                if not ok:
                    if verbose:
                        print(f"[ASR  ] {stage} transcript: FAILED (audio extraction)")
                    return None
                try:
                    tx = transcribe_audio_whisper(client, wav_path)
                except Exception as e:
                    if verbose:
                        print(f"[ERR  ] {stage} transcription failed: {e}")
                    return None
                if not tx:
                    if verbose:
                        print(f"[ASR  ] {stage} transcript: EMPTY")
                    return None
                if verbose:
                    print(f"[ASR  ] {stage} transcript: \"{one_line(tx)}\"")
                return tx

        audio_transcript_primary = None
        audio_transcript_fallback = None

        if evidence_text is None and audio_fallback_enabled:
            audio_transcript_primary = attempt_audio_transcribe(PRIMARY_AUDIO_SECONDS, "PRIMARY")
            if audio_transcript_primary:
                evidence_text = audio_transcript_primary
                evidence_kind = f"audio transcript (Whisper, {PRIMARY_AUDIO_SECONDS:.0f}s @ {AUDIO_START_SECONDS_HARDCODED:.0f}s)"
                used_audio_primary += 1
            else:
                fallback_seconds = float(args.audio_seconds)
                if fallback_seconds < PRIMARY_AUDIO_SECONDS:
                    fallback_seconds = PRIMARY_AUDIO_SECONDS
                if verbose:
                    print(f"[RETRY] PRIMARY transcript unavailable -> trying FALLBACK {fallback_seconds:.0f}s")
                audio_transcript_fallback = attempt_audio_transcribe(fallback_seconds, "FALLBACK")
                if audio_transcript_fallback:
                    evidence_text = audio_transcript_fallback
                    evidence_kind = f"audio transcript (Whisper, {fallback_seconds:.0f}s @ {AUDIO_START_SECONDS_HARDCODED:.0f}s)"
                    used_audio_fallback += 1

        if evidence_text is None:
            skipped_no_evidence += 1
            print("[SKIP ] no usable evidence text (subtitles unavailable and transcription failed/disabled)")
            continue

        if verbose:
            print(f"[EVID ] {evidence_kind}")

        def attempt_llm_with_evidence(stage: str, e_text: str, e_kind: str) -> Tuple[bool, Optional[dict], str]:
            nonlocal errors
            try:
                result = call_llm_identify(client, args.model, show, season, e_text, dur_m, e_kind)
            except Exception as e:
                errors += 1
                print(f"[ERR  ] LLM call failed ({stage}): {e}")
                return False, None, "llm_error"

            compact, passes = format_llm_compact(result, season, args.min_confidence)
            if verbose:
                print(f"[LLM  ] {stage} -> {compact}")
                notes = one_line(result.get("notes", ""), 140)
                if notes and not passes:
                    print(f"       reason: {notes}")

            is_episode = bool(result.get("is_episode"))
            conf = float(result.get("confidence", 0.0))
            ep_num = result.get("episode_number", None)
            ep_title = result.get("episode_title", None)

            if not is_episode:
                return False, result, "non_episode"
            if conf < args.min_confidence:
                return False, result, "low_conf"
            if not isinstance(ep_num, int) or not ep_title:
                return False, result, "missing_fields"

            final_ep_num = ep_num
            final_title = ep_title

            if verify_api_enabled and tmdb_client is not None:
                try:
                    vres = verify_or_correct_with_tmdb(
                        tmdb=tmdb_client,
                        show=show,
                        season=season,
                        proposed_ep_num=ep_num,
                        proposed_title=ep_title,
                        min_title_match=float(args.tmdb_min_title_match),
                    )
                except Exception as e:
                    errors += 1
                    print(f"[ERR  ] TMDB verify failed: {e}")
                    return False, result, "verify_error"

                if verbose:
                    if vres.ok:
                        if vres.corrected:
                            print(f"[API  ] TMDB corrected -> S{season:02d}E{vres.episode_number:02d} \"{vres.episode_title}\" (match={vres.match_score:.2f})")
                        else:
                            print(f"[API  ] TMDB confirmed  -> S{season:02d}E{vres.episode_number:02d} \"{vres.episode_title}\" (match={vres.match_score:.2f})")
                    else:
                        print(f"[API  ] TMDB reject     -> {vres.reason}")

                if not vres.ok or vres.episode_number is None or not vres.episode_title:
                    return False, result, "verify_failed"

                final_ep_num = vres.episode_number
                final_title = vres.episode_title

            ok = rename_in_place(mkv, show, season, final_ep_num, final_title, args.dry_run)
            if ok:
                return True, result, "renamed"
            return False, result, "target_exists"

        renamed_ok, first_result, fail_code = attempt_llm_with_evidence("PRIMARY", evidence_text, evidence_kind)

        if renamed_ok:
            print("[DONE ] renamed")
            if args.dry_run:
                dryrun += 1
            else:
                renamed += 1
            continue

        used_primary_audio = (audio_transcript_primary is not None) and (evidence_text == audio_transcript_primary)
        if used_primary_audio and audio_fallback_enabled:
            fallback_seconds = float(args.audio_seconds)
            if fallback_seconds < PRIMARY_AUDIO_SECONDS:
                fallback_seconds = PRIMARY_AUDIO_SECONDS

            if verbose:
                print(f"[RETRY] primary evidence didn't pass -> trying FALLBACK transcript ({fallback_seconds:.0f}s)")

            if audio_transcript_fallback is None:
                audio_transcript_fallback = attempt_audio_transcribe(fallback_seconds, "FALLBACK")
                if audio_transcript_fallback:
                    used_audio_fallback += 1

            if audio_transcript_fallback:
                fb_kind = f"audio transcript (Whisper, {fallback_seconds:.0f}s @ {AUDIO_START_SECONDS_HARDCODED:.0f}s)"
                renamed_ok2, second_result, fail_code2 = attempt_llm_with_evidence("FALLBACK", audio_transcript_fallback, fb_kind)
                if renamed_ok2:
                    print("[DONE ] renamed (after fallback)")
                    if args.dry_run:
                        dryrun += 1
                    else:
                        renamed += 1
                    continue

                if second_result is not None:
                    first_result = second_result
                    fail_code = fail_code2

        if fail_code in ("llm_error", "verify_error"):
            continue

        if fail_code == "non_episode":
            skipped_non_episode += 1
            print("[SKIP ] LLM says non-episode/unknown")
            continue

        if fail_code == "low_conf":
            skipped_low_conf += 1
            conf = float((first_result or {}).get("confidence", 0.0))
            print(f"[SKIP ] confidence {conf:.2f} < {args.min_confidence:.2f}")
            continue

        if fail_code == "missing_fields":
            skipped_missing_fields += 1
            print("[SKIP ] missing episode_number or episode_title")
            continue

        if fail_code == "verify_failed":
            skipped_verify_failed += 1
            print("[SKIP ] TMDB could not confirm episode number/title (preventing bad rename)")
            continue

        skipped_target_exists += 1
        print("[SKIP ] target exists / rename not performed")

    hr()
    print("\n=== SUMMARY ===")
    print(f"Total files scanned: {total}")
    print(f"Renamed:            {renamed}")
    print(f"Dry-run renames:    {dryrun}")
    print(f"Errors:             {errors}")
    print(f"Evidence used: subtitles={used_subtitles} audio_primary={used_audio_primary} audio_fallback={used_audio_fallback}")
    print(f"Skipped (TMDB verify failed):       {skipped_verify_failed}")
    print("Skipped:")
    print(f"  already named (SxxEyy):           {skipped_already_named}")
    print(f"  can't parse show/season:          {skipped_parse}")
    print(f"  no duration (ffprobe):            {skipped_no_duration}")
    print(f"  duration out of range:            {skipped_duration_range}")
    print(f"  no usable evidence text:          {skipped_no_evidence}")
    print(f"  LLM says non-episode:             {skipped_non_episode}")
    print(f"  confidence below threshold:       {skipped_low_conf}")
    print(f"  missing episode number/title:     {skipped_missing_fields}")
    print(f"  target exists/other:              {skipped_target_exists}")


if __name__ == "__main__":
    main()
