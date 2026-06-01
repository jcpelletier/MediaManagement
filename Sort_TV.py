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
- pydantic + requests python packages
- faster-whisper python package (pip install faster-whisper) — only for audio transcription fallback
- DEEPSEEK_API_KEY env var set
- TMDB_API_KEY env var set (or pass --tmdb-api-key)

TMDB endpoints used:
- /3/search/tv
- /3/tv/{series_id}/season/{season_number}
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import struct
import tempfile
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import requests
try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FasterWhisperModel = None
    _FASTER_WHISPER_AVAILABLE = False
from pydantic import BaseModel

from llm_deepseek import DeepSeekClient, DeepSeekError, DeepSeekAuthError


# ----------------------------
# Constants (fallback strategy)
# ----------------------------

PRIMARY_AUDIO_SECONDS = 240.0

# Shift audio sampling to 2 minutes in (120 seconds).
AUDIO_START_SECONDS_HARDCODED = 300.0

DEFAULT_FALLBACK_AUDIO_SECONDS = 600.0

# Random jitter (seconds) added to every audio sampling start offset. Shows
# in a season often share consistent intro/cold-open timings, so a fixed start
# tends to land on the same intro across episodes; jitter de-correlates samples
# so we capture episode-distinguishing dialog instead.
AUDIO_START_JITTER_MIN = 2.0
AUDIO_START_JITTER_MAX = 20.0

INVALID_FILENAME_CHARS = r'<>:"/\\|?*'

OPENSUBTITLES_API_URL = "https://api.opensubtitles.com/api/v1/subtitles"
OPENSUBTITLES_USER_AGENT = "MediaManagement/1.0"


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
# OpenSubtitles (knowledge boost)
# ----------------------------

def compute_opensubtitles_hash(file_path: Path) -> Optional[str]:
    try:
        size = file_path.stat().st_size
    except OSError:
        return None

    # OpenSubtitles hash needs at least 128 KiB (64 KiB head + 64 KiB tail).
    if size < 131072:
        return None

    try:
        with file_path.open("rb") as f:
            head = f.read(65536)
            f.seek(max(0, size - 65536))
            tail = f.read(65536)
    except OSError:
        return None

    total = size
    for buf in (head, tail):
        for i in range(0, len(buf), 8):
            (val,) = struct.unpack_from("<Q", buf, i)
            total = (total + val) & 0xFFFFFFFFFFFFFFFF

    return f"{total:016x}"


def opensubtitles_exact_match(
    api_key: str,
    user_agent: str,
    mkv_path: Path,
    show: str,
    season: int,
    timeout_s: float = 10.0,
    verbose: bool = False,
) -> Optional[Tuple[int, str]]:
    moviehash = compute_opensubtitles_hash(mkv_path)
    if not moviehash:
        return None
    if verbose:
        print(f"[OSDB ] hash {moviehash}")

    headers = {
        "Api-Key": api_key,
        "User-Agent": user_agent,
    }
    params = {
        "moviehash": moviehash,
        "moviehash_match": "only",
        "order_by": "download_count",
        "order_direction": "desc",
    }

    r = requests.get(OPENSUBTITLES_API_URL, headers=headers, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json().get("data") or []
    if verbose:
        print(f"[OSDB ] results returned: {len(data)}")

    show_norm = norm_title(show)
    for item in data:
        attrs = item.get("attributes") or {}
        details = attrs.get("feature_details") or {}
        parent_title = details.get("parent_title") or details.get("parent_name") or ""
        season_num = details.get("season_number")
        ep_num = details.get("episode_number")
        ep_title = details.get("title") or details.get("feature_title") or ""
        if verbose:
            print(
                "[OSDB ] candidate "
                f"parent_title=\"{parent_title}\" season={season_num} episode={ep_num}"
            )

        if not isinstance(season_num, int) or not isinstance(ep_num, int):
            continue
        if season_num != season:
            continue
        if not parent_title or not ep_title:
            continue
        if norm_title(parent_title) != show_norm:
            continue

        return ep_num, str(ep_title)

    return None


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


_SRT_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _srt_ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def extract_subtitle_excerpt(
    mkv_path: Path,
    max_lines: int = 80,
    max_chars: int = 4000,
    skip_head_seconds: float = 300.0,
) -> Optional[str]:
    """Sample dialogue from three points in the episode, skipping the first
    `skip_head_seconds` of timeline. Recurring intros / opening narration live
    in that window and tend to dominate Phase-1 identification — sampling past
    it forces the LLM to disambiguate on episode-specific dialogue. Files
    whose total runtime falls inside the skip window get sampled from the
    start as a fallback (they're almost certainly extras anyway)."""
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

        # Parse SRT into (start_seconds, dialogue_text) cues so we can filter
        # by timeline position rather than line index.
        cues: List[Tuple[float, str]] = []
        current_start: Optional[float] = None
        current_buf: List[str] = []

        def _flush():
            if current_start is not None and current_buf:
                text = " ".join(current_buf).strip()
                if text:
                    cues.append((current_start, text))

        for raw_line in txt.splitlines():
            line = raw_line.strip()
            if not line:
                _flush()
                current_start = None
                current_buf = []
                continue
            if re.fullmatch(r"\d+", line):
                continue
            ts = _SRT_TS_RE.match(line)
            if ts:
                _flush()
                current_start = _srt_ts_to_seconds(ts.group(1), ts.group(2), ts.group(3), ts.group(4))
                current_buf = []
                continue
            line = re.sub(r"</?i>", "", line)
            line = re.sub(r"</?b>", "", line)
            line = re.sub(r"</?u>", "", line)
            line = re.sub(r"{\\.*?}", "", line)
            line = re.sub(r"^\[.*?\]\s*", "", line)
            line = re.sub(r"^\(.*?\)\s*", "", line)
            if line:
                current_buf.append(line)
        _flush()

        if not cues:
            return None

        filtered = [(t, dlg) for (t, dlg) in cues if t >= skip_head_seconds]
        if not filtered:
            # Whole file fits inside the skip window — fall back to all cues.
            filtered = cues
            head_skipped = False
        else:
            head_skipped = True

        end_time = filtered[-1][0] if filtered else 0.0
        start_time = filtered[0][0] if filtered else 0.0
        span = max(1.0, end_time - start_time)

        per_section = max(1, max_lines // 3)
        # Sample at 0% / 33% / 66% of the post-skip window so the three picks
        # are well separated even on short files.
        sections = []
        for label, frac in [("EARLY", 0.0), ("MID", 0.34), ("LATE", 0.67)]:
            target_t = start_time + frac * span
            # Find the first cue at or after target_t.
            idx = next((i for i, (t, _) in enumerate(filtered) if t >= target_t), 0)
            chunk = [dlg for _, dlg in filtered[idx : idx + per_section]]
            if chunk:
                tag = f"[{label}]" if head_skipped else f"[{label}*]"
                sections.append(f"{tag}\n" + "\n".join(chunk))

        excerpt = "\n\n".join(sections)[:max_chars].strip()
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


def transcribe_with_faster_whisper(model: "_FasterWhisperModel", wav_path: Path) -> Optional[str]:
    segments, _ = model.transcribe(str(wav_path), beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    return text if text else None


# ----------------------------
# Parsing show/season from folder
# ----------------------------

_DISC_MARKER_RE = re.compile(
    r"(?:^|(?<=[\s._-]))(?:disc[\s._-]*\d+|d\d+)(?=$|[\s._-])",
    re.IGNORECASE,
)


def _strip_disc_markers(folder_name: str) -> str:
    """Remove 'Disc N' / 'D<N>' tokens. They identify the physical disc, not the
    season — without stripping, the LLM mistakes 'ARRESTED_D2' for season 2."""
    cleaned = _DISC_MARKER_RE.sub(" ", folder_name)
    return re.sub(r"[\s._-]+", " ", cleaned).strip(" ._-")


def parse_show_and_season_with_llm(client: DeepSeekClient, folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    class _ShowSeason(BaseModel):
        show: Optional[str] = None
        season: Optional[int] = None

    cleaned = _strip_disc_markers(folder_name)

    try:
        result = client.parse(
            system="Return only the structured JSON object that matches the schema.",
            user=(
                "Extract the show name and season number from this folder name. "
                "Disc markers ('Disc 2', 'D2', 'D3', etc.) identify the physical disc "
                "and are NOT season numbers — ignore them. Return season=null if the "
                "folder name does not state a season. Return only the JSON object.\n\n"
                f"Folder name: {cleaned}"
            ),
            schema=_ShowSeason,
            max_tokens=256,
        )
    except Exception:
        return None, None

    show = result.show
    season = result.season

    if isinstance(show, str):
        show = show.strip() or None
    else:
        show = None

    if not isinstance(season, int) or season < 1:
        season = None

    return show, season


def parse_show_and_season_from_folder(folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    name = _strip_disc_markers(folder_name)
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


def find_sibling_disc_hints(folder_path: Path) -> Optional[dict]:
    """Look for a sibling disc folder (e.g. ARRESTED_D2 -> ARRESTED_D1) with a
    sort_hints.json and inherit its show/season. Only fires when the current
    folder has a disc marker, so unrelated siblings are not consulted. If
    multiple siblings have hints, all must agree on show + season."""
    if not _DISC_MARKER_RE.search(folder_path.name):
        return None
    parent = folder_path.parent
    if not parent.exists():
        return None
    base = _strip_disc_markers(folder_path.name).lower()
    if not base:
        return None

    candidates: List[Tuple[str, int, str]] = []
    try:
        siblings = list(parent.iterdir())
    except OSError:
        return None
    for sib in siblings:
        if sib == folder_path or not sib.is_dir():
            continue
        if _strip_disc_markers(sib.name).lower() != base:
            continue
        hp = sib / "sort_hints.json"
        if not hp.exists():
            continue
        try:
            data = json.loads(hp.read_text())
        except Exception:
            continue
        show = str(data.get("show", "")).strip()
        season = data.get("season")
        if show and isinstance(season, int) and season >= 1:
            candidates.append((show, season, sib.name))

    if not candidates:
        return None
    shows = {norm_title(s) for s, _, _ in candidates}
    seasons = {se for _, se, _ in candidates}
    if len(shows) != 1 or len(seasons) != 1:
        return None
    show, season, src = candidates[0]
    return {"show": show, "season": season, "_inherited_from": src}


def write_auto_sort_hints(folder_path: Path, show: str, season: int, source_file: str) -> bool:
    """Persist a hint after a confident, TMDB-confirmed identification so the
    rest of this folder (and future runs) skip blind mode. Refuses to overwrite
    an existing hints file."""
    hints_path = folder_path / "sort_hints.json"
    if hints_path.exists():
        return False
    payload = {
        "show": show,
        "season": season,
        "_auto": True,
        "_source_file": source_file,
    }
    try:
        hints_path.write_text(json.dumps(payload, indent=2))
        return True
    except Exception:
        return False


# ----------------------------
# LLM: identify episode
# ----------------------------

def build_prompt(show: str, season: int, evidence_text: str, duration_minutes: float,
                 evidence_kind: str, episode_guide: Optional[str] = None,
                 disc_context: Optional[List[str]] = None,
                 is_collection: bool = False,
                 cross_disc_last_ep: Optional[int] = None) -> str:
    guide_section = ""
    if episode_guide:
        guide_section = (
            f"\n## Season {season} Episode Guide (from TMDB)\n"
            f"Match the evidence against this list. Episodes flagged '<-- duration match' have a "
            f"runtime within 12 minutes of this file's duration ({duration_minutes:.1f} min) "
            f"and are the most likely candidates.\n"
            f"{episode_guide}\n"
        )

    context_section = ""
    if disc_context:
        # Extract episode numbers already claimed on this disc to infer likely next episode.
        claimed_nums = []
        for entry in disc_context:
            m = re.search(r"E(\d+)", entry, flags=re.IGNORECASE)
            if m:
                claimed_nums.append(int(m.group(1)))

        ordering_hint = ""
        if claimed_nums:
            next_expected = max(claimed_nums) + 1
            ordering_hint = (
                f"\nDisc ordering hint: the highest episode identified so far on this disc is "
                f"E{max(claimed_nums):02d}. Discs are usually sequential, so this file is most "
                f"likely E{next_expected:02d} or nearby. Only deviate from this if the evidence "
                f"strongly and clearly points to a different episode.\n"
            )

        context_section = (
            "\n## Already identified from this disc (earlier files in this folder)\n"
            + "\n".join(f"  - {ep}" for ep in disc_context)
            + "\nThese episodes are already taken — do not assign the same episode number to this file."
            + ordering_hint
        )

    if is_collection:
        episode_hint = (
            "- This is a compilation disc of standalone programs/specials. Each program is a "
            "separate item — identify it and set is_episode=true if you can name it. "
            "Only mark is_episode=false if the content is completely unidentifiable."
        )
    else:
        episode_hint = (
            "- The file may be a NON-EPISODE extra (featurette, recap, deleted scenes). "
            "If so, mark is_episode=false."
        )

    cross_disc_section = ""
    if cross_disc_last_ep is not None and not disc_context:
        next_ep = cross_disc_last_ep + 1
        cross_disc_section = (
            f"\n## Cross-disc context\n"
            f"Previous disc(s) for {show} Season {season:02d} have been identified up to "
            f"E{cross_disc_last_ep:02d}. This disc most likely starts around E{next_ep:02d}. "
            f"Only assign an episode number far from E{next_ep:02d} if the evidence clearly "
            f"and unambiguously points elsewhere.\n"
        )

    return f"""You are identifying TV episodes for file renaming.

HINTS:
- Show: {show}
- Season: {season}
- File duration: {duration_minutes:.1f} minutes
{episode_hint}
{cross_disc_section}{guide_section}{context_section}
EVIDENCE TYPE:
- {evidence_kind}

TASK:
Using the evidence text and the episode guide above (if provided), decide:
- is_episode (true/false)
- If is_episode=true: episode_number (1-based integer within the season) and episode_title
- confidence from 0.0 to 1.0
Evidence may include multiple separate clips from the same file; they are not necessarily contiguous and should be treated as separate excerpts.

Evidence text:
\"\"\"
{evidence_text}
\"\"\"
"""


def build_blind_prompt(evidence_text: str, duration_minutes: float, evidence_kind: str,
                       disc_context: Optional[List[str]] = None) -> str:
    """Prompt used when no show/season hint is available from the folder name.
    Claude must identify the show, season, and episode entirely from content."""

    context_section = ""
    if disc_context:
        claimed_nums = []
        for entry in disc_context:
            m = re.search(r"E(\d+)", entry, flags=re.IGNORECASE)
            if m:
                claimed_nums.append(int(m.group(1)))
        ordering_hint = ""
        if claimed_nums:
            next_expected = max(claimed_nums) + 1
            ordering_hint = (
                f"\nDisc ordering hint: the highest episode identified so far is E{max(claimed_nums):02d}. "
                f"This file is most likely E{next_expected:02d} or nearby.\n"
            )
        context_section = (
            "\n## Already identified from this disc\n"
            + "\n".join(f"  - {ep}" for ep in disc_context)
            + "\nThese are already taken — assign a different episode number to this file."
            + ordering_hint
        )

    return f"""You are identifying video content for file renaming.

No folder name is available. Identify the content entirely from the evidence below.

FILE INFO:
- Duration: {duration_minutes:.1f} minutes

WHAT COUNTS AS is_episode=true:
- TV series episodes
- Standalone TV specials, holiday specials, or animated adaptations (e.g. "Dr. Seuss: Horton Hatches the Egg")
- Short films or TV movies with a clear identifiable title
- Any named video content you can identify

WHAT COUNTS AS is_episode=false (only these):
- Production featurettes, behind-the-scenes, making-of clips
- Trailers or promotional material
- Disc menus or navigation content
- Content you genuinely cannot identify at all

EVIDENCE TYPE:
- {evidence_kind}
{context_section}
TASK:
Using ONLY the evidence text below, decide:
- is_episode (true/false — use the definitions above)
- If is_episode=true: show name (or collection name), season number (integer >= 1), episode_number (1-based), and episode_title
- confidence from 0.0 to 1.0

For standalone specials/films: use a consistent collection name as the show (e.g. "Dr. Seuss Specials"), season 1, and number episodes sequentially. Be conservative with confidence — only high confidence when evidence clearly and unambiguously identifies the content.
Evidence may include multiple separate clips from the same file; they are not necessarily contiguous and should be treated as separate excerpts.

Evidence text:
\"\"\"
{evidence_text}
\"\"\"
"""


def call_llm_identify(
    client: DeepSeekClient,
    model: str,
    show: str,
    season: int,
    evidence_text: str,
    duration_minutes: float,
    evidence_kind: str,
    episode_guide: Optional[str] = None,
    disc_context: Optional[List[str]] = None,
    is_collection: bool = False,
    cross_disc_last_ep: Optional[int] = None,
) -> dict:
    class _EpisodeID(BaseModel):
        is_episode: bool
        show: str
        season: int
        episode_number: Optional[int] = None
        episode_title: Optional[str] = None
        confidence: float
        notes: str = ""

    prompt = build_prompt(show, season, evidence_text, duration_minutes, evidence_kind,
                          episode_guide=episode_guide, disc_context=disc_context,
                          is_collection=is_collection, cross_disc_last_ep=cross_disc_last_ep)

    result = client.parse(
        system="Return only the structured JSON result matching the schema.",
        user=prompt,
        schema=_EpisodeID,
        model=model,
        max_tokens=1024,
    )

    return {
        "is_episode": result.is_episode,
        "show": result.show,
        "season": result.season,
        "episode_number": result.episode_number,
        "episode_title": result.episode_title,
        "confidence": result.confidence,
        "notes": result.notes,
    }


def call_llm_identify_blind(
    client: DeepSeekClient,
    model: str,
    evidence_text: str,
    duration_minutes: float,
    evidence_kind: str,
    disc_context: Optional[List[str]] = None,
) -> dict:
    """Blind identification — no show/season hints. Claude infers everything from content."""
    class _EpisodeIDBlind(BaseModel):
        is_episode: bool
        show: Optional[str] = None
        season: Optional[int] = None
        episode_number: Optional[int] = None
        episode_title: Optional[str] = None
        confidence: float
        notes: str = ""

    prompt = build_blind_prompt(evidence_text, duration_minutes, evidence_kind,
                               disc_context=disc_context)

    result = client.parse(
        system="Return only the structured JSON result matching the schema.",
        user=prompt,
        schema=_EpisodeIDBlind,
        model=model,
        max_tokens=1024,
    )

    return {
        "is_episode": result.is_episode,
        "show": result.show,
        "season": result.season,
        "episode_number": result.episode_number,
        "episode_title": result.episode_title,
        "confidence": result.confidence,
        "notes": result.notes,
    }


def call_llm_synopsis_pick(
    client: DeepSeekClient,
    model: str,
    evidence_text: str,
    evidence_kind: str,
    candidates: List[Tuple[int, str, str]],
) -> Optional[int]:
    """Given an evidence excerpt and a small set of candidate episodes with their
    TMDB plot synopses, ask the LLM which candidate's synopsis best matches the
    excerpt. Used as the sanity check after the reconciler's contiguous-run
    shortcut: if the LLM systematically picks the episode-before / episode-after
    the Phase 1 proposal, the whole disc was off by one and the shortcut needs
    correcting. Returns the picked episode_number, or None if no candidate
    plausibly matches."""
    class _SynopsisPick(BaseModel):
        episode_number: Optional[int] = None
        reason: str = ""

    if not candidates:
        return None

    candidate_block = "\n\n".join(
        f"Episode {ep}: \"{title}\"\nSynopsis: {overview or '(no synopsis available)'}"
        for ep, title, overview in candidates
    )

    prompt = (
        "You are given an excerpt of dialogue or audio transcript from one TV "
        "episode, and several candidate episodes with their plot synopses. Pick "
        "the candidate whose synopsis is the best match for what is happening in "
        "the excerpt. Match on specific plot beats, named characters, settings, "
        "and events — not on the show's recurring style or general tone. If no "
        "candidate's synopsis plausibly matches, return episode_number=null.\n\n"
        f"EVIDENCE TYPE: {evidence_kind}\n\n"
        f"EVIDENCE:\n\"\"\"\n{evidence_text}\n\"\"\"\n\n"
        f"CANDIDATES:\n{candidate_block}\n\n"
        "Return only the JSON object."
    )

    try:
        result = client.parse(
            system="Return only the structured JSON result matching the schema.",
            user=prompt,
            schema=_SynopsisPick,
            model=model,
            max_tokens=512,
        )
    except Exception:
        return None

    picked = result.episode_number
    if not isinstance(picked, int):
        return None
    valid_eps = {ep for ep, _, _ in candidates}
    if picked not in valid_eps:
        return None
    return picked


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
    runtime: Optional[int] = None   # minutes, may be None if TMDB lacks data
    overview: str = ""              # plot synopsis


@dataclass
class TmdbShow:
    tv_id: int
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
        self._tv_cache: Dict[str, Optional[TmdbShow]] = {}
        self._season_cache: Dict[Tuple[int, int], List[TmdbEpisode]] = {}

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        params = dict(params or {})
        params["api_key"] = self.api_key
        # Retry transient network failures / 5xx with exponential backoff. A
        # single api.themoviedb.org blip used to drop the affected file
        # entirely; now we try four times before giving up.
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
                continue
            if r.status_code == 404:
                return {}
            if r.status_code >= 500 or r.status_code == 429:
                last_exc = requests.HTTPError(f"TMDB returned {r.status_code}")
                # 429 honours Retry-After when present, else exponential backoff.
                retry_after = 0.0
                if r.status_code == 429:
                    try:
                        retry_after = float(r.headers.get("Retry-After", "0"))
                    except ValueError:
                        retry_after = 0.0
                time.sleep(max(retry_after, min(8.0, 0.5 * (2 ** attempt))))
                continue
            r.raise_for_status()
            return r.json()
        if last_exc is not None:
            raise last_exc
        return {}

    def find_tv(self, show_name: str) -> Optional[TmdbShow]:
        key = norm_title(show_name)
        if key in self._tv_cache:
            return self._tv_cache[key]

        data = self._get("/search/tv", {"query": show_name, "include_adult": "false"})
        results = data.get("results") or []
        if not results:
            self._tv_cache[key] = None
            return None

        # Pick the first result; optionally you could improve with year matching.
        top = results[0]
        tv_id = top.get("id")
        name = top.get("name") or top.get("original_name") or show_name
        if not tv_id:
            self._tv_cache[key] = None
            return None
        show = TmdbShow(tv_id=int(tv_id), name=str(name))
        self._tv_cache[key] = show
        return show

    def find_tv_id(self, show_name: str) -> Optional[int]:
        show = self.find_tv(show_name)
        return show.tv_id if show else None

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
                eps.append(TmdbEpisode(
                    episode_number=num,
                    name=str(name),
                    runtime=e.get("runtime") if isinstance(e.get("runtime"), int) else None,
                    overview=str(e.get("overview") or ""),
                ))
        self._season_cache[ck] = eps
        return eps


def format_episode_guide(episodes: List[TmdbEpisode], season: int,
                          file_duration_minutes: float,
                          duration_tolerance_minutes: float = 12.0) -> str:
    """
    Format a season's episode list as a guide for the LLM.
    Episodes whose runtime is within tolerance of the file duration are flagged.
    """
    lines = []
    for ep in episodes:
        parts = [f"S{season:02d}E{ep.episode_number:02d} \"{ep.name}\""]
        if ep.runtime:
            parts.append(f"({ep.runtime}min)")
            if abs(ep.runtime - file_duration_minutes) <= duration_tolerance_minutes:
                parts.append("<-- duration match")
        else:
            parts.append("(?min)")
        if ep.overview:
            parts.append(f"— {ep.overview[:150]}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


@dataclass
class VerificationResult:
    ok: bool
    corrected: bool
    episode_number: Optional[int]
    episode_title: Optional[str]
    match_score: float
    reason: str


def _runtime_disagrees(
    canon_runtime: Optional[int],
    file_minutes: Optional[float],
    max_relative: float = 0.30,
    max_absolute: float = 15.0,
) -> bool:
    """Tolerance uses whichever is LARGER (relative or absolute) so commercial cuts,
    pilots, and finales still match. Returns False (no signal) when either side
    is missing — runtime can only reject, never falsely confirm."""
    if canon_runtime is None or canon_runtime <= 0 or file_minutes is None:
        return False
    delta = abs(canon_runtime - file_minutes)
    tolerance = max(max_absolute, canon_runtime * max_relative)
    return delta > tolerance


def verify_or_correct_with_tmdb(
    tmdb: TmdbClient,
    show: str,
    season: int,
    proposed_ep_num: int,
    proposed_title: str,
    min_title_match: float,
    file_duration_minutes: Optional[float] = None,
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

    # 1) If episode number exists, verify title similarity AND runtime.
    ep_by_num = {e.episode_number: e for e in episodes}
    if proposed_ep_num in ep_by_num:
        canon = ep_by_num[proposed_ep_num]
        score = similarity(proposed_title, canon.name)
        if score >= min_title_match and not _runtime_disagrees(canon.runtime, file_duration_minutes):
            return VerificationResult(
                ok=True,
                corrected=(norm_title(proposed_title) != norm_title(canon.name)),
                episode_number=canon.episode_number,
                episode_title=canon.name,
                match_score=score,
                reason="tmdb: confirmed by episode number + title match"
            )
        # Title matched but runtime way off — fall through to look for a better
        # episode in this season whose runtime fits.

    # 2) Score every episode by title similarity; use runtime as tiebreaker
    #    when multiple episodes are close on title.
    scored = sorted(
        ((e, similarity(proposed_title, e.name)) for e in episodes),
        key=lambda x: x[1],
        reverse=True,
    )

    best_ep, best_score = (scored[0] if scored else (None, 0.0))

    if best_ep and best_score >= min_title_match:
        # Tiebreaker: among candidates within 0.05 of the top score and above
        # threshold, prefer the one whose runtime is closest to the file.
        if file_duration_minutes is not None:
            tied = [(e, s) for e, s in scored
                    if s >= min_title_match and s >= best_score - 0.05]
            if len(tied) > 1:
                def _runtime_dist(ep):
                    return abs(ep.runtime - file_duration_minutes) if ep.runtime else float("inf")
                tied.sort(key=lambda x: (_runtime_dist(x[0]), -x[1]))
                best_ep, best_score = tied[0]

        if _runtime_disagrees(best_ep.runtime, file_duration_minutes):
            return VerificationResult(
                ok=False,
                corrected=False,
                episode_number=None,
                episode_title=None,
                match_score=best_score,
                reason=(f"tmdb: best title match S{season:02d}E{best_ep.episode_number:02d} "
                        f"runtime {best_ep.runtime}min disagrees with file {file_duration_minutes:.1f}min"),
            )

        corrected = (best_ep.episode_number != proposed_ep_num) or (norm_title(best_ep.name) != norm_title(proposed_title))
        return VerificationResult(
            ok=True,
            corrected=corrected,
            episode_number=best_ep.episode_number,
            episode_title=best_ep.name,
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

def rename_and_move(
    src: Path,
    show: str,
    season: int,
    ep: int,
    title: str,
    dry_run: bool,
    dest_root: Optional[Path] = None,
) -> bool:
    show_s = sanitize_filename(show)
    title_s = sanitize_filename(title)
    new_name = f"{show_s} - S{season:02d}E{ep:02d} - {title_s}{src.suffix}"

    if dest_root is not None:
        season_dir = dest_root / show_s / f"Season {season:02d}"
        dst = season_dir / new_name
    else:
        dst = src.with_name(new_name)

    if dst.exists():
        print(f"[SKIP ] target exists: {dst}")
        return False

    if dry_run:
        if dest_root is not None:
            print(f"[RENAME] DRYRUN {src.name} -> {dst}")
        else:
            print(f"[RENAME] DRYRUN {src.name} -> {new_name}")
        return True

    if dest_root is not None:
        season_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"[MOVE  ] {src.name} -> {dst}")
    else:
        src.rename(dst)
        print(f"[RENAME] {src.name} -> {new_name}")
    return True


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing season/disc folders (or a single folder).")
    ap.add_argument("--dest", default=None,
                    help="Library root to move identified episodes into (e.g. /mnt/media/Media/Shows). "
                         "Creates <Show>/Season NN/ subdirs as needed. Omit to rename files in place.")
    ap.add_argument("--model", default="deepseek-chat", help="DeepSeek model for episode identification when folder/season is known (default: deepseek-chat)")
    ap.add_argument("--blind-model", default="deepseek-chat", help="DeepSeek model for blind identification (no folder hint; default: deepseek-chat)")
    ap.add_argument("--min-minutes", type=float, default=6.0, help="Skip files shorter than this (default: 6). Low enough to keep short-form kids' episodes, e.g. Bluey (~7 min, often just under).")
    ap.add_argument("--max-minutes", type=float, default=100.0, help="Skip files longer than this (default: 100)")
    ap.add_argument("--min-confidence", type=float, default=0.85, help="Only consider LLM result when confidence >= this (default: 0.85)")
    ap.add_argument("--max-sub-lines", type=int, default=80, help="Subtitle lines to include (default: 80)")
    ap.add_argument("--dry-run", action="store_true", help="Print planned renames, do not rename.")

    # Audio fallback strategy:
    ap.add_argument("--audio-fallback", action="store_true", default=True,
                    help="Enable audio transcription when subtitles fail (default: on).")
    ap.add_argument("--no-audio-fallback", action="store_true",
                    help="Disable audio transcription fallback.")
    ap.add_argument("--audio-start-seconds", type=float, default=AUDIO_START_SECONDS_HARDCODED,
                    help="Primary audio start offset in seconds (default: 120).")
    ap.add_argument("--audio-seconds", type=float, default=DEFAULT_FALLBACK_AUDIO_SECONDS,
                    help="Fallback audio clip length in seconds (default: 300). Primary is always 120s.")
    ap.add_argument("--whisper-model", default="small",
                    help="faster-whisper model size (default: small). Options: tiny, base, small, medium, large-v3.")
    ap.add_argument("--whisper-device", default="cpu",
                    help="Device for faster-whisper (default: cpu). Use 'cuda' if GPU is available.")

    # OpenSubtitles knowledge boost
    ap.add_argument("--opensubtitles-exact-rename", action="store_true",
                    help="Rename immediately when OpenSubtitles hash lookup returns an exact show/season match.")
    ap.add_argument("--opensubtitles-user-agent", default=OPENSUBTITLES_USER_AGENT,
                    help="OpenSubtitles User-Agent header (default: MediaManagement/1.0).")

    # Logging
    ap.add_argument("--quiet", action="store_true", help="Reduce logging (only renames/skips/errors). Default is verbose.")

    # TMDB verification
    ap.add_argument("--no-verify-api", action="store_true",
                    help="Disable TMDB verification (not recommended if you see wrong episode numbers).")
    ap.add_argument("--tmdb-api-key", default=None,
                    help="TMDB API key (or set TMDB_API_KEY env var).")
    ap.add_argument("--tmdb-min-title-match", type=float, default=0.78,
                    help="Minimum title similarity (0-1) to confirm/correct using TMDB (default: 0.78).")
    ap.add_argument("--summary-json", default=None,
                    help="Write a JSON run summary to this path for post-build notifications.")
    args = ap.parse_args()

    dest_root: Optional[Path] = Path(args.dest).expanduser().resolve() if args.dest else None

    verbose = not args.quiet
    audio_fallback_enabled = (not args.no_audio_fallback) and bool(args.audio_fallback)

    def vlog(msg: str) -> None:
        if verbose:
            print(msg)

    if verbose and not os.environ.get("DEEPSEEK_API_KEY"):
        vlog("[WARN ] DEEPSEEK_API_KEY is not set. LLM calls will fail.\n")

    verify_api_enabled = (not args.no_verify_api)
    tmdb_key = args.tmdb_api_key or os.environ.get("TMDB_API_KEY") or ""
    tmdb_client = None

    if verify_api_enabled:
        if not tmdb_key:
            vlog("[WARN ] TMDB verification is ON but TMDB_API_KEY is not set; verification will be skipped.\n")
            verify_api_enabled = False
        else:
            tmdb_client = TmdbClient(api_key=tmdb_key)

    llm_client = DeepSeekClient()

    # Local faster-whisper model for audio transcription fallback
    whisper_model = None
    if audio_fallback_enabled:
        if not _FASTER_WHISPER_AVAILABLE:
            vlog("[WARN ] faster-whisper not installed; audio transcription fallback disabled.\n")
            vlog("        Run: pip install faster-whisper\n")
            audio_fallback_enabled = False
        else:
            # Try CUDA compute types first; if all fail (e.g. GPU unavailable in container)
            # fall back to CPU int8 which always works.
            if args.whisper_device == "cuda":
                candidates = [("cuda", "float16"), ("cuda", "int8"), ("cpu", "int8")]
            else:
                candidates = [("cpu", "int8")]
            vlog(f"[INFO ] Loading Whisper model '{args.whisper_model}' on {args.whisper_device}...")
            for device, compute_type in candidates:
                try:
                    whisper_model = _FasterWhisperModel(
                        args.whisper_model,
                        device=device,
                        compute_type=compute_type,
                    )
                    vlog(f"[INFO ] Whisper model loaded (device={device}, compute_type={compute_type}).\n")
                    break
                except Exception as e:
                    next_idx = candidates.index((device, compute_type)) + 1
                    if next_idx < len(candidates):
                        nd, nc = candidates[next_idx]
                        vlog(f"[INFO ] {device}/{compute_type} not available, retrying with {nd}/{nc}...\n")
                    else:
                        vlog(f"[WARN ] Failed to load Whisper model: {e}; audio fallback disabled.\n")
                        audio_fallback_enabled = False

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    vlog(f"Scanning for .mkv under: {root}")
    mkvs = list(root.rglob("*.mkv"))
    if not mkvs:
        print("No .mkv files found.")
        return

    vlog(f"Found {len(mkvs)} .mkv file(s).")
    vlog(f"Mode: {'DRY-RUN' if args.dry_run else 'RENAME'} | Model: {args.model} (blind: {args.blind_model}) | LLM conf >= {args.min_confidence}")
    if dest_root:
        vlog(f"Dest : {dest_root}")
    else:
        vlog("Dest : rename in place")
    vlog(
        f"Audio fallback: {'ON' if audio_fallback_enabled else 'OFF'} "
        f"(primary {PRIMARY_AUDIO_SECONDS:.0f}s @ {args.audio_start_seconds:.0f}s, "
        f"second {PRIMARY_AUDIO_SECONDS:.0f}s @ {args.audio_start_seconds + PRIMARY_AUDIO_SECONDS:.0f}s, "
        f"deep fallback {args.audio_seconds:.0f}s)"
    )
    if audio_fallback_enabled:
        vlog("[INFO ] --audio-seconds is used only as a deep fallback after combined clips.")
    vlog(f"TMDB verify: {'ON' if verify_api_enabled else 'OFF'} (min title match {args.tmdb_min_title_match:.2f})\n")

    total = 0
    renamed = 0
    dryrun = 0
    errors = 0

    skipped_already_named = 0
    skipped_parse = 0
    skipped_no_duration = 0
    renamed_blind = 0
    skipped_duration_range = 0
    skipped_no_evidence = 0
    skipped_non_episode = 0
    skipped_low_conf = 0
    skipped_missing_fields = 0
    skipped_target_exists = 0
    skipped_verify_failed = 0
    renamed_opensubtitles = 0
    skipped_conflict = 0

    # For summary notification
    renamed_episodes: List[str] = []   # "Show - SxxEyy - Title" per successful rename/move
    skipped_files: List[dict] = []     # {"file": str, "reason": str} per truly unresolvable skip
    extras_queue: List[Tuple[Path, str, int]] = []  # (path, show, season) — unidentified files to move to Extras
    extras_moved: List[str] = []       # "Show - S01 - Extra_N.mkv" per file moved to Extras

    used_subtitles = 0
    used_audio_primary = 0
    used_audio_fallback = 0

    # Track which (show, season, episode) have been claimed this run to detect duplicates.
    episode_claims: Dict[Tuple, Path] = {}

    # Per-folder episode guide (fetched once from TMDB, keyed by (show, season)).
    episode_guide_cache: Dict[Tuple, Optional[str]] = {}

    # Per-folder sort_hints.json overrides (keyed by folder name).
    folder_hints_cache: Dict[str, dict] = {}

    # Per-folder count of TMDB-clean identifications keyed by (folder, show, season).
    # When any (show, season) reaches 2 confirmations in a folder, write
    # sort_hints.json to that folder so the remaining files skip blind mode.
    # Each folder is treated in isolation — no cross-folder inference.
    folder_id_counts: Dict[Tuple[str, str, int], int] = {}
    folder_hints_written: set = set()

    def _maybe_write_folder_hints(folder_key: str, folder_path: Path, show_name: str,
                                  season_num: int, src_name: str,
                                  tmdb_clean: bool, llm_conf: float) -> None:
        if not tmdb_clean or llm_conf < 0.90:
            return
        if folder_key in folder_hints_written:
            return
        if not show_name or not isinstance(season_num, int) or season_num < 1:
            return
        ck = (folder_key, norm_title(show_name), season_num)
        folder_id_counts[ck] = folder_id_counts.get(ck, 0) + 1
        if folder_id_counts[ck] < 2:
            return
        folder_hints_written.add(folder_key)
        folder_hints_cache[folder_key] = {
            "show": show_name,
            "season": season_num,
            "_auto": True,
            "_source_file": src_name,
        }
        if args.dry_run:
            if verbose:
                print(f"[HINT ] DRYRUN would auto-write sort_hints.json after 2 confirmations "
                      f"(show={show_name!r} season={season_num})")
            return
        if write_auto_sort_hints(folder_path, show_name, season_num, src_name) and verbose:
            print(f"[HINT ] auto-wrote sort_hints.json after 2 confirmations "
                  f"(show={show_name!r} season={season_num})")

    # Highest episode number identified so far per (show_key, season) across ALL disc folders.
    # Used to give the LLM a cross-disc "previous disc ended at E{N}" hint.
    show_season_last_ep: Dict[Tuple, int] = {}

    # Per-folder disc context: episodes already identified from the current folder.
    # Reset each time we move to a new parent folder.
    _current_folder: Optional[str] = None
    disc_context_list: List[str] = []

    # Files skipped during blind-mode processing that may be salvageable in a
    # second pass once their folder gains auto-written hints from later files.
    retry_candidates: List[Dict[str, Any]] = []

    # Phase 2 reconciliation state. Phase 1 used to call rename_and_move
    # immediately; now it queues entries here and a folder-boundary flush
    # reconciles them against contiguousness + TMDB runtime range before
    # actually moving anything. Each pending entry carries the Phase 1
    # proposal plus enough metadata for the reconciler to override it.
    folder_pending: Dict[str, List[Dict[str, Any]]] = {}
    folder_show_season: Dict[str, Tuple[str, int]] = {}

    def _queue_for_reconcile(folder_key: str, mkv_path: Path, show_name: str,
                              season_num: int, ep_num: int, ep_title: str,
                              llm_conf: float, dur_minutes: Optional[float],
                              tmdb_clean: bool, evidence_text: Optional[str] = None,
                              evidence_kind: Optional[str] = None,
                              is_blind: bool = False) -> None:
        folder_pending.setdefault(folder_key, []).append({
            "mkv": mkv_path,
            "show": show_name,
            "season": season_num,
            "proposed_ep": ep_num,
            "proposed_title": ep_title,
            "confidence": llm_conf,
            "dur_m": dur_minutes,
            "tmdb_confirmed_clean": tmdb_clean,
            "evidence_text": evidence_text,
            "evidence_kind": evidence_kind,
            "is_blind": is_blind,
        })
        # Remember the canonical (show, season) for the folder so the
        # reconciler can fetch the right TMDB episode list. First-write wins
        # because later files in the same folder always share the season.
        folder_show_season.setdefault(folder_key, (show_name, season_num))

    def _synopsis_consensus_shift(
        folder_key: str,
        entries: List[Dict[str, Any]],
        tmdb_eps: List[TmdbEpisode],
        max_canon: int,
    ) -> Any:
        """For each file in a contiguous-run shortcut candidate, ask the LLM
        which of the nearby episodes (proposed ± 2) best matches the file's
        evidence based on plot synopses. Each file's pick votes for a shift
        relative to its Phase 1 proposed episode. Returns:
          - None if the check cannot run (no synopses, no evidence retained)
          - "refuse" if the LLM broadly rejects nearby candidates (Phase 1 is
            wrong in a way a simple shift can't fix)
          - int (negative, zero, positive) shift to apply when a majority of
            files agree on a single shift
        Ambiguous votes (no majority) fall back to the original Phase 1
        assignment (shift = 0) so we never make things worse than the prior
        unchecked shortcut behaviour."""
        if not tmdb_eps or not any(e.overview for e in tmdb_eps):
            return None  # synopsis data unavailable, can't check

        shift_votes: Dict[int, int] = {}
        null_count = 0
        checked = 0

        for entry in entries:
            evidence_text = entry.get("evidence_text")
            evidence_kind = entry.get("evidence_kind") or "evidence"
            if not evidence_text:
                continue
            proposed = entry["proposed_ep"]
            candidates: List[Tuple[int, str, str]] = []
            for delta in (-2, -1, 0, 1, 2):
                cand_ep = proposed + delta
                if cand_ep < 1 or (max_canon and cand_ep > max_canon):
                    continue
                ep_obj = next((x for x in tmdb_eps if x.episode_number == cand_ep), None)
                if ep_obj is None:
                    continue
                # Require some kind of synopsis for a non-trivial check.
                # Allow the proposed ep through even without an overview so it
                # remains a default option for the LLM to fall back to.
                if not ep_obj.overview and delta != 0:
                    continue
                candidates.append((cand_ep, ep_obj.name, ep_obj.overview))
            if len(candidates) < 2:
                continue

            try:
                picked = call_llm_synopsis_pick(
                    llm_client, args.model, evidence_text, evidence_kind, candidates
                )
            except Exception as e:
                if verbose:
                    print(f"[SYNOPSIS] check failed for {entry['mkv'].name}: {e}")
                continue

            checked += 1
            if picked is None:
                null_count += 1
                if verbose:
                    print(f"[SYNOPSIS] {entry['mkv'].name}: no nearby episode matched")
            else:
                shift = picked - proposed
                shift_votes[shift] = shift_votes.get(shift, 0) + 1
                if verbose and shift != 0:
                    print(f"[SYNOPSIS] {entry['mkv'].name}: Phase 1 said "
                          f"E{proposed:02d}, synopsis best matches E{picked:02d} "
                          f"(vote shift {shift:+d})")

        if checked == 0:
            return None
        # Broad rejection: more than half the checked files said "no match."
        # That means the contiguous shortcut isn't just shifted, it's wrong.
        if null_count * 2 > checked:
            return "refuse"
        if not shift_votes:
            return "refuse"

        consensus_shift, votes = max(shift_votes.items(), key=lambda kv: kv[1])
        total_voters = sum(shift_votes.values())
        # Require a strict majority of voters to agree before we accept a
        # non-zero shift. Otherwise default to Phase 1's original assignment.
        if votes * 2 > total_voters:
            return consensus_shift
        return 0

    def _flush_folder(folder_key: str) -> None:
        """Reconcile Phase 1 proposals for a folder against contiguousness
        and TMDB runtime range, then actually rename. Per the design:
        - runtime outliers move to Extras (TMDB season [min, max] ± 20%)
        - the remaining files in disc filename order are assumed contiguous
          in episode number; the starting offset is found by anchor agreement
        - contiguousness beats individual high-confidence Phase 1 answers"""
        nonlocal renamed, dryrun, renamed_blind, skipped_target_exists, errors
        pending = folder_pending.pop(folder_key, [])
        if not pending:
            return
        show, season = folder_show_season.pop(folder_key, (pending[0]["show"], pending[0]["season"]))

        # Pull TMDB episode list for runtime range + title lookup on override.
        tmdb_eps: List[TmdbEpisode] = []
        if tmdb_client is not None:
            try:
                tv_id = tmdb_client.find_tv_id(show)
                if tv_id:
                    tmdb_eps = tmdb_client.get_season_episodes(tv_id, season) or []
            except Exception as e:
                if verbose:
                    print(f"[RECONCILE] TMDB season fetch failed for {show!r} S{season:02d}: {e}")

        runtimes = [e.runtime for e in tmdb_eps if e.runtime]
        if runtimes:
            r_min = min(runtimes) * 0.8
            r_max = max(runtimes) * 1.2
        else:
            r_min = r_max = None

        episode_entries: List[Dict[str, Any]] = []
        for entry in pending:
            d = entry["dur_m"]
            if r_min is not None and r_max is not None and d is not None and (d < r_min or d > r_max):
                if verbose:
                    print(f"[EXTRA-RANGE] {entry['mkv'].name} {d:.1f}m outside season runtime "
                          f"range [{r_min:.1f}, {r_max:.1f}]m -> Extras")
                extras_queue.append((entry["mkv"], show, season))
                # Also free its claim so we don't block a future legitimate ID.
                claim = (show.lower(), season, entry["proposed_ep"])
                if episode_claims.get(claim) == entry["mkv"]:
                    episode_claims.pop(claim, None)
            else:
                episode_entries.append(entry)

        if not episode_entries:
            return

        episode_entries.sort(key=lambda e: e["mkv"].name)

        # Drop Phase 1 outliers via median + MAD. A file whose proposed
        # episode sits far outside the rest of the folder's proposals is
        # almost certainly an alt-cut, commentary track, or otherwise extra
        # whose content fooled the LLM into a different-season slot. Keeping
        # it in pending forces the offset search to fit it, which can shift
        # the rest of the folder off by one and drop the highest episode.
        if len(episode_entries) >= 4:
            props_for_med = sorted(e["proposed_ep"] for e in episode_entries)
            mid = len(props_for_med) // 2
            median_prop = props_for_med[mid]
            mad_sorted = sorted(abs(e["proposed_ep"] - median_prop) for e in episode_entries)
            mad = mad_sorted[mid]
            if mad > 0:
                kept_entries: List[Dict[str, Any]] = []
                for entry in episode_entries:
                    mod_z = abs(entry["proposed_ep"] - median_prop) / (1.4826 * mad)
                    if mod_z > 3.5:
                        print(f"[EXTRA-OUTLIER] {entry['mkv'].name} Phase 1 said "
                              f"S{season:02d}E{entry['proposed_ep']:02d} — far from "
                              f"folder median E{median_prop:02d}, routing to Extras")
                        extras_queue.append((entry["mkv"], show, season))
                        claim = (show.lower(), season, entry["proposed_ep"])
                        if episode_claims.get(claim) == entry["mkv"]:
                            episode_claims.pop(claim, None)
                    else:
                        kept_entries.append(entry)
                episode_entries = kept_entries

        if not episode_entries:
            return

        # Shortcut: if Phase 1's IDs (after outlier removal) form a contiguous
        # episode run when sorted, Phase 1 is internally self-consistent — the
        # LLM correctly identified the set of episodes, even if disc filename
        # order doesn't match episode order. But "internally consistent" is not
        # the same as "correct" — the LLM can be systematically off by N across
        # a whole disc, with every individual ID title-matching TMDB. The
        # synopsis cross-check below catches that: each file votes for a shift
        # by comparing its evidence against the plot synopses of nearby
        # episodes. Consensus shift wins.
        if len(episode_entries) >= 2:
            sorted_proposed = sorted(e["proposed_ep"] for e in episode_entries)
            if sorted_proposed[-1] - sorted_proposed[0] == len(sorted_proposed) - 1 \
                    and all(e["confidence"] >= 0.85 for e in episode_entries):
                max_canon_for_shift = max((e.episode_number for e in tmdb_eps), default=0)
                consensus_shift = _synopsis_consensus_shift(
                    folder_key, episode_entries, tmdb_eps, max_canon_for_shift
                )
                # consensus_shift is None = couldn't run check; "refuse" = check
                # said Phase 1 broadly wrong; int = apply this shift (may be 0).
                if consensus_shift is None:
                    print(f"[RECONCILE] folder {folder_key!r}: Phase 1 IDs form contiguous "
                          f"run E{sorted_proposed[0]:02d}-E{sorted_proposed[-1]:02d} "
                          f"(synopsis check unavailable) — trusting Phase 1 as-is")
                    _apply_reconciled_renames(folder_key, show, season, episode_entries,
                                              [e["proposed_ep"] for e in episode_entries],
                                              tmdb_eps)
                    return
                if consensus_shift == "refuse":
                    print(f"[SYNOPSIS] folder {folder_key!r}: synopsis check rejected the "
                          f"contiguous run — falling through to file-order offset search")
                    # do NOT return; fall through to the offset-search path below
                else:
                    shifted = [e["proposed_ep"] + consensus_shift for e in episode_entries]
                    if any(a < 1 or a > max_canon_for_shift for a in shifted):
                        print(f"[SYNOPSIS] folder {folder_key!r}: consensus shift "
                              f"{consensus_shift:+d} would place episodes outside the "
                              f"season — falling through to offset search")
                    else:
                        if consensus_shift == 0:
                            print(f"[RECONCILE] folder {folder_key!r}: Phase 1 IDs form "
                                  f"contiguous run E{sorted_proposed[0]:02d}-"
                                  f"E{sorted_proposed[-1]:02d}, synopsis check confirms — "
                                  f"trusting per-file assignments")
                        else:
                            new_lo = sorted_proposed[0] + consensus_shift
                            new_hi = sorted_proposed[-1] + consensus_shift
                            print(f"[RECONCILE] folder {folder_key!r}: Phase 1 said "
                                  f"E{sorted_proposed[0]:02d}-E{sorted_proposed[-1]:02d}, "
                                  f"synopsis consensus shifts run to E{new_lo:02d}-"
                                  f"E{new_hi:02d} (shift {consensus_shift:+d})")
                        _apply_reconciled_renames(folder_key, show, season, episode_entries,
                                                  shifted, tmdb_eps)
                        return

        proposed = [e["proposed_ep"] for e in episode_entries]
        n = len(episode_entries)

        # Anchors: Phase 1 entries we trust enough to pin the offset. We accept
        # any conf >= 0.85; TMDB-confirmed-without-correction is given extra
        # weight inside the score function below.
        anchors_idx = [i for i, e in enumerate(episode_entries)
                       if e["confidence"] >= 0.85]
        if not anchors_idx:
            print(f"[RECONCILE] folder {folder_key!r}: no high-confidence anchor — "
                  f"refusing to guess offset, files left in place")
            return

        # Search every plausible starting episode and score how well it
        # agrees with anchors + runtimes.
        max_canon = max((e.episode_number for e in tmdb_eps), default=0) or (max(proposed) + n)
        search_lo = 1
        search_hi = max(1, max_canon - n + 1)

        def _score(start: int) -> Tuple[int, float]:
            anchor_score = 0
            runtime_bonus = 0.0
            for i, entry in enumerate(episode_entries):
                expected = start + i
                if expected == entry["proposed_ep"]:
                    pts = int(round(entry["confidence"] * 100))
                    if entry.get("tmdb_confirmed_clean"):
                        pts += 20  # extra weight for TMDB-clean confirmations
                    anchor_score += pts
                if tmdb_eps and entry["dur_m"] is not None:
                    ep_obj = next((x for x in tmdb_eps if x.episode_number == expected), None)
                    if ep_obj and ep_obj.runtime and abs(ep_obj.runtime - entry["dur_m"]) <= 3.0:
                        runtime_bonus += 5.0
            return anchor_score, runtime_bonus

        best_start = None
        best_score = (-1, -1.0)
        for start in range(search_lo, search_hi + 1):
            s = _score(start)
            if s > best_score:
                best_score = s
                best_start = start

        if best_start is None:
            print(f"[RECONCILE] folder {folder_key!r}: no plausible offset, files left in place")
            return

        anchor_pts, runtime_pts = best_score
        print(f"[RECONCILE] folder {folder_key!r}: best offset E{best_start:02d} "
              f"(anchor_score={anchor_pts} runtime_bonus={runtime_pts:.0f}, {n} file(s))")

        _apply_reconciled_renames(folder_key, show, season, episode_entries,
                                  [best_start + i for i in range(n)], tmdb_eps)

    def _apply_reconciled_renames(folder_key: str, show: str, season: int,
                                   entries: List[Dict[str, Any]],
                                   assigned_eps: List[int],
                                   tmdb_eps: List[TmdbEpisode]) -> None:
        nonlocal renamed, dryrun, renamed_blind, skipped_target_exists
        # Drop stale claims from Phase 1 for this (show, season) before we
        # rebuild them from the reconciled assignments.
        stale = [k for k in episode_claims
                 if k[0] == show.lower() and k[1] == season]
        for k in stale:
            episode_claims.pop(k, None)

        for entry, assigned_ep in zip(entries, assigned_eps):
            assigned_title = entry["proposed_title"]
            ep_obj = next((x for x in tmdb_eps if x.episode_number == assigned_ep), None)
            if ep_obj:
                assigned_title = ep_obj.name

            if assigned_ep != entry["proposed_ep"]:
                print(f"[OVERRIDE] {entry['mkv'].name}: Phase 1 said "
                      f"S{season:02d}E{entry['proposed_ep']:02d} \"{entry['proposed_title']}\" "
                      f"-> reconciled to S{season:02d}E{assigned_ep:02d} \"{assigned_title}\" "
                      f"(order)")

            ok = rename_and_move(entry["mkv"], show, season, assigned_ep,
                                 assigned_title, args.dry_run, dest_root)
            if ok:
                claim = (show.lower(), season, assigned_ep)
                episode_claims[claim] = entry["mkv"]
                disc_context_list.append(f"S{season:02d}E{assigned_ep:02d} \"{assigned_title}\"")
                renamed_episodes.append(
                    f"{sanitize_filename(show)} - S{season:02d}E{assigned_ep:02d} - {sanitize_filename(assigned_title)}"
                )
                sk = (norm_title(show), season)
                show_season_last_ep[sk] = max(show_season_last_ep.get(sk, 0), assigned_ep)
                if args.dry_run:
                    dryrun += 1
                else:
                    renamed += 1
                    if entry.get("is_blind"):
                        renamed_blind += 1
                print(f"[DONE ] renamed (reconciled)")
            else:
                skipped_target_exists += 1
                print(f"[SKIP ] target exists / rename not performed: "
                      f"S{season:02d}E{assigned_ep:02d}")

    for mkv in mkvs:
        total += 1

        hr()
        print(f"FILE : {short_path(mkv, 110)}")

        if re.search(r"\bS\d{2}E\d{2}\b", mkv.stem, flags=re.IGNORECASE):
            skipped_already_named += 1
            print(f"[SKIP ] already looks named (SxxEyy) -> {mkv.name}")
            continue

        folder = mkv.parent.name

        # Load sort_hints.json from the source folder (cached per folder).
        if folder not in folder_hints_cache:
            hints_path = mkv.parent / "sort_hints.json"
            try:
                folder_hints_cache[folder] = json.loads(hints_path.read_text()) if hints_path.exists() else {}
            except Exception as e:
                print(f"[WARN ] Could not read sort_hints.json in {folder}: {e}")
                folder_hints_cache[folder] = {}
            # Multi-disc box sets (ARRESTED_D1/D2/D3) usually share a season.
            # If this folder has no own hints, inherit from a sibling disc that
            # already has clean hints — manual or auto-written from an earlier
            # disc in this same run. Stops D2/D3 from running blind and
            # anchoring on the wrong season.
            if not folder_hints_cache[folder]:
                sib_hints = find_sibling_disc_hints(mkv.parent)
                if sib_hints:
                    folder_hints_cache[folder] = sib_hints
                    folder_hints_written.add(folder)
                    vlog(f"[HINT ] inherited from sibling disc {sib_hints['_inherited_from']!r}: "
                         f"show={sib_hints['show']!r} season={sib_hints['season']}")
        hints = folder_hints_cache[folder]
        hints_show = str(hints.get("show", "")).strip()
        hints_season = hints.get("season")
        hints_skip_tmdb = bool(hints.get("skip_tmdb", False))

        if hints_show and isinstance(hints_season, int) and hints_season >= 1:
            show, season = hints_show, hints_season
            folder_parse_failed = False
            vlog(f"[HINT ] sort_hints.json: show={show!r} season={season} skip_tmdb={hints_skip_tmdb}")
        else:
            hints_skip_tmdb = False
            vlog(f"[PARSE] LLM parsing folder name: \"{folder}\"")
            show, season = parse_show_and_season_with_llm(llm_client, folder)
            if not show or not season:
                vlog("[PARSE] LLM parse missing fields; falling back to regex parsing.")
                show, season = parse_show_and_season_from_folder(folder)
            # Boxed-set discs ('ARRESTED_D1', 'Show Disc 2') almost always sit inside
            # a single season — default to season 1 when we have a show but no season
            # info. sort_hints.json overrides this for the rare multi-season set.
            if show and not season and _DISC_MARKER_RE.search(folder):
                vlog(f"[PARSE] Folder \"{folder}\" has disc marker but no season — defaulting season=1.")
                season = 1
            folder_parse_failed = not show or not season
            if folder_parse_failed:
                vlog(f"[PARSE] Folder \"{folder}\" yielded no show/season — will attempt blind identification from content.")

        # TMDB show name canonicalization — skipped when hints override or skip_tmdb set
        if not folder_parse_failed and not hints_skip_tmdb and tmdb_client is not None:
            try:
                tmdb_show = tmdb_client.find_tv(show)
            except Exception as e:
                errors += 1
                print(f"[ERR  ] TMDB canonicalize failed: {e}")
                continue
            if tmdb_show and norm_title(tmdb_show.name) != norm_title(show):
                if verbose:
                    print(f"[API  ] TMDB canonical show -> \"{tmdb_show.name}\"")
                show = tmdb_show.name

        # Folder boundary: flush the previous folder's Phase 1 proposals
        # through the reconciler, then reset per-folder state.
        if folder != _current_folder:
            if _current_folder is not None:
                _flush_folder(_current_folder)
            _current_folder = folder
            disc_context_list = []

        dur_s = ffprobe_duration_seconds(mkv)
        if dur_s is None:
            skipped_no_duration += 1
            print("[SKIP ] ffprobe couldn't read duration (is ffprobe on PATH? file readable?)")
            continue

        dur_m = dur_s / 60.0
        if show and season:
            print(f"HINT : {show} | S{season:02d} | {dur_m:.1f}m")
        else:
            print(f"HINT : [blind] | {dur_m:.1f}m")

        if dur_m < args.min_minutes or dur_m > args.max_minutes:
            skipped_duration_range += 1
            print(f"[SKIP ] duration {dur_m:.1f}m out of range [{args.min_minutes:.1f}, {args.max_minutes:.1f}]")
            if dest_root and show and season:
                extras_queue.append((mkv, show, season))
            continue

        # Fetch episode guide from TMDB (cached per show+season) for LLM context
        episode_guide: Optional[str] = None
        if not folder_parse_failed and not hints_skip_tmdb and tmdb_client is not None and show and season:
            guide_key = (show.lower(), season)
            if guide_key not in episode_guide_cache:
                try:
                    tv_id = tmdb_client.find_tv_id(show)
                    if tv_id:
                        eps = tmdb_client.get_season_episodes(tv_id, season)
                        episode_guide_cache[guide_key] = format_episode_guide(eps, season, dur_m) if eps else None
                    else:
                        episode_guide_cache[guide_key] = None
                except Exception:
                    episode_guide_cache[guide_key] = None
            episode_guide = episode_guide_cache.get(guide_key)

        # OpenSubtitles exact hash rename requires a known show + season
        if not folder_parse_failed and args.opensubtitles_exact_rename:
            api_key = os.environ.get("OPENSUB_API_KEY")
            if not api_key:
                if verbose:
                    print("[OSDB ] OPENSUB_API_KEY not set; skipping OpenSubtitles lookup")
            else:
                try:
                    os_match = opensubtitles_exact_match(
                        api_key=api_key,
                        user_agent=args.opensubtitles_user_agent,
                        mkv_path=mkv,
                        show=show,
                        season=season,
                        verbose=verbose,
                    )
                except Exception as e:
                    errors += 1
                    print(f"[ERR  ] OpenSubtitles lookup failed: {e}")
                    os_match = None

                if os_match:
                    ep_num, ep_title = os_match
                    if verbose:
                        print(f"[OSDB ] exact match -> S{season:02d}E{ep_num:02d} \"{ep_title}\"")
                    ok = rename_and_move(mkv, show, season, ep_num, ep_title, args.dry_run, dest_root)
                    if ok:
                        print("[DONE ] renamed (OpenSubtitles exact match)")
                        if args.dry_run:
                            dryrun += 1
                        else:
                            renamed += 1
                        renamed_opensubtitles += 1
                        continue
                elif verbose:
                    print("[OSDB ] lookup completed, no exact match")

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

        def clamp_clip_start(requested_start: float, duration_seconds: float) -> Optional[float]:
            latest_start = max(0.0, dur_s - duration_seconds - 1.0)
            if latest_start == 0.0 and dur_s < (duration_seconds + 1.0):
                return None
            return min(requested_start, latest_start)

        def attempt_audio_transcribe(seconds: float, stage: str, start_seconds: float) -> Tuple[Optional[str], Optional[float]]:
            jitter = random.uniform(AUDIO_START_JITTER_MIN, AUDIO_START_JITTER_MAX)
            jittered_start = start_seconds + jitter
            actual_start = clamp_clip_start(jittered_start, seconds)
            if actual_start is None:
                if verbose:
                    print(f"[AUDIO] {stage} clip skipped (duration {seconds:.0f}s exceeds file length {dur_s:.0f}s)")
                return None, None
            if verbose:
                print(f"[AUDIO] {stage} clip attempt ({seconds:.0f}s @ {actual_start:.0f}s, jitter +{jitter:.1f}s)")
            with tempfile.TemporaryDirectory() as td:
                wav_path = Path(td) / "clip.wav"
                ok = extract_audio_clip_wav(
                    mkv,
                    wav_path,
                    actual_start,
                    seconds,
                    verbose=verbose
                )
                if not ok:
                    if verbose:
                        print(f"[ASR  ] {stage} transcript: FAILED (audio extraction)")
                    return None, None
                try:
                    tx = transcribe_with_faster_whisper(whisper_model, wav_path)
                except Exception as e:
                    if verbose:
                        print(f"[ERR  ] {stage} transcription failed: {e}")
                    return None, None
                if not tx:
                    if verbose:
                        print(f"[ASR  ] {stage} transcript: EMPTY")
                    return None, None
                if verbose:
                    print(f"[ASR  ] {stage} transcript: \"{one_line(tx)}\"")
                return tx, actual_start

        audio_transcript_primary = None
        audio_transcript_second = None
        audio_transcript_fallback = None
        primary_actual_start = None
        second_actual_start = None
        deep_actual_start = None
        primary_start = float(args.audio_start_seconds)
        second_start = primary_start + PRIMARY_AUDIO_SECONDS
        deep_fallback_start = second_start + PRIMARY_AUDIO_SECONDS

        if evidence_text is None and audio_fallback_enabled:
            audio_transcript_primary, primary_actual_start = attempt_audio_transcribe(
                PRIMARY_AUDIO_SECONDS,
                "PRIMARY",
                primary_start
            )
            if audio_transcript_primary:
                evidence_text = audio_transcript_primary
                evidence_kind = (
                    f"audio transcript (Whisper, {PRIMARY_AUDIO_SECONDS:.0f}s @ "
                    f"{primary_actual_start:.0f}s)"
                )
                used_audio_primary += 1
            else:
                if verbose:
                    print("[RETRY] PRIMARY transcript unavailable -> trying SECOND clip")
                audio_transcript_second, second_actual_start = attempt_audio_transcribe(
                    PRIMARY_AUDIO_SECONDS,
                    "SECOND",
                    second_start
                )
                if audio_transcript_second:
                    evidence_text = audio_transcript_second
                    evidence_kind = (
                        f"audio transcript (Whisper, {PRIMARY_AUDIO_SECONDS:.0f}s @ "
                        f"{second_actual_start:.0f}s)"
                    )
                    used_audio_fallback += 1
                else:
                    fallback_seconds = float(args.audio_seconds)
                    if fallback_seconds < PRIMARY_AUDIO_SECONDS:
                        fallback_seconds = PRIMARY_AUDIO_SECONDS
                    if verbose:
                        print(f"[RETRY] SECOND transcript unavailable -> trying DEEP FALLBACK {fallback_seconds:.0f}s")
                    audio_transcript_fallback, deep_actual_start = attempt_audio_transcribe(
                        fallback_seconds,
                        "DEEP",
                        deep_fallback_start
                    )
                    if audio_transcript_fallback:
                        evidence_text = audio_transcript_fallback
                        evidence_kind = (
                            f"audio transcript (Whisper, {fallback_seconds:.0f}s @ "
                            f"{deep_actual_start:.0f}s)"
                        )
                        used_audio_fallback += 1

        if evidence_text is None:
            skipped_no_evidence += 1
            print("[SKIP ] no usable evidence text (subtitles unavailable and transcription failed/disabled)")
            if dest_root and show and season:
                extras_queue.append((mkv, show, season))
            continue

        if verbose:
            print(f"[EVID ] {evidence_kind}")

        _show_season_key = (norm_title(show) if show else "", season or 0)
        _cross_disc_last = show_season_last_ep.get(_show_season_key) if (show and season and not disc_context_list) else None

        def attempt_llm_with_evidence(stage: str, e_text: str, e_kind: str) -> Tuple[bool, Optional[dict], str]:
            nonlocal errors
            try:
                result = call_llm_identify(
                    llm_client, args.model, show, season, e_text, dur_m, e_kind,
                    episode_guide=episode_guide,
                    disc_context=list(disc_context_list) or None,
                    is_collection=hints_skip_tmdb,
                    cross_disc_last_ep=_cross_disc_last,
                )
            except DeepSeekAuthError:
                raise  # fatal, run-wide — abort instead of silently skipping
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

            # In collection mode (skip_tmdb), trust the LLM's title identification
            # even if it flagged is_episode=false (standalone specials are not "episodes"
            # in the traditional sense, but we still want to name them).
            if not is_episode and not hints_skip_tmdb:
                return False, result, "non_episode"
            if not is_episode and hints_skip_tmdb and (not ep_title or conf < args.min_confidence):
                # Truly unidentifiable even in collection mode
                return False, result, "non_episode"
            if conf < args.min_confidence:
                return False, result, "low_conf"
            if not isinstance(ep_num, int) or not ep_title:
                return False, result, "missing_fields"

            final_ep_num = ep_num
            final_title = ep_title
            tmdb_confirmed_clean = False

            if verify_api_enabled and not hints_skip_tmdb and tmdb_client is not None:
                try:
                    vres = verify_or_correct_with_tmdb(
                        tmdb=tmdb_client,
                        show=show,
                        season=season,
                        proposed_ep_num=ep_num,
                        proposed_title=ep_title,
                        min_title_match=float(args.tmdb_min_title_match),
                        file_duration_minutes=dur_m,
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
                tmdb_confirmed_clean = not vres.corrected
            elif hints_skip_tmdb and verbose:
                print(f"[HINT ] TMDB verify skipped (sort_hints.json skip_tmdb=true)")

            # Duplicate detection: if another file this run already claimed this episode,
            # return "conflict" so the caller can retry with audio evidence instead.
            claim_key = (show.lower() if show else "", season or 0, final_ep_num)
            if claim_key in episode_claims:
                if verbose:
                    print(f"[WARN ] S{season:02d}E{final_ep_num:02d} already claimed by "
                          f"{episode_claims[claim_key].name} — forcing audio retry")
                return False, result, "conflict"
            episode_claims[claim_key] = mkv

            # Queue for Phase 2 reconciliation instead of renaming now. The
            # reconciler at the folder boundary may override final_ep_num /
            # final_title before the actual move.
            _queue_for_reconcile(folder, mkv, show, season, final_ep_num,
                                  final_title, conf, dur_m, tmdb_confirmed_clean,
                                  evidence_text=e_text, evidence_kind=e_kind)
            disc_context_list.append(f"S{season:02d}E{final_ep_num:02d} \"{final_title}\"")
            sk = (norm_title(show), season)
            show_season_last_ep[sk] = max(show_season_last_ep.get(sk, 0), final_ep_num)
            _maybe_write_folder_hints(folder, mkv.parent, show, season, mkv.name,
                                      tmdb_confirmed_clean, conf)
            return True, result, "queued"

        if folder_parse_failed:
            # Blind mode: redefine attempt_llm_with_evidence to identify show+season+episode
            # from content alone, with no folder-name hints.
            def attempt_llm_with_evidence(stage: str, e_text: str, e_kind: str) -> Tuple[bool, Optional[dict], str]:  # noqa: F811
                nonlocal errors, renamed_blind
                try:
                    result = call_llm_identify_blind(llm_client, args.blind_model, e_text, dur_m, e_kind,
                                                    disc_context=list(disc_context_list) or None)
                except DeepSeekAuthError:
                    raise  # fatal, run-wide — abort instead of silently skipping
                except Exception as e:
                    errors += 1
                    print(f"[ERR  ] Blind LLM call failed ({stage}): {e}")
                    return False, None, "llm_error"

                blind_show = (result.get("show") or "").strip()
                blind_season = result.get("season")
                ep_num = result.get("episode_number")
                ep_title = result.get("episode_title")
                conf = float(result.get("confidence", 0.0))
                is_ep = bool(result.get("is_episode"))

                if verbose:
                    if is_ep and blind_show and isinstance(blind_season, int) and isinstance(ep_num, int) and ep_title:
                        mark = "✓" if conf >= args.min_confidence else "✗"
                        print(f"[LLM  ] {stage} (blind) -> {blind_show} S{blind_season:02d}E{ep_num:02d} \"{ep_title}\" (conf={conf:.2f}) {mark}")
                    else:
                        print(f"[LLM  ] {stage} (blind) -> non-episode/unknown (conf={conf:.2f}) ✗")
                    notes = one_line(result.get("notes", ""), 140)
                    if notes and (not is_ep or conf < args.min_confidence):
                        print(f"       reason: {notes}")

                if not is_ep:
                    return False, result, "non_episode"
                if conf < args.min_confidence:
                    return False, result, "low_conf"
                if not blind_show or not isinstance(blind_season, int) or blind_season < 1 \
                        or not isinstance(ep_num, int) or not ep_title:
                    return False, result, "missing_fields"

                final_show = blind_show
                final_season = blind_season
                final_ep_num = ep_num
                final_title = ep_title
                tmdb_confirmed_clean = False

                if verify_api_enabled and tmdb_client is not None:
                    # Canonicalize show name first
                    try:
                        tmdb_show_result = tmdb_client.find_tv(blind_show)
                    except Exception as e:
                        errors += 1
                        print(f"[ERR  ] TMDB show lookup (blind) failed: {e}")
                        return False, result, "verify_error"

                    if tmdb_show_result:
                        if verbose and norm_title(tmdb_show_result.name) != norm_title(blind_show):
                            print(f"[API  ] TMDB canonical show -> \"{tmdb_show_result.name}\"")
                        final_show = tmdb_show_result.name

                        # Show found — verify the episode
                        try:
                            vres = verify_or_correct_with_tmdb(
                                tmdb=tmdb_client,
                                show=final_show,
                                season=blind_season,
                                proposed_ep_num=ep_num,
                                proposed_title=ep_title,
                                min_title_match=float(args.tmdb_min_title_match),
                                file_duration_minutes=dur_m,
                            )
                        except Exception as e:
                            errors += 1
                            print(f"[ERR  ] TMDB verify (blind) failed: {e}")
                            return False, result, "verify_error"

                        if verbose:
                            if vres.ok:
                                tag = "corrected" if vres.corrected else "confirmed"
                                print(f"[API  ] TMDB {tag}  -> S{blind_season:02d}E{vres.episode_number:02d} \"{vres.episode_title}\" (match={vres.match_score:.2f})")
                            else:
                                print(f"[API  ] TMDB reject     -> {vres.reason}")

                        if not vres.ok or vres.episode_number is None or not vres.episode_title:
                            return False, result, "verify_failed"

                        final_ep_num = vres.episode_number
                        final_title = vres.episode_title
                        tmdb_confirmed_clean = not vres.corrected
                    else:
                        # Show not found in TMDB — likely a standalone special, TV movie, or
                        # compilation content not catalogued as a TV series. Trust the LLM.
                        if verbose:
                            print(f"[API  ] TMDB: show not found — trusting LLM identification (conf={conf:.2f})")
                else:
                    if verbose:
                        print("[WARN ] Blind identification without TMDB verification — accuracy not guaranteed.")

                # Duplicate detection for blind mode
                claim_key = (final_show.lower(), final_season, final_ep_num)
                if claim_key in episode_claims:
                    if verbose:
                        print(f"[WARN ] {final_show} S{final_season:02d}E{final_ep_num:02d} already claimed by "
                              f"{episode_claims[claim_key].name} — forcing audio retry")
                    return False, result, "conflict"
                episode_claims[claim_key] = mkv

                _queue_for_reconcile(folder, mkv, final_show, final_season, final_ep_num,
                                      final_title, conf, dur_m, tmdb_confirmed_clean,
                                      evidence_text=e_text, evidence_kind=e_kind,
                                      is_blind=True)
                disc_context_list.append(f"{final_show} S{final_season:02d}E{final_ep_num:02d} \"{final_title}\"")
                sk = (norm_title(final_show), final_season)
                show_season_last_ep[sk] = max(show_season_last_ep.get(sk, 0), final_ep_num)
                _maybe_write_folder_hints(folder, mkv.parent, final_show, final_season,
                                          mkv.name, tmdb_confirmed_clean, conf)
                return True, result, "queued"

        renamed_ok, first_result, fail_code = attempt_llm_with_evidence("PRIMARY", evidence_text, evidence_kind)

        if renamed_ok:
            print("[QUEUE] held for folder reconciliation" + (" (blind)" if folder_parse_failed else ""))
            continue

        # Conflict: subtitle identified an already-claimed episode — retry with audio
        if fail_code == "conflict" and audio_fallback_enabled and whisper_model is not None:
            if verbose:
                print("[RETRY] Duplicate episode from subtitles — forcing audio identification")
            audio_tx, audio_start = attempt_audio_transcribe(PRIMARY_AUDIO_SECONDS, "CONFLICT-AUDIO", primary_start)
            if audio_tx:
                used_audio_primary += 1
                conflict_kind = f"audio transcript (Whisper, conflict-retry {PRIMARY_AUDIO_SECONDS:.0f}s @ {audio_start:.0f}s)"
                renamed_ok, first_result, fail_code = attempt_llm_with_evidence("CONFLICT-AUDIO", audio_tx, conflict_kind)
                if renamed_ok:
                    print("[QUEUE] held for folder reconciliation (after conflict audio retry)")
                    continue
            else:
                if verbose:
                    print("[RETRY] Audio unavailable for conflict retry")
                fail_code = "conflict"

        # Low-confidence subtitle result: retry with subtitle + audio combined evidence.
        # This covers episodes where the subtitle beginning is generic but audio reveals more.
        used_subtitles_as_primary = (evidence_text is not None) and (evidence_text != audio_transcript_primary)
        if fail_code == "low_conf" and used_subtitles_as_primary and audio_fallback_enabled and whisper_model is not None:
            if verbose:
                print("[RETRY] Low subtitle confidence — supplementing with audio")
            if audio_transcript_primary is None:
                audio_transcript_primary, primary_actual_start = attempt_audio_transcribe(
                    PRIMARY_AUDIO_SECONDS, "LOWCONF-AUDIO", primary_start
                )
                if audio_transcript_primary:
                    used_audio_primary += 1
            if audio_transcript_primary:
                primary_label = primary_actual_start if primary_actual_start is not None else primary_start
                combined_ev = (
                    f"SUBTITLE EXCERPT:\n\"{evidence_text}\"\n\n"
                    f"AUDIO TRANSCRIPT (Whisper, {PRIMARY_AUDIO_SECONDS:.0f}s @ {primary_label:.0f}s):\n"
                    f"\"{audio_transcript_primary}\""
                )
                combined_kind = (
                    f"subtitle excerpt + audio transcript "
                    f"(Whisper, {PRIMARY_AUDIO_SECONDS:.0f}s @ {primary_label:.0f}s)"
                )
                renamed_ok2, second_result, fail_code2 = attempt_llm_with_evidence(
                    "SUBTITLE+AUDIO", combined_ev, combined_kind
                )
                if renamed_ok2:
                    print("[QUEUE] held for folder reconciliation (after subtitle+audio fallback)")
                    continue
                if second_result is not None:
                    first_result = second_result
                    fail_code = fail_code2

        used_primary_audio = (audio_transcript_primary is not None) and (evidence_text == audio_transcript_primary)
        if used_primary_audio and audio_fallback_enabled:
            if audio_transcript_second is None:
                if verbose:
                    print("[RETRY] primary evidence didn't pass -> trying SECOND clip for combined evidence")
                audio_transcript_second, second_actual_start = attempt_audio_transcribe(
                    PRIMARY_AUDIO_SECONDS,
                    "SECOND",
                    second_start
                )
                if audio_transcript_second:
                    used_audio_fallback += 1

            if audio_transcript_second:
                primary_label_start = primary_actual_start if primary_actual_start is not None else primary_start
                second_label_start = second_actual_start if second_actual_start is not None else second_start
                combined_evidence = (
                    f"CLIP A (Whisper transcript, {PRIMARY_AUDIO_SECONDS:.0f}s @ {primary_label_start:.0f}s):\n"
                    f"\"{audio_transcript_primary}\"\n\n"
                    f"CLIP B (Whisper transcript, {PRIMARY_AUDIO_SECONDS:.0f}s @ {second_label_start:.0f}s):\n"
                    f"\"{audio_transcript_second}\""
                )
                combined_kind = (
                    "audio transcript (Whisper, multi-clip: "
                    f"A {PRIMARY_AUDIO_SECONDS:.0f}s @ {primary_label_start:.0f}s; "
                    f"B {PRIMARY_AUDIO_SECONDS:.0f}s @ {second_label_start:.0f}s)"
                )
                if verbose:
                    print("[EVID ] combined audio transcripts (A+B)")
                renamed_ok2, second_result, fail_code2 = attempt_llm_with_evidence(
                    "FALLBACK",
                    combined_evidence,
                    combined_kind
                )
                if renamed_ok2:
                    print("[QUEUE] held for folder reconciliation (after combined fallback)")
                    continue

                if second_result is not None:
                    first_result = second_result
                    fail_code = fail_code2
            else:
                if verbose:
                    print("[RETRY] SECOND clip unavailable -> skipping combined evidence")

            fallback_seconds = float(args.audio_seconds)
            if fallback_seconds < PRIMARY_AUDIO_SECONDS:
                fallback_seconds = PRIMARY_AUDIO_SECONDS
            if verbose:
                print(f"[RETRY] combined evidence didn't pass -> trying DEEP FALLBACK ({fallback_seconds:.0f}s)")
            if audio_transcript_fallback is None:
                audio_transcript_fallback, deep_actual_start = attempt_audio_transcribe(
                    fallback_seconds,
                    "DEEP",
                    deep_fallback_start
                )
                if audio_transcript_fallback:
                    used_audio_fallback += 1

            if audio_transcript_fallback:
                fb_kind = (
                    f"audio transcript (Whisper, {fallback_seconds:.0f}s @ "
                    f"{deep_actual_start:.0f}s)"
                )
                renamed_ok2, second_result, fail_code2 = attempt_llm_with_evidence(
                    "DEEP",
                    audio_transcript_fallback,
                    fb_kind
                )
                if renamed_ok2:
                    print("[DONE ] renamed (after deep fallback)")
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

        # Queue blind-mode skips for the end-of-run retry pass — later files in
        # this folder may auto-write hints that let us identify this file in a
        # second attempt using the TMDB episode guide.
        if (folder_parse_failed
                and evidence_text is not None
                and fail_code in ("low_conf", "missing_fields", "verify_failed", "non_episode")):
            retry_candidates.append({
                "mkv": mkv,
                "folder": folder,
                "evidence_text": evidence_text,
                "evidence_kind": evidence_kind,
                "dur_m": dur_m,
                "fail_code": fail_code,
            })

        def _queue_or_skip(reason: str) -> None:
            """Add to extras_queue when dest+context available, else skipped_files."""
            if dest_root and show and season:
                extras_queue.append((mkv, show, season))
            else:
                skipped_files.append({"file": mkv.name, "reason": reason})

        if fail_code == "conflict":
            skipped_conflict += 1
            print("[SKIP ] duplicate episode — audio retry could not resolve conflict")
            _queue_or_skip("duplicate episode — conflict unresolved")
            continue

        if fail_code == "non_episode":
            skipped_non_episode += 1
            print("[SKIP ] LLM says non-episode/unknown")
            _queue_or_skip("identified as non-episode (featurette/extra)")
            continue

        if fail_code == "low_conf":
            skipped_low_conf += 1
            conf = float((first_result or {}).get("confidence", 0.0))
            print(f"[SKIP ] confidence {conf:.2f} < {args.min_confidence:.2f}")
            _queue_or_skip(f"low confidence ({conf:.2f})")
            continue

        if fail_code == "missing_fields":
            skipped_missing_fields += 1
            print("[SKIP ] missing episode_number or episode_title")
            _queue_or_skip("LLM returned incomplete result")
            continue

        if fail_code == "verify_failed":
            skipped_verify_failed += 1
            print("[SKIP ] TMDB could not confirm episode number/title (preventing bad rename)")
            _queue_or_skip("TMDB could not verify episode")
            continue

        # Catch-all for any fall-through that didn't queue or hit a labelled
        # skip code (e.g. verify_error from an exception). Logged as
        # "unresolved" so the reconciler's "target exists" stays unambiguous.
        skipped_target_exists += 1
        print(f"[SKIP ] no rename queued (fail_code={fail_code!r})")

    # Final folder flush: the loop ended without crossing one more folder
    # boundary, so the last folder's pending entries still need reconciling.
    if _current_folder is not None:
        _flush_folder(_current_folder)

    # ----------------------------
    # Retry pass: re-attempt blind-mode skips for folders that gained hints
    # mid-run. With (show, season) now known, the hinted prompt + TMDB episode
    # guide can often identify content that was too ambiguous in blind mode.
    # ----------------------------
    retry_renamed_count = 0
    retry_eligible = [c for c in retry_candidates if c["folder"] in folder_hints_written]
    if retry_eligible:
        hr()
        print(f"\n=== RETRY PASS ({len(retry_eligible)} candidate(s)) ===")
        for cand in retry_eligible:
            mkv = cand["mkv"]
            folder = cand["folder"]
            fail_code = cand["fail_code"]
            evidence_text = cand["evidence_text"]
            evidence_kind = cand["evidence_kind"]
            dur_m = cand["dur_m"]

            if not mkv.exists():
                continue

            hints = folder_hints_cache.get(folder) or {}
            show = str(hints.get("show", "")).strip()
            season = hints.get("season")
            if not show or not isinstance(season, int) or season < 1:
                continue

            hr()
            print(f"RETRY: {short_path(mkv, 110)}")
            print(f"HINT (auto): {show} | S{season:02d} | {dur_m:.1f}m | prior={fail_code}")

            guide_key = (show.lower(), season)
            episode_guide = episode_guide_cache.get(guide_key)
            if episode_guide is None and tmdb_client is not None and verify_api_enabled:
                try:
                    tv_id = tmdb_client.find_tv_id(show)
                    if tv_id:
                        eps = tmdb_client.get_season_episodes(tv_id, season)
                        episode_guide = format_episode_guide(eps, season, dur_m) if eps else None
                        episode_guide_cache[guide_key] = episode_guide
                except Exception:
                    pass

            cross_last = show_season_last_ep.get((norm_title(show), season))

            try:
                result = call_llm_identify(
                    llm_client, args.model, show, season, evidence_text, dur_m, evidence_kind,
                    episode_guide=episode_guide,
                    disc_context=None,
                    is_collection=False,
                    cross_disc_last_ep=cross_last,
                )
            except Exception as e:
                errors += 1
                print(f"[ERR  ] retry LLM call failed: {e}")
                continue

            compact, _ = format_llm_compact(result, season, args.min_confidence)
            print(f"[LLM  ] RETRY -> {compact}")

            is_episode = bool(result.get("is_episode"))
            conf = float(result.get("confidence", 0.0))
            ep_num = result.get("episode_number")
            ep_title = result.get("episode_title")

            if not is_episode:
                print("[SKIP ] retry: LLM still says non-episode")
                continue
            if conf < args.min_confidence:
                print(f"[SKIP ] retry: confidence {conf:.2f} < {args.min_confidence:.2f}")
                continue
            if not isinstance(ep_num, int) or not ep_title:
                print("[SKIP ] retry: missing episode_number or episode_title")
                continue

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
                        file_duration_minutes=dur_m,
                    )
                except Exception as e:
                    errors += 1
                    print(f"[ERR  ] retry TMDB verify failed: {e}")
                    continue

                if not vres.ok or vres.episode_number is None or not vres.episode_title:
                    print(f"[SKIP ] retry: TMDB reject ({vres.reason})")
                    continue

                tag = "corrected" if vres.corrected else "confirmed"
                print(f"[API  ] retry TMDB {tag} -> S{season:02d}E{vres.episode_number:02d} \"{vres.episode_title}\" (match={vres.match_score:.2f})")
                final_ep_num = vres.episode_number
                final_title = vres.episode_title

            claim_key = (show.lower(), season, final_ep_num)
            if claim_key in episode_claims:
                print(f"[SKIP ] retry: S{season:02d}E{final_ep_num:02d} already claimed by {episode_claims[claim_key].name}")
                continue
            episode_claims[claim_key] = mkv

            ok = rename_and_move(mkv, show, season, final_ep_num, final_title, args.dry_run, dest_root)
            if not ok:
                print("[SKIP ] retry: target exists / rename not performed")
                continue

            print("[DONE ] renamed (retry pass)")
            renamed_episodes.append(f"{sanitize_filename(show)} - S{season:02d}E{final_ep_num:02d} - {sanitize_filename(final_title)}")
            sk = (norm_title(show), season)
            show_season_last_ep[sk] = max(show_season_last_ep.get(sk, 0), final_ep_num)
            retry_renamed_count += 1
            if args.dry_run:
                dryrun += 1
            else:
                renamed += 1

            # Rewind the original skip counter so the SUMMARY reflects the net outcome.
            if fail_code == "low_conf":
                skipped_low_conf = max(0, skipped_low_conf - 1)
            elif fail_code == "missing_fields":
                skipped_missing_fields = max(0, skipped_missing_fields - 1)
            elif fail_code == "verify_failed":
                skipped_verify_failed = max(0, skipped_verify_failed - 1)
            elif fail_code == "non_episode":
                skipped_non_episode = max(0, skipped_non_episode - 1)

            extras_queue[:] = [(p, s, sn) for (p, s, sn) in extras_queue if p != mkv]
            skipped_files[:] = [sf for sf in skipped_files if sf.get("file") != mkv.name]

    hr()
    print("\n=== SUMMARY ===")
    print(f"Total files scanned: {total}")
    print(f"Renamed:            {renamed}")
    print(f"  of which blind:   {renamed_blind}  (no folder hint — identified from content)")
    print(f"  of which retry:   {retry_renamed_count}  (re-IDed after folder gained auto-hints)")
    print(f"Dry-run renames:    {dryrun}")
    print(f"Errors:             {errors}")
    print(f"Evidence used: subtitles={used_subtitles} audio_primary={used_audio_primary} audio_fallback={used_audio_fallback}")
    print(f"Renamed (OpenSubtitles exact):      {renamed_opensubtitles}")
    print(f"Skipped (TMDB verify failed):       {skipped_verify_failed}")
    print("Skipped:")
    print(f"  already named (SxxEyy):           {skipped_already_named}")
    print(f"  no duration (ffprobe):            {skipped_no_duration}")
    print(f"  duration out of range:            {skipped_duration_range}")
    print(f"  no usable evidence text:          {skipped_no_evidence}")
    print(f"  LLM says non-episode:             {skipped_non_episode}")
    print(f"  confidence below threshold:       {skipped_low_conf}")
    print(f"  duplicate episode (conflict):     {skipped_conflict}")
    print(f"  missing episode number/title:     {skipped_missing_fields}")
    print(f"  target exists/other:              {skipped_target_exists}")

    # Move unidentified files to Extras folders in the destination library.
    if extras_queue:
        from collections import defaultdict

        def _next_extra_number(extras_dir: Path) -> int:
            """Return the next available Extra_N index in extras_dir."""
            if not extras_dir.exists():
                return 1
            highest = 0
            for f in extras_dir.iterdir():
                m = re.match(r"Extra_(\d+)", f.stem, re.IGNORECASE)
                if m:
                    highest = max(highest, int(m.group(1)))
            return highest + 1

        by_season: dict = defaultdict(list)
        for mkv_path, show_name, season_num in extras_queue:
            by_season[(show_name, season_num)].append(mkv_path)

        for (show_name, season_num), paths in by_season.items():
            show_s = sanitize_filename(show_name)
            extras_dir = dest_root / show_s / f"Season {season_num:02d}" / "Extras"

            if args.dry_run:
                for p in paths:
                    print(f"[EXTRA] DRYRUN {p.name} -> {extras_dir}/Extra_N{p.suffix}")
                continue

            extras_dir.mkdir(parents=True, exist_ok=True)
            counter = _next_extra_number(extras_dir)
            for p in paths:
                dst = extras_dir / f"Extra_{counter}{p.suffix}"
                while dst.exists():
                    counter += 1
                    dst = extras_dir / f"Extra_{counter}{p.suffix}"
                try:
                    shutil.move(str(p), str(dst))
                    print(f"[EXTRA] {p.name} -> {dst}")
                    extras_moved.append(f"{show_s} - S{season_num:02d} - {dst.name}")
                    counter += 1
                except Exception as e:
                    print(f"[ERR  ] Could not move {p.name} to Extras: {e}")
                    skipped_files.append({"file": p.name, "reason": f"Extras move failed: {e}"})

    # Remove empty directories left behind in --root after episodes were moved out.
    # Walk bottom-up so nested empties are removed before their parents are checked.
    # Skipped in dry-run mode.
    if dest_root and not args.dry_run:
        removed_dirs = 0
        for dirpath, dirnames, filenames in os.walk(args.root, topdown=False):
            d = Path(dirpath)
            if d == Path(args.root):
                continue  # never remove the root itself
            try:
                if not any(d.iterdir()):
                    d.rmdir()
                    print(f"[RMDIR] {d}")
                    removed_dirs += 1
            except OSError:
                pass  # non-empty or permission issue — leave it
        if removed_dirs:
            print(f"Removed {removed_dirs} empty director{'y' if removed_dirs == 1 else 'ies'} from {args.root}")

    if args.summary_json:
        summary = {
            "total": total,
            "renamed": renamed + dryrun,
            "dry_run": args.dry_run,
            "skipped_count": len(skipped_files),
            "renamed_episodes": renamed_episodes,
            "skipped_files": skipped_files,
            "extras_moved": extras_moved,
        }
        try:
            Path(args.summary_json).write_text(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"[WARN ] Could not write summary JSON: {e}")


if __name__ == "__main__":
    import sys
    try:
        main()
    except DeepSeekAuthError as e:
        # Hard API failure (bad key / out of credits). Fail the build loudly so
        # the Jenkins notification fires, rather than silently leaving files
        # unsorted and reporting SUCCESS.
        print(f"[FATAL] DeepSeek auth/billing failure — aborting run (no silent skips): {e}")
        sys.exit(1)
