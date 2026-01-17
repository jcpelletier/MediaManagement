#!/usr/bin/env python3
"""

Path 2 (LLM-assisted) + API verification:
- LLM proposes episode_number/title from subtitles or audio transcript
- Then VERIFY using TMDB API:
    - search TV series
    - fetch season episode list
    - confirm or correct episode number/title
- Rename in-place (default) to: "{Show} - SxxEyy - {Title}.mkv"
- Optional: if --parent-path is provided, move the (renamed) file to:
    {parent_path}/Season {season}/

Rules:
- If file is not renamed (low confidence / non-episode / verify failed / etc.), it is NOT moved.
- No overwrites: if target exists, skip.
"""

import argparse
import json
import os
import re
import shutil
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
AUDIO_START_SECONDS_HARDCODED = 600.0
DEFAULT_FALLBACK_AUDIO_SECONDS = 300.0
FALLBACK_START_GAP_SECONDS = 120.0
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
    # ASCII-only separator (safe for Jenkins / Windows consoles)
    print("-" * 60)


def short_path(p: Path, max_len: int = 90) -> str:
    s = str(p)
    return s if len(s) <= max_len else ("..." + s[-(max_len - 3):])


def norm_title(s: str) -> str:
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

def parse_show_and_season_from_folder(folder_name: str) -> Tuple[Optional[str], Optional[int]]:
    name = (folder_name or "").strip()
    name = re.sub(r"[_]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Accept: "Season 1", "Season1", "S1", "S01", "Series 1"
    season = None
    m = re.search(r"\b(?:season|series)\s*(\d{1,2})\b", name, flags=re.IGNORECASE)
    if m:
        season = int(m.group(1))
        show_part = name[:m.start()].strip()
        tail_part = name[m.end():].strip()
        show = show_part
        # If show_part is empty (rare), fall back to stripping tail markers
        if not show:
            show = re.sub(r"\b(?:disc|disk|d)\s*\d{1,2}\b", "", name, flags=re.IGNORECASE).strip()
    else:
        m = re.search(r"\bS\s*(\d{1,2})\b", name, flags=re.IGNORECASE)
        if m:
            season = int(m.group(1))
            show = name[:m.start()].strip()
        else:
            show = name

    show = show.strip(" -_.")
    if show:
        # normalize separators like "Star Trek- The Next Generation"
        show = show.replace("-", " ")
        show = re.sub(r"\s+", " ", show).strip()

    # If show is something like "STAR TREK TNG", you might want a custom mapping,
    # but at least we won't fail parsing season anymore.
    return (show if show else None), season



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
        return VerificationResult(False, False, None, None, 0.0, "tmdb: show not found")

    episodes = tmdb.get_season_episodes(tv_id, season)
    if not episodes:
        return VerificationResult(False, False, None, None, 0.0, "tmdb: season not found or has no episodes")

    ep_by_num = {e.episode_number: e for e in episodes}
    if proposed_ep_num in ep_by_num:
        canon = ep_by_num[proposed_ep_num]
        score = similarity(proposed_title, canon.name)
        if score >= min_title_match:
            return VerificationResult(
                ok=True,
                corrected=(norm_title(proposed_title) != norm_title(canon.name)),
                episode_number=canon.episode_number,
                episode_title=canon.name,
                match_score=score,
                reason="tmdb: confirmed by episode number + title match"
            )

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

    return VerificationResult(
        ok=False,
        corrected=False,
        episode_number=None,
        episode_title=None,
        match_score=best_score,
        reason=f"tmdb: could not confirm title (best_score={best_score:.2f} < {min_title_match:.2f})"
    )


# ----------------------------
# Rename (+ optional move)
# ----------------------------

def compute_target_dir(src: Path, season: int, parent_path: Optional[Path]) -> Path:
    if not parent_path:
        return src.parent
    return parent_path / f"Season {season}"


def rename_and_maybe_move(
    src: Path,
    show: str,
    season: int,
    ep: int,
    title: str,
    dry_run: bool,
    parent_path: Optional[Path],
) -> bool:
    show_s = sanitize_filename(show)
    title_s = sanitize_filename(title)
    new_name = f"{show_s} - S{season:02d}E{ep:02d} - {title_s}{src.suffix}"

    target_dir = compute_target_dir(src, season, parent_path)
    dst = target_dir / new_name

    if dst.exists():
        print(f"[SKIP ] target exists: {src.name} -> {dst.name}")
        return False

    if dry_run:
        print(f"[RENAME] DRYRUN {src.name} -> {dst.name}")
        if parent_path:
            print(f"[MOVE  ] DRYRUN -> {short_path(target_dir, 110)}")
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"[RENAME] {src.name} -> {dst.name}")
    if parent_path:
        print(f"[MOVE  ] -> {short_path(target_dir, 110)}")
    return True


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--min-minutes", type=float, default=20.0)
    ap.add_argument("--max-minutes", type=float, default=55.0)
    ap.add_argument("--min-confidence", type=float, default=0.85)
    ap.add_argument("--max-sub-lines", type=int, default=80)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parent-path", default=None)
    ap.add_argument("--no-verify-api", action="store_true")
    ap.add_argument("--tmdb-api-key", default=None)
    ap.add_argument("--tmdb-min-title-match", type=float, default=0.78)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    client = OpenAI()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    print(f"Scanning for .mkv under: {root}")
    mkvs = list(root.rglob("*.mkv"))
    print(f"Found {len(mkvs)} .mkv file(s).")

    for mkv in mkvs:
        hr()
        print(f"FILE : {short_path(mkv, 110)}")

        folder = mkv.parent.name
        show, season = parse_show_and_season_from_folder(folder)
        if not show or not season:
            print(f"[SKIP ] can't parse show/season from folder: \"{folder}\"")
            continue

        dur_s = ffprobe_duration_seconds(mkv)
        if dur_s is None:
            print("[SKIP ] ffprobe couldn't read duration")
            continue

        print(f"HINT : {show} | S{season:02d} | {dur_s/60.0:.1f}m")


if __name__ == "__main__":
    main()
