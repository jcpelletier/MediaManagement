#!/usr/bin/env python3
"""Shared Whisper transcript cache for Sort_Rips.py and Sort_TV.py.

Whisper transcription is the slow, repeated cost of identification. A clip's
transcript is cached keyed by the file fingerprint (size + duration) AND the
transcription parameters (model + clip offset/length), so a cached transcript is
only ever reused under identical conditions — reuse is then functionally
identical to re-transcribing, just faster and deterministic. Changing the model
or clip window misses and re-transcribes.

The key is content-based (not path/filename based), so:
  - the accuracy harness, which stages hardlinks with the same size + duration as
    the real library file, shares entries with production, and
  - movies and TV episodes never collide (different files -> different keys),
so both sorters can safely share one cache directory.
"""

import json
import re
from pathlib import Path
from typing import Optional


def transcript_cache_path(
    cache_dir: Path, size_bytes: int, duration_s: Optional[float],
    model: str, start: float, seconds: float,
) -> Path:
    key = f"{size_bytes}_{int(round(duration_s or 0))}_{model}_{int(round(start))}_{int(round(seconds))}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
    return cache_dir / f"{safe}.json"


def load_cached_transcript(
    cache_dir: Optional[Path], size_bytes: int, duration_s: Optional[float],
    model: str, start: float, seconds: float,
) -> Optional[str]:
    if not cache_dir:
        return None
    p = transcript_cache_path(cache_dir, size_bytes, duration_s, model, start, seconds)
    if not p.is_file():
        return None
    try:
        text = json.loads(p.read_text(encoding="utf-8")).get("transcript")
        return text or None
    except Exception:
        return None


def save_cached_transcript(
    cache_dir: Optional[Path], size_bytes: int, duration_s: Optional[float],
    model: str, start: float, seconds: float, transcript: str,
) -> None:
    if not cache_dir or not transcript:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = transcript_cache_path(cache_dir, size_bytes, duration_s, model, start, seconds)
        p.write_text(json.dumps({
            "transcript": transcript, "size_bytes": size_bytes,
            "duration_s": duration_s, "whisper_model": model,
            "clip_start": start, "clip_seconds": seconds,
        }, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  [CACHE] could not write transcript cache: {e}")
