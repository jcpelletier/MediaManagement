#!/usr/bin/env python3
"""subtitle_ocr.py — OCR text from a video's *bitmap* subtitle track.

Disc rips carry their subtitles as bitmap images (PGS on Blu-ray, VobSub on DVD),
not text, so the sorter can't read them and falls back to Whisper. This module
renders the bitmap subtitle stream onto a black canvas with ffmpeg, samples
frames, and OCRs them with tesseract — one method that covers both PGS and
VobSub, using only ffmpeg + tesseract (no format-specific tooling).

It returns a text sample suitable as identification evidence, or None on any
failure / when there is no usable bitmap subtitle. There is deliberately NO
Whisper fallback in here — the caller decides what to do when this returns None.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

BITMAP_SUB_CODECS = {
    "hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub", "dvb_subtitle",
}

# Sample windows for identification: (fraction-of-duration, seconds). Two short
# windows spread across the film give plenty of dialogue without OCRing the
# whole track.
OCR_WINDOWS = ((0.25, 120.0), (0.60, 120.0))
OCR_FPS = 1                # frames/sec to sample (subs change slowly; deduped)
OCR_CANVAS = "720x480"
_MIN_DURATION = 120.0
_MIN_LINE_LEN = 3

_OCR_FIXUPS = [("|", "I"), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')]


def _run(cmd: List[str], timeout: float) -> str:
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, timeout=timeout,
    ).stdout


def probe_duration(video_path: Path) -> Optional[float]:
    try:
        out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1", str(video_path)], 30).strip()
        return float(out)
    except Exception:
        return None


def list_subtitle_streams(video_path: Path) -> List[dict]:
    """Subtitle streams in order; each {s_index, codec, lang}. s_index is the
    position among subtitle streams (used as the ffmpeg specifier 0:s:<s_index>)."""
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "s",
                    "-show_entries", "stream=codec_name:stream_tags=language",
                    "-of", "json", str(video_path)], 30)
        streams = json.loads(out).get("streams", []) or []
    except Exception:
        return []
    return [
        {"s_index": i,
         "codec": (s.get("codec_name") or "").lower(),
         "lang": ((s.get("tags") or {}).get("language") or "und").lower()}
        for i, s in enumerate(streams)
    ]


def pick_bitmap_stream(streams: List[dict],
                       lang_prefs=("eng", "en", "und")) -> Optional[int]:
    bitmap = [s for s in streams if s["codec"] in BITMAP_SUB_CODECS]
    if not bitmap:
        return None
    for lp in lang_prefs:
        for s in bitmap:
            if s["lang"].startswith(lp):
                return s["s_index"]
    return bitmap[0]["s_index"]


def _clean(line: str) -> str:
    for a, b in _OCR_FIXUPS:
        line = line.replace(a, b)
    return re.sub(r"\s+", " ", line).strip()


def _ocr_window(video_path: Path, s_index: int, start: float,
                seconds: float, lang: str) -> List[str]:
    lines: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        out_pat = str(Path(td) / "f_%04d.png")
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path),
            "-t", f"{seconds:.3f}",
            "-filter_complex",
            f"color=c=black:s={OCR_CANVAS}:d={seconds:.3f}:r={OCR_FPS}[bg];"
            f"[bg][0:s:{s_index}]overlay",
            "-an", out_pat,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=seconds * 4 + 60)
        except Exception:
            return lines
        for png in sorted(Path(td).glob("f_*.png")):
            try:
                txt = _run(["tesseract", str(png), "stdout", "-l", lang], 30)
            except Exception:
                continue
            for raw in txt.splitlines():
                c = _clean(raw)
                if len(c) >= _MIN_LINE_LEN and lines[-1:] != [c]:
                    lines.append(c)
    return lines


def ocr_subtitle_text(video_path: Path, duration: Optional[float] = None,
                      lang: str = "eng", max_lines: int = 120) -> Optional[str]:
    """OCR a sample of the best bitmap subtitle track to deduped dialogue text.
    Returns None on any failure, no bitmap subtitle, or empty result."""
    try:
        s_index = pick_bitmap_stream(list_subtitle_streams(video_path))
        if s_index is None:
            return None
        dur = duration or probe_duration(video_path)
        if not dur or dur < _MIN_DURATION:
            return None
        collected: List[str] = []
        for frac, secs in OCR_WINDOWS:
            start = max(0.0, min(dur * frac, dur - secs - 1))
            window = min(secs, dur - start - 1)
            if window >= 10:
                collected += _ocr_window(video_path, s_index, start, window, lang)
        seen, out = set(), []
        for line in collected:
            key = line.lower()
            if key not in seen:
                seen.add(key)
                out.append(line)
            if len(out) >= max_lines:
                break
        text = "\n".join(out).strip()
        return text or None
    except Exception:
        return None


if __name__ == "__main__":  # quick manual test: python subtitle_ocr.py <video>
    import sys
    t = ocr_subtitle_text(Path(sys.argv[1]))
    print(t if t else "(no bitmap subtitle text)")
