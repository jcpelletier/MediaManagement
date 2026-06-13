#!/usr/bin/env python3
"""
Fix_Sub_Timing.py

Repairs "gap-held" subtitle cues — a cue whose on-screen duration is wildly
longer than its text could possibly take to read, because Whisper dragged the
cue's START back across a non-speech gap (e.g. the line right after a show's
title sequence gets anchored to where the music began and is held for the whole
~2-minute intro).

These cues have a *correct end* (it abuts the real next line) but a wrong start.
The fix: snap the start forward so the cue is shown for a normal reading time,
ending where it already ends (i.e. when the line is actually spoken).

Detection is conservative — a cue is only rewritten when its duration is many
times its reading time AND over an absolute floor — so genuine long cues full of
dialogue are left untouched.

Idempotent and safe to re-run. Default is a dry run; pass --apply to write.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

CHARS_PER_SEC = 15.0      # subtitle reading speed used to estimate a sane duration
MIN_READ_SEC = 1.5        # floor for very short lines
MIN_DUR_SEC = 12.0        # never touch a cue at/under this duration
ALWAYS_CAP_SEC = 50.0     # any cue longer than this is an artifact (no real cue runs this long)
RATIO = 8.0               # ...or short text held *vastly* longer than its reading time

TS_RE = re.compile(r'(\d\d):(\d\d):(\d\d),(\d+)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d+)')


def parse_ts(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def fmt_ts(x: float) -> str:
    if x < 0:
        x = 0.0
    ms = int(round(x * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def reading_time(text: str) -> float:
    return max(MIN_READ_SEC, len(text) / CHARS_PER_SEC)


def parse_srt(raw: str) -> List[dict]:
    cues = []
    for block in raw.split("\n\n"):
        lines = [l for l in block.splitlines() if l.strip() != ""]
        if not lines:
            continue
        ts_idx = next((i for i, l in enumerate(lines) if TS_RE.search(l)), None)
        if ts_idx is None:
            continue
        m = TS_RE.search(lines[ts_idx])
        start = parse_ts(*m.group(1, 2, 3, 4))
        end = parse_ts(*m.group(5, 6, 7, 8))
        text_lines = lines[ts_idx + 1:]
        cues.append({"start": start, "end": end, "text_lines": text_lines})
    return cues


def fix_cues(cues: List[dict]) -> List[Tuple[int, float, float, float, str]]:
    """Rewrite gap-held cues in place. Returns list of (idx, old_start, new_start, end, text)."""
    changes = []
    for i, c in enumerate(cues):
        dur = c["end"] - c["start"]
        text = " ".join(c["text_lines"]).strip()
        if dur <= MIN_DUR_SEC:
            continue
        rt = reading_time(text)
        # Only a gap-held artifact: either absurdly long for any cue, or short
        # text held many times its reading time. Real multi-turn cues (lots of
        # text, duration ~1-3x reading) are left untouched.
        if not (dur > ALWAYS_CAP_SEC or dur > rt * RATIO):
            continue
        new_start = c["end"] - rt
        prev_end = cues[i - 1]["end"] if i > 0 else 0.0
        if new_start < prev_end + 0.1:
            new_start = prev_end + 0.1
        if new_start <= c["start"] + 0.05:
            continue  # nothing meaningful to shorten
        changes.append((i, c["start"], new_start, c["end"], text))
        c["start"] = new_start
    return changes


def render_srt(cues: List[dict]) -> str:
    out = []
    n = 0
    for c in cues:
        text = "\n".join(c["text_lines"]).strip()
        if not text:
            continue
        n += 1
        out.append(f"{n}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{text}\n")
    return "\n".join(out) + "\n"


def process_file(path: Path, apply: bool) -> int:
    raw = path.read_text(encoding="utf-8")
    cues = parse_srt(raw)
    changes = fix_cues(cues)
    if not changes:
        return 0
    print(f"\n{path.name}: {len(changes)} cue(s) capped")
    for idx, old_s, new_s, end, text in changes:
        print(f"  [{fmt_ts(old_s)} -> {fmt_ts(new_s)}] end {fmt_ts(end)} "
              f"(was {end-old_s:.0f}s, now {end-new_s:.1f}s)  {text[:60]}")
    if apply:
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(render_srt(cues), encoding="utf-8")
        os.replace(tmp, path)
    return len(changes)


def collect_srts(target: Path) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() == ".srt" else []
    return sorted(target.rglob("*.srt"))


def main():
    ap = argparse.ArgumentParser(description="Cap 'gap-held' subtitle cues to a sane reading duration.")
    ap.add_argument("target", type=Path, help="SRT file or folder to scan recursively.")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    target = args.target.expanduser().resolve()
    if not target.exists():
        print(f"ERROR: '{target}' does not exist.")
        sys.exit(2)

    srts = collect_srts(target)
    if not srts:
        print("No .srt files found.")
        return

    mode = "APPLY" if args.apply else "DRY-RUN (no files written; pass --apply to write)"
    print(f"{mode} — scanning {len(srts)} subtitle file(s).")
    total = sum(process_file(p, args.apply) for p in srts)
    print(f"\nDone. {total} cue(s) {'rewritten' if args.apply else 'would be rewritten'}.")


if __name__ == "__main__":
    main()
