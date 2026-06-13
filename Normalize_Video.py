#!/usr/bin/env python3
"""
Normalize_Video.py

Detects videos whose frame timeline is *damaged* — variable-frame-rate or with
dropped frames — and repairs them to a clean constant-frame-rate (CFR) stream.

Why this exists
---------------
A bad rip/encode can leave a file whose frames carry irregular timestamps: the
nominal rate (container `r_frame_rate`) says 29.97, but the frames actually
average well below that because thousands are missing or unevenly spaced.

Desktop players (VLC/mpv) render such a file fine — they hold each frame to its
own timestamp, so the picture tracks the audio. But Jellyfin streams to browser
clients via HLS/MSE, and the browser's media pipeline does NOT tolerate that
irregular cadence: the *picture* drifts away from the audio during playback.
Because external subtitles are timed to the audio, they end up looking 2-3s off
even though the .srt is correct. (Observed on Star Trek Voyager S01E05.)

The only durable fix is to give the file a sane, constant-cadence video stream.

What it does
------------
For each flagged file it re-encodes the VIDEO to CFR at the file's own nominal
rate (duplicating frames to fill gaps so the picture stays aligned to the audio)
and **stream-copies the audio untouched**. Keeping the audio bit-for-bit means
the audio timeline is unchanged, so any existing `Name.en.srt` sidecar stays
valid — no subtitle regeneration needed. The leftover data/timecode track is
dropped.

Healthy files (nominal ≈ actual frame rate) are skipped, so this is safe to
point at the whole library as a sweep and safe to re-run: a repaired file is now
CFR and won't be flagged again.

Video: H.264 (NVENC, CQ 19, preset p4) by default; --encoder libx264 for CPU.
Audio: copied as-is.   MP4: +faststart.

Requires: ffmpeg/ffprobe on PATH (NVENC build for the default encoder).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv"}

# A file is "damaged" if this fraction or more of the frames the nominal rate
# implies are actually missing (actual avg rate has fallen that far below
# nominal). 0.03 → flag when >3% of frames are gone. E05 was ~17%.
DEFAULT_MISSING_THRESHOLD = 0.03


def run_cmd(cmd: list) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def is_in_extras_folder(path: Path) -> bool:
    return any(part.lower() == "extras" for part in path.parts)


def _parse_rate(rate: str) -> Optional[float]:
    """Turn an ffprobe rate like '30000/1001' into a float; None if unusable."""
    if not rate or rate in ("0/0", "N/A"):
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/")
            den = float(den)
            return float(num) / den if den else None
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return None


def probe_timeline(video_path: Path) -> Optional[dict]:
    """
    Return timing facts for the first video stream, or None if it can't be read.
    Keys: r_rate, avg_rate, nb_frames, duration, r_rate_str.
    """
    rc, out, _ = run_cmd([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "default=nw=1",
        str(video_path),
    ])
    if rc != 0:
        return None
    fields = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    r_str = fields.get("r_frame_rate", "")
    r_rate = _parse_rate(r_str)
    avg_rate = _parse_rate(fields.get("avg_frame_rate", ""))
    duration = None
    for key in ("duration",):
        try:
            duration = float(fields.get(key))
            break
        except (TypeError, ValueError):
            continue
    nb_frames = None
    try:
        nb_frames = int(fields.get("nb_frames"))
    except (TypeError, ValueError):
        nb_frames = None
    return {
        "r_rate": r_rate,
        "avg_rate": avg_rate,
        "nb_frames": nb_frames,
        "duration": duration,
        "r_rate_str": r_str,
    }


def assess(info: dict, threshold: float) -> Tuple[bool, float, str]:
    """
    Decide whether a file's timeline is damaged.
    Returns (is_damaged, missing_fraction, human_reason).

    missing_fraction = 1 - actual_frames / expected_frames, where expected is
    duration * nominal_rate and actual is nb_frames (or duration * avg_rate).
    """
    r_rate = info.get("r_rate")
    avg_rate = info.get("avg_rate")
    duration = info.get("duration")
    nb_frames = info.get("nb_frames")

    if not r_rate or not duration or duration <= 0:
        return False, 0.0, "insufficient probe data — skipped"

    expected = r_rate * duration
    if nb_frames and nb_frames > 0:
        actual = float(nb_frames)
        basis = "nb_frames"
    elif avg_rate:
        actual = avg_rate * duration
        basis = "avg_frame_rate"
    else:
        return False, 0.0, "no actual frame count — skipped"

    if expected <= 0:
        return False, 0.0, "bad nominal rate — skipped"

    missing = 1.0 - (actual / expected)
    # Negative (actual > expected) just means it's fine; clamp for reporting.
    if missing < threshold:
        return False, missing, (f"ok ({basis}: {actual:.0f}/{expected:.0f} frames, "
                                f"{missing*100:.1f}% missing)")
    return True, missing, (f"DAMAGED ({basis}: {actual:.0f}/{expected:.0f} frames, "
                           f"{missing*100:.1f}% missing; nominal {r_rate:.3f}fps "
                           f"vs actual {actual/duration:.3f}fps)")


def normalize_file(video_path: Path, target_rate: str, encoder: str,
                   keep_backup: bool) -> str:
    """
    Re-encode video to CFR at target_rate, copy audio, replace original in place.
    Returns "repaired", "error".
    """
    tmp_path = video_path.with_suffix(video_path.suffix + ".normtmp.mp4")
    if encoder == "nvenc":
        venc = ["-c:v", "h264_nvenc", "-rc:v", "vbr", "-cq", "19",
                "-preset", "p4", "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-crf", "19", "-preset", "medium"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-map", "0:v:0", "-map", "0:a:0",   # video + audio only; drop data/timecode
        "-vsync", "cfr", "-r", target_rate,  # duplicate/drop to a constant cadence
        *venc,
        "-c:a", "copy",                       # audio untouched → subtitles stay valid
        "-movflags", "+faststart",
        str(tmp_path),
    ]
    print(f"  Normalizing video to CFR {target_rate} ({encoder}), copying audio...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        print("  ffmpeg output:\n" + (proc.stdout or ""))
        print(f"  ERROR: encode failed (exit {proc.returncode}).")
        tmp_path.unlink(missing_ok=True)
        return "error"

    # Sanity-check the result before swapping it in: duration must be preserved
    # (CFR fill shouldn't change wall-clock length by more than a rounding hair).
    new = probe_timeline(tmp_path)
    old_dur = probe_timeline(video_path).get("duration") if video_path.exists() else None
    new_dur = new.get("duration") if new else None
    if old_dur and new_dur and abs(new_dur - old_dur) > 1.0:
        print(f"  ERROR: duration drifted {old_dur:.2f}s -> {new_dur:.2f}s (>1s); "
              f"refusing to replace. Left temp at {tmp_path.name}.")
        return "error"

    if keep_backup:
        backup = video_path.with_suffix(video_path.suffix + ".orig")
        try:
            shutil.move(str(video_path), str(backup))
            print(f"  Backed up original -> {backup.name}")
        except Exception as e:
            print(f"  ERROR: could not back up original: {e}")
            tmp_path.unlink(missing_ok=True)
            return "error"

    try:
        os.replace(tmp_path, video_path)   # atomic same-name swap
    except Exception as e:
        print(f"  ERROR: could not replace original: {e}")
        return "error"
    print(f"  REPAIRED {video_path.name}"
          + (f" — {new['nb_frames']} frames CFR" if new and new.get("nb_frames") else ""))
    return "repaired"


def collect_videos(target: Path) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in VIDEO_EXTS else []
    return sorted(p for p in target.rglob("*")
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def main():
    parser = argparse.ArgumentParser(
        description="Detect frame-damaged/VFR videos and repair them to clean CFR "
                    "(video re-encode, audio copied so subtitles stay valid).",
    )
    parser.add_argument("target", type=Path, help="Video file or folder to scan recursively.")
    parser.add_argument("--encoder", default="nvenc", choices=["nvenc", "libx264"],
                        help="Video encoder for the repair (default: nvenc).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MISSING_THRESHOLD,
                        help=f"Flag a file when this fraction of frames are missing "
                             f"(default: {DEFAULT_MISSING_THRESHOLD} = 3%%).")
    parser.add_argument("--keep-backup", action="store_true",
                        help="Keep the original alongside the repair as <name>.<ext>.orig.")
    parser.add_argument("--force", action="store_true",
                        help="Normalize every file regardless of the damage check.")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Max files to repair this run (0 = unlimited). Skipped/healthy "
                             "files do not count, so a nightly run chips away at the backlog.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report damaged files without re-encoding.")
    args = parser.parse_args()

    # Jenkins pipes stdout; line-buffer so per-file progress shows live.
    sys.stdout.reconfigure(line_buffering=True)

    if args.limit < 0:
        print("ERROR: --limit must be >= 0.")
        sys.exit(2)

    target = args.target.expanduser().resolve()
    if not target.exists():
        print(f"ERROR: '{target}' does not exist.")
        sys.exit(2)

    videos = collect_videos(target)
    if not videos:
        print("No video files found.")
        return

    print(f"Found {len(videos)} video file(s) under {target}.")
    if args.dry_run:
        print("Dry-run mode — nothing will be re-encoded.\n")

    counts = {"repaired": 0, "skipped": 0, "flagged": 0, "error": 0}
    processed = 0
    for i, video in enumerate(videos, 1):
        try:
            rel = video.relative_to(target) if target.is_dir() else video.name
        except ValueError:
            rel = video.name
        print(f"\n[{i}/{len(videos)}] {rel}")

        if is_in_extras_folder(video):
            print("  SKIP (Extras folder)")
            counts["skipped"] += 1
            continue

        info = probe_timeline(video)
        if info is None:
            print("  SKIP — could not probe.")
            counts["skipped"] += 1
            continue

        damaged, missing, reason = assess(info, args.threshold)
        print(f"  {reason}")

        if not damaged and not args.force:
            counts["skipped"] += 1
            continue

        counts["flagged"] += 1
        target_rate = info.get("r_rate_str") or "30000/1001"

        if args.dry_run:
            print(f"  would normalize to CFR {target_rate}")
            processed += 1
            if args.limit and processed >= args.limit:
                print(f"\nReached --limit of {args.limit} — stopping.")
                break
            continue

        result = normalize_file(video, target_rate, args.encoder, args.keep_backup)
        counts[result] = counts.get(result, 0) + 1

        processed += 1
        if args.limit and processed >= args.limit:
            print(f"\nReached --limit of {args.limit} file(s) this run — stopping "
                  f"({len(videos) - i} not yet examined).")
            break

    print(f"\nDone. {counts['repaired']} repaired, {counts['flagged']} flagged, "
          f"{counts['skipped']} skipped, {counts['error']} errors.")
    if counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
