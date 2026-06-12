#!/usr/bin/env python3
"""
Generate_Subs.py

Generates English subtitle sidecar files by transcribing a video's audio with
faster-whisper. This is the *last-resort* tier in the subtitle pipeline:

    Extract_Subs.py   embedded tracks (free, if the MKV has them)
    Fetch_Subs.py     human subs from OpenSubtitles (best quality when matched)
    Generate_Subs.py  Whisper transcription  ← this script (always works, always synced)

Because the transcript comes from the file's own audio, the output is inherently
in sync — no ffsubsync pass (AudioSync_Subs.py) is needed afterward.

Output naming mirrors the rest of the pipeline so Jellyfin auto-loads it:
    MovieName.en.srt

A video is SKIPPED if a sidecar for the target language already exists, so this
is safe to point at the whole library as a fallback sweep and safe to re-run.

CPU note: panda's GPU (GTX 970, Maxwell) cannot run ctranslate2 (no int8/fp16),
so this defaults to CPU int8. On the box's i7-4790K, `base` runs ~1.4x real-time
and `small.en` a few times slower — fine for an overnight batch, but `medium`/
`large` are impractically slow there. Pick the model with that tradeoff in mind.

Requires: ffmpeg/ffprobe on PATH, and `pip install faster-whisper`.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _WhisperModel = None
    _FASTER_WHISPER_AVAILABLE = False


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv"}

DEFAULT_MODEL = "small.en"   # quality/speed sweet spot for English on this CPU
DEFAULT_LANGUAGE = "en"
DEFAULT_BEAM_SIZE = 5


# ─── small utilities ─────────────────────────────────────────────────────────

def run_cmd(cmd: list) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def is_in_extras_folder(path: Path) -> bool:
    return any(part.lower() == "extras" for part in path.parts)


def ffprobe_duration_seconds(video_path: Path) -> Optional[float]:
    rc, out, _ = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ])
    if rc != 0:
        return None
    try:
        return float(out.strip())
    except ValueError:
        return None


def extract_full_audio_wav(video_path: Path, out_wav: Path) -> bool:
    """Decode the whole audio track to 16 kHz mono WAV — what Whisper expects."""
    rc, _, _ = run_cmd([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        str(out_wav),
    ])
    return rc == 0 and out_wav.exists() and out_wav.stat().st_size > 0


# ─── SRT formatting ──────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments, dest_path: Path) -> int:
    """
    Write Whisper segments to an SRT file atomically. Returns the number of
    subtitle blocks written. Writes to a temp file then renames, so a killed
    run never leaves a partial .srt that would block a future re-run.
    """
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            count += 1
            f.write(f"{count}\n")
            f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            f.write(f"{text}\n\n")
    if count == 0:
        tmp_path.unlink(missing_ok=True)
        return 0
    os.replace(tmp_path, dest_path)
    return count


# ─── model loading ───────────────────────────────────────────────────────────

def load_whisper_model(model_name: str, device: str):
    """
    Load a faster-whisper model, degrading gracefully. CUDA is attempted only
    if explicitly requested; on this hardware it will fall through to CPU int8
    (Maxwell can't do int8/fp16 in ctranslate2). Returns the model or None.
    """
    if device == "cuda":
        candidates = [("cuda", "float16"), ("cuda", "int8"), ("cpu", "int8")]
    else:
        candidates = [("cpu", "int8")]

    print(f"Loading Whisper model '{model_name}' (requested device={device})...")
    for dev, compute_type in candidates:
        try:
            model = _WhisperModel(model_name, device=dev, compute_type=compute_type)
            print(f"Whisper model loaded (device={dev}, compute_type={compute_type}).")
            return model
        except Exception as e:
            next_idx = candidates.index((dev, compute_type)) + 1
            if next_idx < len(candidates):
                nd, nc = candidates[next_idx]
                print(f"  {dev}/{compute_type} not available, retrying with {nd}/{nc}...")
            else:
                print(f"ERROR: Failed to load Whisper model: {e}")
                return None
    return None


# ─── per-file transcription ──────────────────────────────────────────────────

def generate_subs_for_video(
    model,
    video_path: Path,
    language: str,
    beam_size: int,
    use_vad: bool,
    translate: bool,
    dry_run: bool,
) -> str:
    """
    Generate a subtitle sidecar for one video.
    Returns one of: "generated", "skipped", "empty", "error".
    """
    if is_in_extras_folder(video_path):
        print(f"  SKIP (Extras folder)")
        return "skipped"

    subs_path = video_path.with_suffix(f".{language}.srt")
    if subs_path.exists():
        print(f"  SKIP — {subs_path.name} already exists.")
        return "skipped"

    if dry_run:
        print(f"  would generate {subs_path.name}")
        return "generated"

    duration = ffprobe_duration_seconds(video_path)

    with tempfile.TemporaryDirectory(prefix="gensubs_") as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        print(f"  Extracting audio...")
        if not extract_full_audio_wav(video_path, wav_path):
            print(f"  ERROR: audio extraction failed.")
            return "error"

        print(f"  Transcribing (model beam={beam_size}, vad={'on' if use_vad else 'off'}"
              f"{', translate' if translate else ''})...")
        t0 = time.time()
        try:
            segments, _info = model.transcribe(
                str(wav_path),
                language=None if translate else language,
                task="translate" if translate else "transcribe",
                beam_size=beam_size,
                vad_filter=use_vad,
            )
            # segments is a generator; write_srt consumes it (transcription
            # actually runs here, lazily, as blocks are pulled).
            written = write_srt(segments, subs_path)
        except Exception as e:
            print(f"  ERROR during transcription: {e}")
            subs_path.with_suffix(subs_path.suffix + ".part").unlink(missing_ok=True)
            return "error"

    elapsed = time.time() - t0
    if written == 0:
        print(f"  EMPTY — no speech transcribed; no file written.")
        return "empty"

    rtf = (duration / elapsed) if duration and elapsed > 0 else None
    speed = f" ({rtf:.1f}x real-time)" if rtf else ""
    print(f"  WROTE {subs_path.name} — {written} cues in {elapsed:.0f}s{speed}")
    return "generated"


# ─── target walking ──────────────────────────────────────────────────────────

def collect_videos(target: Path) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in VIDEO_EXTS else []
    return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def main():
    parser = argparse.ArgumentParser(
        description="Generate subtitle sidecars by transcribing audio with faster-whisper (CPU by default).",
    )
    parser.add_argument("target", type=Path, help="Video file or folder to scan recursively.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"faster-whisper model (default: {DEFAULT_MODEL}). "
                             f"e.g. tiny.en, base.en, small.en, medium.en, large-v3.")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help=f"Language code; sets the sidecar suffix .<lang>.srt (default: {DEFAULT_LANGUAGE}).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu). cuda falls back to cpu on unsupported GPUs.")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE,
                        help=f"Decoding beam size; lower is faster, higher is more accurate (default: {DEFAULT_BEAM_SIZE}).")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable voice-activity filtering (VAD reduces hallucinated text over music/silence).")
    parser.add_argument("--translate", action="store_true",
                        help="Translate speech to English instead of transcribing in the original language.")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Max number of files to transcribe this run (0 = unlimited). "
                             "Files skipped because a sidecar already exists do NOT count "
                             "toward the limit, so a nightly run keeps chipping away at the backlog.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be generated without transcribing.")
    args = parser.parse_args()

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
        print("Dry-run mode — nothing will be transcribed.\n")

    model = None
    if not args.dry_run:
        if not _FASTER_WHISPER_AVAILABLE:
            print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
            sys.exit(3)
        model = load_whisper_model(args.model, args.device)
        if model is None:
            sys.exit(3)

    counts = {"generated": 0, "skipped": 0, "empty": 0, "error": 0}
    processed = 0  # files we actually ran transcription on (skips don't count)
    for i, video in enumerate(videos, 1):
        try:
            rel = video.relative_to(target) if target.is_dir() else video.name
        except ValueError:
            rel = video.name
        print(f"\n[{i}/{len(videos)}] {rel}")
        result = generate_subs_for_video(
            model, video,
            language=args.language,
            beam_size=args.beam_size,
            use_vad=not args.no_vad,
            translate=args.translate,
            dry_run=args.dry_run,
        )
        counts[result] += 1

        if result != "skipped":
            processed += 1
            if args.limit and processed >= args.limit:
                print(f"\nReached --limit of {args.limit} file(s) processed this run — stopping "
                      f"({len(videos) - i} not yet examined).")
                break

    print(f"\nDone. {counts['generated']} generated, {counts['skipped']} skipped, "
          f"{counts['empty']} empty, {counts['error']} errors.")


if __name__ == "__main__":
    main()
