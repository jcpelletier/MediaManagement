#!/usr/bin/env python3
"""
Cache tier-2 identification evidence into a ground-truth fixture, so the
identification test runs fast and self-contained (no media, no Whisper).

Run this on panda, where the media lives. Two flavors, picked by what the title
has (see fixtures/sort_groundtruth/README.md):

  subtitle_raw : raw .srt text (title has text subtitles). Default.
  transcript   : Whisper transcript (bitmap-only / no subtitles). Pass --whisper.

Examples:
  python3 capture_evidence.py --fixture fixtures/sort_groundtruth/aladdin_1992.json \
      --media "/mnt/media/Media/Movies/Aladdin (1992)/Aladdin (1992).mp4" --src "<src in fixture>"

  python3 capture_evidence.py --fixture fixtures/sort_groundtruth/paw_patrol_pups_save_puplantis_9b2b.json \
      --media "/mnt/media/.../B2_t01.mkv" --src "B2_t01.mkv" --whisper
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import Sort_TV  # provides subtitle extraction + Whisper helpers; run from repo root


def capture_subtitle_raw(media: Path):
    text = Sort_TV._srt_text_from_mkv(media)
    if not text or not text.strip():
        return None
    return {"kind": "subtitle_raw", "text": text}


def capture_transcript(media: Path, model_size: str, device: str,
                       audio_start: float, audio_seconds: float):
    if not Sort_TV._FASTER_WHISPER_AVAILABLE:
        sys.exit("faster-whisper is not installed; cannot capture a transcript.")
    # Match production's CPU/GPU fallback order.
    candidates = [("cuda", "float16"), ("cuda", "int8"), ("cpu", "int8")] if device == "cuda" else [("cpu", "int8")]
    model = None
    for dev, compute in candidates:
        try:
            model = Sort_TV._FasterWhisperModel(model_size, device=dev, compute_type=compute)
            break
        except Exception:
            continue
    if model is None:
        sys.exit("could not load a Whisper model on any device.")

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "clip.wav"
        ok = Sort_TV.extract_audio_clip_wav(media, wav, audio_start, audio_seconds, verbose=True)
        if not ok:
            return None
        text = Sort_TV.transcribe_with_faster_whisper(model, wav)
    if not text or not text.strip():
        return None
    return {"kind": "transcript", "text": text}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--media", type=Path, required=True, help="Path to the media file to extract evidence from")
    ap.add_argument("--src", required=True, help="The title's `src` value in the fixture to attach evidence to")
    ap.add_argument("--whisper", action="store_true", help="Transcribe audio instead of using subtitles")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--whisper-device", default="cpu")
    ap.add_argument("--audio-start", type=float, default=60.0, help="Audio clip start seconds (default: 60)")
    ap.add_argument("--audio-seconds", type=float, default=Sort_TV.PRIMARY_AUDIO_SECONDS,
                    help=f"Audio clip length seconds (default: {Sort_TV.PRIMARY_AUDIO_SECONDS:.0f})")
    ap.add_argument("--force", action="store_true", help="Overwrite existing evidence on the title")
    args = ap.parse_args()

    if not args.fixture.is_file():
        ap.error(f"fixture not found: {args.fixture}")
    if not args.media.is_file():
        ap.error(f"media not found: {args.media}")

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    titles = fixture.get("titles", [])
    match = next((t for t in titles if t.get("src") == args.src), None)
    if match is None:
        ap.error(f"no title with src={args.src!r} in {args.fixture.name}. "
                 f"Available: {[t.get('src') for t in titles]}")
    if match.get("evidence") and not args.force:
        ap.error(f"title {args.src!r} already has evidence (use --force to overwrite)")

    if args.whisper:
        evidence = capture_transcript(args.media, args.whisper_model, args.whisper_device,
                                      args.audio_start, args.audio_seconds)
    else:
        evidence = capture_subtitle_raw(args.media)
        if evidence is None:
            sys.exit("no usable text subtitles found; re-run with --whisper to transcribe audio.")

    if evidence is None:
        sys.exit("evidence capture produced no text.")

    match["evidence"] = evidence
    args.fixture.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"attached {evidence['kind']} evidence ({len(evidence['text'])} chars) to {args.src} in {args.fixture.name}")


if __name__ == "__main__":
    main()
