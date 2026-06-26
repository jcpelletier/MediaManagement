"""Tier 2: end-to-end episode identification accuracy, from cached evidence.

This drives the REAL Sort_TV.main() over the full pipeline (identification +
voting + TMDB verify + per-folder reconciliation + extras/skip routing), so the
number reflects what would actually land in the library, not just what the
identification call proposed.

It needs no media and no Whisper: each verified fixture's titles carry cached
evidence (see capture_evidence.py), and this harness stubs only the media-touching
leaf functions to serve that evidence. The stubs assert they were called, so if
production stops routing through them the test fails loudly instead of silently
testing nothing. Real DeepSeek + TMDB still run (that is the thing under test).

Slow + non-deterministic + costs API calls, so it is skipped unless run with:

    pytest --run-identification        # direct, no Jenkins needed

It needs DEEPSEEK_API_KEY and TMDB_API_KEY in the environment. Accuracy is
reported overall and per evidence kind (subtitle_raw vs transcript). Set a
regression gate with --id-accuracy-floor once the corpus is large enough; by
default nothing is asserted to 100% because the LLM is non-deterministic.
"""

import json
import os
from pathlib import Path

import pytest

import Sort_TV
from fixtures_lib import verified_fixtures


def _evidenced_tv_fixtures():
    """Fixtures with at least one verified TV title that has cached evidence,
    an episode label, and a usable .mkv src (Sort_TV only processes .mkv)."""
    out = []
    for f in verified_fixtures():
        scorable = [
            t for t in f["titles"]
            if t.get("evidence") and t["evidence"].get("text")
            and t.get("expected") and t["expected"].get("type") == "tv"
            and isinstance(t["expected"].get("episode"), int)
            and (t.get("src") or "").lower().endswith(".mkv")
        ]
        if scorable:
            out.append((f, scorable))
    return out


def _text_sub_streams():
    return [{"index": 0, "codec_name": "subrip", "codec_type": "subtitle", "tags": {"language": "eng"}}]


def _bitmap_sub_streams():
    return [{"index": 0, "codec_name": "hdmv_pgs_subtitle", "codec_type": "subtitle", "tags": {"language": "eng"}}]


def _run_disc(fixture, scorable_titles, tmp_path, monkeypatch):
    """Stub the media leaves to serve this disc's cached evidence, then run the
    real Sort_TV.main() over placeholder files. Returns the summary 'results'."""
    disc_dir = tmp_path / "src" / "disc"
    disc_dir.mkdir(parents=True)
    dest_dir = tmp_path / "dest"

    # Placeholder .mkv per title (0-byte; main() only stats name/size/mtime).
    by_name = {}
    for t in fixture["titles"]:
        src = t.get("src") or ""
        if not src.lower().endswith(".mkv"):
            continue
        (disc_dir / src).write_bytes(b"")
        by_name[src] = t

    def _title_for(path):
        t = by_name.get(Path(path).name)
        assert t is not None, f"unexpected file passed to a stub: {path}"
        return t

    def fake_duration(path):
        return float(_title_for(path).get("duration_s") or 0)

    def fake_streams(path):
        ev = _title_for(path).get("evidence")
        if ev and ev["kind"] == "subtitle_raw":
            return _text_sub_streams()
        if ev and ev["kind"] == "transcript":
            return _bitmap_sub_streams()  # forces production down the audio path
        return []

    def fake_srt(path):
        ev = _title_for(path).get("evidence")
        return ev["text"] if ev and ev["kind"] == "subtitle_raw" else None

    def fake_extract_audio(path, out_wav, start, seconds, verbose=False):
        ev = _title_for(path).get("evidence")
        if not ev or ev["kind"] != "transcript":
            return False
        # Courier the transcript to the transcriber via the temp wav file.
        Path(out_wav).write_text(ev["text"], encoding="utf-8")
        return True

    def fake_transcribe(model, wav_path):
        return Path(wav_path).read_text(encoding="utf-8")

    monkeypatch.setattr(Sort_TV, "ffprobe_duration_seconds", fake_duration)
    monkeypatch.setattr(Sort_TV, "ffprobe_subtitle_streams", fake_streams)
    monkeypatch.setattr(Sort_TV, "_srt_text_from_mkv", fake_srt)
    monkeypatch.setattr(Sort_TV, "extract_audio_clip_wav", fake_extract_audio)
    monkeypatch.setattr(Sort_TV, "transcribe_with_faster_whisper", fake_transcribe)
    # Make the Whisper model "load" without faster-whisper installed.
    monkeypatch.setattr(Sort_TV, "_FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(Sort_TV, "_FasterWhisperModel", lambda *a, **k: object())

    summary_path = tmp_path / "summary.json"
    Sort_TV.main([
        "--root", str(disc_dir.parent),
        "--dest", str(dest_dir),
        "--dry-run",
        "--quiet",
        "--max-minutes", "100",
        "--audio-start-seconds", "60",
        "--summary-json", str(summary_path),
    ])
    return json.loads(summary_path.read_text(encoding="utf-8")).get("results", [])


@pytest.mark.identification
def test_episode_identification_accuracy(request, tmp_path, monkeypatch, capsys):
    fixtures = _evidenced_tv_fixtures()
    if not fixtures:
        pytest.skip("no verified TV fixtures with cached evidence + episode labels + .mkv src yet")
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("TMDB_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY and TMDB_API_KEY required")

    by_kind = {}  # kind -> [correct, total]
    misses = []
    for fixture, scorable in fixtures:
        disc_tmp = tmp_path / fixture["_name"]
        disc_tmp.mkdir()
        results = _run_disc(fixture, scorable, disc_tmp, monkeypatch)
        by_src = {r.get("src"): r for r in results}

        for t in scorable:
            exp = t["expected"]
            kind = t["evidence"]["kind"]
            r = by_src.get(t["src"])
            ok = (
                r is not None and r.get("decision") == "renamed"
                and r.get("season") == exp["season"] and r.get("episode") == exp["episode"]
            )
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][1] += 1
            if ok:
                by_kind[kind][0] += 1
            else:
                if r and r.get("decision") == "renamed":
                    got = f"S{r.get('season'):02d}E{r.get('episode'):02d}"
                elif r:
                    got = r.get("decision")
                else:
                    got = "absent"
                misses.append(f"{fixture['_name']}:{t['src']} [{kind}] got {got}, "
                              f"expected S{exp['season']:02d}E{exp['episode']:02d}")

    correct = sum(c for c, _ in by_kind.values())
    total = sum(n for _, n in by_kind.values())
    accuracy = correct / total if total else 0.0

    with capsys.disabled():
        print(f"\n[identification accuracy] {correct}/{total} = {accuracy:.1%}")
        for kind in sorted(by_kind):
            c, n = by_kind[kind]
            if n:
                print(f"  {kind}: {c}/{n} = {c / n:.1%}")
        for m in misses:
            print(f"  MISS {m}")

    floor = float(request.config.getoption("--id-accuracy-floor"))
    assert accuracy >= floor, f"identification accuracy {accuracy:.1%} below floor {floor:.1%}"
