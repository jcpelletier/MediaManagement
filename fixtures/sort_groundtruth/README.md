# Sort-accuracy ground-truth dataset

Each JSON file here is one **ripped disc**: the raw pre-sort input the sorter saw,
plus the verified-correct answer. The test suite replays this input through the
live sort code (`Sort_Rips.py`, `Sort_TV.py`) and scores the result, so we can
watch a single master accuracy number move with every change instead of chasing
one-off misclassifications.

A disc is the unit of sorting, so it is the unit of a fixture. One disc can
contain a mix (a movie plus extras, or episodes spanning several seasons), so the
per-title `expected` field is what gets scored.

## Two artifacts: the manifest (automatic) and the fixture (hand-labeled)

There is a deliberate split so we never have to hand-label everything:

- **`rip_manifest.json`** is written by `rip-video.sh` on **every** rip. It is raw
  provenance only: the disc title and each title's original output filename,
  duration, and size. It contains **no labels**. A copy lives next to the rip and
  a durable copy goes to the manifest archive (`/mnt/media/rip_manifests/` on
  panda) so it survives sorting and cleanup. This is the pool of candidate test
  inputs.
- **A fixture** (a file in this directory) is a manifest that a human has picked
  out and hand-labeled with the correct answer, then marked `verified: true`. We
  only do this for a representative **sample**, and **only verified fixtures count
  toward the accuracy number**. The truth comes from a human, never from the
  sorter labeling itself.

## Why the data lives here

These fixtures version with the code they validate: a change to `Sort_Rips.py`
and any change to its expected behavior land in the same commit. The data is
small (filenames, durations, sizes, TMDB IDs, never media) and safe to commit to
this private repo.

## Schema (`schema_version: 1`)

```jsonc
{
  "schema_version": 1,
  "disc_title": "PAW PATROL_ PUPS SAVE PUPLANTIS#9B2B", // raw MakeMKV disc name = staging folder name
  "ripped_at": "2026-06-25",
  "source": "makemkv",            // "makemkv" (rip-time manifest) or "backfill" (reconstructed)
  "notes": "free text",

  "verified": true,               // human-confirmed. ONLY verified discs count toward the master number.

  "expected_routing": "tv",       // tier-1 ground truth: "tv" | "movie". The looks_like_tv_disc decision.

  "titles": [
    {
      "src": "title_t00.mkv",     // raw output filename from the rip (pre-rename)
      "size_bytes": 992609474,
      "duration_s": 1404,         // null if unknown
      "subs": { "codecs": ["pgs"], "bitmap_only": true },   // optional

      // Cached evidence so tier-2 runs fast and self-contained (no media, no Whisper).
      // Optional; populate with capture_evidence.py at label time. See "Evidence cache" below.
      "evidence": {
        "kind": "transcript",     // "subtitle_raw" | "transcript"
        "text": "..."
      },

      // tier-2 ground truth (per file). null when not yet labeled (tier-2 scoring skips it).
      "expected": {
        "type": "tv",             // "movie" | "tv" | "extra"
        "show": "PAW Patrol",     // tv only
        "tmdb_tv_id": 57532,      // tv only, preferred exact key
        "season": 3,              // tv only
        "episode": 1,             // tv only
        "title": "Pups Save Puplantis",
        "tmdb_movie_id": null,    // movie only
        "year": null              // movie only
      }
    }
  ]
}
```

### Field rules

- **`expected_routing`** is the only field the tier-1 (routing) test needs. Set it
  whenever you know whether the disc is a movie or a TV disc, even if you have not
  yet labeled the individual episodes.
- **`titles[].expected`** may be `null` until someone identifies the episode/movie.
  The tier-2 (identification) test skips `null` entries, so a partially labeled
  fixture is still useful for routing.
- **`verified`** must be `true` for a fixture to count. The current library
  contains known-wrong placements (this is what we are trying to measure), so an
  unverified or auto-scaffolded fixture is excluded until a human confirms it.
- Prefer **TMDB IDs** (`tmdb_tv_id` + `season` + `episode`, or `tmdb_movie_id`) as
  the label. They make scoring exact; titles have punctuation and spelling
  variance that fuzzy matching gets wrong.

## Evidence cache (tier-2 speed + self-containment)

Tier-2 identification normally reads the media (subtitles via ffmpeg, or audio via
Whisper). Whisper is by far the slowest step and needs the file present. To avoid
both, we cache the evidence into the fixture once, at label time, then the tier-2
test injects it instead of touching the media. Two flavors, chosen by what the
title has:

- **`subtitle_raw`**: the raw `.srt` text, for titles that have text subtitles.
  The test replays the *real* parsing/sampling (`Sort_TV.parse_full_dialogue`) on
  it, so this also covers the extraction-sampling logic, still with no media and
  no Whisper.
- **`transcript`**: the Whisper transcript, for titles with bitmap-only or no
  subtitles (e.g. many kids' discs). The test injects it straight into the
  identifier, since the transcript cannot be re-derived without the audio.

Either way the slow Whisper pass is removed and the fixture works after the media
is deleted. Because `kind` is recorded per title, the tier-2 test reports accuracy
**broken down by evidence kind**, so we can see how the sorter does on
subtitle-driven vs transcript-driven identification.

Populate it on panda (where the media lives):

```bash
# text subtitles -> caches raw .srt; bitmap/no subs -> add --whisper to transcribe
python3 capture_evidence.py \
  --fixture fixtures/sort_groundtruth/<disc>.json \
  --media "/mnt/media/.../<title>.mkv" --src "<title filename in the fixture>" [--whisper]
```

Sizing: cached evidence is a few KB to ~15 KB of text per title, so the curated
sample stays well under a megabyte in the repo. Cache only titles you actually
score by content; kept media can always re-extract on demand.

## Adding a fixture

1. **From a rip manifest** (preferred, captured automatically by `rip-video.sh`):
   ```bash
   python3 make_fixture.py --from-manifest "/mnt/media/Video/Processed/<disc>/rip_manifest.json"
   ```
2. **From an already-sorted library folder** (backfill; durations/sizes via ffprobe):
   ```bash
   python3 make_fixture.py --from-folder "/mnt/media/Media/Movies/<title>"
   ```

Both emit an **unverified** skeleton with `expected` fields blank. Fill in the
correct answers, set `verified: true`, and commit.
