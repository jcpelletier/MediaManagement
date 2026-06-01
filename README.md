# MediaManagement

Scripts for ripping, sorting, and converting media on **panda** (Ubuntu Server 24.04, `192.168.1.100`). Run as Jenkins jobs or directly on the server.

## Environment variables

| Variable | Used by |
|---|---|
| `DEEPSEEK_API_KEY` | Sort_Rips.py, Sort_TV.py |
| `DEEPSEEK_BASE_URL` | Sort_Rips.py, Sort_TV.py (optional; default `https://api.deepseek.com`) |
| `DEEPSEEK_MODEL` | Sort_Rips.py, Sort_TV.py (optional; default `deepseek-chat`) |
| `TMDB_API_KEY` | Sort_Rips.py, Sort_TV.py |
| `OPENSUB_API_KEY` | Sort_TV.py (optional) |

---

## Rip scripts (`rip/`)

Shell scripts triggered by udev when a disc is inserted. Deployed to `/opt/rip/` on the server.

### Deploy

```bash
ssh panda "sudo git -C /opt/MediaManagement pull && sudo bash /opt/MediaManagement/rip/install.sh"
```

On first setup, clone the repo first:
```bash
ssh panda "sudo git clone https://github.com/jcpelletier/MediaManagement.git /opt/MediaManagement && sudo bash /opt/MediaManagement/rip/install.sh"
```

### rip-video.sh

Rips a DVD/Blu-ray via MakeMKV into `/mnt/media/Video/<disc_title>/`. Triggered by udev on disc insert.

- Retries disc info up to 3× while the drive spins up
- Logs 10% progress milestones to `/var/log/rip-video.log` and App Insights
- Scans MakeMKV output for corruption indicators (damaged VOB, read errors, DvdFab/MacTheRipper warnings) — sends a Discord alert if found
- Ejects the disc on completion

### rip-cd.sh

Rips a CD to FLAC via `abcde` (config at `/etc/abcde-rip.conf`) into `/mnt/media/Media/Music/`. Logs to `/var/log/rip-cd.log` and App Insights.

---

## Python scripts

### Sort_Rips.py

Identifies and moves ripped movie folders from `/mnt/media/Video` to `/mnt/media/Media/Movies`. Uses DeepSeek to guess the title from folder name, file list, and subtitle/audio evidence, then verifies against TMDB.

**Usage**
```bash
python Sort_Rips.py \
  [--source /mnt/media/Video] \
  [--dest /mnt/media/Media/Movies] \
  [--processed /mnt/media/Video/Processed] \
  [--min-confidence 0.6] \
  [--dry-run]
```

**Output structure**

Each identified movie gets its own subfolder under `--dest`:
```
Movies/
  The Matrix (1999)/
    The Matrix (1999).mkv      ← largest video file, renamed
    Extras/
      bonus_feature.mkv        ← any other video files from the source folder
```

- TV show folders are correctly skipped — a TV-disc heuristic (folder name markers like `Season`/`SxxExx`, or several similarly-sized titles with episode-length median runtime) defers them to `Sort_TV.py` without sending them through TMDB's movie index. Disney/Pixar discs that expose the same feature as multiple large playlists are disambiguated by median duration (≥75 min → movie).
- After processing, non-video leftovers (subtitles, nfo files, etc.) are moved to `--processed`. If all files were moved to the library the source folder is deleted.

### Sort_TV.py

Identifies and renames TV episode `.mkv` files in place. Parses show/season from the parent folder name (e.g. `DS9S1D2`), extracts subtitle or audio evidence, asks DeepSeek to identify the episode, then verifies against TMDB.

**Usage**
```bash
python Sort_TV.py --root /mnt/media/Video/Processed [flags]
```

**Key flags**

| Flag | Default | Description |
|---|---|---|
| `--root` | *(required)* | Folder to scan recursively for `.mkv` files |
| `--model` | `deepseek-chat` | Model for guided identification (show/season known) |
| `--blind-model` | `deepseek-chat` | Model for blind identification (no folder hint) |
| `--min-minutes` | `6.0` | Skip files shorter than this (low enough for short-form kids' episodes) |
| `--max-minutes` | `100.0` | Skip files longer than this (allows 2-part episodes) |
| `--min-confidence` | `0.85` | Minimum LLM confidence to rename |
| `--whisper-model` | `small` | faster-whisper model size |
| `--whisper-device` | `cpu` | `cpu` or `cuda` |
| `--audio-start-seconds` | `300` | Start offset for primary audio clip (past most recurring intros). A small random jitter is added each call so episodes in a season don't all sample at the same intro timestamp. |
| `--no-audio-fallback` | | Disable Whisper transcription fallback |
| `--no-verify-api` | | Disable TMDB verification |
| `--tmdb-min-title-match` | `0.78` | Minimum title similarity for TMDB confirm |
| `--dry-run` | | Print planned renames without renaming |

**Identification pipeline**

1. Parse show + season from folder name via DeepSeek with regex fallback. Disc markers (`Disc 2`, `D2`, etc.) are stripped before parsing so they aren't mistaken for season numbers.
2. Fetch episode guide (titles, runtimes, synopses) from TMDB for LLM context. Previously-identified episode numbers from sibling discs in the same show/season are passed as a "previous disc ended at E{N}" hint.
3. Extract subtitle excerpt sampled at three timeline positions, skipping the first 5 minutes to avoid recurring intros / cold-open narration dominating the LLM context.
4. If no subtitles: Whisper audio transcription (primary clip → second clip → deep fallback). Sample start offset gets per-call random jitter to de-correlate samples across episodes that share intro timing.
5. DeepSeek identifies episode against the episode guide; TMDB verifies/corrects.
6. Per-folder reconciliation (Phase 2): proposals are queued and reconciled against contiguousness + TMDB runtime range at the folder boundary — outliers get rejected and retried.
7. Duplicate detection — if two files claim the same episode, the second retries with audio.

**Auto-written hints + sibling disc inheritance**

- After two TMDB-clean, high-confidence (≥0.90) identifications agree on `(show, season)` within a folder, Sort_TV writes a `sort_hints.json` to that folder so the remaining files skip blind mode.
- When a folder name has a disc marker (e.g. `ARRESTED_D2`), Sort_TV looks at sibling folders sharing the same base name (`ARRESTED_D1`, `ARRESTED_D3`, …); if their `sort_hints.json` files all agree on show + season, the current folder inherits them.

**Handling compilation discs / kids specials not in TMDB**

Place a `sort_hints.json` file inside the source folder to override folder-name parsing and bypass TMDB for content that isn't listed as a TV series (e.g. compilation discs, Dr. Seuss specials, holiday specials):

```json
{
  "show": "Dr. Seuss Specials",
  "season": 1,
  "skip_tmdb": true
}
```

- `show` + `season` — used directly, no DeepSeek folder parse or TMDB canonicalization
- `skip_tmdb: true` — skips TMDB episode guide fetch and verification; also tells the LLM that these are standalone programs in a compilation (so it won't reject them as "non-episodes")
- Files are renamed `Dr. Seuss Specials - S01E01 - Daisy-Head Mayzie.mkv` etc.
- Unidentifiable files still fall through to `Extras/` as usual

### Convert_Video.py

Recursively re-encodes `.mkv` files to H.264/AAC using `ffmpeg` (GPU-accelerated via `h264_nvenc` when available).

```bash
python Convert_Video.py /path/to/folder [--overwrite] [--dry-run]
```

### Extract_Subs.py

Extracts subtitle tracks from `.mkv` files into sidecar `.srt` files.

### AudioSync_Subs.py

Auto-syncs existing `.srt` files to their video using `ffsubsync`.

```bash
python AudioSync_Subs.py /path/to/library
```

### Fetch_Subs.py

Downloads subtitles from OpenSubtitles for `.mp4` files. Requires `OPENSUB_API_KEY`.

```bash
python Fetch_Subs.py /path/to/video_or_directory [name_filter]
```
