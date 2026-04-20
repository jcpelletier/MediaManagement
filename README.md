# MediaManagement

Scripts for ripping, sorting, and converting media on **panda** (Ubuntu Server 24.04, `192.168.1.100`). Run as Jenkins jobs or directly on the server.

## Environment variables

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | Sort_Rips.py, Sort_TV.py |
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

Identifies and moves ripped movie folders from `/mnt/media/Video` to `/mnt/media/Media/Movies`. Uses Claude to guess the title from folder name, file list, and subtitle/audio evidence, then verifies against TMDB.

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

- TV show folders are correctly skipped — TMDB movie search rejects them, and they pass through to `Sort_TV.py` via the Processed directory.
- After processing, non-video leftovers (subtitles, nfo files, etc.) are moved to `--processed`. If all files were moved to the library the source folder is deleted.

### Sort_TV.py

Identifies and renames TV episode `.mkv` files in place. Parses show/season from the parent folder name (e.g. `DS9S1D2`), extracts subtitle or audio evidence, asks Claude to identify the episode, then verifies against TMDB.

**Usage**
```bash
python Sort_TV.py --root /mnt/media/Video/Processed [flags]
```

**Key flags**

| Flag | Default | Description |
|---|---|---|
| `--root` | *(required)* | Folder to scan recursively for `.mkv` files |
| `--model` | `claude-sonnet-4-5` | Model for guided identification (show/season known) |
| `--blind-model` | `claude-sonnet-4-5` | Model for blind identification (no folder hint) |
| `--min-minutes` | `7.0` | Skip files shorter than this |
| `--max-minutes` | `100.0` | Skip files longer than this (allows 2-part episodes) |
| `--min-confidence` | `0.85` | Minimum LLM confidence to rename |
| `--whisper-model` | `small` | faster-whisper model size |
| `--whisper-device` | `cpu` | `cpu` or `cuda` |
| `--audio-start-seconds` | `120` | Start offset for primary audio clip |
| `--no-audio-fallback` | | Disable Whisper transcription fallback |
| `--no-verify-api` | | Disable TMDB verification |
| `--tmdb-min-title-match` | `0.78` | Minimum title similarity for TMDB confirm |
| `--dry-run` | | Print planned renames without renaming |

**Identification pipeline**

1. Parse show + season from folder name via Claude (Sonnet) with regex fallback
2. Fetch episode guide (titles, runtimes, synopses) from TMDB for LLM context
3. Extract subtitle excerpt (sampled from beginning, 25%, and 50% of file)
4. If no subtitles: Whisper audio transcription (primary clip → second clip → deep fallback)
5. Claude identifies episode against the episode guide; TMDB verifies/corrects
6. Duplicate detection — if two files claim the same episode, the second retries with audio

**Handling compilation discs / kids specials not in TMDB**

Place a `sort_hints.json` file inside the source folder to override folder-name parsing and bypass TMDB for content that isn't listed as a TV series (e.g. compilation discs, Dr. Seuss specials, holiday specials):

```json
{
  "show": "Dr. Seuss Specials",
  "season": 1,
  "skip_tmdb": true
}
```

- `show` + `season` — used directly, no Claude folder parse or TMDB canonicalization
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
