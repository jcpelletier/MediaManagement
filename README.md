# MediaManagement

A small collection of Python scripts for managing media files.

## Installation

1. **Install system tools**
   - [ffmpeg](https://ffmpeg.org/) is required for video conversion (used by `Convert_Video.py`).
   - `ffsubsync` depends on `ffmpeg` for audio analysis; ensure `ffmpeg` is on your `PATH`.
2. **Install Python dependencies**
   ```bash
   pip install ffsubsync requests
   ```

For fetching subtitles from OpenSubtitles, set an API key in the environment:
```bash
export OPENSUBTITLES_API_KEY="<your-api-key>"
```

## Scripts

### Convert_Video.py
Recursively converts `.mkv` files to `.mp4` with H.264 video and AAC audio.

**Usage**
```bash
python Convert_Video.py /path/to/folder [--overwrite] [--dry-run]
```
- `--overwrite` replaces existing `.mp4` files and deletes the source `.mkv` after successful conversion.
- `--dry-run` shows the planned `ffmpeg` commands without executing them.

### Fetch_Subs.py
Downloads high-quality subtitles from OpenSubtitles for a single `.mp4` file or every `.mp4` inside a directory.

**Usage**
```bash
python Fetch_Subs.py /path/to/video_or_directory [name_filter]
```
- Requires `OPENSUBTITLES_API_KEY` to be set.
- Optional `name_filter` restricts results to releases/filenames that contain the provided substring.
- Saves subtitles next to each video as `<video>.en.srt` by default.
- Skips files inside `Extras/` folders and stops early when the OpenSubtitles daily limit is reached (tracked in `~/.fetch_subs_opensubtitles_state.json`).
- Applies quality filters (minimum rating, avoids machine/AI translated results by default) when ranking candidates.

### AudioSync_Subs.py
Auto-syncs existing `.srt` subtitle files with their matching video tracks using `ffsubsync`.

**Usage**
```bash
python AudioSync_Subs.py "/path/to/library"
```
- Scans the directory recursively for matching video/subtitle pairs (same filename stem).
- Overwrites the subtitle file with the synced version and writes a log to `ffsubsync_audit.log`.

### Sort_Rips.py
Renames and moves ripped movies by asking OpenAI to guess the movie title from the folder name and contained video files. Intended for one-movie-per-folder rips; ambiguous folders (e.g., TV shows) are skipped.

**Prerequisites**
- Set `OPENAI_API_KEY` in your environment.
- Python dependencies: `requests`

**Usage**
```bash
python Sort_Rips.py [--source D:\Video] [--dest D:\Media\Movies] [--processed D:\Video\Processed] [--extensions .mkv,.mp4,...] [--min-confidence 0.6] [--overwrite] [--dry-run]
```
- Scans each immediate subfolder of `--source` for video files (default extensions include `.mkv`, `.mp4`, `.avi`, `.mov`, `.wmv`, `.m4v`, `.mpg`, `.mpeg`, `.ts`, `.flv`).
- Automatically skips the configured `--processed` directory to avoid reprocessing prior results.
- Sends a concise folder+file summary to the OpenAI API and expects JSON containing `title`, optional `year`, and `confidence`.
- Renames the largest video file in the folder to `Title.ext` or `Title (Year).ext` and moves it into `--dest`.
- Skips if the model returns a low/empty title or confidence below `--min-confidence`.
- Existing destination files are preserved unless `--overwrite` is provided.
- Use `--dry-run` to preview actions without renaming or moving files.
- After a folder is reviewed, it is moved to a `Processed` directory inside `--source` by default (override with `--processed`) so it is not re-checked.
- Empty source folders are deleted instead of being moved.

### Sort_TV.py
Renames TV episodes in place by parsing show/season from the parent folder name, extracting subtitle or audio evidence, asking OpenAI to identify the episode, and optionally verifying against TMDB. Can also rename immediately when OpenSubtitles returns an exact file hash match.

**Prerequisites**
- `ffmpeg` and `ffprobe` on `PATH`
- `OPENAI_API_KEY` set (LLM and transcription)
- `TMDB_API_KEY` set (optional verification)
- `OPENSUB_API_KEY` set (optional OpenSubtitles exact hash rename)

**Usage**
```bash
python Sort_TV.py --root /path/to/shows [flags]
```

**Flags**
- `--root` Root folder to scan recursively for `.mkv` files (required).
- `--model` OpenAI model name (default: `gpt-5.2`).
- `--min-minutes` Skip files shorter than this (default: 20).
- `--max-minutes` Skip files longer than this (default: 60).
- `--min-confidence` Minimum LLM confidence required to rename (default: 0.85).
- `--max-sub-lines` Subtitle lines to include as evidence (default: 80).
- `--dry-run` Print planned renames without renaming.
- `--audio-fallback` Enable audio transcription when subtitles fail (default: on).
- `--no-audio-fallback` Disable audio transcription fallback.
- `--audio-start-seconds` Start offset for primary audio clip (default: 120).
- `--audio-seconds` Deep fallback audio clip length in seconds (default: 300).
- `--opensubtitles-exact-rename` Rename immediately when OpenSubtitles hash lookup returns an exact show/season match.
- `--opensubtitles-user-agent` User-Agent header for OpenSubtitles API (default: `MediaManagement/1.0`).
- `--quiet` Reduce logging (only renames/skips/errors).
- `--no-verify-api` Disable TMDB verification.
- `--tmdb-api-key` TMDB API key (or use `TMDB_API_KEY` env var).
- `--tmdb-min-title-match` Minimum title similarity to confirm/correct using TMDB (default: 0.78).
