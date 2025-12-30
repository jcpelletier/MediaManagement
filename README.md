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
python Fetch_Subs.py /path/to/video_or_directory
```
- Requires `OPENSUBTITLES_API_KEY` to be set.
- Saves subtitles next to each video as `<video>.en.srt` by default.
- Skips files inside `Extras/` folders and stops early when the OpenSubtitles daily limit is reached.

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
