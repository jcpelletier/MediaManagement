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

