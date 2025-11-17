#!/usr/bin/env python3
import os
import re
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import requests

# -----------------------------------
# Config
# -----------------------------------

# IMPORTANT: OpenSubtitles expects 2-letter codes like "en"
PREFERRED_LANGUAGE = "en"

OPENSUBTITLES_API_URL = "https://api.opensubtitles.com/api/v1/subtitles"
OPENSUBTITLES_DOWNLOAD_URL = "https://api.opensubtitles.com/api/v1/download"

OPENSUBTITLES_API_KEY = os.environ.get("OPENSUBTITLES_API_KEY")

# REQUIRED by OpenSubtitles – must be a real app name/version string
USER_AGENT = "MySubFetcher/1.0"

# Self-imposed limits (tuned for free/dev-tier)
MAX_DOWNLOADS_PER_24H = 20       # tweak as you like
API_CALL_MIN_INTERVAL = 1.0      # seconds between API calls

# Where we track recent downloads
STATE_FILE = Path.home() / ".fetch_subs_opensubtitles_state.json"

# Global state
_last_api_call_ts: float = 0.0
DAILY_LIMIT_REACHED: bool = False


# -----------------------------------
# Rate limit helpers
# -----------------------------------

def _sleep_for_rate_limit():
    """Simple delay to keep requests spaced out."""
    global _last_api_call_ts
    now = time.time()
    elapsed = now - _last_api_call_ts
    if elapsed < API_CALL_MIN_INTERVAL:
        time.sleep(API_CALL_MIN_INTERVAL - elapsed)
    _last_api_call_ts = time.time()


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"downloads": []}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "downloads" not in data:
            return {"downloads": []}
        return data
    except Exception:
        return {"downloads": []}


def _save_state(state: dict):
    try:
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        # Not critical enough to kill the script
        pass


def _prune_old_downloads(downloads: list) -> list:
    """Keep only timestamps from the last 24 hours."""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    new_list = []
    for ts in downloads:
        try:
            dt = datetime.fromisoformat(ts)
            if dt >= cutoff:
                new_list.append(ts)
        except Exception:
            # Skip malformed timestamps
            pass
    return new_list


def can_download_more() -> Tuple[bool, int]:
    """
    Check if we are under our daily download limit.
    Returns (ok_to_download, remaining_count).
    """
    state = _load_state()
    downloads = _prune_old_downloads(state.get("downloads", []))
    count = len(downloads)
    remaining = MAX_DOWNLOADS_PER_24H - count
    return remaining > 0, remaining


def record_download():
    """Record one successful subtitle download."""
    state = _load_state()
    downloads = _prune_old_downloads(state.get("downloads", []))
    downloads.append(datetime.utcnow().isoformat())
    state["downloads"] = downloads
    _save_state(state)


# -----------------------------------
# Subtitle helpers
# -----------------------------------

def parse_title_and_year(filename: str) -> Tuple[str, Optional[int]]:
    """
    Supports:
      - 'MyMovie (1999).mp4' -> ('MyMovie', 1999)
      - 'MyMovie.mp4'        -> ('MyMovie', None)
    """
    stem = Path(filename).stem
    m = re.search(r"\(([12][0-9]{3})\)$", stem)
    year = None
    title = stem

    if m:
        year = int(m.group(1))
        title = stem[:m.start()].strip()

    title = re.sub(r"[._]+", " ", title).strip()
    return title, year


def _os_headers() -> dict:
    if not OPENSUBTITLES_API_KEY:
        raise RuntimeError("OPENSUBTITLES_API_KEY environment variable is not set.")
    return {
        "Api-Key": OPENSUBTITLES_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def _search_once(params: dict, label: str):
    """
    Helper to call the search API once with given params.
    Returns the full JSON data or None if no results.
    """
    headers = _os_headers()
    _sleep_for_rate_limit()
    resp = requests.get(OPENSUBTITLES_API_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    total = data.get("total_count", 0)
    print(f"    [{label}] total_count={total}")
    if not data.get("data"):
        return None
    return data


def search_subtitles(title: str, year: Optional[int], language: str = PREFERRED_LANGUAGE):
    """
    Try a few strategies in order:
      1) query + year + language
      2) query + language (no year)
      3) query only (no language filter)
    Returns best result dict or None.
    """
    print("  Searching subtitles...")

    # 1) query + year + language
    if year is not None:
        params1 = {
            "query": title,
            "year": str(year),
            "languages": language,
            "order_by": "download_count",
            "order_direction": "desc",
        }
        data = _search_once(params1, "q+year+lang")
        if data:
            return data["data"][0]

    # 2) query + language
    params2 = {
        "query": title,
        "languages": language,
        "order_by": "download_count",
        "order_direction": "desc",
    }
    data = _search_once(params2, "q+lang")
    if data:
        return data["data"][0]

    # 3) query only (no language restriction)
    params3 = {
        "query": title,
        "order_by": "download_count",
        "order_direction": "desc",
    }
    data = _search_once(params3, "q only")
    if data:
        return data["data"][0]

    print("  No subtitles found after all strategies.")
    return None


def download_subtitle_file(file_id: int, dest_path: Path):
    """
    Download a subtitle file identified by file_id and save it to dest_path.
    Sets DAILY_LIMIT_REACHED when limit is hit.
    """
    global DAILY_LIMIT_REACHED

    can_dl, remaining = can_download_more()
    if not can_dl:
        print(f"  Daily download limit reached ({MAX_DOWNLOADS_PER_24H}/24h). Stopping.")
        DAILY_LIMIT_REACHED = True
        return

    headers = _os_headers()
    payload = {"file_id": file_id}

    print(f"  Requesting download info (remaining today: {remaining})...")
    _sleep_for_rate_limit()
    resp = requests.post(OPENSUBTITLES_DOWNLOAD_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()

    download_data = resp.json()
    url = download_data["link"]

    print(f"  Downloading subtitles from {url}")
    _sleep_for_rate_limit()
    sub_resp = requests.get(url, timeout=60)
    sub_resp.raise_for_status()

    dest_path.write_bytes(sub_resp.content)
    record_download()
    print(f"  Saved subtitles to {dest_path}")


def is_in_extras_folder(path: Path) -> bool:
    """True if any part of the path is an 'Extras' folder (case-insensitive)."""
    return any(part.lower() == "extras" for part in path.parts)


def ensure_subtitles_for_video(video_path: Path, language: str = PREFERRED_LANGUAGE):
    """Main flow for a single video. Does nothing if limit reached."""
    global DAILY_LIMIT_REACHED

    if DAILY_LIMIT_REACHED:
        return

    if is_in_extras_folder(video_path):
        print(f"Skipping (Extras folder): {video_path}")
        return

    if not video_path.exists():
        print(f"File not found: {video_path}")
        return

    subs_path = video_path.with_suffix(f".{language}.srt")
    if subs_path.exists():
        print(f"Skipping (already exists): {subs_path}")
        return

    title, year = parse_title_and_year(video_path.name)
    print(f"Processing: {video_path}")
    print(f"  Parsed -> title='{title}', year={year}")

    try:
        result = search_subtitles(title, year, language)
    except Exception as e:
        print(f"  Error talking to OpenSubtitles: {e}")
        return

    if not result:
        return

    files = result.get("attributes", {}).get("files", [])
    if not files:
        print("  Entry has no downloadable files.")
        return

    file_id = files[0]["file_id"]

    try:
        download_subtitle_file(file_id, subs_path)
    except Exception as e:
        print(f"  Error downloading subtitle file: {e}")
        if subs_path.exists() and subs_path.stat().st_size == 0:
            subs_path.unlink(missing_ok=True)


def process_target(target: Path):
    """
    If target is a file: process it (unless limit reached).
    If target is a directory: walk all .mp4 files and stop early
    when DAILY_LIMIT_REACHED is set.
    """
    global DAILY_LIMIT_REACHED

    if target.is_file():
        if target.suffix.lower() == ".mp4":
            ensure_subtitles_for_video(target)
        else:
            print(f"Not an mp4: {target}")
        if DAILY_LIMIT_REACHED:
            print("\nDaily limit reached — stopping.")
        return

    if target.is_dir():
        print(f"Scanning directory: {target}")
        for p in sorted(target.rglob("*.mp4")):
            if DAILY_LIMIT_REACHED:
                print("\nDaily limit reached — stopping early.")
                break
            ensure_subtitles_for_video(p)
        return

    print(f"Invalid target: {target}")


# -----------------------------------
# CLI
# -----------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_subs.py <video_file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1]).expanduser().resolve()
    process_target(target)

    # Always exit 0 unless something raised an uncaught exception
    # (Jenkins will see success even when limit was reached)
    return


if __name__ == "__main__":
    main()
