#!/usr/bin/env python3
import os
import re
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import requests

# ----------------------------------- 
# Config
# ----------------------------------- 
PREFERRED_LANGUAGE = "en"
OPENSUBTITLES_API_URL = "https://api.opensubtitles.com/api/v1/subtitles"
OPENSUBTITLES_DOWNLOAD_URL = "https://api.opensubtitles.com/api/v1/download"
OPENSUBTITLES_API_KEY = os.environ.get("OPENSUBTITLES_API_KEY")
USER_AGENT = "MySubFetcher/1.0"
MAX_DOWNLOADS_PER_24H = 20
API_CALL_MIN_INTERVAL = 1.0
STATE_FILE = Path.home() / ".fetch_subs_opensubtitles_state.json"

# Quality thresholds
MIN_RATING = 6.0  # Minimum acceptable rating (0-10 scale)
MIN_DOWNLOADS = 100  # Prefer subs downloaded at least this many times
AVOID_MACHINE_TRANSLATED = True
AVOID_HEARING_IMPAIRED = False  # Set True if you don't want SDH subtitles
MAX_RESULTS_TO_FETCH = 20  # Get more results to score

_last_api_call_ts: float = 0.0
DAILY_LIMIT_REACHED: bool = False
CONSECUTIVE_FAILURES: int = 0
MAX_CONSECUTIVE_FAILURES: int = 5

# ----------------------------------- 
# Rate limit helpers
# ----------------------------------- 
def _sleep_for_rate_limit():
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
        pass

def _prune_old_downloads(downloads: list) -> list:
    cutoff = datetime.utcnow() - timedelta(hours=24)
    new_list = []
    for ts in downloads:
        try:
            dt = datetime.fromisoformat(ts)
            if dt >= cutoff:
                new_list.append(ts)
        except Exception:
            pass
    return new_list

def can_download_more() -> Tuple[bool, int]:
    state = _load_state()
    downloads = _prune_old_downloads(state.get("downloads", []))
    count = len(downloads)
    remaining = MAX_DOWNLOADS_PER_24H - count
    return remaining > 0, remaining

def record_download():
    state = _load_state()
    downloads = _prune_old_downloads(state.get("downloads", []))
    downloads.append(datetime.utcnow().isoformat())
    state["downloads"] = downloads
    _save_state(state)

# ----------------------------------- 
# Subtitle quality scoring
# ----------------------------------- 
def calculate_quality_score(result: dict, title: str) -> float:
    """
    Calculate a quality score for a subtitle result.
    Higher score = better quality.
    """
    score = 0.0
    attrs = result.get("attributes", {})
    
    # 1. Rating (0-10) - weight heavily
    rating = attrs.get("ratings", 0) or 0
    score += rating * 10  # Max +100 points
    
    # 2. Download count (popularity) - logarithmic scale
    downloads = attrs.get("download_count", 0) or 0
    if downloads > 0:
        import math
        score += min(math.log10(downloads) * 15, 50)  # Max +50 points
    
    # 3. Penalize machine/AI translated
    if AVOID_MACHINE_TRANSLATED:
        if attrs.get("machine_translated"):
            score -= 30
        if attrs.get("ai_translated"):
            score -= 25
    
    # 4. Penalize hearing impaired if unwanted
    if AVOID_HEARING_IMPAIRED and attrs.get("hearing_impaired"):
        score -= 20
    
    # 5. Recency bonus (uploaded in last year)
    upload_date = attrs.get("upload_date")
    if upload_date:
        try:
            upload_dt = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
            age_days = (datetime.now(upload_dt.tzinfo) - upload_dt).days
            if age_days < 365:
                score += (365 - age_days) / 365 * 20  # Max +20 points for very recent
        except Exception:
            pass
    
    # 6. Title match quality (check if movie name matches)
    feature_details = attrs.get("feature_details", {})
    movie_name = feature_details.get("movie_name", "")
    if movie_name:
        # Simple fuzzy match - you can enhance this
        title_lower = title.lower()
        movie_lower = movie_name.lower()
        if title_lower in movie_lower or movie_lower in title_lower:
            score += 15
    
    # 7. Prefer non-foreign parts only
    if attrs.get("foreign_parts_only"):
        score -= 40
    
    return score

def filter_and_rank_results(results: List[dict], title: str) -> List[Tuple[float, dict]]:
    """
    Filter and rank subtitle results by quality score.
    Returns list of (score, result) tuples, sorted best first.
    """
    scored = []
    
    for result in results:
        attrs = result.get("attributes", {})
        
        # Hard filters
        rating = attrs.get("ratings", 0) or 0
        downloads = attrs.get("download_count", 0) or 0
        
        # Skip if below minimum thresholds
        if rating < MIN_RATING and rating > 0:  # rating=0 means unrated, allow those
            continue
        
        # Skip foreign parts only
        if attrs.get("foreign_parts_only"):
            continue
            
        score = calculate_quality_score(result, title)
        scored.append((score, result))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# ----------------------------------- 
# Subtitle search
# ----------------------------------- 
def parse_title_and_year(filename: str) -> Tuple[str, Optional[int]]:
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

def _search_once(params: dict, label: str) -> Optional[List[dict]]:
    """
    Call the search API once with given params.
    Returns list of all results or None if no results.
    """
    headers = _os_headers()
    _sleep_for_rate_limit()
    
    resp = requests.get(OPENSUBTITLES_API_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    
    total = data.get("total_count", 0)
    results = data.get("data", [])
    
    print(f"  [{label}] total_count={total}, fetched={len(results)}")
    
    if not results:
        return None
    return results

def search_subtitles(title: str, year: Optional[int], language: str = PREFERRED_LANGUAGE):
    """
    Search for subtitles with quality scoring.
    Returns best result dict or None.
    """
    print("  Searching subtitles...")
    
    all_results = []
    
    # Strategy 1: query + year + language
    if year is not None:
        params1 = {
            "query": title,
            "year": str(year),
            "languages": language,
        }
        results = _search_once(params1, "q+year+lang")
        if results:
            all_results.extend(results)
    
    # Strategy 2: query + language (if we don't have enough results yet)
    if len(all_results) < MAX_RESULTS_TO_FETCH:
        params2 = {
            "query": title,
            "languages": language,
        }
        results = _search_once(params2, "q+lang")
        if results:
            # Avoid duplicates
            existing_ids = {r.get("id") for r in all_results}
            all_results.extend([r for r in results if r.get("id") not in existing_ids])
    
    if not all_results:
        print("  No subtitles found.")
        return None
    
    # Score and rank all results
    scored_results = filter_and_rank_results(all_results, title)
    
    if not scored_results:
        print("  No subtitles passed quality filters.")
        return None
    
    # Show top 3 candidates
    print(f"\n  Top subtitle candidates (from {len(scored_results)} filtered):")
    for i, (score, result) in enumerate(scored_results[:3], 1):
        attrs = result.get("attributes", {})
        rating = attrs.get("ratings", 0) or 0
        downloads = attrs.get("download_count", 0) or 0
        upload_date = attrs.get("upload_date", "unknown")[:10]
        machine = "[MT]" if attrs.get("machine_translated") else ""
        ai = "[AI]" if attrs.get("ai_translated") else ""
        hi = "[HI]" if attrs.get("hearing_impaired") else ""
        
        # Get release/file name
        release = attrs.get("release", "")
        files = attrs.get("files", [])
        filename = files[0].get("file_name", "") if files else ""
        display_name = release or filename or "unknown"
        
        print(f"    {i}. Score: {score:.1f} | Rating: {rating}/10 | "
              f"Downloads: {downloads} | Uploaded: {upload_date} {machine}{ai}{hi}")
        print(f"       File: {display_name}")
    
    best_score, best_result = scored_results[0]
    print(f"\n  [OK] Selected best subtitle (score: {best_score:.1f})")
    return best_result

def download_subtitle_file(file_id: int, dest_path: Path):
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
    return any(part.lower() == "extras" for part in path.parts)

def ensure_subtitles_for_video(video_path: Path, language: str = PREFERRED_LANGUAGE):
    global DAILY_LIMIT_REACHED, CONSECUTIVE_FAILURES
    if DAILY_LIMIT_REACHED:
        return
    
    if CONSECUTIVE_FAILURES >= MAX_CONSECUTIVE_FAILURES:
        print(f"\n!!! {MAX_CONSECUTIVE_FAILURES} consecutive failures detected. Stopping script. !!!")
        sys.exit(1)
    
    if is_in_extras_folder(video_path):
        print(f"Skipping (Extras folder): {video_path}")
        return
    
    if not video_path.exists():
        print(f"File not found: {video_path}")
        CONSECUTIVE_FAILURES += 1
        return
    
    subs_path = video_path.with_suffix(f".{language}.srt")
    if subs_path.exists():
        print(f"Skipping (already exists): {subs_path}")
        CONSECUTIVE_FAILURES = 0  # Reset on success (skip counts as success)
        return
    
    title, year = parse_title_and_year(video_path.name)
    print(f"\nProcessing: {video_path}")
    print(f"  Parsed -> title='{title}', year={year}")
    
    try:
        result = search_subtitles(title, year, language)
    except Exception as e:
        print(f"  Error talking to OpenSubtitles: {e}")
        CONSECUTIVE_FAILURES += 1
        return
    
    if not result:
        CONSECUTIVE_FAILURES += 1
        return
    
    files = result.get("attributes", {}).get("files", [])
    if not files:
        print("  Entry has no downloadable files.")
        CONSECUTIVE_FAILURES += 1
        return
    
    file_id = files[0]["file_id"]
    try:
        download_subtitle_file(file_id, subs_path)
        CONSECUTIVE_FAILURES = 0  # Reset on successful download
    except Exception as e:
        print(f"  Error downloading subtitle file: {e}")
        CONSECUTIVE_FAILURES += 1
        if subs_path.exists() and subs_path.stat().st_size == 0:
            subs_path.unlink(missing_ok=True)

def process_target(target: Path):
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_subs.py <video_file_or_directory>")
        sys.exit(1)
    
    target = Path(sys.argv[1]).expanduser().resolve()
    process_target(target)

if __name__ == "__main__":
    main()