#!/usr/bin/env python3
"""
Rename movie folders + their files in the library to TMDB's canonical title.

Older library entries predate the canonical-naming convention (e.g. folder
"The Beast with a Billion Backs" whose canonical TMDB title is
"Futurama: The Beast with a Billion Backs"). This brings them in line so the
library, Jellyfin, and the accuracy test all agree on names.

SAFE BY DEFAULT: dry-run unless --apply is passed. Only AUTO proposals (where
the canonical title is clearly the same movie - a squashed superset of the
current name, or >= 0.85 similar, with the year agreeing) are applied; REVIEW
items are listed for a human and never auto-renamed. Requires TMDB_API_KEY (or
TMDB_KEY) in the environment.
"""
import argparse
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import requests

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".flv"}
MOVIE_DIR_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")


def norm(s):
    s = (s or "").lower().strip()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def squash(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def sim(a, b):
    a, b = norm(a), norm(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def sanitize(t):
    return re.sub(r'[<>:"/\\|?*]+', " ", t).strip()


def tmdb_search(key, title, year):
    params = {"api_key": key, "query": title, "include_adult": "false"}
    if year:
        params["primary_release_year"] = str(year)
    r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=10)
    r.raise_for_status()
    res = r.json().get("results") or []
    if not res and year:
        return tmdb_search(key, title, None)
    if not res:
        return None
    top = res[0]
    t = top.get("title") or top.get("original_title") or ""
    rd = top.get("release_date") or ""
    yr = int(rd[:4]) if rd[:4].isdigit() else None
    return (str(t), yr)


def plan_for_folder(folder, key):
    """Return (status, current_name, new_name, detail). status in
    skip/auto/review/nomatch."""
    vids = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    if not vids:
        return ("skip", folder.name, None, "no video files")
    m = MOVIE_DIR_RE.match(folder.name)
    cur_title = m.group("title").strip() if m else folder.name.strip()
    cur_year = int(m.group("year")) if m else None

    try:
        res = tmdb_search(key, cur_title, cur_year)
    except Exception as e:
        return ("nomatch", folder.name, None, f"tmdb error: {e}")
    if not res:
        return ("nomatch", folder.name, None, "no TMDB result")
    canon_title, canon_year = res
    canon_title_s = sanitize(canon_title)
    yr = canon_year or cur_year
    new_name = f"{canon_title_s} ({yr})" if yr else canon_title_s

    if new_name == folder.name:
        return ("skip", folder.name, new_name, "already canonical")

    s = sim(cur_title, canon_title)
    year_ok = (cur_year is None or canon_year is None or cur_year == canon_year)
    superset = squash(cur_title) in squash(canon_title) or squash(canon_title) in squash(cur_title)
    detail = (f"tmdb='{canon_title}' ({canon_year}) sim={s:.2f}"
              f"{' superset' if superset else ''}{'' if year_ok else ' YEAR-MISMATCH'}")

    if year_ok and (superset or s >= 0.85):
        return ("auto", folder.name, new_name, detail)
    return ("review", folder.name, new_name, detail)


def apply_rename(folder, new_name):
    target = folder.parent / new_name
    if target.exists():
        print(f"  COLLISION: {target} exists - skipping")
        return False
    old_stem = folder.name
    new_has_year = bool(re.search(r"\(\d{4}\)\s*$", new_name))
    for f in list(folder.iterdir()):
        if f.is_file() and f.name.startswith(old_stem):
            rest = f.name[len(old_stem):]
            # If the file already carried a "(YYYY)" that new_name also has,
            # drop the duplicate so we don't produce "Title (2002) (2002).mp4".
            if new_has_year:
                rest = re.sub(r"^\s*\(\d{4}\)", "", rest)
            newfn = new_name + rest
            f.rename(folder / newfn)
            print(f"  file: {f.name} -> {newfn}")
    folder.rename(target)
    print(f"  dir : {old_stem} -> {new_name}")
    return True


# ---- flat-file movies (bare Title.ext directly under Movies/) ----

def parse_title_year(stem):
    m = MOVIE_DIR_RE.match(stem)
    if m:
        return m.group("title").strip(), int(m.group("year"))
    return stem.strip(), None


def associated_files(root, video):
    """The video itself plus its sidecars - other (non-video) files whose name
    is exactly the video stem or starts with 'stem.' (a dot after the stem).
    The dot guard stops 'Sister Act 2.mp4' being pulled into 'Sister Act'."""
    stem = video.stem
    group = [video]
    for f in root.iterdir():
        if f == video or not f.is_file() or f.suffix.lower() in VIDEO_EXTS:
            continue
        if f.name == stem or f.name.startswith(stem + "."):
            group.append(f)
    return group


def plan_flat(video, key):
    """Plan wrapping a bare movie file into a canonical Title (Year)/ folder.
    Returns (status, video, new_folder, detail)."""
    cur_title, cur_year = parse_title_year(video.stem)
    try:
        res = tmdb_search(key, cur_title, cur_year)
    except Exception as e:
        return ("nomatch", video, None, f"tmdb error: {e}")
    if not res:
        return ("nomatch", video, None, "no TMDB result")
    canon_title, canon_year = res
    yr = canon_year or cur_year
    new_folder = f"{sanitize(canon_title)} ({yr})" if yr else sanitize(canon_title)

    s = sim(cur_title, canon_title)
    year_ok = (cur_year is None or canon_year is None or cur_year == canon_year)
    superset = squash(cur_title) in squash(canon_title) or squash(canon_title) in squash(cur_title)
    detail = (f"tmdb='{canon_title}' ({canon_year}) sim={s:.2f}"
              f"{' superset' if superset else ''}{'' if year_ok else ' YEAR-MISMATCH'}")

    if not yr:
        return ("review", video, new_folder, detail + " no-year")
    if (video.parent / new_folder).exists():
        return ("review", video, new_folder, detail + " TARGET-FOLDER-EXISTS")
    if year_ok and (superset or s >= 0.85):
        return ("auto", video, new_folder, detail)
    return ("review", video, new_folder, detail)


def apply_flat(root, video, new_folder):
    target = root / new_folder
    if target.exists():
        print(f"  COLLISION: {target} exists - skipping")
        return False
    group = associated_files(root, video)
    base = video.stem  # full stem (may include a year) - replaced wholesale
    target.mkdir(parents=True)
    for f in group:
        newfn = new_folder + f.name[len(base):]
        f.rename(target / newfn)
        print(f"  {f.name} -> {new_folder}/{newfn}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movies-root", type=Path, default=Path("/mnt/media/Media/Movies"))
    ap.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run).")
    ap.add_argument("--mode", choices=["folders", "flat", "both"], default="both",
                    help="folders = canonicalize existing Title (Year)/ dirs; "
                         "flat = wrap bare Title.ext files into canonical folders; both (default).")
    ap.add_argument("--tmdb-api-key",
                    default=os.environ.get("TMDB_API_KEY") or os.environ.get("TMDB_KEY"))
    args = ap.parse_args()
    if not args.tmdb_api_key:
        print("ERROR: set TMDB_API_KEY (or TMDB_KEY)")
        return 2
    root = args.movies_root

    # ---- folder pass ----
    f_auto, f_review, f_nomatch, f_canon = [], [], [], 0
    if args.mode in ("folders", "both"):
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            status, cur, new, detail = plan_for_folder(folder, args.tmdb_api_key)
            if status == "skip":
                f_canon += 1
            elif status == "auto":
                f_auto.append((folder, cur, new, detail))
            elif status == "review":
                f_review.append((cur, new, detail))
            else:
                f_nomatch.append((cur, detail))

    # ---- flat-file pass ----
    x_auto, x_review, x_nomatch = [], [], []
    if args.mode in ("flat", "both"):
        videos = sorted(f for f in root.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
        for video in videos:
            status, vid, new, detail = plan_flat(video, args.tmdb_api_key)
            if status == "auto":
                x_auto.append((vid, vid.name, new, detail))
            elif status == "review":
                x_review.append((vid.name, new, detail))
            else:
                x_nomatch.append((vid.name, detail))

    print(f"\nFOLDERS: already-canonical {f_canon} | AUTO {len(f_auto)} | "
          f"REVIEW {len(f_review)} | no-match {len(f_nomatch)}")
    print(f"FLAT   : AUTO {len(x_auto)} | REVIEW {len(x_review)} | no-match {len(x_nomatch)}\n")

    if f_auto:
        print("== FOLDER AUTO ==")
        for _, cur, new, detail in f_auto:
            print(f"  '{cur}' -> '{new}'   [{detail}]")
    if x_auto:
        print("\n== FLAT AUTO (wrap into folder on --apply) ==")
        for _, cur, new, detail in x_auto:
            print(f"  '{cur}' -> '{new}/'   [{detail}]")
    print("\n== REVIEW (left alone - decide manually) ==")
    for cur, new, detail in f_review + x_review:
        print(f"  '{cur}'  ~?  '{new}'   [{detail}]")
    print("\n== NO TMDB MATCH ==")
    for cur, detail in f_nomatch + x_nomatch:
        print(f"  '{cur}'   [{detail}]")

    if args.apply:
        done = 0
        if f_auto:
            print(f"\n--- APPLYING {len(f_auto)} FOLDER rename(s) ---")
            for folder, cur, new, _ in f_auto:
                print(f"* {cur}")
                if apply_rename(folder, new):
                    done += 1
        if x_auto:
            print(f"\n--- APPLYING {len(x_auto)} FLAT wrap(s) ---")
            for video, cur, new, _ in x_auto:
                print(f"* {cur}")
                if apply_flat(root, video, new):
                    done += 1
        print(f"\nApplied {done} rename(s).")
    else:
        print("\n(dry-run - pass --apply to perform the AUTO actions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
