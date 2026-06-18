#!/usr/bin/env python3
"""
Resolve a hand-picked list of loose movie files that TMDB's bare-title search
mis-ranks (a same-named sequel/featurette/parody outranks the real film). For
each, we do what Sort_Rips does: query TMDB *with the correct release year*
(primary_release_year), apply the same 0.78 title-similarity gate, and on a
pass wrap the file + sidecars into a canonical "Title (Year)/" folder.

Below the gate it is left alone (exactly as Sort_Rips would skip it). Dry-run
unless --apply. Requires TMDB_API_KEY (or TMDB_KEY).
"""
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import requests

VID = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".flv"}
MIN_MATCH = 0.78  # Sort_Rips --tmdb-min-title-match default

# (loose filename in Movies/, query title, correct year)
TARGETS = [
    ("Beetlejuice.mp4", "Beetlejuice", 1988),
    ("Indiana Jones and the Raiders of the Lost Ark.mp4", "Raiders of the Lost Ark", 1981),
    ("Wall-E (2008).mp4", "WALL-E", 2008),
    ("Nausicaa of the Valley of the Wind.mp4", "Nausicaa of the Valley of the Wind", 1984),
    ("Me Eloise.mp4", "Me Eloise", 2006),
    ("The Pink Panther (1964).mp4", "The Pink Panther", 1963),
]


def norm(s):
    s = (s or "").lower().strip()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


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
    if not res:
        return None
    top = res[0]
    t = top.get("title") or top.get("original_title") or ""
    rd = top.get("release_date") or ""
    yr = int(rd[:4]) if rd[:4].isdigit() else None
    return (str(t), yr)


def main():
    apply = "--apply" in sys.argv
    root = Path("/mnt/media/Media/Movies")
    key = os.environ.get("TMDB_API_KEY") or os.environ.get("TMDB_KEY")
    if not key:
        print("ERROR: set TMDB_API_KEY (or TMDB_KEY)")
        return 2

    done = 0
    for fname, qtitle, year in TARGETS:
        src = root / fname
        if not src.exists():
            print(f"MISSING src: {fname}")
            continue
        res = tmdb_search(key, qtitle, year)
        if not res:
            print(f"SKIP  {fname}  (no TMDB result for '{qtitle}' {year})")
            continue
        ctitle, cyear = res
        s = sim(qtitle, ctitle)
        if s < MIN_MATCH:
            print(f"SKIP  {fname}  -> tmdb='{ctitle}' ({cyear}) sim={s:.2f} < {MIN_MATCH} (gate)")
            continue
        folder = f"{sanitize(ctitle)} ({cyear or year})"
        tgt = root / folder
        if tgt.exists():
            print(f"COLLISION {fname} -> {folder}/ (exists)")
            continue
        stem = src.stem
        group = [src] + [
            f for f in root.iterdir()
            if f.is_file() and f != src and f.suffix.lower() not in VID
            and (f.name == stem or f.name.startswith(stem + "."))
        ]
        print(f"{fname}  ->  {folder}/   [tmdb='{ctitle}' ({cyear}) sim={s:.2f}]")
        for f in group:
            newfn = folder + f.name[len(stem):]
            print(f"      {f.name} -> {newfn}")
            if apply:
                tgt.mkdir(exist_ok=True)
                f.rename(tgt / newfn)
        done += 1
    print(f"\n{'Applied' if apply else 'Would apply'} {done}.{'' if apply else '  (dry-run; pass --apply)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
