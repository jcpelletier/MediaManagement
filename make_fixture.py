#!/usr/bin/env python3
"""
Scaffold a sort-accuracy ground-truth fixture (fixtures/sort_groundtruth/).

Two sources:
  --from-manifest <rip_manifest.json>   # preferred; written automatically by rip-video.sh
  --from-folder   <library folder>      # backfill an already-sorted item; durations via ffprobe

Both emit an UNVERIFIED skeleton (verified=false, expected fields blank). Fill in
expected_routing + per-title expected, set verified=true, and commit. Only verified
fixtures count toward the master accuracy number (see fixtures/sort_groundtruth/README.md).
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "sort_groundtruth"

VIDEO_EXTS = {
    ".mkv", ".mp4", ".avi", ".mov", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".flv",
}


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "fixture"


def ffprobe_duration_seconds(path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        return round(float(out)) if out else None
    except (ValueError, FileNotFoundError):
        return None


def from_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    titles = []
    for t in data.get("titles", []):
        # Skip titles that failed to save: the sorter never sees them, so they
        # should not appear in the fixture. (saved absent => treat as present.)
        if t.get("saved") is False:
            continue
        titles.append({
            "src": t.get("src"),
            "size_bytes": t.get("size_bytes"),
            "duration_s": t.get("duration_s"),
            "expected": None,
        })
    return {
        "disc_title": data.get("disc_title", manifest_path.parent.name),
        "ripped_at": data.get("ripped_at", date.today().isoformat()),
        "source": data.get("source", "makemkv"),
        "titles": titles,
    }


def from_folder(folder: Path) -> dict:
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    titles = []
    for f in files:
        titles.append({
            "src": f.name,
            "size_bytes": f.stat().st_size,
            "duration_s": ffprobe_duration_seconds(f),
            "expected": None,
        })
    return {
        "disc_title": folder.name,
        "ripped_at": date.today().isoformat(),
        "source": "backfill",
        "titles": titles,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-manifest", type=Path, help="Path to a rip_manifest.json")
    src.add_argument("--from-folder", type=Path, help="Path to a sorted library folder")
    ap.add_argument("--out", type=Path, default=None, help="Output path (default: fixtures/sort_groundtruth/<slug>.json)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing fixture")
    args = ap.parse_args()

    if args.from_manifest:
        if not args.from_manifest.is_file():
            ap.error(f"manifest not found: {args.from_manifest}")
        core = from_manifest(args.from_manifest)
    else:
        if not args.from_folder.is_dir():
            ap.error(f"folder not found: {args.from_folder}")
        core = from_folder(args.from_folder)

    fixture = {
        "schema_version": 1,
        "disc_title": core["disc_title"],
        "ripped_at": core["ripped_at"],
        "source": core["source"],
        "notes": "",
        "verified": False,
        "expected_routing": None,
        "titles": core["titles"],
    }

    out = args.out or (FIXTURE_DIR / f"{slugify(core['disc_title'])}.json")
    if out.exists() and not args.force:
        print(f"refusing to overwrite existing fixture: {out} (use --force)", file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(fixture['titles'])} titles, UNVERIFIED; fill in expected_routing + expected, then set verified=true)")


if __name__ == "__main__":
    main()
