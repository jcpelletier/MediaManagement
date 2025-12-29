#!/usr/bin/env python3
"""
Sort ripped movie folders by asking an LLM to guess the movie title.

For each subdirectory inside the source root (default: D:\\Video), the script:
- collects video filenames and sizes
- asks the OpenAI API for a best-guess movie title/year
- renames the largest video file in place to "Title.ext" or "Title (Year).ext"
- moves that renamed file into the destination root (default: D:\\Media\\Movies)

Other files are left untouched. Folders with low/unknown confidence are skipped.
"""

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests


DEFAULT_EXTENSIONS = [
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".flv",
]


@dataclass
class FolderGuess:
    title: str
    year: Optional[int]
    confidence: float


def parse_extensions(ext_list: Iterable[str]) -> List[str]:
    return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext_list]


def collect_video_files(folder: Path, extensions: List[str]) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]


def format_files_for_prompt(files: List[Path]) -> str:
    lines = []
    for file in sorted(files):
        size_mb = file.stat().st_size / (1024 * 1024)
        lines.append(f"- {file.name} ({size_mb:.1f} MB)")
    return "\n".join(lines)


def call_openai(folder_name: str, files_summary: str, api_key: str) -> Optional[FolderGuess]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_prompt = (
        "You are a movie identifier. Given a folder name and contained video files, "
        "return the most likely movie title and optional year. Respond ONLY with JSON "
        "using keys: title (string), year (integer or null), confidence (0-1). If unsure, "
        "set an empty title and confidence 0."
    )
    user_prompt = f"Folder: {folder_name}\nFiles:\n{files_summary}"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR: OpenAI request failed for '{folder_name}': {exc}")
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Could not parse OpenAI response for '{folder_name}': {exc}")
        return None

    title = (data.get("title") or "").strip()
    year = data.get("year")
    confidence = float(data.get("confidence") or 0)

    if year is not None:
        try:
            year_int = int(year)
            year = year_int if 1800 <= year_int <= 2100 else None
        except (TypeError, ValueError):
            year = None

    return FolderGuess(title=title, year=year, confidence=confidence)


def sanitize_title(title: str) -> str:
    # Remove characters that are invalid in Windows filenames.
    return re.sub(r'[<>:"/\\|?*]+', " ", title).strip()


def rename_and_move(largest_file: Path, guess: FolderGuess, dest_root: Path, overwrite: bool, dry_run: bool) -> None:
    title = sanitize_title(guess.title)
    if not title:
        print(f"SKIP: Empty title after sanitization for {largest_file.parent.name}")
        return

    new_name = f"{title}{f' ({guess.year})' if guess.year else ''}{largest_file.suffix}"
    renamed_path = largest_file.with_name(new_name)
    dest_path = dest_root / new_name

    if dest_path.exists() and not overwrite:
        print(f"SKIP: Destination exists, use --overwrite to replace: {dest_path}")
        return

    print(f"  Rename: {largest_file.name} -> {renamed_path.name}")
    print(f"  Move  : {renamed_path} -> {dest_path}")

    if dry_run:
        return

    if not dest_root.exists():
        dest_root.mkdir(parents=True, exist_ok=True)

    try:
        if largest_file != renamed_path:
            largest_file.rename(renamed_path)
        if dest_path.exists() and overwrite:
            dest_path.unlink()
        shutil.move(str(renamed_path), str(dest_path))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Could not move '{largest_file}': {exc}")


def move_folder_to_processed(folder: Path, processed_root: Path, dry_run: bool) -> None:
    if not any(folder.iterdir()):
        print(f"  Delete empty folder: {folder}")
        if not dry_run:
            try:
                folder.rmdir()
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: Could not delete empty folder '{folder}': {exc}")
        return

    target = processed_root / folder.name

    if folder.resolve() == target.resolve():
        return

    if target.exists():
        print(f"SKIP: Processed destination already exists, not moving folder: {target}")
        return

    print(f"  Processed move: {folder} -> {target}")
    if dry_run:
        return

    try:
        processed_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(target))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Could not move folder to processed location '{target}': {exc}")


def process_folder(folder: Path, extensions: List[str], api_key: str, min_confidence: float, dest_root: Path, overwrite: bool, dry_run: bool) -> None:
    video_files = collect_video_files(folder, extensions)
    if not video_files:
        return

    largest_file = max(video_files, key=lambda p: p.stat().st_size)
    files_summary = format_files_for_prompt(video_files)

    guess = call_openai(folder.name, files_summary, api_key)
    if not guess:
        return

    if not guess.title or guess.confidence < min_confidence:
        print(
            f"SKIP: Low confidence ({guess.confidence:.2f}) or missing title for folder '{folder.name}'"
        )
        return

    rename_and_move(largest_file, guess, dest_root, overwrite, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename and move ripped movies using OpenAI title guesses.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\\Video"),
        help="Source root containing ripped folders (default: D:\\Video)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(r"D:\\Media\\Movies"),
        help="Destination root for renamed movies (default: D:\\Media\\Movies)",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Folder to move processed source subfolders into (default: <source>/Processed).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list of video extensions to consider (default: common formats)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show actions without renaming or moving files.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence (0-1) required to rename and move a folder (default: 0.6)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(2)

    source_root: Path = args.source
    dest_root: Path = args.dest
    processed_root: Path = args.processed or (source_root / "Processed")
    extensions = parse_extensions([ext.strip() for ext in args.extensions.split(",") if ext.strip()])

    if not source_root.exists() or not source_root.is_dir():
        print(f"ERROR: Source root is not a directory: {source_root}")
        sys.exit(2)

    print(f"Source: {source_root}")
    print(f"Dest  : {dest_root}")
    print(f"Processed: {processed_root}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Minimum confidence: {args.min_confidence}")

    processed_root_resolved = processed_root.resolve()
    subfolders = [
        p
        for p in source_root.iterdir()
        if p.is_dir() and p.resolve() != processed_root_resolved
    ]
    if not subfolders:
        print("No subdirectories found to process.")
        return

    for folder in sorted(subfolders):
        print(f"\nFolder: {folder.name}")
        process_folder(
            folder,
            extensions=extensions,
            api_key=api_key,
            min_confidence=args.min_confidence,
            dest_root=dest_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        move_folder_to_processed(folder, processed_root, args.dry_run)


if __name__ == "__main__":
    main()
