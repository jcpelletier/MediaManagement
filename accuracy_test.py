#!/usr/bin/env python3
"""
accuracy_test.py - identification-accuracy harness for Sort_Rips.py and Sort_TV.py.

The library under <media-root>/{Movies,Shows} is already correctly named, so it is a
labeled ground-truth set. This harness feeds the *real media bytes* through the *real
sorter scripts* but with folder/file names replaced by obfuscated stand-ins that mimic
real MakeMKV disc labels, then compares what the sorters decided against the known-correct
library names and reports per-index accuracy.

How it stays safe and faithful:
- The real library is NEVER modified. Staging contains only hardlinks (symlink fallback)
  to the real files; the sorters rename/move/delete only those links. ffprobe reads follow
  the link to the real bytes. Targets are untouched.
- Decisions are mapped back to ground truth by inode identity: a hardlink shares
  (st_dev, st_ino) with its target, and a stat-followed symlink resolves to the same. After
  a real run we walk the output tree, stat each file, and match (st_dev, st_ino) back to the
  ground-truth file. No edits to the sorters, no console scraping.

Indexes (obfuscation schemes):
  Sort_Rips (single-file movies):
    1  folder vaguely hints the title (MENINBLACK / MIBMOVIE); random file name
    2  folder constant MOVIEFOLDER_<NNN> (no hint); random file name
  Sort_TV (seasons split into discs of --episodes-per-disc, default 4):
    1  <SHOWTOKEN>_SEASON<N>_DISC<k>   show + correct season known
    2  <SHOWTOKEN>_DISC<k>             show known, season absent
    3  <random token> per disc         fully blind

Report-only: always exits 0 (non-zero only on harness errors - bad paths, no media,
link mode unavailable).

Requires: ffprobe/ffmpeg + the sorters' own deps (run on panda). Reads
DEEPSEEK_API_KEY / TMDB_API_KEY from the environment and passes them through.
"""

import argparse
import datetime as _dt
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

# Reuse the sorter's normalization/matching so the test scores titles the same way the
# sorter does. Fall back to local copies if Sort_Rips can't be imported (e.g. its optional
# deps are missing on a dev box) so --help and unit-y checks still work.
try:
    from Sort_Rips import DEFAULT_EXTENSIONS, norm_title, similarity  # type: ignore
except Exception:  # pragma: no cover - import side effects only matter on panda
    from difflib import SequenceMatcher

    DEFAULT_EXTENSIONS = [
        ".mkv", ".mp4", ".avi", ".mov", ".wmv",
        ".m4v", ".mpg", ".mpeg", ".ts", ".flv",
    ]

    def norm_title(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"&", "and", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def similarity(a: str, b: str) -> float:
        a_n, b_n = norm_title(a), norm_title(b)
        if not a_n or not b_n:
            return 0.0
        return SequenceMatcher(None, a_n, b_n).ratio()


TITLE_MATCH_THRESHOLD = 0.90
VIDEO_EXTS = {e.lower() for e in DEFAULT_EXTENSIONS}

# Human-readable labels for the obfuscation pattern each index applies. Kept in
# sync with the staging functions (stage_movies / stage_tv) so the Jenkins log
# explains what each index actually tests.
RIPS_PATTERNS = {
    1: "With title hint",
    2: "No title hint",
}
TV_PATTERNS = {
    1: "Show + season known",
    2: "Show known, season hidden",
    3: "Fully blind (no show/season)",
}
EPISODE_RE = re.compile(
    r"^(?P<show>.+?)\s*-\s*S(?P<s>\d{1,2})E(?P<e>\d{1,3})\s*-\s*(?P<title>.+)\.(?P<ext>[A-Za-z0-9]+)$"
)
SXXEYY_RE = re.compile(r"\bS(?P<s>\d{1,2})E(?P<e>\d{1,3})\b", re.IGNORECASE)
MOVIE_DIR_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")


# ---------------------------- data classes ----------------------------

@dataclass
class MovieGT:
    title: str
    year: Optional[int]
    real_path: Path
    key: Tuple[int, int]


@dataclass
class EpisodeGT:
    show: str
    season: int
    episode: int
    title: str
    real_path: Path
    key: Tuple[int, int]


@dataclass
class Season:
    show: str
    season: int
    episodes: List[EpisodeGT] = field(default_factory=list)


# ---------------------------- small helpers ----------------------------

def inode_key(p: Path) -> Tuple[int, int]:
    st = p.stat()  # follows symlinks -> resolves to the real file
    return (st.st_dev, st.st_ino)


def rand_token(n: int = 10) -> str:
    # lowercase letters only -> cannot accidentally form an SxxEyy marker
    return "".join(random.choices(string.ascii_lowercase, k=n))


def squash_label(name: str) -> str:
    """Uppercase underscore label like a MakeMKV disc volume name."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return s or "TITLE"


def squash_title(name: str) -> str:
    """Run-together uppercase form: 'Men in Black' -> 'MENINBLACK'."""
    return re.sub(r"[^A-Za-z0-9]+", "", name).upper() or "MOVIE"


def acronym_label(name: str) -> str:
    """Initials + MOVIE: 'Men in Black' -> 'MIBMOVIE'."""
    words = re.findall(r"[A-Za-z0-9]+", name)
    initials = "".join(w[0] for w in words).upper()
    return f"{initials}MOVIE" if initials else "MOVIE"


def uniquify(name: str, used: set) -> str:
    candidate = name
    n = 2
    while candidate in used:
        candidate = f"{name}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den):.1f}%" if den else "n/a"


def make_link(real: Path, dest: Path, mode: str) -> str:
    """Create dest as a link to real. Returns the mode actually used.
    Never copies - multi-GB media must not be duplicated."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    real = real.resolve()
    if mode in ("auto", "hardlink"):
        try:
            os.link(real, dest)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    # symlink (absolute target so the link survives being moved)
    os.symlink(real, dest)
    return "symlink"


# ---------------------------- ground truth ----------------------------

def harvest_movies(movies_root: Path) -> List[MovieGT]:
    out: List[MovieGT] = []
    if not movies_root.is_dir():
        return out
    for folder in sorted(p for p in movies_root.iterdir() if p.is_dir()):
        vids = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
        if not vids:
            continue
        main = max(vids, key=lambda f: f.stat().st_size)
        m = MOVIE_DIR_RE.match(folder.name)
        if m:
            title = m.group("title").strip()
            year: Optional[int] = int(m.group("year"))
        else:
            title, year = folder.name.strip(), None
        try:
            key = inode_key(main)
        except OSError:
            continue
        out.append(MovieGT(title=title, year=year, real_path=main, key=key))
    return out


def harvest_shows(shows_root: Path) -> List[Season]:
    seasons: Dict[Tuple[str, int], Season] = {}
    if not shows_root.is_dir():
        return []
    for show_dir in sorted(p for p in shows_root.iterdir() if p.is_dir()):
        for season_dir in sorted(p for p in show_dir.iterdir() if p.is_dir()):
            if season_dir.name.lower() == "extras":
                continue
            for f in sorted(season_dir.iterdir()):
                # Accept any video container (the library is mostly .mp4). They are
                # staged as .mkv-named hardlinks below; ffprobe/ffmpeg detect format
                # from content, not extension, so Sort_TV processes them unchanged.
                if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
                    continue
                em = EPISODE_RE.match(f.name)
                if not em:
                    continue
                # The filename's own SxxEyy is the authoritative ground-truth label.
                season = int(em.group("s"))
                episode = int(em.group("e"))
                try:
                    key = inode_key(f)
                except OSError:
                    continue
                ep = EpisodeGT(
                    show=show_dir.name, season=season, episode=episode,
                    title=em.group("title").strip(), real_path=f, key=key,
                )
                seasons.setdefault((show_dir.name, season), Season(show_dir.name, season)).episodes.append(ep)
    result = list(seasons.values())
    for s in result:
        s.episodes.sort(key=lambda e: e.episode)
    return result


# ---------------------------- real disc labels ----------------------------
# Index 1 ("with title hint") normally mangles the real title into a synthetic
# stand-in (MENINBLACK / MIBMOVIE). When rip_manifest.json archives are present
# we instead feed Sort_Rips the *real* MakeMKV disc volume label captured at rip
# time — the exact string production sees. A manifest is written pre-sort (it has
# no final "Title (Year)"), so each is joined back to a library movie by its main
# feature's duration (which survives the Nightly_Convert re-encode), with an
# exact byte-size fast path for not-yet-converted rips. The label is attached
# verbatim; movies with no matching manifest keep the synthetic scheme.

@dataclass
class RipLabel:
    disc_title: str
    main_duration_s: Optional[float]
    main_size_bytes: Optional[int]


def _manifest_main_title(titles: List[dict]) -> Optional[dict]:
    """The feature title of a movie disc = the longest one that actually saved."""
    saved = [t for t in titles if t.get("saved")] or list(titles)
    if not saved:
        return None
    return max(saved, key=lambda t: (t.get("duration_s") or 0, t.get("size_bytes") or 0))


def load_rip_labels(manifests_dir: Optional[Path]) -> List[RipLabel]:
    out: List[RipLabel] = []
    if not manifests_dir or not manifests_dir.is_dir():
        return out
    for jf in sorted(manifests_dir.glob("*.json")):
        try:
            m = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        disc = (m.get("disc_title") or "").strip()
        if not disc:
            continue
        main = _manifest_main_title(m.get("titles") or []) or {}
        out.append(RipLabel(
            disc_title=disc,
            main_duration_s=main.get("duration_s"),
            main_size_bytes=main.get("size_bytes"),
        ))
    return out


def probe_duration_seconds(path: Path) -> Optional[float]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return float(proc.stdout.strip())
    except (ValueError, OSError):
        return None


def safe_disc_folder(disc_title: str) -> str:
    """Keep the real label verbatim except for path-illegal characters, so the
    sorter sees exactly what MakeMKV produced."""
    s = disc_title.replace("/", "_").replace("\0", "").strip().strip(".")
    return s or "DISC"


class RealLabelIndex:
    """Attaches real MakeMKV disc labels to sampled movies. A disc holds a single
    movie, so each manifest is consumed at most once per pass (the caller passes a
    ``used`` set of indices into ``self.labels``)."""

    DURATION_TOL_FLOOR_S = 10.0

    def __init__(self, labels: List[RipLabel]):
        self.labels = [l for l in labels if l.main_duration_s or l.main_size_bytes]

    def __bool__(self) -> bool:
        return bool(self.labels)

    def label_for(self, mv: MovieGT, used: set) -> Optional[str]:
        """Real disc label for this movie, or None if no manifest matches.
        ``used`` indices are excluded and the chosen one is added to it."""
        if not self.labels:
            return None
        try:
            size: Optional[int] = mv.real_path.stat().st_size
        except OSError:
            size = None

        # 1) Exact byte-size match (rip not yet re-encoded): physically unambiguous.
        if size:
            exact = [i for i, l in enumerate(self.labels)
                     if i not in used and l.main_size_bytes == size]
            if exact:
                pick = max(exact, key=lambda i: similarity(self.labels[i].disc_title, mv.title))
                used.add(pick)
                return safe_disc_folder(self.labels[pick].disc_title)

        # 2) Duration match within tolerance (survives the Nightly_Convert re-encode).
        dur = probe_duration_seconds(mv.real_path)
        if dur:
            tol = max(self.DURATION_TOL_FLOOR_S, 0.01 * dur)
            cands = [i for i, l in enumerate(self.labels)
                     if i not in used and l.main_duration_s is not None
                     and abs(l.main_duration_s - dur) <= tol]
            if cands:
                # Disambiguate duration collisions by which label best resembles
                # the movie's real title (the disc IS labeled for that movie).
                pick = max(cands, key=lambda i: (
                    similarity(self.labels[i].disc_title, mv.title),
                    -abs((self.labels[i].main_duration_s or 0) - dur),
                ))
                used.add(pick)
                return safe_disc_folder(self.labels[pick].disc_title)
        return None


# ---------------------------- sampling ----------------------------

def sample_movies(movies: List[MovieGT], limit: Optional[int], rng: random.Random) -> List[MovieGT]:
    if limit is None or limit >= len(movies):
        return list(movies)
    pool = list(movies)
    rng.shuffle(pool)
    return pool[:limit]


def sample_seasons(seasons: List[Season], limit: Optional[int], rng: random.Random) -> List[Season]:
    """Select whole seasons until we reach `limit` episodes, trimming the last
    season's highest-numbered episodes so the total is exactly `limit` (keeps
    E01.. contiguity the reconciler relies on)."""
    if limit is None:
        return list(seasons)
    pool = list(seasons)
    rng.shuffle(pool)
    chosen: List[Season] = []
    count = 0
    for s in pool:
        if count >= limit:
            break
        remaining = limit - count
        if len(s.episodes) <= remaining:
            chosen.append(s)
            count += len(s.episodes)
        else:
            trimmed = Season(s.show, s.season, s.episodes[:remaining])
            chosen.append(trimmed)
            count += remaining
    return chosen


# ---------------------------- staging ----------------------------

def discs_for_season(season: Season, eps_per_disc: int) -> List[List[EpisodeGT]]:
    if eps_per_disc and eps_per_disc > 0:
        return [season.episodes[i:i + eps_per_disc] for i in range(0, len(season.episodes), eps_per_disc)]
    return [season.episodes]


def stage_movies(movies: List[MovieGT], dest_dir: Path, index: int, mode: str,
                 rng: random.Random, label_index: Optional["RealLabelIndex"] = None
                 ) -> Tuple[Dict[Tuple[int, int], MovieGT], str, Dict[Tuple[int, int], str]]:
    """Build one obfuscated source tree for Sort_Rips. Returns key->GT map, the
    link mode used, and key->label-source ("real" | "cohortA" | "cohortB" |
    "blind"). cohortA = full run-together title (realistic disc label);
    cohortB = initials only (forces the audio-fallback path)."""
    keymap: Dict[Tuple[int, int], MovieGT] = {}
    label_sources: Dict[Tuple[int, int], str] = {}
    used_folders: set = set()
    used_discs: set = set()
    link_used = mode
    for i, mv in enumerate(movies):
        real_label = label_index.label_for(mv, used_discs) if (index == 1 and label_index) else None
        if real_label is not None:
            base = real_label
            label_sources[mv.key] = "real"
        elif index == 1:
            # Two synthetic cohorts, alternating: A = full run-together title
            # (realistic disc label, e.g. THEGOONIES); B = initials only
            # (e.g. TGMOVIE), which forces the audio-fallback path.
            if i % 2 == 0:
                base = squash_title(mv.title)
                label_sources[mv.key] = "cohortA"
            else:
                base = acronym_label(mv.title)
                label_sources[mv.key] = "cohortB"
        else:
            base = f"MOVIEFOLDER_{i + 1:03d}"
            label_sources[mv.key] = "blind"
        folder = uniquify(base, used_folders)
        ext = mv.real_path.suffix
        dest = dest_dir / folder / f"{rand_token()}{ext}"
        link_used = make_link(mv.real_path, dest, mode)
        keymap[mv.key] = mv
    return keymap, link_used, label_sources


def stage_tv(seasons: List[Season], dest_dir: Path, index: int, eps_per_disc: int,
             mode: str, rng: random.Random) -> Tuple[Dict[Tuple[int, int], EpisodeGT], str]:
    """Build one obfuscated source tree for Sort_TV (seasons split into disc folders)."""
    keymap: Dict[Tuple[int, int], EpisodeGT] = {}
    show_tokens: Dict[str, str] = {}
    used_tokens: set = set()
    used_folders: set = set()
    link_used = mode

    def token_for(show: str) -> str:
        if show not in show_tokens:
            show_tokens[show] = uniquify(squash_label(show), used_tokens)
        return show_tokens[show]

    for season in seasons:
        discs = discs_for_season(season, eps_per_disc)
        for k, disc_eps in enumerate(discs, start=1):
            if index == 1:
                folder = f"{token_for(season.show)}_SEASON{season.season}_DISC{k}"
            elif index == 2:
                folder = f"{token_for(season.show)}_DISC{k}"
            else:
                folder = rand_token(12).upper()
            folder = uniquify(folder, used_folders)
            for rank, ep in enumerate(disc_eps):
                dest = dest_dir / folder / f"{rank:03d}_{rand_token()}.mkv"
                link_used = make_link(ep.real_path, dest, mode)
                keymap[ep.key] = ep
    return keymap, link_used


# ---------------------------- run sorters ----------------------------

def child_env(threads: int) -> dict:
    """Cap each sorter's CPU-Whisper thread count so several can run concurrently
    without oversubscribing the cores. The sorters build WhisperModel with the
    default cpu_threads, which ctranslate2 derives from OMP_NUM_THREADS."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max(1, threads))
    return env


def run_subprocess(cmd: List[str], log_path: Path, env: Optional[dict]) -> int:
    """Run a sorter, capturing its output to log_path (parallel passes would
    otherwise interleave unreadably). Returns the exit code."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode


def run_sort_rips(source: Path, dest: Path, processed: Path, no_audio: bool,
                  summary_json: Path, whisper_model: Optional[str],
                  log_path: Path, env: Optional[dict],
                  transcript_cache: Optional[Path] = None) -> int:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "Sort_Rips.py"),
        "--source", str(source),
        "--dest", str(dest),
        "--processed", str(processed),
        "--summary-json", str(summary_json),
    ]
    if whisper_model:
        cmd += ["--whisper-model", whisper_model]
    if transcript_cache:
        cmd += ["--transcript-cache", str(transcript_cache)]
    if no_audio:
        cmd.append("--no-whisper-fallback")
    return run_subprocess(cmd, log_path, env)


def run_sort_tv(root: Path, dest: Path, no_audio: bool, summary_json: Path,
                whisper_model: Optional[str], log_path: Path, env: Optional[dict]) -> int:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "Sort_TV.py"),
        "--root", str(root),
        "--dest", str(dest),
        "--summary-json", str(summary_json),
    ]
    if whisper_model:
        cmd += ["--whisper-model", whisper_model]
    if no_audio:
        cmd.append("--no-audio-fallback")
    return run_subprocess(cmd, log_path, env)


# ---------------------------- transcript cache fill ----------------------------
# Sort_Rips writes the cache as it transcribes during a run. This is the harness's
# "also write it if missing" backstop: after the passes, transcribe (once) any
# sampled movie whose first-clip transcript is still absent, so the next run is
# fully cache-served. Keyed identically to Sort_Rips (size+duration+model+window).

def _load_whisper_model(model_name: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        return None
    for device, compute in (("cuda", "float16"), ("cuda", "int8"), ("cpu", "int8")):
        try:
            return WhisperModel(model_name, device=device, compute_type=compute)
        except Exception:
            continue
    return None


def fill_transcript_cache(movies: List[MovieGT], cache_dir: Optional[Path],
                          model_name: str) -> None:
    if not cache_dir:
        return
    try:
        import Sort_Rips as SR  # type: ignore
    except Exception:
        print("[CACHE] Sort_Rips unavailable; skipping transcript-cache fill.")
        return

    start = SR.WHISPER_INTERVAL_SECONDS_DEFAULT       # first clip: 5 min in
    base = SR.WHISPER_BASE_SECONDS_DEFAULT            # 60s
    todo = []
    for mv in movies:
        dur = SR.ffprobe_duration_seconds(mv.real_path)
        if dur is None or start + 1.0 >= dur:
            continue  # Sort_Rips would not take a first clip here either
        seconds = min(base, dur - start - 1.0)
        size = mv.real_path.stat().st_size
        if SR.load_cached_transcript(cache_dir, size, dur, model_name, start, seconds) is None:
            todo.append((mv, size, dur, seconds))

    if not todo:
        print(f"[CACHE] transcript cache already complete for {len(movies)} sampled movie(s).")
        return
    model = _load_whisper_model(model_name)
    if model is None:
        print(f"[CACHE] {len(todo)} transcript(s) missing but no Whisper model loaded; skipping fill.")
        return
    print(f"[CACHE] filling {len(todo)} missing transcript(s) with model '{model_name}'...")
    for mv, size, dur, seconds in todo:
        SR.get_clip_transcript(mv.real_path, size, dur, start, seconds, model, model_name, cache_dir)
    print("[CACHE] transcript-cache fill done.")


# ---------------------------- per-pass workers ----------------------------

def job_rips(idx: int, sampled: List[MovieGT], run_dir: Path, args, threads: int,
             label_index: Optional["RealLabelIndex"] = None) -> dict:
    src = run_dir / f"rips_idx{idx}"
    out = run_dir / f"out_rips_idx{idx}"
    proc = run_dir / f"proc_rips_idx{idx}"
    summ = run_dir / f"rips_idx{idx}_summary.json"
    log = run_dir / f"rips_idx{idx}.log"
    rng = random.Random(args.seed + 100 + idx)
    keymap, mode, label_sources = stage_movies(
        sampled, src, idx, args.link_mode, rng,
        label_index=label_index if idx == 1 else None)
    rc = run_sort_rips(src, out, proc, args.no_audio_fallback, summ,
                       args.whisper_model, log, child_env(threads),
                       transcript_cache=getattr(args, "transcript_cache", None))
    return {"script": "rips", "idx": str(idx),
            "result": score_rips(keymap, out, [src, proc], label_sources),
            "log": log, "mode": mode, "n": len(keymap), "rc": rc}


def job_tv(idx: int, sampled: List[Season], run_dir: Path, args, eps_per_disc: int,
           threads: int) -> dict:
    src = run_dir / f"tv_idx{idx}"
    out = run_dir / f"out_tv_idx{idx}"
    summ = run_dir / f"tv_idx{idx}_summary.json"
    log = run_dir / f"tv_idx{idx}.log"
    rng = random.Random(args.seed + 200 + idx)
    keymap, mode = stage_tv(sampled, src, idx, eps_per_disc, args.link_mode, rng)
    rc = run_sort_tv(src, out, args.no_audio_fallback, summ,
                     args.whisper_model, log, child_env(threads))
    return {"script": "tv", "idx": str(idx), "result": score_tv(keymap, out, [src]),
            "log": log, "mode": mode, "n": len(keymap), "rc": rc}


# ---------------------------- scoring ----------------------------

def _walk_video_files(root: Path):
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in VIDEO_EXTS:
                yield p


def score_rips(keymap: Dict[Tuple[int, int], MovieGT], out_dir: Path,
               leftover_dirs: List[Path],
               label_sources: Optional[Dict[Tuple[int, int], str]] = None) -> dict:
    label_sources = label_sources or {}
    # key -> identified "Title (Year)" folder name
    identified: Dict[Tuple[int, int], str] = {}
    for p in _walk_video_files(out_dir):
        try:
            key = inode_key(p)
        except OSError:
            continue
        identified[key] = p.parent.name
    leftover: set = set()
    for d in leftover_dirs:
        for p in _walk_video_files(d):
            try:
                leftover.add(inode_key(p))
            except OSError:
                continue

    items = []
    title_correct = title_year_correct = wrong = miss = unmapped = 0
    for key, gt in keymap.items():
        row = {"gt_title": gt.title, "gt_year": gt.year, "identified": None,
               "correct_title": False, "correct_year": False, "outcome": "",
               "label_source": label_sources.get(key, "")}
        if key in identified:
            folder = identified[key]
            row["identified"] = folder
            m = MOVIE_DIR_RE.match(folder)
            id_title = m.group("title").strip() if m else folder.strip()
            id_year = int(m.group("year")) if m else None
            ok_title = (norm_title(id_title) == norm_title(gt.title)
                        or similarity(id_title, gt.title) >= TITLE_MATCH_THRESHOLD)
            row["correct_title"] = ok_title
            row["correct_year"] = bool(ok_title and gt.year is not None and id_year == gt.year)
            if ok_title:
                title_correct += 1
                if row["correct_year"]:
                    title_year_correct += 1
                row["outcome"] = "correct"
            else:
                wrong += 1
                row["outcome"] = "wrong_title"
        elif key in leftover:
            miss += 1
            row["outcome"] = "unidentified"
        else:
            unmapped += 1
            row["outcome"] = "unmapped"
        items.append(row)

    return {
        "total": len(keymap),
        "title_correct": title_correct,
        "title_year_correct": title_year_correct,
        "wrong": wrong,
        "miss": miss,
        "unmapped": unmapped,
        "items": items,
    }


def score_tv(keymap: Dict[Tuple[int, int], EpisodeGT], out_dir: Path,
             leftover_dirs: List[Path]) -> dict:
    # key -> {"type": "episode"/"extras", "season":, "ep":}
    identified: Dict[Tuple[int, int], dict] = {}
    for p in _walk_video_files(out_dir):
        try:
            key = inode_key(p)
        except OSError:
            continue
        rel_parts = {part.lower() for part in p.relative_to(out_dir).parts[:-1]}
        if "extras" in rel_parts:
            identified[key] = {"type": "extras"}
            continue
        m = SXXEYY_RE.search(p.name)
        if m:
            identified[key] = {"type": "episode", "season": int(m.group("s")), "ep": int(m.group("e"))}
        else:
            identified[key] = {"type": "extras"}  # in library tree but unparseable -> treat as extra
    leftover: set = set()
    for d in leftover_dirs:
        for p in _walk_video_files(d):
            try:
                leftover.add(inode_key(p))
            except OSError:
                continue

    items = []
    episode_correct = season_correct = wrong = extras = miss = unmapped = 0
    for key, gt in keymap.items():
        row = {"show": gt.show, "season": gt.season, "episode": gt.episode,
               "gt_title": gt.title,
               "id_season": None, "id_episode": None,
               "correct_season": False, "correct_episode": False, "outcome": ""}
        info = identified.get(key)
        if info and info["type"] == "episode":
            row["id_season"] = info["season"]
            row["id_episode"] = info["ep"]
            row["correct_season"] = (info["season"] == gt.season)
            row["correct_episode"] = (info["season"] == gt.season and info["ep"] == gt.episode)
            if row["correct_season"]:
                season_correct += 1
            if row["correct_episode"]:
                episode_correct += 1
                row["outcome"] = "correct"
            else:
                wrong += 1
                row["outcome"] = "wrong_episode"
        elif info and info["type"] == "extras":
            extras += 1
            row["outcome"] = "routed_to_extras"
        elif key in leftover:
            miss += 1
            row["outcome"] = "unidentified"
        else:
            unmapped += 1
            row["outcome"] = "unmapped"
        items.append(row)

    return {
        "total": len(keymap),
        "episode_correct": episode_correct,
        "season_correct": season_correct,
        "wrong": wrong,
        "extras": extras,
        "miss": miss,
        "unmapped": unmapped,
        "items": items,
    }


# ---------------------------- report ----------------------------

def _fmt_movie_gt(row: dict) -> str:
    year = f" ({row['gt_year']})" if row.get("gt_year") else ""
    return f"{row['gt_title']}{year}"


def _fmt_episode_gt(row: dict) -> str:
    title = row.get("gt_title") or ""
    suffix = f' "{title}"' if title else ""
    return f"{row['show']} S{row['season']:02d}E{row['episode']:02d}{suffix}"


def _rips_source_breakdown(items: List[dict]) -> Dict[str, Tuple[int, int]]:
    """correct/total title accuracy grouped by label source."""
    by_src: Dict[str, List[int]] = {}
    for row in items:
        src = row.get("label_source") or "?"
        agg = by_src.setdefault(src, [0, 0])
        agg[1] += 1
        if row.get("correct_title"):
            agg[0] += 1
    return {s: (c, n) for s, (c, n) in by_src.items()}


_LABEL_SOURCE_NAMES = {
    "real": "real disc label",
    "cohortA": "Cohort A (full title)",
    "cohortB": "Cohort B (initials)",
    "blind": "blind",
    "?": "unknown",
}


def _print_rips_label_sources(items: List[dict]) -> None:
    """Break Index-1 accuracy out by label source so each cohort — real disc
    labels, Cohort A (full title), Cohort B (initials) — is reported separately
    instead of being averaged into one misleading number."""
    breakdown = _rips_source_breakdown(items)
    if not ({"real", "cohortA", "cohortB"} & set(breakdown)):
        return
    order = ["real", "cohortA", "cohortB", "blind", "?"]
    parts = [f"{_LABEL_SOURCE_NAMES.get(s, s)}: {c}/{n} ({pct(c, n)})"
             for s in order if s in breakdown
             for (c, n) in [breakdown[s]]]
    print("  by label source -> " + "   ".join(parts))


def _print_rips_failures(items: List[dict]) -> None:
    buckets = {"wrong_title": [], "unidentified": [], "unmapped": []}
    for row in items:
        if row["outcome"] == "correct":
            continue
        if row["outcome"] in buckets:
            buckets[row["outcome"]].append(row)
    labels = {
        "wrong_title": "wrong title",
        "unidentified": "unidentified (left in source / processed)",
        "unmapped": "unmapped (vanished — neither in output nor source)",
    }
    for outcome, label in labels.items():
        rows = buckets[outcome]
        if not rows:
            continue
        print(f"  {label}:")
        for row in rows:
            got = row.get("identified") or "(no folder)"
            tag = {"real": " [real-label]", "cohortA": " [cohort A]",
                   "cohortB": " [cohort B]"}.get(row.get("label_source"), "")
            print(f"    - {_fmt_movie_gt(row)}{tag}  ->  got: {got}")


def _print_tv_failures(items: List[dict]) -> None:
    buckets = {
        "wrong_episode": [],
        "routed_to_extras": [],
        "unidentified": [],
        "unmapped": [],
    }
    for row in items:
        if row["outcome"] == "correct":
            continue
        if row["outcome"] in buckets:
            buckets[row["outcome"]].append(row)
    labels = {
        "wrong_episode": "wrong episode / season",
        "routed_to_extras": "routed to extras (sorter saw episode as non-episode)",
        "unidentified": "unidentified (left in source — TMDB verify failed or no usable evidence)",
        "unmapped": "unmapped (vanished — neither in output nor source)",
    }
    for outcome, label in labels.items():
        rows = buckets[outcome]
        if not rows:
            continue
        print(f"  {label}:")
        for row in rows:
            if outcome == "wrong_episode":
                got = f"got S{row['id_season']:02d}E{row['id_episode']:02d}"
            else:
                got = ""
            line = f"    - {_fmt_episode_gt(row)}"
            if got:
                line += f"  ->  {got}"
            print(line)


def print_report(results: dict) -> None:
    print("\n" + "=" * 64)
    print("ACCURACY REPORT")
    print("Each suite/index reflects a different obfuscation pattern. The")
    print("real media bytes are unchanged; only folder/file names vary.")
    print("=" * 64)
    rips = results.get("rips", {})
    for idx in sorted(rips):
        r = rips[idx]
        t = r["total"]
        pattern = RIPS_PATTERNS.get(int(idx), "unknown pattern")
        print(f"Sort_Rips Index {idx} ({pattern}):")
        print(f"  title {r['title_correct']}/{t} ({pct(r['title_correct'], t)})  "
              f"title+year {r['title_year_correct']}/{t}  wrong {r['wrong']}  miss {r['miss']}"
              + (f"  unmapped {r['unmapped']}" if r['unmapped'] else ""))
        _print_rips_label_sources(r.get("items", []))
        _print_rips_failures(r.get("items", []))
    tv = results.get("tv", {})
    for idx in sorted(tv):
        r = tv[idx]
        t = r["total"]
        pattern = TV_PATTERNS.get(int(idx), "unknown pattern")
        print(f"Sort_TV   Index {idx} ({pattern}):")
        print(f"  episode {r['episode_correct']}/{t} ({pct(r['episode_correct'], t)})  "
              f"season {r['season_correct']}/{t}  wrong {r['wrong']}  extras {r['extras']}  miss {r['miss']}"
              + (f"  unmapped {r['unmapped']}" if r['unmapped'] else ""))
        _print_tv_failures(r.get("items", []))
    print("=" * 64 + "\n")


# ---------------------------- main ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media-root", type=Path, default=Path("/mnt/media/Media"),
                    help="Library root; movies=<root>/Movies, shows=<root>/Shows.")
    ap.add_argument("--movies-root", type=Path, default=None, help="Override movies dir.")
    ap.add_argument("--shows-root", type=Path, default=None, help="Override shows dir.")
    ap.add_argument("--staging-root", type=Path, default=Path("/mnt/media/.accuracy_test"),
                    help="Scratch dir for link staging - MUST be on the same volume as the media.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run this many movies AND this many episodes. Omit for ALL.")
    ap.add_argument("--episodes-per-disc", type=int, default=4,
                    help="Split each season into discs of this size (default 4).")
    ap.add_argument("--no-disc-split", action="store_true", help="Keep whole seasons as one disc.")
    ap.add_argument("--only", choices=["rips", "tv", "both"], default="both")
    ap.add_argument("--indexes", default="1,2,3", help="Comma list (rips uses 1,2; tv uses 1,2,3).")
    ap.add_argument("--link-mode", choices=["auto", "hardlink", "symlink"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument("--no-audio-fallback", action="store_true",
                    help="Disable Whisper/audio fallback in the sorters (faster, less faithful).")
    ap.add_argument("--jobs", type=int, default=None,
                    help="How many sorter passes to run concurrently (default: min(#passes, cores)). "
                         "Each pass is capped to cores/jobs CPU threads.")
    ap.add_argument("--whisper-model", default=None,
                    help="Override the Whisper model the sorters use (e.g. tiny, base, small). "
                         "Smaller = much faster CPU transcription, slightly lower ID accuracy.")
    ap.add_argument("--fast", action="store_true",
                    help="Speed preset for large sweeps: uses the 'base' Whisper model unless "
                         "--whisper-model is given. Opt-in; off = production-faithful models.")
    ap.add_argument("--rip-manifests", type=Path, default=Path("/mnt/media/rip_manifests"),
                    help="Dir of rip_manifest.json archives. Index 1 ('with title hint') feeds "
                         "Sort_Rips the real MakeMKV disc label from a matching manifest (joined "
                         "by main-feature duration/size) instead of a synthetic stand-in when one "
                         "is found. Movies with no match keep the synthetic label.")
    ap.add_argument("--no-real-labels", action="store_true",
                    help="Ignore rip manifests; always use the synthetic Index-1 labels.")
    ap.add_argument("--transcript-cache", type=Path, default=Path("/mnt/media/.transcript_cache"),
                    help="Shared transcript cache dir passed to Sort_Rips. On by default so repeat "
                         "runs reuse Whisper transcripts instead of re-transcribing. After each run "
                         "the harness fills any still-missing entries for the sampled movies.")
    ap.add_argument("--no-transcript-cache", action="store_true",
                    help="Disable the transcript cache (always re-transcribe).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    movies_root = args.movies_root or (args.media_root / "Movies")
    shows_root = args.shows_root or (args.media_root / "Shows")
    staging_root = args.staging_root.resolve()
    eps_per_disc = 0 if args.no_disc_split else args.episodes_per_disc

    # Safety: staging must not live inside the library.
    for lib in (movies_root.resolve(), shows_root.resolve()):
        if staging_root == lib or lib in staging_root.parents:
            print(f"ERROR: --staging-root ({staging_root}) is inside the library ({lib}). Refusing.")
            return 2

    want_indexes = {int(x) for x in str(args.indexes).split(",") if x.strip()}
    do_rips = args.only in ("rips", "both")
    do_tv = args.only in ("tv", "both")
    limit = args.limit if (args.limit is not None and args.limit > 0) else None

    if args.fast and not args.whisper_model:
        args.whisper_model = "base"

    # Transcript cache: on by default; job_rips reads it off args and passes it to
    # Sort_Rips. Disabled by --no-transcript-cache.
    args.transcript_cache = None if args.no_transcript_cache else args.transcript_cache

    run_id = "run_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = staging_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"rips": {}, "tv": {}}

    try:
        # -- harvest + sample once (shared across all index passes) ----
        rips_indexes = sorted(i for i in want_indexes if i in (1, 2)) if do_rips else []
        tv_indexes = sorted(i for i in want_indexes if i in (1, 2, 3)) if do_tv else []

        sampled_movies: List[MovieGT] = []
        sampled_seasons: List[Season] = []
        label_index: Optional[RealLabelIndex] = None
        if rips_indexes:
            movies = harvest_movies(movies_root)
            sampled_movies = sample_movies(movies, limit, rng)
            print(f"[RIPS] {len(sampled_movies)}/{len(movies)} movies sampled.")
            if 1 in rips_indexes and not args.no_real_labels:
                label_index = RealLabelIndex(load_rip_labels(args.rip_manifests))
                if label_index:
                    print(f"[RIPS] Index 1 will use real disc labels where available "
                          f"({len(label_index.labels)} manifest(s) in {args.rip_manifests}).")
                else:
                    print(f"[RIPS] no rip manifests in {args.rip_manifests}; "
                          f"Index 1 uses synthetic labels.")
        if tv_indexes:
            seasons = harvest_shows(shows_root)
            sampled_seasons = sample_seasons(seasons, limit, rng)
            n_eps = sum(len(s.episodes) for s in sampled_seasons)
            n_all = sum(len(s.episodes) for s in seasons)
            print(f"[TV]   {n_eps}/{n_all} episodes across {len(sampled_seasons)} season(s) sampled.")

        # -- plan the parallel passes ----------------------------------
        cores = os.cpu_count() or 8
        n_passes = len(rips_indexes) + len(tv_indexes)
        # Default to ~2 CPU threads per concurrent pass: CPU Whisper scales poorly
        # past a couple of threads, so more passes x fewer threads beats few passes
        # x many threads. --jobs overrides.
        default_jobs = max(1, min(n_passes, cores // 2)) if n_passes else 1
        workers = max(1, min(args.jobs or default_jobs, n_passes)) if n_passes else 1
        threads = max(1, cores // workers)
        model = args.whisper_model or "sorter defaults"

        print(f"Staging: {run_dir}")
        print(f"Media  : movies={movies_root}  shows={shows_root}")
        print(f"Limit  : {'ALL' if limit is None else limit}   eps/disc: {eps_per_disc or 'whole-season'}   "
              f"link-mode: {args.link_mode}   audio: {'OFF' if args.no_audio_fallback else 'ON'}")
        print(f"Run    : {n_passes} sorter pass(es), {workers} concurrent x {threads} CPU thread(s), "
              f"whisper={model}")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for idx in rips_indexes:
                futures[ex.submit(job_rips, idx, sampled_movies, run_dir, args, threads, label_index)] = ("rips", idx)
            for idx in tv_indexes:
                futures[ex.submit(job_tv, idx, sampled_seasons, run_dir, args, eps_per_disc, threads)] = ("tv", idx)
            print(f"\nLaunched {len(futures)} pass(es); logs print below as each finishes.\n")
            for fut in as_completed(futures):
                script, idx = futures[fut]
                patterns = RIPS_PATTERNS if script == "rips" else TV_PATTERNS
                header = f"{'Sort_Rips' if script == 'rips' else 'Sort_TV'} Index {idx} ({patterns.get(idx, '?')})"
                try:
                    jr = fut.result()
                except Exception as e:
                    print("#" * 64)
                    print(f"# {header} - FAILED: {type(e).__name__}: {e}")
                    print("#" * 64)
                    continue
                results[script][str(idx)] = jr["result"]
                print("#" * 64)
                print(f"# {header}  staged {jr['n']}  link={jr['mode']}  exit={jr['rc']}")
                print("#" * 64)
                try:
                    print(jr["log"].read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass

        print_report(results)

        # Backstop: ensure every sampled movie has a cached first-clip transcript
        # so the next run is fully cache-served (Sort_Rips already wrote most of
        # them during the passes). Skipped when audio is off (no transcripts).
        if rips_indexes and args.transcript_cache and not args.no_audio_fallback:
            fill_transcript_cache(sampled_movies, args.transcript_cache,
                                  args.whisper_model or "base")

        # Stamp the human-readable pattern label into each index's results so
        # downstream consumers (the Discord notify script) don't need to know the
        # index numbering.
        for _idx, _r in results.get("rips", {}).items():
            _r["label"] = RIPS_PATTERNS.get(int(_idx), f"index {_idx}")
        for _idx, _r in results.get("tv", {}).items():
            _r["label"] = TV_PATTERNS.get(int(_idx), f"index {_idx}")

        summary = {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "limit": limit,
            "episodes_per_disc": eps_per_disc,
            "link_mode": args.link_mode,
            "seed": args.seed,
            "results": results,
        }
        if args.summary_json:
            try:
                args.summary_json.write_text(json.dumps(summary, indent=2))
                print(f"Summary written to {args.summary_json}")
            except Exception as e:
                print(f"WARN: could not write summary JSON: {e}")
    finally:
        if args.keep_staging:
            print(f"[KEEP] staging left at {run_dir}")
        else:
            # Removes only links + sorter scratch (sort_hints.json, empty dirs).
            # Link targets - the real library - are never touched by rmtree of links.
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"[CLEAN] removed {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
