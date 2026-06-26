"""Deterministic tests for chapter-based disc segmentation (Segment_Disc).

Runs the pure boundary planner against cached real chapter data (no media), so
it is fast and runs on every commit. This is the "test against Pokemon" for the
Play-All disc case: confirm a concatenated disc is cut into the right episodes
on the right chapter boundaries.
"""

import json
from pathlib import Path

import pytest

from Segment_Disc import plan_episode_segments

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "segmentation"


def _load():
    out = []
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("verified") is True:
            d["_name"] = p.stem
            out.append(d)
    return out


_fixtures = _load()


@pytest.mark.skipif(not _fixtures, reason="no segmentation fixtures yet")
@pytest.mark.parametrize("fx", _fixtures, ids=lambda f: f["_name"])
def test_segments_match_expected_boundaries(fx):
    chapters = [tuple(c) for c in fx["chapters"]]
    segments = plan_episode_segments(chapters, fx["expected_episode_minutes"] * 60.0)

    # Right number of episodes.
    assert len(segments) == fx["expected_episodes"], (
        f"{fx['_name']}: got {len(segments)} segments, expected {fx['expected_episodes']}"
    )

    # Boundaries land where expected (within 1s; cuts are exact chapter marks).
    got_bounds = [round(segments[0][0], 3)] + [round(s[1], 3) for s in segments]
    for got, exp in zip(got_bounds, fx["expected_boundaries_s"]):
        assert abs(got - exp) <= 1.0, f"{fx['_name']}: boundary {got} != expected {exp}"

    # Every cut is an actual chapter boundary (no mid-chapter cuts).
    chapter_marks = {round(c[0], 3) for c in chapters} | {round(c[1], 3) for c in chapters}
    for s, e in segments:
        assert round(s, 3) in chapter_marks and round(e, 3) in chapter_marks, (
            f"{fx['_name']}: segment ({s}, {e}) not on chapter boundaries"
        )

    # Episodes are all close to the expected length (none merged or split).
    target = fx["expected_episode_minutes"]
    for s, e in segments:
        assert abs((e - s) / 60.0 - target) <= 1.5, (
            f"{fx['_name']}: segment length {(e - s) / 60.0:.2f} min off target {target}"
        )


def test_short_file_not_split():
    """A single short title (already episode-sized) is returned whole."""
    chapters = [(0.0, 600.0), (600.0, 1320.0)]  # 22 min, 2 act chapters
    segments = plan_episode_segments(chapters, 22 * 60.0)
    assert segments == [(0.0, 1320.0)]
