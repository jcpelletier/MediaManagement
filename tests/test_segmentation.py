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


# ---------------------------------------------------------------------------
# Split decision: is a long title a Play-All, or one long episode?
# ---------------------------------------------------------------------------

from Segment_Disc import decide_split_target  # noqa: E402

# Outlander-S6-like runtimes in minutes (approximated from the disc title
# lengths, not copied from TMDB): hour-plus drama episodes.
OUTLANDER_S6 = [80, 72, 63, 58, 60, 71, 63, 63]
POKEMON_S21 = [22] * 8


@pytest.mark.parametrize("minutes", [57.4, 59.4, 62.7, 63.0, 71.0, 72.0, 80.4])
def test_long_drama_episodes_are_not_split(minutes):
    """Regression: Outlander S6 titles (57-80 min) were cut into ~22 min pieces
    by the old fixed 60-min threshold, then routed to Extras by Sort_TV."""
    target, reason = decide_split_target(minutes * 60, OUTLANDER_S6)
    assert target is None, reason


def test_long_drama_play_all_split_at_median_runtime():
    """A real Play-All of two Outlander episodes is split at ~hour length, not 22."""
    target, _ = decide_split_target(152.4 * 60, OUTLANDER_S6)
    assert target == pytest.approx(63 * 60)
    chapters = [(0.0, 3780.0), (3780.0, 4560.0), (4560.0, 9144.0)]
    segments = plan_episode_segments(chapters, target)
    assert len(segments) == 2


def test_pokemon_play_all_still_split_with_known_runtimes():
    target, _ = decide_split_target(170.4 * 60, POKEMON_S21)
    assert target == pytest.approx(22 * 60)


def test_short_show_double_feature_split_below_old_threshold():
    """Known 22-min show: a 44-min two-episode title is now split too."""
    target, _ = decide_split_target(44.5 * 60, POKEMON_S21)
    assert target == pytest.approx(22 * 60)


def test_double_length_special_within_band_not_split():
    """A season whose TMDB data includes a 44-min special keeps it whole."""
    target, _ = decide_split_target(44.0 * 60, [22, 22, 44, 22])
    assert target is None


def test_unknown_runtimes_conservative_threshold():
    """Without TMDB data only titles Sort_TV would reject (>100 min) are split."""
    assert decide_split_target(80.4 * 60, [])[0] is None
    assert decide_split_target(99.0 * 60, [])[0] is None
    target, _ = decide_split_target(170.4 * 60, [])
    assert target == pytest.approx(22 * 60)


def test_episode_minutes_override():
    target, _ = decide_split_target(170.4 * 60, POKEMON_S21, episode_minutes=21.3)
    assert target == pytest.approx(21.3 * 60)
    target, _ = decide_split_target(170.4 * 60, [], episode_minutes=21.3)
    assert target == pytest.approx(21.3 * 60)


def test_missing_runtimes_ignored():
    """TMDB episodes without a runtime (None/0) do not drag the band down."""
    target, _ = decide_split_target(72.0 * 60, [None, 0, 72, 63])
    assert target is None


# ---------------------------------------------------------------------------
# Folder -> show/season -> TMDB runtimes (TMDB client mocked)
# ---------------------------------------------------------------------------

class _FakeEp:
    def __init__(self, runtime):
        self.runtime = runtime


class _FakeShow:
    def __init__(self, tv_id, name):
        self.tv_id, self.name = tv_id, name


class _FakeTmdb:
    """Matches only the exact show names in ``shows``; records queries."""
    queries = []

    def __init__(self, api_key, timeout_s=10.0):
        pass

    def find_tv(self, name):
        _FakeTmdb.queries.append(name)
        return {"outlander": _FakeShow(56570, "Outlander")}.get(name.lower())

    def get_season_episodes(self, tv_id, season):
        return [_FakeEp(r) for r in ({6: OUTLANDER_S6}.get(season, []) + [None])]


@pytest.fixture
def fake_tmdb(monkeypatch):
    import Sort_TV
    import Segment_Disc
    monkeypatch.setenv("TMDB_API_KEY", "test")
    monkeypatch.setattr(Sort_TV, "TmdbClient", _FakeTmdb)
    _FakeTmdb.queries = []
    return Segment_Disc


def test_lookup_strips_year_from_folder_name(fake_tmdb, tmp_path):
    folder = tmp_path / "OUTLANDER_2014_SEASON_6_DISC_1"
    folder.mkdir()
    label, runtimes = fake_tmdb.lookup_season_runtimes(folder)
    assert label == "Outlander S06"
    assert runtimes == OUTLANDER_S6
    assert _FakeTmdb.queries == ["OUTLANDER 2014", "OUTLANDER"]


def test_lookup_prefers_sort_hints(fake_tmdb, tmp_path):
    folder = tmp_path / "SOME_BOX_SET_D1"
    folder.mkdir()
    (folder / "sort_hints.json").write_text(json.dumps({"show": "Outlander", "season": 6}))
    label, runtimes = fake_tmdb.lookup_season_runtimes(folder)
    assert label == "Outlander S06" and runtimes == OUTLANDER_S6


def test_lookup_without_api_key_falls_back(monkeypatch, tmp_path):
    import Segment_Disc
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    folder = tmp_path / "OUTLANDER_2014_SEASON_6_DISC_1"
    folder.mkdir()
    assert Segment_Disc.lookup_season_runtimes(folder) == ("", [])
