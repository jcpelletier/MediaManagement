"""TMDB type-aware verification (issue #25).

Covers the Outlander-style regression: a confident TV-show guess must never be
"corrected" into a same-named film of a different year, and TMDB's TV index is
consulted before the folder is skipped or misfiled.
"""

from pathlib import Path

import Sort_Rips
from Sort_Rips import (
    FolderGuess,
    TmdbMovie,
    TmdbTv,
    TmdbMovieClient,
    TMDB_TV_DEFER,
    verify_movie_with_tmdb,
)

MIN_MATCH = 0.78


class FakeTmdbClient:
    """Stands in for TmdbMovieClient; returns canned search results, no network."""

    def __init__(self, movie=None, tv=None):
        self.movie = movie
        self.tv = tv
        self.movie_calls = []
        self.tv_calls = []

    def search_movie(self, title, year=None):
        self.movie_calls.append((title, year))
        return self.movie

    def search_tv(self, title, year=None):
        self.tv_calls.append((title, year))
        return self.tv

    def alternative_titles(self, movie_id):
        return []


class FakeLlmClient:
    """Stands in for the LLM client; returns a canned guess for every call."""

    def __init__(self, guess):
        self.guess = guess

    def parse(self, **kwargs):
        class D:
            pass
        d = D()
        d.title = self.guess.title
        d.year = self.guess.year
        d.confidence = self.guess.confidence
        return d


def _outlander_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "Outlander S06 Disc"
    folder.mkdir()
    (folder / "video.mkv").write_bytes(b"dummy")
    srt = folder / "video.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:04,000\nOutlander\n", encoding="utf-8")
    return folder


def test_outlander_case_defers_to_sort_tv(tmp_path, capsys):
    # The subtitle pass identified 'Outlander' (2014) at conf=0.99. The movie
    # index only has the 2008 film; the TV index has the series. process_folder
    # must return the existing TV defer reason, leave the folder in place, and
    # write nothing under the movie root.
    folder = _outlander_folder(tmp_path)
    dest_root = tmp_path / "Movies"
    dest_root.mkdir()
    guess = FolderGuess(title="Outlander", year=2014, confidence=0.99)
    client = FakeTmdbClient(
        movie=TmdbMovie(movie_id=12000, title="Outlander", year=2008),
        tv=TmdbTv(tv_id=2116, name="Outlander", first_air_year=2014),
    )

    moved, reason = Sort_Rips.process_folder(
        folder=folder,
        extensions=[".mkv"],
        llm_client=FakeLlmClient(guess),
        min_confidence=0.6,
        dest_root=dest_root,
        overwrite=False,
        dry_run=False,
        tmdb_client=client,
        tmdb_min_title_match=MIN_MATCH,
    )

    assert moved is None
    assert reason == Sort_Rips.TV_DISC_DEFER_REASON
    # Folder is left in place for Sort_TV; nothing was filed under the movie root.
    assert folder.exists()
    assert list(dest_root.iterdir()) == []
    out = capsys.readouterr().out
    assert out.count("TMDB identified it as a TV series:") == 1
    assert "Outlander" in out and "2014" in out
    assert "could not confirm" not in out


def test_outlander_verification_returns_tv_defer_sentinel(capsys):
    # At the verify level the Outlander case yields the TV-defer sentinel.
    guess = FolderGuess(title="Outlander", year=2014, confidence=0.99)
    client = FakeTmdbClient(
        movie=TmdbMovie(movie_id=12000, title="Outlander", year=2008),
        tv=TmdbTv(tv_id=2116, name="Outlander", first_air_year=2014),
    )

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is TMDB_TV_DEFER
    out = capsys.readouterr().out
    assert "TMDB identified it as a TV series:" in out
    assert "2014" in out
    assert "could not confirm" not in out


def test_confident_guess_with_different_year_movie_hit_is_skipped(capsys):
    # Confident guess, movie index only knows a different-year title, and the
    # TV index has nothing either: skip, never rename under the movie hit.
    guess = FolderGuess(title="Outlander", year=2014, confidence=0.99)
    client = FakeTmdbClient(movie=TmdbMovie(movie_id=12000, title="Outlander",
                                            year=2008), tv=None)

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is None
    out = capsys.readouterr().out
    # Both sides of the rejected correction are named.
    assert "Outlander (2014) conf=0.99" in out
    assert "not accepting Outlander (2008)" in out


def test_confident_guess_accepts_same_year_movie_hit():
    # Ordinary movie: same-title, same-year hit is confirmed as before, and
    # the TV index is never consulted.
    guess = FolderGuess(title="Dune", year=2021, confidence=0.95)
    client = FakeTmdbClient(movie=TmdbMovie(
        movie_id=414906, title="Dune: Part One", year=2021, original_title="Dune"))

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is not None and result is not TMDB_TV_DEFER
    assert result.title == "Dune"  # best-known title for the guess is preserved
    assert result.year == 2021
    assert result.confidence == 0.95
    assert client.tv_calls == []
def test_same_year_canonical_title_correction_still_applies(capsys):
    # A same-year canonical-title correction (e.g. a regional title) still
    # applies for a confident guess.
    guess = FolderGuess(title="Warriors of the Wind", year=1984, confidence=0.92)
    client = FakeTmdbClient(movie=TmdbMovie(
        movie_id=632, title="Nausicaä of the Valley of the Wind", year=1984,
        original_title="風のとこやけ娑婆鎮"))

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is not None and result is not TMDB_TV_DEFER
    assert result.year == 1984
    assert "Corrected" in capsys.readouterr().out


def test_both_searches_match_same_title_year_files_as_movie(capsys):
    # When the movie index confirms the guess (same title+year), the folder is
    # filed as a movie and no TV defer line is logged.
    guess = FolderGuess(title="Outlander", year=2008, confidence=0.99)
    client = FakeTmdbClient(
        movie=TmdbMovie(movie_id=12000, title="Outlander", year=2008),
        tv=TmdbTv(tv_id=2116, name="Outlander", first_air_year=2008),
    )

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result == FolderGuess(title="Outlander", year=2008, confidence=0.99)
    out = capsys.readouterr().out
    assert "TV series" not in out
    assert client.tv_calls == []  # movie search decided first


def test_low_confidence_guess_still_accepts_different_year_correction():
    # Below 0.90 the year-correction behaviour is unchanged.
    guess = FolderGuess(title="Outlander", year=2014, confidence=0.89)
    client = FakeTmdbClient(movie=TmdbMovie(movie_id=12000, title="Outlander",
                                            year=2008), tv=None)

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result == FolderGuess(title="Outlander", year=2008, confidence=0.89)


def test_confidence_boundary_090_rejects_different_year(capsys):
    # Exactly 0.90 is already confident: a different-year hit is not accepted.
    guess = FolderGuess(title="Outlander", year=2014, confidence=0.90)
    client = FakeTmdbClient(movie=TmdbMovie(movie_id=12000, title="Outlander",
                                            year=2008), tv=None)

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is None
    assert "not accepting Outlander (2008)" in capsys.readouterr().out


def test_title_match_is_case_insensitive(capsys):
    # A guess of 'OUTLANDER' still matches TMDB's 'Outlander' in both indexes.
    guess = FolderGuess(title="OUTLANDER", year=2014, confidence=0.99)
    client = FakeTmdbClient(
        movie=TmdbMovie(movie_id=12000, title="Outlander", year=2008),
        tv=TmdbTv(tv_id=2116, name="Outlander", first_air_year=2014),
    )

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is TMDB_TV_DEFER
    out = capsys.readouterr().out
    assert "TMDB identified it as a TV series:" in out
    assert "2014" in out


def test_no_results_anywhere_is_skipped(capsys):
    # Neither search confirms the guess: skipped as today.
    guess = FolderGuess(title="Slightly Misheard Title", year=2019, confidence=0.95)
    client = FakeTmdbClient(movie=None, tv=None)

    result = verify_movie_with_tmdb(client, guess, MIN_MATCH)

    assert result is None


def test_real_client_has_tv_search():
    # The real client must expose the TV search surface used by verification.
    client = TmdbMovieClient(api_key="dummy")
    assert hasattr(client, "search_tv")
    import dataclasses
    fields = {f.name for f in dataclasses.fields(TmdbTv)}
    assert {"tv_id", "name", "first_air_year"} <= fields
