"""Tier 1: deterministic movie-vs-TV routing accuracy.

Replays each verified fixture's pre-sort title features (name + per-title size +
duration) through the live routing decision in Sort_Rips.classify_disc_routing
and scores it against expected_routing. No media files, no ffprobe, no API calls,
so this runs in CI on every change; it is the layer that catches misroutes like
the Paw Patrol compilation disc.
"""

import pytest

from Sort_Rips import classify_disc_routing
from fixtures_lib import verified_fixtures


def _route(fixture: dict) -> str:
    sizes = [t.get("size_bytes") or 0 for t in fixture["titles"]]
    durations = [t.get("duration_s") for t in fixture["titles"]]
    is_tv, _ = classify_disc_routing(fixture["disc_title"], sizes, durations)
    return "tv" if is_tv else "movie"


_routing_fixtures = [
    f for f in verified_fixtures()
    if f.get("expected_routing") in ("tv", "movie")
]


@pytest.mark.skipif(not _routing_fixtures, reason="no verified routing fixtures yet")
@pytest.mark.parametrize("fixture", _routing_fixtures, ids=lambda f: f["_name"])
def test_routing_matches_ground_truth(fixture):
    predicted = _route(fixture)
    assert predicted == fixture["expected_routing"], (
        f"{fixture['_name']}: routed as {predicted!r}, "
        f"expected {fixture['expected_routing']!r}"
    )


def test_master_routing_accuracy(capsys):
    """Headline number: % of verified discs routed correctly. Asserts a clean
    100% so any regression turns CI red, and prints the score + every miss."""
    if not _routing_fixtures:
        pytest.skip("no verified routing fixtures yet")

    misses = []
    for f in _routing_fixtures:
        predicted = _route(f)
        if predicted != f["expected_routing"]:
            misses.append((f["_name"], predicted, f["expected_routing"]))

    total = len(_routing_fixtures)
    correct = total - len(misses)
    accuracy = correct / total

    with capsys.disabled():
        print(f"\n[routing accuracy] {correct}/{total} = {accuracy:.1%}")
        for name, got, exp in misses:
            print(f"  MISS {name}: got {got!r}, expected {exp!r}")

    assert not misses, f"{len(misses)} routing miss(es): {misses}"
