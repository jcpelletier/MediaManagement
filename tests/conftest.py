"""Tier-2 identification tests are slow (real media + TMDB/DeepSeek) and are
skipped unless explicitly enabled with --run-identification."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-identification",
        action="store_true",
        default=False,
        help="Run the slow tier-2 identification tests (need media files + API keys).",
    )
    parser.addoption(
        "--media-root",
        action="store",
        default=None,
        help="Root of the media library on disk (e.g. /mnt/media/Media) for tier-2 tests.",
    )
    parser.addoption(
        "--id-accuracy-floor",
        action="store",
        default="0.0",
        help="Fail tier-2 identification if accuracy drops below this fraction (default: 0.0, informational).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-identification"):
        return
    skip = pytest.mark.skip(reason="needs --run-identification (media files + API keys)")
    for item in items:
        if "identification" in item.keywords:
            item.add_marker(skip)
