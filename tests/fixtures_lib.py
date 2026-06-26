"""Shared loader for the sort-accuracy ground-truth fixtures.

Fixtures live in fixtures/sort_groundtruth/*.json (schema in that dir's README).
Only `verified: true` fixtures count toward accuracy numbers.
"""

import json
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "fixtures" / "sort_groundtruth"


def load_fixtures() -> List[dict]:
    """All fixtures with a stable `_name` (the filename stem) attached."""
    out = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_name"] = path.stem
        out.append(data)
    return out


def verified_fixtures() -> List[dict]:
    return [f for f in load_fixtures() if f.get("verified") is True]
