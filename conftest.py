"""Put the repo root on sys.path so tests can import the top-level sort scripts
(Sort_Rips, Sort_TV) directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
