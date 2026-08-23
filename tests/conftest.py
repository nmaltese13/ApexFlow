"""Shared pytest fixtures + path setup so tests can `import apexflow` cleanly."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("APEXFLOW_ATLAS_DISABLE", "1")
os.environ.setdefault("APEXFLOW_BRIEFING_DISABLE", "1")
