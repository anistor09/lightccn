"""Shared test fixtures."""

import sys
from pathlib import Path

# Ensure src is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
