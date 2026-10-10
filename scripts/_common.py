"""Make ``contamination_audit`` importable when running a script directly.

Importing this module puts ``src/`` on the path, so the numbered entry points
work with ``python scripts/02_results.py`` without installing the package first.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
