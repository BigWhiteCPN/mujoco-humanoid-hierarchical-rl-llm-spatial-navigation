"""Runtime path setup for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
