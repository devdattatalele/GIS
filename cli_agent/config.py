"""Utility functions: project root discovery, gh CLI check."""

import shutil
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root (parent of cli_agent/)."""
    return Path(__file__).resolve().parent.parent


def ensure_gh_cli_available() -> bool:
    """Check if the GitHub CLI (gh) is installed and on PATH."""
    return shutil.which("gh") is not None


def setup_sys_path():
    """Add src/ and project root to sys.path so service imports work."""
    root = get_project_root()
    src_dir = str(root / "src")
    root_str = str(root)
    for p in (src_dir, root_str):
        if p not in sys.path:
            sys.path.insert(0, p)
