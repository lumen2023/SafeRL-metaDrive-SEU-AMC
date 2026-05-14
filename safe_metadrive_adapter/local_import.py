"""Utilities for consistently importing the vendored local MetaDrive tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def prefer_local_metadrive() -> None:
    """Put ``./metadrive`` before site-packages and clear stale imports.

    The project vendors a modified MetaDrive tree. This helper makes every
    script resolve ``import metadrive`` to that local tree first, even if a
    different MetaDrive version was already imported by the current process.
    """

    local_root = repo_root() / "metadrive"
    package_dir = local_root / "metadrive"
    if not package_dir.is_dir():
        return

    local_root_str = str(local_root)
    if local_root_str in sys.path:
        sys.path.remove(local_root_str)
    sys.path.insert(0, local_root_str)

    expected_init = os.path.abspath(str(package_dir / "__init__.py"))
    cached = sys.modules.get("metadrive")
    cached_file = getattr(cached, "__file__", None) if cached is not None else None
    if cached is not None and os.path.abspath(str(cached_file or "")) != expected_init:
        for module_name in [name for name in sys.modules if name == "metadrive" or name.startswith("metadrive.")]:
            del sys.modules[module_name]


__all__ = ["prefer_local_metadrive", "repo_root"]
