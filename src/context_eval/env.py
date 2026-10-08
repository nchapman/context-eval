"""Minimal .env loader (hand-rolled; no python-dotenv dependency).

Real environment variables always win over .env values, so CI and shells
override the file. Entry points call ``load_env()`` before constructing
clients; library code keeps reading ``os.environ`` only.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    """Populate unset env vars from a KEY=VALUE file. Returns what was loaded."""
    file = Path(path)
    if not file.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
