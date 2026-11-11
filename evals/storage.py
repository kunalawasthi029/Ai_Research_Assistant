"""Atomic JSON stage cache. No prompts, credentials or raw model messages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class StageCache:
    """Repetitions have distinct keys; completed stages alone are reusable."""

    def __init__(self, root: Path, namespace: str):
        self.root = root / namespace

    def path(self, key: Any) -> Path:
        return self.root / f"{digest(key)}.json"

    def get(self, key: Any) -> dict[str, Any] | None:
        try:
            value = json.loads(self.path(key).read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def put(self, key: Any, value: dict[str, Any]) -> None:
        write_json(self.path(key), value)
