"""Frozen dataset and reproducibility fingerprint (including prompts and implementation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evals.storage import digest

ROOT = Path(__file__).resolve().parents[1]


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^q\d+$")
    query: str = Field(min_length=3, max_length=4000)
    expected_topics: list[str] = Field(min_length=1)
    category: str
    difficulty: str


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    # JSON is a YAML 1.2 subset: no executable tags or extra runtime dependency.
    data = json.loads((path or ROOT / "evals/queries.yaml").read_text(encoding="utf-8"))
    if data.get("version") != 1 or len(data.get("queries", [])) < 20:
        raise ValueError("Dataset v1 requires at least 20 queries")
    queries = [Query.model_validate(q).model_dump() for q in data["queries"]]
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    return {**data, "queries": queries}


def fingerprint(settings: Any, dataset: dict[str, Any], throttle: float) -> str:
    files = {}
    for base in (ROOT / "evals", ROOT / "src/research_agent"):
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".py", ".md"}
                and not any(part in {".cache", "results", "__pycache__"} for part in path.parts)
            ):
                files[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return digest(
        {
            "protocol": 2,
            "settings": settings.model_dump(mode="json"),
            "dataset": dataset,
            "files": files,
            "throttle": throttle,
        }
    )
