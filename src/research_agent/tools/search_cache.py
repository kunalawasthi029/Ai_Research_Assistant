"""Optional, bounded SQLite cache of validated public search results, never LLM thoughts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from research_agent.tools.results import ToolOutput


def search_key(query: str, provider: str, depth: str, limit: int) -> str:
    payload = json.dumps([1, query, provider, depth, limit], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SearchCache:
    """Cache failures never prevent research; each operation owns its connection.

    No credentials or raw query are stored in keys. Result text can still contain
    query terms: enable only on storage appropriate for the application's users.
    Concurrent misses may both fetch; this is not an exactly-once billing ledger.
    """

    def __init__(self, path: str, ttl_seconds: int) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> ToolOutput | None:
        if not self.path or self.ttl_seconds <= 0:
            return None
        try:
            # Do not create an empty cache on reads.
            if not Path(self.path).is_file():
                return None
            with closing(sqlite3.connect(self.path, timeout=1)) as connection:
                row = connection.execute(
                    "SELECT saved_at, payload FROM search_results WHERE cache_key = ?", (key,)
                ).fetchone()
            if row is None or not 0 <= time.time() - row[0] < self.ttl_seconds:
                return None
            output = ToolOutput.model_validate_json(row[1])
            if output.status != "success" or not output.items:
                return None
            output.data.update(cache_hit=True, tavily_credits_upper_bound=0)
            for item in output.items:
                item.metadata["search_cache_hit"] = True
            return output
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return None

    def put(self, key: str, output: ToolOutput) -> None:
        if not self.path or self.ttl_seconds <= 0 or output.status != "success" or not output.items:
            return
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=1)) as connection, connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS search_results "
                    "(cache_key TEXT PRIMARY KEY, saved_at REAL NOT NULL, payload TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT OR REPLACE INTO search_results VALUES (?, ?, ?)",
                    (key, time.time(), output.model_dump_json()),
                )
                connection.execute(
                    "DELETE FROM search_results WHERE saved_at < ?",
                    (time.time() - self.ttl_seconds,),
                )
                connection.execute(
                    "DELETE FROM search_results WHERE cache_key NOT IN "
                    "(SELECT cache_key FROM search_results ORDER BY saved_at DESC LIMIT 512)"
                )
        except (OSError, sqlite3.Error):
            pass
