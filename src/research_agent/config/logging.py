"""Emit JSON logs with credential redaction and observable execution context."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime

_CREDENTIAL = re.compile(r"(?:gsk_|tvly-)[A-Za-z0-9_-]+")
_ASSIGNMENT = re.compile(r"(?i)((?:api[_-]?key|authorization|token|password)\s*[=:]\s*)[^\s,;]+")


class SafeJsonFormatter(logging.Formatter):
    """Serialize event metadata without provider response bodies or secret values."""

    def format(self, record: logging.LogRecord) -> str:
        """Redact credential patterns and omit exception values that may echo inputs."""
        message = _ASSIGNMENT.sub(
            r"\1[REDACTED]", _CREDENTIAL.sub("[REDACTED]", record.getMessage())
        )
        data: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }
        for key in (
            "research_id",
            "thread_id",
            "agent",
            "node",
            "tool",
            "duration_seconds",
            "total_tokens",
        ):
            if hasattr(record, key):
                value = str(getattr(record, key))
                data[key] = _CREDENTIAL.sub("[REDACTED]", value)
        if record.exc_info and record.exc_info[0]:
            data["error_type"] = record.exc_info[0].__name__
        return json.dumps(data)


def configure_logging(level: int = logging.INFO) -> None:
    """Install one safe stderr sink and suppress provider wire-debug logging."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeJsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ("groq", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
