"""SqliteSaver setup for persistent graph checkpointing."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data") / "checkpoints.sqlite"


@contextmanager
def create_checkpointer(db_path: str | Path | None = None) -> Iterator[SqliteSaver]:
    """Create a SqliteSaver checkpointer for the graph.

    Args:
        db_path: Path to the SQLite database file. Defaults to data/checkpoints.sqlite.

    Returns:
        A configured SqliteSaver instance.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(connection)
        saver.setup()
        logger.info("Checkpoint storage opened")
        yield saver
    finally:
        connection.close()
