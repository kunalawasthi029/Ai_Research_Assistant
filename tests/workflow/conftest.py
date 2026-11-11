"""Keep scripted workflow source identifiers stable without changing production UUIDs."""

from __future__ import annotations

import pytest

from research_agent.schemas.evidence import Evidence
from research_agent.tools.results import ToolOutput


@pytest.fixture(autouse=True)
def stable_evidence_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give scripted responses known IDs without altering production UUID generation."""
    from research_agent.tools.results import evidence_from_output

    count = 0

    def collect(output: ToolOutput, task_id: str, tool_name: str) -> list[Evidence]:
        nonlocal count
        records = evidence_from_output(output, task_id, tool_name)
        for record in records:
            count += 1
            record.evidence_id = f"ev{count}"
        return records

    monkeypatch.setattr("research_agent.agents.researcher.evidence_from_output", collect)
