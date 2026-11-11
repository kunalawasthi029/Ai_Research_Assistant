"""Preferences reach actual role prompts and finite snapshots return before a run completes."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from research_agent.config.settings import Settings
from research_agent.persistence.models import EventRow
from research_agent.service.research_service import ResearchService
from tests.api.test_routes import parse_sse, wait_idle


def test_preferences_are_persisted_checkpointed_and_supplied_to_roles(
    client: TestClient,
    settings: Settings,
    scripted_worker: Any,
) -> None:
    _, models = scripted_worker
    preferences = {
        "depth": "brief",
        "scope": "broad",
        "source_preferences": ["academic", "wikipedia"],
    }
    response = client.post("/api/v1/research", json={"query": "What is X?", "options": preferences})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    result = wait_idle(client, run_id)
    assert result["status"] == "completed"
    assert result["research_preferences"] == preferences
    assert result["research_plan"]["scope"] == "broad"
    with ResearchService(settings) as service:
        config = json.loads(service.get_run(run_id).config_json)
        assert config["research_depth"] == "brief" and config["max_research_iterations"] == 1
        assert config["max_tool_calls_per_task"] <= 3
        assert service.inspect_checkpoint(run_id)["research_preferences"] == preferences
    for role in ("planner", "report"):
        prompt = str(models[run_id][role].invocations[0][-1].content)
        assert "UNTRUSTED_RESEARCH_PREFERENCES" in prompt and "academic" in prompt


@pytest.mark.parametrize(
    "options",
    [
        {"depth": "unbounded"},
        {"scope": "unknown"},
        {"source_preferences": ["web", "web"]},
        {"source_preferences": ["invented"]},
        {"source_preferences": "web"},
    ],
)
def test_preference_bounds_remain_on_the_backend(client: TestClient, options: Any) -> None:
    assert (
        client.post(
            "/api/v1/research", json={"query": "What is X?", "options": options}
        ).status_code
        == 422
    )


def test_finite_events_return_observed_activity_while_the_run_is_running(
    client: TestClient, settings: Settings
) -> None:
    with ResearchService(settings) as service:
        run = service.create_run("What is X?")
        service._run_repo.claim(run.id, ("queued",))
        service._event_repo.create(
            EventRow(
                id="observed",
                run_id=run.id,
                event_type="tool_started",
                data_json='{"tool":"web_search"}',
            )
        )
    response = client.get(f"/api/v1/research/{run.id}/events?follow=false")
    rows = parse_sse(response.text)
    assert response.status_code == 200
    assert rows[-1]["event"] == "snapshot" and rows[-1]["data"]["status"] == "running"
    assert any(row["event"] == "tool_started" for row in rows)
    assert not any(row["event"] == "done" for row in rows)
    assert client.get(f"/api/v1/research/{run.id}").json()["status"] == "running"
