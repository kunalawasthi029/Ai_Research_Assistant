"""API fixtures use real thread-owned SQLite and the real graph with scripted inference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import Command

from research_agent.api.app import create_app
from research_agent.config.settings import Settings
from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.runtime import consume_graph
from research_agent.graph.state import create_initial_state
from research_agent.service.research_service import ResearchService
from tests.fake_llm import FakeChatModel
from tests.workflow.test_lifecycle import make_graph
from tests.workflow.test_workflow import _build_happy_path_llms


class ObservedEvidenceModel(FakeChatModel):
    """Use the real collected UUID visible in the prompt; do not replace retrieval IDs."""

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            match = re.search(
                r"<UNTRUSTED_EVIDENCE>\n(.*?)\n</UNTRUSTED_EVIDENCE>",
                str(messages[-1].content),
                re.S,
            )
            if match and isinstance(response.content, str):
                items = json.loads(match.group(1))["items"]
                if items:
                    self.responses[self._call_index] = AIMessage(
                        content=response.content.replace(
                            '"ev1"', json.dumps(items[0]["evidence_id"])
                        )
                    )
        return super()._generate(messages, *args, **kwargs)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        groq_api_key="dummy-private-key",
        database_path=str(tmp_path / "runs.sqlite"),
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        api_max_workers=1,
        api_max_queued_jobs=1,
        cors_origins='["http://allowed.test"]',
        overall_run_timeout_seconds=30,
    )


@pytest.fixture
def scripted_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    models_by_run = {}

    def local(
        service: ResearchService,
        run_id: str,
        thread_id: str,
        query: str,
        objective: str,
        use_checkpointer: bool,
        **options: Any,
    ) -> dict[str, Any]:
        settings = options.get("settings") or service.settings
        if run_id not in models_by_run:
            models_by_run[run_id] = {
                name: ObservedEvidenceModel(responses=model.responses)
                for name, model in _build_happy_path_llms().items()
            }
        with create_checkpointer(settings.checkpoint_db_path) as saver:
            graph = make_graph(models_by_run[run_id], settings=settings, checkpointer=saver)
            value = (
                Command(resume=options.get("approved"))
                if options.get("resume")
                else create_initial_state(run_id, thread_id, query, objective)
            )
            return consume_graph(
                graph,
                value,
                {"configurable": {"thread_id": thread_id}},
                lambda event: service._handle_worker_event(run_id, event),
            )

    monkeypatch.setattr(ResearchService, "_run_worker", local)
    return local, models_by_run


@pytest.fixture
def client(settings: Settings, scripted_worker: Any) -> Any:
    with TestClient(create_app(settings)) as client:
        yield client
