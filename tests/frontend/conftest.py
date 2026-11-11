"""Frontend test imports and an HTTP-boundary client double for native AppTest journeys."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))
from api_client import APIError  # noqa: E402 -- frontend entrypoint imports use its script directory.


@pytest.fixture(autouse=True)
def isolate_frontend_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui_state

    monkeypatch.setattr(ui_state, "dotenv_values", lambda *args, **kwargs: {})


@pytest.fixture(autouse=True)
def restore_python_main() -> Any:
    """AppTest replaces __main__; later Windows spawn must retain pytest's entrypoint."""
    original = sys.modules.get("__main__")
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = original


def frame(
    sequence: int,
    event_type: str,
    data: dict[str, Any],
    agent: str = "researcher",
    node: str = "researcher",
) -> dict[str, Any]:
    return {
        "event_id": sequence,
        "event_type": event_type,
        "data": {
            "event_id": f"event-{sequence}",
            "agent": agent,
            "node": node,
            "timestamp": "2026-11-10T00:00:00+00:00",
            "data": data,
        },
    }


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.runs: dict[str, Any] = {}
        self.events: dict[str, list[Any]] = {}
        self.fail: set[str] = set()
        self.closed = 0
        self.plan = {
            "research_objective": "Compare retrieval approaches",
            "scope": "balanced",
            "research_tasks": [
                {
                    "task_id": "t1",
                    "description": "Compare source evidence",
                    "status": "completed",
                    "required_evidence": "Observed sources",
                    "completion_criteria": "Compare evidence",
                    "expected_source_types": ["web"],
                }
            ],
        }
        self.validation = {
            "status": "pass",
            "overall_score": 0.9,
            "scores": [{"criterion": "traceability", "score": 1.0, "passed": True}],
            "issues": [],
            "unsupported_claims": [],
            "feedback": "All checks passed",
        }
        self.report = {
            "title": "Retrieval comparison",
            "executive_summary": "Observed summary [ref1]",
            "research_objective": "Compare retrieval",
            "methodology": "Observed sources",
            "key_findings": ["Source finding [ref1]"],
            "detailed_analysis": [
                {
                    "heading": "Comparison",
                    "content": "Evidence supports the comparison [ref1]",
                    "evidence_ids": ["ev1"],
                }
            ],
            "comparative_insights": ["Different methods"],
            "recommendations": ["Review coverage"],
            "limitations": ["Limited sample"],
            "conclusion": "Observed conclusion [ref1]",
            "references": [
                {
                    "ref_id": "ref1",
                    "evidence_id": "ev1",
                    "url": "https://example.com/source",
                    "title": "Observed source",
                }
            ],
        }

    def check(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise APIError(
                "The server is busy or unavailable. Try again shortly.", 503, "request-error"
            )

    def health_check(self) -> dict[str, Any]:
        self.check("health")
        return {"status": "ok"}

    def start_research(
        self, query: str, objective: str = "", *, options: Any = None
    ) -> dict[str, Any]:
        self.check("start")
        run_id = f"run-{len(self.runs) + 1}"
        self.calls.append(("submission", query, objective, options))
        self.runs[run_id] = {
            "run_id": run_id,
            "status": "awaiting_approval" if options.get("human_approval_enabled") else "running",
            "job_active": not options.get("human_approval_enabled"),
            "query": query,
            "research_plan": copy.deepcopy(self.plan),
            "report_version": None,
            "total_tool_calls": 0,
            "total_tokens": 0,
            "evidence_count": 0,
            "elapsed_seconds": 1.5,
            "revision_count": 0,
            "research_iteration": 0,
            "research_preferences": {
                "depth": options.get("depth"),
                "scope": options.get("scope"),
                "source_preferences": options.get("source_preferences", []),
            },
        }
        self.events[run_id] = []
        return {"run_id": run_id, "status": "accepted"}

    def complete(self, run_id: str = "run-1", status: str = "completed") -> None:
        self.runs[run_id].update(
            status=status,
            job_active=False,
            report_version=1,
            evidence_count=1,
            total_tool_calls=1,
            total_tokens=31,
            revision_count=1,
            research_iteration=1,
            validation_result=copy.deepcopy(self.validation),
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        self.check("run")
        if run_id not in self.runs:
            raise APIError("The run or report is not available.", 404)
        return copy.deepcopy(self.runs[run_id])

    def get_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        self.check("events")
        self.calls.append(("after", after))
        rows = [row for row in self.events[run_id] if row["event_id"] > after]
        return copy.deepcopy(rows[:200]) + [
            {"event_type": "snapshot", "event_id": None, "data": {"has_more": len(rows) > 200}}
        ]

    def get_report(self, run_id: str) -> dict[str, Any]:
        self.check("report")
        if self.runs[run_id].get("report_version") is None:
            raise APIError("The run or report is not available.", 404)
        return {
            "version": 1,
            "created_at": "2026-11-10",
            "report_json": copy.deepcopy(self.report),
            "report_markdown": "# Retrieval comparison\n\n[ref1] https://example.com/source",
        }

    def get_sources(self, run_id: str) -> list[dict[str, Any]]:
        self.check("sources")
        return [
            {
                "evidence_id": "ev1",
                "title": "Observed source",
                "url": "https://example.com/source",
                "content": "Observed source text",
                "source_type": "web",
                "tool_name": "web_search",
                "retrieved_at": "2026-11-10",
                "source": {"is_accessible": None},
                "findings": [{"claim": "Observed claim"}],
            }
        ]

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        self.check("history")
        return [
            {**copy.deepcopy(run), "tool_call_count": run.get("total_tool_calls", 0)}
            for run in self.runs.values()
        ][:limit]

    def resume_research(self, run_id: str, approved: bool | None = None) -> dict[str, Any]:
        self.check("resume")
        self.calls.append(("approval", approved))
        self.runs[run_id].update(
            status="incomplete" if approved is False else "running",
            job_active=approved is not False,
        )
        return {"run_id": run_id, "status": "accepted"}

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeAPI:
    import ui_state

    api = FakeAPI()
    monkeypatch.setattr(ui_state, "APIClient", lambda base_url, **kwargs: api)
    return api


@pytest.fixture
def ui(fake_api: FakeAPI) -> AppTest:
    app = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=15)
    app.session_state["api_url"] = f"http://ui-{uuid4().hex}.test/api/v1"
    return app.run()
