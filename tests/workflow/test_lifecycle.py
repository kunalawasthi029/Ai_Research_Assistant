"""Exercise real SQLite checkpoints, approval, recovery, streaming and finite graph limits."""

from __future__ import annotations

import json
import queue
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from research_agent.config.settings import Settings
from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.nodes import recovery_node
from research_agent.graph.researcher_loop import build_researcher_subgraph
from research_agent.graph.runtime import consume_graph, graph_worker
from research_agent.graph.state import create_initial_state
from research_agent.graph.workflow import build_graph
from research_agent.schemas.plan import ResearchPlan
from research_agent.service.research_service import ResearchService
from research_agent.tools.results import evidence_from_output
from tests.fake_llm import FakeChatModel
from tests.workflow.test_workflow import (
    _build_happy_path_llms,
    _make_mock_registry,
    _make_plan_json,
    _make_supervisor_decision_json,
    _make_validation_result_json,
)


def test_each_completed_task_is_durable_and_skipped_on_restart(tmp_path: Path) -> None:
    plan = json.loads(_make_plan_json())
    second = dict(plan["research_tasks"][0], task_id="t2")
    plan["research_tasks"].append(second)
    models = _build_happy_path_llms()
    models["planner"] = FakeChatModel(responses=[AIMessage(content=json.dumps(plan))])
    config = {"configurable": {"thread_id": "tasks"}}
    path = tmp_path / "tasks.sqlite"
    with create_checkpointer(path) as saver:
        graph = make_graph(models, checkpointer=saver)
        graph.invoke(
            create_initial_state("run", "tasks", "What is X?"),
            config,
            interrupt_after=["researcher"],
        )
        saved = graph.get_state(config)
        assert saved.next == ("researcher",)
        assert saved.values["completed_tasks"] == ["t1"]
        assert [task.task_id for task in saved.values["task_queue"]] == ["t2"]
        assert len(saved.values["evidence"]) == 1
    replacement = _build_happy_path_llms()
    replacement["planner"] = FakeChatModel(responses=[])
    with create_checkpointer(path) as saver:
        graph = make_graph(replacement, checkpointer=saver)
        graph.invoke(None, config, interrupt_after=["researcher"])
        saved = graph.get_state(config)
        assert saved.next == ("analyst",)
        assert saved.values["completed_tasks"] == ["t1", "t2"]
        assert len(saved.values["tool_calls"]) == 2
        assert not saved.values["task_queue"]
        assert not replacement["planner"].invocations


def make_graph(models: dict[str, Any] | None = None, **options: Any) -> Any:
    """Compile the real coordinator around separately scripted role models."""
    models = models or _build_happy_path_llms()
    return build_graph(
        planner_llm=models["planner"],
        researcher_llm=models["researcher"],
        analyst_llm=models["analyst"],
        report_llm=models["report"],
        supervisor_llm=models["supervisor"],
        validator_llm=models["validator"],
        registry=_make_mock_registry(),
        **options,
    )


def test_checkpoint_reopen_resume_and_thread_isolation(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite"
    settings = Settings(groq_api_key="dummy", human_approval_enabled=True)
    models = _build_happy_path_llms()
    config = {"configurable": {"thread_id": "original"}}
    with create_checkpointer(path) as saver:
        graph = make_graph(models, checkpointer=saver, settings=settings)
        paused = graph.invoke(create_initial_state("run", "original", "What is X?"), config)
        assert paused["__interrupt__"]
        snapshot = graph.get_state(config)
        assert snapshot.next == ("approval",)
        assert isinstance(snapshot.values["research_plan"], ResearchPlan)
        assert not snapshot.values["tool_calls"]
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        saver.get_tuple(config)
    # New models and compiled graph: persisted planning must not be repeated.
    resumed_models = _build_happy_path_llms()
    resumed_models["planner"] = FakeChatModel(responses=[])
    resumed_models["supervisor"] = FakeChatModel(
        responses=resumed_models["supervisor"].responses[1:]
    )
    with create_checkpointer(path) as saver:
        graph = make_graph(resumed_models, checkpointer=saver, settings=settings)
        assert not graph.get_state({"configurable": {"thread_id": "other"}}).values
        final = graph.invoke(Command(resume=True), config)
        assert final["execution_status"] == "completed"
        assert len(final["tool_calls"]) == 1
        assert len(resumed_models["planner"].invocations) == 0
        assert not graph.get_state(config).next
        assert graph.get_state(config).values["report"].references[0].evidence_id == "ev1"


def test_approval_off_by_default_and_decline_is_incomplete(tmp_path: Path) -> None:
    graph = make_graph()
    assert "__interrupt__" not in graph.invoke(create_initial_state("run", "thread", "What is X?"))
    with create_checkpointer(tmp_path / "decline.sqlite") as saver:
        graph = make_graph(
            checkpointer=saver, settings=Settings(groq_api_key="dummy", human_approval_enabled=True)
        )
        config = {"configurable": {"thread_id": "decline"}}
        graph.invoke(create_initial_state("run", "decline", "What is X?"), config)
        final = graph.invoke(Command(resume=False), config)
        assert final["execution_status"] == "incomplete"
        assert final["report"].references == []
        assert final["tool_calls"] == []


def test_approval_requires_checkpointer() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        make_graph(settings=Settings(groq_api_key="dummy", human_approval_enabled=True))


def test_runtime_worker_reopens_approval_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        groq_api_key="dummy",
        human_approval_enabled=True,
        checkpoint_db_path=str(tmp_path / "runtime.sqlite"),
    )
    models = _build_happy_path_llms()

    def configured(settings: Settings, saver: Any, **options: Any) -> Any:
        assert options["reserved_calls"] == {"web_search": 1}
        return make_graph(models, checkpointer=saver, settings=settings)

    monkeypatch.setattr("research_agent.graph.runtime.configured_graph", configured)
    channel: queue.Queue[Any] = queue.Queue()
    args = (channel, settings, "run", "thread", "What is X?", "", True)
    paused = graph_worker(*args, reserved_calls={"web_search": 1})
    assert paused["execution_status"] == "awaiting_approval"
    resumed = graph_worker(
        *args, resume=True, approved=True, remaining_seconds=30, reserved_calls={"web_search": 1}
    )
    assert resumed["execution_status"] == "completed"
    assert len(models["planner"].invocations) == 1
    with pytest.raises(ValueError, match="no pending work"):
        graph_worker(*args, resume=True, reserved_calls={"web_search": 1})


@pytest.mark.parametrize("failure", ["bad_plan", "invalid_input", "model_timeout"])
def test_recoverable_failures_return_honest_artifact(failure: str) -> None:
    models = _build_happy_path_llms()
    if failure == "bad_plan":
        models["planner"] = FakeChatModel(
            responses=[AIMessage(content="invalid"), AIMessage(content="invalid")]
        )
    elif failure == "model_timeout":

        class TimedOut(FakeChatModel):
            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                raise TimeoutError("private-provider-body")

        models["planner"] = TimedOut(responses=[])
    result = make_graph(models).invoke(
        create_initial_state("run", "thread", "" if failure == "invalid_input" else "What is X?")
    )
    assert result["execution_status"] == "incomplete"
    assert result["report"].references == []
    assert "private-provider-body" not in str(result)
    assert result["failure_node"] in {"input_validation", "planner"}


def test_custom_events_are_observed_during_execution() -> None:
    graph = make_graph()
    seen = []
    saw_tool_before_terminal = False
    terminal = False
    for mode, payload in graph.stream(
        create_initial_state("run", "thread", "What is X?"), stream_mode=["values", "custom"]
    ):
        if mode == "custom":
            seen.append(payload["event_type"])
            if payload["event_type"] == "tool_started":
                saw_tool_before_terminal = not terminal
        else:
            terminal = payload.get("execution_status") in {"completed", "incomplete", "best_effort"}
    assert saw_tool_before_terminal
    assert {"tool_started", "tool_finished", "source_found", "decision", "validation"}.issubset(
        seen
    )


def test_compiled_graphs_have_real_conditional_sources_and_tool_loop() -> None:
    graph = make_graph().get_graph()
    assert len({edge.source for edge in graph.edges if edge.conditional}) >= 4
    child = build_researcher_subgraph(
        FakeChatModel(responses=[]), _make_mock_registry(), evidence_from_output
    ).get_graph()
    edges = {(edge.source, edge.target) for edge in child.edges}
    assert ("execute_tools", "model") in edges
    assert ("model", "execute_tools") in edges
    assert {"replan", "approval", "recovery", "semantic_validator"}.issubset(graph.nodes)


def test_plan_revision_zero_cap_does_not_force_approval() -> None:
    models = _build_happy_path_llms()
    models["supervisor"] = FakeChatModel(
        responses=[
            AIMessage(content=_make_supervisor_decision_json("revise", "planner", "plan_review"))
        ]
    )
    result = make_graph(
        models, settings=Settings(groq_api_key="dummy", max_plan_revisions=0)
    ).invoke(create_initial_state("run", "thread", "What is X?"))
    assert result["execution_status"] == "incomplete"
    assert not result["tool_calls"]
    assert any("Plan revision limit" in warning for warning in result["warnings"])


def test_first_plan_revision_receives_feedback_without_consuming_initial_budget() -> None:
    models = _build_happy_path_llms()
    models["planner"] = FakeChatModel(
        responses=[AIMessage(content=_make_plan_json()), AIMessage(content=_make_plan_json())]
    )
    decision = json.loads(_make_supervisor_decision_json("revise", "planner", "plan_review"))
    decision["reasoning"] = "Add a source diversity requirement"
    models["supervisor"].responses.insert(0, AIMessage(content=json.dumps(decision)))
    result = make_graph(models).invoke(create_initial_state("run", "thread", "What is X?"))
    assert result["execution_status"] == "completed"
    assert result["plan_revision_count"] == 1
    assert "Add a source diversity requirement" in str(models["planner"].invocations[1])


def test_zero_report_revision_cap_preserves_best_effort_report() -> None:
    models = _build_happy_path_llms()
    models["validator"] = FakeChatModel(
        responses=[AIMessage(content=_make_validation_result_json("needs_revision", 0.4))]
    )
    models["supervisor"].responses[-1] = AIMessage(
        content=_make_supervisor_decision_json("needs_revision", "report_generator", "final_review")
    )
    result = make_graph(
        models, settings=Settings(groq_api_key="dummy", max_validation_revisions=0)
    ).invoke(create_initial_state("run", "thread", "What is X?"))
    assert result["execution_status"] == "best_effort"
    assert result["revision_count"] == 0
    assert result["report"].version == 1
    assert any("Report revision limit" in warning for warning in result["warnings"])
    assert len(models["report"].invocations) == 1


def test_low_recursion_limit_recovers_via_stream_runner() -> None:
    graph = make_graph(settings=Settings(groq_api_key="dummy", graph_recursion_limit=1))
    result = consume_graph(
        graph,
        create_initial_state("run", "thread", "What is X?"),
        {"recursion_limit": 1},
        lambda event: None,
    )
    assert result["execution_status"] == "incomplete"
    assert result["failure_node"] == "recursion_limit"


def test_service_restarts_and_resumes_same_run_with_real_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        groq_api_key="dummy-secret",
        human_approval_enabled=True,
        database_path=str(tmp_path / "runs.sqlite"),
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    models = _build_happy_path_llms()

    def local_worker(
        service: ResearchService,
        run_id: str,
        thread_id: str,
        query: str,
        objective: str,
        use_checkpointer: bool,
        **options: Any,
    ) -> dict[str, Any]:
        with create_checkpointer(settings.checkpoint_db_path) as saver:
            graph = make_graph(models, checkpointer=saver, settings=settings)
            value = (
                Command(resume=options.get("approved"))
                if options.get("resume")
                else create_initial_state(run_id, thread_id, query, objective)
            )

            def publish(event: dict[str, Any]) -> None:
                service._handle_worker_event(run_id, event)

            return consume_graph(graph, value, {"configurable": {"thread_id": thread_id}}, publish)

    monkeypatch.setattr(ResearchService, "_run_worker", local_worker)
    service = ResearchService(settings)
    started = service.run_research("What is X?")
    assert started["status"] == "awaiting_approval"
    assert "dummy-secret" not in service.get_run(started["run_id"]).config_json
    assert service.inspect_checkpoint(started["run_id"])["research_plan"].research_tasks
    service.conn.close()
    replacement = ResearchService(settings)
    resumed = replacement.resume_research(started["run_id"], approved=True)
    assert resumed["run_id"] == started["run_id"]
    assert resumed["thread_id"] == started["thread_id"]
    assert resumed["status"] == "completed"
    assert len(replacement.list_runs()) == 1
    assert any(
        event.event_type == "tool_started" for event in replacement.get_events(started["run_id"])
    )
    assert replacement.get_report(started["run_id"]).version == 1
    assert len(models["planner"].invocations) == 1
    replacement.conn.close()


@pytest.mark.parametrize("has_report", [False, True])
def test_service_never_infers_completion_from_report_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_report: bool,
) -> None:
    state = dict(create_initial_state("run", "thread", "What is X?"))
    if has_report:
        state["report"] = recovery_node(state)["report"]
    state.pop("execution_status")
    monkeypatch.setattr(ResearchService, "_execute_graph", lambda *args, **kwargs: state)
    service = ResearchService(
        Settings(groq_api_key="dummy", database_path=str(tmp_path / "runs.sqlite"))
    )
    try:
        result = service.run_research("What is X?")
        assert result["status"] == ("best_effort" if has_report else "incomplete")
        assert service.get_run(result["run_id"]).status != "completed"
    finally:
        service.conn.close()
