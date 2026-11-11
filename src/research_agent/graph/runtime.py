"""Build configured live graphs and stream their observable events in an isolated worker."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from research_agent.config.settings import Settings
from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.nodes import recovery_node
from research_agent.graph.state import create_initial_state
from research_agent.graph.workflow import build_graph
from research_agent.llm.groq_client import create_llm
from research_agent.tools.registry import create_default_registry


def configured_graph(
    settings: Settings, checkpointer: Any, *, reserved_calls: dict[str, int] | None = None
) -> Any:
    """Create role models and a separately budgeted registry without performing any LLM calls."""
    registry = create_default_registry(settings=settings, isolate_calls=True)
    registry.restore_counts(reserved_calls or {})
    return build_graph(
        planner_llm=create_llm("planner", settings),
        researcher_llm=create_llm("researcher", settings),
        analyst_llm=create_llm("analyst", settings),
        report_llm=create_llm("report_generator", settings),
        supervisor_llm=create_llm("supervisor", settings),
        validator_llm=create_llm("validator", settings),
        registry=registry,
        checkpointer=checkpointer,
        settings=settings,
    )


def consume_graph(
    graph: Any, input_value: Any, config: dict[str, Any], publish: Any
) -> dict[str, Any]:
    """Forward real custom events and keep the most recent committed graph state."""
    latest: dict[str, Any] = {}
    try:
        for mode, payload in graph.stream(
            input_value, config=config, stream_mode=["values", "custom"]
        ):
            if mode == "custom":
                publish(payload)
            else:
                latest = dict(payload)
                publish(
                    {
                        "event_type": "state_snapshot",
                        "state": {
                            key: latest[key]
                            for key in (
                                "evidence",
                                "sources",
                                "findings",
                                "report",
                                "validation_history",
                                "tool_calls",
                                "token_usage",
                            )
                            if key in latest
                        },
                    }
                )
    except GraphRecursionError:
        latest = {
            **latest,
            "failure_node": "recursion_limit",
            "warnings": [*latest.get("warnings", []), "Graph recursion limit reached"],
        }
        latest.update(recovery_node(latest))
        publish({"event_type": "state_snapshot", "state": latest})
    if graph.checkpointer is not None:
        snapshot = graph.get_state(config)
        if snapshot.next and any(task.interrupts for task in snapshot.tasks):
            latest["execution_status"] = "awaiting_approval"
            publish(
                {
                    "event_type": "approval_required",
                    "node": "approval",
                    "research_id": latest.get("research_id"),
                }
            )
    return latest


def graph_worker(
    channel: Any,
    settings: Settings,
    run_id: str,
    thread_id: str,
    query: str,
    objective: str,
    use_checkpointer: bool,
    resume: bool = False,
    approved: bool | None = None,
    remaining_seconds: float | None = None,
    reserved_calls: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Own checkpoint lifetime for one active execution slice; waiting for humans owns no process."""
    manager = (
        create_checkpointer(settings.checkpoint_db_path) if use_checkpointer else nullcontext(None)
    )
    with manager as saver:
        graph = configured_graph(settings, saver, reserved_calls=reserved_calls)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.graph_recursion_limit,
        }
        value: Any
        if resume:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                raise ValueError("Checkpoint has no pending work")
            update = {
                "deadline_at": time.time()
                + (remaining_seconds or settings.overall_run_timeout_seconds)
            }
            if any(task.interrupts for task in snapshot.tasks):
                value = Command(resume=approved, update=update)
            else:
                graph.update_state(config, update)
                value = None
        else:
            value = create_initial_state(run_id, thread_id, query, objective)
        return consume_graph(graph, value, config, lambda event: channel.put(("event", event)))


def checkpoint_state(settings: Settings, thread_id: str) -> dict[str, Any]:
    """Read checkpoint state without constructing models, tools or contacting providers."""
    if not Path(settings.checkpoint_db_path).is_file():
        return {}
    with create_checkpointer(settings.checkpoint_db_path) as saver:
        checkpoint = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        return dict(checkpoint.checkpoint["channel_values"]) if checkpoint else {}
