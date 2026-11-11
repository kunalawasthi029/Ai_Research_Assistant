"""Importable, deadline-isolated live benchmark stages using production agents/graph."""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import Field, TypeAdapter

from evals.baselines import run_baseline_b0, run_baseline_b1
from evals.judge import ensure_independent, run_judge
from research_agent.config.settings import Settings
from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.events import event_scope
from research_agent.graph.nodes import analyst_node
from research_agent.graph.runtime import consume_graph
from research_agent.graph.state import ResearchState, create_initial_state
from research_agent.graph.workflow import build_graph
from research_agent.llm.context import context_for_model
from research_agent.llm.model_config import get_model_config
from research_agent.llm.reliable import ReliableChatGroq
from research_agent.tools.registry import create_default_registry
from research_agent.tools.source_verifier import verify_sources

STATE = TypeAdapter(ResearchState)
_last_request = 0.0


class ThrottledChatGroq(ReliableChatGroq):
    """Sequential eval workers enforce spacing before every provider call."""

    minimum_interval: float = Field(default=0, ge=0, exclude=True)

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        global _last_request
        pause = self.minimum_interval - (time.monotonic() - _last_request)
        if pause > 0:
            time.sleep(pause)
        _last_request = time.monotonic()
        return super()._generate(*args, **kwargs)


def model(role: str, settings: Settings, throttle: float) -> Any:
    config = get_model_config(role, settings.get_model_for_role(role))
    if not config.supports_structured_output:
        raise ValueError("Model capability is not documented")
    return ThrottledChatGroq(
        model=config.model_id,
        api_key=settings.groq_api_key,
        temperature=settings.default_temperature,
        max_tokens=min(config.max_tokens, settings.max_tokens),
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        concurrency_limit=1,
        slot_timeout=settings.llm_timeout_seconds,
        repair_attempts=settings.structured_repair_attempts,
        supports_parallel_tools=config.supports_parallel_tools,
        agent_role=role,
        minimum_interval=throttle,
        structured_output_method=config.structured_output_method,
        evidence_context_max_chars=settings.evidence_context_max_chars,
        reasoning_effort=config.reasoning_effort,
    )


def configured(
    settings: Settings, saver: Any, throttle: float, reserved: dict[str, int] | None = None
) -> Any:
    registry = create_default_registry(settings=settings, isolate_calls=True)
    registry.restore_counts(reserved or {})
    return build_graph(
        planner_llm=model("planner", settings, throttle),
        researcher_llm=model("researcher", settings, throttle),
        analyst_llm=model("analyst", settings, throttle),
        report_llm=model("report_generator", settings, throttle),
        supervisor_llm=model("supervisor", settings, throttle),
        validator_llm=model("validator", settings, throttle),
        registry=registry,
        checkpointer=saver,
        settings=settings,
    )


def resume_full_system(
    graph: Any,
    state: dict[str, Any],
    config: dict[str, Any],
    publish: Any,
    remaining_seconds: float,
) -> dict[str, Any]:
    """Resume the actual coordinator after its shared analyst prefix, on a fresh thread."""
    seed = STATE.validate_python(state)
    seed["research_id"] = str(uuid4())
    seed["thread_id"] = config["configurable"]["thread_id"]
    seed["deadline_at"] = time.time() + remaining_seconds
    graph.update_state(config, seed, as_node="analyst")
    return cast(dict[str, Any], consume_graph(graph, None, config, publish))


def stage_worker(
    channel: Any,
    kind: str,
    settings: Settings,
    query: str,
    prepared: dict[str, Any] | None,
    throttle: float,
    report_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return parsed public artifacts plus measured call metadata; no raw LLM responses."""
    started = time.monotonic()
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def observe(event: dict[str, Any]) -> None:
        event_kind = event.get("event_type")
        if event_kind == "llm_call":
            calls.append(
                {
                    k: event[k]
                    for k in (
                        "agent",
                        "model",
                        "duration_seconds",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                }
            )
        if event_kind in {"node_error", "llm_error", "agent_output_invalid"}:
            errors.append(
                {
                    k: event[k]
                    for k in (
                        "node",
                        "error_type",
                        "http_status",
                        "agent",
                        "schema",
                        "issue_kind",
                        "attempt",
                        "schema_issues",
                    )
                    if k in event
                }
            )
        if event_kind in {
            "node_started",
            "node_finished",
            "node_error",
            "llm_call",
            "llm_error",
            "agent_output_invalid",
        }:
            channel.put(
                (
                    "event",
                    {
                        k: event[k]
                        for k in (
                            "event_type",
                            "node",
                            "agent",
                            "model",
                            "error_type",
                            "duration_seconds",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "http_status",
                            "schema",
                            "issue_kind",
                            "attempt",
                            "schema_issues",
                        )
                        if k in event
                    },
                )
            )

    result: dict[str, Any]
    state: Any  # TypedAdapter validates persisted input; graph outputs are dynamic mappings.
    with event_scope(observe):
        if kind == "judge":
            ensure_independent(settings)
            assert prepared is not None and report_json is not None
            # Same initial reference corpus for every arm, including B0. This is NOT
            # supplied to B0 generation. Extra B2 evidence receives a separate judge run.
            judge_model = model("judge", settings, throttle)
            judge_state = STATE.validate_python(prepared["state"])
            context = context_for_model(judge_model, judge_state["evidence"])
            evidence_text = context.text
            result = {
                "judge": run_judge(
                    query,
                    json.dumps(
                        {
                            k: v
                            for k, v in report_json.items()
                            if k not in {"model_used", "version", "generated_at"}
                        }
                    ),
                    evidence_text,
                    judge_model,
                ).model_dump(),
                "judge_evidence_ids": [item.evidence_id for item in context.records],
                "judge_omitted_ids": list(context.omitted_ids),
            }
        elif kind == "analyze":
            assert prepared is not None
            state = STATE.validate_python(prepared["state"])
            state.update(analyst_node(dict(state), llm=model("analyst", settings, throttle)))
            result = {"state": json.loads(STATE.dump_json(state)), "execution_status": "prepared"}
        elif kind == "b0":
            result = {
                "report": run_baseline_b0(
                    query, model("report_generator", settings, throttle)
                ).model_dump(mode="json"),
                "evidence": [],
                "execution_status": "generated",
            }
        elif kind == "b1":
            assert prepared is not None
            state = STATE.validate_python(prepared["state"])
            result = {
                "report": run_baseline_b1(
                    state, model("report_generator", settings, throttle)
                ).model_dump(mode="json"),
                "evidence": prepared["state"]["evidence"],
                "execution_status": "generated",
            }
        else:
            with (
                tempfile.TemporaryDirectory(prefix="research-eval-") as directory,
                create_checkpointer(Path(directory) / "checkpoints.db") as saver,
            ):
                state = STATE.validate_python(prepared["state"]) if prepared else None
                reserved = (
                    dict(Counter(call.tool_name for call in state["tool_calls"])) if state else {}
                )
                graph = configured(settings, saver, throttle, reserved)
                thread = str(uuid4())
                config = {
                    "configurable": {"thread_id": thread},
                    "recursion_limit": settings.graph_recursion_limit,
                }
                if kind == "prepare":
                    initial = create_initial_state(str(uuid4()), thread, query)
                    state = graph.invoke(initial, config, interrupt_before=["analyst"])
                    state["messages"] = []
                    result = {
                        "state": json.loads(STATE.dump_json(state)),
                        "execution_status": "retrieved",
                        "failure_node": state.get("failure_node")
                        or ("preparation" if not state.get("evidence") else ""),
                    }
                elif kind == "b2":
                    assert state is not None and prepared is not None
                    remaining = settings.overall_run_timeout_seconds - prepared["elapsed_seconds"]
                    if remaining <= 0:
                        raise TimeoutError("Shared prefix exhausted the run budget")
                    state = resume_full_system(graph, state, config, lambda _: None, remaining)
                    report = state.get("report")
                    result = {
                        "report": report.model_dump(mode="json") if report else None,
                        "evidence": [e.model_dump(mode="json") for e in state.get("evidence", [])],
                        "execution_status": state.get("execution_status"),
                        "failure_node": state.get("failure_node"),
                        "revision_count": state.get("revision_count"),
                        "validation_history": json.loads(STATE.dump_json(state)).get(
                            "validation_history", []
                        ),
                    }
                else:
                    raise ValueError("Unknown evaluation stage")
    return {
        **result,
        "status": "failed"
        if result.get("failure_node") or result.get("execution_status") == "failed"
        else "ok",
        "elapsed_seconds": time.monotonic() - started,
        "calls": calls,
        "errors": errors,
    }


def url_worker(channel: Any, urls: list[str]) -> dict[str, Any]:
    """Use production public-only DNS-pinned requests, bounded to 20 URLs per stage."""
    return cast(dict[str, Any], json.loads(verify_sources(urls, timeout_seconds=10)))
