"""Build guarded typed graph orchestration with recovery, replanning and optional approval."""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from research_agent.config.settings import Settings
from research_agent.graph.events import emit_event, event_scope
from research_agent.graph.nodes import (
    analyst_node,
    finalize_node,
    planner_node,
    recovery_node,
    replan_node,
    report_generator_node,
    researcher_node,
    semantic_validator_node,
    supervisor_evidence_review_node,
    supervisor_final_review_node,
    supervisor_plan_review_node,
    validator_node,
)
from research_agent.graph.routing import (
    route_evidence_review,
    route_final_review,
    route_plan_review,
)
from research_agent.graph.state import ResearchState
from research_agent.schemas.decisions import TokenUsage
from research_agent.tools.registry import ToolRegistry


def build_graph(
    planner_llm: Any,
    researcher_llm: Any,
    analyst_llm: Any,
    report_llm: Any,
    supervisor_llm: Any,
    validator_llm: Any,
    registry: ToolRegistry,
    checkpointer: Any | None = None,
    settings: Settings | None = None,
) -> Any:
    """Compile finite routing with a correctly owned saver supplied by the caller."""
    names = (
        "max_plan_revisions",
        "max_research_iterations",
        "max_validation_revisions",
        "graph_recursion_limit",
        "overall_run_timeout_seconds",
        "human_approval_enabled",
        "semantic_validation_threshold",
        "semantic_criterion_threshold",
    )
    limits = {
        name: getattr(settings, name) if settings else Settings.model_fields[name].default
        for name in names
    }
    if limits["human_approval_enabled"] and checkpointer is None:
        raise ValueError("Human approval requires persistent checkpointing")

    def input_validation(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("research_query", "").strip()
        objective = state.get("research_objective", "").strip() or query
        if not 3 <= len(query) <= 4000 or len(objective) > 4000:
            raise ValueError("Research input length is invalid")
        return {
            "research_query": query,
            "research_objective": objective,
            "research_preferences": {
                "depth": settings.research_depth if settings else "standard",
                "scope": settings.research_scope if settings else "balanced",
                "source_preferences": settings.source_preferences if settings else [],
            },
            "limits": limits,
            "deadline_at": time.time() + float(limits["overall_run_timeout_seconds"]),
            "execution_status": "running",
        }

    def approval(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("limits", {}).get("human_approval_enabled"):
            allowed = interrupt(
                {
                    "kind": "plan_approval",
                    "research_id": state["research_id"],
                    "plan": state["research_plan"].model_dump(mode="json"),
                }
            )
            if allowed is not True:
                raise ValueError("Plan approval was declined")
        return {"approval_granted": True, "execution_status": "approved"}

    graph = StateGraph(ResearchState)
    functions = {
        "input_validation": input_validation,
        "planner": partial(planner_node, llm=planner_llm),
        "supervisor_plan_review": partial(supervisor_plan_review_node, llm=supervisor_llm),
        "approval": approval,
        "researcher": partial(researcher_node, llm=researcher_llm, registry=registry),
        "analyst": partial(analyst_node, llm=analyst_llm),
        "supervisor_evidence_review": partial(supervisor_evidence_review_node, llm=supervisor_llm),
        "replan": partial(replan_node, llm=planner_llm),
        "report_generator": partial(report_generator_node, llm=report_llm),
        "validator": validator_node,
        "semantic_validator": partial(semantic_validator_node, validator_llm=validator_llm),
        "supervisor_final_review": partial(supervisor_final_review_node, llm=supervisor_llm),
        "finalize": finalize_node,
    }

    def guard(name: str, function: Any) -> Any:
        def execute(state: ResearchState) -> dict[str, Any]:
            started = time.monotonic()
            usages: list[TokenUsage] = []

            def observe(event: dict[str, Any]) -> None:
                if event["event_type"] == "llm_call":
                    usages.append(
                        TokenUsage(
                            agent=event["agent"],
                            model=event["model"],
                            prompt_tokens=event["prompt_tokens"],
                            completion_tokens=event["completion_tokens"],
                            total_tokens=event["total_tokens"],
                            latency_seconds=event["duration_seconds"],
                        )
                    )

            with event_scope(observe):
                emit_event("node_started", node=name)
                try:
                    deadline = state.get("deadline_at", 0)
                    if deadline and time.time() >= deadline:
                        raise TimeoutError("Run deadline exhausted")
                    update: dict[str, Any] = function(dict(state))
                    if deadline and time.time() >= deadline:
                        raise TimeoutError("Run deadline exhausted")
                    update["token_usage"] = [*update.get("token_usage", []), *usages]
                    emit_event(
                        "node_finished",
                        node=name,
                        duration_seconds=time.monotonic() - started,
                        status=update.get("execution_status", state.get("execution_status")),
                        research_iteration=update.get(
                            "research_iteration", state.get("research_iteration", 0)
                        ),
                        revision_count=update.get("revision_count", state.get("revision_count", 0)),
                    )
                    if update.get("agent_decisions"):
                        emit_event(
                            "decision",
                            node=name,
                            **update["agent_decisions"][0].model_dump(mode="json"),
                        )
                    if update.get("validation_result"):
                        emit_event(
                            "validation",
                            node=name,
                            status=update["validation_result"].status,
                            score=update["validation_result"].overall_score,
                        )
                    return update
                except GraphInterrupt:
                    raise
                except Exception as error:
                    emit_event("node_error", node=name, error_type=type(error).__name__)
                    return {
                        "failure_node": name,
                        "error_details": [f"{name} failed ({type(error).__name__})"],
                        "token_usage": usages,
                    }

        return execute

    for name, function in functions.items():
        graph.add_node(name, guard(name, function))
    graph.add_node("recovery", lambda state: recovery_node(dict(state)))
    graph.add_edge(START, "input_validation")
    fixed = {
        "input_validation": "planner",
        "planner": "supervisor_plan_review",
        "approval": "researcher",
        "analyst": "supervisor_evidence_review",
        "replan": "researcher",
        "report_generator": "validator",
        "validator": "semantic_validator",
        "semantic_validator": "supervisor_final_review",
    }
    choices = {
        "supervisor_plan_review": (route_plan_review, ["planner", "approval", "recovery"]),
        "supervisor_evidence_review": (
            route_evidence_review,
            ["replan", "report_generator", "recovery"],
        ),
        "supervisor_final_review": (
            route_final_review,
            ["replan", "report_generator", "finalize", "recovery"],
        ),
    }
    for name, target in fixed.items():

        def route(state: ResearchState, next_node: str = target) -> str:
            return "recovery" if state.get("failure_node") else next_node

        graph.add_conditional_edges(name, route, {target: target, "recovery": "recovery"})

    def after_task(state: ResearchState) -> str:
        return (
            "recovery"
            if state.get("failure_node")
            else "researcher"
            if state.get("task_queue")
            else "analyst"
        )

    graph.add_conditional_edges(
        "researcher",
        after_task,
        {"researcher": "researcher", "analyst": "analyst", "recovery": "recovery"},
    )
    for name, (router, destinations) in choices.items():

        def choose(state: ResearchState, decide: Any = router) -> str:
            return "recovery" if state.get("failure_node") else str(decide(dict(state)))

        graph.add_conditional_edges(name, choose, {target: target for target in destinations})
    graph.add_edge("finalize", END)
    graph.add_edge("recovery", END)
    return graph.compile(checkpointer=checkpointer).with_config(
        recursion_limit=int(limits["graph_recursion_limit"])
    )
