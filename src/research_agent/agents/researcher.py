"""Researcher agent: prepare a task and execute its explicit LangGraph tool subgraph."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.graph.events import event_scope
from research_agent.graph.researcher_loop import build_researcher_subgraph
from research_agent.llm.context import untrusted_block
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.results import evidence_from_output

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher.md"


def _load_prompt() -> str:
    """Load the researcher's distinct tool-use policy."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def run_researcher_task(
    task_id: str,
    task_description: str,
    search_queries: list[str],
    llm: Any,
    registry: ToolRegistry,
    max_tool_calls: int = 5,
    required_evidence: str = "",
    completion_criteria: str = "",
    expected_source_types: list[str] | None = None,
    research_objective: str = "",
) -> dict[str, Any]:
    """Invoke model -> conditional tool execution -> model with bounded repairs/calls."""
    budget = min(max_tool_calls, registry.max_task_calls)
    if budget < 1:
        raise ValueError("Task tool budget must be positive")
    registry.extraction_focus = re.sub(
        r"https?://\S+", "", f"{research_objective} {task_description} {required_evidence}"
    )[:500]
    source_urls = list(
        dict.fromkeys(
            url.rstrip(".,);]") for url in re.findall(r"https?://[^\s<>\"']+", research_objective)
        )
    )[:20]
    prompt = untrusted_block(
        "TASK",
        {
            "description": task_description,
            "search_queries": search_queries,
            "required_evidence": required_evidence,
            "completion_criteria": completion_criteria,
            "expected_source_types": expected_source_types or [],
            "research_objective": research_objective,
            "provided_source_urls": source_urls,
        },
    )
    graph = build_researcher_subgraph(llm, registry, evidence_from_output)
    with event_scope():
        result = graph.invoke(
            {
                "task_id": task_id,
                "messages": [
                    SystemMessage(content=_load_prompt()),
                    HumanMessage(
                        content=f"Research this task within {budget} tool calls.\n{prompt}"
                    ),
                ],
                "pending_calls": [],
                "max_calls": budget,
                "calls_made": 0,
                "repair_count": 0,
                "status": "model",
                "evidence": [],
                "tool_calls": [],
                "tool_results": [],
                "error_details": [],
            },
            {"recursion_limit": 3 * budget + 12},
        )
    return {
        key: result[key]
        for key in ("evidence", "tool_calls", "tool_results", "error_details", "status")
    }
