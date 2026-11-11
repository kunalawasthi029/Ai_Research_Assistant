"""Execute each research task as an explicit LangGraph model/tool conditional loop."""

from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from research_agent.graph.events import emit_event
from research_agent.llm.retry import MalformedToolCallError
from research_agent.schemas.decisions import ToolCallRecord, ToolResult
from research_agent.schemas.evidence import Evidence
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.results import ToolExecutionError, ToolOutput, safe_arguments


class TaskState(TypedDict):
    """Private in-task messages plus observable append-only tool/evidence records."""

    task_id: str
    messages: list[BaseMessage]
    pending_calls: list[dict[str, Any]]
    max_calls: int
    calls_made: int
    repair_count: int
    status: str
    evidence: Annotated[list[Evidence], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    error_details: Annotated[list[str], operator.add]


def route_tool_turn(state: TaskState) -> str:
    """Route valid calls to execution, repair feedback to model, and all stops to END."""
    if state["status"] == "repair":
        return "model"
    return "execute_tools" if state["status"] == "tools" else END


def build_researcher_subgraph(
    llm: Any, registry: ToolRegistry, collect: Callable[[ToolOutput, str, str], list[Evidence]]
) -> Any:
    """Build the required loop explicitly; raw model messages are never checkpointed."""
    bound = llm.bind_tools(registry.get_all_tools())
    repair_limit = min(2, max(0, int(getattr(llm, "repair_attempts", 1))))

    def model(state: TaskState) -> dict[str, Any]:
        final_review = state["calls_made"] >= state["max_calls"]
        messages = list(state["messages"])
        if final_review:
            messages.append(
                HumanMessage(
                    content="The tool budget is closed. Review the last tool result against every task requirement. Do not request another tool. Reply exactly TASK_COMPLETE only if the retrieved evidence meets the requirements; otherwise reply TASK_INCOMPLETE. This is an execution assessment, not factual validation."
                )
            )
        malformed = False
        try:
            response = bound.invoke(messages)
            malformed = bool(response.invalid_tool_calls) or (
                len(response.tool_calls) > 1 and not getattr(llm, "supports_parallel_tools", False)
            )
        except MalformedToolCallError:
            malformed = True
        except Exception as error:
            return {
                "status": "llm_error",
                "pending_calls": [],
                "error_details": [f"Research model failed ({type(error).__name__})"],
            }
        if final_review:
            # One terminal assessment sees the last result; it cannot execute tools,
            # retry malformed output, or re-enter the loop. Independent analysis
            # and validation still decide whether evidence supports the objective.
            complete = (
                not malformed
                and not response.tool_calls
                and isinstance(response.content, str)
                and response.content.strip() == "TASK_COMPLETE"
            )
            return {"status": "done" if complete else "budget_exhausted", "pending_calls": []}
        if malformed:
            if state["repair_count"] >= repair_limit:
                return {
                    "status": "invalid_output",
                    "pending_calls": [],
                    "error_details": ["Malformed tool calls exhausted the repair budget"],
                }
            messages.append(
                HumanMessage(
                    content="Tool-call repair: provide one valid tool name with schema-valid JSON arguments at a time, or finish if no tool is needed."
                )
            )
            emit_event(
                "tool_call_repair",
                agent="researcher",
                task_id=state["task_id"],
                attempt=state["repair_count"] + 1,
            )
            return {
                "messages": messages,
                "repair_count": state["repair_count"] + 1,
                "pending_calls": [],
                "status": "repair",
            }
        messages.append(response)
        return {
            "messages": messages,
            "pending_calls": response.tool_calls,
            "status": "tools" if response.tool_calls else "done",
        }

    def execute_tools(state: TaskState) -> dict[str, Any]:
        messages = list(state["messages"])
        records: list[ToolCallRecord] = []
        evidence: list[Evidence] = []
        results: list[ToolResult] = []
        errors: list[str] = []
        calls = state["calls_made"]
        for call in state["pending_calls"]:
            call_id = call.get("id") or str(uuid.uuid4())
            if calls >= state["max_calls"]:
                messages.append(ToolMessage(content="Tool budget exhausted", tool_call_id=call_id))
                continue
            name, arguments = call.get("name", ""), call.get("args", {})
            started = time.monotonic()
            calls += 1
            emit_event(
                "tool_started",
                agent="researcher",
                task_id=state["task_id"],
                call_id=call_id,
                tool=name,
                arguments=safe_arguments(arguments),
            )
            success = False
            detail = None
            try:
                raw = registry.execute(name, arguments)
                output = ToolOutput.model_validate_json(raw)
                if output.status == "error":
                    raise ToolExecutionError(output.error or "Tool retrieval failed")
                sources = collect(output, state["task_id"], name)
                evidence.extend(sources)
                results.append(
                    ToolResult(
                        call_id=call_id,
                        tool_name=name,
                        output=raw[:5000],
                        evidence_ids=[item.evidence_id for item in sources],
                    )
                )
                preview = raw[:5000].replace("<", "\\u003c").replace(">", "\\u003e")
                messages.append(
                    ToolMessage(
                        content=f"<RETRIEVED_CONTENT>\n{preview}\n</RETRIEVED_CONTENT>",
                        tool_call_id=call_id,
                    )
                )
                for item in sources:
                    emit_event(
                        "source_found",
                        agent="researcher",
                        task_id=state["task_id"],
                        evidence_id=item.evidence_id,
                        url=item.url,
                        title=item.title,
                        tool=name,
                    )
                success = True
            except Exception as error:
                detail = (
                    str(error) if isinstance(error, ToolExecutionError) else type(error).__name__
                )
                errors.append(f"Tool failed: {detail}")
                messages.append(
                    ToolMessage(
                        content=f"Tool error: {detail}. Check arguments or use another tool.",
                        tool_call_id=call_id,
                    )
                )
            duration = time.monotonic() - started
            record = ToolCallRecord(
                call_id=call_id,
                tool_name=name,
                arguments=safe_arguments(arguments),
                task_id=state["task_id"],
                success=success,
                error=detail,
                duration_seconds=duration,
            )
            records.append(record)
            emit_event(
                "tool_finished", agent="researcher", tool=name, **record.model_dump(mode="json")
            )
        return {
            "messages": messages,
            "pending_calls": [],
            "calls_made": calls,
            "tool_calls": records,
            "tool_results": results,
            "evidence": evidence,
            "error_details": errors,
            "status": "model",
        }

    graph = StateGraph(TaskState)
    graph.add_node("model", model)
    graph.add_node("execute_tools", execute_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", route_tool_turn, {"model": "model", "execute_tools": "execute_tools", END: END}
    )
    graph.add_edge("execute_tools", "model")
    return graph.compile(checkpointer=False, name="researcher_tool_loop")
