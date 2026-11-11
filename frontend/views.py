"""Native Streamlit views of public backend artifacts."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import streamlit as st
from live_state import LiveState


def status_message(status: str) -> None:
    messages = {
        "completed": (st.success, "Research completed. The saved report is available."),
        "best_effort": (
            st.warning,
            "Best-effort report: some completion or validation checks were not met.",
        ),
        "incomplete": (st.warning, "Research is incomplete. A partial artifact may be available."),
        "interrupted": (
            st.warning,
            "Execution was interrupted. Resume requires a saved checkpoint and remaining budget.",
        ),
        "failed": (st.error, "Research failed. Review the execution events."),
        "awaiting_approval": (st.info, "Review the saved plan before approving or rejecting it."),
        "queued": (st.info, "Queued for an available worker."),
        "running": (st.info, "Research is in progress."),
    }
    show, message = messages.get(status, (st.info, "Backend status: " + status))
    show(message)


def plan_view(plan: dict[str, Any] | None, *, expanded: bool = False) -> None:
    with st.expander("Research plan", expanded=expanded):
        if not plan:
            st.caption("A plan has not been reported yet.")
            return
        st.text(plan.get("research_objective", ""))
        st.caption("Scope: " + str(plan.get("scope", "balanced")))
        tasks = [
            {
                "Task": row.get("task_id"),
                "Description": row.get("description"),
                "Status": row.get("status"),
                "Evidence needed": row.get("required_evidence"),
                "Completion criteria": row.get("completion_criteria"),
                "Source types": ", ".join(row.get("expected_source_types", [])),
            }
            for row in plan.get("research_tasks", [])
        ]
        if tasks:
            st.dataframe(
                tasks,
                hide_index=True,
                alt="Planned tasks, evidence requirements and observed task status",
            )


def validation_view(validation: dict[str, Any] | None) -> None:
    with st.expander("Validation", expanded=True):
        if not validation:
            st.caption("Validation has not been reported yet.")
            return
        st.text("Verdict: " + str(validation.get("status", "unknown")))
        st.metric("Assessment score", f"{validation.get('overall_score', 0):.2f} / 1")
        st.caption(
            "An assessment score is not proof of factual correctness. Review the issues and source text."
        )
        if validation.get("scores"):
            st.dataframe(
                validation["scores"],
                hide_index=True,
                alt="Validation criterion scores and pass flags",
            )
        for issue in validation.get("issues", []):
            st.text(f"{issue.get('severity', '')}: {issue.get('description', '')}")
        for claim in validation.get("unsupported_claims", []):
            st.text("Unsupported claim: " + str(claim))
        if validation.get("feedback"):
            st.text(validation["feedback"])


def metrics_view(run: dict[str, Any], live: LiveState) -> None:
    tokens = max(int(run.get("total_tokens", 0) or 0), live.tokens)
    first = st.columns(4)
    first[0].metric("Status", str(run.get("status", "unknown")).replace("_", " "))
    first[1].metric("Tool calls", max(run.get("total_tool_calls", 0), live.observed_tool_calls))
    first[2].metric("Sources", max(run.get("evidence_count", 0), len(live.sources)))
    first[3].metric("Reported tokens", tokens if tokens or live.usage_observed else "Not reported")
    second = st.columns(4)
    elapsed = run.get("elapsed_seconds")
    second[0].metric("Elapsed", f"{elapsed:.1f}s" if elapsed is not None else "Not reported")
    second[1].metric("Revisions", run.get("revision_count", 0))
    second[2].metric("Research rounds", run.get("research_iteration", 0))
    second[3].metric(
        "Observed model time",
        f"{live.model_seconds:.2f}s" if live.usage_observed else "Not reported",
    )
    active = run.get("job_active") and run.get("status") == "running"
    step = live.current_node if active else ""
    st.text(
        "Latest step: "
        + (step or str(run.get("current_agent") or "Not reported")).replace("_", " ")
    )
    task = run.get("active_task") or {}
    st.text("Latest task: " + str(task.get("description") or "Not reported"))
    if live.agent:
        st.caption("Latest observed agent: " + live.agent.replace("_", " "))


def event_views(live: LiveState) -> None:
    with st.expander("Tool activity", expanded=True):
        if not live.tools:
            st.caption("No tool calls have been reported yet.")
        else:
            st.dataframe(
                [
                    {
                        "Tool": item.get("tool") or item.get("tool_name"),
                        "Task": item.get("task_id"),
                        "Status": item.get("status"),
                        "Duration (s)": item.get("duration_seconds"),
                        "Error": item.get("error"),
                        "Arguments": str(item.get("arguments", {})),
                    }
                    for item in live.tools.values()
                ],
                hide_index=True,
                alt="Observed tool calls, arguments, results and durations",
            )
    with st.expander("Discovered sources", expanded=True):
        if live.sources:
            st.dataframe(
                list(live.sources.values()),
                hide_index=True,
                alt="Source URLs and titles discovered by backend tools",
            )
        else:
            st.caption("No sources have been reported yet.")
    with st.expander("Execution events"):
        if live.events:
            st.dataframe(
                live.events,
                hide_index=True,
                alt="Latest 200 persisted execution events in delivery order",
            )
        else:
            st.caption("No execution events have been received yet.")


def source_link(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"https", "http"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def sources_view(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No retrieved sources are available.")
    for index, source in enumerate(sources[:30], 1):
        with st.expander(f"{index}. {str(source.get('title', 'Source'))[:100]}"):
            st.text(f"Evidence: {source.get('evidence_id', '')}")
            st.text(f"Collected: {source.get('retrieved_at', '')}")
            st.text(f"Type: {source.get('source_type', '')}; tool: {source.get('tool_name', '')}")
            reachable = source.get("source", {}).get("is_accessible")
            st.caption(
                "URL reachability: "
                + (
                    "not assessed"
                    if reachable is None
                    else "reachable when checked"
                    if reachable
                    else "unreachable when checked"
                )
            )
            url = source.get("url", "")
            if source_link(url):
                st.link_button("Open source", url)
            st.text(str(source.get("content", ""))[:5000])
            for finding in source.get("findings", []):
                st.text("Finding: " + str(finding.get("claim", "")))
    if len(sources) > 30:
        st.caption(f"Showing 30 of {len(sources)} sources.")


def report_view(report: dict[str, Any]) -> None:
    st.subheader(report.get("title", "Research report"))
    for key, title in (
        ("executive_summary", "Executive summary"),
        ("research_objective", "Research objective"),
        ("methodology", "Methodology"),
    ):
        st.markdown("### " + title)
        st.markdown(report.get(key, ""))
    st.markdown("### Key findings")
    for item in report.get("key_findings", []):
        st.markdown("- " + str(item))
    for section in report.get("detailed_analysis", []):
        st.markdown("### " + str(section.get("heading", "")))
        st.markdown(section.get("content", ""))
    for key, title in (
        ("comparative_insights", "Comparative insights"),
        ("recommendations", "Recommendations"),
        ("limitations", "Limitations"),
    ):
        if report.get(key):
            st.markdown("### " + title)
            for item in report[key]:
                st.markdown("- " + str(item))
    st.markdown("### Conclusion")
    st.markdown(report.get("conclusion", ""))
    st.markdown("### References")
    for ref in report.get("references", []):
        st.text(
            f"[{ref.get('ref_id', '')}] {ref.get('title', '')} | evidence: {ref.get('evidence_id', '')}"
        )
        if source_link(ref.get("url", "")):
            st.link_button("Open reference " + str(ref.get("ref_id", "")), ref["url"])
