"""Refresh observed events in a fragment instead of blocking the page."""

import streamlit as st
from api_client import APIError
from ui_state import client, event_state, select_from_input, show_error
from views import event_views, metrics_view, plan_view, status_message, validation_view

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = ""
st.text_input(
    "Run ID",
    key="live_run_input",
    on_change=select_from_input,
    args=("live_run_input",),
    max_chars=100,
    persist_state="session",
)


@st.fragment(
    run_every=2 if st.session_state.auto_refresh and st.session_state.live_should_poll else None
)
def live_panel() -> None:
    run_id = st.session_state.selected_run_id
    if not run_id:
        if st.session_state.live_should_poll:
            st.session_state.live_should_poll = False
            st.rerun()
        st.info("Start research or choose a run from History.")
        return
    st.button("Refresh now", key="refresh_live")
    try:
        api = client()
        run = api.get_run(run_id)
    except APIError as error:
        if error.status_code == 404 and st.session_state.live_should_poll:
            st.session_state.live_should_poll = False
            st.rerun()
        show_error(error)
        return
    live = event_state(run_id)
    try:
        live.ingest(api.get_events(run_id, after=live.cursor))
    except APIError as error:
        show_error(error)
        st.caption("Last observed events are retained. Refresh to retry.")
    status = run.get("status", "unknown")
    status_message(status)
    st.text(run.get("query", ""))
    st.caption("Run ID: " + run_id)
    metrics_view(run, live)
    preferences = run.get("research_preferences", {})
    if preferences:
        st.caption(
            f"Depth: {preferences.get('depth', 'standard')} | Scope: {preferences.get('scope', 'balanced')} | Preferred sources: {', '.join(preferences.get('source_preferences', [])) or 'any'}"
        )
    plan_view(run.get("research_plan"), expanded=status == "awaiting_approval")
    if status == "awaiting_approval":
        with st.container(horizontal=True):
            approve = st.button(
                "Approve plan",
                type="primary",
                disabled=bool(run.get("job_active")),
                key="approve_plan",
            )
            reject = st.button(
                "Reject plan", disabled=bool(run.get("job_active")), key="reject_plan"
            )
        if approve or reject:
            try:
                api.resume_research(run_id, approved=approve)
                st.session_state.live_should_poll = True
                st.rerun()
            except APIError as error:
                show_error(error)
    elif status == "interrupted" and not run.get("job_active"):
        if st.button("Resume from checkpoint", key="resume_checkpoint"):
            try:
                api.resume_research(run_id)
                st.session_state.live_should_poll = True
                st.rerun()
            except APIError as error:
                show_error(error)
    if run.get("report_version") is not None:
        if st.button("Open report", key="open_live_report", type="primary"):
            st.session_state.report_run_input = run_id
            st.switch_page("app_pages/report.py")
    validation_view(run.get("validation_result"))
    event_views(live)
    if live.has_more:
        st.caption(
            "Additional recorded events are available. Automatic refresh or Refresh now loads more."
        )
    for warning in run.get("warnings", []):
        st.warning(warning)
    should_poll = status in {"queued", "running"} or bool(run.get("job_active")) or live.has_more
    if should_poll != st.session_state.live_should_poll:
        st.session_state.live_should_poll = should_poll
        st.rerun()


live_panel()
