"""Display persisted reports, plan, validation and optional source details."""

import json

import streamlit as st
from api_client import APIError
from ui_state import client, select_from_input, show_error
from views import plan_view, report_view, sources_view, status_message, validation_view

st.text_input(
    "Run ID",
    key="report_run_input",
    on_change=select_from_input,
    args=("report_run_input",),
    max_chars=100,
    persist_state="session",
)
run_id = st.session_state.selected_run_id
if not run_id:
    st.info("Choose a run from History or enter its ID.")
    st.stop()
try:
    api = client()
    run = api.get_run(run_id)
    data = api.get_report(run_id)
except APIError as error:
    if error.status_code == 404:
        st.info("A report is not available for this run yet.")
        if st.button("Open live execution", key="report_to_live"):
            st.session_state.live_run_input = run_id
            st.switch_page("app_pages/live.py")
    else:
        show_error(error)
    st.stop()
status_message(run.get("status", "unknown"))
st.caption(
    f"Version {data.get('version', 0)} | Saved: {data.get('created_at', '')} | Run: {run_id}"
)
report = data.get("report_json", {})
formatted, markdown, raw = st.tabs(["Report", "Markdown", "JSON"])
with formatted:
    report_view(report)
with markdown:
    st.code(data.get("report_markdown", ""), language="markdown")
with raw:
    st.json(report)
with st.container(horizontal=True):
    st.download_button(
        "Download JSON",
        json.dumps(report, ensure_ascii=False, indent=2),
        file_name="research-report.json",
        mime="application/json",
        key="download_json",
        on_click="ignore",
    )
    st.download_button(
        "Download Markdown",
        data.get("report_markdown", ""),
        file_name="research-report.md",
        mime="text/markdown",
        key="download_markdown",
        on_click="ignore",
    )
plan_view(run.get("research_plan"))
validation_view(run.get("validation_result"))
if st.toggle("Show source details", key="show_sources"):
    try:
        sources_view(api.get_sources(run_id))
    except APIError as error:
        show_error(error)
