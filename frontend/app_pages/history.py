"""History navigation carries a selected run into the other pages."""

from typing import Any

import streamlit as st
from api_client import APIError
from ui_state import client, select_run, show_error

limit = st.selectbox("Recent runs", [20, 50, 100], key="history_limit")
st.button("Refresh history", key="refresh_history")
runs: list[dict[str, Any]] | None
try:
    runs = client().list_runs(limit)
except APIError as error:
    show_error(error)
    runs = None
if runs == []:
    st.info("No research runs yet. Start from New research.")
if runs:
    for run in runs:
        run_id = run["run_id"]
        with st.container(border=True):
            st.subheader(str(run.get("query", "Research"))[:120])
            st.text(
                f"Status: {run.get('status', 'unknown')} | Sources: {run.get('evidence_count', 0)} | Tool calls: {run.get('tool_call_count', 0)}"
            )
            st.caption(f"Run: {run_id} | Created: {run.get('created_at', '')}")
            with st.container(horizontal=True):
                if st.button("Monitor run", key="monitor_" + run_id):
                    select_run(run_id)
                    st.switch_page("app_pages/live.py")
                if st.button("View report", key="report_" + run_id):
                    select_run(run_id)
                    st.switch_page("app_pages/report.py")
