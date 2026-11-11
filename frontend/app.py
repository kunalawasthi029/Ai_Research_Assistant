"""HTTP-only Streamlit entrypoint; run from the project root."""

from __future__ import annotations

import streamlit as st
from api_client import APIError
from ui_state import configuration, health, initialize, reset_backend

st.set_page_config(page_title="Research workspace", layout="wide")
initialize()
# Resolve the initial page before registering persistent widgets. Otherwise the
# first browser rerun can restore their defaults under a different page identity.
page = st.navigation(
    [
        st.Page(
            "app_pages/new_research.py",
            title="New research",
            icon=":material/search:",
            default=True,
        ),
        st.Page("app_pages/live.py", title="Live execution", icon=":material/monitoring:"),
        st.Page("app_pages/report.py", title="Report", icon=":material/description:"),
        st.Page("app_pages/history.py", title="History", icon=":material/history:"),
    ],
    position="top",
)
with st.sidebar:
    st.header("Research workspace")
    st.text_input(
        "API URL",
        key="api_url",
        on_change=reset_backend,
        persist_state="session",
        disabled=bool(configuration("APP_API_KEY")),
    )
    if st.button("Check API", key="check_api"):
        health.clear()
    try:
        result = health(st.session_state.api_url)
        st.caption("API: available" if result.get("status") == "ok" else "API: unavailable")
    except APIError:
        st.caption("API: unavailable. Check the URL and server.")
    st.toggle("Refresh live data automatically", key="auto_refresh", persist_state="session")
    st.caption("Live data refreshes every two seconds while its page is open.")
st.title(page.title)
page.run()
