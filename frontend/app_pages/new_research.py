"""Submit a research run without waiting for inference."""

import streamlit as st
from api_client import APIError
from ui_state import client, select_run, show_error

st.caption(
    "Set a question and research preferences. Execution continues in the API after submission."
)
with st.form("new_research_form"):
    query = st.text_area(
        "Research question",
        key="research_query",
        max_chars=4000,
        height=110,
        placeholder="What would you like to investigate?",
        persist_state="session",
    )
    objective = st.text_area(
        "Detailed instructions",
        key="research_objective",
        max_chars=4000,
        placeholder="Focus, comparison criteria, time period or intended audience",
        persist_state="session",
    )
    left, right = st.columns(2)
    depth = left.selectbox(
        "Depth", ["standard", "brief", "deep"], key="research_depth", persist_state="session"
    )
    scope = right.selectbox(
        "Scope", ["balanced", "narrow", "broad"], key="research_scope", persist_state="session"
    )
    sources = st.multiselect(
        "Preferred source types",
        ["web", "academic", "wikipedia"],
        key="research_sources",
        persist_state="session",
    )
    approval = st.checkbox(
        "Review the plan before research begins", key="human_approval", persist_state="session"
    )
    st.caption(
        "Brief uses a smaller research budget. Deep requests more analysis within server limits. Source choices guide planning; availability is not guaranteed."
    )
    submitted = st.form_submit_button("Start research", type="primary", key="start_research")

if submitted:
    if not 3 <= len(query.strip()) <= 4000:
        st.warning("Enter a research question of 3 to 4,000 characters.")
    else:
        try:
            with st.spinner("Submitting research..."):
                result = client().start_research(
                    query.strip(),
                    objective.strip(),
                    options={
                        "depth": depth,
                        "scope": scope,
                        "source_preferences": sources,
                        "human_approval_enabled": approval,
                    },
                )
            select_run(result["run_id"])
            st.session_state.flash = "Research accepted. This run remains available in History."
            st.switch_page("app_pages/live.py")
        except APIError as error:
            show_error(error)
            st.caption(
                "If acceptance could not be confirmed, check History before submitting again."
            )
