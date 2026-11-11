"""Per-session HTTP resources and shared run selection for the four UI pages."""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import streamlit as st
from api_client import APIClient, APIError
from dotenv import dotenv_values
from live_state import LiveState


def configuration(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    with suppress(FileNotFoundError):
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    return dotenv_values(Path(__file__).resolve().parents[1] / ".env").get(name) or ""


def configured_api_url() -> str:
    return configuration("RESEARCH_API_URL") or "http://localhost:8000/api/v1"


@st.cache_resource(
    scope="session",
    ttl=900,
    max_entries=1,
    show_spinner=False,
    on_release=lambda client: client.close(),
)
def get_client(base_url: str) -> APIClient:
    key = configuration("APP_API_KEY")
    if key:
        if base_url.strip().rstrip("/") != configured_api_url().strip().rstrip("/"):
            raise APIError("The backend URL is managed by the deployment configuration.")
        return APIClient(base_url, api_key=key)
    return APIClient(base_url)


@st.cache_data(scope="session", ttl=15, max_entries=4, show_spinner=False)
def health(base_url: str) -> dict[str, Any]:
    return get_client(base_url).health_check()


def initialize() -> None:
    defaults = {
        "api_url": configured_api_url(),
        "selected_run_id": "",
        "live_run_input": "",
        "report_run_input": "",
        "auto_refresh": True,
        "live_run_id": "",
        "live_events": None,
        "live_should_poll": True,
        "flash": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_backend() -> None:
    get_client.clear()
    health.clear()
    for key in ("selected_run_id", "live_run_input", "report_run_input", "live_run_id", "flash"):
        st.session_state[key] = ""
    st.session_state.live_events = None
    st.session_state.live_should_poll = True


def select_run(run_id: str) -> None:
    st.session_state.selected_run_id = run_id
    st.session_state.live_run_input = run_id
    st.session_state.report_run_input = run_id
    st.session_state.live_should_poll = True


def select_from_input(key: str) -> None:
    st.session_state.selected_run_id = st.session_state[key].strip()
    other = "report_run_input" if key == "live_run_input" else "live_run_input"
    st.session_state[other] = st.session_state.selected_run_id
    st.session_state.live_should_poll = True


def event_state(run_id: str) -> LiveState:
    if st.session_state.live_run_id != run_id or st.session_state.live_events is None:
        st.session_state.live_run_id = run_id
        st.session_state.live_events = LiveState()
    return cast(LiveState, st.session_state.live_events)


def client() -> APIClient:
    return get_client(st.session_state.api_url)


def show_error(error: APIError) -> None:
    st.error(str(error))
    if error.request_id:
        st.caption(f"Request ID: {error.request_id}")
