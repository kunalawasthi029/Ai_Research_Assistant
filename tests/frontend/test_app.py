"""Native Streamlit widget/navigation journeys, mocking only the HTTP client boundary."""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from tests.frontend.conftest import FRONTEND, FakeAPI, frame


def test_deployment_api_url_initializes_session_and_preserves_user_change(
    fake_api: FakeAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_API_URL", "https://backend.example.org/api/v1")
    ui = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=15).run()
    assert not ui.exception
    assert ui.text_input(key="api_url").value == "https://backend.example.org/api/v1"
    ui.text_input(key="api_url").set_value("https://other.example.org/api/v1").run()
    assert ui.text_input(key="api_url").value == "https://other.example.org/api/v1"


def test_cloud_secrets_api_url_without_backend_credentials(fake_api: FakeAPI) -> None:
    ui = AppTest.from_file(str(FRONTEND / "app.py"))
    ui.secrets["RESEARCH_API_URL"] = "https://cloud-backend.example.org/api/v1"
    ui.run()
    assert not ui.exception
    assert ui.text_input(key="api_url").value == "https://cloud-backend.example.org/api/v1"


def test_first_submission_keeps_untouched_backend_and_refresh_defaults(
    fake_api: FakeAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_API_URL", "http://initial-api.test/api/v1")
    ui = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=15).run()
    assert ui.toggle(key="auto_refresh").value is True
    submit(ui)
    assert ui.text_input(key="api_url").value == "http://initial-api.test/api/v1"
    assert ui.toggle(key="auto_refresh").value is True
    assert (
        len([row for row in fake_api.calls if isinstance(row, tuple) and row[0] == "submission"])
        == 1
    )


def submit(ui: AppTest) -> AppTest:
    ui.text_area(key="research_query").set_value("Compare retrieval approaches")
    ui.button(key="start_research").click().run()
    assert not ui.exception
    assert ui.title[0].value == "Live execution"
    # AppTest renders st.switch_page immediately but its next explicit run keeps
    # the harness-selected page. Pin that page through the documented test API.
    ui.switch_page("app_pages/live.py").run()
    return ui


def metric(ui: AppTest, label: str) -> str:
    return next(item.value for item in ui.metric if item.label == label)


def test_sidebar_backend_and_refresh_preference_survive_submit_and_page_navigation(
    ui: AppTest,
) -> None:
    backend = "http://custom-api.test/api/v1"
    ui.text_input(key="api_url").set_value(backend).run()
    ui.toggle(key="auto_refresh").set_value(False).run()
    ui = submit(ui)
    assert ui.text_input(key="api_url").value == backend
    assert ui.toggle(key="auto_refresh").value is False
    selected = ui.session_state["selected_run_id"]
    assert selected
    for page in ("app_pages/history.py", "app_pages/report.py", "app_pages/live.py"):
        ui.switch_page(page).run()
        assert not ui.exception
        assert ui.text_input(key="api_url").value == backend
        assert ui.toggle(key="auto_refresh").value is False
        assert ui.session_state["selected_run_id"] == selected


def test_new_research_sends_async_options_and_retains_identity(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    assert not ui.exception
    ui.selectbox(key="research_depth").select("deep")
    ui.selectbox(key="research_scope").select("narrow")
    ui.multiselect(key="research_sources").set_value(["academic"])
    ui.text_area(key="research_objective").set_value("Compare evaluation methods")
    submit(ui)
    assert ui.session_state.selected_run_id == "run-1"
    sent = next(row for row in fake_api.calls if isinstance(row, tuple) and row[0] == "submission")
    assert sent[2] == "Compare evaluation methods"
    assert sent[3] == {
        "depth": "deep",
        "scope": "narrow",
        "source_preferences": ["academic"],
        "human_approval_enabled": False,
    }
    assert metric(ui, "Status") == "running"
    assert len(fake_api.runs) == 1 and not ui.get("progress")
    ui.run()
    assert len(fake_api.runs) == 1 and not ui.exception


def test_empty_question_and_api_outage_do_not_fabricate_success(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    ui.button(key="start_research").click().run()
    assert ui.warning and "start" not in fake_api.calls
    fake_api.fail.add("start")
    ui.text_area(key="research_query").set_value("Compare retrieval approaches")
    ui.button(key="start_research").click().run()
    assert ui.title[0].value == "New research"
    assert ui.error and "request-error" in " ".join(row.value for row in ui.caption)
    assert ui.session_state.selected_run_id == ""
    assert fake_api.runs == {}


def test_live_data_uses_real_events_and_deduplicates_refresh(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    submit(ui)
    fake_api.events["run-1"] = [
        frame(1, "node_started", {}, node="researcher"),
        frame(
            2,
            "tool_started",
            {
                "call_id": "call",
                "task_id": "t1",
                "tool": "web_search",
                "arguments": {"query": "sources"},
            },
        ),
        frame(
            3,
            "source_found",
            {"evidence_id": "ev1", "title": "Observed source", "url": "https://example.com/source"},
        ),
        frame(
            4,
            "tool_finished",
            {
                "call_id": "call",
                "tool_name": "web_search",
                "task_id": "t1",
                "success": False,
                "error": "Tool wall-clock timeout",
                "duration_seconds": 0.4,
            },
        ),
        frame(5, "llm_call", {"total_tokens": 31, "duration_seconds": 0.25}),
    ]
    ui.run()
    assert not ui.exception
    assert metric(ui, "Reported tokens") == "31" and metric(ui, "Observed model time") == "0.25s"
    assert metric(ui, "Tool calls") == "1" and metric(ui, "Sources") == "1"
    tools = next(table.value for table in ui.dataframe if "Tool" in table.value.columns)
    assert tools.iloc[0]["Status"] == "failed"
    assert ui.session_state.live_events.cursor == 5
    ui.run()
    assert (
        ui.session_state.live_events.tokens == 31
        and ui.session_state.live_events.observed_tool_calls == 1
    )
    assert ("after", 5) in fake_api.calls
    assert not ui.get("progress")


def test_event_error_retains_observed_data_and_retries(ui: AppTest, fake_api: FakeAPI) -> None:
    submit(ui)
    fake_api.events["run-1"] = [frame(1, "llm_call", {"total_tokens": 31, "duration_seconds": 0.2})]
    ui.run()
    fake_api.fail.add("events")
    ui.run()
    assert ui.error and metric(ui, "Reported tokens") == "31"
    assert ui.session_state.live_events.cursor == 1
    fake_api.fail.clear()
    ui.run()
    assert not ui.error and not ui.exception


def test_approval_and_rejection_are_explicit_controls(ui: AppTest, fake_api: FakeAPI) -> None:
    ui.checkbox(key="human_approval").check()
    submit(ui)
    assert metric(ui, "Status") == "awaiting approval"
    assert not ui.session_state.live_should_poll
    ui.button(key="approve_plan").click().run()
    assert ("approval", True) in fake_api.calls and metric(ui, "Status") == "running"
    ui.switch_page("app_pages/new_research.py").run()
    submit(ui)
    ui.button(key="reject_plan").click().run()
    assert ("approval", False) in fake_api.calls
    assert metric(ui, "Status") == "incomplete" and not ui.session_state.live_should_poll


def test_completed_history_report_sources_and_downloads_remain_linked(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    submit(ui)
    fake_api.complete()
    ui.run()
    assert not ui.session_state.live_should_poll
    ui.switch_page("app_pages/history.py").run()
    ui.button(key="report_run-1").click().run()
    assert ui.title[0].value == "Report"
    ui.switch_page("app_pages/report.py").run()
    assert not ui.exception and ui.session_state.selected_run_id == "run-1"
    assert any(row.value == "Retrieval comparison" for row in ui.subheader)
    assert len(ui.get("download_button")) == 2
    assert json.loads(ui.json[0].value)["references"][0]["evidence_id"] == "ev1"
    ui.toggle(key="show_sources").set_value(True).run()
    assert not ui.exception and any("Observed source text" in row.value for row in ui.text)
    assert any("not assessed" in row.value for row in ui.caption)
    ui.switch_page("app_pages/history.py").run()
    ui.button(key="monitor_run-1").click().run()
    assert ui.session_state.selected_run_id == "run-1" and metric(ui, "Status") == "completed"


def test_best_effort_interrupted_and_report_not_ready_states(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    submit(ui)
    ui.switch_page("app_pages/report.py").run()
    assert any("not available" in row.value for row in ui.info)
    assert not ui.get("download_button")
    ui.button(key="report_to_live").click().run()
    assert ui.title[0].value == "Live execution"
    ui.switch_page("app_pages/live.py").run()
    fake_api.complete(status="best_effort")
    fake_api.runs["run-1"]["validation_result"]["status"] = "needs_revision"
    ui.run()
    assert ui.warning and not any("Research completed" in row.value for row in ui.success)
    fake_api.runs["run-1"].update(status="interrupted", job_active=False)
    ui.run()
    ui.button(key="resume_checkpoint").click().run()
    assert ("approval", None) in fake_api.calls and metric(ui, "Status") == "running"


def test_history_empty_errors_and_api_url_change_are_isolated(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    ui.switch_page("app_pages/history.py").run()
    assert any("No research" in row.value for row in ui.info)
    fake_api.fail.add("history")
    ui.run()
    assert ui.error and not ui.exception
    fake_api.fail.clear()
    ui.switch_page("app_pages/new_research.py").run()
    submit(ui)
    ui.text_input(key="api_url").set_value("http://another-api.test/api/v1").run()
    assert ui.session_state.selected_run_id == "" and ui.session_state.live_events is None
    assert fake_api.closed > 0 and not ui.exception


def test_session_selection_does_not_leak_to_another_apptest(ui: AppTest, fake_api: FakeAPI) -> None:
    submit(ui)
    other = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=15).run()
    assert not other.exception and other.session_state.selected_run_id == ""


def test_completed_run_drains_event_backlog_before_stopping_refresh(
    ui: AppTest, fake_api: FakeAPI
) -> None:
    submit(ui)
    fake_api.complete()
    fake_api.events["run-1"] = [
        frame(index, "llm_call", {"total_tokens": 2, "duration_seconds": 0.1})
        for index in range(1, 251)
    ]
    ui.run()
    assert ui.session_state.live_events.cursor == 200 and ui.session_state.live_should_poll
    ui.run()
    assert not ui.exception and ui.session_state.live_events.cursor == 250
    assert not ui.session_state.live_should_poll
    assert len(ui.session_state.live_events.events) == 200
    assert metric(ui, "Reported tokens") == "500"


def test_source_failure_preserves_report_and_downloads(ui: AppTest, fake_api: FakeAPI) -> None:
    submit(ui)
    fake_api.complete()
    ui.switch_page("app_pages/report.py").run()
    fake_api.fail.add("sources")
    ui.toggle(key="show_sources").set_value(True).run()
    assert ui.error and not ui.exception
    assert len(ui.get("download_button")) == 2
