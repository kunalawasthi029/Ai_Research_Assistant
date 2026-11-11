"""AppTest drives the production HTTP client through FastAPI, real graph and SQLite."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from api_client import APIClient
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from research_agent.api.app import create_app
from research_agent.config.settings import Settings
from research_agent.service.research_service import ResearchService
from tests.api.conftest import scripted_worker as scripted_worker
from tests.frontend.conftest import FRONTEND


def test_native_ui_to_http_to_sqlite_report_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted_worker: Any
) -> None:
    import ui_state

    settings = Settings(
        groq_api_key="dummy-not-a-live-key",
        database_path=str(tmp_path / "runs.sqlite"),
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        overall_run_timeout_seconds=30,
    )
    clients = []
    with TestClient(create_app(settings)) as backend:

        def bridge(request: httpx.Request) -> httpx.Response:
            target = request.url.path + (
                "?" + request.url.query.decode() if request.url.query else ""
            )
            response = backend.request(
                request.method, target, content=request.content, headers=dict(request.headers)
            )
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        def factory(base_url: str) -> APIClient:
            client = APIClient(base_url)
            client._client.close()
            client._client = httpx.Client(
                base_url=base_url.rstrip("/") + "/", transport=httpx.MockTransport(bridge)
            )
            clients.append(client)
            return client

        monkeypatch.setattr(ui_state, "APIClient", factory)
        ui = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=20)
        ui.session_state["api_url"] = f"http://bridge-{uuid4().hex}.test/api/v1"
        ui.run()
        ui.text_area(key="research_query").set_value("What is X?")
        ui.selectbox(key="research_depth").select("brief")
        ui.selectbox(key="research_scope").select("broad")
        ui.multiselect(key="research_sources").set_value(["academic"])
        ui.button(key="start_research").click().run()
        assert not ui.exception and ui.title[0].value == "Live execution"
        ui.switch_page("app_pages/live.py").run()
        run_id = ui.session_state.selected_run_id
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            assert not ui.exception
            status = next(row.value for row in ui.metric if row.label == "Status")
            if status == "completed":
                break
            time.sleep(0.02)
            ui.run()
        assert status == "completed"
        assert not ui.get("progress")
        ui.button(key="open_live_report").click().run()
        assert ui.title[0].value == "Report"
        ui.switch_page("app_pages/report.py").run()
        assert not ui.exception and len(ui.get("download_button")) == 2
        report = json.loads(ui.json[0].value)
        with ResearchService(settings) as service:
            assert len(service.list_runs()) == 1
            assert report["references"][0]["evidence_id"] == service.get_evidence(run_id)[0].id
            assert service.get_run(run_id).status == "completed"
            assert len(service.get_validations(run_id)) == 3
            assert service.inspect_checkpoint(run_id)["research_preferences"]["scope"] == "broad"
        ui.toggle(key="show_sources").set_value(True).run()
        assert not ui.exception and any("X has been demonstrated" in row.value for row in ui.text)
        ui.switch_page("app_pages/history.py").run()
        assert ui.button(key="monitor_" + run_id)
    for client in clients:
        client.close()
