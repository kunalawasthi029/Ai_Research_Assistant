"""Deployment authentication protects data/jobs while allowing health checks."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from research_agent.api.app import create_app
from research_agent.config.settings import Settings


def test_required_auth_fails_closed_without_a_key() -> None:
    with pytest.raises(ValidationError, match="APP_API_KEY is required"):
        Settings(groq_api_key="test", require_api_auth=True)


def test_api_key_protects_reads_and_job_creation_but_not_health(settings: Settings) -> None:
    key = "test-server-key-" + "x" * 32
    settings = settings.model_copy(update={"app_api_key": SecretStr(key), "require_api_auth": True})
    assert "app_api_key" not in settings.model_dump()
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").status_code == 200
        for headers in ({}, {"X-API-Key": "wrong"}):
            response = client.get("/api/v1/research", headers=headers)
            assert response.status_code == 401
            assert key not in response.text
            assert response.headers["x-request-id"]
            assert (
                client.post(
                    "/api/v1/research", json={"query": "Never queued"}, headers=headers
                ).status_code
                == 401
            )
        assert client.get("/api/v1/research", headers={"X-API-Key": key}).json() == []
        assert client.get("/api/v1/tools", headers={"X-API-Key": key}).status_code == 200
        response = client.options(
            "/api/v1/research",
            headers={
                "Origin": "http://allowed.test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key,Content-Type",
            },
        )
        assert response.status_code == 200
