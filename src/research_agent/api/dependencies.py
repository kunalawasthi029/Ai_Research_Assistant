"""Dependency injection of immutable configuration and the lifespan-owned dispatcher."""

from __future__ import annotations

from fastapi import HTTPException, Request

from research_agent.api.jobs import JobManager
from research_agent.config.settings import Settings


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise HTTPException(status_code=503, detail="Research service is not initialized")
    return settings


def get_jobs(request: Request) -> JobManager:
    jobs = getattr(request.app.state, "jobs", None)
    if not isinstance(jobs, JobManager):
        raise HTTPException(status_code=503, detail="Research service is not initialized")
    return jobs
