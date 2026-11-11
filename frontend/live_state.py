"""Bounded projection of observed backend events; no inferred progress percentage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiveState:
    cursor: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_node: str = ""
    agent: str = ""
    tokens: int = 0
    model_seconds: float = 0.0
    usage_observed: bool = False
    observed_tool_calls: int = 0
    has_more: bool = False

    def ingest(self, frames: list[dict[str, Any]]) -> None:
        for frame in frames:
            if frame.get("event_type") == "snapshot":
                self.has_more = bool(frame["data"].get("has_more"))
                continue
            sequence = frame.get("event_id")
            if not isinstance(sequence, int) or sequence <= self.cursor:
                continue
            payload = frame["data"]
            data = payload.get("data", {})
            kind = frame["event_type"]
            self.cursor = sequence
            self.events.append(
                {
                    "Time": payload.get("timestamp", ""),
                    "Event": kind,
                    "Agent": payload.get("agent", ""),
                    "Step": payload.get("node", ""),
                    "Detail": self._detail(kind, data),
                }
            )
            self.events = self.events[-200:]
            if kind == "node_started":
                self.current_node = payload.get("node", "")
            elif kind == "node_finished" and self.current_node == payload.get("node"):
                self.current_node = ""
            if payload.get("agent"):
                self.agent = payload["agent"]
            if kind in {"tool_started", "tool_finished"}:
                if kind == "tool_started":
                    self.observed_tool_calls += 1
                call = str(data.get("call_id", sequence))
                self.tools[call] = {
                    **self.tools.get(call, {}),
                    **data,
                    "status": "running"
                    if kind == "tool_started"
                    else "succeeded"
                    if data.get("success")
                    else "failed",
                }
                if len(self.tools) > 200:
                    self.tools.pop(next(iter(self.tools)))
            elif kind == "source_found":
                self.sources[str(data.get("evidence_id", sequence))] = data
                if len(self.sources) > 200:
                    self.sources.pop(next(iter(self.sources)))
            elif kind == "llm_call":
                self.usage_observed = True
                self.tokens += int(data.get("total_tokens", 0) or 0)
                self.model_seconds += float(data.get("duration_seconds", 0) or 0)

    @staticmethod
    def _detail(kind: str, data: dict[str, Any]) -> str:
        if kind == "decision":
            return str(data.get("selected_action", ""))
        if kind in {"tool_started", "tool_finished"}:
            return str(data.get("tool") or data.get("tool_name", ""))
        if kind == "source_found":
            return str(data.get("title", ""))
        return str(data.get("status") or data.get("error_type") or "")
