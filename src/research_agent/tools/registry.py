"""Tool registry for managing research tools."""

from __future__ import annotations

import logging
import threading
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from research_agent.config.settings import Settings

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages tool registration, lookup, and per-run execution limits.

    All tools are registered here and retrieved by name. The registry
    tracks call counts per run and enforces configurable per-tool limits.
    """

    def __init__(
        self, default_call_limit: int = 20, max_total_calls: int = 100, max_task_calls: int = 5
    ) -> None:
        if min(default_call_limit, max_total_calls, max_task_calls) < 1:
            raise ValueError("Tool budgets must be positive")
        self._tools: dict[str, BaseTool] = {}
        self._call_counts: dict[str, int] = {}
        self._limits: dict[str, int] = {}
        self._default_call_limit = default_call_limit
        self.max_total_calls = max_total_calls
        self.max_task_calls = max_task_calls
        self._lock = threading.RLock()
        self.settings: Settings | None = None
        self.isolate_calls = False
        self.extraction_focus = ""

    def register(self, tool: BaseTool, call_limit: int | None = None) -> None:
        """Register a tool with an optional per-run call limit."""
        name = tool.name
        if name in self._tools:
            logger.warning("Tool '%s' already registered — overwriting", name)
        self._tools[name] = tool
        self._call_counts[name] = 0
        if call_limit is not None and call_limit < 1:
            raise ValueError("Tool budget must be positive")
        self._limits[name] = call_limit if call_limit is not None else self._default_call_limit
        logger.info("Registered tool: %s (limit=%d)", name, self._limits[name])

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools as a list."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> list[dict[str, str]]:
        """Get name and description of each tool for API/UI display."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def can_call(self, name: str) -> bool:
        """Check if a tool has remaining calls within its limit."""
        if name not in self._tools:
            return False
        return (
            self._call_counts[name] < self._limits[name]
            and sum(self._call_counts.values()) < self.max_total_calls
        )

    def record_call(self, name: str) -> None:
        """Record a tool call. Raises if limit exceeded."""
        with self._lock:
            if not self.can_call(name):
                raise RuntimeError("Tool or run call limit exceeded")
            self._call_counts[name] += 1

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Reserve before validating/invoking so failed attempts consume the run budget."""
        tool = self.get_tool(name)
        self.record_call(name)
        if name == "webpage_extractor" and self.extraction_focus and not arguments.get("focus"):
            arguments = {**arguments, "focus": self.extraction_focus}
        if self.isolate_calls and self.settings is not None:
            from research_agent.execution.isolation import WorkerTimeoutError, run_isolated
            from research_agent.tools.results import failure

            if isinstance(tool, StructuredTool) and isinstance(tool.args_schema, type):
                tool.args_schema.model_validate(arguments)
            try:
                return str(
                    run_isolated(
                        _tool_worker,
                        (name, arguments, self.settings),
                        self.settings.per_tool_timeout_seconds,
                    )
                )
            except WorkerTimeoutError:
                return failure("Tool wall-clock timeout")
        return str(tool.invoke(arguments))

    def restore_counts(self, counts: dict[str, int]) -> None:
        """Restore persisted invocation reservations when resuming a run."""
        with self._lock:
            for name in self._tools:
                self._call_counts[name] = max(self._call_counts[name], counts.get(name, 0))

    def get_call_count(self, name: str) -> int:
        """Get the number of times a tool has been called this run."""
        return self._call_counts.get(name, 0)

    def reset_counts(self) -> None:
        """Reset all call counts (for a new run)."""
        for name in self._call_counts:
            self._call_counts[name] = 0

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def create_default_registry(
    call_limit: int = 20, settings: Settings | None = None, *, isolate_calls: bool = False
) -> ToolRegistry:
    """Create a ToolRegistry with all six research tools registered.

    Import and instantiate each tool here so the rest of the system
    gets a ready-to-use registry.
    """
    from research_agent.tools.arxiv_search import arxiv_search_tool
    from research_agent.tools.calculator import calculator_tool
    from research_agent.tools.source_verifier import source_verifier_tool
    from research_agent.tools.web_search import create_web_search_tool
    from research_agent.tools.webpage_extractor import webpage_extractor_tool
    from research_agent.tools.wikipedia_search import wikipedia_search_tool

    registry = ToolRegistry(
        default_call_limit=call_limit,
        max_total_calls=settings.max_tool_calls_per_run if settings else 100,
        max_task_calls=settings.max_tool_calls_per_task if settings else 5,
    )
    registry.settings = settings
    registry.isolate_calls = isolate_calls and settings is not None
    for tool in (
        create_web_search_tool(),
        webpage_extractor_tool,
        wikipedia_search_tool,
        arxiv_search_tool,
        calculator_tool,
        source_verifier_tool,
    ):
        configured = tool
        if (
            settings
            and isinstance(tool, StructuredTool)
            and tool.func
            and tool.name != "calculator"
        ):
            options: dict[str, Any] = {"timeout_seconds": settings.per_tool_timeout_seconds}
            if tool.name == "web_search":
                options["api_key"] = settings.tavily_api_key or ""
                options.update(
                    search_depth=settings.search_depth,
                    result_limit=settings.search_max_results,
                    fallback_enabled=settings.search_fallback_enabled,
                    cache_path=settings.search_cache_path,
                    cache_ttl_seconds=settings.search_cache_ttl_seconds,
                )
            configured = tool.model_copy(update={"func": partial(tool.func, **options)})
        registry.register(configured)
    return registry


def _tool_worker(channel: Any, name: str, arguments: dict[str, Any], settings: Settings) -> str:
    """Execute one registered public tool in an independently terminable process."""
    registry = create_default_registry(settings=settings)
    return str(registry.get_tool(name).invoke(arguments))
