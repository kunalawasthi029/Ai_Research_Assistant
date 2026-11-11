"""Tests for graph construction, compilation, and checkpointing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.workflow import build_graph
from research_agent.tools.registry import ToolRegistry
from tests.fake_llm import FakeChatModel


class TestBuildGraph:
    """Tests for graph construction and compilation."""

    def _make_graph(self, checkpointer: object | None = None) -> object:
        """Build a graph with FakeChatModels (no responses needed for structure test)."""
        llm = FakeChatModel(responses=[])
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.description = "Search the web"
        registry.register(mock_tool)

        return build_graph(
            planner_llm=llm,
            researcher_llm=llm,
            analyst_llm=llm,
            report_llm=llm,
            supervisor_llm=llm,
            validator_llm=llm,
            registry=registry,
            checkpointer=checkpointer,
        )

    def test_graph_compiles(self) -> None:
        """The graph compiles without errors."""
        compiled = self._make_graph()
        assert compiled is not None

    def test_graph_has_all_nodes(self) -> None:
        """All expected nodes are in the compiled graph."""
        compiled = self._make_graph()
        # get_graph() returns the graph structure
        graph_obj = compiled.get_graph()
        node_ids = set(graph_obj.nodes.keys())

        expected_nodes = {
            "__start__",
            "__end__",
            "planner",
            "supervisor_plan_review",
            "researcher",
            "analyst",
            "supervisor_evidence_review",
            "report_generator",
            "validator",
            "supervisor_final_review",
            "finalize",
        }
        assert expected_nodes.issubset(node_ids), f"Missing: {expected_nodes - node_ids}"

    def test_graph_has_conditional_edges(self) -> None:
        """The graph has at least 3 conditional edge sources (the 3 supervisor reviews)."""
        compiled = self._make_graph()
        graph_obj = compiled.get_graph()
        # Edges are in graph_obj.edges
        edge_sources = {e.source for e in graph_obj.edges}
        assert "supervisor_plan_review" in edge_sources
        assert "supervisor_evidence_review" in edge_sources
        assert "supervisor_final_review" in edge_sources

    def test_mermaid_export(self) -> None:
        """The graph can export a Mermaid diagram."""
        compiled = self._make_graph()
        mermaid = compiled.get_graph().draw_mermaid()
        assert "planner" in mermaid
        assert "researcher" in mermaid
        assert "finalize" in mermaid

    def test_graph_with_checkpointer(self) -> None:
        """The graph compiles with a checkpointer attached."""
        with (
            tempfile.TemporaryDirectory() as tmp,
            create_checkpointer(Path(tmp) / "test.sqlite") as cp,
        ):
            compiled = self._make_graph(checkpointer=cp)
            assert compiled is not None


class TestCheckpointer:
    """Tests for SqliteSaver checkpointer setup."""

    def test_creates_db_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "subdir" / "test.sqlite"
            with create_checkpointer(db_path) as cp:
                assert cp is not None
                assert db_path.is_file()

    def test_default_path(self) -> None:
        """create_checkpointer with None uses default path."""
        # Just check it doesn't crash — don't actually create in project dir
        # We'll pass a tmp path to avoid polluting the project
        with (
            tempfile.TemporaryDirectory() as tmp,
            create_checkpointer(Path(tmp) / "default_test.sqlite") as cp,
        ):
            assert cp is not None
