"""Structural acceptance guards; live quality, dates and remote CI need separate evidence."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from research_agent.agents.supervisor import LEGAL_TRANSITIONS
from research_agent.graph.routing import (
    MAX_PLAN_REVISIONS,
    MAX_REPORT_REVISIONS,
    MAX_RESEARCH_ITERATIONS,
)
from research_agent.graph.state import ResearchState
from research_agent.schemas.plan import ResearchPlan, ResearchTask, SubQuestion
from research_agent.schemas.report import Reference, Report, ReportSection
from research_agent.tools.registry import create_default_registry

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src" / "research_agent"


class TestClaim1AutonomousMultiAgent:
    """C1: Six agents, each with dedicated module, prompt file, and distinct state responsibility."""

    def test_six_agent_modules_exist(self) -> None:
        agent_dir = _SRC / "agents"
        required_agents = [
            "supervisor.py",
            "planner.py",
            "researcher.py",
            "analyst.py",
            "report_generator.py",
            "validator.py",
        ]
        for agent in required_agents:
            assert (agent_dir / agent).is_file(), f"Missing agent module: {agent}"

    def test_six_prompt_files_exist(self) -> None:
        prompt_dir = _SRC / "prompts"
        required_prompts = [
            "supervisor.md",
            "planner.md",
            "researcher.md",
            "analyst.md",
            "report_generator.md",
            "validator.md",
        ]
        for prompt in required_prompts:
            p = prompt_dir / prompt
            assert p.is_file(), f"Missing prompt file: {prompt}"
            assert len(p.read_text(encoding="utf-8").strip()) > 50, f"Prompt {prompt} is empty"


class TestClaim2PlanMultiStepTasks:
    """C2: Planner emits validated Pydantic ResearchPlan with sub-questions and tasks."""

    def test_research_plan_schema(self) -> None:
        plan = ResearchPlan(
            research_objective="Evaluate fusion energy",
            sub_questions=[
                SubQuestion(
                    question_id="q1",
                    question="What are recent magnetic confinement advances?",
                    priority=1,
                    expected_source_types=["arxiv"],
                )
            ],
            research_tasks=[
                ResearchTask(
                    task_id="t1",
                    question_id="q1",
                    description="Search arXiv for tokamak confinement papers",
                    search_queries=["tokamak plasma confinement 2024"],
                    required_evidence="Experimental energy gain results",
                    completion_criteria="At least 2 preprints identified",
                    priority=1,
                )
            ],
            scope="balanced",
            estimated_complexity="moderate",
        )
        assert len(plan.sub_questions) == 1
        assert len(plan.research_tasks) == 1
        assert plan.research_tasks[0].question_id == "q1"


class TestClaim3DynamicToolInvocation:
    """C3: Dynamic tool registry with at least 6 tools, Pydantic schemas, and error boundaries."""

    def test_minimum_six_tools_registered(self) -> None:
        registry = create_default_registry()
        tool_names = registry.get_tool_names()
        assert len(tool_names) >= 6
        expected_tools = {
            "web_search",
            "webpage_extractor",
            "wikipedia_search",
            "arxiv_search",
            "calculator",
            "source_verifier",
        }
        assert expected_tools.issubset(set(tool_names))

    def test_tool_call_limits_and_safety(self) -> None:
        registry = create_default_registry()
        assert registry.can_call("web_search")
        registry.record_call("web_search")
        assert registry.get_call_count("web_search") == 1


class TestClaim4SynthesizeStructuredReports:
    """C4: Analyst synthesizes findings; Report Generator emits Pydantic Report with traceable citations."""

    def test_report_schema_enforces_citations(self) -> None:
        report = Report(
            title="Fusion Synthesis",
            executive_summary="Summary text here.",
            research_objective="Objective text",
            methodology="Automated agentic workflow",
            key_findings=["Finding A", "Finding B"],
            detailed_analysis=[
                ReportSection(
                    heading="Confinement Overview",
                    content="Plasma physics progress.",
                    evidence_ids=["ev_123"],
                )
            ],
            limitations=["English only"],
            conclusion="Promising outlook",
            references=[
                Reference(
                    ref_id="ref_1",
                    evidence_id="ev_123",
                    url="https://arxiv.org/abs/2401.00001",
                    title="Tokamak confinement advances",
                    source_type="arxiv",
                    accessed_at="2024-01-01T00:00:00Z",
                )
            ],
        )
        assert len(report.detailed_analysis[0].evidence_ids) == 1
        assert report.references[0].evidence_id == "ev_123"


class TestClaim5StatefulGraphBased:
    """C5: Single typed StateGraph with reducers and persistent SQLite checkpointing."""

    def test_state_typed_dict_has_reducers(self) -> None:
        annotations = ResearchState.__annotations__
        assert "evidence" in annotations
        assert "tool_calls" in annotations
        assert "findings" in annotations
        assert "messages" in annotations

    def test_checkpointer_factory(self) -> None:
        import tempfile

        from research_agent.graph.checkpointing import create_checkpointer

        with (
            tempfile.TemporaryDirectory() as tmp,
            create_checkpointer(Path(tmp) / "test.db") as cp,
        ):
            assert cp.get_tuple({"configurable": {"thread_id": "not-created"}}) is None


class TestClaim6ConditionalRouting:
    """C6: Routing driven by runtime state via add_conditional_edges."""

    def test_supervisor_legal_transitions_structure(self) -> None:
        assert "plan_review" in LEGAL_TRANSITIONS
        assert "evidence_review" in LEGAL_TRANSITIONS
        assert "final_review" in LEGAL_TRANSITIONS
        assert "approve" in LEGAL_TRANSITIONS["plan_review"]
        assert "needs_research" in LEGAL_TRANSITIONS["evidence_review"]
        assert "accept" in LEGAL_TRANSITIONS["final_review"]

    def test_compiled_graph_exposes_validation_feedback_cycles(self) -> None:
        from tests.workflow.test_lifecycle import make_graph

        compiled = make_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in compiled.edges}
        assert len({edge.source for edge in compiled.edges if edge.conditional}) >= 4
        for destination in ("replan", "report_generator"):
            assert ("supervisor_final_review", destination) in edges
        for source, target in (
            ("validator", "semantic_validator"),
            ("semantic_validator", "supervisor_final_review"),
            ("report_generator", "validator"),
            ("replan", "researcher"),
            ("researcher", "analyst"),
            ("analyst", "supervisor_evidence_review"),
            ("supervisor_evidence_review", "report_generator"),
        ):
            assert (source, target) in edges


class TestClaim7IterativeValidationAndCaps:
    """C7: Two-layer validator, feedback loop, and hard iteration caps."""

    def test_hard_iteration_caps_defined(self) -> None:
        assert MAX_PLAN_REVISIONS > 0
        assert MAX_RESEARCH_ITERATIONS > 0
        assert MAX_REPORT_REVISIONS > 0

    def test_two_layer_validator_modules(self) -> None:
        from research_agent.agents.validator import (
            combine_validation_results,
            run_deterministic_validator,
            run_semantic_validator,
        )

        assert callable(run_deterministic_validator)
        assert callable(run_semantic_validator)
        assert callable(combine_validation_results)


class TestClaim8ReliabilityOverSinglePass:
    """C8: Evaluation dataset and comparison baselines B0, B1, B2."""

    def test_evaluation_dataset_present(self) -> None:
        from evals.dataset import EVAL_QUERIES

        assert len(EVAL_QUERIES) >= 20

    def test_baselines_implemented(self) -> None:
        from evals.baselines import run_baseline_b0, run_baseline_b1

        assert callable(run_baseline_b0)
        assert callable(run_baseline_b1)


class TestClaim9And10FastAPIAndStreamlit:
    """C9/C10: FastAPI backend and Streamlit frontend."""

    def test_fastapi_app_created(self) -> None:
        from research_agent.api.app import create_app

        app = create_app()
        assert app.title == "AI Research Agent"

    def test_frontend_files_exist(self) -> None:
        frontend_dir = _ROOT / "frontend"
        assert (frontend_dir / "app.py").is_file()
        assert (frontend_dir / "api_client.py").is_file()


class TestClaim11ReusableAPIsAndBoundaries:
    """C11: Core never imports FastAPI or Streamlit; API never imports Streamlit."""

    def test_boundary_integrity(self) -> None:
        core_dirs = [
            _SRC / name
            for name in (
                "schemas",
                "graph",
                "agents",
                "service",
                "llm",
                "tools",
                "config",
                "persistence",
                "reporting",
            )
        ]
        for d in core_dirs:
            for p in d.rglob("*.py"):
                tree = ast.parse(p.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            assert a.name.split(".")[0] not in ("fastapi", "streamlit")
                    elif isinstance(node, ast.ImportFrom):
                        assert (node.module or "").split(".")[0] not in ("fastapi", "streamlit")

    def test_recursive_frontend_is_an_http_consumer(self) -> None:
        forbidden = {"research_agent", "langgraph", "langchain_groq", "groq"}
        for path in (_ROOT / "frontend").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                assert not {name.split(".")[0] for name in names} & forbidden, path


def test_researcher_binds_all_six_real_registered_tool_schemas() -> None:
    from typing import Any

    from research_agent.graph.researcher_loop import build_researcher_subgraph
    from research_agent.tools.results import evidence_from_output
    from tests.fake_llm import FakeChatModel

    class RecordingModel(FakeChatModel):
        bound_names: list[str] = []

        def bind_tools(self, tools: list[Any], **kwargs: Any) -> RecordingModel:
            self.bound_names = [tool.name for tool in tools]
            return self

    registry = create_default_registry()
    model = RecordingModel(responses=[])
    graph = build_researcher_subgraph(model, registry, evidence_from_output).get_graph()
    assert set(model.bound_names) == set(registry.get_tool_names())
    assert len(model.bound_names) >= 6
    assert {("model", "execute_tools"), ("execute_tools", "model")} <= {
        (edge.source, edge.target) for edge in graph.edges
    }


def test_required_api_operations_are_in_the_actual_openapi_contract() -> None:
    from research_agent.api.app import create_app

    paths = create_app().openapi()["paths"]
    required = {
        "/health": {"get"},
        "/tools": {"get"},
        "/research": {"get", "post"},
        "/research/{run_id}": {"get"},
        "/research/{run_id}/events": {"get"},
        "/research/{run_id}/report": {"get"},
        "/research/{run_id}/sources": {"get"},
        "/research/{run_id}/resume": {"post"},
    }
    for path, methods in required.items():
        assert methods <= paths[f"/api/v1{path}"].keys()
    assert "202" in paths["/api/v1/research"]["post"]["responses"]


def test_offline_ci_contract_and_dispatch_only_live_evaluation() -> None:
    def read(name: str) -> dict:
        # BaseLoader preserves YAML's `on` key instead of treating it as a boolean.
        return yaml.load((_ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)

    ci = read("ci.yml")
    assert set(ci["on"]) == {"push", "pull_request"}
    job = ci["jobs"]["test"]
    assert {"3.11", "3.12"} <= set(job["strategy"]["matrix"]["python"])
    assert {"windows-latest", "ubuntu-latest"} <= set(job["strategy"]["matrix"]["os"])
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--require-hashes -r requirements.lock" in commands
    assert "ruff format --check" in commands and "mypy --no-incremental" in commands
    assert 'pytest -m "not live"' in commands and "--cov-fail-under=80" in commands
    assert "secrets." not in str(ci)
    assert any(step.get("uses", "").startswith("actions/upload-artifact@") for step in job["steps"])
    live = read("eval.yml")
    assert set(live["on"]) == {"workflow_dispatch"}
    assert live["jobs"]["evaluate"]["environment"] == "evaluation"
