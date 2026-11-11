"""Controlled tests of benchmark fairness, provenance, independent judging and B2 routing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from queue import SimpleQueue
from unittest.mock import MagicMock, patch

import pytest
from evals.baselines import run_baseline_b0, run_baseline_b1
from evals.judge import ensure_independent, run_judge
from evals.workers import STATE, resume_full_system, stage_worker
from langchain_core.messages import AIMessage

from research_agent.graph.checkpointing import create_checkpointer
from research_agent.graph.state import create_initial_state
from research_agent.graph.workflow import build_graph
from research_agent.schemas.evidence import AnalysisResult, Evidence
from research_agent.schemas.plan import ResearchPlan
from research_agent.tools.registry import ToolRegistry
from tests.fake_llm import FakeChatModel
from tests.workflow.test_workflow import (
    _make_analysis_json,
    _make_plan_json,
    _make_report_json,
    _make_supervisor_decision_json,
    _make_validation_result_json,
)


def prepared_state():
    state = create_initial_state("original", "original", "What is X?")
    state.update(
        research_plan=ResearchPlan.model_validate_json(_make_plan_json()),
        analyst_summary=AnalysisResult.model_validate_json(_make_analysis_json()),
        evidence=[
            Evidence(
                evidence_id="ev1",
                research_task_id="t1",
                tool_name="web_search",
                url="https://example.com/x",
                title="Source about X",
                content="X has been demonstrated",
                source_type="web",
            )
        ],
        completed_tasks=["t1"],
        research_iteration=1,
        limits={"max_research_iterations": 2, "max_validation_revisions": 1},
    )
    return state


def fake(value):
    return FakeChatModel(
        responses=[AIMessage(content=json.dumps(value) if isinstance(value, dict) else value)]
    )


def test_b0_single_call_and_rejects_invented_provenance():
    report = json.loads(_make_report_json())
    llm = fake(report)
    with pytest.raises(ValueError, match="invented"):
        run_baseline_b0("X", llm)
    assert len(llm.invocations) == 1
    report["references"] = []
    report["detailed_analysis"][0]["evidence_ids"] = []
    assert run_baseline_b0("X", fake(report)).references == []


def test_b0_execution_metadata_is_not_model_supplied():
    report = json.loads(_make_report_json())
    report.update(generated_at="2000-01-01T00:00:00Z", model_used="invented", version=9)
    report["references"] = []
    report["detailed_analysis"][0]["evidence_ids"] = []
    before = datetime.now(UTC)
    result = run_baseline_b0("Objective", fake(report))
    assert result.version == 1
    assert result.model_used == ""
    assert result.research_objective == "Objective"
    assert before <= result.generated_at <= datetime.now(UTC)


def test_b1_uses_production_generator_and_actual_reference():
    state = prepared_state()
    llm = fake(_make_report_json())
    result = run_baseline_b1(state, llm)
    assert result.references[0].url == state["evidence"][0].url
    assert result.references[0].accessed_at == state["evidence"][0].retrieved_at
    assert len(llm.invocations) == 1
    with patch("evals.baselines.run_report_generator", return_value=result) as generator:
        run_baseline_b1(state, llm)
        assert generator.call_args.kwargs["evidence"] is state["evidence"]
    with pytest.raises(ValueError, match="incomplete"):
        run_baseline_b1({}, llm)


def test_judge_is_separate_blinded_and_rejects_impossible_counts(sample_settings):
    ensure_independent(sample_settings)
    for field in ("report_generator_model", "validator_model"):
        with pytest.raises(ValueError, match="differ"):
            ensure_independent(
                sample_settings.model_copy(update={field: sample_settings.judge_model})
            )
    scores = dict(
        factual_grounding=8,
        comprehensiveness=6,
        structural_coherence=8,
        attribution_integrity=6,
        relevance=0.8,
        factual_claims=10,
        unsupported_claims=2,
        summary="Two claims lack evidence.",
    )
    llm = fake(scores)
    result = run_judge("query", "report", "evidence", llm)
    assert result.overall_score == 7
    assert result.unsupported_claim_rate == 0.2
    assert "reasoning" not in result.model_dump()
    assert "b2" not in llm.invocations[0][-1].content
    with pytest.raises(ValueError, match="exceeds"):
        run_judge("query", "report", "", fake({**scores, "unsupported_claims": 11}))


def test_b2_restores_shared_state_and_runs_real_validation(tmp_path: Path, sample_settings):
    state = prepared_state()
    empty = FakeChatModel(responses=[])
    supervisor = FakeChatModel(
        responses=[
            AIMessage(
                content=_make_supervisor_decision_json(
                    "sufficient", "report_generator", "evidence_review"
                )
            ),
            AIMessage(content=_make_supervisor_decision_json("accept", "finalize", "final_review")),
        ]
    )
    generator = fake(_make_report_json())
    with create_checkpointer(tmp_path / "b2.db") as saver:
        graph = build_graph(
            empty,
            empty,
            empty,
            generator,
            supervisor,
            fake(_make_validation_result_json()),
            ToolRegistry(),
            saver,
            sample_settings,
        )
        final = resume_full_system(
            graph,
            json.loads(STATE.dump_json(state)),
            {"configurable": {"thread_id": "new-thread"}},
            lambda _: None,
            30,
        )
    assert final["execution_status"] == "completed"
    assert len(final["evidence"]) == 1  # append reducers did not duplicate the seed
    assert final["evidence"][0].evidence_id == "ev1"
    assert len(generator.invocations) == 1
    assert final["deterministic_validation"].status == "pass"
    assert final["semantic_validation"].status == "pass"
    assert not empty.invocations  # shared preparation was not repeated or replaced
    assert state["thread_id"] == "original"


def test_worker_retrieval_stops_before_analysis_and_b2_branch_executes(sample_settings):
    graph = MagicMock()
    state = prepared_state()
    state["analyst_summary"] = None
    graph.invoke.return_value = state
    with patch("evals.workers.configured", return_value=graph):
        retrieved = stage_worker(SimpleQueue(), "prepare", sample_settings, "What is X?", None, 0)
    assert retrieved["status"] == "ok"
    assert retrieved["state"]["analyst_summary"] is None
    assert graph.invoke.call_args.kwargs["interrupt_before"] == ["analyst"]
    final = prepared_state()
    from research_agent.schemas.report import Report

    final["report"] = Report.model_validate_json(_make_report_json())
    final["execution_status"] = "completed"
    with (
        patch("evals.workers.configured", return_value=graph),
        patch("evals.workers.resume_full_system", return_value=final) as resumed,
    ):
        value = stage_worker(SimpleQueue(), "b2", sample_settings, "What is X?", retrieved, 0)
    assert resumed.call_count == 1
    assert value["report"]["title"] == "Report on X"


def test_failed_preparation_retains_evidence_for_diagnostics(sample_settings):
    state = prepared_state()
    state["failure_node"] = "researcher"
    graph = MagicMock()
    graph.invoke.return_value = state
    with patch("evals.workers.configured", return_value=graph):
        value = stage_worker(SimpleQueue(), "prepare", sample_settings, "What is X?", None, 0)
    assert value["status"] == "failed"
    assert value["state"]["evidence"][0]["evidence_id"] == "ev1"
    assert value["state"]["messages"] == []
