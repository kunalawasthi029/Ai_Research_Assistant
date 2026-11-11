"""Benchmark cache, aggregation, partial results and privacy regression tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from evals.protocol import fingerprint, load_dataset
from evals.runner import Benchmark, evaluate, main
from evals.statistics import paired_comparison
from evals.storage import StageCache, write_json

from research_agent.execution.isolation import WorkerError


def test_dataset_single_source_and_fingerprint_excludes_credentials(sample_settings):
    dataset = load_dataset()
    assert len(dataset["queries"]) >= 20
    first = fingerprint(sample_settings, dataset, 0)
    changed_key = sample_settings.model_copy(update={"groq_api_key": "another-test-key"})
    assert fingerprint(changed_key, dataset, 0) == first
    assert fingerprint(sample_settings.model_copy(update={"max_tokens": 1}), dataset, 0) != first


def test_dataset_rejects_duplicate_ids_and_missing_queries(tmp_path):
    value = load_dataset()
    value["queries"][1]["id"] = value["queries"][0]["id"]
    path = tmp_path / "queries.yaml"
    write_json(path, value)
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)
    value["queries"] = []
    write_json(path, value)
    with pytest.raises(ValueError, match="20"):
        load_dataset(path)


def test_cache_corruption_independent_repetitions_and_no_rebilling(tmp_path, sample_settings):
    cache = StageCache(tmp_path, "test")
    cache.path("bad").parent.mkdir(parents=True)
    cache.path("bad").write_text("broken")
    assert cache.get("bad") is None
    benchmark = Benchmark(sample_settings, cache, 0)
    with patch(
        "evals.runner.run_isolated",
        return_value={"status": "ok", "elapsed_seconds": 2, "calls": [{"total_tokens": 12}]},
    ) as worker:
        first = benchmark.stage(["q01", "b0", 1], "b0", "query")
        cached = benchmark.stage(["q01", "b0", 1], "b0", "query")
        second = benchmark.stage(["q01", "b0", 2], "b0", "query")
    assert worker.call_count == 2
    assert first["actual_tokens"] == second["actual_tokens"] == 12
    assert cached["actual_tokens"] == cached["actual_elapsed_seconds"] == 0
    assert cached["elapsed_seconds"] == 2


def test_failures_are_not_cached_or_leaked(tmp_path, sample_settings):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "safe"), 0)
    with patch("evals.runner.run_isolated", side_effect=WorkerError("private-provider-body")):
        value = benchmark.stage("fail", "b0", "query")
    assert value["status"] == "failed"
    assert "private-provider-body" not in json.dumps(value)
    assert benchmark.cache.get("fail") is None


def test_safe_worker_exception_class_is_retained_without_raw_provider_body(
    tmp_path, sample_settings
):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "class"), 0)
    with patch(
        "evals.runner.run_isolated", side_effect=WorkerError("Worker failed (AgentOutputError)")
    ):
        value = benchmark.stage("bad", "b1", "query")
    assert value["worker_error_type"] == "AgentOutputError"


def test_unattempted_urls_are_unassessed_not_failed(tmp_path, sample_settings):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "deadline"), 0)
    payload = {
        "data": {
            "verifications": [
                {"url": "https://example.com/a", "is_accessible": True, "error": None},
                {
                    "url": "https://example.com/b",
                    "is_accessible": False,
                    "error": "Timeout: batch deadline exceeded",
                },
            ]
        }
    }
    with patch("evals.runner.run_isolated", return_value=payload):
        result = benchmark.urls(["https://example.com/a", "https://example.com/b"])
    assert result["reachable_rate"] == 1.0
    assert result["unassessed"] == 1


def test_cache_invalidates_judge_when_report_or_evidence_changes(tmp_path, sample_settings):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "inputs"), 0)
    with patch(
        "evals.runner.run_isolated",
        return_value={"status": "ok", "elapsed_seconds": 1, "calls": []},
    ) as worker:
        for evidence, report in (("a", "first"), ("a", "first"), ("a", "second"), ("b", "second")):
            benchmark.stage(
                ["q01", "judge", 1],
                "judge",
                "query",
                {"state": {"evidence": [evidence]}},
                {"title": report},
            )
    assert worker.call_count == 3


def test_lower_unsupported_rate_counts_as_win():
    rows = [
        {
            "query_id": "q01",
            "arm": arm,
            "repetition": 1,
            "status": "ok",
            "metrics": {"unsupported_claim_rate": rate},
        }
        for arm, rate in (("b1", 0.5), ("b2", 0.1))
    ]
    result = paired_comparison(rows, "b1", "unsupported_claim_rate", 1)
    assert result["wins"] == 1
    assert not result["higher_is_better"]


def test_offline_summary_preserves_raw_observations_and_has_provenance(tmp_path):
    from evals.summarize import main as summarize_main

    source = tmp_path / "raw.json"
    write_json(
        source,
        {
            "protocol_version": 1,
            "experiment": "frozen",
            "limit": 1,
            "repetitions": 3,
            "records": [],
        },
    )
    original = source.read_bytes()
    assert summarize_main([str(source)]) == 0
    summary = json.loads(source.with_suffix(".summary.json").read_text())
    assert summary["source_sha256"]
    assert summary["summary"]["arms"]["b2"]["attempted"] == 0
    assert source.read_bytes() == original
    with pytest.raises(SystemExit):
        summarize_main([str(source), "--output", str(source)])


def test_explicit_retrieval_reuse_preserves_lineage_and_rejects_different_producer(
    tmp_path, sample_settings
):
    from evals.runner import load_retrievals
    from evals.workers import STATE

    from tests.evaluation.test_baselines import prepared_state

    dataset = load_dataset()
    dataset["queries"][0]["query"] = "What is X?"
    from evals.storage import digest

    source = {
        "dataset_sha256": digest(dataset),
        "settings": sample_settings.model_dump(mode="json"),
        "preparations": [
            {
                "query_id": "q01",
                "retrieval": {
                    "status": "ok",
                    "execution_status": "retrieved",
                    "elapsed_seconds": 10,
                    "state": json.loads(STATE.dump_json(prepared_state())),
                    "calls": [],
                },
            }
        ],
    }
    path = tmp_path / "prior.json"
    source["preparations"][0]["retrieval"]["raw_model_text"] = "private-scratch"
    source["preparations"][0]["retrieval"]["state"]["messages"] = [
        {"type": "ai", "content": "private-scratch"}
    ]
    write_json(path, source)
    reused = load_retrievals(path, sample_settings, dataset)["q01"]
    assert reused["actual_tokens"] == reused["actual_elapsed_seconds"] == 0
    assert reused["reuse_source_sha256"] == digest(source)
    assert reused["state"]["evidence"][0]["evidence_id"] == "ev1"
    assert reused["reuse_hash_format"] == "sorted-json-sha256"
    assert "private-scratch" not in json.dumps(reused)
    assert reused["state"]["messages"] == []
    with pytest.raises(ValueError, match="configuration"):
        load_retrievals(
            path, sample_settings.model_copy(update={"planner_model": "different"}), dataset
        )


def test_query_blocked_statistics_include_losses_and_require_complete_pairs():
    records = [
        dict(query_id=q, arm=a, repetition=r, status="ok", metrics={"score": score})
        for q, b2 in (("q1", 1), ("q2", 3))
        for r in range(1, 4)
        for a, score in (("b1", 2), ("b2", b2))
    ]
    comparison = paired_comparison(records, "b1", "score", 3)
    assert comparison["complete_paired_queries"] == 2
    assert comparison["wins"] == comparison["losses"] == 1
    assert comparison["ci95"] == [-1, 1]
    records[0]["status"] = "failed"
    comparison = paired_comparison(records, "b1", "score", 3)
    assert comparison["complete_paired_queries"] == 1
    assert comparison["ci95"] is None


def test_failed_preparation_still_runs_b0_and_saves_all_blocked_arms(tmp_path, sample_settings):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "failed"), 0)
    with patch.object(
        benchmark, "stage", return_value={"status": "failed", "error_type": "NetworkError"}
    ):
        output = tmp_path / "result.json"
        result = evaluate(load_dataset(), benchmark, 1, 3, output, "test")
    assert len(result["records"]) == 9
    assert sum(r["status"] == "blocked" for r in result["records"]) == 6
    assert result["summary"]["arms"]["b0"]["failed"] == 3
    assert json.loads(output.read_text())["finished_at"]
    assert "groq_api_key" not in output.read_text()


def test_url_failure_is_unassessed_not_zero_quality(tmp_path, sample_settings):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "urls"), 0)
    with patch("evals.runner.run_isolated", side_effect=WorkerError()):
        value = benchmark.urls(["https://example.com"])
    assert value["unassessed"] == 1
    assert value["reachable_rate"] is None


def test_cli_rejects_invalid_limits_and_judge_configuration():
    with pytest.raises(SystemExit):
        main(["--limit", "0"])


def test_single_arm_smoke_does_not_create_baseline_observations(tmp_path, sample_settings):
    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "single"), 0)
    with patch.object(
        benchmark, "stage", return_value={"status": "failed", "error_type": "NetworkError"}
    ):
        result = evaluate(load_dataset(), benchmark, 1, 1, tmp_path / "one.json", "smoke", ("b2",))
    assert result["arms"] == ["b2"]
    assert len(result["records"]) == 1
    assert result["summary"]["arms"]["b0"]["attempted"] == 0
    assert result["summary"]["arms"]["b2"]["attempted"] == 1


def test_failed_system_keeps_extra_evidence_and_observed_resource_costs(tmp_path, sample_settings):
    from evals.workers import STATE

    from tests.evaluation.test_baselines import prepared_state

    state = json.loads(STATE.dump_json(prepared_state()))
    extra = {**state["evidence"][0], "evidence_id": "additional"}

    def stage(key, kind, query, prepared=None, report=None):
        if kind in {"prepare", "analyze"}:
            return {"status": "ok", "state": state, "elapsed_seconds": 1, "calls": []}
        return {
            "status": "failed",
            "evidence": [*state["evidence"], extra],
            "elapsed_seconds": 2,
            "calls": [{"total_tokens": 12}],
        }

    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "partial"), 0)
    with patch.object(benchmark, "stage", side_effect=stage):
        result = evaluate(
            load_dataset(), benchmark, 1, 1, tmp_path / "partial.json", "test", ("b2",)
        )
    assert result["records"][0]["extra_evidence_ids"] == ["additional"]
    assert result["summary"]["arms"]["b2"]["failed"] == 1
    assert (
        result["summary"]["arms"]["b2"]["downstream_reported_workload_tokens_all_attempts"]["mean"]
        == 12
    )


def test_summary_counts_new_reported_tokens_from_failed_work_without_double_charging_prefix(
    tmp_path,
):
    from evals.summarize import summarize_file

    source = tmp_path / "costs.json"
    write_json(
        source,
        {
            "protocol_version": 2,
            "experiment": "costs",
            "limit": 1,
            "repetitions": 1,
            "preparations": [{"actual_tokens": 5, "retrieval": {"actual_tokens": 10}}],
            "records": [
                {
                    "query_id": "q01",
                    "arm": "b2",
                    "repetition": 1,
                    "status": "failed",
                    "actual_tokens": 7,
                    "calls": [{"total_tokens": 7}],
                    "judge": {"status": "failed", "actual_tokens": 3},
                }
            ],
        },
    )
    result = summarize_file(source, tmp_path / "summary.json")
    assert result["actual_reported_tokens_total"] == 25
    assert result["raw_protocol_version"] == 2


def test_successful_runner_costs_common_evidence_and_missing_judges_are_explicit(
    tmp_path, sample_settings
):
    from evals.workers import STATE

    from tests.evaluation.test_baselines import prepared_state

    state = json.loads(STATE.dump_json(prepared_state()))
    preparation = {
        "status": "ok",
        "state": state,
        "elapsed_seconds": 10,
        "calls": [{"total_tokens": 100}],
    }
    keys = []

    def stage(key, kind, query, prepared=None, report=None):
        keys.append(key)
        if kind == "prepare":
            return preparation
        if kind == "analyze":
            return {**preparation, "elapsed_seconds": 2, "calls": [{"total_tokens": 5}]}
        if kind == "judge":
            return {"status": "failed", "error_type": "JudgeQuotaError"}
        from tests.workflow.test_workflow import _make_report_json

        arm_report = json.loads(_make_report_json())
        if kind == "b0":
            arm_report["references"] = []
            arm_report["detailed_analysis"][0]["evidence_ids"] = []
        return {
            "status": "ok",
            "report": arm_report,
            "elapsed_seconds": 2,
            "calls": [{"total_tokens": 20}],
            "evidence": state["evidence"] if kind != "b0" else [],
        }

    benchmark = Benchmark(sample_settings, StageCache(tmp_path, "success"), 0)
    with (
        patch.object(benchmark, "stage", side_effect=stage),
        patch.object(benchmark, "urls", return_value={"reachable_rate": 1}),
    ):
        result = evaluate(load_dataset(), benchmark, 1, 3, tmp_path / "results.json", "test")
    assert keys.count(["q01", "prepare"]) == 1
    for row in result["records"]:
        expected = 2 if row["arm"] == "b0" else 14
        assert row["metrics"]["elapsed_seconds"] == expected
        assert row["metrics"]["total_tokens"] == (20 if row["arm"] == "b0" else 125)
        assert "judge_overall" not in row["metrics"]
        if row["arm"] != "b0":
            assert row["initial_evidence_ids"] == ["ev1"]
            assert row["extra_evidence_ids"] == []
    assert all(
        c["complete_paired_queries"] == 0
        for c in result["summary"]["comparisons"]
        if c["metric"] == "judge_overall"
    )
