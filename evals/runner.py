"""Run `python -m evals.runner --limit 1` for a bounded live B0/B1/B2 benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from evals.judge import ensure_independent
from evals.metrics import compute_all_metrics
from evals.protocol import ROOT, fingerprint, load_dataset
from evals.statistics import summarize
from evals.storage import StageCache, digest, write_json
from evals.workers import STATE, stage_worker, url_worker
from research_agent.config.settings import Settings
from research_agent.execution.isolation import run_isolated


class Benchmark:
    """Cache complete parsed stages, with different cache keys for independent draws."""

    def __init__(
        self,
        settings: Settings,
        cache: StageCache,
        throttle: float,
        reused: dict[str, Any] | None = None,
    ):
        self.settings, self.cache, self.throttle = settings, cache, throttle
        self.reused = reused or {}

    def stage(
        self, key: Any, kind: str, query: str, prepared: Any = None, report: Any = None
    ) -> dict[str, Any]:
        key = [
            key,
            digest(
                {
                    "query": query,
                    "state": prepared.get("state") if prepared else None,
                    "report": report,
                }
            ),
        ]
        cached = self.cache.get(key)
        if cached is not None and cached.get("status") == "ok":
            return {**cached, "cache_hit": True, "actual_elapsed_seconds": 0.0, "actual_tokens": 0}
        print(json.dumps({"stage": kind, "key": key, "cache_hit": False}), flush=True)
        # Independent workers also leave a minimum inter-stage gap.
        if self.throttle:
            time.sleep(self.throttle)
        started = time.monotonic()
        observed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        def progress(event: dict[str, Any]) -> None:
            if event.get("event_type") == "llm_call":
                observed.append(event)
            if event.get("event_type") in {"llm_error", "agent_output_invalid"}:
                errors.append(event)
            if event.get("event_type") in {
                "node_finished",
                "node_error",
                "llm_error",
                "agent_output_invalid",
            }:
                print(json.dumps(event), flush=True)

        budget = self.settings.overall_run_timeout_seconds
        if kind in {"b1", "b2", "analyze"} and prepared:
            budget -= prepared["elapsed_seconds"]
        if kind == "judge":
            budget = min(120, self.settings.overall_run_timeout_seconds)
        value: dict[str, Any]
        try:
            value = run_isolated(
                stage_worker,
                (kind, self.settings, query, prepared, self.throttle, report),
                budget,
                on_event=progress,
            )
        except Exception as error:
            # No provider bodies or raw exception messages in artifacts.
            value = {
                "status": "failed",
                "error_type": type(error).__name__,
                "calls": observed,
                "elapsed_seconds": time.monotonic() - started,
                "errors": errors,
            }
            match = re.fullmatch(r"Worker failed \(([A-Za-z][A-Za-z0-9_]{0,80})\)", str(error))
            if match:
                value["worker_error_type"] = match.group(1)
        value["cache_hit"] = False
        value["actual_elapsed_seconds"] = time.monotonic() - started
        value["actual_tokens"] = sum(c["total_tokens"] for c in value.get("calls", []))
        if value["status"] == "ok":
            self.cache.put(key, value)
        return cast(dict[str, Any], value)

    def urls(self, urls: list[str]) -> dict[str, Any]:
        if not urls:
            return {"verifications": [], "reachable_rate": None, "unassessed": 0}
        key = ["url-reachability", sorted(set(urls))]
        cached = self.cache.get(key)
        if cached is not None:
            return {**cached, "cache_hit": True}
        observations = []
        unique = sorted(set(urls))
        unassessed = 0
        for start in range(0, len(unique), 20):
            batch = unique[start : start + 20]
            try:
                value = run_isolated(url_worker, (batch,), 20)
                observations.extend(value["data"]["verifications"])
            except Exception:
                unassessed += len(batch)
        unattempted = [
            v for v in observations if v.get("error") == "Timeout: batch deadline exceeded"
        ]
        assessed = [v for v in observations if v.get("error") != "Timeout: batch deadline exceeded"]
        unassessed += len(unattempted)
        result = {
            "verifications": observations,
            "unassessed": unassessed,
            "reachable_rate": sum(v["is_accessible"] for v in assessed) / len(assessed)
            if assessed
            else None,
            "observed_at": datetime.now(UTC).isoformat(),
            "cache_hit": False,
        }
        self.cache.put(key, result)
        return result


def report_text(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            report["executive_summary"],
            *report["key_findings"],
            *(s["content"] for s in report["detailed_analysis"]),
            report["conclusion"],
        ]
    )


def evaluate(
    dataset: dict[str, Any],
    benchmark: Benchmark,
    limit: int,
    repetitions: int,
    output: Path,
    experiment: str,
    arms: tuple[str, ...] = ("b0", "b1", "b2"),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol_version": 2,
        "citation_validity_semantics": "Observed public-only HEAD accessibility; null until assessed",
        "experiment": experiment,
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset["version"],
        "dataset_sha256": digest(dataset),
        "dataset_size": len(dataset["queries"]),
        "limit": limit,
        "repetitions": repetitions,
        "arms": list(arms),
        "settings": benchmark.settings.model_dump(mode="json"),
        "throttle_seconds": benchmark.throttle,
        "judge_reference_policy": "Same initial corpus for all arms; B2 full corpus judged separately",
        "cache_policy": "Completed parsed stages; distinct query/arm/repetition keys; no raw messages",
        "metric_notes": "Schema, ID resolution and section completeness are construction metrics. "
        "Topic coverage is keyword mentions. Reachability is HEAD accessibility, not truth. "
        "Unsupported means absent from the supplied corpus, not necessarily false. "
        "Means exclude missing metrics; failures and denominators are explicit. "
        "Costs charge shared preparation to B1 and B2, not judge/URL overhead.",
        "preparations": [],
        "records": [],
    }

    def save() -> None:
        result["summary"] = summarize(result["records"], repetitions)
        result["updated_at"] = datetime.now(UTC).isoformat()
        write_json(output, result)

    save()
    for item in dataset["queries"][:limit]:
        qid, query = item["id"], item["query"]
        retrieved = benchmark.reused.get(qid) or benchmark.stage([qid, "prepare"], "prepare", query)
        prepared = retrieved
        if retrieved["status"] == "ok":
            analyzed = benchmark.stage([qid, "analyze"], "analyze", query, retrieved)
            prepared = {
                **analyzed,
                "retrieval": retrieved,
                "elapsed_seconds": retrieved["elapsed_seconds"] + analyzed["elapsed_seconds"],
                "calls": retrieved.get("calls", []) + analyzed.get("calls", []),
            }
        result["preparations"].append({"query_id": qid, **prepared})
        save()
        for repetition in range(1, repetitions + 1):
            order = list(arms)
            random.Random(f"{qid}:{repetition}:42").shuffle(order)
            for arm in order:
                key = [qid, arm, repetition]
                value: dict[str, Any] = (
                    {"status": "blocked", "error_type": "SharedPreparationFailed", "calls": []}
                    if arm != "b0" and prepared["status"] != "ok"
                    else benchmark.stage(key, arm, query, prepared if arm != "b0" else None)
                )
                row = {"query_id": qid, "repetition": repetition, "arm": arm, **value}
                shared = prepared.get("state", {}).get("evidence", [])
                shared_ids = {e["evidence_id"] for e in shared}
                row["initial_evidence_ids"] = sorted(shared_ids) if arm != "b0" else []
                row["extra_evidence_ids"] = (
                    sorted(
                        e["evidence_id"]
                        for e in value["evidence"]
                        if e["evidence_id"] not in shared_ids
                    )
                    if "evidence" in value
                    else None
                )
                report = value.get("report")
                if value["status"] == "ok" and report:
                    evidence = value.get("evidence", [])
                    metrics = compute_all_metrics(
                        report,
                        report_text(report),
                        item["expected_topics"],
                        evidence,
                        {e["evidence_id"] for e in evidence},
                    )
                    prefix = prepared if arm != "b0" else {}
                    metrics.update(
                        elapsed_seconds=value["elapsed_seconds"] + prefix.get("elapsed_seconds", 0),
                        total_tokens=sum(
                            c["total_tokens"]
                            for c in value.get("calls", []) + prefix.get("calls", [])
                        ),
                        extra_evidence_count=len(row["extra_evidence_ids"]),
                    )
                    row["url_observations"] = benchmark.urls(
                        [ref["url"] for ref in report["references"]]
                    )
                    metrics["url_reachable_rate"] = row["url_observations"]["reachable_rate"]
                    metrics["citation_validity"] = metrics["url_reachable_rate"]
                    if prepared["status"] == "ok":
                        judged = benchmark.stage(
                            [*key, "judge-common"], "judge", query, prepared, report
                        )
                        row["judge"] = judged
                        if judged["status"] == "ok":
                            scores = judged["judge"]
                            metrics.update(
                                judge_overall=scores["overall_score"],
                                relevance=scores["relevance"],
                                unsupported_claim_rate=scores["unsupported_claim_rate"],
                            )
                        if row["extra_evidence_ids"]:
                            full = {
                                **prepared,
                                "state": {**prepared["state"], "evidence": evidence},
                            }
                            row["judge_full_corpus"] = benchmark.stage(
                                [*key, "judge-full"], "judge", query, full, report
                            )
                    row["metrics"] = metrics
                result["records"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            "query_id": qid,
                            "arm": arm,
                            "repetition": repetition,
                            "status": row["status"],
                            "judge_status": row.get("judge", {}).get("status"),
                        }
                    ),
                    flush=True,
                )
    result["finished_at"] = datetime.now(UTC).isoformat()
    save()
    return result


def load_retrievals(path: Path, settings: Settings, dataset: dict[str, Any]) -> dict[str, Any]:
    """Explicitly reuse actual source observations across analysis-only changes."""
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("dataset_sha256") != digest(dataset):
        raise ValueError("Retrieval dataset differs")
    fields = (
        "planner_model",
        "researcher_model",
        "supervisor_model",
        "research_depth",
        "research_scope",
        "source_preferences",
        "max_tool_calls_per_task",
        "max_tool_calls_per_run",
        "per_tool_timeout_seconds",
        "max_plan_revisions",
    )
    for name in fields:
        if source["settings"].get(name) != getattr(settings, name):
            raise ValueError("Retrieval-producing configuration differs")
    queries = {q["id"]: q["query"] for q in dataset["queries"]}
    reused = {}
    for entry in source["preparations"]:
        candidate = entry.get("retrieval", entry)
        if candidate.get("status") != "ok" or candidate.get("execution_status") != "retrieved":
            continue
        state = STATE.validate_python(candidate["state"])
        if state["research_query"] != queries.get(entry["query_id"]):
            raise ValueError("Retrieval query differs")
        state["messages"] = []
        reused[entry["query_id"]] = {
            **{
                key: candidate[key]
                for key in ("status", "execution_status", "elapsed_seconds", "failure_node")
                if key in candidate
            },
            "calls": [
                {
                    k: c[k]
                    for k in (
                        "agent",
                        "model",
                        "duration_seconds",
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                    if k in c
                }
                for c in candidate.get("calls", [])
            ],
            "errors": [
                {k: e[k] for k in ("node", "error_type", "http_status", "agent") if k in e}
                for e in candidate.get("errors", [])
            ],
            "state": json.loads(STATE.dump_json(state)),
            "cache_hit": True,
            "actual_elapsed_seconds": 0.0,
            "actual_tokens": 0,
            "reuse_source": path.name,
            "reuse_source_sha256": digest(source),
            "reuse_hash_format": "sorted-json-sha256",
        }
    if not reused:
        raise ValueError("No compatible completed retrieval was found")
    return reused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--throttle", type=float, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "evals/.cache")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment", default="phase-8-v1")
    parser.add_argument("--arms", nargs="+", choices=["b0", "b1", "b2"], default=["b0", "b1", "b2"])
    parser.add_argument(
        "--reuse-retrieval",
        type=Path,
        help="Reuse validated real retrieval from a prior result, retaining its provenance",
    )
    args = parser.parse_args(argv)
    if (
        not 1 <= args.limit <= 20
        or not 1 <= args.repetitions <= 10
        or not 0 <= args.throttle <= 60
        or len(set(args.arms)) != len(args.arms)
    ):
        parser.error("limit must be 1..20, repetitions 1..10, throttle 0..60, arms unique")
    logging.basicConfig(level=logging.ERROR)
    # Avoid HTTP/SDK logging raw provider errors; run_isolated returns type names only.
    try:
        settings = Settings(  # type: ignore[call-arg]  # Required key is loaded from env/dotenv.
            max_research_iterations=2,
            max_tool_calls_per_task=3,
            max_tool_calls_per_run=15,
            max_validation_revisions=1,
            max_plan_revisions=1,
            overall_run_timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
            human_approval_enabled=False,
            research_depth="brief",
            research_scope="narrow",
        )
        ensure_independent(settings)
        dataset = load_dataset()
        namespace = digest([args.experiment, fingerprint(settings, dataset, args.throttle)])
        cache = StageCache(args.cache_dir, namespace)
        reused = (
            load_retrievals(args.reuse_retrieval, settings, dataset) if args.reuse_retrieval else {}
        )
        benchmark = Benchmark(settings, cache, args.throttle, reused)
        output = args.output or ROOT / "evals/results" / f"{namespace[:16]}.json"
        result = evaluate(
            dataset, benchmark, args.limit, args.repetitions, output, namespace, tuple(args.arms)
        )
    except (ValidationError, ValueError, OSError) as error:
        print(f"Evaluation configuration failed ({type(error).__name__})")
        return 2
    print(f"Results: {output}")
    return (
        0
        if all(
            r["status"] == "ok" and r.get("judge", {}).get("status") == "ok"
            for r in result["records"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
