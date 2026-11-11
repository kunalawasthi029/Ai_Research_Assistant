"""Query-blocked paired bootstrap; repetitions are not independent queries."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Any


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "std": stdev(values) if len(values) > 1 else None,
    }


def paired_comparison(
    records: list[dict[str, Any]],
    comparator: str,
    metric: str,
    repetitions: int,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        value = record.get("metrics", {}).get(metric)
        if record["status"] == "ok" and isinstance(value, (float, int)):
            grouped[record["query_id"]][record["arm"]][record["repetition"]] = float(value)
    expected = set(range(1, repetitions + 1))
    deltas = []
    for arms in grouped.values():
        if set(arms["b2"]) == expected and set(arms[comparator]) == expected:
            deltas.append(mean(arms["b2"][r] - arms[comparator][r] for r in expected))
    direction = (
        -1
        if metric
        in {"unsupported_claim_rate", "elapsed_seconds", "total_tokens", "duplicate_section_rate"}
        else 1
    )
    result = {
        "metric": metric,
        "comparison": f"b2-{comparator}",
        "complete_paired_queries": len(deltas),
        "query_delta": describe(deltas),
        "wins": sum(d * direction > 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas),
        "losses": sum(d * direction < 0 for d in deltas),
        "higher_is_better": direction == 1,
        "ci95": None,
        "ci_method": "query-blocked percentile bootstrap; positive is higher, not always better",
    }
    if len(deltas) >= 2:
        rng = random.Random(seed)
        draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
        result["ci95"] = [
            draws[int(samples * 0.025)],
            draws[min(samples - 1, int(samples * 0.975))],
        ]
    return result


def summarize(records: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    arms = {}
    for arm in ("b0", "b1", "b2"):
        rows = [r for r in records if r["arm"] == arm]
        names = {k for r in rows for k in r.get("metrics", {})}
        arms[arm] = {
            "attempted": len(rows),
            "failed": sum(r["status"] != "ok" for r in rows),
            "downstream_reported_workload_tokens_all_attempts": describe(
                [
                    sum(call["total_tokens"] for call in r.get("calls", []))
                    for r in rows
                    if "elapsed_seconds" in r
                ]
            ),
            "downstream_stage_seconds_all_attempts": describe(
                [
                    r["elapsed_seconds"]
                    for r in rows
                    if isinstance(r.get("elapsed_seconds"), (int, float))
                ]
            ),
            "best_effort": sum(r.get("execution_status") == "best_effort" for r in rows),
            "judge_ok": sum(r.get("judge", {}).get("status") == "ok" for r in rows),
            "judge_failed": sum("judge" in r and r["judge"].get("status") != "ok" for r in rows),
            "judge_unassessed": sum("judge" not in r for r in rows),
            "metrics": {
                name: describe(
                    [
                        r["metrics"][name]
                        for r in rows
                        if r["status"] == "ok"
                        and isinstance(r.get("metrics", {}).get(name), (int, float))
                    ]
                )
                for name in sorted(names)
            },
        }
    metrics = (
        "judge_overall",
        "relevance",
        "unsupported_claim_rate",
        "elapsed_seconds",
        "total_tokens",
        "url_reachable_rate",
        "citation_id_resolution",
        "evidence_coverage",
        "schema_valid",
        "duplicate_section_rate",
        "report_completeness",
        "topic_coverage",
    )
    return {
        "arms": arms,
        "comparisons": [
            paired_comparison(records, arm, metric, repetitions)
            for arm in ("b1", "b0")
            for metric in metrics
        ],
    }
