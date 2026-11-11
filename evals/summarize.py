"""Recompute descriptive statistics without a provider call or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.statistics import summarize
from evals.storage import write_json


def summarize_file(path: Path, output: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("protocol_version") not in {1, 2} or not 1 <= data.get("repetitions", 0) <= 10:
        raise ValueError("Unsupported result protocol")
    result = {
        "analysis_version": 3,
        "raw_protocol_version": data["protocol_version"],
        "source": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "experiment": data["experiment"],
        "limit": data["limit"],
        "repetitions": data["repetitions"],
        "actual_reported_tokens_total": sum(
            p.get("actual_tokens", 0) + p.get("retrieval", {}).get("actual_tokens", 0)
            for p in data.get("preparations", [])
        )
        + sum(
            r.get("actual_tokens", 0)
            + r.get("judge", {}).get("actual_tokens", 0)
            + r.get("judge_full_corpus", {}).get("actual_tokens", 0)
            for r in data["records"]
        ),
        "summary": summarize(data["records"], data["repetitions"]),
    }
    write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.results.with_suffix(".summary.json")
    if output.resolve() == args.results.resolve():
        parser.error("Summary output must preserve the raw observations")
    summarize_file(args.results, output)
    print(f"Summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
