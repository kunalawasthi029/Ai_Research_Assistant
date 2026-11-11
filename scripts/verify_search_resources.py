"""Explicit live Tavily + persistent cache probe; stores metadata, not source text or secrets."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from research_agent.config.settings import Settings
from research_agent.tools.results import ToolOutput
from research_agent.tools.web_search import web_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings()
    if not settings.tavily_api_key:
        parser.error("Configure TAVILY_API_KEY for the live probe")
    cache_path = Path(".release/section19") / f"search-{uuid4().hex}.db"
    options = {
        "api_key": settings.tavily_api_key,
        "max_results": 1,
        "result_limit": 1,
        "search_depth": "basic",
        "timeout_seconds": 15,
        "fallback_enabled": False,
        "cache_path": str(cache_path),
    }
    query = "Streamlit Community Cloud official documentation"
    first = ToolOutput.model_validate_json(web_search(query, **options))
    # Never send a second potentially billable call after a failed/empty first result.
    second = (
        ToolOutput.model_validate_json(web_search(query, **options))
        if first.status == "success" and cache_path.is_file()
        else None
    )
    passed = bool(
        second
        and second.data.get("cache_hit")
        and first.items
        and second.items
        and first.items[0].retrieved_at == second.items[0].retrieved_at
    )
    artifact = {
        "observed_at": datetime.now(UTC).isoformat(),
        "verification_level": "3 - live provider, local application",
        "provider": "Tavily",
        "search_depth": "basic",
        "max_results": 1,
        "first_status": first.status,
        "first_count": len(first.items),
        "first_cache_hit": first.data.get("cache_hit"),
        "first_credits_upper_bound": first.data.get("tavily_credits_upper_bound"),
        "error": first.error,
        "second_cache_hit": second.data.get("cache_hit") if second else None,
        "second_credits_upper_bound": second.data.get("tavily_credits_upper_bound")
        if second
        else None,
        "original_retrieval_time_preserved": passed,
        "passed": passed,
        "limits": "Not a report quality, quota-exhaustion, account balance or deployment test.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
