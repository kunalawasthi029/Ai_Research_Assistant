"""CLI entry point for running research from the command line."""

from __future__ import annotations

import argparse
import logging
import sys

from research_agent.config.logging import configure_logging
from research_agent.config.settings import get_settings
from research_agent.service.research_service import ResearchService


def main(argv: list[str] | None = None) -> int:
    """Run a research query from the command line.

    Usage:
        python -m research_agent.cli "What is quantum computing?"
        python -m research_agent.cli --query "AI in healthcare" --objective "Survey recent advances"
    """
    parser = argparse.ArgumentParser(
        description="AI Research Agent — run a research query",
        prog="research-agent",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query (positional or via --query)",
    )
    parser.add_argument(
        "--query",
        "-q",
        dest="query_flag",
        help="Research query (alternative to positional)",
    )
    parser.add_argument(
        "--objective",
        "-o",
        default="",
        help="Refined research objective (defaults to query)",
    )
    parser.add_argument(
        "--output",
        "-O",
        choices=["json", "markdown", "both"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--no-checkpointer",
        action="store_true",
        help="Disable SQLite checkpointing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--resume", metavar="RUN_ID", help="Resume an existing interrupted/approval run"
    )
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument(
        "--approve", action="store_true", help="Approve the saved plan when resuming"
    )
    approval.add_argument(
        "--reject", action="store_true", help="Reject the saved plan when resuming"
    )
    args = parser.parse_args(argv)

    query = args.query or args.query_flag
    if args.resume and query:
        parser.error("Use either a query or --resume")
    if not args.resume and (not query or not 3 <= len(query.strip()) <= 4000):
        parser.error("A research query of 3 to 4000 characters is required")
    if (args.approve or args.reject) and not args.resume:
        parser.error("Approval flags require --resume")
    if len(args.objective) > 4000:
        parser.error("The objective must not exceed 4000 characters")
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        with ResearchService(settings=get_settings()) as service:
            if args.resume:
                approved = True if args.approve else False if args.reject else None
                result = service.resume_research(args.resume, approved=approved)
            else:
                result = service.run_research(
                    query=query, objective=args.objective, use_checkpointer=not args.no_checkpointer
                )
            status = result["status"]
            print(f"Research status: {status} (run: {result['run_id']})", file=sys.stderr)
            print(
                f"Elapsed: {result.get('elapsed_seconds', 0):.1f}s; "
                f"evidence: {result.get('evidence_count', 0)}; "
                f"tool calls: {result.get('tool_call_count', 0)}",
                file=sys.stderr,
            )
            report = result.get("report")
            if report:
                if args.output in ("json", "both"):
                    print(report.model_dump_json(indent=2))
                if args.output in ("markdown", "both"):
                    row = service.get_report(result["run_id"])
                    print(row.report_markdown if row else service._render_markdown(report))
            elif status == "awaiting_approval":
                print(
                    "Review the saved plan with ResearchService.inspect_checkpoint before "
                    "using --resume RUN_ID --approve or --reject.",
                    file=sys.stderr,
                )
            else:
                print("No report generated.", file=sys.stderr)
            return 0 if status == "completed" else 1 if status == "failed" else 2
    except Exception as error:
        print(f"Research could not execute ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
