"""Export compiled graph structure from the checkout; no inference or tool calls."""

from pathlib import Path

from tests.fake_llm import FakeChatModel

from research_agent.graph.researcher_loop import build_researcher_subgraph
from research_agent.graph.workflow import build_graph
from research_agent.tools.registry import ToolRegistry
from research_agent.tools.results import evidence_from_output


def main() -> None:
    """Use the test adapter only to bind schemas; never invoke its empty response list."""
    model = FakeChatModel(responses=[])
    registry = ToolRegistry()
    coordinator = build_graph(model, model, model, model, model, model, registry)
    researcher = build_researcher_subgraph(model, registry, evidence_from_output)
    parts = [
        "# Compiled workflow diagrams\n",
        "Generated with `python -m scripts.export_graph`; both diagrams come from "
        "`get_graph().draw_mermaid()`. The model adapter is used only to compile structure. "
        "This export performs no inference and is not execution evidence.\n",
    ]
    for title, graph in (("Coordinator", coordinator), ("Researcher tool loop", researcher)):
        parts.append(f"## {title}\n\n```mermaid\n{graph.get_graph().draw_mermaid()}\n```\n")
    target = Path(__file__).resolve().parents[1] / "docs" / "graph.md"
    target.write_text("\n".join(parts), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
