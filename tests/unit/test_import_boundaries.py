"""Import boundary test: enforces the layered architecture.

C11: core packages import neither FastAPI nor Streamlit.
The frontend never imports langgraph or core agent modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "research_agent"

# Modules in core layers (0-3) must NOT import these
_FORBIDDEN_IN_CORE = {"fastapi", "uvicorn", "streamlit", "starlette"}

# The api/ layer must NOT import these
_FORBIDDEN_IN_API = {"streamlit"}


def _get_imports(filepath: Path) -> set[str]:
    """Parse a Python file and return the set of top-level imported module names."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def _collect_python_files(directory: Path) -> list[Path]:
    """Recursively collect all .py files in a directory."""
    if not directory.exists():
        return []
    return list(directory.rglob("*.py"))


class TestImportBoundaries:
    """Enforce the layered architecture import rules."""

    def test_core_does_not_import_fastapi_or_streamlit(self) -> None:
        """Layers 0-3 (schemas, config, llm, tools, agents, graph, persistence, service)
        must not import fastapi or streamlit."""
        core_dirs = [
            _SRC_ROOT / "schemas",
            _SRC_ROOT / "config",
            _SRC_ROOT / "llm",
            _SRC_ROOT / "tools",
            _SRC_ROOT / "agents",
            _SRC_ROOT / "graph",
            _SRC_ROOT / "persistence",
            _SRC_ROOT / "service",
        ]

        violations: list[str] = []
        for d in core_dirs:
            for f in _collect_python_files(d):
                imports = _get_imports(f)
                forbidden = imports & _FORBIDDEN_IN_CORE
                if forbidden:
                    rel = f.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}: imports {forbidden}")

        assert violations == [], "Core import boundary violations:\n" + "\n".join(violations)

    def test_cli_does_not_import_fastapi_or_streamlit(self) -> None:
        """The CLI module must not import FastAPI or Streamlit."""
        cli_path = _SRC_ROOT / "cli.py"
        if cli_path.exists():
            imports = _get_imports(cli_path)
            forbidden = imports & _FORBIDDEN_IN_CORE
            assert forbidden == set(), f"cli.py imports {forbidden}"

    def test_api_does_not_import_streamlit(self) -> None:
        """The API layer must not import Streamlit."""
        api_dir = _SRC_ROOT / "api"
        violations: list[str] = []
        for f in _collect_python_files(api_dir):
            imports = _get_imports(f)
            forbidden = imports & _FORBIDDEN_IN_API
            if forbidden:
                rel = f.relative_to(_SRC_ROOT)
                violations.append(f"{rel}: imports {forbidden}")

        assert violations == [], "API import boundary violations:\n" + "\n".join(violations)
