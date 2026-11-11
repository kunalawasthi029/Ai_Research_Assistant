# AI Research Assistant

A Python research application with six agent roles, six tools, a bounded LangGraph
workflow, a FastAPI backend, and a Streamlit frontend. Groq supplies model inference;
Tavily is the primary search provider, with an explicit optional fallback.

SQLite stores runs, source excerpts, report versions, validation results, events,
and graph checkpoints. Citations link to retrieved evidence. Provider failures and
exhausted budgets produce labelled partial results rather than invented success.

## Run locally

Use Python 3.11 or 3.12. From the project root:

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install setuptools==84.2.1 wheel==0.48.0
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

Use **one root `.env`** with your credentials and backend address:

```ini
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
RESEARCH_API_URL=http://127.0.0.1:8000/api/v1
```

The existing local `.env` is private and excluded from Git. See
[configuration](docs/ENVIRONMENT.md) for model selection, budgets, and optional
API authentication. Do not commit credentials.

Start the backend:

```powershell
python -m uvicorn research_agent.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second activated terminal:

```powershell
python -m streamlit run frontend/app.py --server.port 8501
```

Open [the workspace](http://localhost:8501).
The [backend health endpoint](http://127.0.0.1:8000/api/v1/health) remains public.
The CLI uses the same research service:

```powershell
python -m research_agent.cli "Compare TCP and UDP" --output both
```

## Verify

```powershell
python -m ruff check src frontend tests evals scripts
python -m ruff format --check src frontend tests evals scripts
python -m mypy src/research_agent frontend evals
python -m pytest -m "not live" --cov=research_agent --cov=frontend --cov=evals
```

GitHub Actions runs offline checks on Windows/Linux and Python 3.11/3.12.
Live evaluation is a separate manual workflow requiring provider credentials.
See [verification status](docs/VERIFICATION.md) and [evaluation status](docs/EVAL_RESULTS.md).

## Layout

| Directory | Purpose |
|---|---|
| `src/research_agent/` | Agents, graph, tools, API, persistence, and CLI |
| `frontend/` | Streamlit pages and HTTP client |
| `tests/` | Unit, mocked integration, and explicitly marked live checks |
| `evals/` | Frozen queries, baselines, judging, and statistics |
| `scripts/` | Graph export and search-resource verification |
| `docs/` | Architecture, configuration, verification, and limitations |

Read the [architecture](docs/ARCHITECTURE.md), [workflow diagrams](docs/graph.md),
and [limitations](docs/LIMITATIONS.md). `uv.lock` is the dependency resolution source;
`requirements.lock` is its hashed pip export used by installation and CI. The
frontend requirements declare its smaller HTTP-only dependency set.

Deployment is cancelled. No cloud resources are created by this repository.
