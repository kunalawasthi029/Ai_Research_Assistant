# Configuration

Use one ignored root `.env`. The backend resolves it from the source checkout;
an installed package resolves it from the launch directory. Restart both services
after configuration changes.

| Setting | Purpose |
|---|---|
| `GROQ_API_KEY` | Required model credential |
| `TAVILY_API_KEY` | Optional primary search credential; fallback is explicit |
| `RESEARCH_API_URL` | Frontend backend address ending in `/api/v1` |
| `APP_API_KEY` | Optional shared backend key, sent in `X-API-Key` |
| `REQUIRE_API_AUTH` | Reject startup without an application key when true |
| `DATABASE_PATH` / `CHECKPOINT_DB_PATH` | Local SQLite files |
| `API_MAX_WORKERS` | Concurrent research worker cap |
| `LLM_MIN_INTERVAL_SECONDS` | Per-worker spacing for calls sharing a model |
| `MAX_TOKENS` / `EVIDENCE_CONTEXT_MAX_CHARS` | Output/context budgets |
| `OVERALL_RUN_TIMEOUT_SECONDS` | Total execution deadline |
| `SEARCH_DEPTH` / `SEARCH_MAX_RESULTS` | Search cost controls |
| `SEARCH_CACHE_PATH` | Opt-in persistent search cache |

Model overrides: `SUPERVISOR_MODEL`, `PLANNER_MODEL`, `RESEARCHER_MODEL`,
`ANALYST_MODEL`, `REPORT_GENERATOR_MODEL`, `VALIDATOR_MODEL`, and `JUDGE_MODEL`.
Authoritative defaults/ranges are in `src/research_agent/config/settings.py`.
Local values override defaults. Model availability, capability, and quota are
separate checks; small probes do not establish capacity for a full report.

Process variables override backend dotenv settings. Frontend precedence is process
environment, Streamlit secrets, then root `.env`. Provider keys stay on the backend.
With `APP_API_KEY` set, the frontend rejects an alternate destination before
constructing its HTTP client. Keys never belong in URLs or checked-in files.

Keep both databases on durable local storage with one API owner. The ignored
`data/` contains user research history and is intentionally preserved. It is not
disposable build output. Use consistent SQLite backups, not a copy of a live main
database file alone.

No paid hosting configuration is included. Deployment remains cancelled.
