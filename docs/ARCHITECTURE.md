# Architecture

Streamlit calls FastAPI over HTTP. The backend owns credentials, dispatch, SQLite
state, and graph execution. Core agents/persistence do not import Streamlit or
FastAPI. The CLI uses the same research service.

| Role | Responsibility |
|---|---|
| Supervisor | Review plans, evidence, and validation; select legal routes |
| Planner | Produce typed tasks and fresh tasks for identified gaps |
| Researcher | Choose tools and assess task completion within budgets |
| Analyst | Map evidence to findings and identify gaps/contradictions |
| Report generator | Produce versioned reports with source citations |
| Validator | Deterministic provenance/coverage checks, then semantic assessment |

Roles have separate prompts and graph boundaries. Model IDs are configurable;
roles can share a model. This does not establish independent model criticism.
Benchmark judging is a separate evaluation path.

The six tools are web search, webpage extraction, Wikipedia search, arXiv search,
calculator, and source verifier. Calculations/accessibility observations are not
factual source evidence. Network tools use public-only DNS-pinned fetching,
checked redirects, bounded downloads, and invocation/time budgets.

## Execution

The API returns a run ID when a job is queued. A bounded dispatcher owns worker
subprocesses, deadlines, cancellation, and shutdown. An ownership lease prevents
overlapping API instances from marking each other's jobs abandoned. Run one
Uvicorn process per database.

SQLite stores runs, append-only events, evidence, versioned reports, and validation.
A separate database holds LangGraph checkpoints. SSE exposes stored events with
reconnect cursors. Eligible interrupted work can resume; in-flight operations may
replay. Frontend reruns do not own backend jobs.

Research/revision/tool/time limits are finite. After the last allowed tool result,
the researcher gets one terminal assessment that cannot execute more tools or
retry malformed output. Independent analysis and validation remain required.

## Evidence and citations

Evidence IDs, task IDs, timestamps, content, and metadata remain separate records.
Focused extraction selects bounded verbatim passages with original character
ranges. Context packing prioritizes full documents and distinct URLs, distributes
space across extracted records, and reports omissions/truncation.

One reference represents one source URL. Its `evidence_id` identifies the primary
excerpt; `supporting_evidence_ids` links other retrieved excerpts from the exact
same URL and source type. Legacy references default to an empty list. Sections
declare the precise excerpt IDs they use; reference membership alone does not
establish coverage of every task represented by those excerpts.

Generation guards and independent deterministic checks reject invented IDs,
metadata mismatches, duplicate sources, invalid excerpt membership, unresolved
citations, and uncited task evidence. Semantic validation receives the actual
report and visible excerpts. Structural validity does not prove factual accuracy.

## Access

Optional shared-key authentication protects data/job routes. Health and CORS
preflight remain public. The frontend sends the key in a header and pins requests
to its configured backend. CORS is not authentication. There is no per-user history
isolation. See [configuration](ENVIRONMENT.md), [diagrams](graph.md), and
[limitations](LIMITATIONS.md).
