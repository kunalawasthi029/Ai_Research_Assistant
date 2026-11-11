# Verification status

Reviewed 2026-11-11. Evidence levels are separate; passing offline checks does not
establish successful live research, measured accuracy, or hosted readiness.

## Unit and mocked integration evidence

The regression suite exercises schemas, bounded SDK repair, model pacing, tools,
SSRF/redirect checks, source provenance, graph routing, process deadlines,
SQLite persistence/resume, API authentication/SSE, Streamlit behavior, and evaluation.
The exact final test result is recorded in `checks.json` after the release checks.

Recent repairs cover:

- Actual planner data-object output, required fields, budgets, and supplied URLs.
- Safe field/type diagnostics from provider-rejected JSON without accepting or
  retaining rejected/private output; the existing finite repair budget remains.
- Original research objectives and source URLs reaching each researcher task.
- Bounded verbatim extraction, explicit truncation, full-document preference, and
  source diversity within small contexts.
- One terminal assessment of the last allowed tool result without excess tool calls.
- One reference linking multiple same-source excerpts, with independent membership,
  source/type, duplicate, citation, and task-coverage checks. Markdown exports retain
  every linked excerpt ID. Existing single-excerpt reports remain compatible.
- Shared-key backend authentication, fail-closed required-key configuration, and
  a server-pinned frontend destination so edited UI state cannot redirect the key.

## Historical real-provider evidence

Live development runs progressed past the original planner failure and retrieved
primary sources/generated reports. Other runs hit schema errors, quota limits,
context omissions, or validation issues and remained best effort/incomplete.

Run `27cc1476-98bc-4e5d-ab6f-3852273b07d7` received an automated semantic pass but
manual inspection found an unsupported causal statement in its conclusion. It is
not treated as factual acceptance. A later run, `9587bedd-7aab-45cb-a50a-ca3bbaee3baa`,
retrieved both RFCs but exposed repeated-source context crowding and duplicate
reference limitations; these motivated the latest citation/context changes.

The final citation changes are verified offline, not by a fresh complete
real-provider acceptance run. An actual Tavily search/cache hit and a Groq tool
round trip were verified earlier. These do not prove full workflow reliability.
Earlier raw diagnostics and failed attempts are preserved in the external local
archive, outside the cleaned project.

## Remote and deployment evidence

GitHub CI is an additional independent execution of the offline matrix. Check the
repository's Actions page for its current result; local tests alone do not imply
that remote CI passed. Provider-consuming evaluation remains manual only.

No deployment was performed. Deployment was cancelled at the user's request.
Cloud secrets, persistence, proxy SSE, and restart recovery remain unverified.
The full comparative benchmark remains incomplete; see [evaluation status](EVAL_RESULTS.md).

## Cleanup and publication

Outdated phase diaries, repeated diagnostic artifacts, cancelled hosting config,
duplicate environments, and generated build/cache output are archived outside the
project. Required source, tests, lockfiles, current docs, the active Python runtime,
one local environment file, and user research databases are preserved.

The previous local Git history contains credential-shaped values and is archived
privately instead of pushed. The GitHub publication is a new scanned source snapshot;
it does not rewrite the preserved original history or backdate commits.
