# Known limitations

- Models can produce malformed or unsupported output despite strict schemas.
  Repairs are bounded. Semantic judges can miss errors, especially when roles
  share a model. A passing verdict is not independent factual verification.
- Research can end best effort or incomplete when retrieval, budgets, deadlines,
  or provider quota prevent completion. Partial artifacts remain labelled.
- Bounded context and lexical excerpt selection can omit relevant facts. Source
  traceability is not semantic support. Cached results preserve old retrieval dates.
- Per-model pacing is local to one worker; it does not track organization-wide
  quota or other clients. Unlimited free provider usage is never assumed.
- Shared-key authentication does not provide per-user history isolation. Use the
  app locally or as a controlled single-owner system.
- SQLite requires one backend owner. Restart recovery may replay an in-flight
  operation. Cloud persistence, proxy streaming, and hosted recovery are unverified.
- Free ephemeral hosting cannot retain these local SQLite files across redeploys.
  No free durable cloud architecture has been implemented or deployed.
- The complete comparative evaluation remains unfinished. No measured general
  reliability improvement or production-readiness claim is made.

See [verification](VERIFICATION.md) for separate evidence levels.
