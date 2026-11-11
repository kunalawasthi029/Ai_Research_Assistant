# Evaluation status

The implementation includes a frozen 20-query dataset, B0/B1/B2 arms, shared
initial evidence, independent judging, throttling, stage caching, retained failed
attempts, and paired query-blocked statistics.

Development smoke attempts exercised only the first query with up to three
repetitions per arm. They exposed schema, provider compatibility, context, and
quota failures. Configuration changed between attempts; they must not be pooled
as independent confirmation of improvement. The complete comparative study remains
unverified. Old development outputs are archived outside the project; failed
attempts have not been reclassified as successes.

To write new real-provider observations:

```powershell
python -m evals.runner --limit 1 --repetitions 1 --throttle 60 --timeout 1800 --output evals/results/local.json
```

This consumes provider quota. GitHub live evaluation is manual only and uses an
`evaluation` environment with provider secrets. Offline CI does not call providers.
