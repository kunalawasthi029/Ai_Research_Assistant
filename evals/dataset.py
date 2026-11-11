"""Compatibility export; queries.yaml is the single versioned dataset source."""

from evals.protocol import load_dataset

EVAL_QUERIES = load_dataset()["queries"]
