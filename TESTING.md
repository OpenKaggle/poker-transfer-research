# Public testing boundary

Run the source-only test suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite uses synthetic data and validates the feature, metric, pipeline,
source separation, and frozen-core behavior without accessing competition
records. The public-root adapter makes this command work directly from a
normal clone rather than requiring the former parent-workspace layout.

Some retained assertions are explicitly skipped because they require omitted
competition-linked evidence receipts or single-use authorizations under
`work/`. Their skip reason is reported by `unittest`; they are not silently
redirected to substitute material. The `static_audit_*.py` scripts retain the
historical audit logic and likewise print an explicit skip when that private
receipt bundle is absent.
