# Public testing boundary

Run the source-only test suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The retained tests use synthetic data and validate the feature, metric,
pipeline, and frozen-core behavior without accessing competition records.

The private workspace also contains integration checks coupled to `work/`
evidence tables, single-use authorization receipts, and an exact local project
root. Publishing those fixtures would disclose non-redistributable derived
competition material; they are deliberately omitted rather than weakened or
silently redirected.
