# Release manifest

## Included

- user-authored feature, validation, model, and evidence-gating code;
- 15 source-only tests, requirements, and self-authored preregistration/
  experiment records;
- links to the official competition and metric.

## Excluded

- `work/` (all row-level and derived tables), competition data, and submissions;
- `reference/` (third-party notebook and metric copies);
- competition-linked evidence fixtures and single-use authorizations under
  `work/`; the corresponding public test assertions are retained with explicit
  skip reasons;
- virtual environments, caches, credentials, model artifacts, and bytecode.

This public source archive documents the method, not any participant-only data
or a claim that the recorded submission is currently eligible or competitive.
