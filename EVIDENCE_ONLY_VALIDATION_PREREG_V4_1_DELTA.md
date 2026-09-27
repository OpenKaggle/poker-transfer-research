# EVP-DRAFT-04.1 — source-integrity implementation delta

**Status:** `IMPLEMENTATION_DELTA_ONLY_AWAITING_FRESH_SOURCE_AUDIT`  
**Date:** 2026-09-11  
**Authority:** local versioned implementation repair only. This document does not
authorize development-data access, real model fitting, aggregate performance,
evaluation access, experiment numbering, candidate promotion, submission, or a
rerun.

This delta supplements, and does not overwrite, EVP-DRAFT-04. The statistical
hypothesis, source universe, folds, features, models, C grid, tie handling,
coverage contract, weight streams, gates, and forbidden actions remain exactly
as specified by the base document. The v4 core, runner, tests, static audit, and
containment receipt remain immutable rejected audit objects.

## Immutable base binding

| Artifact | SHA-256 |
|---|---|
| EVP-DRAFT-04 | `3a9f20936f4173f72cb4ff867d296eb2945b43ca40cac0ee587b7fb4b83da22c` |
| v4 coverage contract | `4e821092161be0b0c80892af41ed61718e0a4930562b748a826da8a8bf9001e6` |
| v4 method-audit PASS | `7ab04d40cff489323ee85f27484b93d57e6efdce06b09a78eeccc9cd0d22e913` |
| rejected v4 core | `1d02aa88b4a690aa088ea1dcdfda32ead25acb2e9236ccb7457c17e0a484a62d` |
| rejected v4 runner | `4737f3c4983b37042d30d801dc0407625d7dc95a155a97bcb9f92417cfd59098` |
| rejected v4 core tests | `fa1df57317f09dd2a6de09bfcc9b1d2fc09a1ed7d53351e8fdc0cfdab29d5304` |
| rejected v4 runner tests | `bbb3a6ca2c8dc73010aaa68e994dce3f9292569e02c3ea6bbf24e3a068c42224` |
| rejected v4 static audit | `3c6b77d9bc15a92ea8a3874d4af99770f1cab31696564a7bedbd02f008703711` |
| v4 containment | `f8a2e27b4e41044f165badd05bc4f96d4883d6a7204d1f59f02a18292ac9e232` |

## Mandatory v4.1 repairs

### 1. Pre-data runtime and dependency lock

Before consuming a future authorization or hashing/opening any competition
source, the executor must verify the exact Python, NumPy, pandas,
scikit-learn, SciPy, joblib, threadpoolctl, and PyArrow versions frozen by the
v4.1 source. It must also hash the imported local schema dependency. Any
mismatch stops at Stage 0 without a development-data read.

### 2. Independent immutable raw receipts

Before any aggregate metric or gate is computed, every applicable stage must
write with exclusive-create semantics and hash independent receipts for:

- selected C and all four inner-OOF means;
- every fold-local transform used by inner selection or outer fitting;
- every per-query pairwise construction receipt, including zero-preference
  queries;
- Stage-3 eligibility masks;
- routes, per-hand scores, and deterministic top five.

The terminal journal may point to these hashes but is not itself a substitute
for a pre-aggregate raw receipt. Stage 2 must reconstruct its L0 fitting states
only by reloading and validating the frozen Stage-1 selected-C and outer
transform receipts. An in-memory Stage-1 fit object is not an authorized input.

### 3. Append-only execution and exception terminalization

Every result artifact and route file uses exclusive-create publication. No
result target is replaced or overwritten. After a future exact authorization
has been accepted, every caught exception writes one exclusive terminal STOP
journal before re-raising. A failed or interrupted run remains consumed and is
never resumed or rerun. Best-effort temporary-file cleanup is not deletion of a
published result artifact.

### 4. Complete terminal controls

Every normal PASS/STOP journal and exception STOP journal contains explicit
booleans for each stage's development-data access, route fit, L1 fit, L0 fit,
score serialization, aggregate metric computation, reporting-family join, and
gate result. It also contains explicit booleans for real model fitting,
aggregate performance viewed, evaluation rows loaded/scored, unknown labels
assigned, experiment number assigned, candidate promoted, submission
created/modified, and rerun allowed.

### 5. Strict held-out-family freeze barrier

Before the applicable score and top-five hashes exist, held-out true family may
not be joined or inspected for coverage, readiness, routing, selection,
eligibility, features, or scoring. Pre-score structural checks are restricted to
overall and outer-fold coverage without family. Exact family and
outer-fold-by-family coverage is validated only after the reporting-only family
join, following immutable score and top-five hashes. The historical special
zero-preference query is checked pre-score only by its count and outer fold; its
true-family classification is checked only after the reporting barrier.

## Required next gate

Only pure synthetic/unit/static verification is allowed during implementation.
When the v4.1 implementation, tests, static audit, this delta, and the v4.1
containment receipt have exact hashes, a new independent source audit must bind
that entire bundle. This implementation may not self-issue that PASS. Even a
fresh source-audit PASS would not authorize result-bearing execution; a later,
separate exact v4.1 result authorization would still be required.
