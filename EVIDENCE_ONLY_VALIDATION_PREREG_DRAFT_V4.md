# EVP-DRAFT-04 — sequential evidence-only validation

**Status:** `DRAFT_ONLY_AWAITING_INDEPENDENT_METHOD_AUDIT`  
**Date:** 2026-09-09  
**Authority:** design revision only; it authorizes neither source changes nor
result-bearing execution.

This version supersedes EVP-DRAFT-03 only as a proposal for future audit. The
v3 document and its receipts remain immutable history. The v3 implementation
is independently classified `SOURCE_REJECTED_UNEXECUTED`; no real model or
development performance has been run from it.

## Immutable provenance

| Artifact | SHA-256 | Meaning |
|---|---|---|
| EVP-DRAFT-03 | `54776c546377da214626828f71deed383302ceee95142073da7b031a0d1723e5` | rejected predecessor design |
| historical v3 static audit | `a45c55aae9cfac2b4caafb99511815b38b4da9c87c0b431edb985d554e117b91` | historical pre-implementation audit |
| v3 implementation containment | `f04b10963b9e466cda2cb819a668344464c7efd7ead98ebb772b6879b1e5f75c` | implementation present, unexecuted |
| v3 source-audit rejection | `90bb142996798652cc64adb863c93ff845452fb56363500cab969b68de213a1d` | four required v4 repairs |
| v4 metric-local coverage contract | `4e821092161be0b0c80892af41ed61718e0a4930562b748a826da8a8bf9001e6` | exact query/hand/evidence/pool cells |
| rejected v3 core source | `5871093ce9df04c7370db9c462fc85f77d15b8ebfb0e0dc65e65d7808c54005e` | provenance only, not executable authority |
| rejected v3 runner source | `5a0f5aec85d2bad1ea852c4782200316a57704aa28edc8a99e7fb27b2c7a5fad` | provenance only, not executable authority |

The rejected source files must remain byte-identical until this v4 document
receives an independent method-audit PASS and a separate implementation-only
authorization. A later implementation must have new versioned containment and
source-audit receipts; the hashes above can never be treated as a result-run
approval.

## Frozen source universe and label semantics

Only confirmed development labels and organizer evidence may participate in
supervision or validation. The following source hashes remain frozen:

| Source | SHA-256 |
|---|---|
| `development_labels.csv` | `40fcfa99d67db2ccbc37c56d4ef858676f1a677128db873de0e94bcea15f62a9` |
| `development_evidence.csv` | `f5889ee490b7dd0d03341b5505a1789071558bd5a5d1a31d61f25541460d33ba` |
| `development_pair_hands.parquet` | `cf197d2bf8b8242b83515b6db1a5a0c73c214ed91fc46fce9205014f07d265af` |
| B001 full OOF receipt | `71d7ce45170267060f7a1a6ae3e4a78a425d37bc4b4d88d392d8fe08b8baee36` |
| B001 time OOF receipt | `4f85a92ab6e72f3c71dd4d14b7f5ba57e621bd87c08a41955576d7d612273147` |
| B001 implementation | `4cae28dba762ba301bb58d0c75fa5533bd8d5e979a52764732b506e831a87b49` |
| B001 full features | `756d2e1e1b9d031755534ff975023a5f958ab36c53ec342b71c9f03f6448dc8b` |
| B001 early features | `0622e8f2b59cf1fb7b9109c4b9e536d5f5773d0bdee630b361134d9116db471d` |
| B001 late features | `943fbf666fc19ac2c1fb086e7c32f94df4ead0dcc8b65e85b9feade85cd22e1a` |
| `hands.parquet` | `82c9ad1c01cae6a68b28e194b9dd0bad9ca09f94628b429c0e58e970cc29eecd` |
| official metric notebook | `3cb11be5c999ada91aa91002f18aabc0c1c48548f1b6aedf3939f31fcd511c5c` |
| EV000 implementation/report | `7518ed8731fa128cf66abee99dc95dc639673dfd0c025735ad4bdb028e09a49d` / `27a8245e99419fce163e5d90843cd2585a1273a467d7841433dea732384a4726` |

- The supervised evidence universe is exactly 372 `confirmed_target` pairs,
  45,129 legal shared candidate hands, and 1,817 organizer evidence hands.
- The 1,488 `confirmed_non_target` pairs may not enter any evidence model or
  evidence metric. They remain available only to the separately frozen pair-risk
  workstream.
- Every unlisted development pair is unknown. Unknown is never a negative,
  never receives a ground-truth label, and never enters this study.
- Within a confirmed-target query, a non-evidence candidate is **unjudged**,
  not negative. It may appear only as the lower member of a within-query
  evidence-over-unjudged ranking preference.
- IDs are equality keys only. Identifier text, numeric content, substring,
  format, ordering, file order, and generator internals are prohibited from
  features, eligibility, hyperparameter choice, or scores.

## Hypothesis and methods

The single hypothesis remains: a family-conditioned pairwise linear evidence
ranker improves frozen family-routed heuristic MAP@5, and the improvement is
not explained solely by the global pairwise linear component.

### Shared five-by-three split

- Use the exact B001 five-fold whole-`table_id` outer mapping. A pool occurs in
  one outer fold only.
- Inside each outer-training partition, create three
  `StratifiedGroupKFold` splits grouped by `table_id`, with seed
  `8803 + outer_fold`.
- Refit the frozen B001 router on the two inner-training folds and predict the
  inner-valid fold. The exact router is
  `HistGradientBoostingClassifier(learning_rate=0.055, max_iter=280,
  max_leaf_nodes=31, min_samples_leaf=12, l2_regularization=2.5,
  early_stopping=False)`.
- Full, early-to-late, and late-to-early router seeds are 18401, 19401, and
  20401 respectively, plus `outer_fold * 10 + inner_fold`.
- Full routing uses the frozen 95 B001 features. Each time direction uses the
  frozen 77 time-stable B001 features from its permitted source and target
  halves.
- Persist and hash each nested route receipt before fitting any evidence model
  that consumes it. The receipt contains pair equality key, pool equality key,
  outer/inner fold, view, predicted family, and the three probabilities. It
  must not contain true family.

### H0, L0, and L1

H0 chooses exactly one descending gameplay signal from the held-out predicted
family: `directed_signal`, `soft_signal`, or `isolation_signal` for directed
transfer, soft play, or coordinated isolation.

L0 uses these 20 fields in this exact order:

`pair_contribution_bb`, `contribution_gap_bb`, `net_gap_bb`, `max_win_bb`,
`max_loss_bb`, `transfer_1_to_2_bb`, `transfer_2_to_1_bb`, `transfer_any_bb`,
`both_showdown`, `one_folded`, `pair_aggression`, `pair_passive`,
`expected_aggression`, `expected_passive`, `aggression_residual`,
`passive_residual`, `pair_pot_share`, `pair_raises`, `pair_folds`, and
`max_hole_strength`.

For every fitting partition, use training-hand median imputation, training-hand
linear 1%/99% winsorization, then training-hand median/IQR scaling after
winsorization; replace zero IQR by one. Transforms are fold-local. Identifiers,
identifier-derived values, labels, behavior columns, `phase_progress`, and row
order are prohibited numeric inputs.

For each fitting query, create every organizer-evidence over unjudged-candidate
difference and its symmetric reverse row. Both directions together give the
query total weight one. The frozen early directed-transfer/fold-3 query with
one evidence candidate and no unjudged comparator remains in preprocessing and
receipts with zero preferences; it is not counted as trainable.

Fit `LogisticRegression(C=selected_C, fit_intercept=False, solver="lbfgs",
max_iter=2000, tol=1e-8)` under deterministic single-thread execution. A
convergence warning is a hard stop.

L1 appends the 20 base fields multiplied by each of the three one-hot family
indicators, yielding 20 base plus 60 interaction columns. Training-side true
family is permitted for the interaction. Every held-out hand uses only its
cross-fitted B001 predicted family. L0 uses the exact C selected for L1, so the
full attribution changes only the 60 interaction columns.

## C-selection semantics — option B is frozen

This version deliberately chooses the no-family-grouping option from the source
audit. For each outer fold and view:

1. Evaluate only `C in [0.01, 0.1, 1.0, 10.0]` on pooled three-inner-fold OOF
   predictions.
2. Compute exact random-tie AP@5 once per eligible inner-valid query.
3. Select the greatest **unweighted query mean AP@5 over all eligible
   inner-valid queries**, with no family grouping, family weighting, family
   macro, family column, or family target in the selection frame.
4. Exact ties choose the smaller C.

Inner-valid true family is prohibited from the C-selection metric, feature
matrix, router input, eligibility, and score. Training-side true family remains
permitted only for fitting the B001 router and constructing L1 interactions.
Outer-valid true family is likewise never an input to a model, route,
eligibility rule, feature, or score. Only after the stage's per-hand scores and
deterministic top-five receipts have immutable hashes may the frozen confirmed
family label be joined as reporting metadata for the preregistered family
strata and gates. The score-freeze barrier and post-score join order must be
machine-audited.

## Canonical authorization contract

No version of this draft authorizes execution. A future executor must stop
before opening any competition-data path unless all of the following are true:

1. The authorization status is exactly `RESULT_BEARING_EXECUTION_APPROVED`.
2. The executor itself constructs and resolves the canonical containment path
   `poker/work/EVP_implementation_containment_audit.json`; it verifies SHA-256
   `f04b10963b9e466cda2cb819a668344464c7efd7ead98ebb772b6879b1e5f75c`.
   No path supplied by an authorization document is ever dereferenced.
3. The authorization's exact binding repeats that canonical path/hash and the
   rejected predecessor core/runner provenance hashes
   `5871093ce9df04c7370db9c462fc85f77d15b8ebfb0e0dc65e65d7808c54005e`
   and
   `5a0f5aec85d2bad1ea852c4782200316a57704aa28edc8a99e7fb27b2c7a5fad`.
   Any extra or alternative audit path is rejected.
4. The authorization also binds the eventual v4 draft hash, independent v4
   method-audit PASS hash, v4 implementation-containment canonical path/hash,
   v4 source-audit PASS canonical path/hash, and the exact executable v4 core
   and runner hashes established by that source audit. These values do not yet
   exist and may not be guessed.
5. Permissions are the exact narrow map: one preregistered development run is
   true; evaluation loading/scoring, E-number assignment, submission creation
   or modification, candidate promotion, and rerun after any stop/failure are
   false.
6. No consumption marker or stage/result artifact already exists. The executor
   atomically creates the one-run consumption marker before the first permitted
   development-data read. A failed or interrupted run consumes the permission.

Any missing field, extra permission, hash mismatch, noncanonical path, symlink
escape, pre-existing artifact, or runtime-version mismatch stops before data.

## Strict sequential execution and data-access barriers

The run is an append-only state machine. Each stage writes and hashes its raw
receipts before computing its aggregate metric, then writes a terminal PASS or
STOP journal record. A later stage may begin only from the immediately prior
stage's PASS hash. No rescue, reorder, overwrite, deletion, or rerun exists.

### Stage 0 — authorization and document integrity

Read only authorization, v3/v4 design receipts, source code, and runtime
versions. Do not open any development parquet/CSV or evaluation source. On
PASS, atomically consume the single authorization.

### Stage 1 — full-period primary

- Hash and open only development labels, organizer evidence, full pair-hand
  columns excluding `phase_progress`, B001 full OOF, B001 full features,
  gameplay `started_at`, and the official metric reference.
- Validate full coverage against the v4 coverage contract before any fit.
- Fit only nested full B001 routers and full L1; compute H0 and L1 scores.
  Do not construct or fit L0. Do not hash, open, or inspect B001 time OOF,
  early features, late features, or the `phase_progress` column.
- Persist and hash full nested routes, selected C, transforms, pairwise query
  receipts, per-hand H0/L1 scores, and deterministic top-five receipts. Only
  then join reporting family and compute the full primary metrics, sensitivity
  summaries, and primary gate.
- If the primary gate fails, write `STOP_PRIMARY_UNEXECUTED_LATER_STAGES` and
  return. Attribution and time models, metrics, score receipts, top-five
  receipts, serializers, source hashes, and data reads must all remain absent.

### Stage 2 — full-period family-conditioning attribution

Begin only from the exact Stage-1 PASS hash. Reuse the frozen Stage-1 selected
C and transform receipts. Fit only the matched-C full L0 on the same training
query/pool universe, then persist and hash L0 per-hand and top-five receipts.
Only afterward compute L1-minus-L0 attribution metrics and its gate. Do not
open any time-specific source or `phase_progress`.

If attribution fails, write `STOP_ATTRIBUTION_UNEXECUTED_TIME` and return. All
time source hashes, reads, models, metrics, and serializers must remain absent.

### Stage 3 — bidirectional time validation

Begin only from the exact Stage-2 PASS hash. Only now hash and open B001 time
OOF, early/late B001 features, and `phase_progress`. Freeze eligible masks
before any time score: early is `phase_progress <= 0.5`, late is `> 0.5`; a
direction includes only a query with at least one organizer evidence hand in
its target half.

Fit early-to-late and late-to-early H0/L1/L0 paths separately with their own
inner overall-MAP C selection and matched-C L0. Persist/hash all routes,
transforms, pairwise receipts, eligibility masks, per-hand scores, and top-five
receipts before joining reporting family or computing any time aggregate. Then
compute both directions and the frozen bidirectional gate. A failure is
terminal and never authorizes a variant or rerun.

## Metric-local coverage contract

The exact design artifact is
`poker/work/EVP_v4_metric_coverage_contract.json`, SHA-256
`4e821092161be0b0c80892af41ed61718e0a4930562b748a826da8a8bf9001e6`.
Its tuple order is `(eligible_queries, candidate_hands, evidence_hands,
eligible_pools)` and it freezes overall, family, outer-fold, and every
outer-fold-by-family cell for all three views.

Every scalar method metric and paired contrast must embed—beside the value,
not merely through a pointer—its exact local eligible-query, candidate-hand,
evidence-hand, and pool counts. This includes full/time overall, each family,
each outer fold, and all 45 view/fold/family cells. Methods in a view must have
identical query and candidate masks; contrasts require exact query-key equality
before subtraction.

Coverage is checked before each metric value is computed. Stop before that
metric and all later stages on any missing cell, nonexact frozen tuple, silent
exclusion, denominator change, full fold-family cell below 18 queries / 2,058
candidates / 90 evidence / 14 pools, or time fold-family cell below 7 queries /
458 candidates / 17 evidence / 6 pools. The exact-tuple rule is stricter than
the floor; both must independently pass. The two seven-query early-to-late
isolation cells are explicit hard stops.

Headline frozen totals are:

| View | Eligible queries | Candidate hands | Evidence hands | Pools |
|---|---:|---:|---:|---:|
| full | 372 | 45,129 | 1,817 | 245 |
| early-to-late target late | 264 | 17,375 | 617 | 197 |
| late-to-early target early | 347 | 22,560 | 1,200 | 237 |

## AP@5, ties, and immutable receipts

- Gate and selection AP@5 use the exact expectation under uniform order within
  exact score ties. For a tie block of `n` hands with `r` relevant, `a`
  positions and `R` relevant before it, position `j` contributes
  `(r/n) * (R + 1 + (j-1)*(r-1)/(n-1)) / (a+j)` for `n > 1`; use the observed
  contribution for `n = 1`. Sum through rank five and divide by
  `min(total_relevant, 5)`.
- The implementation must pass all 55 exhaustive one/two-block synthetic cases
  of total length at most four before any authorization.
- Deterministic top-five materialization sorts descending score, then official
  gameplay `started_at` ascending inside an exact tie. If tied hands also share
  `started_at`, stop. No pair/hand ID, file order, or row order fallback exists.
- Each stage's per-hand receipt includes equality keys, official timestamp,
  pool/fold, eligibility, predicted route, method scores, evidence flag, and
  time half when applicable. The score receipt and top-five receipt are hashed
  before aggregate metrics. Reporting true family is stored only in a separate
  post-score join receipt.

## Stage-local paired pool-weight sensitivity

The output name is exactly **90% paired Bayesian cluster-weight sensitivity
interval**. It is not a confidence interval, credible interval, significance
test, posterior probability, or coverage statement.

For each eligible view and outer fold, lexicographically enumerate `table_id`
equality keys solely to assign weights. Initialize
`numpy.random.default_rng(numpy.random.SeedSequence([12673, view_index,
outer_fold]))`, with indices full=0, early-to-late=1, late-to-early=2. Generate
exactly 5,000 independent Exp(1) pool weights and divide each draw by its
within-view/fold mean. All queries inherit their pool weight. Identical realized
weights are shared across H0/L0/L1, contrasts, folds, and reporting families.
Never resample hands/queries, create family-specific weights, reject a draw,
omit a draw, or zero-fill a cell.

Form query-level paired deltas before weighting. Report original unweighted
point means. Sensitivity endpoints are unrounded linear 0.05/0.95 quantiles of
weighted query means; macro-family draws are arithmetic means of the three
same-draw family means.

To preserve stage barriers, verify only the weight stream for the stage whose
data are authorized:

| Stream | With view/fold headers | No-header negative control |
|---|---|---|
| Stage 1/2 full only | `a00923786962622bacfd63fb211670ef14eb657a4a96e5411274ff4034777a2e` | `e57b0a7b15873641e429a964d79075a0f56e997b675caab8e9a2a7f40ab1f16d` |
| Stage 3 early-to-late | `4b009fee27edc3e662fe9cda00bcc9ca5ad84e1d16c7b4d253ce2f9d1dd39471` | `5afbd956bc156cd2b35f0270ba5637f8fbd4b6a973b6c2c4465b721987ae856f` |
| Stage 3 late-to-early | `938473e74b7609cccea365606f1f468ecd7958782704229ba8f40cd31f057cf0` | `ee671cf04e877e747254b61c669013fc41957fb91dd08e26dd5c67f3498a68fc` |
| Stage 3 both time views | `380969a4fa506e7cd9734be8ba5b0e55fc9d3a7ebd82864762e43ec36a7468fe` | `4bad16bff34f63c0caa336b128ce2994e70b49c0ec85bd582f3b44ab213ee3b2` |
| Historical all views | `c833ac7d6dd02d2009f41f3566667dcf37e59071646cc1afe9929395b601975c` | `08ee7564ab1c5aa7736c1fc541a006f6c5181b2638088a6b91d24e034a222e1a` |

Each block is UTF-8 view bytes, one raw fold byte 0 through 4, then its
mean-normalized `(5000, pool_count)` matrix as little-endian float64 C-order
bytes. The historical hashes do not bind pool IDs or shapes; the exact coverage
contract, source hashes, pool membership, per-fold pool counts, shapes, and
positive denominators are mandatory companion checks.

## Frozen sequential gates

All comparisons use unrounded values.

1. **Stage 1 primary full-period:** L1 minus H0 query-weighted overall MAP@5
   at least +0.040; overall lower sensitivity endpoint above zero; every
   post-score true-family point delta nonnegative; at least four of five outer
   fold point deltas positive.
2. **Stage 2 attribution:** full L1 minus matched-C L0 unweighted macro-family
   point delta at least +0.005; coordinated-isolation point delta nonnegative.
3. **Stage 3 bidirectional time:** separately in each direction, L1 minus H0
   unweighted macro-family point delta at least +0.020, query-weighted overall
   point delta positive, overall lower sensitivity endpoint above zero, every
   direction-family point delta nonnegative, and at least four of five outer
   fold point deltas positive.

Integrity, authorization, source, route, tie, weight, and metric-local coverage
checks precede the applicable stage's performance. A failed check or gate is
terminal. Passing all gates establishes only an owned development signal; it
does not authorize evaluation scoring, candidate promotion, an experiment
number, or submission.

## Required audit and terminal controls

Before any source implementation change, an independent method audit must bind
this v4 document and coverage contract and return PASS. After any separately
authorized implementation-only work, a different independent source audit must
verify canonical authorization binding, absence of caller-selected audit paths,
held-out-family barriers, AST/data-open stage order, exact local coverage,
stage-local weights, tie logic, and tests. Only then may supervision consider a
one-run result authorization.

Every terminal journal must state explicit booleans for each stage's data
access, route/model fit, score serialization, aggregate metric computation,
family reporting join, gate result, real model fit, aggregate performance
viewed, evaluation rows loaded/scored, unknown labels assigned, experiment
number assigned, candidate promoted, submission created/modified, and rerun
allowed. Before separate result authorization, all real-run fields remain
false.
