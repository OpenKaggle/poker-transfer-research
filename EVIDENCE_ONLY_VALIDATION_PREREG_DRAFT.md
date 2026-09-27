# Nested evidence-only validation preregistration — draft

Document ID: `EVP-DRAFT-01`  
State: **draft only; execution forbidden pending a separate supervisory review**  
Experiment ID: none. This document is not E013 and does not reserve an E-number.

## Authorization boundary

This draft specifies one owned, nested validation of evidence ranking. Creating
the document does not authorize implementation, model fitting, metric
calculation, evaluation scoring, candidate promotion, or a Kaggle submission.
After implementation is separately approved, its source and frozen constants
must be audited before one result-bearing run is authorized.

Pair risk, behavior activation, and every evaluation artifact remain untouched.
The only public-notebook pattern retained is the need for a common outer fold
and cross-fitted predicted-family routing on held-out queries. The proposed
learner does **not** reuse the public notebook's XGBoost model, PU weights,
feature list, rank-percentile features, hyperparameters, same-OOF activation
selection, or outer-validation tree selection.

## Frozen question and inputs

The sole primary question is whether an independently specified,
family-conditioned pairwise linear ranker improves the frozen family-routed
heuristic on exactly the same held-out confirmed-target queries and legal
shared hands.

Frozen sources:

| Artifact | SHA-256 |
| --- | --- |
| `development_labels.csv` | `40fcfa99d67db2ccbc37c56d4ef858676f1a677128db873de0e94bcea15f62a9` |
| `development_evidence.csv` | `f5889ee490b7dd0d03341b5505a1789071558bd5a5d1a31d61f25541460d33ba` |
| `development_pair_hands.parquet` | `cf197d2bf8b8242b83515b6db1a5a0c73c214ed91fc46fce9205014f07d265af` |
| B001 full OOF receipt | `71d7ce45170267060f7a1a6ae3e4a78a425d37bc4b4d88d392d8fe08b8baee36` |
| B001 time OOF receipt | `4f85a92ab6e72f3c71dd4d14b7f5ba57e621bd87c08a41955576d7d612273147` |
| B001 behavior implementation | `4cae28dba762ba301bb58d0c75fa5533bd8d5e979a52764732b506e831a87b49` |
| B001 full pair features | `756d2e1e1b9d031755534ff975023a5f958ab36c53ec342b71c9f03f6448dc8b` |
| B001 early pair features | `0622e8f2b59cf1fb7b9109c4b9e536d5f5773d0bdee630b361134d9116db471d` |
| B001 late pair features | `943fbf666fc19ac2c1fb086e7c32f94df4ead0dcc8b65e85b9feade85cd22e1a` |

Only the 372 `confirmed_target` pairs and their 1,817 organizer evidence hands
enter this study. The 1,488 confirmed non-target pairs and every unlisted pair
are excluded from fitting, preprocessing, selection, and metrics. Unknown
ground-truth labels assigned: zero. A shared hand not listed by the organizer
is an **unjudged candidate**, never a ground-truth negative; it may appear only
as the lower member of a within-query preference used to learn a ranking.

## Frozen split and routing contract

- The outer split is the exact five-fold whole-pool mapping in the B001 full
  receipt. A table pool may occur in one outer fold only.
- For each outer fold, all preprocessing and fitting use the other four folds.
  The outer-validation fold is scored once and is never used for feature
  choice, regularization choice, early stopping, ablation choice, or reruns.
- Within each outer-training partition, recreate three whole-pool
  `StratifiedGroupKFold` splits with seed `8803 + outer_fold`, stratified by true
  family and grouped by `table_id`. For each inner fold, refit the exact frozen
  B001 behavior recipe on the other two inner folds' confirmed targets and
  predict the inner-valid targets. Full-period routing uses the 95 frozen B001
  features; early-to-late and late-to-early routing use the 77 frozen
  time-stable features from the permitted source and target halves. The B001
  router is `HistGradientBoostingClassifier(learning_rate=0.055,
  max_iter=280, max_leaf_nodes=31, min_samples_leaf=12,
  l2_regularization=2.5, early_stopping=False)`; full, early-to-late, and
  late-to-early nested seeds are respectively `18401`, `19401`, and `20401`,
  plus `outer_fold * 10 + inner_fold`.
- Evidence hyperparameters are chosen only from pooled inner OOF predictions
  routed by those nested cross-fitted B001 predictions. True family is allowed
  only on the fitting side of each inner or outer evidence model. In
  particular, an inner-valid family is never used to choose `C`, features, or
  an evidence score. No tree-count selection exists in the evidence method.
- Full-period outer-validation hands are routed with B001 `predicted_family`;
  early-to-late hands use B001
  `early_to_late__predicted_family`; late-to-early hands use B001
  `late_to_early__predicted_family`. The organizer family of an outer-valid pair
  is used only after scores are frozen, for reporting family metrics.
- Full-period scoring uses every legal shared hand. A time-direction query is
  included only when its target half contains at least one organizer evidence
  hand. Early is `phase_progress <= 0.5`; late is `phase_progress > 0.5`.
  The inclusion mask is frozen before any score is computed.

## Frozen methods

### H0 — family-routed heuristic

For every held-out query, use the same frozen predicted-family routing as the
learned method. Sort descending by `directed_signal`, `soft_signal`, or
`isolation_signal` for predicted directed transfer, soft play, or coordinated
isolation respectively. H0 and all learned variants receive identical query
and candidate-hand masks.

### L0 — independent global pairwise linear ranker

Use exactly these 20 gameplay fields, in this order:

`pair_contribution_bb`, `contribution_gap_bb`, `net_gap_bb`, `max_win_bb`,
`max_loss_bb`, `transfer_1_to_2_bb`, `transfer_2_to_1_bb`, `transfer_any_bb`,
`both_showdown`, `one_folded`, `pair_aggression`, `pair_passive`,
`expected_aggression`, `expected_passive`, `aggression_residual`,
`passive_residual`, `pair_pot_share`, `pair_raises`, `pair_folds`, and
`max_hole_strength`.

For each fitting partition, impute by the training-hand median, winsorize at
training-hand 1%/99% quantiles, and standardize by the training-hand median and
IQR; a zero IQR is replaced by one. All fitted transforms are fold-local.
Identifiers, identifier-derived values, row order, `phase_progress`, labels,
and behavior columns are prohibited as numeric features.

Within each training query, create a preference for every organizer evidence
hand over every unjudged candidate hand. Add the reverse ordered difference so
binary classes are symmetric. Weight the resulting rows so each query has
total weight one, regardless of its number of candidates or evidence hands.
One frozen early-half directed-transfer query has one candidate hand and that
hand is organizer evidence, so it has no unjudged comparator and contributes
zero preference rows. It is retained in receipts and preprocessing coverage but
is not counted as pairwise-trainable; no query may be silently dropped.
Fit `LogisticRegression` with `fit_intercept=False`, `solver="lbfgs"`,
`max_iter=2000`, `tol=1e-8`, and deterministic single-thread execution. Select
`C` only from `[0.01, 0.1, 1.0, 10.0]` by pooled three-inner-fold macro-family
random-tie MAP@5; exact ties choose the smaller `C`. Score an individual hand
by the learned coefficient dot product.

### L1 — sole controlled ablation and primary learned method

Append the 20 base fields multiplied by each of the three one-hot family
indicators. The training indicator is the permitted true family; the held-out
indicator is the applicable frozen B001 predicted family. This produces 80
columns: 20 base fields plus 60 interactions. Use the L1-selected `C` for the
matched L0 ablation, so L0 versus L1 changes only the interaction block. No
other feature set, model family, seed, hyperparameter grid, blend, or rescue
variant is allowed.

The full-period, early-to-late, and late-to-early models are fitted separately.
Each direction selects its own `C` inside the corresponding outer-training
partition, using only the permitted source half for fitting and the target half
for inner validation. Persist and hash every nested B001 inner-route receipt
before fitting an evidence ranker; inner family accuracy is an integrity
diagnostic only and cannot choose or rescue a configuration.

## Frozen coverage receipt

All sequences below are ordered outer folds 0 through 4. Exact equality is an
integrity condition, not a result-dependent filter.

Full-period outer-validation coverage:

| Family | Total pairs | Candidate hands | Evidence hands | Valid pairs F0–F4 | Evidence hands F0–F4 |
| --- | ---: | ---: | ---: | --- | --- |
| Directed transfer | 148 | 18,150 | 725 | 30/29/30/30/29 | 148/142/149/147/139 |
| Soft play | 132 | 15,699 | 632 | 26/26/27/27/26 | 121/128/124/132/127 |
| Coordinated isolation | 92 | 11,280 | 460 | 18/18/19/18/19 | 90/90/95/90/95 |

Frozen feature and preference readiness:

| Scope | Evidence-bearing pairs | Pairwise-trainable pairs | Candidate minimum | Unjudged minimum | Preference pairs | Symmetric fit rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 372 | 372 | 57 | 52 | 212,888 | 425,776 |
| Early | 347 | **346** | 1 | **0** | 82,266 | 164,532 |
| Late | 264 | 264 | 10 | 9 | 42,763 | 85,526 |

All 45,129 rows contain finite values for all 20 frozen features: missing
values zero, non-finite values zero, and globally constant frozen features
zero. “Preference pairs” counts evidence-vs-unjudged ordered differences before
adding the symmetric reverse rows.

Time-direction outer coverage:

The source-pair column reports `evidence-bearing / pairwise-trainable` for each
fold.

| Direction | Family | Source pairs F0–F4 (bearing/trainable) | Source evidence F0–F4 | Target-valid pairs F0–F4 | Target evidence F0–F4 |
| --- | --- | --- | --- | --- | --- |
| Early→late | Directed transfer | 108/107 · 108/107 · 110/109 · 109/109 · 109/108 | 362/377/367/356/382 | 23/23/22/19/24 | 49/58/55/42/60 |
| Early→late | Soft play | 100/100 · 100/100 · 98/98 · 97/97 · 101/101 | 326/324/317/312/329 | 20/21/20/22/23 | 45/50/39/42/54 |
| Early→late | Coordinated isolation | 70/70 · 72/72 · 68/68 · 69/69 · 69/69 | 264/283/270/271/260 | **7/11/12/10/7** | 17/36/28/24/18 |
| Late→early | Directed transfer | 88/88 · 88/88 · 89/89 · 92/92 · 87/87 | 215/206/209/222/204 | 28/28/26/27/27 | 99/84/94/105/79 |
| Late→early | Soft play | 86/86 · 85/85 · 86/86 · 84/84 · 83/83 | 185/180/191/188/176 | 24/24/26/27/23 | 76/78/85/90/73 |
| Late→early | Coordinated isolation | 40/40 · 36/36 · 35/35 · 37/37 · 40/40 | 106/87/95/99/105 | 17/15/19/18/18 | 73/54/67/66/77 |

Nested time-selection ranges across all outer fold, inner fold, and family
cells:

The source column reports the range of evidence-bearing pairs followed by the
range of pairwise-trainable pairs.

| Direction | Family | Source pairs (bearing / trainable) | Source evidence | Target-inner-valid pairs | Target evidence |
| --- | --- | ---: | ---: | ---: | ---: |
| Early→late | Directed transfer | 70–76 / **69–75** | 231–263 | 26–33 | 62–80 |
| Early→late | Soft play | 63–68 / 63–68 | 202–228 | 24–31 | 53–70 |
| Early→late | Coordinated isolation | 44–49 / 44–49 | 170–192 | 9–16 | 25–42 |
| Late→early | Directed transfer | 56–65 / 56–65 | 125–153 | 34–38 | 112–135 |
| Late→early | Soft play | 54–60 / 54–60 | 109–137 | 31–35 | 98–115 |
| Late→early | Coordinated isolation | 19–29 / 19–29 | 53–74 | 21–25 | 78–100 |

The early-to-late coordinated-isolation cells in outer folds 0 and 4 contain
only seven validation pairs each. Seven is the frozen minimum, not evidence of
adequate power. If either cell loses even one pair or has fewer than its frozen
17/18 target evidence hands because of a join, missing feature, invalid score,
or post-hoc exclusion, the entire study stops before performance is interpreted.
Those two cells and their wide uncertainty must be reported even if all other
gates pass.

## Metric, ties, receipts, and uncertainty

- Query AP@5 is the organizer definition: sum precision at each relevant rank
  through rank five, divided by `min(number_of_organizer_evidence_hands, 5)`.
  Overall MAP@5 is the unweighted mean over eligible queries.
- Score ties use exact expected AP under a uniform random ordering within every
  tied-score block. Neither `pair_id`, `hand_id`, file order, nor a library's
  grouped-tie convention may resolve metric ties. Exact organizer stable-order
  MAP may be recorded only as a reproduction check and cannot select or pass a
  method.
- Report H0, L0, and L1 for full overall, macro family, each family, every outer
  fold, both time directions, and each direction-family cell. Report exact
  eligible query/evidence/candidate coverage beside every time metric.
- Persist one immutable per-hand outer-OOF receipt with pair, hand, outer fold,
  eligibility mask, predicted route, H0/L0/L1 scores, evidence flag, and time
  half, plus exact top-five receipts. Hash receipts before aggregate metrics are
  viewed.
- For paired L1-minus-H0 and L1-minus-L0 deltas, report a 90% cluster-bootstrap
  interval from 5,000 resamples of table pools within outer fold, seed 12673.
  Intervals are descriptive where the number of eligible pools is small; no
  cell may be suppressed for width or sign.
- Evidence scores are ranks, not probabilities. No calibration, risk threshold,
  behavior activation rate, or evaluation prevalence is estimated or changed.

## Frozen stop/go gates

Evaluate the following in order. Any failure ends the study; it cannot trigger
a rescue model, a second run, evaluation scoring, or submission.

1. **Integrity and coverage:** all frozen hashes, fold mappings, legal-hand
   joins, row counts, and every coverage sequence above match exactly; unknown
   pairs and confirmed non-target pairs used are both zero. The seven-pair late
   isolation minimum and the 17/18 evidence-hand counts must remain intact.
   Exactly one, and no more than one, early evidence-bearing query may have zero
   pairwise preferences; it must be the frozen directed-transfer/fold-3 cell,
   remain visible in receipts, and never be counted as trainable.
   Every inner-valid query must have exactly one nested B001 predicted-family
   route generated without its own family label; any missing route, true-family
   substitution, or use of an outer-valid row in an inner router stops the run.
2. **Primary full-period gate:** L1 minus H0 overall MAP@5 is at least +0.040,
   its paired 90% cluster-bootstrap lower bound is above zero, all three family
   deltas are non-negative, and at least four of five outer-fold deltas are
   positive.
3. **Family-conditioning attribution gate:** L1 minus the matched-C L0
   macro-family MAP@5 is at least +0.005 and the coordinated-isolation delta is
   non-negative. This is the only authorized ablation.
4. **Bidirectional time gate:** in **each** direction, L1 minus H0
   macro-family MAP@5 is at least +0.020, the overall delta is positive with a
   paired 90% cluster-bootstrap lower bound above zero, every one of the six
   direction-family deltas is non-negative, and at least four of five
   direction-level outer-fold deltas are positive.

Passing every gate would establish only an owned development evidence
candidate eligible for supervisory review. It would not validate pair-risk
transfer, authorize evaluation scoring, produce either of the two final frozen
candidates, or authorize a Kaggle submission.

## Required terminal controls

The run receipt must end with explicit booleans for model fit, aggregate metrics
viewed, evaluation rows loaded/scored, submission created/modified, unknown
labels assigned, and gate status. Before separate execution authorization, all
must remain false except the existence of this draft and the receipt-only
EV000 feasibility audit.
