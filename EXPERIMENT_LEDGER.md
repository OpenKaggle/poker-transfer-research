# Poker experiment ledger

All times are Asia/Shanghai. Unknown development pairs are never treated as
negative unless an entry explicitly names a PU method and keeps their target
latent. Player, pair, hand, and table identifier formats and source-file row
order are prohibited features.

## Competition snapshot

- Snapshot time: 2026-09-09 04:48 CST.
- Account entry: confirmed (`userHasEntered=True`).
- Deadline: 2026-09-21 06:00 CST (2026-09-20 22:00 UTC).
- Metric: 0.70 Pair AP + 0.20 Evidence MAP@5 + 0.10 three-family
  Behavior macro AP.
- Dynamic public frontier reported by research A at this snapshot: 0.93601,
  0.90098, 0.89765; 99 teams. These are timestamped observations, not fixed
  targets.

## Data receipt

- Official archive SHA-256:
  `8080c32f7acd6ad1e76dd332885210db3a2e5625f170db80578626259aae4c45`.
- Eight extracted tables pass the local official-schema audit.
- Counts: 12,000 players; 2,000,000 hands; 12,000,000 seat rows;
  18,609,028 ordered actions; 1,860 confirmed development labels; 1,817
  evidence labels; 112,540 evaluation pairs.
- Confirmed supervision: 372 `confirmed_target`, 1,488
  `confirmed_non_target`; all other development pairs remain unknown.
- Pool boundary check: co-seating graph has exactly 400 connected components
  of 30 players; each table maps to one component and each pair stays within a
  single component. Whole-table validation is therefore whole-pool validation
  in this release.

## E000 — emergency schema (local only)

- Artifact: `submissions/poker_emergency_schema.csv`.
- Purpose: answer only whether dynamic row coverage, value domains, and the
  eight-column submission schema are valid.
- Model: exposure-rank placeholder; behavior `none`; evidence `NO_EVIDENCE`.
- Decision: never use as a competitive submission. Remote read-only API calls
  already verify connectivity.

## E001 — owned residual_v1 anchor

- Parent: owned `poker_attack` baseline.
- Training labels: confirmed rows only; unknown pairs used as negatives: 0.
- Split: five whole-pool folds.
- Features: gameplay-only directionality, chip flow, player-versus-own-style
  residuals, phase-quartile burst, exposure, and permitted player metadata
  comparisons. Identifier values and row order are excluded.
- Pair AP OOF: 0.948387. Fold AP: 0.932719, 0.959274, 0.975610,
  0.942361, 0.930684.
- Confirmed-label Brier score: 0.037542.
- Positive-family OOF accuracy: 0.943548.
- Behavior macro AP proxy at the model's 4% sparse cutoff on the enriched
  confirmed set: 0.274939; family-conditional all-row proxy: 0.854226. The
  cutoff proxy is not prevalence-calibrated to evaluation and is retained as a
  diagnostic, not an estimated leaderboard component.
- Rule-based Evidence MAP@5 proxy: 0.135258; directed 0.185608, soft
  0.142064, isolation 0.044493.
- Submission question: can the strong confirmed-only pair ranking transfer to
  the hidden evaluation pairs despite a weak evidence component?
- Gate: if public score is below 0.75, stop submissions, retain E001 as the
  anchor, and diagnose PU/CV/features before any further trial.
- Artifact SHA-256:
  `319f304dd4bc3716a13e4ebf9f543b19fc1409b0a46a9cac004c3000a04407b8`.
- Exact legality receipt: 112,540/112,540 pair coverage; 112,540 unique
  continuous risk values; no null scores; 562,700/562,700 evidence cells are
  exact shared evaluation hands; every pair has five distinct legal hands.
- Kaggle submission ID: `56106882`.
- Submitted/scored: 2026-09-09 04:53 CST.
- Public score: **0.44388**.
- Gate decision: FAIL (`0.44388 < 0.75`). No subsequent candidate may be
  submitted until the PU/CV/feature failure is resolved offline.
- Primary failure: `GENERALIZATION` / confirmed-negative selection shift. The
  0.948 confirmed-only pool OOF does not represent ranking among the broad
  evaluation candidate population. Secondary failure: weak rule-based
  evidence proxy (0.135 MAP@5).

## E002 — family-conditioned evidence ranker (local candidate, not submitted)

- Parent: E001; single intended algorithmic change is evidence ranking.
- Training data: organizer evidence hands from confirmed target pairs only;
  unknown pairs used as negatives: 0.
- Split: five whole-pool folds; all hands from one pair remain in one query and
  one fold.
- Model: separate LightGBM LambdaRank model for each disclosed behavior;
  within-pair percentile features plus gameplay-derived hand features.
- Evidence MAP@5 OOF: 0.398819; directed 0.465815, soft 0.451490,
  isolation 0.215471.
- Coverage policy: five legal shared evaluation hands for every pair, not only
  a top-risk subset. Risk scores remain continuous and unchanged from E001.
- Decision: retain locally until E001 clears the 0.75 public gate.

### E002 retrospective methodology correction — candidate status removed

- A later source-level audit found that the reported 0.398819 OOF MAP@5 is
  oracle-routed. The validation code first filters each held-out pair by its
  true `behavior_family` and then evaluates the ranker for that same family.
  Production code instead chooses a ranker from predicted
  `evidence_behavior`. The reported overall and family scores remain valid only
  as true-family component ceilings, not as an end-to-end evidence estimate.
- Each family also creates a separate `GroupKFold` assignment after filtering
  to that family. Those folds are not the common sealed outer folds used for
  pair risk and behavior routing. In addition, each scored validation fold
  selects its own LightGBM tree count through early stopping, introducing a
  secondary optimistic-selection risk.
- The current work directory contains no E002-specific OOF hand-score receipt
  or JSON result; `model_diagnostics.json` has no `evidence_validation` key.
  Therefore the ledger numbers cannot be independently reproduced from frozen
  predictions without rerunning a new, separately authorized validation.
- E006 remains a public-research control rather than an owned candidate. Its
  notebook does use cross-fitted predicted family on held-out evidence rows and
  reports 0.443468 MAP@5 under a common pair fold, but it bundles a different
  public risk/PU/evidence stack and was not submitted.
- Decision: remove E002's local-candidate status. The owned end-to-end Evidence
  MAP@5 is currently unmeasured; retain 0.398819 only as an oracle-family
  ceiling. Any future evidence validation must share one outer pool fold across
  behavior routing and ranker scoring, choose tree counts only inside outer
  training, persist per-hand OOF scores/top-five receipts, and never use
  unknown labels. No model was rerun, no evaluation row was scored, and no
  submission was created or modified for this audit.
- Method-audit SHA-256:
  `ffd2765a382b2cb07f88d3250a00877561bd9f4dbe18d336933364857c63179c`.
  Audited implementation SHA-256:
  `d5fc22acc542bdd73fef8af5db17b6602ceeb59ac0b96acce23b471898330819`.

## Next preregistered ablations

1. Pair module only: confirmed-negative-only anchor versus explicit bagged PU
   or Elkan–Noto, with unknown targets kept latent. Naive unknown-as-negative is
   a negative control and cannot become the default merely from one score.
2. Pair features, one factor at a time: residual, directionality, temporal
   burst. Require overall Pair AP and worst-fold stability.
3. Behavior: freeze pair ranking, compare sparse cutoff versus prevalence-aware
   PU calibration; report all three family AP values.
4. Evidence: only after the public gate, compare E001 heuristic against E002
   family rankers while requiring 100% exact shared-hand legality.

## Post-E001 stop-loss plan

The next work is offline only and must keep E001 risk/evidence receipts intact.

1. Freeze a sealed set of whole pools before PU tuning. Compare feature
   distributions for confirmed negatives, confirmed positives, eligible
   unknown development pairs, and evaluation pairs. Do not attach class zero to
   the unknown rows.
2. Re-run E001 on the frozen pools with two diagnostics: confirmed-label AP and
   a prevalence-stress ranking metric where unknown targets remain latent.
   Identify whether residual, directionality, temporal, or exposure features
   separate label-selection policy rather than gameplay target behavior.
3. Single-method PU comparison: bagged PU with repeated unknown subsamples and
   confirmed negatives retained as trusted negatives. The unknown rows may
   influence the decision boundary as unlabeled observations but never receive
   ground-truth negative status. Compare against Elkan–Noto in a separate run;
   do not bundle the methods.
4. Keep the E002 evidence improvement (`0.135258 -> 0.398819` OOF MAP@5) local
   until the pair-ranking candidate passes both sealed-pool stability and the
   original `>=0.75` remote gate on an authorized future submission.

## E003 preregistration — bagged PU only

- Parent comparison: identical E001 features, learner class, seeds, and five
  sealed whole-pool folds. The sole method change is adding repeated unlabeled
  subsamples through a biased bagged-PU learner.
- Unknown storage invariant: `label_status=unlabeled`, no `label` column, no
  overlap with confirmed pairs. A temporary zero inside a PU learner is a
  surrogate selection indicator only and is never persisted or described as
  ground truth.
- Candidate population: every unlisted development pair with at least 57
  shared development hands, corresponding to the evaluation minimum of 38
  after the 60/40 phase-duration adjustment. No random pair-selection or ID
  parsing is used.
- Promotion requires all three before looking at results: confirmed Pair AP
  delta >= -0.01; median known-positive percentile versus same-pool unlabeled
  delta >= +0.02; worst-fold confirmed Pair AP delta >= -0.03.
- The positive-versus-unlabeled percentile is a contaminated distribution
  stress diagnostic, not a target metric and not a leaderboard estimate.
- Failure action: archive E003 without submission and move to a different PU
  family. Passing action: add the required early/late time-block stability
  audit before considering any candidate file.

## E003 result — failed gate, not submitted

- Unknown artifact: 135,879 exposure-matched pairs across all 400 pools; no
  confirmed overlap, no `label` column, role exactly `unlabeled`.
- Unknown pair SHA-256:
  `38eeec55e8b28ba3947c6cfb0049918ae9b9be8fde826cf9c07bfc007e6def05`.
- Unknown feature SHA-256:
  `5526883dd7cafc4d588b6c37f2eb18e5c44714bc410b9800c1636d2a0cda690f`.
- Confirmed Pair AP: E001 0.948387; E003 0.954367; delta +0.005981.
- Worst-fold confirmed Pair AP: E001 0.930684; E003 0.932180; delta
  +0.001496.
- Confirmed Brier: E001 0.037542; E003 0.036450.
- Same-pool known-positive versus unlabeled percentile mean: E001 0.978122;
  E003 0.980834; delta +0.002712.
- Median percentile: 1.0 for both; preregistered delta 0.0 versus required
  +0.02. The statistic was saturated and cannot be repaired retroactively.
- Positive-above-unlabeled-P95 rate: 0.897849 to 0.911290 (+0.013441).
- Positive-above-unlabeled-P99 rate: 0.793011 to 0.819892 (+0.026882).
- Decision: FAIL because the predeclared median-percentile gate failed. E003 is
  archived without a candidate submission even though several secondary
  diagnostics improved. The next gate must use unsaturated mean/tail rates and
  be declared before the next method runs.
- Labelled OOF SHA-256:
  `0b62fab38c42bffa2f01ca8b18fc9b5de17dc6eec428a6dda1c5cb0dcd4526bb`.
- Unlabeled OOF SHA-256:
  `3ff22a5ae1edc3a7d2e31f149494b5b1f7d8f8510bee13d10bfac579061ce405`.

## E004 preregistration — pool-relative domain normalization

- Failure evidence: confirmed-negative versus unlabeled feature distributions
  are close, while both shift against evaluation. `shared_hands_calc` has KS
  0.316 because development has 1.2M hands and evaluation 0.8M; max, cumulative
  flow, and some time-bin features also shift with phase length.
- Parent comparison: same 95 E001 gameplay features, HistGradientBoosting
  learner, seeds, confirmed labels, and sealed whole-pool folds. The sole change
  is representing every feature by its percentile among candidate pairs from
  the same pool and phase.
- Unknown is used only as the unlabeled development reference distribution for
  percentile calculation. It receives no surrogate or ground-truth class in
  E004. Evaluation percentiles use evaluation gameplay features only.
- Promotion requires, before results: confirmed Pair AP delta >= -0.015;
  worst-fold AP delta >= -0.03; unlabeled-to-evaluation score-distribution KS
  reduced by at least 30%; known-positive-above-unlabeled-P95 rate delta >=
  -0.02; absolute evaluation score correlation with shared-hand count <= 0.10.
- Passing still does not authorize submission: it first triggers early/late
  time-block stability and behavior/evidence reconstruction. Failure archives
  the representation and resumes a different PU family.

## E004 result — failed gate, not submitted

- Confirmed Pair AP: raw 0.948387; pool-rank 0.942667; delta -0.005720.
- Worst-fold Pair AP: raw 0.930684; pool-rank 0.920715; delta -0.009970.
- Known-positive-above-unlabeled-P95 rate: 0.897849 to 0.900538; delta
  +0.002688.
- Unlabeled-to-evaluation score KS: raw 0.061287; pool-rank 0.082313. This is
  a 34.31% deterioration rather than the required 30% reduction.
- Evaluation score/shared-hands Spearman: raw -0.084755; pool-rank 0.102147,
  just beyond the declared absolute 0.10 limit.
- Decision: FAIL. Marginal feature drift did not propagate to harmful raw score
  drift; full pool-relative ranking discarded useful absolute gameplay
  information. Archive this representation family without submission.
- Labelled OOF SHA-256:
  `6db47a6c4672359e8963da4a142c1e3f4d5f92a80d42dc7fa3721c9463a816ef`.
- Unlabeled OOF SHA-256:
  `3f48a6f600f20d235b41f9cba0e757e7c1aff2670d5d0f4dea34ca574b94af60`.
- Evaluation score SHA-256:
  `276067dd1e3f3543ddc3cdba2de9c1c65a03f3a93bafd1697609dbaa930cce29`.

## E005 preregistration — narrow exposure normalization

- This is the last experiment in the exposure-normalization family. Unlike
  E004, it does not rank every feature.
- On the frozen E001 drift table, remove only `__max` and time-bin fields whose
  unlabeled/evaluation KS is at least 0.075; keep all other E001 features.
- Convert development `shared_hands_calc` to the evaluation-duration scale by
  multiplying by 2/3. Convert `dominant_flow_bb` in both phases to flow per 100
  shared hands. Other values, learner, labels, seeds, and pool folds are
  unchanged.
- Gate is frozen before model output and matches E004: confirmed AP delta >=
  -0.015; worst-fold AP delta >= -0.03; score KS reduction >=30%; P95
  known-positive separation delta >= -0.02; absolute score/shared-hands
  Spearman <=0.10.
- Failure stops the exposure-normalization family. Passing triggers time-block
  stability but still does not itself authorize a Kaggle submission.

## E005 result — failed gate; exposure-normalization family stopped

- Frozen selector removed 32 sample-sensitive max/time-bin fields, leaving 63
  features; shared-hands was scaled 2/3 in development and dominant flow became
  per-100-shared-hands.
- Confirmed Pair AP: raw 0.948387; E005 0.951019; delta +0.002633.
- Worst-fold Pair AP delta: +0.002232.
- P95 known-positive separation delta: +0.008065.
- Score/shared-hands Spearman: 0.082221, within the 0.10 limit.
- Unlabeled-to-evaluation score KS: 0.061287 to 0.168920, a 175.62%
  deterioration instead of the required 30% reduction.
- Decision: FAIL. The primary domain-stability gate failed catastrophically.
  With E004 and E005 both failing, stop this representation family and do not
  tune another exposure/rank normalization variant.
- Labelled OOF SHA-256:
  `98145f81d9d84b9ce9a50def9c0d0f9a9a7d5f1840a89ec93a8e7eabfe0622f9`.
- Unlabeled OOF SHA-256:
  `bf58bc00ac6d7727cc9de19e843823f8594a13656829ef87de59a25611214289`.
- Evaluation score SHA-256:
  `02eddb710a59c5871a4dee2622fbb1b9f8f53588506fb72cf5afa6bd0d40a4da`.

## E006 preregistration — public PU notebook reproduction

- Purpose: establish a stronger legal reference anchor and isolate whether
  E001's missing partner/outsider action-response features explain the remote
  gap. This is a reproduction, not an owned-method claim and not yet a
  submission candidate.
- Source: retained untouched notebook
  `nomannic19/poker-collusion-pu-aware-evidence-ranker`; execute only its data
  preparation, modelling, validation, and submission cells through a local
  path adapter. Record source hash, cells, runtime artifacts, and all deltas.
- Compliance checks after execution: unknown rows retain null ground-truth
  labels and explicit PU role; ID fields are absent from feature columns; every
  evidence hand is an exact shared evaluation hand; continuous risk coverage
  is complete.
- Metrics to record without retuning: confirmed whole-pool Pair AP, PU stress
  AP (explicitly not truth), confirmed/PU behavior MAP, positive-family
  accuracy, evidence OOF MAP@5, coverage, and hashes.
- No-submit rule: reproduction output remains local regardless of its proxy
  score. It may inform a later single-feature-family owned ablation only after
  source/license attribution and time-block leakage checks.

## E006 result — reproduced locally, not submitted

- Source notebook SHA-256:
  `58acf6069c61016d55583c1dcfc88c97aa936cd51623830257dd55a1dbbf67f0`.
- Executed original source cells: 0, 2, 3, 29-32, 34-37, 39-40, with only
  three local path substitutions.
- Confirmed whole-pool Pair AP: 0.934033; fold range 0.9107-0.9497.
- PU stress Pair AP: 0.493722. This is a contaminated proxy, not truth.
- Confirmed Behavior MAP: 0.831020; PU stress Behavior MAP: 0.452244;
  positive-family accuracy: 0.967742.
- Evidence OOF MAP@5: 0.443468; directed 0.4505, soft 0.4814,
  isolation 0.3777.
- Submission coverage: 112,540 pairs and 562,700/562,700 exact shared
  evidence hands; identifier features: 0.
- Unknown policy: 24,000 rows retain null ground-truth label; the notebook's
  0.35 value is recorded only as biased-PU background weight.
- Local output SHA-256:
  `33bbd53b772367be591a84f5e7d68dc6a333743d38efb5f906f2051b5801955d`.
- Decision: keep as a legal reference anchor. Do not submit it and do not claim
  its code or result as an owned method.

## E007 preregistration — partner/outsider interaction features only

- Parent: E001 raw 95 features and confirmed-only learner. Add only public-game
  action-response aggregates reimplemented from the E006 prepared tables:
  true heads-up actions, partner fold/call/raise responses, outsider
  fold/call/raise responses, HU passivity/aggression, and response contrasts.
- Aggregate each interaction by mean, P95, top-five mean, plus five normalized
  response ratios. Do not add E006's XGBoost model, PU weights, or evidence
  model in this experiment.
- Same confirmed labels, HistGradientBoosting parameters, seeds, and sealed
  whole-pool folds. Unknown remains unlabeled and is used only for stress
  diagnostics.
- Promotion requires: confirmed AP delta >= -0.005; worst-fold AP delta >=
  -0.02; P95 known-positive separation delta >= +0.01; score-domain KS may
  worsen by at most 0.02 absolute; absolute shared-hands Spearman <=0.10.
- Passing triggers early/late time-block stability before any PU combination or
  candidate file. Failure archives this feature family without submission.

## E007 result — passed local gate; packaging repaired

- Added interaction features: 41. No model, PU, behavior, or evidence change.
- Confirmed Pair AP: 0.948387 to 0.958898; delta +0.010512.
- Worst-fold Pair AP delta: +0.016993; interaction fold range
  0.947677-0.985548.
- P95 known-positive versus same-pool unlabeled rate: 0.913978 to 0.935484;
  delta +0.021505.
- Unlabeled/evaluation score KS: 0.065096 to 0.075742; absolute deterioration
  +0.010646, inside the predeclared +0.02 limit.
- Evaluation score/shared-hands Spearman: -0.096058, inside the absolute 0.10
  limit.
- Gate decision: PASS. This authorizes only the preregistered early/late
  time-block audit, not a submission or PU combination.
- A receipt-path collision was detected immediately after training: the
  evaluation score receipt had overwritten the evaluation interaction feature
  artifact. The score result was preserved under a new explicit path, the
  feature aggregate was rebuilt from the frozen E006 hand features, and all
  paths/hashes in the JSON receipt were corrected before promotion.
- Development interaction SHA-256:
  `9b3f718d0520352013c4698d69437563e9c708377fa6b8809433a77168e8b12c`.
- Rebuilt evaluation interaction SHA-256:
  `d0738a1e920d1775a6f24556e00073545015941ac9128afdee387a8522fc3df0`.
- Evaluation score SHA-256:
  `12aa0f5bf57e4b3a41bc5f0af10f10c82adfaa1ab55aceb0c8ab14ff80c53a33`.
- Labelled OOF SHA-256:
  `4c235116348d727980c76ab28cd15b22fc8554b39a6ffdfb67853cb11efe484d`.
- Unlabeled OOF SHA-256:
  `5326c997b331131e4826c297f8b01369a1f07e13daaae9a1e941d0f4a8f5497f`.

## E007b preregistration — early/late time stability

- Split every confirmed pair's development shared hands at phase progress 0.5.
  All pool labels remain in the same whole-pool fold; no unknown pair enters a
  class target.
- Exclude phase-progress, quartile-bin, and temporal-range fields so an
  early-trained model cannot identify the half from structural zeros. Keep the
  same E001 learner and compare raw versus raw+E007 interactions.
- Evaluate within-half and, critically, early-train to late-test and late-train
  to early-test OOF predictions on held-out pools.
- Promotion requires both directional interaction deltas >= -0.005; worst
  per-fold transfer delta >= -0.03; worst interaction cross-time AP >=0.90;
  directional AP gap <=0.05.
- Failure revokes E007 promotion. Passing permits a separate interaction+PU
  experiment, but still no submission until behavior/evidence and exact
  packaging audits pass.

## E007b result — failed stability gate; E007 promotion revoked

- Stable base fields: 77; interaction fields: 41; all phase-progress,
  time-quartile, and temporal-range fields excluded from this audit.
- Raw early-train to late-test Pair AP: 0.823499; interaction: 0.852369;
  delta +0.028871.
- Raw late-train to early-test Pair AP: 0.825570; interaction: 0.863528;
  delta +0.037958.
- Worst per-fold transfer delta: +0.005628; directional interaction gap:
  0.011158. Interactions improved every aggregate stability comparison.
- Worst interaction cross-time AP: 0.852369, below the preregistered 0.90
  requirement.
- Decision: FAIL. The local signal is real but insufficiently stable under the
  declared absolute time-transfer floor, so E007 cannot become a submission
  candidate. Retain it as evidence for future model design only.
- OOF SHA-256:
  `eb20d9cb9b35c88b704497665ac7c78ed6cffbee7d147ab399a5a132307640eb`.

## E008 preregistration — outcome-conditioned marginal impact

- This starts a different method family from E007. The sole added feature is
  one theory-motivated, family-agnostic marginal-impact scalar; none of E007's
  41 partner/outsider count aggregates enter the model.
- Build directed pressure edges from ordered public actions where `to_call>0`
  and the previous aggressor is known. Collapse repeated responses to one
  aggressor/responder observation per hand, join only the responder's public
  realized net result, and winsorize that result to [-20, 20] big blinds.
- For directed edge A->B, subtract A's leave-B-out mean responder result from
  B's result under A's pressure. Shrink the residual by using denominator
  `edge_count + 10`; sum A->B and B->A for the pair. Missing edges contribute
  zero. IDs are join keys only and label data never enters feature creation.
- Compute full-development, full-evaluation, early-development, and
  late-development versions independently. Early/late uses phase progress 0.5
  and recomputes every edge baseline inside that half; it cannot inherit a
  full-period target statistic.
- Parent comparison: E001's confirmed-only learner, raw features, seeds, and
  sealed whole-pool folds. Unknown rows remain explicitly unlabeled and are
  used only for stress diagnostics.
- Joint promotion requires all of the following: confirmed AP delta >=-0.005;
  worst-fold AP delta >=-0.02; positive-above-unlabeled-P95 rate delta >=0;
  unlabeled/evaluation score KS deterioration <=0.02; absolute evaluation
  score/shared-hands Spearman <=0.10; both early->late and late->early AP
  deltas >=+0.01; worst cross-time fold delta >=-0.02; cross-time directional
  gap <=0.05.
- Failure archives E008 without tuning this scalar. Passing permits only a
  separate PU-method experiment and behavior/evidence reconstruction; it does
  not authorize a Kaggle submission.

## E008 result — failed directional time gate, not submitted

- Added exactly one non-ID feature. Coverage is 1,860/1,860 confirmed pairs,
  24,000/24,000 public-reference unknown pairs, and 112,540/112,540 evaluation
  pairs; 112,531 evaluation values are nonzero. Unknown receipts contain no
  `label` column and retain only `label_status=unlabeled` and
  `training_role=unlabeled`.
- Confirmed Pair AP: 0.948387 to 0.948759; delta +0.000373. Worst-fold AP
  delta: +0.003156.
- Known-positive-above-same-pool-unlabeled-P95 rate: 0.913978 to 0.919355;
  delta +0.005376.
- Unlabeled/evaluation score KS: 0.065096 to 0.064641; delta -0.000454.
  Evaluation score/shared-hands Spearman: -0.083209, inside the 0.10 gate.
- Raw early->late AP 0.823499; marginal-impact AP 0.823214; delta -0.000284,
  failing the required +0.01. Raw late->early AP 0.825570; marginal-impact AP
  0.836116; delta +0.010547. Worst cross-time fold delta was -0.012156 and the
  directional gap was 0.012902, both within their limits.
- Decision: FAIL because one required time direction failed. Archive this
  exact outcome-residual scalar without tuning or submission. The asymmetric
  result is useful evidence that realized-payoff impact is not temporally
  robust enough for promotion.
- All nine artifact hashes were independently recomputed and matched the JSON
  receipt. Response-outcome SHA-256:
  `8dbbcb8f541f1371f7dbee54d08ecfd578c37a036a93a37691aa50bf8eb4215a`.
  Labelled OOF SHA-256:
  `f1a71adc99c4110644d5c86a136a12fbe47a9984775b3c0f940850c9dd821c9b`.
  Unlabeled OOF SHA-256:
  `01c88639036ddbbdb03fd23ce1181f78ab11e1dd945e1bfde3be590f7d775760`.
  Evaluation score SHA-256:
  `545f4c9b371fb5ac5f9fa0b418b4b24dafddb614ff32e19730d4b6accf726893`.
  Time OOF SHA-256:
  `983d967866eae1bf1fdfc25419d87d4aa190819d3ddac8bea2d0b4a68e3ecffa`.

## E009 preregistration — sequential conditional-information influence

- New method family based on Bonjour, Aggarwal, and Bhargava, UAI 2022,
  <https://proceedings.mlr.press/v180/bonjour22a.html>. The paper is only a
  proof of concept on three-player synthetic RPS/Leduc; its threshold and
  transfer to this benchmark are explicitly unproven.
- Build only same-street adjacent ordered-action transitions. The influencer
  action must precede the target action. Collapse actions into fold, check,
  call, small aggression, and large aggression; condition on the target's
  action-preceding public state: street, active-player bucket (2, 3-4, 5-6),
  and to-call/big-blind bucket (0, (0,1], (1,4], >4). No result, card, ID
  format, row order, label, or future-action field enters the state.
- Estimate directed empirical conditional mutual information from transition
  counts. Multiply each directed estimate by `n/(n+50)` to shrink sparse
  edges. Following the paper, directed net influence A->B is A's influence on
  B minus the largest other-player influence on B. The sole pair feature is
  `min(net_A_to_B, net_B_to_A)`, because both directions must be positive in
  the paper's criterion.
- Compute full-development, full-evaluation, early-development, and
  late-development estimates independently. Early/late recomputes counts,
  shrinkage, and outsider maxima within its own half. Unknown pairs remain
  target-latent and are used only for stress diagnostics.
- Parent comparison: raw E001 features, confirmed-only HistGradientBoosting,
  identical seeds, and sealed whole-pool folds. Do not add E007 counts, E008
  outcomes, PU weights, behavior changes, or evidence changes.
- Before promotion the feature's absolute evaluation shared-hands Spearman
  must be <=0.20. Joint model gate: confirmed AP delta >=+0.003; worst-fold AP
  delta >=-0.02; positive-above-unlabeled-P95 delta >=+0.005;
  unlabeled/evaluation score-KS deterioration <=0.02; absolute evaluation
  score/shared-hands Spearman <=0.10; both early->late and late->early AP
  deltas >=+0.01; worst cross-time fold delta >=-0.02; worst CMI cross-time AP
  >=0.84; directional gap <=0.05.
- Failure archives this exact empirical-CMI feature without submission.
  Passing permits behavior/evidence reconstruction and a separate PU
  experiment, but still does not authorize Kaggle submission.

## E009 result — failed full and time gates, not submitted

- Added exactly one feature. All 1,860 confirmed, 24,000 public-reference
  unknown, and 112,540 evaluation pairs have nonzero values. Feature/shared
  hands Spearman is 0.164300, inside the preregistered 0.20 guard.
- Confirmed Pair AP: 0.948387 to 0.947576; delta -0.000811 versus the required
  +0.003. Worst-fold AP delta: -0.000447.
- Known-positive-above-unlabeled-P95 rate improved +0.005376. Score KS improved
  0.065096 to 0.061749, and evaluation score/shared-hands Spearman is
  -0.083633. These secondary passes cannot override the primary AP failures.
- Raw early->late AP 0.823499; influence AP 0.819826; delta -0.003672. Raw
  late->early AP 0.825570; influence AP 0.827393; delta +0.001823. Both miss
  the required +0.01, and worst influence transfer 0.819826 misses the 0.84
  floor. Worst fold delta -0.010907 and directional gap 0.007566 passed.
- Decision: FAIL. Archive the exact adjacent-action empirical-CMI estimator;
  do not tune state bins, `n/(n+50)` shrinkage, or direction aggregation. Move
  away from the influence family to a different learning objective.
- Unknown receipt has no `label` column and retains only the explicit
  `unlabeled` status/role. All nine referenced artifacts were independently
  rehashed and matched the JSON receipt.
- Transition SHA-256:
  `af5cc705cb99cf5ae3b9b187f7b5e407d3d7b3f7e05c12697b6ad1f33e03b8be`.
  Labelled OOF SHA-256:
  `da8353c9b28915dba6ac6ad33f2d4c722439da8089925480090b828ec1801148`.
  Unlabeled OOF SHA-256:
  `4e1d9d8cabc50578e1ea141f17fbf0d2995eead06794dfeb6a07ed1e421c7855`.
  Evaluation score SHA-256:
  `7b33bc0949164fab7df68e67173b4964457cc31ab5c93d73f58fa80044a126dd`.
  Time OOF SHA-256:
  `0efd830ca0fa0530d9d972b102a2d6e9b1b27e090780082185eb65b4e334e379`.

## E010 preregistration — confirmed-only global pairwise objective

- Pool-query feasibility audit: 397 pools contain confirmed labels, but 152
  are all-negative, five are all-positive, and only 240 contain both classes.
  A separate LambdaRank query per pool would discard the ranking gradient from
  a large fraction of trusted supervision, so E010 uses one global confirmed
  query while retaining whole-pool held-out validation boundaries.
- Sole change from E001 is the learning objective: replace
  HistGradientBoosting classification with deterministic LightGBM LambdaRank
  on all confirmed target/non-target pairs. Raw E001 gameplay features stay
  fixed; there are no E007/E008/E009 features, PU weights, behavior changes,
  evidence changes, ID values, or row-order features.
- Frozen ranker: `objective=lambdarank`, 280 trees, learning rate 0.035, 15
  leaves, minimum child samples 12, L2 3.0, feature fraction 0.9, deterministic
  column-wise training. Convert raw margins with a fixed logistic sigmoid only;
  do not fit a calibration model.
- Unknown pairs remain target-latent and are predicted only for stress
  diagnostics. E001 raw receipts supply the unchanged baseline on identical
  pairs/folds; no baseline model is retrained.
- Joint promotion gate: confirmed AP delta >=+0.003; worst-fold AP delta
  >=-0.02; positive-above-unlabeled-P95 delta >=+0.005;
  unlabeled/evaluation score-KS deterioration <=0.02; absolute evaluation
  score/shared-hands Spearman <=0.10; early->late and late->early AP deltas
  each >=+0.01; worst cross-time fold delta >=-0.02; worst pairwise
  cross-time AP >=0.84; directional gap <=0.05.
- Failure archives the global pairwise objective without tuning tree or sigmoid
  parameters. Passing permits behavior/evidence reconstruction and a separate
  PU experiment, but still does not authorize a Kaggle submission.

## E010 result — catastrophic gate failure, not submitted

- Confirmed Pair AP fell from 0.948387 to 0.699491; delta -0.248896. The
  pairwise fold range was 0.741291-0.793729 and worst-fold delta was -0.189394.
- Known-positive-above-unlabeled-P95 rate fell from 0.913978 to 0.594086;
  delta -0.319892. Unlabeled/evaluation score KS rose from 0.065096 to
  0.524167 (+0.459071), and score/shared-hands Spearman was -0.120867.
- Early->late AP fell from 0.823499 to 0.642373; late->early fell from
  0.825570 to 0.571096. Worst time-fold delta was -0.273549 and directional
  gap 0.071276.
- Decision: FAIL on every primary quality/domain/time gate. A single global
  LambdaRank query destroys the calibrated classification structure needed by
  this benchmark. Stop the pairwise-objective family; do not tune query,
  tree, or sigmoid parameters and do not submit.
- Unknown receipt contains no `label` column and retains only explicit
  `unlabeled` status/role. All four receipt hashes were independently
  recomputed and matched the JSON receipt.
- Labelled OOF SHA-256:
  `808ffe1ed3333fd3c52d266522eb2f0ab7f708f0ec0a82124280a40b6bc1078d`.
  Unlabeled OOF SHA-256:
  `0de085cb2780bf0999825502f57a27cd859c97096669c637c6e609b78ed30300`.
  Evaluation score SHA-256:
  `a85550761756f233c65c706e39d31ab752e583f8146e1cdde3591bc9e6419006`.
  Time OOF SHA-256:
  `bbe1ecb2bc94cbae0b61ab37c235f10f57c69410fda8eafdfbfe3ae6a7eb8a77`.

## E011 preregistration — nested trusted hard-negative weighting

- New learning method family; parent is E001 with the same 95 raw gameplay
  features, confirmed-only HistGradientBoosting parameters, seeds, and sealed
  whole-pool outer folds. Do not add PU rows, E007-E009 features, behavior, or
  evidence changes.
- Within each outer fold, fit a pilot E001 model using only that fold's
  confirmed training pools. Rank only its confirmed non-target training rows
  by in-sample pilot probability; the highest quartile is the fixed hard set.
  No global OOF score is allowed for mining because it could depend on the
  outer validation pools.
- Fit the final model on the same confirmed training rows with weights:
  hard confirmed negative 2.5, easy confirmed negative 0.5, confirmed positive
  1.0. This approximately preserves total negative-class weight while moving
  emphasis from easy to hard negatives. Unknown rows never enter pilot,
  threshold, weighting, or final fit; they remain latent stress diagnostics.
- Stage A promotion requires confirmed AP delta >=+0.003; worst-fold delta
  >=-0.02; positive-above-unlabeled-P95 delta >=+0.005;
  unlabeled/evaluation score-KS deterioration <=0.02; absolute evaluation
  score/shared-hands Spearman <=0.10. Failure stops E011 before time fitting.
- Only after Stage A passes, recompute pilot hardness independently inside
  early and late outer-training halves. Final promotion additionally requires
  early->late and late->early AP deltas each >=+0.01; worst time-fold delta
  >=-0.02; worst hard-negative cross-time AP >=0.84; directional gap <=0.05.
- Passing both stages permits behavior/evidence reconstruction but does not
  authorize submission. Any Stage A failure archives this exact weighting
  scheme without tuning quartile or weights.

## E011 result — Stage A failed; time gate not run; not submitted

- Each outer fold selected 298 hard confirmed negatives, 25.02%-25.04% of
  its confirmed-negative training rows. Negative total weights were
  1,191.0-1,191.5, matching the original negative counts up to the quantile
  boundary; unknown rows never entered mining or fitting.
- Confirmed Pair AP: 0.948387 to 0.945789; delta -0.002598 versus required
  +0.003. Worst-fold AP delta: -0.008371.
- Known-positive-above-unlabeled-P95 rate fell 0.913978 to 0.908602; delta
  -0.005376 versus required +0.005. Score KS worsened 0.065096 to 0.081627
  (+0.016531, inside its separate limit), while score/shared-hands Spearman
  was -0.117752 and failed the absolute 0.10 guard.
- Decision: FAIL Stage A. Per preregistration, no early/late models were run;
  JSON status is `not_run_stage_a_failed`. Do not tune the hard quartile or
  weights and do not submit.
- Unknown receipt contains no `label` column and retains explicit
  `unlabeled` status/role. All three receipt hashes were independently
  recomputed and matched the JSON receipt.
- Labelled OOF SHA-256:
  `28b89639e9f74feac84d154b0b8629252399c451046b136905dfa64bd1db45c8`.
  Unlabeled OOF SHA-256:
  `ff4185467ebb10d094bd1c97ee04ddef1b3adbf6a8ea1e77f0ed4ffb02dbddd1`.
  Evaluation score SHA-256:
  `621d7d44fc66334fa956a2b15c7aff1c10c498a5c96c30c8a49b6ecbe4ccab17`.

## E012 preregistration — Elkan–Noto positive-unlabeled selector

- New PU estimator, distinct from E003 bagging. Raw E001 gameplay features and
  sealed whole-pool outer folds stay fixed. Unknown target labels remain null;
  no unknown row is stored or described as a ground-truth negative.
- The temporary selector indicator is `s=1` only for confirmed targets and
  `s=0` for confirmed non-targets plus target-latent unknown rows. In each
  selector fit, confirmed rows have weight 1.0 and all unknown weights sum to
  the number of confirmed-negative rows, preventing the 24,000 unknown sample
  from overwhelming trusted supervision. The indicator is never persisted.
- Inside each outer training fold, create a deterministic three-way
  stratified whole-pool split. Use its first held-out partition only to
  estimate `c=P(s=1|y=1)` from confirmed-positive predictions of a selector
  trained on the other two partitions and their unknown pools. Clip c to
  [0.05, 0.95]. Refit on every outer-training confirmed/unknown row and emit
  `clip(P(s=1|x)/c, 0, 1)` for outer validation, held-out unknown, and
  evaluation. The outer validation pools never enter c or selector fitting.
- Stage A promotion requires confirmed AP delta >=-0.01; worst-fold delta
  >=-0.03; positive-above-unlabeled-P95 delta >=+0.01; P99 delta >=0;
  unlabeled/evaluation score-KS deterioration <=0.02; absolute evaluation
  score/shared-hands Spearman <=0.10.
- Stage A failure archives E012 without time fitting, parameter changes, or
  submission. Passing requires building independent early/late unknown
  features and then both time-transfer deltas >=0, worst time-fold delta
  >=-0.03, worst transfer AP >=0.82, and directional gap <=0.05 before any
  behavior/evidence reconstruction. Passing never itself authorizes a Kaggle
  submission.

## E012 execution status — suspended before results

- A cross-experiment supervisory stop line arrived after preregistration but
  before any E012 code, model, score, or receipt was produced. E007b and
  E008-E011 constitute five consecutive feature/objective/weight failures, so
  running another PU estimator without a residual mechanism would violate the
  required switch-and-diagnose discipline.
- Decision: keep the preregistration as an immutable proposed design, but do
  not execute it. This is not a failed numerical experiment and consumes no
  candidate slot or submission.

### E012 supervisory reactivation before execution

- After the residual audit, the supervisory channel allowed E012 as the one
  cooling-off exception because Elkan–Noto was already named in the E001
  stop-loss plan and E003 supplied positive P95/P99 evidence for a distinct PU
  estimator. This authorization arrived before any E012 result.
- The original Stage A and time gates remain unchanged. Any Stage A failure
  ends E012 without time fitting. Regardless of outcome, no E013 or Kaggle
  submission may be created automatically in this run.

## E007-E011 cross-failure residual audit

- Authoritative report: `poker/work/E007_E011_failure_audit.json`. It compares
  all 553,536 confirmed positive-negative inversions, family AP, 240 mixed-pool
  AP values, score-delta correlations, and both time directions.
- Raw family AP exposes the stable bottleneck: coordinated isolation 0.716667,
  directed transfer 0.919071, soft play 0.977378.
- E007 is the only candidate that improves full Pair AP and both time
  directions. Isolation family AP rises to 0.804722 full-period, 0.572214
  early->late from 0.462772, and 0.497086 late->early from 0.322589. Overall
  time deltas are +0.028871 and +0.037958.
- E007 fixes 4,553 raw inversions while creating 2,964, a net removal of 1,589;
  123 positives improve versus 70 that worsen. Across mixed pools it wins nine,
  loses four, and ties 227. Its static gain is therefore localized but real.
- E008 outcome impact removes only 320 net inversions and is directionally
  unstable. E009 conditional influence creates 31 net inversions. E010 creates
  80,303 net inversions, and E011 creates 1,712. These failures do not support
  further tuning of their families.
- Score-delta Spearman between E007 and E008/E009 is only 0.1708/0.0954, so the
  methods are not merely numerical copies; nevertheless neither independent
  mechanism repairs isolation across time.
- Evidence-backed next proposal: a behavior-conditioned risk target that
  keeps directed/soft heads on raw E001 fields and routes E007 interactions
  only to an isolation head. This changes the target decomposition rather than
  tuning a frozen feature, PU, pairwise, or weighting parameter. It must be
  reviewed and preregistered before execution.
- Hard-positive residual SHA-256:
  `14b77dc3376cdead05e3e2af31515e0281301b91f58c26c810815555180b2099`.
  Score-delta correlation SHA-256:
  `a2934b11be382cb6d8c77f8074b348a6a33158f72d15b51d6e193afb1d5096b6`.

## E012 result — numerical Stage A pass, method boundary failure

- The frozen numerical Stage A comparisons all passed: confirmed Pair AP rose
  from 0.948387 to 0.958382 (+0.009996); worst-fold delta was +0.002138;
  positive-above-unlabeled P95/P99 rates rose by +0.016129/+0.064516; score KS
  improved by 0.029483; and evaluation score/shared-hands Spearman was
  -0.033481. These figures are retained as measurements, not as promotion
  evidence.
- Post-result method audit found a calibration mismatch. Each `c` came from a
  selector trained on two inner partitions, while outer, unknown, and
  evaluation scores came from a different selector refit on all outer-training
  rows. Thus the estimated `c` did not calibrate the `g` used for final scores.
- The 24,000 unknown rows were also case-control reweighted so their total
  weight equaled the confirmed-negative count. No organizer sampling
  probability or inverse-probability correction is known, so this changes the
  selector base rate and cannot be presented as standard Elkan-Noto
  probability recovery. The SCAR assumption for confirmed positives is also
  unverified under the confirmation policy.
- Saturation independently fails the revised validity guard. The five outer
  validation clip-to-one rates were 10.78%-15.55%, all above 0.5%. Overall,
  251/1,860 labelled OOF scores were exactly one and only 1,610/1,860
  (86.56%) were unique, below the 99.5% requirement. Unknown and evaluation
  unique-score fractions were 99.77% and 99.64%, respectively.
- Division by a positive fold constant preserves ranking until clipping, and
  clipping only creates ties. Therefore the observed AP signal is attributable
  to the unknown-aware selector construction, not identified as an
  Elkan-Noto correction effect. E012 may be retained only as a biased PU
  sensitivity probe / development signal.
- Decision: `PROVISIONAL_METHOD_MISMATCH` and `STOP_BEFORE_TIME`. The time gate
  was not run, the unexecuted time-feature builder was removed, no behavior or
  evidence reconstruction was performed, and there was no Kaggle submission.
  Do not repair or rerun the same E012 after seeing these results; any future
  faithful design requires a new experiment ID and a preregistered same-model
  three-way `g_k/c_k` rotation or explicitly justified case-control correction.
  Per supervisory stop, do not create E013 automatically; return to E001 as the
  sole remote anchor and end this run.
- Unknown ground-truth labels assigned: zero. The temporary selection indicator
  was not persisted. Independent row/schema/hash checks remain valid.
- Raw numerical report SHA-256:
  `cac16b65b0f3aa19b3ae6768448c2653f00439c0c441e87f6c1002f384c5f9d4`.
  Independent method-audit SHA-256:
  `d41d45256bacea4be9c78ebfebbe6582034c904b1882f22f11f6d126b8667ba5`.
- Labelled OOF SHA-256:
  `e546d880a35fa524adf579083721794567d82b8a4b2b4912acdc7e0987a8f0be`.
  Unlabeled OOF SHA-256:
  `06e2d94b41fff29e7e610350fa78a5432cfbd122efe3a5d112aa8eeac1d0cbf5`.
  Evaluation score SHA-256:
  `6a588ba69ea1a65c238fdd9d6aaf07e858e91cf4854adc26cd647952fa8abbd9`.
- Primary method references: Elkan and Noto,
  `https://cseweb.ucsd.edu/~elkan/posonly.pdf`; Bekker and Davis,
  `https://proceedings.mlr.press/v94/bekker18a.html`.

### E012 source-metadata correction

- The PMLR 94 URL above is Jessa Bekker and Jesse Davis, “Learning from
  Positive and Unlabeled Data under the Selected At Random Assumption,” not
  the previously recorded title/authorship. Only citation metadata changed.
  All E012 measurements, validity failures, `STOP_BEFORE_TIME` decision, and
  no-submission controls remain unchanged. The corrected method-audit hash is
  the value recorded above.

## B001 preregistration — exact behavior baseline audit

- This is a behavior-validation audit, not E013, a pair-risk candidate, or a
  submission authorization. Pair risk is frozen to the E001 raw OOF scores in
  the E007 receipt (SHA-256
  `4c235116348d727980c76ab28cd15b22fc8554b39a6ffdfb67853cb11efe484d`).
  Evidence is unchanged and evaluation rows are not scored in this audit.
- Training uses only confirmed-target pairs for the three-class family model.
  Confirmed non-targets receive no family training label; unknown pairs are
  absent from training, validation, thresholds, and metrics. Unknown
  ground-truth labels assigned: zero.
- Reuse the exact five sealed whole-pool folds stored in the E007 receipt.
  Within each fold, fit the unchanged E001 HistGradientBoosting learner on the
  other folds' confirmed targets using the same 95 gameplay-only feature
  columns. Predict family probabilities for every held-out confirmed row;
  family supervision for that row never enters its model.
- Primary outputs are positive-family accuracy, macro recall, per-family
  recall, confusion counts, multiclass log loss and Brier score, plus the
  official-format family AP values and macro Behavior MAP at fixed active-risk
  fractions 0.5%, 1%, 2%, 4%, 5%, 10%, 20%, and 100%. The grid is descriptive:
  no rate is selected, promoted, or used to fit a new candidate after results.
- The official scorer assigns exactly zero to every pair not predicted as a
  given family, then resolves that zero-score tie by lexicographic `pair_id`
  order. Because identifier order is prohibited as a modelling signal, B001
  must report three views for every rate/family: exact official-order AP,
  deterministic random-tie expected AP (512 permutations, seed 9917), and the
  analytical best/worst zero-tie bounds. Only the tie-randomized result may be
  used as a model-quality diagnostic; official-order values are reproduction
  checks, never selection evidence.
- Cross-time audit uses the existing 77 stable early/late raw fields and the
  exact E007 time-fold receipt. Fit family models on confirmed targets from the
  early training halves and predict late held-out pools, and conversely late
  to early. Report positive-family accuracy, macro recall, per-family recall,
  confusion counts, and tie-randomized Behavior MAP at the fixed 4% and 100%
  active fractions. No interaction feature enters B001.
- Required integrity checks: every pool represented among the 1,860 confirmed
  rows maps to exactly one fold, and its observed pool count is recorded;
  feature lists contain no pair/player/hand/table identifier, ID-derived value,
  row order, label, or behavior field; all source hashes and output hashes are
  recorded; risk scores match the frozen receipts exactly. B001 cannot trigger
  E013, time-risk fitting, behavior/evidence reconstruction, or submission.

## B001 result — behavior baseline measured; diagnostic only

- Independent receipt audit passed: 1,860 unique confirmed pairs, exact E001
  risk-score equality, one fold per observed pool, 95 full-period and 77
  time-stable gameplay-only fields, no unknown rows or unknown labels, and no
  evaluation scoring. Pair AP remained exactly 0.948387 because risk was
  frozen.
- Confirmed-target family accuracy was 0.943548 and macro recall 0.938447.
  Recall by family was 0.939189 directed transfer, 0.984848 soft play, and
  0.891304 coordinated isolation. Isolation supplied 19 of the 21 total family
  errors and remains the classification bottleneck. Positive-only multiclass
  log loss was 0.163134 and Brier score 0.088087.
- At the frozen 4% risk-active policy, random-tie Behavior MAP was 0.262344
  (directed 0.311515, soft 0.409650, isolation 0.065866). Exact official
  pair-ID order gives 0.274939, while sklearn's grouped-tie implementation gives
  0.232909; this confirms that sparse zero ties materially change the reported
  proxy and that ID-order values must not select models.
- With every confirmed row assigned its predicted family, random-tie Behavior
  MAP was 0.854915: directed 0.891437, soft 0.980111, isolation 0.693198. The
  official-order value was 0.854226 and the analytical zero-tie interval was
  0.852508-0.861947. This is a descriptive policy comparison, not a
  post-result authorization to select 100% activation.
- Early->late family accuracy/macro recall were 0.852151/0.847081;
  late->early were 0.879032/0.869046. Isolation recall fell to 0.804348 and
  0.782609. Random-tie all-active Behavior MAP was 0.689634/0.679294, versus
  0.261408/0.259238 at the fixed 4% policy. The full-period strength therefore
  does not eliminate the time-stability bottleneck.
- Coverage limitation: confirmed labels occur in 397 of the 400 gameplay
  pools. Those 397 all map to exactly one fold, but three label-empty pools
  cannot supply direct behavior validation. Record the audit as
  `diagnostic_only_with_three_unlabelled_pool_coverage_gap`; do not extrapolate
  behavior validation to the three unseen pools.
- Decision: B001 establishes a strict behavior baseline and confirms that
  isolation plus sparse risk activation, not broad family confusion, is the
  dominant behavior weakness. Per supervisory boundary, it does not promote a
  policy, trigger E013, score evaluation rows, alter evidence, or authorize a
  submission.
- Report SHA-256:
  `02e7e03e217cdf403d120e938461482228d1e5cbfc55d24f751f7dae295fecf6`.
  Implementation SHA-256:
  `4cae28dba762ba301bb58d0c75fa5533bd8d5e979a52764732b506e831a87b49`.
  Full OOF receipt SHA-256:
  `71d7ce45170267060f7a1a6ae3e4a78a425d37bc4b4d88d392d8fe08b8baee36`.
  Time OOF receipt SHA-256:
  `4f85a92ab6e72f3c71dd4d14b7f5ba57e621bd87c08a41955576d7d612273147`.

### B001 audit-guard clarification

- Before any cross-time receipt is consumed, the implementation now requires
  exact SHA-256
  `eb20d9cb9b35c88b704497665ac7c78ed6cffbee7d147ab399a5a132307640eb`
  and verifies all 1,860 pair/fold assignments exactly match the frozen
  full-period receipt. The independent check passed.
- Seed 9917 is the deterministic base, not the literal seed for every view.
  Full-period views use `9917 + rate_index * 10 + family_index`; time views add
  a fixed 1000 early->late or 1100 late->early offset before the same
  rate/family offsets. Only guard and metadata records changed; all B001 model
  scores and OOF receipts are unchanged, and no new model was run.

## Public-research refresh — 2026-09-09 16:46 CST

- A live Kaggle API inventory ordered by latest run returned exactly three
  public notebooks for this competition: the official metric, the official
  getting-started notebook, and Nomannic's PU-aware evidence ranker. All three
  already exist locally with matching references and recorded hashes; no new
  public notebook method family was found. Files named Ravaghi/Nihilistic in
  the shared workspace belong to a separate wellbore-geology competition and
  are explicitly excluded from poker research.
- The official tutorial uses gameplay-only feature concepts and legal evidence
  examples, but provides no strict whole-pool or PU validation and no learned
  evidence-ranker improvement claim.
- The public PU notebook remains a useful research control: it assigns zero
  unknown ground-truth labels, excludes identifiers from features, and uses a
  common pair fold with cross-fitted predicted-family routing for held-out
  evidence scoring. Its local reproduction reports confirmed Pair AP 0.934033,
  PU-stress Pair AP 0.493722, Behavior MAP 0.831020, and Evidence MAP@5
  0.443468.
- Those public scores are not an owned candidate. The notebook bundles a new
  XGBoost risk model, biased-PU weighting, and a larger feature set; selects the
  behavior activation rate on the same OOF grid; reports behavior with
  sklearn's grouped-tie AP instead of the organizer's stable-order AP; and
  chooses evidence tree count on the same outer validation rows whose MAP@5 it
  reports. Its local reproduction has no authorized Kaggle submission or
  remote score.
- Decision: the only reusable public pattern is joint outer-fold,
  cross-fitted predicted-family evidence routing. It requires a separately
  reviewed owned validation with inner-only tree selection and frozen per-hand
  receipts. This refresh does not authorize or launch a new experiment, score
  evaluation rows, assign unknown labels, or create/modify a submission.
- Research-inventory SHA-256:
  `9b3adde347c213462ad30f22c72b904c5fafee78ff0d5394b4100289e1920401`.

## EV000 — strict joint evidence-validation feasibility audit

- EV000 is a receipt-only audit, not a model experiment or candidate. It fit no
  model, viewed no evidence-performance score, loaded/scored no evaluation row,
  assigned no unknown ground-truth label, and created or modified no
  submission.
- Frozen inputs passed exact hashes: B001 full OOF
  `71d7ce45170267060f7a1a6ae3e4a78a425d37bc4b4d88d392d8fe08b8baee36`,
  B001 time OOF
  `4f85a92ab6e72f3c71dd4d14b7f5ba57e621bd87c08a41955576d7d612273147`,
  and development pair hands
  `cf197d2bf8b8242b83515b6db1a5a0c73c214ed91fc46fce9205014f07d265af`.
  The full/time receipts have identical pair/fold assignments for all 1,860
  confirmed rows. The B001 router implementation and its full/early/late
  feature receipts are additionally hard-pinned at
  `4cae28dba762ba301bb58d0c75fa5533bd8d5e979a52764732b506e831a87b49`,
  `756d2e1e1b9d031755534ff975023a5f958ab36c53ec342b71c9f03f6448dc8b`,
  `0622e8f2b59cf1fb7b9109c4b9e536d5f5773d0bdee630b361134d9116db471d`,
  and `943fbf666fc19ac2c1fb086e7c32f94df4ead0dcc8b65e85b9feade85cd22e1a`.
- All 372 confirmed targets join one-to-one to the frozen whole-pool folds and
  contain 45,129 legal candidate hands. All 1,817 organizer evidence rows are
  legal pair-hand keys, match the organizer family, have unique ranks 1-5, and
  leave every target pair with three to five evidence hands. Confirmed
  non-target pairs used: zero; unknown pairs used: zero.
- Full-period family coverage is 148 directed-transfer pairs/725 evidence
  hands, 132 soft-play/632, and 92 coordinated-isolation/460. The deterministic
  three-inner-fold construction (`8803 + outer_fold`) leaves every
  outer/inner/family cell non-empty: minimum training/validation pair counts
  are 48/24 for isolation, 78/39 for directed transfer, and 69/34 for soft
  play.
- Static model-input readiness covers only the 45,129 candidate rows belonging
  to the 372 confirmed targets. All 20 frozen preregistration fields are finite,
  with zero missing values and zero globally constant fields. Full-period
  pairwise construction has 372/372 trainable queries and 212,888 preferences;
  late has 264/264 and 42,763. Early has 347 evidence-bearing queries but only
  346 pairwise-trainable queries and 82,266 preferences: one directed-transfer
  query in outer fold 3 has a single candidate which is evidence, hence no
  unjudged comparator. It must remain visible as a zero-preference query and
  may not be silently counted as trainable or discarded.
- Both time directions are structurally feasible. Early->late has minimum
  outer source-training/target-validation coverage of 68/7 pairs; late->early
  has 35/15. Nested time selection is also non-empty: global minima are 44/9
  for early->late and 19/21 for late->early. The decisive weakness is late
  coordinated-isolation validation: only 47 eligible pairs total, with outer
  fold counts 7/11/12/10/7 and evidence counts 17/36/28/24/18. Any future
  validation must retain those cells exactly and report their wide uncertainty.
- An independent reconstruction repeated five outer folds, 15 inner splits,
  30 outer time cells, and 90 nested time cells directly from frozen sources;
  every count matched. Decision: a strict owned validation is feasible, but
  EV000 authorizes no implementation run or model.
- Historical pre-tie feasibility implementation SHA-256:
  `76404aaa300c47138679ffd602a45e94a62fd3f3e02f5f3410ffbb43c74cc306`.
  Historical pre-tie feasibility report SHA-256:
  `7c0a3d2683b1e32d31a6137216a88f0b6e4f4abb73ade465ae5da192206f3824`.
  Preserved historical PASS receipt SHA-256:
  `eeb72462842522fd26882bec89ae19969c165f9594f51e8750d9a6fa6ffdc74b`.

## Nested evidence-only preregistration — draft, not authorized

- Draft `EVP-DRAFT-01` specifies an independent pairwise linear ranker and a
  single matched family-interaction ablation. It does not reuse the public
  notebook's model, PU weights, features, hyperparameters, OOF activation
  selection, or outer-validation tree selection. The only retained validation
  principle is common whole-pool outer folds plus predicted-family routing for
  held-out queries.
- The draft freezes B001 outer folds, three inner whole-pool folds inside every
  outer-training partition, fold-local preprocessing, true-family construction
  only on fitting-side rows, direction-specific B001 predicted-family routing,
  exact random-tie AP@5, immutable per-hand/top-five OOF receipts, paired
  pool-bootstrap uncertainty, and exact family/direction coverage tables.
- To close the E002-style oracle-family loophole, every inner validation fold
  must now receive a B001 family prediction from a behavior model refit on the
  other two inner folds. Inner-valid true family cannot select regularization
  or construct an evidence score. Full/early-to-late/late-to-early nested
  router seed bases are frozen to 18401/19401/20401 plus
  `outer_fold * 10 + inner_fold`; outer validation continues to use the frozen
  B001 full/time OOF receipts.
- Frozen gates cover integrity, full-period improvement, attribution to family
  conditioning, and both time directions. The two seven-pair late-isolation
  outer cells are explicit stop lines: loss of one pair or the frozen 17/18
  evidence counts ends the study. Failure of any gate forbids a rescue variant,
  evaluation scoring, or submission.
- A static readiness amendment separates evidence-bearing from
  pairwise-trainable source coverage. It freezes the single early
  directed-transfer/fold-3 zero-preference query, all 20 feature checks, exact
  preference volumes, and trainable counts before any implementation exists.
- State remains documentation-only. The draft cannot be implemented or run
  until a separate supervisory review explicitly authorizes it, and passing it
  would still require another review before any evaluation action.
- Preserved historical `EVP-DRAFT-01` SHA-256:
  `7a170fd003ff8f07fc4f918250b239832d0c34c76384b82c1379769f8fe10301`.

## EV000 binding-integrity repair — amended draft v2

- The earlier independent PASS became invalid when a later receipt-only tie
  audit deliberately added official-metric and `hands.parquet` hash pins plus
  H0 tie-structure fields to EV000, while the in-place draft was also amended.
  The modified files had mtimes later than the audit and hashes no longer
  matched. No model, evidence-performance metric, evaluation row, or submission
  was involved in the change.
- The original draft was restored byte-for-byte and retained as
  `EVP-DRAFT-01`, SHA-256
  `7a170fd003ff8f07fc4f918250b239832d0c34c76384b82c1379769f8fe10301`.
  The original PASS content is separately preserved at
  `poker/work/EV000_independent_audit_historical_pass.json`, SHA-256
  `eeb72462842522fd26882bec89ae19969c165f9594f51e8750d9a6fa6ffdc74b`.
- The mismatched audit remains retained as
  `PROVISIONAL_HASH_MISMATCH`, SHA-256
  `e5d8293a2629f0639b33e030a0252f3da5d8654b342ce574476e562b362e2fcd`.
  Its historical code/report/draft binding is explicitly invalid and its
  execution lock remains true.
- Replacement `EVP-DRAFT-02` is a separately versioned amendment. It binds the
  current feasibility implementation SHA-256
  `7518ed8731fa128cf66abee99dc95dc639673dfd0c025735ad4bdb028e09a49d`
  and current EV000 report SHA-256
  `27a8245e99419fce163e5d90843cd2585a1273a467d7841433dea732384a4726`.
  Its own SHA-256 is
  `a689703b2f3fc69cc054000f040b43cca80cee6bc4051feb62f8bd5d30810054`.
- The added audit freezes exact random-tie AP@5 mathematically and separates it
  from deterministic top-five materialization. Exhaustive enumeration of 55
  synthetic one/two-block cases matched the closed form. H0 has 4,522 full
  exact tie blocks, 65 crossing rank five; the two time views have 2,580/2,635
  blocks and 70 rank-five crossings each. Development `started_at` resolves all
  observed tied hands; any future timestamp-unresolved score tie is a stop,
  with no ID or row-order fallback. These are structural counts, not MAP scores.
- A complete source-level reconstruction reverified 11 input hashes, 372
  targets, 1,817 legal organizer evidence hands, 45,129 target-hand rows, all
  20 finite features, 30 outer time cells, 90 nested time cells, and the full
  H0 tie structure. Replacement audit status is `PASS_CURRENT_BINDING` at
  `poker/work/EV000_independent_audit_v2.json`, SHA-256
  `85adea3bc23ac6f6c25bd15cd2ce70135a383aeee2530d222a04836e9eea09fc`.
- Repair conclusion: the v2 hash chain is internally valid, but execution
  remains forbidden pending a new supervisory decision. No E-number is
  created, and no validation run, evaluation scoring, or submission is
  authorized.

## EVP pool-provenance companion audit

- This is a read-only companion to the frozen v2 binding; none of the bound
  implementation, report, draft, or audit-v2 files changed. It loaded no
  evaluation rows, fit no model, and viewed no performance metric.
- All 45,129 confirmed-target candidate rows match across five independent
  pool fields: pair-hand `table_id`, both player-side table fields, source-hand
  `table_id`, and B001 receipt pool. Every candidate is development phase,
  every one of the 372 queries belongs to exactly one pool and outer fold, and
  all 1,817 organizer evidence rows match that same query pool.
- The 372 positive queries occupy 245 pools, split 47/50/52/45/51 across outer
  folds. Pool family multiplicity is 183 one-family, 54 two-family, and eight
  three-family pools. Within-family multi-query pools number 25 directed, 17
  soft, and 11 isolation; independent query or hand bootstrap would therefore
  break real dependence.
- Decision: `table_id` is the mandatory uncertainty-resampling cluster. This
  receipt validates the v2 pool boundary but does not amend its method or
  authorize execution. Companion receipt SHA-256:
  `eb11f4528c406d04b577aa254a3537cbe92cb080bf8e65a266153d95ab0b4545`.

## EVP bootstrap-support companion audit

- This read-only audit tested the literal v2 wording: within each view and
  outer fold, resample the observed eligible `table_id` pools with replacement.
  It computed label-support probabilities only; no model or performance metric
  was run and the v2 binding remains byte-identical.
- Ordinary resampling can yield undefined family/macro cells. In early->late
  outer fold 0, the seven coordinated-isolation queries occupy six of 38 view
  pools, so a draw has empty-isolation probability 0.00145855, or 7.293 expected
  empty replicates in 5,000. Fold 4 has seven isolation queries/pools among 41,
  giving 2.320 expected empty replicates. Other views are much less exposed but
  the probability remains nonzero.
- v2 does not preregister whether to reject, omit, zero-fill, or otherwise
  handle an empty cell. Any of those choices made after scores are visible
  would invalidate the interval. Independent hand/query bootstrap and separate
  family bootstrap are also prohibited because the pool-provenance audit shows
  shared queries and families inside pools.
- Decision: status `REVIEW_REQUIRED_BOOTSTRAP_AMBIGUITY`. Do not implement or
  execute v2 until supervision chooses a separately versioned rule. The
  recommended review option is 5,000 paired Bayesian cluster-bootstrap draws:
  positive Exp(1) weights per eligible `table_id` within outer fold, inherited
  by all its queries and shared across H0/L0/L1/families, seed 12673. This is a
  recommendation only, not an amendment or authorization.
- Companion receipt SHA-256:
  `71b02ce03d923346a62ebb01e34ad606aa039c96a3d7ce53a4c26c0098e9092a`.

## EVP v2 methodology withdrawal and v3 draft

- Supervisory review withdrew v2's methodology-execution status while
  preserving its exact file and its historical hash-binding PASS. Withdrawal
  reason is solely the undefined empty-family bootstrap path; all prior source,
  fold, legality, routing, and readiness audits remain evidence. Withdrawal
  receipt SHA-256:
  `98b1cea1bc56394eb1b801091dd269c9250857b524a65867b66d3667276e0804`.
- A proposal-only structural check generated 5,000 Bayesian pool-weight draws
  for all full/early->late/late->early outer-fold universes. Each draw used
  positive Exp(1) weights per eligible `table_id`, normalized to mean one in
  its view/fold, with the same weight inherited by all queries. All 45
  view/fold/family denominators remained positive; the smallest was 1.0232903
  in early->late fold-0 isolation. Weight-stream SHA-256 is
  `c833ac7d6dd02d2009f41f3566667dcf37e59071646cc1afe9929395b601975c`;
  feasibility-receipt SHA-256 is
  `7a89801e9e5bd1bbd46bfb270bdbed04e06d525f3ce18f581254b4a7d1772849`.
- Separately versioned `EVP-DRAFT-03` replaces ordinary bootstrap text with a
  complete paired Bayesian cluster-bootstrap contract: exact RNG schedule
  `SeedSequence([12673, view_index, outer_fold])`, 5,000 draws, mean-one pool
  weights shared by H0/L0/L1 and all families, exact weighted-query and
  macro-family formulas, linear 5%/95% quantiles, and no reject/omit/zero-fill
  path. Independent hand/query or per-family weights remain prohibited.
- v3 retains every prior data, nested-routing, tie, thin-cell, and stop/go
  control. Its SHA-256 is
  `9464378b8406586367cbf3e26af708e5cf2573dccf03cb4bacd1f1a61276c045`.
  Static document audit status is `PASS_DRAFT_ONLY_AWAITING_SUPERVISION`,
  SHA-256
  `cfe4f8daddf25b81a5b8fe82c76ba30be8733e6ed33d01454cc3ba3e73dbc790`.
- Decision: no implementation exists and no model, performance metric,
  evaluation row, or submission was touched. v3 remains execution-forbidden
  until a new supervisory decision; no E-number is created.

## EVP v3 final interval-semantics review

- A later, explicitly documentation-only supervisory review required the v3
  output to be named exactly a “90% paired Bayesian cluster-weight sensitivity
  interval,” not a confidence interval. It also required exact separation of
  the query-weighted overall, within-family pool-weighted mean, unweighted
  macro-family mean, and query-level paired delta. No coverage simulation was
  designed or run.
- The numerical mechanism is unchanged: 5,000 paired positive `table_id`
  weights, identical draws shared across H0/L0/L1 and all families, the frozen
  RNG schedule and weight-stream hash, and unrounded linear 5%/95% quantiles.
  A positive lower sensitivity bound is only a perturbation-robustness gate;
  it is not a significance, posterior-probability, or coverage claim. Any
  result-bearing output that calls it a confidence/credible/coverage interval
  must stop before aggregate performance is viewed.
- Rubin (1981), Field and Welsh (2007), and Præstgaard and Wellner (1993) were
  checked as the methodological boundary: their Bayesian, clustered-data, and
  exchangeably weighted results depend on model assumptions or sufficient
  conditions and do not automatically establish finite-sample coverage for
  this small, stratified, nonlinear AP@5 design.
- Under the explicit final-wording instruction, `EVP-DRAFT-03` now has SHA-256
  `54776c546377da214626828f71deed383302ceee95142073da7b031a0d1723e5`.
  Its static audit retains status `PASS_DRAFT_ONLY_AWAITING_SUPERVISION`, adds
  execution state `DRAFT_ONLY_AWAITING_SUPERVISION`, and has SHA-256
  `a45c55aae9cfac2b4caafb99511815b38b4da9c87c0b431edb985d554e117b91`.
  The earlier `9464378b...c045` / `cfe4f8da...c790` hashes remain historical
  records of the pre-terminology wording and its audit; v2 withdrawal and its
  historical binding PASS remain byte-identical.
- Decision: this was document revision plus read-only static audit only. No
  implementation, model fit, performance metric, evaluation access, E-number,
  candidate promotion, or submission was authorized or performed.

## EVP v3 implementation-only source audit and containment

- Supervision subsequently authorized only implementation plus non-result
  static and synthetic tests, bound to v3 SHA-256
  `54776c546377da214626828f71deed383302ceee95142073da7b031a0d1723e5`
  and its historical preregistration static-audit SHA-256
  `a45c55aae9cfac2b4caafb99511815b38b4da9c87c0b431edb985d554e117b91`.
  Result-bearing execution, real evidence-model fitting, aggregate performance,
  evaluation access, E-number assignment, candidate promotion, submission, and
  reruns after a stop remained forbidden.
- The frozen implementation is split between core mechanics SHA-256
  `5871093ce9df04c7370db9c462fc85f77d15b8ebfb0e0dc65e65d7808c54005e`
  and the one-run hash-locked executor SHA-256
  `5a0f5aec85d2bad1ea852c4782200316a57704aa28edc8a99e7fb27b2c7a5fad`.
  The executor rejects implementation-only permission and requires a later
  `RESULT_BEARING_EXECUTION_APPROVED` receipt that binds the draft, historical
  static audit, both source files, and the current containment receipt. It
  creates an exclusive consumption marker before any competition-data access
  and contains no evaluation or submission path.
- Source tests SHA-256
  `6817117188a1277e2487f979f3077e0177f1100c227d98286a80f43a280fac57`
  and executor tests SHA-256
  `b907b921bbf01c96fd73838ef945f1acc94d4a0fe9e6761baceafb8a69428d9a`
  passed 17/17 EVP tests, including all 55 exhaustive random-tie cases, nested
  B001 cross-fitting, exact C tie handling, authorization ordering, single-run
  consumption, and a wholly synthetic nested-selection/outer-scoring
  integration. The full repository suite passed 23/23; compilation also
  passed. Synthetic models were fitted by tests, but no real model was fitted.
- A non-result structural audit reverified the 12 frozen source hashes, 372
  targets, 45,129 legal candidate hands, 1,817 organizer evidence hands, the
  single retained early zero-preference query, and all frozen fold/family/time
  coverage. It reproduced the weight hash
  `c833ac7d6dd02d2009f41f3566667dcf37e59071646cc1afe9929395b601975c`
  with view/fold headers and the no-header negative control
  `08ee7564ab1c5aa7736c1fc541a006f6c5181b2638088a6b91d24e034a222e1a`.
  The historical weight hash does not bind pool IDs or shapes; exact source,
  pool, fold, and population checks are retained as companion controls, with no
  in-place amendment to v3.
- The older preregistration static audit remains byte-identical and historical.
  Its then-true `implementation_created=false` claim is superseded for current
  state only by `poker/work/EVP_implementation_containment_audit.json`, status
  `PASS_IMPLEMENTATION_PRESENT_UNEXECUTED`, SHA-256
  `f04b10963b9e466cda2cb819a668344464c7efd7ead98ebb772b6879b1e5f75c`.
  No result artifact or authorization-consumption marker exists. The work stops
  at the source-audit gate; no E-number is created.

## EVP v3 source rejection and versioned v4 design

- Independent source audit formally rejected the contained v3 implementation
  without executing it. The decision is preserved as
  `SOURCE_REJECTED_UNEXECUTED` at
  `poker/work/EVP_v3_source_audit_rejection.json`, SHA-256
  `90bb142996798652cc64adb863c93ff845452fb56363500cab969b68de213a1d`.
  Rejected core/runner hashes remain byte-identical at
  `5871093c...005e` / `5a0f5aec...5fad`; no implementation modification is
  permitted before a separately versioned v4 method-audit PASS.
- The rejection identifies four mandatory repairs: hard-code the canonical
  containment path/hash instead of dereferencing an authorization-supplied
  audit path; remove the family-semantics conflict in C selection; make the
  primary, attribution, and time gates strictly sequential at both compute and
  data-access levels; and embed exact local query/candidate/evidence coverage
  beside every time/family/fold/direction-family metric with explicit stops.
- `EVP-DRAFT-04`, SHA-256
  `3a9f20936f4173f72cb4ff867d296eb2945b43ca40cac0ee587b7fb4b83da22c`,
  resolves the C ambiguity by selecting L1 C on the unweighted mean inner-OOF
  query AP@5 with no family grouping or family field in the selection frame.
  Held-out true family cannot enter features, routing, eligibility, selection,
  or scores; it may be joined only after score/top-five hashes as reporting
  metadata for frozen strata and gates.
- v4 freezes the state machine as full L1-vs-H0 primary, then matched-C full L0
  attribution, then bidirectional time validation. Primary failure prevents
  any L0/time fit, metric, serialization, source hash, or time-data read;
  attribution failure similarly prevents all time access. Stage-local weight
  hashes allow full-only sensitivity verification before any time source is
  opened.
- The exact metric-local design contract is
  `poker/work/EVP_v4_metric_coverage_contract.json`, SHA-256
  `4e821092161be0b0c80892af41ed61718e0a4930562b748a826da8a8bf9001e6`.
  A read-only structural self-check matched all 72 overall/family/fold/fold-by-
  family cells. Every metric must embed the local tuple `(eligible queries,
  candidate hands, evidence hands, eligible pools)`. Exact mismatch, missing
  cells, or time fold-family coverage below 7/458/17/6 is a hard stop.
- v4 is design-only and awaits independent method audit. No source change,
  real model fit, performance metric, result artifact, consumption marker,
  evaluation access, E-number, candidate promotion, or submission occurred.
