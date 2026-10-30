# Poker (Detect Suspicious Value Transfers in Poker) after-action review — 2026-10-29

> After-action review prepared 2026-10-29 from public post-competition sources;
> internal evidence grades mapped to Verified/Unverified.

Competition context: ended 2026-09-20; our line is sealed (fail-closed after an
incomplete long-run termination). Primary sources: the 8th-place team's public
solution repository (`github.com/stonesalltheway1/poker-collusion-detection`
README), title-level retrieval of Kaggle writeup/discussion pages (body text is
JavaScript-rendered and was not retrieved), and our own archived validation
records.

## A. Leaderboard mechanisms (Verified/Unverified split)

- The #1 team, score, and mechanism: **Unverified** — the winner's writeup was
  not indexed by search engines, Kaggle writeup pages are JavaScript-rendered,
  and web archives hold only shells.
- Metric [Verified, two sources — the official page and the 8th-place README]:
  Pair AP 70% / Evidence MAP@5 20% / Behavior macro-AP 10%; ~370 teams, $5,000
  total.
- Verified mechanism near the top 10 (8th place of ~370):
  PU (positive-unlabeled) learning over the 372 labeled pairs with
  XGBoost/LightGBM; the core feature contrasts behavior "toward the partner"
  vs "toward the other 28 tablemates" to cancel persona effects, plus a
  label-free normal-play policy baseline (a donor calling a partner's raise
  with probability <0.15 flags anomalous directed transfers); **phase-local
  windows** (first ~3000 hands of each pool labeled, last ~2000 hands scored —
  on the **mixing-period** convention, a public notebook caps at ~0.695); the
  evidence-list scorer's deterministic ordering rule was reverse-engineered
  (`sorted(EVENTS, key=(event_type, hand_seq))[:5]`) with a Poisson-binomial
  dynamic program building listing probabilities. Private score 0.92817;
  components Pair AP 0.9807 / Evidence 0.7218 / Behavior 0.9692.
- Other title-level reads [Verified as titles only]: 9th place ran "PU pair
  ranking on eval-length windows"; 12th place ran a dual pipeline of "policy
  residuals + evidence experts"; 4th place published a reproduction package;
  one writeup is titled "What actually moved the needle (200 experiments)."
  The 9th-place title confirms PU was a consensus route inside the top 10.
- Official clarification [Verified, single source]: discussion #742244 — only
  coordination inferable from poker activity counts; exploiting ID formats,
  row/file order, or data-generator artifacts is prohibited. Community
  complaints about label construction / the validation regime: Unverified.

## B. Mechanism differences vs our line

1. **Feature contrast.** They explicitly contrast partner-vs-other-28 to cancel
   persona effects; our feature work aggregated hand interactions only, with no
   per-opponent contrast family.
2. **Policy baseline.** They used a label-free normal-play policy to turn
   "abnormal pricing / abnormal response" into discriminative features; our
   leave-one-out residual approximation of marginal impact did not pass its
   preregistered gate.
3. **Validation windows.** They discovered coordination is phase-local and
   restricted features to the scoring period; our whole-pool OOF of 0.948
   collapsed to an LB of 0.444, and our early-vs-late window check (AP only
   0.82–0.86) observed the same phenomenon without promoting the phase-local
   window into a first-class gate [Unverified interpretation].
4. **Evidence component.** They reverse-engineered the scorer's list-ordering
   rule and modeled listing probability to reach 0.7218 on the Evidence
   component; our evidence-preregistration spec (v2→v4.2) stopped at a strict
   nested-validation specification — a 72-cell structural audit passed, but the
   corresponding method was never executed [Verified].
5. **PU usage.** Same 372 labeled pairs: their naive PU reached 8th–9th place,
   while our PU variant was judged a provisional-method mismatch (calibration
   scale mismatch, 13.5% clipped-to-one scores, 251/1860 scores exactly 1.0).
   What blocked us was the claimability of a faithful Elkan–Noto correction,
   not score feasibility [Unverified interpretation].

## C. Seal-decision retrospective (with final-board hindsight; not a reversal)

- What the seal protected [Verified]: the preregistered gates blocked six
  families of pseudo-gains before they could burn submission slots; the
  fail-closed source rejection preserved the evidence chain; and after the
  incomplete long-run termination, sealing with no automatic resume avoided
  burning quota on an unverified state.
- What the seal cost [Verified]: the one-shot long-run authorization was
  consumed by an infrastructure anomaly — after two outer folds with no
  checkpointing, the run terminated with an unknown final state. A pure
  execution loss; the science was never falsified.
- The earlier, real stop-loss target [Unverified interpretation]: the
  validation proxy itself — whole-pool folds do not match the phase-local
  scoring window. Even with the long run completed, submitting under the then-
  current validation regime would most plausibly have landed around the ~0.7
  band (compare the 8th-place team's ~0.695 ceiling under the mixing-period
  convention), not 0.948.
- **Calibration conclusion: the seal verdict stands — sealing on the incomplete
  termination was proportionate, and this review does not overturn it.** The
  direct loss (execution completeness) was acceptable relative to the cost of
  re-authorization. The genuine improvement point is decoupling: "sealing the
  line" and "fixing the validation regime" are separate decisions, and the
  window-mismatch evidence already pointed to a phase-local regime before the
  seal; not converting that into a regime revision was the largest missed
  improvement.

## D. Lessons to institutionalize

1. PU / anomaly-detection lines: the headline metric (confirmed-pair AP)
   saturates above ~0.94 and loses discrimination; the first-class
   preregistered gate must be a cross-period / cross-window recomputation
   (phase-local or eval-length windows), and mixing-period scores must be
   treated as optimistic upper bounds. [Unverified rule; direction supported by
   the 8th-place Verified evidence and our own window-mismatch data.]
2. The infrastructure-failure triplet for ROI cards: run-duration budget ×
   checkpoint/resume capability × re-authorization cost. A one-shot
   authorization plus no checkpointing plus a long run must be priced as a
   one-shot bet where failure = zero, and occupancy expectations multiplied by
   the probability of completion. [Unverified rule; matches the backtest's R4.]
3. Deterministic ordering rules in multi-output scorers (e.g., a sorted
   evidence list) are a legitimate scouting surface and can yield structural
   gains on the affected component; add them to the standard scouting
   checklist. [Unverified rule; the 8th-place Evidence gain is Verified.]

## E. Open questions

- The #1 team, the final top-3 scores, and the winner's mechanism: Unverified
  (requires manually opening the writeup pages or waiting for search indexing).
- The 9th-place "eval-length windows" construction: Unverified (title only).
- The full picture of discussion-thread complaints about label construction and
  the validation regime: Unverified (only the #742244 anti-cheat clarification
  was retrieved at title level).
- If our evidence-preregistration v4.2 were redesigned around a phase-local
  regime, the reachable ceilings for Evidence MAP@5 and Pair AP: no experiment
  exists; cannot be calibrated.
