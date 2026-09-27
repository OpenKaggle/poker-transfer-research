# Detect Suspicious Value Transfers in Poker — research archive

An OpenKaggle source-and-provenance archive for the Kaggle competition. The
implementation is retained for methodology and reproducibility review; it is
not a data or submission release.

This repository contains the user-authored pipeline, tests, preregistration
records, and experiment-level narrative. It intentionally excludes the
competition download, all row-level derived tables, submissions, and copies of
third-party reference notebooks. See [DATA_SOURCES.md](DATA_SOURCES.md) and
[RELEASE_MANIFEST.md](RELEASE_MANIFEST.md).

Local, reproducible baseline for Kaggle's **Detect Suspicious Value Transfers in
Poker** competition.

The pipeline deliberately uses poker activity only. It does not use ID formats,
file order, row order, generator artifacts, or private labels. The local
private workspace retains a downloaded public notebook for audit purposes; it
is cited but not redistributed here. The code in `src/` is our own implementation.

## Current state

The Kaggle account has entered the competition. The official data is downloaded
under `data/raw/poker_comp`, all eight files pass the local audit, and the first
owned anchor has been submitted. Submission `56106882` scored `0.44388` public,
below the preregistered `0.75` stop-loss gate, so additional submissions are
paused while the confirmed-negative selection shift is diagnosed offline. See
`poker/STATUS_2026-09-09.md` and `poker/EXPERIMENT_LEDGER.md`.

To reproduce the current pipeline after accepting the official rules and
downloading the data privately, run:

```bash
python -m kaggle competitions download \
  -c detect-suspicious-value-transfers-in-poker \
  -p /private/path/to/poker_comp
unzip /private/path/to/poker_comp/detect-suspicious-value-transfers-in-poker.zip \
  -d /private/path/to/poker_comp
PYTHONPATH=src python -m poker_attack.pipeline \
  --data-dir /private/path/to/poker_comp \
  --work-dir /private/path/to/work \
  --output /private/path/to/submissions/poker_residual_v1.csv
```

The pipeline performs four stages:

1. validates the eight official input tables and submission coverage;
2. aggregates action logs into player-hand features without loading all actions
   into memory at once;
3. creates pair-hand and pair-level features, including one-way chip flow,
   behavior relative to each player's normal style, and episodic time segments;
4. trains group-held-out risk and behavior models, ranks evidence hands, validates
   the exact submission schema, and writes a diagnostic JSON report.

To run only the data audit or create a schema-valid emergency submission:

```bash
PYTHONPATH=src python -m poker_attack.pipeline \
  --data-dir /private/path/to/poker_comp --work-dir /private/path/to/work --audit-only

PYTHONPATH=src python -m poker_attack.pipeline \
  --data-dir /private/path/to/poker_comp --work-dir /private/path/to/work \
  --minimal --output /private/path/to/submissions/poker_minimal.csv
```

The emergency submission ranks `shared_hands` and intentionally uses
`NO_EVIDENCE`; it is a connectivity/schema check, not a competitive model.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## First-submission gate

The public reference notebook scored about `0.5576` when inspected.  The current
top-three public scores were around `0.8977` or higher.  Keep this branch active
only if the first legal submission clears roughly `0.75`; target `0.82+` after the
residual/time-burst ablation.  These thresholds are tactical, not guarantees of a
private-leaderboard prize.
