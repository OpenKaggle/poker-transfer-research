"""Exact local reproduction of the organizer's public metric.

The official reference notebook is retained at
``poker/reference/official_metric/slash-poker-competition-metric.ipynb``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .audit import DataAuditError, validate_submission
from .schema import EVIDENCE_COLUMNS, NO_EVIDENCE, TARGET_BEHAVIORS


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = y_true[order]
    true_positives = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((true_positives / ranks) * ranked) / positives)


def _clean_evidence(values: list[object]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != NO_EVIDENCE:
            cleaned.append(text)
    return cleaned


def competition_score(solution: pd.DataFrame, submission: pd.DataFrame) -> dict[str, float]:
    """Return the final score and its three components."""

    validate_submission(submission, solution["pair_id"])
    truth = solution.set_index("pair_id").sort_index()
    predictions = submission.set_index("pair_id").loc[truth.index]

    y_true = pd.to_numeric(truth["risk_score"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise DataAuditError("solution risk_score values must be binary labels")
    risk = pd.to_numeric(predictions["risk_score"], errors="raise").to_numpy(dtype=float)
    pair_ap = average_precision(y_true, risk)

    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    pred_behavior = predictions["predicted_behavior"].astype(str).to_numpy()
    behavior_aps: list[float] = []
    for family in TARGET_BEHAVIORS:
        family_truth = (true_behavior == family).astype(int)
        family_scores = np.where(pred_behavior == family, risk, 0.0)
        behavior_aps.append(average_precision(family_truth, family_scores))
    behavior_map = float(np.mean(behavior_aps))

    evidence_aps: list[float] = []
    for position in np.flatnonzero(y_true == 1):
        relevant = set(_clean_evidence(truth.iloc[position][list(EVIDENCE_COLUMNS)].tolist()))
        submitted = _clean_evidence(
            predictions.iloc[position][list(EVIDENCE_COLUMNS)].tolist()
        )[:5]
        if not relevant:
            evidence_aps.append(0.0)
            continue
        hits = 0
        precision_sum = 0.0
        for rank, hand_id in enumerate(submitted, start=1):
            if hand_id in relevant:
                hits += 1
                precision_sum += hits / rank
        evidence_aps.append(precision_sum / min(len(relevant), 5))
    evidence_map = float(np.mean(evidence_aps)) if evidence_aps else 0.0

    final = 0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map
    return {
        "score": float(final),
        "pair_ap": float(pair_ap),
        "evidence_map5": float(evidence_map),
        "behavior_map": float(behavior_map),
    }

