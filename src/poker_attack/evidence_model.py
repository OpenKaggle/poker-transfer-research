"""Pool-held-out evidence-hand ranking for confirmed positive pairs only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold

from .schema import EVIDENCE_COLUMNS, TARGET_BEHAVIORS


@dataclass
class EvidenceModelResult:
    models: dict[str, LGBMRanker]
    base_features: list[str]
    rank_features: list[str]
    diagnostics: dict[str, object]


_IDENTITY_COLUMNS = {
    "pair_id",
    "hand_id",
    "player_1",
    "player_2",
    "table_id",
    "table_id_1",
    "table_id_2",
    "target",
}

_RANK_TOKENS = (
    "signal",
    "transfer",
    "contribution",
    "net_",
    "pot_",
    "aggress",
    "raise",
    "bet",
    "call",
    "check",
    "fold",
    "showdown",
    "pressure",
    "amount",
    "active",
    "hole",
)


def _feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select gameplay fields while explicitly excluding every identifier."""

    base = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in _IDENTITY_COLUMNS
    ]
    ranked = [column for column in base if any(token in column for token in _RANK_TOKENS)]
    return base, ranked


def _matrix(
    frame: pd.DataFrame,
    base_features: list[str],
    rank_features: list[str],
) -> pd.DataFrame:
    base = (
        frame.reindex(columns=base_features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
        .reset_index(drop=True)
    )
    if not rank_features:
        return base
    percentile = (
        frame.groupby("pair_id", sort=False)[rank_features]
        .rank(method="average", pct=True)
        .astype(np.float32)
        .reset_index(drop=True)
    )
    percentile.columns = [f"{column}__pair_pct" for column in rank_features]
    return pd.concat([base, percentile], axis=1)


def _map5(frame: pd.DataFrame, scores: np.ndarray) -> float:
    ranked = frame[["pair_id", "target"]].copy()
    ranked["score"] = scores
    values: list[float] = []
    for _, pair in ranked.groupby("pair_id", sort=False):
        relevance = pair.sort_values("score", ascending=False, kind="stable")[
            "target"
        ].to_numpy()[:5]
        denominator = min(int(pair["target"].sum()), 5)
        if denominator == 0:
            values.append(0.0)
            continue
        hits = np.cumsum(relevance)
        values.append(
            float(
                np.sum((hits / np.arange(1, len(relevance) + 1)) * relevance)
                / denominator
            )
        )
    return float(np.mean(values)) if values else 0.0


def _new_ranker(seed: int, n_estimators: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="map",
        eval_at=[5],
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=25,
        reg_alpha=0.1,
        reg_lambda=4.0,
        colsample_bytree=0.8,
        verbosity=-1,
        random_state=seed,
        n_jobs=4,
    )


def _query_order(frame: pd.DataFrame, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = positions[
        np.argsort(frame.iloc[positions]["pair_id"].astype(str).to_numpy(), kind="stable")
    ]
    sizes = frame.iloc[ordered].groupby("pair_id", sort=False).size().to_numpy()
    return ordered, sizes


def train_evidence_models(
    development_pair_hands: pd.DataFrame,
    labels: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    seed: int = 7401,
) -> EvidenceModelResult:
    """Train one family-conditional ranker with whole-pool OOF validation.

    Only confirmed positive pairs and their organizer-provided evidence labels
    enter this fit. Unlisted development pairs are never assigned a target.
    """

    confirmed = labels[
        (labels["label"] == 1) & (labels["label_status"] == "confirmed_target")
    ][["pair_id", "behavior_family"]]
    frame = development_pair_hands.merge(
        confirmed, on="pair_id", how="inner", validate="many_to_one"
    )
    evidence_keys = pd.MultiIndex.from_frame(
        evidence[["pair_id", "hand_id"]].astype(str)
    )
    frame_keys = pd.MultiIndex.from_frame(frame[["pair_id", "hand_id"]].astype(str))
    frame["target"] = frame_keys.isin(evidence_keys).astype(np.int8)
    if int(frame["target"].sum()) != len(evidence):
        raise ValueError("not every development evidence hand is a legal shared hand")

    base_features, rank_features = _feature_columns(frame)
    matrix = _matrix(frame, base_features, rank_features)
    models: dict[str, LGBMRanker] = {}
    family_diagnostics: dict[str, object] = {}
    weighted_sum = 0.0
    total_pairs = 0

    for family_index, family in enumerate(TARGET_BEHAVIORS):
        family_frame = frame[frame["behavior_family"] == family].copy().reset_index(drop=True)
        family_matrix = matrix.loc[frame["behavior_family"] == family].reset_index(drop=True)
        pair_meta = (
            family_frame.groupby("pair_id", sort=False)
            .agg(table_id=("table_id", "first"), evidence=("target", "sum"))
            .reset_index()
        )
        n_folds = min(5, int(pair_meta["table_id"].nunique()))
        oof = np.zeros(len(family_frame), dtype=np.float32)
        best_rounds: list[int] = []
        fold_scores: list[float] = []
        if n_folds >= 2:
            pair_fold: dict[object, int] = {}
            splitter = GroupKFold(n_folds)
            for fold, (_, pair_valid) in enumerate(
                splitter.split(pair_meta, pair_meta["evidence"], pair_meta["table_id"])
            ):
                pair_fold.update({pair_id: fold for pair_id in pair_meta.iloc[pair_valid]["pair_id"]})
            folds = family_frame["pair_id"].map(pair_fold).to_numpy(dtype=int)
            for fold in range(n_folds):
                train_positions = np.flatnonzero(folds != fold)
                valid_positions = np.flatnonzero(folds == fold)
                train_order, train_groups = _query_order(family_frame, train_positions)
                valid_order, valid_groups = _query_order(family_frame, valid_positions)
                ranker = _new_ranker(seed + family_index * 20 + fold, 500)
                ranker.fit(
                    family_matrix.iloc[train_order],
                    family_frame.iloc[train_order]["target"],
                    group=train_groups,
                    eval_set=[
                        (
                            family_matrix.iloc[valid_order],
                            family_frame.iloc[valid_order]["target"],
                        )
                    ],
                    eval_group=[valid_groups],
                    callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
                )
                oof[valid_order] = ranker.predict(family_matrix.iloc[valid_order])
                best_rounds.append(max(1, int(ranker.best_iteration_)))
                fold_scores.append(
                    _map5(
                        family_frame.iloc[valid_order],
                        oof[valid_order],
                    )
                )
        else:
            best_rounds.append(80)

        full_order, full_groups = _query_order(
            family_frame, np.arange(len(family_frame), dtype=int)
        )
        final_rounds = max(10, int(round(float(np.median(best_rounds)))))
        final_model = _new_ranker(seed + 100 + family_index, final_rounds)
        final_model.fit(
            family_matrix.iloc[full_order],
            family_frame.iloc[full_order]["target"],
            group=full_groups,
            callbacks=[log_evaluation(0)],
        )
        models[family] = final_model

        family_score = _map5(family_frame, oof) if n_folds >= 2 else 0.0
        pair_count = int(family_frame["pair_id"].nunique())
        weighted_sum += family_score * pair_count
        total_pairs += pair_count
        family_diagnostics[family] = {
            "pairs": pair_count,
            "hands": int(len(family_frame)),
            "evidence_hands": int(family_frame["target"].sum()),
            "oof_map5": family_score,
            "fold_map5": fold_scores,
            "final_trees": final_rounds,
        }

    return EvidenceModelResult(
        models=models,
        base_features=base_features,
        rank_features=rank_features,
        diagnostics={
            "training_pairs": int(confirmed["pair_id"].nunique()),
            "training_hands": int(len(frame)),
            "training_evidence_hands": int(frame["target"].sum()),
            "n_features": int(len(base_features) + len(rank_features)),
            "oof_map5": float(weighted_sum / total_pairs),
            "families": family_diagnostics,
            "unknown_pairs_used_as_negative": 0,
        },
    )


def rank_evidence_with_models(
    pair_hands: pd.DataFrame,
    predictions: pd.DataFrame,
    result: EvidenceModelResult,
    *,
    pairs_per_batch: int = 512,
) -> pd.DataFrame:
    """Score evaluation hands in bounded, whole-pair batches and return top five."""

    if "evidence_behavior" not in predictions:
        raise ValueError("predictions must include evidence_behavior for every pair")
    behavior = predictions.set_index("pair_id")["evidence_behavior"]
    pair_ids = pair_hands["pair_id"].astype(str).to_numpy()
    if len(pair_ids) == 0:
        return pd.DataFrame(columns=["pair_id", *EVIDENCE_COLUMNS])
    boundaries = np.flatnonzero(pair_ids[1:] != pair_ids[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(pair_ids)]
    top_frames: list[pd.DataFrame] = []

    for group_start in range(0, len(starts), pairs_per_batch):
        group_end = min(group_start + pairs_per_batch, len(starts))
        chunk = pair_hands.iloc[starts[group_start] : ends[group_end - 1]].reset_index(drop=True)
        matrix = _matrix(chunk, result.base_features, result.rank_features)
        family = chunk["pair_id"].map(behavior).astype(str).to_numpy()
        scores = np.zeros(len(chunk), dtype=np.float32)
        for name, model in result.models.items():
            positions = np.flatnonzero(family == name)
            if len(positions):
                scores[positions] = model.predict(matrix.iloc[positions])
        ranked = chunk[["pair_id", "hand_id"]].copy()
        ranked["evidence_score"] = scores
        top_frames.append(
            ranked.sort_values(
                ["pair_id", "evidence_score", "hand_id"],
                ascending=[True, False, True],
                kind="stable",
            )
            .groupby("pair_id", sort=False)
            .head(5)
        )

    top = pd.concat(top_frames, ignore_index=True)
    top["rank"] = top.groupby("pair_id", sort=False).cumcount() + 1
    return (
        top.pivot(index="pair_id", columns="rank", values="hand_id")
        .rename(columns=lambda rank: f"evidence_hand_{rank}")
        .reset_index()
    )
