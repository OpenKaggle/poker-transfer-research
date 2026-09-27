"""Owned ablation for partner/outsider action-response feature aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp, spearmanr

from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .pu import _assign_folds, _positive_percentiles
from .schema import input_paths


INTERACTION_HAND_FEATURES = [
    "true_hu_actions",
    "true_hu_checks",
    "true_hu_calls",
    "true_hu_aggression",
    "p1_hu_passive",
    "p2_hu_passive",
    "partner_folds_to_pair",
    "partner_calls_to_pair",
    "partner_raises_to_pair",
    "outsider_folds_to_pair",
    "outsider_calls_to_pair",
    "outsider_raises_to_pair",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_interactions(
    source: Path,
    output: Path,
    *,
    phase_half: str | None = None,
    pair_ids: list[str] | None = None,
) -> None:
    if output.exists():
        return
    lazy = pl.scan_parquet(source)
    if phase_half == "early":
        lazy = lazy.filter(pl.col("phase_progress") <= 0.5)
    elif phase_half == "late":
        lazy = lazy.filter(pl.col("phase_progress") > 0.5)
    elif phase_half is not None:
        raise ValueError(f"unknown phase_half: {phase_half}")
    if pair_ids is not None:
        lazy = lazy.filter(pl.col("pair_id").is_in(pair_ids))
    aggregations = []
    for feature in INTERACTION_HAND_FEATURES:
        aggregations.extend(
            [
                pl.col(feature).mean().alias(f"interaction_{feature}__mean"),
                pl.col(feature).quantile(0.95, interpolation="nearest").alias(
                    f"interaction_{feature}__p95"
                ),
                pl.col(feature).top_k(5).mean().alias(
                    f"interaction_{feature}__top5"
                ),
                pl.col(feature).sum().alias(f"_sum_{feature}"),
            ]
        )
    frame = lazy.group_by("pair_id").agg(aggregations).with_columns(
        (
            (pl.col("_sum_true_hu_checks") + pl.col("_sum_true_hu_calls"))
            / (pl.col("_sum_true_hu_actions") + 1)
        ).alias("interaction_true_hu_passive_rate"),
        (
            pl.col("_sum_true_hu_aggression")
            / (pl.col("_sum_true_hu_actions") + 1)
        ).alias("interaction_true_hu_aggression_rate"),
        (
            pl.col("_sum_partner_folds_to_pair")
            / (
                pl.col("_sum_partner_folds_to_pair")
                + pl.col("_sum_partner_calls_to_pair")
                + pl.col("_sum_partner_raises_to_pair")
                + 1
            )
        ).alias("interaction_partner_fold_rate"),
        (
            pl.col("_sum_outsider_folds_to_pair")
            / (
                pl.col("_sum_outsider_folds_to_pair")
                + pl.col("_sum_outsider_calls_to_pair")
                + pl.col("_sum_outsider_raises_to_pair")
                + 1
            )
        ).alias("interaction_outsider_fold_rate"),
        (
            pl.col("_sum_outsider_folds_to_pair")
            - pl.col("_sum_partner_folds_to_pair")
        )
        .truediv(
            pl.col("_sum_outsider_folds_to_pair")
            + pl.col("_sum_partner_folds_to_pair")
            + 1
        )
        .alias("interaction_outsider_partner_fold_contrast"),
    )
    keep = ["pair_id"] + [
        name
        for name in frame.collect_schema().names()
        if name.startswith("interaction_")
    ]
    frame.select(keep).sink_parquet(output)


def _canonical_players(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first = result["player_1"].astype(str)
    second = result["player_2"].astype(str)
    result["_p_low"] = first.where(first <= second, second)
    result["_p_high"] = second.where(first <= second, first)
    return result


def run_interaction_validation(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    public_root = output_root / "public_pu_prepared_v2"
    development_interaction_path = output_root / "E007_interaction_features_development.parquet"
    evaluation_interaction_path = output_root / "E007_interaction_features_evaluation.parquet"
    _aggregate_interactions(
        public_root / "dev_hand_features.parquet", development_interaction_path
    )
    _aggregate_interactions(
        public_root / "eval_hand_features.parquet", evaluation_interaction_path
    )
    development_interactions = pd.read_parquet(development_interaction_path)
    evaluation_interactions = pd.read_parquet(evaluation_interaction_path)
    interaction_features = [
        column for column in development_interactions if column.startswith("interaction_")
    ]

    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    confirmed = (
        pd.read_parquet(output_root / "development_pair_features.parquet")
        .merge(
            labels[["pair_id", "label", "label_status", "behavior_family"]],
            on="pair_id",
            validate="one_to_one",
        )
        .merge(development_interactions, on="pair_id", validate="one_to_one")
    )
    evaluation = pd.read_parquet(output_root / "evaluation_pair_features.parquet").merge(
        evaluation_interactions, on="pair_id", validate="one_to_one"
    )

    public_pairs = pd.read_parquet(public_root / "dev_pairs.parquet")
    public_unknown = _canonical_players(public_pairs[public_pairs["is_pu"]])
    owned_unknown_pairs = _canonical_players(
        pd.read_parquet(output_root / "pu_unknown_pairs.parquet")
    )
    mapping = public_unknown[["pair_id", "_p_low", "_p_high"]].merge(
        owned_unknown_pairs[["pair_id", "_p_low", "_p_high"]].rename(
            columns={"pair_id": "owned_pair_id"}
        ),
        on=["_p_low", "_p_high"],
        validate="one_to_one",
    )
    if len(mapping) != len(public_unknown):
        raise ValueError("not every public PU pair maps to the owned unlabeled universe")
    unknown_base = pd.read_parquet(output_root / "pu_unknown_features.parquet")
    unknown = (
        mapping[["pair_id", "owned_pair_id"]]
        .merge(
            unknown_base.rename(columns={"pair_id": "owned_pair_id"}),
            on="owned_pair_id",
            validate="one_to_one",
        )
        .drop(columns="owned_pair_id")
        .merge(development_interactions, on="pair_id", validate="one_to_one")
        .merge(
            public_unknown[["pair_id", "table_id"]].rename(
                columns={"table_id": "public_table_id"}
            ),
            on="pair_id",
            validate="one_to_one",
        )
    )
    unknown["table_id"] = unknown["public_table_id"].astype(str)
    unknown["label_status"] = "unlabeled"
    unknown["training_role"] = "unlabeled"
    unknown = unknown.drop(columns="public_table_id")
    if "label" in unknown or set(unknown["training_role"]) != {"unlabeled"}:
        raise ValueError("interaction validation must keep unknown targets latent")

    base_features = _numeric_columns(
        confirmed.drop(columns=[*interaction_features, "table_id"], errors="ignore")
    )
    augmented_features = base_features + interaction_features
    raw_confirmed = _matrix(confirmed, base_features)
    raw_unknown = _matrix(unknown, base_features)
    raw_evaluation = _matrix(evaluation, base_features)
    augmented_confirmed = _matrix(confirmed, augmented_features)
    augmented_unknown = _matrix(unknown, augmented_features)
    augmented_evaluation = _matrix(evaluation, augmented_features)
    y = confirmed["label"].to_numpy(dtype=np.int8)
    labelled_folds, unknown_folds = _assign_folds(confirmed, unknown, seed)

    raw_oof = np.zeros(len(confirmed), dtype=np.float32)
    augmented_oof = np.zeros(len(confirmed), dtype=np.float32)
    raw_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    augmented_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    raw_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    augmented_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    augmented_fold_ap: list[float] = []

    for fold in range(5):
        train = np.flatnonzero(labelled_folds != fold)
        valid = np.flatnonzero(labelled_folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        raw_model = _new_model(seed + fold)
        raw_model.fit(raw_confirmed[train], y[train])
        raw_oof[valid] = raw_model.predict_proba(raw_confirmed[valid])[:, 1]
        raw_unknown_scores[unknown_valid] = raw_model.predict_proba(raw_unknown[unknown_valid])[:, 1]
        raw_evaluation_scores += raw_model.predict_proba(raw_evaluation)[:, 1] / 5

        augmented_model = _new_model(seed + 120 + fold)
        augmented_model.fit(augmented_confirmed[train], y[train])
        augmented_oof[valid] = augmented_model.predict_proba(augmented_confirmed[valid])[:, 1]
        augmented_unknown_scores[unknown_valid] = augmented_model.predict_proba(
            augmented_unknown[unknown_valid]
        )[:, 1]
        augmented_evaluation_scores += augmented_model.predict_proba(
            augmented_evaluation
        )[:, 1] / 5
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        augmented_fold_ap.append(average_precision(y[valid], augmented_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    augmented_ap = average_precision(y, augmented_oof)
    raw_separation = _positive_percentiles(
        confirmed, raw_oof, unknown, raw_unknown_scores
    )
    augmented_separation = _positive_percentiles(
        confirmed, augmented_oof, unknown, augmented_unknown_scores
    )
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    augmented_ks = float(
        ks_2samp(augmented_unknown_scores, augmented_evaluation_scores).statistic
    )
    raw_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], raw_evaluation_scores).statistic
    )
    augmented_shared_corr = float(
        spearmanr(
            evaluation["shared_hands_calc"], augmented_evaluation_scores
        ).statistic
    )
    confirmed_ap_delta = float(augmented_ap - raw_ap)
    worst_fold_delta = float(min(augmented_fold_ap) - min(raw_fold_ap))
    p95_delta = float(
        augmented_separation["known_positive_above_unlabeled_p95_rate"]
        - raw_separation["known_positive_above_unlabeled_p95_rate"]
    )
    ks_delta = float(augmented_ks - raw_ks)
    passes = bool(
        confirmed_ap_delta >= -0.005
        and worst_fold_delta >= -0.02
        and p95_delta >= 0.01
        and ks_delta <= 0.02
        and abs(augmented_shared_corr) <= 0.10
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = labelled_folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["interaction_oof"] = augmented_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["interaction_oof_score"] = augmented_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc"]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["interaction_score"] = augmented_evaluation_scores
    labelled_path = output_root / "E007_interaction_labelled_oof.parquet"
    unknown_path = output_root / "E007_interaction_unlabeled_oof.parquet"
    evaluation_path = output_root / "E007_interaction_scores_evaluation.parquet"
    labelled_receipt.to_parquet(labelled_path, index=False)
    unknown_receipt.to_parquet(unknown_path, index=False)
    evaluation_receipt.to_parquet(evaluation_path, index=False)

    report: dict[str, object] = {
        "experiment": "E007_partner_outsider_interactions_only",
        "base_feature_count": int(len(base_features)),
        "interaction_feature_count": int(len(interaction_features)),
        "interaction_features": interaction_features,
        "fold_boundary": "whole_table_equals_whole_pool",
        "unknown_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            "evaluation_score_shared_hands_spearman": raw_shared_corr,
            **raw_separation,
        },
        "interaction": {
            "confirmed_pair_ap": float(augmented_ap),
            "fold_pair_ap": augmented_fold_ap,
            "unlabeled_to_evaluation_score_ks": augmented_ks,
            "evaluation_score_shared_hands_spearman": augmented_shared_corr,
            **augmented_separation,
        },
        "deltas": {
            "confirmed_pair_ap": confirmed_ap_delta,
            "worst_fold_pair_ap": worst_fold_delta,
            "known_positive_above_unlabeled_p95_rate": p95_delta,
            "unlabeled_to_evaluation_score_ks_absolute": ks_delta,
        },
        "preregistered_gate": {
            "confirmed_pair_ap_delta_min": -0.005,
            "worst_fold_pair_ap_delta_min": -0.02,
            "known_positive_above_unlabeled_p95_rate_delta_min": 0.01,
            "unlabeled_to_evaluation_score_ks_absolute_delta_max": 0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "passes": passes,
        },
        "development_interaction_artifact": str(development_interaction_path),
        "evaluation_interaction_artifact": str(evaluation_interaction_path),
        "labelled_oof_artifact": str(labelled_path),
        "unlabeled_oof_artifact": str(unknown_path),
        "evaluation_score_artifact": str(evaluation_path),
    }
    for name, path in (
        ("development_interaction_sha256", development_interaction_path),
        ("evaluation_interaction_sha256", evaluation_interaction_path),
        ("labelled_oof_sha256", labelled_path),
        ("unlabeled_oof_sha256", unknown_path),
        ("evaluation_score_sha256", evaluation_path),
    ):
        report[name] = _sha256(path)
    report_path = output_root / "E007_interaction_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_interaction_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
