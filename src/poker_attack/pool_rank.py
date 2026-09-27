"""Pool-relative domain normalization for development/evaluation pair features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .pu import _assign_folds, _positive_percentiles
from .schema import input_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pool_percentiles(
    confirmed: pd.DataFrame,
    unknown: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    development = pd.concat(
        [
            confirmed[["table_id", *features]].assign(_source="confirmed"),
            unknown[["table_id", *features]].assign(_source="unlabeled"),
        ],
        ignore_index=True,
    )
    development_rank = (
        development.groupby("table_id", sort=False)[features]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )
    confirmed_rank = development_rank.iloc[: len(confirmed)].to_numpy(np.float32)
    unknown_rank = development_rank.iloc[len(confirmed) :].to_numpy(np.float32)
    evaluation_rank = (
        evaluation.groupby("table_id", sort=False)[features]
        .rank(method="average", pct=True)
        .astype(np.float32)
        .to_numpy(np.float32)
    )
    return confirmed_rank, unknown_rank, evaluation_rank


def run_pool_rank_validation(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    confirmed = pd.read_parquet(output_root / "development_pair_features.parquet").merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    unknown_pairs = pd.read_parquet(output_root / "pu_unknown_pairs.parquet")
    unknown = pd.read_parquet(output_root / "pu_unknown_features.parquet").merge(
        unknown_pairs[["pair_id", "label_status", "training_role"]],
        on="pair_id",
        validate="one_to_one",
    )
    evaluation = pd.read_parquet(output_root / "evaluation_pair_features.parquet")
    if "label" in unknown or set(unknown["training_role"]) != {"unlabeled"}:
        raise ValueError("pool-rank normalization must keep unknown targets latent")
    features = _numeric_columns(confirmed.drop(columns=["table_id"], errors="ignore"))
    confirmed_x = _matrix(confirmed, features)
    unknown_x = _matrix(unknown, features)
    evaluation_x = _matrix(evaluation, features)
    confirmed_rank, unknown_rank, evaluation_rank = _pool_percentiles(
        confirmed, unknown, evaluation, features
    )
    y = confirmed["label"].to_numpy(dtype=np.int8)
    labelled_folds, unknown_folds = _assign_folds(confirmed, unknown, seed)

    raw_oof = np.zeros(len(confirmed), dtype=np.float32)
    rank_oof = np.zeros(len(confirmed), dtype=np.float32)
    raw_unknown = np.zeros(len(unknown), dtype=np.float32)
    rank_unknown = np.zeros(len(unknown), dtype=np.float32)
    raw_evaluation = np.zeros(len(evaluation), dtype=np.float32)
    rank_evaluation = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    rank_fold_ap: list[float] = []

    for fold in range(5):
        train = np.flatnonzero(labelled_folds != fold)
        valid = np.flatnonzero(labelled_folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        raw_model = _new_model(seed + fold)
        raw_model.fit(confirmed_x[train], y[train])
        raw_oof[valid] = raw_model.predict_proba(confirmed_x[valid])[:, 1]
        raw_unknown[unknown_valid] = raw_model.predict_proba(unknown_x[unknown_valid])[:, 1]
        raw_evaluation += raw_model.predict_proba(evaluation_x)[:, 1] / 5

        rank_model = _new_model(seed + 50 + fold)
        rank_model.fit(confirmed_rank[train], y[train])
        rank_oof[valid] = rank_model.predict_proba(confirmed_rank[valid])[:, 1]
        rank_unknown[unknown_valid] = rank_model.predict_proba(unknown_rank[unknown_valid])[:, 1]
        rank_evaluation += rank_model.predict_proba(evaluation_rank)[:, 1] / 5
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        rank_fold_ap.append(average_precision(y[valid], rank_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    rank_ap = average_precision(y, rank_oof)
    raw_separation = _positive_percentiles(confirmed, raw_oof, unknown, raw_unknown)
    rank_separation = _positive_percentiles(confirmed, rank_oof, unknown, rank_unknown)
    raw_domain_ks = float(ks_2samp(raw_unknown, raw_evaluation).statistic)
    rank_domain_ks = float(ks_2samp(rank_unknown, rank_evaluation).statistic)
    ks_reduction = float((raw_domain_ks - rank_domain_ks) / max(raw_domain_ks, 1e-12))
    raw_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], raw_evaluation).statistic
    )
    rank_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], rank_evaluation).statistic
    )
    confirmed_ap_delta = float(rank_ap - raw_ap)
    worst_fold_delta = float(min(rank_fold_ap) - min(raw_fold_ap))
    p95_delta = float(
        rank_separation["known_positive_above_unlabeled_p95_rate"]
        - raw_separation["known_positive_above_unlabeled_p95_rate"]
    )
    passes = bool(
        confirmed_ap_delta >= -0.015
        and worst_fold_delta >= -0.03
        and ks_reduction >= 0.30
        and p95_delta >= -0.02
        and abs(rank_shared_corr) <= 0.10
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = labelled_folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["pool_rank_oof"] = rank_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown
    unknown_receipt["pool_rank_oof_score"] = rank_unknown
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc"]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation
    evaluation_receipt["pool_rank_score"] = rank_evaluation
    labelled_path = output_root / "E004_pool_rank_labelled_oof.parquet"
    unknown_path = output_root / "E004_pool_rank_unlabeled_oof.parquet"
    evaluation_path = output_root / "E004_pool_rank_evaluation.parquet"
    labelled_receipt.to_parquet(labelled_path, index=False)
    unknown_receipt.to_parquet(unknown_path, index=False)
    evaluation_receipt.to_parquet(evaluation_path, index=False)

    report: dict[str, object] = {
        "experiment": "E004_pool_relative_domain_normalization",
        "feature_count": int(len(features)),
        "fold_boundary": "whole_table_equals_whole_pool",
        "unknown_ground_truth_labels_assigned": 0,
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "confirmed_brier": float(np.mean((raw_oof - y) ** 2)),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_domain_ks,
            "evaluation_score_shared_hands_spearman": raw_shared_corr,
            **raw_separation,
        },
        "pool_rank": {
            "confirmed_pair_ap": float(rank_ap),
            "confirmed_brier": float(np.mean((rank_oof - y) ** 2)),
            "fold_pair_ap": rank_fold_ap,
            "unlabeled_to_evaluation_score_ks": rank_domain_ks,
            "evaluation_score_shared_hands_spearman": rank_shared_corr,
            **rank_separation,
        },
        "deltas": {
            "confirmed_pair_ap": confirmed_ap_delta,
            "worst_fold_pair_ap": worst_fold_delta,
            "unlabeled_to_evaluation_score_ks_fraction_reduction": ks_reduction,
            "known_positive_above_unlabeled_p95_rate": p95_delta,
        },
        "preregistered_gate": {
            "confirmed_pair_ap_delta_min": -0.015,
            "worst_fold_pair_ap_delta_min": -0.03,
            "unlabeled_to_evaluation_score_ks_fraction_reduction_min": 0.30,
            "known_positive_above_unlabeled_p95_rate_delta_min": -0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "passes": passes,
        },
        "labelled_oof_artifact": str(labelled_path),
        "unlabeled_oof_artifact": str(unknown_path),
        "evaluation_score_artifact": str(evaluation_path),
    }
    for name, path in (
        ("labelled_oof_sha256", labelled_path),
        ("unlabeled_oof_sha256", unknown_path),
        ("evaluation_score_sha256", evaluation_path),
    ):
        report[name] = _sha256(path)
    report_path = output_root / "E004_pool_rank_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_pool_rank_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
