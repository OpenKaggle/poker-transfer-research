"""Narrow exposure-sensitive feature normalization and ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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


def _transform(
    frame: pd.DataFrame,
    features: list[str],
    dropped: set[str],
    *,
    development: bool,
) -> tuple[np.ndarray, list[str]]:
    kept = [feature for feature in features if feature not in dropped]
    transformed = frame[kept].copy()
    original_hands = frame["shared_hands_calc"].replace(0, np.nan).astype(float)
    if "shared_hands_calc" in transformed:
        transformed["shared_hands_calc"] = original_hands * (2.0 / 3.0 if development else 1.0)
    if "dominant_flow_bb" in transformed:
        transformed["dominant_flow_bb"] = (
            100.0 * frame["dominant_flow_bb"].astype(float) / original_hands
        )
    matrix = (
        transformed.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    return matrix, kept


def run_exposure_validation(
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
        raise ValueError("exposure audit must keep unknown targets latent")
    features = _numeric_columns(confirmed.drop(columns=["table_id"], errors="ignore"))
    drift = pd.read_csv(output_root / "E001_feature_shift.csv").set_index("feature")
    missing = set(features) - set(drift.index)
    if missing:
        raise ValueError(f"feature drift audit is missing fields: {sorted(missing)}")
    sample_sensitive = {
        feature
        for feature in features
        if (
            feature.endswith("__max")
            or re.search(r"__t[0-3]$", feature) is not None
        )
        and float(drift.loc[feature, "unlabeled_vs_evaluation_ks"]) >= 0.075
    }
    raw_confirmed = _matrix(confirmed, features)
    raw_unknown = _matrix(unknown, features)
    raw_evaluation = _matrix(evaluation, features)
    normalized_confirmed, normalized_features = _transform(
        confirmed, features, sample_sensitive, development=True
    )
    normalized_unknown, check_features = _transform(
        unknown, features, sample_sensitive, development=True
    )
    normalized_evaluation, evaluation_features = _transform(
        evaluation, features, sample_sensitive, development=False
    )
    if normalized_features != check_features or normalized_features != evaluation_features:
        raise ValueError("exposure-normalized feature schemas do not align")

    y = confirmed["label"].to_numpy(dtype=np.int8)
    labelled_folds, unknown_folds = _assign_folds(confirmed, unknown, seed)
    raw_oof = np.zeros(len(confirmed), dtype=np.float32)
    normalized_oof = np.zeros(len(confirmed), dtype=np.float32)
    raw_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    normalized_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    raw_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    normalized_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    normalized_fold_ap: list[float] = []

    for fold in range(5):
        train = np.flatnonzero(labelled_folds != fold)
        valid = np.flatnonzero(labelled_folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        raw_model = _new_model(seed + fold)
        raw_model.fit(raw_confirmed[train], y[train])
        raw_oof[valid] = raw_model.predict_proba(raw_confirmed[valid])[:, 1]
        raw_unknown_scores[unknown_valid] = raw_model.predict_proba(raw_unknown[unknown_valid])[:, 1]
        raw_evaluation_scores += raw_model.predict_proba(raw_evaluation)[:, 1] / 5

        normalized_model = _new_model(seed + 80 + fold)
        normalized_model.fit(normalized_confirmed[train], y[train])
        normalized_oof[valid] = normalized_model.predict_proba(normalized_confirmed[valid])[:, 1]
        normalized_unknown_scores[unknown_valid] = normalized_model.predict_proba(
            normalized_unknown[unknown_valid]
        )[:, 1]
        normalized_evaluation_scores += normalized_model.predict_proba(
            normalized_evaluation
        )[:, 1] / 5
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        normalized_fold_ap.append(average_precision(y[valid], normalized_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    normalized_ap = average_precision(y, normalized_oof)
    raw_separation = _positive_percentiles(
        confirmed, raw_oof, unknown, raw_unknown_scores
    )
    normalized_separation = _positive_percentiles(
        confirmed, normalized_oof, unknown, normalized_unknown_scores
    )
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    normalized_ks = float(
        ks_2samp(normalized_unknown_scores, normalized_evaluation_scores).statistic
    )
    ks_reduction = float((raw_ks - normalized_ks) / max(raw_ks, 1e-12))
    raw_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], raw_evaluation_scores).statistic
    )
    normalized_shared_corr = float(
        spearmanr(
            evaluation["shared_hands_calc"], normalized_evaluation_scores
        ).statistic
    )
    confirmed_ap_delta = float(normalized_ap - raw_ap)
    worst_fold_delta = float(min(normalized_fold_ap) - min(raw_fold_ap))
    p95_delta = float(
        normalized_separation["known_positive_above_unlabeled_p95_rate"]
        - raw_separation["known_positive_above_unlabeled_p95_rate"]
    )
    passes = bool(
        confirmed_ap_delta >= -0.015
        and worst_fold_delta >= -0.03
        and ks_reduction >= 0.30
        and p95_delta >= -0.02
        and abs(normalized_shared_corr) <= 0.10
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = labelled_folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["exposure_oof"] = normalized_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["exposure_oof_score"] = normalized_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc"]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["exposure_score"] = normalized_evaluation_scores
    labelled_path = output_root / "E005_exposure_labelled_oof.parquet"
    unknown_path = output_root / "E005_exposure_unlabeled_oof.parquet"
    evaluation_path = output_root / "E005_exposure_evaluation.parquet"
    labelled_receipt.to_parquet(labelled_path, index=False)
    unknown_receipt.to_parquet(unknown_path, index=False)
    evaluation_receipt.to_parquet(evaluation_path, index=False)

    report: dict[str, object] = {
        "experiment": "E005_narrow_exposure_normalization",
        "raw_feature_count": int(len(features)),
        "normalized_feature_count": int(len(normalized_features)),
        "dropped_sample_sensitive_features": sorted(sample_sensitive),
        "transformed_features": {
            "shared_hands_calc": "development_count_times_2_over_3",
            "dominant_flow_bb": "flow_per_100_shared_hands",
        },
        "fold_boundary": "whole_table_equals_whole_pool",
        "unknown_ground_truth_labels_assigned": 0,
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            "evaluation_score_shared_hands_spearman": raw_shared_corr,
            **raw_separation,
        },
        "exposure_normalized": {
            "confirmed_pair_ap": float(normalized_ap),
            "fold_pair_ap": normalized_fold_ap,
            "unlabeled_to_evaluation_score_ks": normalized_ks,
            "evaluation_score_shared_hands_spearman": normalized_shared_corr,
            **normalized_separation,
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
    report_path = output_root / "E005_exposure_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_exposure_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
