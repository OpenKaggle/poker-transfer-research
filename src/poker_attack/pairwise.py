"""Confirmed-only global LambdaRank ablation with whole-pool/time validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.special import expit
from scipy.stats import ks_2samp, spearmanr

from .metric import average_precision
from .model import _matrix, _numeric_columns
from .pu import _assign_folds, _positive_percentiles
from .schema import input_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_players(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    first = result["player_1"].astype(str)
    second = result["player_2"].astype(str)
    result["_p_low"] = first.where(first <= second, second)
    result["_p_high"] = second.where(first <= second, first)
    return result


def _new_ranker(seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=280,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=12,
        reg_lambda=3.0,
        feature_fraction=0.9,
        deterministic=True,
        force_col_wise=True,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *targets: np.ndarray,
    seed: int,
) -> list[np.ndarray]:
    model = _new_ranker(seed)
    model.fit(train_x, train_y, group=[len(train_y)])
    return [expit(model.predict(target)).astype(np.float32) for target in targets]


def _load_frames(
    data_root: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    confirmed = pd.read_parquet(output_root / "development_pair_features.parquet").merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    evaluation = pd.read_parquet(output_root / "evaluation_pair_features.parquet")
    public_pairs = pd.read_parquet(output_root / "public_pu_prepared_v2" / "dev_pairs.parquet")
    public_unknown = _canonical_players(public_pairs[public_pairs["is_pu"]])
    owned_unknown = _canonical_players(pd.read_parquet(output_root / "pu_unknown_pairs.parquet"))
    mapping = public_unknown[["pair_id", "_p_low", "_p_high"]].merge(
        owned_unknown[["pair_id", "_p_low", "_p_high"]].rename(
            columns={"pair_id": "owned_pair_id"}
        ),
        on=["_p_low", "_p_high"],
        validate="one_to_one",
    )
    if len(mapping) != len(public_unknown):
        raise ValueError("public PU sample must map exactly to owned unknown pairs")
    unknown = (
        mapping[["pair_id", "owned_pair_id"]]
        .merge(
            pd.read_parquet(output_root / "pu_unknown_features.parquet").rename(
                columns={"pair_id": "owned_pair_id"}
            ),
            on="owned_pair_id",
            validate="one_to_one",
        )
        .drop(columns="owned_pair_id")
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
        raise ValueError("E010 must keep unknown targets latent")
    return confirmed, unknown, evaluation


def _aligned_baseline(
    frame: pd.DataFrame,
    receipt_path: Path,
    columns: list[str],
) -> pd.DataFrame:
    receipt = pd.read_parquet(receipt_path, columns=["pair_id", *columns])
    aligned = frame[["pair_id"]].merge(receipt, on="pair_id", validate="one_to_one")
    if aligned["pair_id"].tolist() != frame["pair_id"].astype(str).tolist():
        raise ValueError("baseline receipt did not preserve pair order")
    return aligned


def _time_frame(
    output_root: Path,
    labels: pd.DataFrame,
    half: str,
    pool_lookup: pd.Series,
    order: list[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(output_root / f"E007_{half}_base_features.parquet").merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    frame["table_id"] = frame["table_id"].fillna(frame["pair_id"].map(pool_lookup))
    if frame["table_id"].isna().any():
        raise ValueError("every time pair must keep its whole-pool boundary")
    return frame.set_index("pair_id").loc[order].reset_index()


def run_pairwise_validation(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    confirmed, unknown, evaluation = _load_frames(data_root, output_root)
    features = _numeric_columns(confirmed.drop(columns="table_id", errors="ignore"))
    confirmed_x = _matrix(confirmed, features)
    unknown_x = _matrix(unknown, features)
    evaluation_x = _matrix(evaluation, features)
    y = confirmed["label"].to_numpy(dtype=np.int8)
    folds, unknown_folds = _assign_folds(confirmed, unknown, seed)

    base_labelled = _aligned_baseline(
        confirmed,
        output_root / "E008_marginal_impact_labelled_oof.parquet",
        ["fold", "raw_oof"],
    )
    base_unknown = _aligned_baseline(
        unknown,
        output_root / "E008_marginal_impact_unlabeled_oof.parquet",
        ["fold", "raw_oof_score"],
    )
    base_evaluation = _aligned_baseline(
        evaluation,
        output_root / "E008_marginal_impact_evaluation_scores.parquet",
        ["raw_score"],
    )
    if not np.array_equal(base_labelled["fold"].to_numpy(dtype=np.int8), folds):
        raise ValueError("sealed labelled folds differ from cached E001 baseline")
    if not np.array_equal(base_unknown["fold"].to_numpy(dtype=np.int8), unknown_folds):
        raise ValueError("sealed unknown folds differ from cached E001 baseline")
    raw_oof = base_labelled["raw_oof"].to_numpy(dtype=np.float32)
    raw_unknown_scores = base_unknown["raw_oof_score"].to_numpy(dtype=np.float32)
    raw_evaluation_scores = base_evaluation["raw_score"].to_numpy(dtype=np.float32)

    pairwise_oof = np.zeros(len(confirmed), dtype=np.float32)
    pairwise_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    pairwise_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    pairwise_fold_ap: list[float] = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        valid_score, unknown_score, eval_score = _fit_predict(
            confirmed_x[train],
            y[train],
            confirmed_x[valid],
            unknown_x[unknown_valid],
            evaluation_x,
            seed=seed + fold,
        )
        pairwise_oof[valid] = valid_score
        pairwise_unknown_scores[unknown_valid] = unknown_score
        pairwise_evaluation_scores += eval_score / 5
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        pairwise_fold_ap.append(average_precision(y[valid], pairwise_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    pairwise_ap = average_precision(y, pairwise_oof)
    raw_separation = _positive_percentiles(confirmed, raw_oof, unknown, raw_unknown_scores)
    pairwise_separation = _positive_percentiles(
        confirmed, pairwise_oof, unknown, pairwise_unknown_scores
    )
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    pairwise_ks = float(
        ks_2samp(pairwise_unknown_scores, pairwise_evaluation_scores).statistic
    )
    pairwise_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], pairwise_evaluation_scores).statistic
    )

    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    order = confirmed["pair_id"].astype(str).tolist()
    pool_lookup = confirmed.set_index("pair_id")["table_id"]
    early = _time_frame(output_root, labels, "early", pool_lookup, order)
    late = _time_frame(output_root, labels, "late", pool_lookup, order)
    if not np.array_equal(early["label"].to_numpy(dtype=np.int8), y):
        raise ValueError("time labels do not align with sealed full-period folds")
    temporal_pattern = re.compile(r"__t[0-3]$")
    time_features = [
        column
        for column in _numeric_columns(early.drop(columns="table_id", errors="ignore"))
        if not column.startswith("phase_progress")
        and "temporal_range" not in column
        and temporal_pattern.search(column) is None
    ]
    early_x = _matrix(early, time_features)
    late_x = _matrix(late, time_features)
    base_time = _aligned_baseline(
        early,
        output_root / "E008_marginal_impact_time_oof.parquet",
        ["fold", "raw_early_to_late", "raw_late_to_early"],
    )
    if not np.array_equal(base_time["fold"].to_numpy(dtype=np.int8), folds):
        raise ValueError("cached time folds differ from E010 folds")
    time_predictions = {
        "raw_early_to_late": base_time["raw_early_to_late"].to_numpy(dtype=np.float32),
        "raw_late_to_early": base_time["raw_late_to_early"].to_numpy(dtype=np.float32),
        "pairwise_early_to_late": np.zeros(len(early), dtype=np.float32),
        "pairwise_late_to_early": np.zeros(len(early), dtype=np.float32),
    }
    time_fold_scores = {name: [] for name in time_predictions}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        early_to_late, = _fit_predict(
            early_x[train], y[train], late_x[valid], seed=seed + 200 + fold
        )
        late_to_early, = _fit_predict(
            late_x[train], y[train], early_x[valid], seed=seed + 220 + fold
        )
        time_predictions["pairwise_early_to_late"][valid] = early_to_late
        time_predictions["pairwise_late_to_early"][valid] = late_to_early
        for name, values in time_predictions.items():
            time_fold_scores[name].append(average_precision(y[valid], values[valid]))
    time_scores = {
        name: average_precision(y, values) for name, values in time_predictions.items()
    }
    early_late_delta = float(
        time_scores["pairwise_early_to_late"] - time_scores["raw_early_to_late"]
    )
    late_early_delta = float(
        time_scores["pairwise_late_to_early"] - time_scores["raw_late_to_early"]
    )
    time_fold_deltas = [
        time_fold_scores["pairwise_early_to_late"][fold]
        - time_fold_scores["raw_early_to_late"][fold]
        for fold in range(5)
    ] + [
        time_fold_scores["pairwise_late_to_early"][fold]
        - time_fold_scores["raw_late_to_early"][fold]
        for fold in range(5)
    ]
    time_directional_gap = float(
        abs(time_scores["pairwise_early_to_late"] - time_scores["pairwise_late_to_early"])
    )
    worst_pairwise_transfer = float(
        min(time_scores["pairwise_early_to_late"], time_scores["pairwise_late_to_early"])
    )
    deltas = {
        "confirmed_pair_ap": float(pairwise_ap - raw_ap),
        "worst_fold_pair_ap": float(min(pairwise_fold_ap) - min(raw_fold_ap)),
        "known_positive_above_unlabeled_p95_rate": float(
            pairwise_separation["known_positive_above_unlabeled_p95_rate"]
            - raw_separation["known_positive_above_unlabeled_p95_rate"]
        ),
        "unlabeled_to_evaluation_score_ks_absolute": float(pairwise_ks - raw_ks),
        "early_to_late": early_late_delta,
        "late_to_early": late_early_delta,
        "worst_time_fold": float(min(time_fold_deltas)),
        "worst_pairwise_transfer": worst_pairwise_transfer,
        "time_directional_gap": time_directional_gap,
    }
    passes = bool(
        deltas["confirmed_pair_ap"] >= 0.003
        and deltas["worst_fold_pair_ap"] >= -0.02
        and deltas["known_positive_above_unlabeled_p95_rate"] >= 0.005
        and deltas["unlabeled_to_evaluation_score_ks_absolute"] <= 0.02
        and abs(pairwise_shared_corr) <= 0.10
        and early_late_delta >= 0.01
        and late_early_delta >= 0.01
        and min(time_fold_deltas) >= -0.02
        and worst_pairwise_transfer >= 0.84
        and time_directional_gap <= 0.05
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["pairwise_oof"] = pairwise_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["pairwise_oof_score"] = pairwise_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc"]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["pairwise_score"] = pairwise_evaluation_scores
    time_receipt = early[["pair_id", "label", "label_status", "table_id"]].copy()
    time_receipt["fold"] = folds
    for name, values in time_predictions.items():
        time_receipt[name] = values
    receipt_paths = {
        "labelled": output_root / "E010_pairwise_labelled_oof.parquet",
        "unlabeled": output_root / "E010_pairwise_unlabeled_oof.parquet",
        "evaluation": output_root / "E010_pairwise_evaluation_scores.parquet",
        "time": output_root / "E010_pairwise_time_oof.parquet",
    }
    labelled_receipt.to_parquet(receipt_paths["labelled"], index=False)
    unknown_receipt.to_parquet(receipt_paths["unlabeled"], index=False)
    evaluation_receipt.to_parquet(receipt_paths["evaluation"], index=False)
    time_receipt.to_parquet(receipt_paths["time"], index=False)
    report: dict[str, object] = {
        "experiment": "E010_confirmed_only_global_pairwise_objective",
        "feature_count": int(len(features)),
        "features_changed_from_E001": 0,
        "learner": {
            "objective": "lambdarank",
            "query_policy": "one global confirmed query per training fold",
            "n_estimators": 280,
            "learning_rate": 0.035,
            "num_leaves": 15,
            "min_child_samples": 12,
            "reg_lambda": 3.0,
            "feature_fraction": 0.9,
            "margin_transform": "fixed logistic sigmoid",
        },
        "unknown_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "fold_boundary": "whole_table_equals_whole_pool",
        "cached_baseline_source": "E008 raw receipts; identical E001 features/folds",
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            **raw_separation,
        },
        "pairwise": {
            "confirmed_pair_ap": float(pairwise_ap),
            "fold_pair_ap": pairwise_fold_ap,
            "unlabeled_to_evaluation_score_ks": pairwise_ks,
            "evaluation_score_shared_hands_spearman": pairwise_shared_corr,
            **pairwise_separation,
        },
        "time_scores": time_scores,
        "time_fold_scores": time_fold_scores,
        "deltas": deltas,
        "preregistered_gate": {
            "confirmed_pair_ap_delta_min": 0.003,
            "worst_fold_pair_ap_delta_min": -0.02,
            "known_positive_above_unlabeled_p95_rate_delta_min": 0.005,
            "unlabeled_to_evaluation_score_ks_absolute_delta_max": 0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "early_to_late_delta_min": 0.01,
            "late_to_early_delta_min": 0.01,
            "worst_time_fold_delta_min": -0.02,
            "worst_pairwise_transfer_ap_min": 0.84,
            "time_directional_gap_max": 0.05,
            "passes": passes,
        },
    }
    for name, path in receipt_paths.items():
        report[f"{name}_receipt_artifact"] = str(path)
        report[f"{name}_receipt_sha256"] = _sha256(path)
    report_path = output_root / "E010_pairwise_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_pairwise_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
