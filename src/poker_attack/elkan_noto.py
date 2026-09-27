"""Nested Elkan-Noto PU selector with a confirmed-only outer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .pairwise import _aligned_baseline, _load_frames
from .pu import _assign_folds, _positive_percentiles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_selector(
    confirmed_x: np.ndarray,
    confirmed_y: np.ndarray,
    confirmed_indices: np.ndarray,
    unknown_x: np.ndarray,
    unknown_indices: np.ndarray,
    *,
    seed: int,
):
    confirmed_selection = confirmed_y[confirmed_indices].astype(np.int8)
    unknown_selection = np.zeros(len(unknown_indices), dtype=np.int8)
    train_x = np.concatenate(
        [confirmed_x[confirmed_indices], unknown_x[unknown_indices]], axis=0
    )
    selection_y = np.concatenate([confirmed_selection, unknown_selection])
    weights = np.ones(len(selection_y), dtype=np.float64)
    confirmed_negative_count = int((confirmed_selection == 0).sum())
    unknown_weight = (
        float(confirmed_negative_count / len(unknown_indices))
        if len(unknown_indices)
        else 0.0
    )
    if len(unknown_indices):
        weights[len(confirmed_indices) :] = unknown_weight
    model = _new_model(seed)
    model.fit(train_x, selection_y, sample_weight=weights)
    diagnostics = {
        "confirmed_rows": int(len(confirmed_indices)),
        "confirmed_positive_rows": int(confirmed_selection.sum()),
        "confirmed_negative_rows": confirmed_negative_count,
        "unknown_rows": int(len(unknown_indices)),
        "unknown_row_weight": unknown_weight,
        "unknown_total_weight": float(unknown_weight * len(unknown_indices)),
        "selection_indicator_persisted": False,
    }
    return model, diagnostics


def run_elkan_noto_validation(
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

    pu_oof = np.zeros(len(confirmed), dtype=np.float32)
    pu_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    pu_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    pu_fold_ap: list[float] = []
    fold_diagnostics: list[dict[str, object]] = []
    confirmed_groups = confirmed["table_id"].astype(str).to_numpy()
    unknown_groups = unknown["table_id"].astype(str).to_numpy()

    for fold in range(5):
        outer_train = np.flatnonzero(folds != fold)
        outer_valid = np.flatnonzero(folds == fold)
        unknown_outer_train = np.flatnonzero(unknown_folds != fold)
        unknown_outer_valid = np.flatnonzero(unknown_folds == fold)

        inner_frame = confirmed.iloc[outer_train]
        inner_splitter = StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=seed + 100 + fold
        )
        inner_train_local, inner_calibration_local = next(
            inner_splitter.split(
                inner_frame,
                inner_frame["behavior_family"].astype(str),
                inner_frame["table_id"].astype(str),
            )
        )
        inner_train = outer_train[inner_train_local]
        inner_calibration = outer_train[inner_calibration_local]
        inner_train_groups = set(confirmed_groups[inner_train])
        unknown_inner_train = np.flatnonzero(
            (unknown_folds != fold)
            & np.fromiter(
                (group in inner_train_groups for group in unknown_groups),
                dtype=bool,
                count=len(unknown_groups),
            )
        )
        calibration_positive = inner_calibration[y[inner_calibration] == 1]
        if len(calibration_positive) == 0:
            raise ValueError(f"outer fold {fold} has no inner calibration positives")
        calibration_model, calibration_fit = _fit_selector(
            confirmed_x,
            y,
            inner_train,
            unknown_x,
            unknown_inner_train,
            seed=seed + 200 + fold,
        )
        calibration_scores = calibration_model.predict_proba(
            confirmed_x[calibration_positive]
        )[:, 1]
        c_raw = float(np.mean(calibration_scores))
        c = float(np.clip(c_raw, 0.05, 0.95))

        final_model, final_fit = _fit_selector(
            confirmed_x,
            y,
            outer_train,
            unknown_x,
            unknown_outer_train,
            seed=seed + fold,
        )
        valid_selection = final_model.predict_proba(confirmed_x[outer_valid])[:, 1]
        unknown_selection = final_model.predict_proba(unknown_x[unknown_outer_valid])[:, 1]
        evaluation_selection = final_model.predict_proba(evaluation_x)[:, 1]
        pu_oof[outer_valid] = np.clip(valid_selection / c, 0.0, 1.0)
        pu_unknown_scores[unknown_outer_valid] = np.clip(unknown_selection / c, 0.0, 1.0)
        pu_evaluation_scores += np.clip(evaluation_selection / c, 0.0, 1.0) / 5
        raw_fold_ap.append(average_precision(y[outer_valid], raw_oof[outer_valid]))
        pu_fold_ap.append(average_precision(y[outer_valid], pu_oof[outer_valid]))
        fold_diagnostics.append(
            {
                "fold": fold,
                "outer_train_pools": int(len(set(confirmed_groups[outer_train]))),
                "outer_valid_pools": int(len(set(confirmed_groups[outer_valid]))),
                "inner_calibration_positive_rows": int(len(calibration_positive)),
                "c_raw": c_raw,
                "c_clipped": c,
                "calibration_fit": calibration_fit,
                "final_fit": final_fit,
                "outer_valid_fraction_clipped_one": float(
                    np.mean(valid_selection / c >= 1.0)
                ),
                "unknown_valid_fraction_clipped_one": float(
                    np.mean(unknown_selection / c >= 1.0)
                ),
                "evaluation_fraction_clipped_one": float(
                    np.mean(evaluation_selection / c >= 1.0)
                ),
            }
        )

    raw_ap = average_precision(y, raw_oof)
    pu_ap = average_precision(y, pu_oof)
    raw_separation = _positive_percentiles(confirmed, raw_oof, unknown, raw_unknown_scores)
    pu_separation = _positive_percentiles(confirmed, pu_oof, unknown, pu_unknown_scores)
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    pu_ks = float(ks_2samp(pu_unknown_scores, pu_evaluation_scores).statistic)
    pu_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], pu_evaluation_scores).statistic
    )
    deltas = {
        "confirmed_pair_ap": float(pu_ap - raw_ap),
        "worst_fold_pair_ap": float(min(pu_fold_ap) - min(raw_fold_ap)),
        "known_positive_above_unlabeled_p95_rate": float(
            pu_separation["known_positive_above_unlabeled_p95_rate"]
            - raw_separation["known_positive_above_unlabeled_p95_rate"]
        ),
        "known_positive_above_unlabeled_p99_rate": float(
            pu_separation["known_positive_above_unlabeled_p99_rate"]
            - raw_separation["known_positive_above_unlabeled_p99_rate"]
        ),
        "unlabeled_to_evaluation_score_ks_absolute": float(pu_ks - raw_ks),
    }
    stage_a_passes = bool(
        deltas["confirmed_pair_ap"] >= -0.01
        and deltas["worst_fold_pair_ap"] >= -0.03
        and deltas["known_positive_above_unlabeled_p95_rate"] >= 0.01
        and deltas["known_positive_above_unlabeled_p99_rate"] >= 0.0
        and deltas["unlabeled_to_evaluation_score_ks_absolute"] <= 0.02
        and abs(pu_shared_corr) <= 0.10
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["elkan_noto_oof"] = pu_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["elkan_noto_oof_score"] = pu_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc"]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["elkan_noto_score"] = pu_evaluation_scores
    receipt_paths = {
        "labelled": output_root / "E012_elkan_noto_labelled_oof.parquet",
        "unlabeled": output_root / "E012_elkan_noto_unlabeled_oof.parquet",
        "evaluation": output_root / "E012_elkan_noto_evaluation_scores.parquet",
    }
    labelled_receipt.to_parquet(receipt_paths["labelled"], index=False)
    unknown_receipt.to_parquet(receipt_paths["unlabeled"], index=False)
    evaluation_receipt.to_parquet(receipt_paths["evaluation"], index=False)
    report: dict[str, object] = {
        "experiment": "E012_nested_elkan_noto_pu_selector",
        "feature_count": int(len(features)),
        "features_changed_from_E001": 0,
        "unknown_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "unknown_temporary_role": "selection_indicator_zero_only_inside_PU_fit",
        "selection_indicator_persisted": False,
        "fold_boundary": "whole_table_equals_whole_pool",
        "c_policy": "first of three inner whole-pool folds; confirmed positives only; clip [0.05,0.95]",
        "fold_diagnostics": fold_diagnostics,
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            **raw_separation,
        },
        "elkan_noto": {
            "confirmed_pair_ap": float(pu_ap),
            "fold_pair_ap": pu_fold_ap,
            "unlabeled_to_evaluation_score_ks": pu_ks,
            "evaluation_score_shared_hands_spearman": pu_shared_corr,
            **pu_separation,
            "labelled_fraction_score_one": float(np.mean(pu_oof >= 1.0)),
            "unlabeled_fraction_score_one": float(np.mean(pu_unknown_scores >= 1.0)),
            "evaluation_fraction_score_one": float(np.mean(pu_evaluation_scores >= 1.0)),
        },
        "stage_a_deltas": deltas,
        "stage_a_gate": {
            "confirmed_pair_ap_delta_min": -0.01,
            "worst_fold_pair_ap_delta_min": -0.03,
            "known_positive_above_unlabeled_p95_rate_delta_min": 0.01,
            "known_positive_above_unlabeled_p99_rate_delta_min": 0.0,
            "unlabeled_to_evaluation_score_ks_absolute_delta_max": 0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "passes": stage_a_passes,
        },
        "time_gate_status": "required_not_run" if stage_a_passes else "not_run_stage_a_failed",
        "passes": False,
    }
    for name, path in receipt_paths.items():
        report[f"{name}_receipt_artifact"] = str(path)
        report[f"{name}_receipt_sha256"] = _sha256(path)
    report_path = output_root / "E012_elkan_noto_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_elkan_noto_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
