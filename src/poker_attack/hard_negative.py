"""Nested trusted hard-negative weighting with a staged promotion gate."""

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
from .pairwise import _aligned_baseline, _load_frames, _time_frame
from .pu import _assign_folds, _positive_percentiles
from .schema import input_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_hard_negative(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *targets: np.ndarray,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    pilot = _new_model(seed)
    pilot.fit(train_x, train_y)
    pilot_scores = pilot.predict_proba(train_x)[:, 1]
    negative = train_y == 0
    threshold = float(np.quantile(pilot_scores[negative], 0.75))
    hard = negative & (pilot_scores >= threshold)
    weights = np.ones(len(train_y), dtype=np.float64)
    weights[negative] = 0.5
    weights[hard] = 2.5
    final = _new_model(seed)
    final.fit(train_x, train_y, sample_weight=weights)
    diagnostics: dict[str, float | int] = {
        "hard_threshold": threshold,
        "hard_negative_count": int(hard.sum()),
        "negative_count": int(negative.sum()),
        "hard_negative_fraction": float(hard.sum() / negative.sum()),
        "negative_weight_sum": float(weights[negative].sum()),
        "positive_weight_sum": float(weights[~negative].sum()),
    }
    predictions = [final.predict_proba(target)[:, 1].astype(np.float32) for target in targets]
    return predictions, diagnostics


def run_hard_negative_validation(
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

    hard_oof = np.zeros(len(confirmed), dtype=np.float32)
    hard_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    hard_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    hard_fold_ap: list[float] = []
    mining_diagnostics: list[dict[str, float | int]] = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        predictions, diagnostics = _fit_hard_negative(
            confirmed_x[train],
            y[train],
            confirmed_x[valid],
            unknown_x[unknown_valid],
            evaluation_x,
            seed=seed + fold,
        )
        valid_score, unknown_score, eval_score = predictions
        hard_oof[valid] = valid_score
        hard_unknown_scores[unknown_valid] = unknown_score
        hard_evaluation_scores += eval_score / 5
        diagnostics["fold"] = fold
        mining_diagnostics.append(diagnostics)
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        hard_fold_ap.append(average_precision(y[valid], hard_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    hard_ap = average_precision(y, hard_oof)
    raw_separation = _positive_percentiles(confirmed, raw_oof, unknown, raw_unknown_scores)
    hard_separation = _positive_percentiles(
        confirmed, hard_oof, unknown, hard_unknown_scores
    )
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    hard_ks = float(ks_2samp(hard_unknown_scores, hard_evaluation_scores).statistic)
    hard_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], hard_evaluation_scores).statistic
    )
    stage_a_deltas = {
        "confirmed_pair_ap": float(hard_ap - raw_ap),
        "worst_fold_pair_ap": float(min(hard_fold_ap) - min(raw_fold_ap)),
        "known_positive_above_unlabeled_p95_rate": float(
            hard_separation["known_positive_above_unlabeled_p95_rate"]
            - raw_separation["known_positive_above_unlabeled_p95_rate"]
        ),
        "unlabeled_to_evaluation_score_ks_absolute": float(hard_ks - raw_ks),
    }
    stage_a_passes = bool(
        stage_a_deltas["confirmed_pair_ap"] >= 0.003
        and stage_a_deltas["worst_fold_pair_ap"] >= -0.02
        and stage_a_deltas["known_positive_above_unlabeled_p95_rate"] >= 0.005
        and stage_a_deltas["unlabeled_to_evaluation_score_ks_absolute"] <= 0.02
        and abs(hard_shared_corr) <= 0.10
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["hard_negative_oof"] = hard_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["hard_negative_oof_score"] = hard_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc"]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["hard_negative_score"] = hard_evaluation_scores
    receipt_paths = {
        "labelled": output_root / "E011_hard_negative_labelled_oof.parquet",
        "unlabeled": output_root / "E011_hard_negative_unlabeled_oof.parquet",
        "evaluation": output_root / "E011_hard_negative_evaluation_scores.parquet",
    }
    labelled_receipt.to_parquet(receipt_paths["labelled"], index=False)
    unknown_receipt.to_parquet(receipt_paths["unlabeled"], index=False)
    evaluation_receipt.to_parquet(receipt_paths["evaluation"], index=False)
    report: dict[str, object] = {
        "experiment": "E011_nested_trusted_hard_negative_weighting",
        "feature_count": int(len(features)),
        "features_changed_from_E001": 0,
        "unknown_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "fold_boundary": "whole_table_equals_whole_pool",
        "hard_negative_policy": {
            "pilot_scope": "outer-fold confirmed training pools only",
            "hard_quantile": 0.75,
            "hard_negative_weight": 2.5,
            "easy_negative_weight": 0.5,
            "positive_weight": 1.0,
        },
        "mining_diagnostics": mining_diagnostics,
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            **raw_separation,
        },
        "hard_negative": {
            "confirmed_pair_ap": float(hard_ap),
            "fold_pair_ap": hard_fold_ap,
            "unlabeled_to_evaluation_score_ks": hard_ks,
            "evaluation_score_shared_hands_spearman": hard_shared_corr,
            **hard_separation,
        },
        "stage_a_deltas": stage_a_deltas,
        "stage_a_gate": {
            "confirmed_pair_ap_delta_min": 0.003,
            "worst_fold_pair_ap_delta_min": -0.02,
            "known_positive_above_unlabeled_p95_rate_delta_min": 0.005,
            "unlabeled_to_evaluation_score_ks_absolute_delta_max": 0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "passes": stage_a_passes,
        },
        "time_gate_status": "required_if_stage_a_passes" if stage_a_passes else "not_run_stage_a_failed",
    }
    for name, path in receipt_paths.items():
        report[f"{name}_receipt_artifact"] = str(path)
        report[f"{name}_receipt_sha256"] = _sha256(path)

    if stage_a_passes:
        labels = pd.read_csv(input_paths(data_root)["development_labels"])
        order = confirmed["pair_id"].astype(str).tolist()
        pool_lookup = confirmed.set_index("pair_id")["table_id"]
        early = _time_frame(output_root, labels, "early", pool_lookup, order)
        late = _time_frame(output_root, labels, "late", pool_lookup, order)
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
            raise ValueError("cached time folds differ from E011 folds")
        time_predictions = {
            "raw_early_to_late": base_time["raw_early_to_late"].to_numpy(dtype=np.float32),
            "raw_late_to_early": base_time["raw_late_to_early"].to_numpy(dtype=np.float32),
            "hard_early_to_late": np.zeros(len(early), dtype=np.float32),
            "hard_late_to_early": np.zeros(len(early), dtype=np.float32),
        }
        time_fold_scores = {name: [] for name in time_predictions}
        time_mining: list[dict[str, float | int | str]] = []
        for fold in range(5):
            train = np.flatnonzero(folds != fold)
            valid = np.flatnonzero(folds == fold)
            early_predictions, early_diagnostics = _fit_hard_negative(
                early_x[train], y[train], late_x[valid], seed=seed + 200 + fold
            )
            late_predictions, late_diagnostics = _fit_hard_negative(
                late_x[train], y[train], early_x[valid], seed=seed + 220 + fold
            )
            time_predictions["hard_early_to_late"][valid] = early_predictions[0]
            time_predictions["hard_late_to_early"][valid] = late_predictions[0]
            early_diagnostics.update({"fold": fold, "training_half": "early"})
            late_diagnostics.update({"fold": fold, "training_half": "late"})
            time_mining.extend([early_diagnostics, late_diagnostics])
            for name, values in time_predictions.items():
                time_fold_scores[name].append(average_precision(y[valid], values[valid]))
        time_scores = {
            name: average_precision(y, values) for name, values in time_predictions.items()
        }
        early_late_delta = float(
            time_scores["hard_early_to_late"] - time_scores["raw_early_to_late"]
        )
        late_early_delta = float(
            time_scores["hard_late_to_early"] - time_scores["raw_late_to_early"]
        )
        time_fold_deltas = [
            time_fold_scores["hard_early_to_late"][fold]
            - time_fold_scores["raw_early_to_late"][fold]
            for fold in range(5)
        ] + [
            time_fold_scores["hard_late_to_early"][fold]
            - time_fold_scores["raw_late_to_early"][fold]
            for fold in range(5)
        ]
        directional_gap = float(
            abs(time_scores["hard_early_to_late"] - time_scores["hard_late_to_early"])
        )
        worst_transfer = float(
            min(time_scores["hard_early_to_late"], time_scores["hard_late_to_early"])
        )
        time_passes = bool(
            early_late_delta >= 0.01
            and late_early_delta >= 0.01
            and min(time_fold_deltas) >= -0.02
            and worst_transfer >= 0.84
            and directional_gap <= 0.05
        )
        time_receipt = early[["pair_id", "label", "label_status", "table_id"]].copy()
        time_receipt["fold"] = folds
        for name, values in time_predictions.items():
            time_receipt[name] = values
        time_path = output_root / "E011_hard_negative_time_oof.parquet"
        time_receipt.to_parquet(time_path, index=False)
        report.update(
            {
                "time_gate_status": "passed" if time_passes else "failed",
                "time_scores": time_scores,
                "time_fold_scores": time_fold_scores,
                "time_mining_diagnostics": time_mining,
                "time_deltas": {
                    "early_to_late": early_late_delta,
                    "late_to_early": late_early_delta,
                    "worst_time_fold": float(min(time_fold_deltas)),
                    "worst_hard_negative_transfer": worst_transfer,
                    "directional_gap": directional_gap,
                },
                "time_gate": {
                    "early_to_late_delta_min": 0.01,
                    "late_to_early_delta_min": 0.01,
                    "worst_time_fold_delta_min": -0.02,
                    "worst_hard_negative_transfer_ap_min": 0.84,
                    "directional_gap_max": 0.05,
                    "passes": time_passes,
                },
                "time_receipt_artifact": str(time_path),
                "time_receipt_sha256": _sha256(time_path),
            }
        )
    report["passes"] = bool(stage_a_passes and report.get("time_gate_status") == "passed")
    report_path = output_root / "E011_hard_negative_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_hard_negative_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
