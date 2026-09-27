"""Confirmed-only, whole-pool behavior validation audit for B001."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .schema import TARGET_BEHAVIORS, input_paths


ACTIVE_FRACTIONS = (0.005, 0.01, 0.02, 0.04, 0.05, 0.10, 0.20, 1.0)
TIE_PERMUTATIONS = 512
TIE_SEED = 9917


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_probabilities(model, matrix: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(matrix)
    result = np.zeros((len(matrix), len(TARGET_BEHAVIORS)), dtype=np.float32)
    for source, family in enumerate(model.classes_):
        family_name = str(family)
        if family_name not in TARGET_BEHAVIORS:
            raise ValueError(f"unexpected behavior class: {family_name}")
        result[:, TARGET_BEHAVIORS.index(family_name)] = raw[:, source]
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("behavior probabilities do not sum to one")
    return result


def _active_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 1.0:
        return np.ones(len(scores), dtype=bool)
    requested = max(1, int(round(len(scores) * fraction)))
    threshold = np.partition(scores, len(scores) - requested)[len(scores) - requested]
    # Include the entire boundary tie instead of using row or identifier order.
    return scores >= threshold


def _tie_views(
    y_true: np.ndarray,
    scores: np.ndarray,
    pair_ids: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(pair_ids, dtype=str)
    if int(y.sum()) == 0:
        return {
            "official_pair_id_order_ap": 0.0,
            "sklearn_grouped_tie_ap": 0.0,
            "random_tie_mean_ap": 0.0,
            "random_tie_std_ap": 0.0,
            "random_tie_min_ap": 0.0,
            "random_tie_max_ap": 0.0,
            "best_tie_ap": 0.0,
            "worst_tie_ap": 0.0,
            "zero_tie_rows": int(np.sum(values == 0)),
            "zero_tie_positive_rows": 0,
        }

    id_order = np.argsort(ids, kind="mergesort")
    official = average_precision(y[id_order], values[id_order])
    grouped = float(average_precision_score(y, values))

    score_order = np.argsort(-values, kind="mergesort")
    ordered_scores = values[score_order]
    boundaries = np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(score_order)]

    best_parts: list[np.ndarray] = []
    worst_parts: list[np.ndarray] = []
    tied_groups: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        group = score_order[start:end]
        best_parts.append(group[np.argsort(-y[group], kind="mergesort")])
        worst_parts.append(group[np.argsort(y[group], kind="mergesort")])
        if end - start > 1:
            tied_groups.append((int(start), int(end)))
    best = average_precision(y[np.concatenate(best_parts)], values[np.concatenate(best_parts)])
    worst = average_precision(y[np.concatenate(worst_parts)], values[np.concatenate(worst_parts)])

    rng = np.random.default_rng(seed)
    samples = np.zeros(TIE_PERMUTATIONS, dtype=np.float64)
    for iteration in range(TIE_PERMUTATIONS):
        randomized = score_order.copy()
        for start, end in tied_groups:
            rng.shuffle(randomized[start:end])
        ranked = y[randomized]
        positives = int(ranked.sum())
        precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
        samples[iteration] = float(np.sum(precision * ranked) / positives)

    return {
        "official_pair_id_order_ap": float(official),
        "sklearn_grouped_tie_ap": grouped,
        "random_tie_mean_ap": float(samples.mean()),
        "random_tie_std_ap": float(samples.std(ddof=0)),
        "random_tie_min_ap": float(samples.min()),
        "random_tie_max_ap": float(samples.max()),
        "best_tie_ap": float(best),
        "worst_tie_ap": float(worst),
        "zero_tie_rows": int(np.sum(values == 0)),
        "zero_tie_positive_rows": int(np.sum((values == 0) & (y == 1))),
    }


def _classification_diagnostics(
    true_family: np.ndarray,
    predicted_family: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    names = np.asarray(TARGET_BEHAVIORS, dtype=object)
    true = np.asarray(true_family, dtype=str)
    predicted = np.asarray(predicted_family, dtype=str)
    true_index = np.asarray([TARGET_BEHAVIORS.index(value) for value in true])
    one_hot = np.eye(len(TARGET_BEHAVIORS), dtype=np.float64)[true_index]
    confusion = {
        family: {
            predicted_name: int(np.sum((true == family) & (predicted == predicted_name)))
            for predicted_name in names
        }
        for family in names
    }
    recall = {
        family: float(np.mean(predicted[true == family] == family))
        for family in names
    }
    return {
        "rows": int(len(true)),
        "accuracy": float(np.mean(true == predicted)),
        "macro_recall": float(np.mean(list(recall.values()))),
        "per_family_recall": recall,
        "confusion": confusion,
        "multiclass_log_loss": float(log_loss(true_index, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def _behavior_curve(
    labels: pd.DataFrame,
    risk: np.ndarray,
    predicted_family: np.ndarray,
    fractions: tuple[float, ...],
    *,
    seed_offset: int = 0,
) -> dict[str, object]:
    pair_ids = labels["pair_id"].astype(str).to_numpy()
    true_family = labels["behavior_family"].astype(str).to_numpy()
    result: dict[str, object] = {}
    for rate_index, fraction in enumerate(fractions):
        active = _active_mask(risk, fraction)
        families: dict[str, object] = {}
        for family_index, family in enumerate(TARGET_BEHAVIORS):
            truth = (true_family == family).astype(np.int8)
            family_score = np.where(active & (predicted_family == family), risk, 0.0)
            families[family] = _tie_views(
                truth,
                family_score,
                pair_ids,
                seed=TIE_SEED + seed_offset + rate_index * 10 + family_index,
            )
        macro_keys = (
            "official_pair_id_order_ap",
            "sklearn_grouped_tie_ap",
            "random_tie_mean_ap",
            "best_tie_ap",
            "worst_tie_ap",
        )
        result[f"{fraction:.3f}"] = {
            "requested_active_fraction": fraction,
            "active_rows": int(active.sum()),
            "actual_active_fraction": float(active.mean()),
            "macro": {
                key: float(np.mean([families[family][key] for family in TARGET_BEHAVIORS]))
                for key in macro_keys
            },
            "families": families,
        }
    return result


def _fit_oof_family(
    frame: pd.DataFrame,
    features: list[str],
    folds: np.ndarray,
    *,
    seed: int,
    predict_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    source_x = _matrix(frame, features)
    target_x = source_x if predict_frame is None else _matrix(predict_frame, features)
    label = frame["label"].to_numpy(dtype=np.int8)
    family = frame["behavior_family"].astype(str).to_numpy()
    probabilities = np.zeros((len(frame), len(TARGET_BEHAVIORS)), dtype=np.float32)
    diagnostics: list[dict[str, object]] = []
    for fold in range(5):
        train = np.flatnonzero((folds != fold) & (label == 1))
        valid = np.flatnonzero(folds == fold)
        counts = {name: int(np.sum(family[train] == name)) for name in TARGET_BEHAVIORS}
        if min(counts.values()) == 0:
            raise ValueError(f"fold {fold} family training set is incomplete: {counts}")
        model = _new_model(seed + fold)
        model.fit(source_x[train], family[train])
        probabilities[valid] = _aligned_probabilities(model, target_x[valid])
        valid_positive = valid[label[valid] == 1]
        predicted = np.asarray(TARGET_BEHAVIORS, dtype=object)[
            probabilities[valid_positive].argmax(axis=1)
        ]
        diagnostics.append(
            {
                "fold": fold,
                "training_positive_counts": counts,
                "validation_rows": int(len(valid)),
                "validation_positive_rows": int(len(valid_positive)),
                "validation_positive_accuracy": float(
                    np.mean(predicted == family[valid_positive])
                ),
            }
        )
    if np.any(probabilities.sum(axis=1) == 0):
        raise ValueError("at least one OOF row has no behavior prediction")
    return probabilities, diagnostics


def _validate_feature_names(features: list[str]) -> None:
    prohibited_exact = {
        "pair_id",
        "player_1",
        "player_2",
        "hand_id",
        "table_id",
        "label",
        "behavior_family",
        "fold",
    }
    prohibited = [
        column
        for column in features
        if column in prohibited_exact or column.endswith("_id") or "row_order" in column
    ]
    if prohibited:
        raise ValueError(f"prohibited behavior features: {prohibited}")


def run_behavior_audit(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    expected_status = np.where(labels["label"].to_numpy() == 1, "confirmed_target", "confirmed_non_target")
    if not np.array_equal(labels["label_status"].astype(str).to_numpy(), expected_status):
        raise ValueError("B001 accepts only confirmed target/non-target rows")

    frozen_path = output_root / "E007_interaction_labelled_oof.parquet"
    frozen = pd.read_parquet(frozen_path)
    if _sha256(frozen_path) != "4c235116348d727980c76ab28cd15b22fc8554b39a6ffdfb67853cb11efe484d":
        raise ValueError("frozen E001 risk receipt hash changed")
    base_path = output_root / "development_pair_features.parquet"
    base = pd.read_parquet(base_path)
    frame = (
        base.merge(
            labels[["pair_id", "label", "label_status", "behavior_family"]],
            on="pair_id",
            validate="one_to_one",
        )
        .merge(
            frozen[["pair_id", "fold", "raw_oof"]],
            on="pair_id",
            validate="one_to_one",
        )
    )
    frame = frame.set_index("pair_id").loc[labels["pair_id"]].reset_index()
    folds = frame["fold"].to_numpy(dtype=np.int8)
    if set(folds) != set(range(5)):
        raise ValueError("frozen fold receipt must contain folds 0 through 4")
    pool_fold_counts = frame.groupby("table_id", sort=False)["fold"].nunique()
    if int(pool_fold_counts.max()) != 1:
        raise ValueError("a confirmed pool crosses B001 folds")

    features = _numeric_columns(
        frame.drop(columns=["table_id", "fold", "raw_oof"], errors="ignore")
    )
    _validate_feature_names(features)
    if len(features) != 95:
        raise ValueError(f"expected 95 E001 behavior features, found {len(features)}")
    probabilities, fold_diagnostics = _fit_oof_family(
        frame, features, folds, seed=seed + 300
    )
    predicted = np.asarray(TARGET_BEHAVIORS, dtype=object)[probabilities.argmax(axis=1)]
    positive = frame["label"].to_numpy(dtype=np.int8) == 1
    classification = _classification_diagnostics(
        frame.loc[positive, "behavior_family"].astype(str).to_numpy(),
        predicted[positive],
        probabilities[positive],
    )
    risk = frame["raw_oof"].to_numpy(dtype=np.float64)
    risk_ap = average_precision(frame["label"].to_numpy(dtype=np.int8), risk)
    behavior_curve = _behavior_curve(frame, risk, predicted, ACTIVE_FRACTIONS)

    receipt = frame[["pair_id", "label", "label_status", "behavior_family", "table_id", "fold", "raw_oof"]].copy()
    for index, family in enumerate(TARGET_BEHAVIORS):
        receipt[f"family_probability__{family}"] = probabilities[:, index]
    receipt["predicted_family"] = predicted
    receipt_path = output_root / "B001_behavior_oof.parquet"
    receipt.to_parquet(receipt_path, index=False)

    time_receipt_path = output_root / "E007_time_stability_oof.parquet"
    expected_time_receipt_sha256 = (
        "eb20d9cb9b35c88b704497665ac7c78ed6cffbee7d147ab399a5a132307640eb"
    )
    if _sha256(time_receipt_path) != expected_time_receipt_sha256:
        raise ValueError("frozen E007 time receipt hash changed")
    time_risk = pd.read_parquet(time_receipt_path)
    fold_comparison = frozen[["pair_id", "fold"]].merge(
        time_risk[["pair_id", "fold"]],
        on="pair_id",
        suffixes=("_full", "_time"),
        validate="one_to_one",
    )
    if len(fold_comparison) != len(frozen) or not np.array_equal(
        fold_comparison["fold_full"].to_numpy(),
        fold_comparison["fold_time"].to_numpy(),
    ):
        raise ValueError("full-period and time receipts use different pair folds")
    early_path = output_root / "E007_early_base_features.parquet"
    late_path = output_root / "E007_late_base_features.parquet"
    early = pd.read_parquet(early_path).merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    late = pd.read_parquet(late_path).merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    order = time_risk["pair_id"].astype(str).tolist()
    early = early.assign(pair_id=early["pair_id"].astype(str)).set_index("pair_id").loc[order].reset_index()
    late = late.assign(pair_id=late["pair_id"].astype(str)).set_index("pair_id").loc[order].reset_index()
    temporal_pattern = re.compile(r"__t[0-3]$")
    time_features = [
        column
        for column in _numeric_columns(early.drop(columns=["table_id"], errors="ignore"))
        if not column.startswith("phase_progress")
        and "temporal_range" not in column
        and temporal_pattern.search(column) is None
    ]
    _validate_feature_names(time_features)
    if len(time_features) != 77 or any(column not in late for column in time_features):
        raise ValueError(f"expected 77 shared time-stable fields, found {len(time_features)}")
    time_folds = time_risk["fold"].to_numpy(dtype=np.int8)
    early_to_late, early_folds = _fit_oof_family(
        early,
        time_features,
        time_folds,
        seed=seed + 400,
        predict_frame=late,
    )
    late_to_early, late_folds = _fit_oof_family(
        late,
        time_features,
        time_folds,
        seed=seed + 500,
        predict_frame=early,
    )
    time_predicted = {
        "early_to_late": np.asarray(TARGET_BEHAVIORS, dtype=object)[early_to_late.argmax(axis=1)],
        "late_to_early": np.asarray(TARGET_BEHAVIORS, dtype=object)[late_to_early.argmax(axis=1)],
    }
    time_probabilities = {
        "early_to_late": early_to_late,
        "late_to_early": late_to_early,
    }
    time_risk_columns = {
        "early_to_late": "raw_early_to_late",
        "late_to_early": "raw_late_to_early",
    }
    time_result: dict[str, object] = {}
    for direction_index, direction in enumerate(("early_to_late", "late_to_early")):
        direction_probabilities = time_probabilities[direction]
        direction_predicted = time_predicted[direction]
        direction_frame = late if direction == "early_to_late" else early
        direction_positive = direction_frame["label"].to_numpy(dtype=np.int8) == 1
        direction_risk = time_risk[time_risk_columns[direction]].to_numpy(dtype=np.float64)
        time_result[direction] = {
            "classification": _classification_diagnostics(
                direction_frame.loc[direction_positive, "behavior_family"].astype(str).to_numpy(),
                direction_predicted[direction_positive],
                direction_probabilities[direction_positive],
            ),
            "behavior_curve": _behavior_curve(
                direction_frame,
                direction_risk,
                direction_predicted,
                (0.04, 1.0),
                seed_offset=1000 + direction_index * 100,
            ),
            "fold_diagnostics": early_folds if direction == "early_to_late" else late_folds,
        }

    time_output = time_risk[["pair_id", "label", "label_status", "table_id", "fold", "raw_early_to_late", "raw_late_to_early"]].copy()
    time_output = time_output.merge(
        labels[["pair_id", "behavior_family"]], on="pair_id", validate="one_to_one"
    )
    for direction in ("early_to_late", "late_to_early"):
        for index, family in enumerate(TARGET_BEHAVIORS):
            time_output[f"{direction}__family_probability__{family}"] = time_probabilities[direction][:, index]
        time_output[f"{direction}__predicted_family"] = time_predicted[direction]
    time_output_path = output_root / "B001_behavior_time_oof.parquet"
    time_output.to_parquet(time_output_path, index=False)

    pool_universe_path = output_root / "public_pu_prepared_v2" / "dev_pairs.parquet"
    pool_universe = pd.read_parquet(
        pool_universe_path, columns=["table_id", "is_labeled"]
    )
    total_gameplay_pools = int(pool_universe["table_id"].nunique())
    confirmed_pools = int(
        pool_universe.loc[pool_universe["is_labeled"].astype(bool), "table_id"].nunique()
    )
    if confirmed_pools != int(frame["table_id"].nunique()):
        raise ValueError("confirmed pool count disagrees with the sealed label frame")

    report: dict[str, object] = {
        "audit": "B001_exact_behavior_baseline",
        "status": "diagnostic_only_with_three_unlabelled_pool_coverage_gap",
        "fold_boundary": "whole_table_equals_whole_pool",
        "confirmed_rows": int(len(frame)),
        "confirmed_positive_rows": int(positive.sum()),
        "total_gameplay_pools": total_gameplay_pools,
        "confirmed_pools_observed": confirmed_pools,
        "pools_without_confirmed_rows": total_gameplay_pools - confirmed_pools,
        "observed_pool_fold_integrity_passed": bool(pool_fold_counts.eq(1).all()),
        "literal_400_pool_mapping_requirement_met": confirmed_pools == total_gameplay_pools,
        "unknown_rows_used": 0,
        "unknown_ground_truth_labels_assigned": 0,
        "pair_risk_changed": False,
        "evidence_changed": False,
        "feature_count": int(len(features)),
        "feature_columns": features,
        "prohibited_feature_columns": [],
        "risk_oof_pair_ap": float(risk_ap),
        "risk_unique_scores": int(pd.Series(risk).nunique()),
        "classification": classification,
        "fold_diagnostics": fold_diagnostics,
        "behavior_curve": behavior_curve,
        "time_feature_count": int(len(time_features)),
        "time_feature_columns": time_features,
        "time": time_result,
        "tie_policy": {
            "official": "sort pair_id lexicographically, then stable descending score exactly as organizer notebook",
            "selection_safe_diagnostic": "mean AP over 512 deterministic random permutations inside every exact score tie",
            "base_random_seed": TIE_SEED,
            "full_period_seed_formula": "9917 + rate_index * 10 + family_index",
            "time_seed_formula": "9917 + (1000 for early_to_late or 1100 for late_to_early) + rate_index * 10 + family_index",
            "permutations": TIE_PERMUTATIONS,
            "identifier_order_may_select_models": False,
        },
        "controls": {
            "active_fraction_grid_is_descriptive_only": True,
            "e013_triggered": False,
            "evaluation_scored": False,
            "submission_allowed": False,
        },
        "sources": {
            "official_metric_notebook": "poker/reference/official_metric/slash-poker-competition-metric.ipynb",
            "official_metric_notebook_sha256": _sha256(
                Path("poker/reference/official_metric/slash-poker-competition-metric.ipynb")
            ),
            "frozen_risk_receipt": str(frozen_path),
            "frozen_risk_receipt_sha256": _sha256(frozen_path),
            "development_features": str(base_path),
            "development_features_sha256": _sha256(base_path),
            "early_features_sha256": _sha256(early_path),
            "late_features_sha256": _sha256(late_path),
            "time_risk_receipt_sha256": _sha256(time_receipt_path),
            "full_and_time_pair_folds_match": True,
            "pool_universe_sha256": _sha256(pool_universe_path),
        },
        "artifacts": {
            "oof_receipt": str(receipt_path),
            "oof_receipt_sha256": _sha256(receipt_path),
            "time_oof_receipt": str(time_output_path),
            "time_oof_receipt_sha256": _sha256(time_output_path),
        },
    }
    report_path = output_root / "B001_behavior_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_behavior_audit(arguments.data_dir, arguments.work_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
