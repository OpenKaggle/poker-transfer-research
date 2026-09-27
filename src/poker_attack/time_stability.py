"""Early/late cross-time, whole-pool stability audit for E007 interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .features import aggregate_pair_features
from .interaction import INTERACTION_HAND_FEATURES, _aggregate_interactions
from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .schema import input_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_half_features(
    data_root: Path,
    output_root: Path,
    half: str,
) -> pd.DataFrame:
    base_path = output_root / f"E007_{half}_base_features.parquet"
    interaction_path = output_root / f"E007_{half}_interaction_features.parquet"
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    if not base_path.exists():
        pair_hands = pd.read_parquet(output_root / "development_pair_hands.parquet")
        if half == "early":
            pair_hands = pair_hands[pair_hands["phase_progress"] <= 0.5]
        elif half == "late":
            pair_hands = pair_hands[pair_hands["phase_progress"] > 0.5]
        else:
            raise ValueError(f"unknown phase half: {half}")
        players = pd.read_parquet(input_paths(data_root)["players"])
        base = aggregate_pair_features(pair_hands, labels, players)
        base.to_parquet(base_path, index=False)
    _aggregate_interactions(
        output_root / "public_pu_prepared_v2" / "dev_hand_features.parquet",
        interaction_path,
        phase_half=half,
        pair_ids=labels["pair_id"].astype(str).tolist(),
    )
    result = pd.read_parquet(base_path).merge(
        pd.read_parquet(interaction_path),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    interaction_columns = [
        column for column in result if column.startswith("interaction_")
    ]
    result[interaction_columns] = result[interaction_columns].fillna(0.0)
    return result


def run_time_stability(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    early = _build_half_features(data_root, output_root, "early").merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    late = _build_half_features(data_root, output_root, "late").merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    )
    pool_lookup = pd.read_parquet(
        output_root / "development_pair_features.parquet",
        columns=["pair_id", "table_id"],
    ).set_index("pair_id")["table_id"]
    early["table_id"] = early["table_id"].fillna(early["pair_id"].map(pool_lookup))
    late["table_id"] = late["table_id"].fillna(late["pair_id"].map(pool_lookup))
    if early["table_id"].isna().any() or late["table_id"].isna().any():
        raise ValueError("every time-split pair must retain its whole-pool boundary")
    if early["pair_id"].tolist() != late["pair_id"].tolist():
        late = late.set_index("pair_id").loc[early["pair_id"]].reset_index()
    interaction_features = [
        column for column in early if column.startswith("interaction_")
    ]
    temporal_pattern = re.compile(r"__t[0-3]$")
    base_features = [
        column
        for column in _numeric_columns(
            early.drop(columns=[*interaction_features, "table_id"], errors="ignore")
        )
        if not column.startswith("phase_progress")
        and "temporal_range" not in column
        and temporal_pattern.search(column) is None
    ]
    augmented_features = base_features + interaction_features
    early_raw = _matrix(early, base_features)
    late_raw = _matrix(late, base_features)
    early_augmented = _matrix(early, augmented_features)
    late_augmented = _matrix(late, augmented_features)
    y = early["label"].to_numpy(dtype=np.int8)
    folds = np.full(len(early), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(
        splitter.split(early, early["behavior_family"], early["table_id"])
    ):
        folds[valid] = fold

    names = (
        "raw_early_to_early",
        "raw_early_to_late",
        "raw_late_to_late",
        "raw_late_to_early",
        "interaction_early_to_early",
        "interaction_early_to_late",
        "interaction_late_to_late",
        "interaction_late_to_early",
    )
    predictions = {name: np.zeros(len(early), dtype=np.float32) for name in names}
    fold_scores = {name: [] for name in names}

    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        raw_early_model = _new_model(seed + fold)
        raw_early_model.fit(early_raw[train], y[train])
        predictions["raw_early_to_early"][valid] = raw_early_model.predict_proba(
            early_raw[valid]
        )[:, 1]
        predictions["raw_early_to_late"][valid] = raw_early_model.predict_proba(
            late_raw[valid]
        )[:, 1]

        raw_late_model = _new_model(seed + 20 + fold)
        raw_late_model.fit(late_raw[train], y[train])
        predictions["raw_late_to_late"][valid] = raw_late_model.predict_proba(
            late_raw[valid]
        )[:, 1]
        predictions["raw_late_to_early"][valid] = raw_late_model.predict_proba(
            early_raw[valid]
        )[:, 1]

        interaction_early_model = _new_model(seed + 40 + fold)
        interaction_early_model.fit(early_augmented[train], y[train])
        predictions["interaction_early_to_early"][valid] = (
            interaction_early_model.predict_proba(early_augmented[valid])[:, 1]
        )
        predictions["interaction_early_to_late"][valid] = (
            interaction_early_model.predict_proba(late_augmented[valid])[:, 1]
        )

        interaction_late_model = _new_model(seed + 60 + fold)
        interaction_late_model.fit(late_augmented[train], y[train])
        predictions["interaction_late_to_late"][valid] = (
            interaction_late_model.predict_proba(late_augmented[valid])[:, 1]
        )
        predictions["interaction_late_to_early"][valid] = (
            interaction_late_model.predict_proba(early_augmented[valid])[:, 1]
        )
        for name in names:
            fold_scores[name].append(average_precision(y[valid], predictions[name][valid]))

    scores = {name: average_precision(y, values) for name, values in predictions.items()}
    early_to_late_delta = float(
        scores["interaction_early_to_late"] - scores["raw_early_to_late"]
    )
    late_to_early_delta = float(
        scores["interaction_late_to_early"] - scores["raw_late_to_early"]
    )
    directional_gap = float(
        abs(
            scores["interaction_early_to_late"]
            - scores["interaction_late_to_early"]
        )
    )
    fold_transfer_deltas = [
        fold_scores["interaction_early_to_late"][fold]
        - fold_scores["raw_early_to_late"][fold]
        for fold in range(5)
    ] + [
        fold_scores["interaction_late_to_early"][fold]
        - fold_scores["raw_late_to_early"][fold]
        for fold in range(5)
    ]
    worst_fold_transfer_delta = float(min(fold_transfer_deltas))
    worst_interaction_transfer = float(
        min(
            scores["interaction_early_to_late"],
            scores["interaction_late_to_early"],
        )
    )
    passes = bool(
        early_to_late_delta >= -0.005
        and late_to_early_delta >= -0.005
        and worst_fold_transfer_delta >= -0.03
        and worst_interaction_transfer >= 0.90
        and directional_gap <= 0.05
    )

    receipt = early[["pair_id", "label", "label_status", "table_id"]].copy()
    receipt["fold"] = folds
    for name, values in predictions.items():
        receipt[name] = values
    receipt_path = output_root / "E007_time_stability_oof.parquet"
    receipt.to_parquet(receipt_path, index=False)
    report: dict[str, object] = {
        "experiment": "E007_early_late_time_stability",
        "base_feature_count": int(len(base_features)),
        "interaction_feature_count": int(len(interaction_features)),
        "excluded_time_fields": "phase_progress, quartile bins, temporal ranges",
        "fold_boundary": "whole_table_equals_whole_pool",
        "unknown_pairs_used_as_negative": 0,
        "scores": scores,
        "fold_scores": fold_scores,
        "deltas": {
            "early_to_late": early_to_late_delta,
            "late_to_early": late_to_early_delta,
            "worst_fold_transfer": worst_fold_transfer_delta,
            "interaction_directional_gap": directional_gap,
            "worst_interaction_transfer": worst_interaction_transfer,
        },
        "preregistered_gate": {
            "early_to_late_delta_min": -0.005,
            "late_to_early_delta_min": -0.005,
            "worst_fold_transfer_delta_min": -0.03,
            "worst_interaction_transfer_ap_min": 0.90,
            "interaction_directional_gap_max": 0.05,
            "passes": passes,
        },
        "oof_artifact": str(receipt_path),
        "oof_sha256": _sha256(receipt_path),
    }
    report_path = output_root / "E007_time_stability.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_time_stability(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
