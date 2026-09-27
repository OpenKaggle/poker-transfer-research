"""Strict positive-unlabelled preparation and whole-pool validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .features import aggregate_pair_features, build_pair_hands
from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .schema import input_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_key(player_1: object, player_2: object) -> frozenset[str]:
    return frozenset((str(player_1), str(player_2)))


def build_unknown_features(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    minimum_development_hands: int = 57,
    force: bool = False,
) -> dict[str, object]:
    """Aggregate every exposure-matched unknown pair one pool at a time.

    Output tables contain an explicit ``unlabeled`` role and deliberately have
    no ``label`` column. Synthetic pair IDs are join keys only and are excluded
    from all model matrices.
    """

    data_root = Path(data_dir)
    output_root = Path(work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_path = output_root / "pu_unknown_pairs.parquet"
    feature_path = output_root / "pu_unknown_features.parquet"
    manifest_path = output_root / "pu_unknown_manifest.json"
    if not force and pair_path.exists() and feature_path.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    paths = input_paths(data_root)
    labels = pd.read_csv(paths["development_labels"])
    players = pd.read_parquet(paths["players"])
    known_keys = {
        _member_key(row.player_1, row.player_2)
        for row in labels[["player_1", "player_2"]].itertuples(index=False)
    }
    player_hands = pd.read_parquet(
        output_root / "player_hands.parquet",
        filters=[("phase", "==", "development")],
    )
    if set(player_hands["phase"].astype(str)) != {"development"}:
        raise ValueError("PU preparation must read development hands only")

    unknown_pair_frames: list[pd.DataFrame] = []
    unknown_feature_frames: list[pd.DataFrame] = []
    all_pair_count = 0
    serial = 0

    for table_id, table_frame in player_hands.groupby("table_id", sort=False):
        members = pd.unique(table_frame["player_id"].astype(str))
        if len(members) != 30:
            raise ValueError(f"table {table_id} has {len(members)} players rather than 30")
        rows = []
        for player_1, player_2 in combinations(members, 2):
            rows.append(
                {
                    "pair_id": f"PU{serial:07d}",
                    "player_1": player_1,
                    "player_2": player_2,
                }
            )
            serial += 1
        all_pairs = pd.DataFrame(rows)
        all_pair_count += len(all_pairs)
        pair_hands = build_pair_hands(table_frame, all_pairs, "development")
        observed = pair_hands.groupby("pair_id", sort=False).size().rename("shared_hands")
        all_pairs = all_pairs.join(observed, on="pair_id", how="left")
        all_pairs["shared_hands"] = all_pairs["shared_hands"].fillna(0).astype(int)
        all_pairs["is_confirmed"] = [
            _member_key(a, b) in known_keys
            for a, b in zip(all_pairs["player_1"], all_pairs["player_2"])
        ]
        unknown_pairs = all_pairs[
            (~all_pairs["is_confirmed"])
            & (all_pairs["shared_hands"] >= minimum_development_hands)
        ].drop(columns="is_confirmed")
        if unknown_pairs.empty:
            continue
        unknown_hands = pair_hands[pair_hands["pair_id"].isin(unknown_pairs["pair_id"])]
        unknown_features = aggregate_pair_features(unknown_hands, unknown_pairs, players)
        unknown_pairs = unknown_pairs.assign(
            table_id=str(table_id),
            label_status="unlabeled",
            training_role="unlabeled",
        )
        unknown_feature_frames.append(unknown_features)
        unknown_pair_frames.append(unknown_pairs)

    unknown_pairs = pd.concat(unknown_pair_frames, ignore_index=True)
    unknown_features = pd.concat(unknown_feature_frames, ignore_index=True)
    if "label" in unknown_pairs or "label" in unknown_features:
        raise ValueError("unknown artifacts must never contain a label column")
    if set(unknown_pairs["label_status"]) != {"unlabeled"}:
        raise ValueError("unknown label status was mutated")
    if unknown_pairs["pair_id"].duplicated().any():
        raise ValueError("synthetic unknown pair IDs must be unique")
    feature_keys = set(unknown_features["pair_id"].astype(str))
    pair_keys = set(unknown_pairs["pair_id"].astype(str))
    if feature_keys != pair_keys:
        raise ValueError("unknown feature coverage does not match unknown pair coverage")
    overlap = sum(
        _member_key(row.player_1, row.player_2) in known_keys
        for row in unknown_pairs[["player_1", "player_2"]].itertuples(index=False)
    )
    if overlap:
        raise ValueError(f"unknown sample overlaps {overlap} confirmed pairs")

    unknown_pairs.to_parquet(pair_path, index=False)
    unknown_features.to_parquet(feature_path, index=False)
    counts = unknown_pairs.groupby("table_id", sort=False).size()
    manifest: dict[str, object] = {
        "all_development_pairs": int(all_pair_count),
        "confirmed_pairs_excluded": int(len(labels)),
        "minimum_development_hands": int(minimum_development_hands),
        "evaluation_equivalent_minimum_hands": float(minimum_development_hands / 1.5),
        "unknown_pairs": int(len(unknown_pairs)),
        "unknown_features": int(len(unknown_features)),
        "unknown_tables": int(unknown_pairs["table_id"].nunique()),
        "unknown_pairs_per_table_min": int(counts.min()),
        "unknown_pairs_per_table_median": float(counts.median()),
        "unknown_pairs_per_table_max": int(counts.max()),
        "unknown_confirmed_overlap": int(overlap),
        "unknown_label_column_present": False,
        "unknown_role": "unlabeled",
        "pair_id_is_feature": False,
        "pair_artifact": str(pair_path),
        "feature_artifact": str(feature_path),
    }
    manifest["pair_sha256"] = _sha256(pair_path)
    manifest["feature_sha256"] = _sha256(feature_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _assign_folds(labelled: pd.DataFrame, unknown: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labelled_folds = np.full(len(labelled), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    group_to_fold: dict[str, int] = {}
    for fold, (_, valid) in enumerate(
        splitter.split(
            labelled,
            labelled["behavior_family"].astype(str),
            labelled["table_id"].astype(str),
        )
    ):
        labelled_folds[valid] = fold
        group_to_fold.update(
            {str(group): fold for group in labelled.iloc[valid]["table_id"].unique()}
        )
    missing_groups = sorted(set(unknown["table_id"].astype(str)) - set(group_to_fold))
    fold_sizes = np.bincount(labelled_folds, minlength=5)
    for group in missing_groups:
        fold = int(np.argmin(fold_sizes))
        group_to_fold[group] = fold
        fold_sizes[fold] += int((unknown["table_id"].astype(str) == group).sum())
    unknown_folds = unknown["table_id"].astype(str).map(group_to_fold).to_numpy(dtype=np.int8)
    if (labelled_folds < 0).any() or pd.isna(unknown_folds).any():
        raise ValueError("not every pair received a sealed pool fold")
    return labelled_folds, unknown_folds


def _positive_percentiles(
    labelled: pd.DataFrame,
    labelled_scores: np.ndarray,
    unknown: pd.DataFrame,
    unknown_scores: np.ndarray,
) -> dict[str, float]:
    unknown_by_table = {
        str(table): scores
        for table, scores in pd.DataFrame(
            {"table_id": unknown["table_id"].astype(str), "score": unknown_scores}
        ).groupby("table_id", sort=False)["score"]
    }
    percentiles: list[float] = []
    positive_mask = labelled["label"].to_numpy(dtype=int) == 1
    for table, score in zip(
        labelled.loc[positive_mask, "table_id"].astype(str),
        labelled_scores[positive_mask],
    ):
        pool_scores = unknown_by_table.get(str(table))
        if pool_scores is None or len(pool_scores) == 0:
            continue
        values = np.asarray(pool_scores, dtype=float)
        percentiles.append(float((np.sum(values < score) + 0.5 * np.sum(values == score)) / len(values)))
    array = np.asarray(percentiles, dtype=float)
    return {
        "known_positive_vs_unlabeled_percentile_mean": float(array.mean()),
        "known_positive_vs_unlabeled_percentile_median": float(np.median(array)),
        "known_positive_above_unlabeled_p95_rate": float(np.mean(array >= 0.95)),
        "known_positive_above_unlabeled_p99_rate": float(np.mean(array >= 0.99)),
    }


def validate_bagged_pu(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
    bags_per_fold: int = 5,
) -> dict[str, object]:
    """Compare E001 to one bagged-PU change on identical sealed pool folds."""

    data_root = Path(data_dir)
    output_root = Path(work_dir)
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    labelled = pd.read_parquet(output_root / "development_pair_features.parquet").merge(
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
    if "label" in unknown or set(unknown["training_role"]) != {"unlabeled"}:
        raise ValueError("PU input must keep unknown targets latent")
    feature_columns = _numeric_columns(labelled.drop(columns=["table_id"], errors="ignore"))
    labelled_x = _matrix(labelled, feature_columns)
    unknown_x = _matrix(unknown, feature_columns)
    labelled_y = labelled["label"].to_numpy(dtype=np.int8)
    labelled_folds, unknown_folds = _assign_folds(labelled, unknown, seed)

    baseline_oof = np.zeros(len(labelled), dtype=np.float32)
    pu_oof = np.zeros(len(labelled), dtype=np.float32)
    baseline_unknown = np.zeros(len(unknown), dtype=np.float32)
    pu_unknown = np.zeros(len(unknown), dtype=np.float32)
    baseline_fold_ap: list[float] = []
    pu_fold_ap: list[float] = []
    rng = np.random.default_rng(seed)

    for fold in range(5):
        train = np.flatnonzero(labelled_folds != fold)
        valid = np.flatnonzero(labelled_folds == fold)
        unknown_train = np.flatnonzero(unknown_folds != fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)

        baseline = _new_model(seed + fold)
        baseline.fit(labelled_x[train], labelled_y[train])
        baseline_oof[valid] = baseline.predict_proba(labelled_x[valid])[:, 1]
        baseline_unknown[unknown_valid] = baseline.predict_proba(unknown_x[unknown_valid])[:, 1]

        positive_train = train[labelled_y[train] == 1]
        negative_train = train[labelled_y[train] == 0]
        unlabeled_sample_size = min(
            len(unknown_train), max(len(negative_train), 4 * len(positive_train))
        )
        fold_valid_scores = np.zeros((len(valid), bags_per_fold), dtype=np.float32)
        fold_unknown_scores = np.zeros((len(unknown_valid), bags_per_fold), dtype=np.float32)
        for bag in range(bags_per_fold):
            sampled_unknown = rng.choice(
                unknown_train, size=unlabeled_sample_size, replace=False
            )
            train_x = np.concatenate(
                [
                    labelled_x[positive_train],
                    labelled_x[negative_train],
                    unknown_x[sampled_unknown],
                ],
                axis=0,
            )
            # The zero below is a PU surrogate used by the biased learner, not
            # a persisted or asserted ground-truth negative label.
            train_y = np.r_[
                np.ones(len(positive_train), dtype=np.int8),
                np.zeros(len(negative_train) + len(sampled_unknown), dtype=np.int8),
            ]
            train_weight = np.r_[
                np.ones(len(positive_train), dtype=np.float32),
                np.ones(len(negative_train), dtype=np.float32),
                np.full(len(sampled_unknown), 0.35, dtype=np.float32),
            ]
            model = _new_model(seed + 100 + fold * bags_per_fold + bag)
            model.fit(train_x, train_y, sample_weight=train_weight)
            fold_valid_scores[:, bag] = model.predict_proba(labelled_x[valid])[:, 1]
            fold_unknown_scores[:, bag] = model.predict_proba(unknown_x[unknown_valid])[:, 1]
        pu_oof[valid] = fold_valid_scores.mean(axis=1)
        pu_unknown[unknown_valid] = fold_unknown_scores.mean(axis=1)
        baseline_fold_ap.append(average_precision(labelled_y[valid], baseline_oof[valid]))
        pu_fold_ap.append(average_precision(labelled_y[valid], pu_oof[valid]))

    baseline_ap = average_precision(labelled_y, baseline_oof)
    pu_ap = average_precision(labelled_y, pu_oof)
    baseline_separation = _positive_percentiles(
        labelled, baseline_oof, unknown, baseline_unknown
    )
    pu_separation = _positive_percentiles(labelled, pu_oof, unknown, pu_unknown)
    confirmed_ap_delta = float(pu_ap - baseline_ap)
    percentile_delta = float(
        pu_separation["known_positive_vs_unlabeled_percentile_median"]
        - baseline_separation["known_positive_vs_unlabeled_percentile_median"]
    )
    worst_fold_delta = float(min(pu_fold_ap) - min(baseline_fold_ap))
    passes = bool(
        confirmed_ap_delta >= -0.01
        and percentile_delta >= 0.02
        and worst_fold_delta >= -0.03
    )

    labelled_receipt = labelled[["pair_id", "label", "label_status", "table_id"]].copy()
    labelled_receipt["fold"] = labelled_folds
    labelled_receipt["baseline_oof"] = baseline_oof
    labelled_receipt["bagged_pu_oof"] = pu_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id"]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["baseline_oof_score"] = baseline_unknown
    unknown_receipt["bagged_pu_oof_score"] = pu_unknown
    labelled_path = output_root / "pu_E003_labelled_oof.parquet"
    unknown_path = output_root / "pu_E003_unlabeled_oof.parquet"
    labelled_receipt.to_parquet(labelled_path, index=False)
    unknown_receipt.to_parquet(unknown_path, index=False)

    report: dict[str, object] = {
        "experiment": "E003_bagged_pu_only",
        "feature_count": int(len(feature_columns)),
        "fold_boundary": "whole_table_equals_whole_pool",
        "bags_per_fold": int(bags_per_fold),
        "confirmed_pairs": int(len(labelled)),
        "confirmed_positive_pairs": int(labelled_y.sum()),
        "unlabeled_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "baseline": {
            "confirmed_pair_ap": float(baseline_ap),
            "confirmed_brier": float(np.mean((baseline_oof - labelled_y) ** 2)),
            "fold_pair_ap": baseline_fold_ap,
            **baseline_separation,
        },
        "bagged_pu": {
            "confirmed_pair_ap": float(pu_ap),
            "confirmed_brier": float(np.mean((pu_oof - labelled_y) ** 2)),
            "fold_pair_ap": pu_fold_ap,
            **pu_separation,
        },
        "deltas": {
            "confirmed_pair_ap": confirmed_ap_delta,
            "known_positive_vs_unlabeled_percentile_median": percentile_delta,
            "worst_fold_pair_ap": worst_fold_delta,
        },
        "preregistered_gate": {
            "confirmed_pair_ap_delta_min": -0.01,
            "known_positive_vs_unlabeled_percentile_median_delta_min": 0.02,
            "worst_fold_pair_ap_delta_min": -0.03,
            "passes": passes,
        },
        "labelled_oof_artifact": str(labelled_path),
        "unlabeled_oof_artifact": str(unknown_path),
    }
    report["labelled_oof_sha256"] = _sha256(labelled_path)
    report["unlabeled_oof_sha256"] = _sha256(unknown_path)
    report_path = output_root / "pu_E003_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    parser.add_argument("--build-unknown", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bags-per-fold", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, object] = {}
    if args.build_unknown:
        result["unknown"] = build_unknown_features(
            args.data_dir, args.work_dir, force=args.force
        )
    if args.validate:
        result["validation"] = validate_bagged_pu(
            args.data_dir, args.work_dir, bags_per_fold=args.bags_per_fold
        )
    if not result:
        raise SystemExit("choose --build-unknown and/or --validate")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
