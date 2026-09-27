"""Receipt-only feasibility audit for strict joint evidence validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .schema import TARGET_BEHAVIORS, input_paths


B001_FULL_SHA256 = "71d7ce45170267060f7a1a6ae3e4a78a425d37bc4b4d88d392d8fe08b8baee36"
B001_TIME_SHA256 = "4f85a92ab6e72f3c71dd4d14b7f5ba57e621bd87c08a41955576d7d612273147"
B001_IMPLEMENTATION_SHA256 = "4cae28dba762ba301bb58d0c75fa5533bd8d5e979a52764732b506e831a87b49"
B001_FULL_FEATURES_SHA256 = "756d2e1e1b9d031755534ff975023a5f958ab36c53ec342b71c9f03f6448dc8b"
B001_EARLY_FEATURES_SHA256 = "0622e8f2b59cf1fb7b9109c4b9e536d5f5773d0bdee630b361134d9116db471d"
B001_LATE_FEATURES_SHA256 = "943fbf666fc19ac2c1fb086e7c32f94df4ead0dcc8b65e85b9feade85cd22e1a"
PAIR_HANDS_SHA256 = "cf197d2bf8b8242b83515b6db1a5a0c73c214ed91fc46fce9205014f07d265af"
HANDS_SHA256 = "82c9ad1c01cae6a68b28e194b9dd0bad9ca09f94628b429c0e58e970cc29eecd"
OFFICIAL_METRIC_SHA256 = "3cb11be5c999ada91aa91002f18aabc0c1c48548f1b6aedf3939f31fcd511c5c"
H0_SIGNALS = ("directed_signal", "soft_signal", "isolation_signal")
PREREG_FEATURES = (
    "pair_contribution_bb",
    "contribution_gap_bb",
    "net_gap_bb",
    "max_win_bb",
    "max_loss_bb",
    "transfer_1_to_2_bb",
    "transfer_2_to_1_bb",
    "transfer_any_bb",
    "both_showdown",
    "one_folded",
    "pair_aggression",
    "pair_passive",
    "expected_aggression",
    "expected_passive",
    "aggression_residual",
    "passive_residual",
    "pair_pot_share",
    "pair_raises",
    "pair_folds",
    "max_hole_strength",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    values = frame[column].astype(str).value_counts()
    return {family: int(values.get(family, 0)) for family in TARGET_BEHAVIORS}


def _nested_inner_summary(positives: pd.DataFrame, seed: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for outer_fold in range(5):
        outer_train = positives[positives["fold"] != outer_fold].reset_index(drop=True)
        splitter = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed + outer_fold,
        )
        inner_rows: list[dict[str, object]] = []
        assigned = np.full(len(outer_train), -1, dtype=np.int8)
        for inner_fold, (train_index, valid_index) in enumerate(
            splitter.split(
                outer_train,
                outer_train["behavior_family"],
                outer_train["table_id"],
            )
        ):
            assigned[valid_index] = inner_fold
            train_pools = set(outer_train.iloc[train_index]["table_id"].astype(str))
            valid_pools = set(outer_train.iloc[valid_index]["table_id"].astype(str))
            if train_pools & valid_pools:
                raise ValueError("an inner evidence fold splits a pool")
            inner_rows.append(
                {
                    "inner_fold": inner_fold,
                    "training_pairs": int(len(train_index)),
                    "validation_pairs": int(len(valid_index)),
                    "training_family_counts": _counts(
                        outer_train.iloc[train_index], "behavior_family"
                    ),
                    "validation_family_counts": _counts(
                        outer_train.iloc[valid_index], "behavior_family"
                    ),
                    "training_pools": int(len(train_pools)),
                    "validation_pools": int(len(valid_pools)),
                }
            )
        if np.any(assigned < 0):
            raise ValueError("an outer-training pair lacks an inner fold")
        if int(
            pd.DataFrame(
                {"table_id": outer_train["table_id"], "inner_fold": assigned}
            )
            .groupby("table_id")["inner_fold"]
            .nunique()
            .max()
        ) != 1:
            raise ValueError("an outer-training pool maps to multiple inner folds")
        result.append(
            {
                "outer_fold": outer_fold,
                "outer_training_pairs": int(len(outer_train)),
                "outer_validation_pairs": int((positives["fold"] == outer_fold).sum()),
                "inner_folds": inner_rows,
            }
        )
    return result


def _family_fold_summary(pair_stats: pd.DataFrame) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for family in TARGET_BEHAVIORS:
        for fold in range(5):
            part = pair_stats[
                (pair_stats["behavior_family"] == family) & (pair_stats["fold"] == fold)
            ]
            result.append(
                {
                    "behavior_family": family,
                    "outer_fold": fold,
                    "pairs": int(len(part)),
                    "candidate_hands": int(part["candidate_hands"].sum()),
                    "evidence_hands": int(part["evidence_hands"].sum()),
                    "pairs_with_early_evidence": int((part["early_evidence"] > 0).sum()),
                    "early_candidate_hands": int(part["early_candidate_hands"].sum()),
                    "early_evidence_hands": int(part["early_evidence"].sum()),
                    "pairs_with_late_evidence": int((part["late_evidence"] > 0).sum()),
                    "late_candidate_hands": int(part["late_candidate_hands"].sum()),
                    "late_evidence_hands": int(part["late_evidence"].sum()),
                }
            )
    return result


def _pairwise_scope_summary(
    pair_stats: pd.DataFrame,
    scope: str,
) -> dict[str, object]:
    candidate_column = "candidate_hands" if scope == "full" else f"{scope}_candidate_hands"
    evidence_column = "evidence_hands" if scope == "full" else f"{scope}_evidence"
    unjudged_column = f"{scope}_unjudged"
    preference_column = f"{scope}_pairwise_preferences"
    eligible = pair_stats[pair_stats[evidence_column] > 0]
    family_rows: dict[str, object] = {}
    for family in TARGET_BEHAVIORS:
        part = eligible[eligible["behavior_family"] == family]
        family_rows[family] = {
            "evidence_bearing_pairs": int(len(part)),
            "pairwise_trainable_pairs": int((part[unjudged_column] > 0).sum()),
            "candidate_hands_minimum": int(part[candidate_column].min()),
            "unjudged_hands_minimum": int(part[unjudged_column].min()),
            "pairwise_preferences": int(part[preference_column].sum()),
        }
    return {
        "evidence_bearing_pairs": int(len(eligible)),
        "pairwise_trainable_pairs": int((eligible[unjudged_column] > 0).sum()),
        "zero_preference_pairs": int((eligible[unjudged_column] == 0).sum()),
        "candidate_hands_minimum": int(eligible[candidate_column].min()),
        "candidate_hands_maximum": int(eligible[candidate_column].max()),
        "unjudged_hands_minimum": int(eligible[unjudged_column].min()),
        "unjudged_hands_maximum": int(eligible[unjudged_column].max()),
        "pairwise_preferences": int(eligible[preference_column].sum()),
        "symmetric_preference_rows": int(2 * eligible[preference_column].sum()),
        "family_summary": family_rows,
    }


def _h0_tie_summary(
    frame: pd.DataFrame,
    family_column: str,
) -> dict[str, object]:
    scores = np.select(
        [
            frame[family_column] == "directed_transfer",
            frame[family_column] == "soft_play",
            frame[family_column] == "coordinated_isolation",
        ],
        [frame[column] for column in H0_SIGNALS],
        default=np.nan,
    )
    work = frame.assign(h0_score=scores)
    if work["h0_score"].isna().any() or work["started_at"].isna().any():
        raise ValueError("H0 tie audit received a missing route, score, or timestamp")
    sizes = (
        work.groupby(["pair_id", "h0_score"], sort=False)
        .size()
        .rename("tie_size")
        .reset_index()
    )
    ties = sizes[sizes["tie_size"] > 1]
    tied_hands = work.merge(
        ties[["pair_id", "h0_score"]],
        on=["pair_id", "h0_score"],
        how="inner",
        validate="many_to_many",
    )
    duplicate_timestamps = int(
        tied_hands.duplicated(
            ["pair_id", "h0_score", "started_at"], keep=False
        ).sum()
    )
    ordered = work.sort_values(
        ["pair_id", "h0_score"],
        ascending=[True, False],
        kind="stable",
    )
    ordered["rank_low"] = ordered.groupby("pair_id", sort=False).cumcount() + 1
    bounds = (
        ordered.groupby(["pair_id", "h0_score"], sort=False)["rank_low"]
        .agg(["min", "max"])
        .reset_index()
    )
    crosses_rank_five = int(
        ((bounds["min"] <= 5) & (bounds["max"] >= 5) & (bounds["min"] != bounds["max"])).sum()
    )
    return {
        "queries_with_candidate_hands": int(work["pair_id"].nunique()),
        "exact_score_tie_blocks": int(len(ties)),
        "hands_in_exact_score_tie_blocks": int(len(tied_hands)),
        "tie_blocks_crossing_rank_five": crosses_rank_five,
        "duplicate_started_at_rows_within_exact_tie_blocks": duplicate_timestamps,
        "started_at_resolves_all_observed_exact_ties": duplicate_timestamps == 0,
    }


def _time_direction_summary(
    pair_stats: pd.DataFrame,
    source: str,
    target: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        for family in TARGET_BEHAVIORS:
            train = pair_stats[
                (pair_stats["fold"] != fold)
                & (pair_stats["behavior_family"] == family)
                & (pair_stats[f"{source}_evidence"] > 0)
            ]
            valid = pair_stats[
                (pair_stats["fold"] == fold)
                & (pair_stats["behavior_family"] == family)
                & (pair_stats[f"{target}_evidence"] > 0)
            ]
            rows.append(
                {
                    "outer_fold": fold,
                    "behavior_family": family,
                    "source_training_pairs_with_evidence": int(len(train)),
                    "source_training_evidence_hands": int(
                        train[f"{source}_evidence"].sum()
                    ),
                    "source_pairwise_trainable_pairs": int(
                        (train[f"{source}_unjudged"] > 0).sum()
                    ),
                    "source_pairwise_preferences": int(
                        train[f"{source}_pairwise_preferences"].sum()
                    ),
                    "target_validation_pairs_with_evidence": int(len(valid)),
                    "target_validation_evidence_hands": int(
                        valid[f"{target}_evidence"].sum()
                    ),
                }
            )
    return {
        "source_half": source,
        "target_half": target,
        "pair_inclusion_rule": "include a query only when its scored target half contains at least one organizer evidence hand",
        "rows": rows,
        "minimum_source_training_pairs_by_family_fold": int(
            min(row["source_training_pairs_with_evidence"] for row in rows)
        ),
        "minimum_target_validation_pairs_by_family_fold": int(
            min(row["target_validation_pairs_with_evidence"] for row in rows)
        ),
    }


def _nested_cross_time_summary(
    pair_stats: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for direction, source, target in (
        ("early_to_late", "early", "late"),
        ("late_to_early", "late", "early"),
    ):
        rows: list[dict[str, object]] = []
        for outer_fold in range(5):
            outer_train = pair_stats[pair_stats["fold"] != outer_fold].reset_index(
                drop=True
            )
            splitter = StratifiedGroupKFold(
                n_splits=3,
                shuffle=True,
                random_state=seed + outer_fold,
            )
            assigned = np.full(len(outer_train), -1, dtype=np.int8)
            for inner_fold, (_, valid_index) in enumerate(
                splitter.split(
                    outer_train,
                    outer_train["behavior_family"],
                    outer_train["table_id"],
                )
            ):
                assigned[valid_index] = inner_fold
            if np.any(assigned < 0):
                raise ValueError("a nested time-audit pair lacks an inner fold")
            for inner_fold in range(3):
                for family in TARGET_BEHAVIORS:
                    train = outer_train[
                        (assigned != inner_fold)
                        & (outer_train["behavior_family"] == family)
                        & (outer_train[f"{source}_evidence"] > 0)
                    ]
                    valid = outer_train[
                        (assigned == inner_fold)
                        & (outer_train["behavior_family"] == family)
                        & (outer_train[f"{target}_evidence"] > 0)
                    ]
                    rows.append(
                        {
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "behavior_family": family,
                            "source_training_pairs_with_evidence": int(len(train)),
                            "source_training_evidence_hands": int(
                                train[f"{source}_evidence"].sum()
                            ),
                            "source_pairwise_trainable_pairs": int(
                                (train[f"{source}_unjudged"] > 0).sum()
                            ),
                            "source_pairwise_preferences": int(
                                train[f"{source}_pairwise_preferences"].sum()
                            ),
                            "target_validation_pairs_with_evidence": int(len(valid)),
                            "target_validation_evidence_hands": int(
                                valid[f"{target}_evidence"].sum()
                            ),
                        }
                    )
        result[direction] = {
            "source_half": source,
            "target_half": target,
            "rows": rows,
            "minimum_source_training_pairs_by_outer_inner_family": int(
                min(row["source_training_pairs_with_evidence"] for row in rows)
            ),
            "minimum_source_pairwise_trainable_pairs_by_outer_inner_family": int(
                min(row["source_pairwise_trainable_pairs"] for row in rows)
            ),
            "minimum_target_validation_pairs_by_outer_inner_family": int(
                min(row["target_validation_pairs_with_evidence"] for row in rows)
            ),
        }
    return result


def run_feasibility_audit(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 8803,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    paths = input_paths(data_root)
    full_receipt_path = output_root / "B001_behavior_oof.parquet"
    time_receipt_path = output_root / "B001_behavior_time_oof.parquet"
    pair_hands_path = output_root / "development_pair_hands.parquet"
    b001_implementation_path = Path(__file__).resolve().parent / "behavior_audit.py"
    b001_full_features_path = output_root / "development_pair_features.parquet"
    b001_early_features_path = output_root / "E007_early_base_features.parquet"
    b001_late_features_path = output_root / "E007_late_base_features.parquet"
    official_metric_path = Path("poker/reference/official_metric/slash-poker-competition-metric.ipynb")
    for path, expected in (
        (full_receipt_path, B001_FULL_SHA256),
        (time_receipt_path, B001_TIME_SHA256),
        (pair_hands_path, PAIR_HANDS_SHA256),
        (b001_implementation_path, B001_IMPLEMENTATION_SHA256),
        (b001_full_features_path, B001_FULL_FEATURES_SHA256),
        (b001_early_features_path, B001_EARLY_FEATURES_SHA256),
        (b001_late_features_path, B001_LATE_FEATURES_SHA256),
        (paths["hands"], HANDS_SHA256),
        (official_metric_path, OFFICIAL_METRIC_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen feasibility source changed: {path}")

    labels = pd.read_csv(paths["development_labels"])
    positives = labels[
        (labels["label"] == 1) & (labels["label_status"] == "confirmed_target")
    ][["pair_id", "behavior_family"]]
    full_receipt = pd.read_parquet(
        full_receipt_path,
        columns=["pair_id", "fold", "table_id", "predicted_family"],
    )
    time_receipt = pd.read_parquet(
        time_receipt_path,
        columns=["pair_id", "fold", "early_to_late__predicted_family", "late_to_early__predicted_family"],
    )
    positives = positives.merge(
        full_receipt,
        on="pair_id",
        how="left",
        validate="one_to_one",
    ).merge(
        time_receipt,
        on=["pair_id", "fold"],
        how="left",
        validate="one_to_one",
    )
    if positives[["fold", "table_id", "predicted_family"]].isna().any().any():
        raise ValueError("a confirmed target lacks a frozen behavior/fold receipt")
    if not set(positives["predicted_family"].astype(str)).issubset(TARGET_BEHAVIORS):
        raise ValueError("B001 full receipt contains an invalid predicted family")
    for column in (
        "early_to_late__predicted_family",
        "late_to_early__predicted_family",
    ):
        if not set(positives[column].astype(str)).issubset(TARGET_BEHAVIORS):
            raise ValueError(f"B001 time receipt contains an invalid family: {column}")
    if int(positives.groupby("table_id")["fold"].nunique().max()) != 1:
        raise ValueError("a confirmed target pool crosses outer folds")

    pair_hands = pd.read_parquet(
        pair_hands_path,
        columns=[
            "pair_id",
            "hand_id",
            "phase_progress",
            *PREREG_FEATURES,
            *H0_SIGNALS,
        ],
    )
    pair_hands["is_early"] = pair_hands["phase_progress"] <= 0.5
    positive_pair_hands = pair_hands[
        pair_hands["pair_id"].isin(positives["pair_id"])
    ].copy()
    prereg_values = positive_pair_hands.loc[:, PREREG_FEATURES].to_numpy(dtype=float)
    prereg_missing_values = int(np.isnan(prereg_values).sum())
    prereg_nonfinite_values = int((~np.isfinite(prereg_values)).sum())
    prereg_constant_features = [
        column
        for column in PREREG_FEATURES
        if positive_pair_hands[column].nunique(dropna=False) <= 1
    ]
    hand_times = pd.read_parquet(paths["hands"], columns=["hand_id", "started_at"])
    if hand_times["hand_id"].duplicated().any():
        raise ValueError("hands receipt contains duplicate hand identifiers")
    h0_frame = (
        positive_pair_hands.merge(
            hand_times,
            on="hand_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            positives[
                [
                    "pair_id",
                    "predicted_family",
                    "early_to_late__predicted_family",
                    "late_to_early__predicted_family",
                ]
            ],
            on="pair_id",
            how="left",
            validate="many_to_one",
        )
    )
    evidence = pd.read_csv(paths["development_evidence"])
    if evidence[["pair_id", "hand_id"]].duplicated().any():
        raise ValueError("duplicate organizer evidence pair-hand key")
    if evidence[["pair_id", "evidence_rank"]].duplicated().any() or not set(
        evidence["evidence_rank"].astype(int)
    ).issubset(range(1, 6)):
        raise ValueError("organizer evidence rank must be unique per pair and in 1..5")
    evidence = evidence.merge(
        positives[["pair_id", "behavior_family"]],
        on="pair_id",
        suffixes=("_evidence", "_label"),
        how="left",
        validate="many_to_one",
    )
    if evidence["behavior_family_label"].isna().any() or not np.array_equal(
        evidence["behavior_family_evidence"].astype(str).to_numpy(),
        evidence["behavior_family_label"].astype(str).to_numpy(),
    ):
        raise ValueError("evidence pair/family does not match a confirmed target")
    evidence = evidence.merge(
        pair_hands[["pair_id", "hand_id", "is_early"]],
        on=["pair_id", "hand_id"],
        how="left",
        validate="one_to_one",
    )
    if evidence["is_early"].isna().any():
        raise ValueError("an organizer evidence hand is not a legal shared hand")

    candidate_stats = pair_hands.groupby("pair_id", sort=False).agg(
        candidate_hands=("hand_id", "size"),
        early_candidate_hands=("is_early", "sum"),
    ).reset_index()
    candidate_stats["late_candidate_hands"] = (
        candidate_stats["candidate_hands"] - candidate_stats["early_candidate_hands"]
    )
    evidence_stats = evidence.groupby("pair_id", sort=False).agg(
        evidence_hands=("hand_id", "size"),
        early_evidence=("is_early", "sum"),
    ).reset_index()
    evidence_stats["late_evidence"] = (
        evidence_stats["evidence_hands"] - evidence_stats["early_evidence"]
    )
    pair_stats = (
        positives.merge(candidate_stats, on="pair_id", validate="one_to_one")
        .merge(evidence_stats, on="pair_id", validate="one_to_one")
    )
    if len(pair_stats) != 372 or int(pair_stats["evidence_hands"].sum()) != 1817:
        raise ValueError("positive/evidence receipt count changed")
    if not pair_stats["evidence_hands"].between(3, 5).all():
        raise ValueError("each confirmed target must retain three to five evidence hands")
    pair_stats["full_unjudged"] = (
        pair_stats["candidate_hands"] - pair_stats["evidence_hands"]
    )
    pair_stats["full_pairwise_preferences"] = (
        pair_stats["evidence_hands"] * pair_stats["full_unjudged"]
    )
    for half in ("early", "late"):
        pair_stats[f"{half}_unjudged"] = (
            pair_stats[f"{half}_candidate_hands"] - pair_stats[f"{half}_evidence"]
        )
        pair_stats[f"{half}_pairwise_preferences"] = (
            pair_stats[f"{half}_evidence"] * pair_stats[f"{half}_unjudged"]
        )

    family_totals: dict[str, object] = {}
    for family in TARGET_BEHAVIORS:
        part = pair_stats[pair_stats["behavior_family"] == family]
        family_totals[family] = {
            "pairs": int(len(part)),
            "candidate_hands": int(part["candidate_hands"].sum()),
            "evidence_hands": int(part["evidence_hands"].sum()),
            "pairs_with_early_evidence": int((part["early_evidence"] > 0).sum()),
            "early_evidence_hands": int(part["early_evidence"].sum()),
            "pairs_with_late_evidence": int((part["late_evidence"] > 0).sum()),
            "late_evidence_hands": int(part["late_evidence"].sum()),
        }

    report: dict[str, object] = {
        "audit": "EV000_joint_evidence_validation_feasibility",
        "status": "receipt_only_no_model_fit_no_experiment_authorization",
        "outer_fold_boundary": "frozen B001 whole-pool fold",
        "confirmed_target_pairs": int(len(pair_stats)),
        "confirmed_non_target_pairs_used": 0,
        "unknown_pairs_used": 0,
        "unknown_ground_truth_labels_assigned": 0,
        "evidence_hands": int(len(evidence)),
        "evidence_pair_hand_legality_passed": True,
        "evidence_behavior_family_matches_label": True,
        "all_positive_pairs_have_evidence": bool((pair_stats["evidence_hands"] > 0).all()),
        "family_totals": family_totals,
        "family_outer_fold_summary": _family_fold_summary(pair_stats),
        "nested_inner_split_seed_policy": "8803 + outer_fold",
        "nested_inner_fold_feasibility": _nested_inner_summary(pair_stats, seed),
        "cross_time": {
            "early_to_late": _time_direction_summary(pair_stats, "early", "late"),
            "late_to_early": _time_direction_summary(pair_stats, "late", "early"),
        },
        "nested_cross_time": _nested_cross_time_summary(pair_stats, seed),
        "frozen_prereg_model_input_readiness": {
            "confirmed_target_feature_rows": int(len(positive_pair_hands)),
            "feature_columns": list(PREREG_FEATURES),
            "feature_values_missing": prereg_missing_values,
            "feature_values_nonfinite": prereg_nonfinite_values,
            "globally_constant_features": prereg_constant_features,
            "full": _pairwise_scope_summary(pair_stats, "full"),
            "early": _pairwise_scope_summary(pair_stats, "early"),
            "late": _pairwise_scope_summary(pair_stats, "late"),
            "interpretation": "Evidence-bearing pairs with no unjudged hand contribute zero pairwise training preferences and must not be counted as pairwise-trainable.",
        },
        "h0_output_tie_readiness": {
            "selection_metric_policy": "exact expected AP under uniform random order inside every exact score tie",
            "deterministic_receipt_only_tie_break": "started_at ascending; never pair_id, hand_id, or row order",
            "unresolved_tie_policy": "stop if an exact score tie also shares started_at; no identifier fallback",
            "full": _h0_tie_summary(h0_frame, "predicted_family"),
            "early_to_late_target_late": _h0_tie_summary(
                h0_frame[h0_frame["phase_progress"] > 0.5],
                "early_to_late__predicted_family",
            ),
            "late_to_early_target_early": _h0_tie_summary(
                h0_frame[h0_frame["phase_progress"] <= 0.5],
                "late_to_early__predicted_family",
            ),
        },
        "strict_future_validation_requirements": [
            "Use the frozen B001 outer pair/fold mapping for behavior routing and evidence scoring.",
            "Train evidence models only on organizer evidence labels from outer-training confirmed targets.",
            "Use true family only on fitting-side rows; route inner validation with B001 models cross-fitted again inside each outer-training partition, and route outer validation with the frozen B001 OOF predicted family.",
            "Select every hyperparameter only in the three inner whole-pool folds, then refit outer training and score outer validation once.",
            "Compare a predicted-family heuristic baseline and learned ranker on identical held-out pairs and legal shared hands.",
            "Persist per-hand OOF scores and exact top-five receipts; report overall, three families, five outer folds, and both time directions.",
            "For a time direction, include a validation query only if its scored target half contains at least one organizer evidence hand; report coverage explicitly.",
            "Do not score evaluation rows or create a submission without separate authorization after the frozen gate is reviewed."
        ],
        "controls": {
            "model_fit_performed": False,
            "new_experiment_authorized": False,
            "evaluation_scored": False,
            "submission_created_or_modified": False,
        },
        "source_artifacts": {
            "development_labels_sha256": _sha256(paths["development_labels"]),
            "development_evidence_sha256": _sha256(paths["development_evidence"]),
            "development_pair_hands_sha256": _sha256(pair_hands_path),
            "b001_full_receipt_sha256": _sha256(full_receipt_path),
            "b001_time_receipt_sha256": _sha256(time_receipt_path),
            "b001_behavior_implementation_sha256": _sha256(
                b001_implementation_path
            ),
            "b001_full_features_sha256": _sha256(b001_full_features_path),
            "b001_early_features_sha256": _sha256(b001_early_features_path),
            "b001_late_features_sha256": _sha256(b001_late_features_path),
            "hands_sha256": _sha256(paths["hands"]),
            "official_metric_notebook_sha256": _sha256(official_metric_path),
        },
    }
    output_path = output_root / "EV000_evidence_feasibility.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_feasibility_audit(arguments.data_dir, arguments.work_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
