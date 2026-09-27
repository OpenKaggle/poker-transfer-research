"""Sequential conditional-information influence feature and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp, spearmanr

from .metric import average_precision
from .model import _matrix, _new_model, _numeric_columns
from .pu import _assign_folds, _positive_percentiles
from .schema import input_paths


FEATURE = "sequential_net_influence_min"


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


def _action_code() -> pl.Expr:
    return (
        pl.when(pl.col("action") == "fold")
        .then(pl.lit(0, dtype=pl.Int8))
        .when(pl.col("action") == "check")
        .then(pl.lit(1, dtype=pl.Int8))
        .when((pl.col("action") == "call") | ((pl.col("action") == "all_in") & ~pl.col("is_aggressive")))
        .then(pl.lit(2, dtype=pl.Int8))
        .when(pl.col("is_aggressive") & (pl.col("amount_pot_ratio").fill_null(0.0) < 0.5))
        .then(pl.lit(3, dtype=pl.Int8))
        .when(pl.col("is_aggressive"))
        .then(pl.lit(4, dtype=pl.Int8))
        .otherwise(pl.lit(None, dtype=pl.Int8))
    )


def build_action_transitions(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Build label-free, explicitly ordered same-street action transitions."""

    data_root = Path(data_dir)
    output_root = Path(work_dir)
    output = output_root / "E009_action_transitions.parquet"
    if output.exists() and not force:
        return output
    paths = input_paths(data_root)
    prepared = output_root / "public_pu_prepared_v2"
    hands = (
        pl.scan_parquet(paths["hands"])
        .select("hand_id", "phase", "table_id", "started_at")
        .sort(["phase", "table_id", "started_at", "hand_id"])
        .with_columns(
            (
                pl.col("started_at").rank(method="ordinal").over(["phase", "table_id"])
                / pl.len().over(["phase", "table_id"])
            )
            .cast(pl.Float32)
            .alias("phase_progress")
        )
        .select("hand_id", "phase", "phase_progress")
    )
    actions = (
        pl.scan_parquet(prepared / "action_context.parquet")
        .sort(["hand_id", "action_no"])
        .with_columns(_action_code().alias("target_action"))
        .with_columns(
            pl.col("player_id").shift(1).over("hand_id").alias("influencer"),
            pl.col("target_action").shift(1).over("hand_id").alias("influencer_action"),
            pl.col("street").shift(1).over("hand_id").alias("influencer_street"),
        )
        .filter(
            pl.col("influencer").is_not_null()
            & (pl.col("influencer") != pl.col("player_id"))
            & (pl.col("influencer_street") == pl.col("street"))
            & pl.col("influencer_action").is_not_null()
            & pl.col("target_action").is_not_null()
        )
        .with_columns(
            pl.when(pl.col("street") == "preflop")
            .then(0)
            .when(pl.col("street") == "flop")
            .then(1)
            .when(pl.col("street") == "turn")
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("street_bin"),
            pl.when(pl.col("players_active") <= 2)
            .then(0)
            .when(pl.col("players_active") <= 4)
            .then(1)
            .otherwise(2)
            .cast(pl.Int8)
            .alias("active_bin"),
            pl.when(pl.col("to_call_bb") <= 0)
            .then(0)
            .when(pl.col("to_call_bb") <= 1)
            .then(1)
            .when(pl.col("to_call_bb") <= 4)
            .then(2)
            .otherwise(3)
            .cast(pl.Int8)
            .alias("to_call_bin"),
        )
        .with_columns(
            (
                pl.col("street_bin") * 12
                + pl.col("to_call_bin") * 3
                + pl.col("active_bin")
            )
            .cast(pl.Int8)
            .alias("state")
        )
        .select(
            "hand_id",
            "influencer",
            pl.col("player_id").alias("target"),
            "state",
            "influencer_action",
            "target_action",
        )
        .join(hands, on="hand_id", how="inner")
        .select(
            "phase",
            "phase_progress",
            "influencer",
            "target",
            "state",
            "influencer_action",
            "target_action",
        )
    )
    temp = output.with_suffix(".tmp.parquet")
    actions.sink_parquet(temp)
    temp.replace(output)
    return output


def _directed_influence(
    transitions_path: str | Path,
    *,
    phase: str,
    phase_half: str | None = None,
) -> pd.DataFrame:
    transitions = pl.scan_parquet(transitions_path).filter(pl.col("phase") == phase)
    if phase_half == "early":
        transitions = transitions.filter(pl.col("phase_progress") <= 0.5)
    elif phase_half == "late":
        transitions = transitions.filter(pl.col("phase_progress") > 0.5)
    elif phase_half is not None:
        raise ValueError(f"unknown phase_half: {phase_half}")
    counts = transitions.group_by(
        ["influencer", "target", "state", "influencer_action", "target_action"]
    ).agg(pl.len().cast(pl.Float64).alias("joint_count"))
    counts = counts.with_columns(
        pl.col("joint_count")
        .sum()
        .over(["influencer", "target", "state"])
        .alias("state_count"),
        pl.col("joint_count")
        .sum()
        .over(["influencer", "target", "state", "influencer_action"])
        .alias("influencer_marginal"),
        pl.col("joint_count")
        .sum()
        .over(["influencer", "target", "state", "target_action"])
        .alias("target_marginal"),
        pl.col("joint_count")
        .sum()
        .over(["influencer", "target"])
        .alias("edge_count"),
    ).with_columns(
        (
            pl.col("joint_count")
            / pl.col("edge_count")
            * (
                pl.col("joint_count")
                * pl.col("state_count")
                / (pl.col("influencer_marginal") * pl.col("target_marginal"))
            ).log()
        ).alias("cmi_contribution")
    )
    return (
        counts.group_by(["influencer", "target"])
        .agg(
            pl.col("cmi_contribution").sum().alias("empirical_cmi"),
            pl.col("edge_count").first().alias("edge_count"),
        )
        .with_columns(
            (
                pl.col("empirical_cmi")
                * pl.col("edge_count")
                / (pl.col("edge_count") + 50.0)
            ).alias("influence")
        )
        .select("influencer", "target", "edge_count", "influence")
        .collect(engine="streaming")
        .to_pandas()
    )


def aggregate_net_influence(
    transitions_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path,
    *,
    phase: str,
    phase_half: str | None = None,
    force: bool = False,
) -> Path:
    """Apply outsider-max subtraction and emit the sole symmetric pair feature."""

    output = Path(output_path)
    if output.exists() and not force:
        return output
    edges = _directed_influence(
        transitions_path,
        phase=phase,
        phase_half=phase_half,
    )
    influence = {
        (str(row.influencer), str(row.target)): float(row.influence)
        for row in edges.itertuples(index=False)
    }
    top_by_target: dict[str, tuple[str | None, float, float]] = {}
    for target, group in edges.groupby("target", sort=False):
        ordered = group.sort_values("influence", ascending=False, kind="stable")
        first_player = str(ordered.iloc[0]["influencer"]) if len(ordered) else None
        first_value = float(ordered.iloc[0]["influence"]) if len(ordered) else 0.0
        second_value = float(ordered.iloc[1]["influence"]) if len(ordered) > 1 else 0.0
        top_by_target[str(target)] = (first_player, first_value, second_value)

    def net(source: object, target: object) -> float:
        source_id, target_id = str(source), str(target)
        own = influence.get((source_id, target_id), 0.0)
        first_player, first_value, second_value = top_by_target.get(
            target_id, (None, 0.0, 0.0)
        )
        outsider = second_value if first_player == source_id else first_value
        return own - outsider

    pairs = pd.read_parquet(pairs_path, columns=["pair_id", "player_1", "player_2"])
    forward = np.fromiter(
        (net(a, b) for a, b in zip(pairs["player_1"], pairs["player_2"])),
        dtype=np.float64,
        count=len(pairs),
    )
    reverse = np.fromiter(
        (net(b, a) for a, b in zip(pairs["player_1"], pairs["player_2"])),
        dtype=np.float64,
        count=len(pairs),
    )
    result = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"].astype(str),
            FEATURE: np.minimum(forward, reverse),
        }
    )
    result.to_parquet(output, index=False)
    return output


def _load_frames(
    data_root: Path,
    output_root: Path,
    development_feature: Path,
    evaluation_feature: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    confirmed = (
        pd.read_parquet(output_root / "development_pair_features.parquet")
        .merge(
            labels[["pair_id", "label", "label_status", "behavior_family"]],
            on="pair_id",
            validate="one_to_one",
        )
        .merge(pd.read_parquet(development_feature), on="pair_id", validate="one_to_one")
    )
    evaluation = pd.read_parquet(
        output_root / "evaluation_pair_features.parquet"
    ).merge(pd.read_parquet(evaluation_feature), on="pair_id", validate="one_to_one")
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
        .merge(pd.read_parquet(development_feature), on="pair_id", validate="one_to_one")
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
        raise ValueError("E009 must keep unknown targets latent")
    return confirmed, unknown, evaluation


def _aligned_baseline(
    frame: pd.DataFrame,
    receipt_path: Path,
    columns: list[str],
) -> pd.DataFrame:
    receipt = pd.read_parquet(receipt_path, columns=["pair_id", *columns])
    aligned = frame[["pair_id"]].merge(receipt, on="pair_id", validate="one_to_one")
    if aligned["pair_id"].tolist() != frame["pair_id"].astype(str).tolist():
        raise ValueError("cached baseline receipt did not preserve pair order")
    return aligned


def _time_frame(
    output_root: Path,
    labels: pd.DataFrame,
    half: str,
    feature_path: Path,
    pool_lookup: pd.Series,
    order: list[str],
) -> pd.DataFrame:
    frame = (
        pd.read_parquet(output_root / f"E007_{half}_base_features.parquet")
        .merge(
            labels[["pair_id", "label", "label_status", "behavior_family"]],
            on="pair_id",
            validate="one_to_one",
        )
        .merge(pd.read_parquet(feature_path), on="pair_id", validate="one_to_one")
    )
    frame["table_id"] = frame["table_id"].fillna(frame["pair_id"].map(pool_lookup))
    if frame["table_id"].isna().any():
        raise ValueError("every time pair must keep its whole-pool boundary")
    return frame.set_index("pair_id").loc[order].reset_index()


def run_influence_validation(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    prepared = output_root / "public_pu_prepared_v2"
    transition_path = build_action_transitions(data_root, output_root)
    feature_paths = {
        "development": aggregate_net_influence(
            transition_path,
            prepared / "dev_pairs.parquet",
            output_root / "E009_influence_development.parquet",
            phase="development",
        ),
        "evaluation": aggregate_net_influence(
            transition_path,
            prepared / "eval_pairs.parquet",
            output_root / "E009_influence_evaluation.parquet",
            phase="evaluation",
        ),
        "early": aggregate_net_influence(
            transition_path,
            prepared / "dev_pairs.parquet",
            output_root / "E009_influence_early.parquet",
            phase="development",
            phase_half="early",
        ),
        "late": aggregate_net_influence(
            transition_path,
            prepared / "dev_pairs.parquet",
            output_root / "E009_influence_late.parquet",
            phase="development",
            phase_half="late",
        ),
    }
    confirmed, unknown, evaluation = _load_frames(
        data_root,
        output_root,
        feature_paths["development"],
        feature_paths["evaluation"],
    )
    base_features = _numeric_columns(
        confirmed.drop(columns=[FEATURE, "table_id"], errors="ignore")
    )
    augmented_features = base_features + [FEATURE]
    confirmed_x = _matrix(confirmed, augmented_features)
    unknown_x = _matrix(unknown, augmented_features)
    evaluation_x = _matrix(evaluation, augmented_features)
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
        raise ValueError("E009 sealed folds differ from cached E001 baseline")
    if not np.array_equal(base_unknown["fold"].to_numpy(dtype=np.int8), unknown_folds):
        raise ValueError("E009 unknown folds differ from cached baseline")
    raw_oof = base_labelled["raw_oof"].to_numpy(dtype=np.float32)
    raw_unknown_scores = base_unknown["raw_oof_score"].to_numpy(dtype=np.float32)
    raw_evaluation_scores = base_evaluation["raw_score"].to_numpy(dtype=np.float32)
    influence_oof = np.zeros(len(confirmed), dtype=np.float32)
    influence_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    influence_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    influence_fold_ap: list[float] = []
    raw_fold_ap: list[float] = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        model = _new_model(seed + fold)
        model.fit(confirmed_x[train], y[train])
        influence_oof[valid] = model.predict_proba(confirmed_x[valid])[:, 1]
        influence_unknown_scores[unknown_valid] = model.predict_proba(unknown_x[unknown_valid])[:, 1]
        influence_evaluation_scores += model.predict_proba(evaluation_x)[:, 1] / 5
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        influence_fold_ap.append(average_precision(y[valid], influence_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    influence_ap = average_precision(y, influence_oof)
    raw_separation = _positive_percentiles(confirmed, raw_oof, unknown, raw_unknown_scores)
    influence_separation = _positive_percentiles(
        confirmed, influence_oof, unknown, influence_unknown_scores
    )
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    influence_ks = float(
        ks_2samp(influence_unknown_scores, influence_evaluation_scores).statistic
    )
    influence_feature_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], evaluation[FEATURE]).statistic
    )
    influence_score_shared_corr = float(
        spearmanr(
            evaluation["shared_hands_calc"], influence_evaluation_scores
        ).statistic
    )

    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    order = confirmed["pair_id"].astype(str).tolist()
    pool_lookup = confirmed.set_index("pair_id")["table_id"]
    early = _time_frame(
        output_root, labels, "early", feature_paths["early"], pool_lookup, order
    )
    late = _time_frame(
        output_root, labels, "late", feature_paths["late"], pool_lookup, order
    )
    if not np.array_equal(early["label"].to_numpy(dtype=np.int8), y):
        raise ValueError("time labels do not align with sealed full-period folds")
    temporal_pattern = re.compile(r"__t[0-3]$")
    time_base_features = [
        column
        for column in _numeric_columns(
            early.drop(columns=[FEATURE, "table_id"], errors="ignore")
        )
        if not column.startswith("phase_progress")
        and "temporal_range" not in column
        and temporal_pattern.search(column) is None
    ]
    time_features = time_base_features + [FEATURE]
    early_x = _matrix(early, time_features)
    late_x = _matrix(late, time_features)
    base_time = _aligned_baseline(
        early,
        output_root / "E008_marginal_impact_time_oof.parquet",
        ["fold", "raw_early_to_late", "raw_late_to_early"],
    )
    if not np.array_equal(base_time["fold"].to_numpy(dtype=np.int8), folds):
        raise ValueError("cached time folds differ from E009 folds")
    time_predictions = {
        "raw_early_to_late": base_time["raw_early_to_late"].to_numpy(dtype=np.float32),
        "raw_late_to_early": base_time["raw_late_to_early"].to_numpy(dtype=np.float32),
        "influence_early_to_late": np.zeros(len(early), dtype=np.float32),
        "influence_late_to_early": np.zeros(len(early), dtype=np.float32),
    }
    time_fold_scores = {name: [] for name in time_predictions}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        early_model = _new_model(seed + 200 + fold)
        early_model.fit(early_x[train], y[train])
        time_predictions["influence_early_to_late"][valid] = early_model.predict_proba(
            late_x[valid]
        )[:, 1]
        late_model = _new_model(seed + 220 + fold)
        late_model.fit(late_x[train], y[train])
        time_predictions["influence_late_to_early"][valid] = late_model.predict_proba(
            early_x[valid]
        )[:, 1]
        for name, values in time_predictions.items():
            time_fold_scores[name].append(average_precision(y[valid], values[valid]))
    time_scores = {
        name: average_precision(y, values) for name, values in time_predictions.items()
    }
    early_late_delta = float(
        time_scores["influence_early_to_late"] - time_scores["raw_early_to_late"]
    )
    late_early_delta = float(
        time_scores["influence_late_to_early"] - time_scores["raw_late_to_early"]
    )
    time_fold_deltas = [
        time_fold_scores["influence_early_to_late"][fold]
        - time_fold_scores["raw_early_to_late"][fold]
        for fold in range(5)
    ] + [
        time_fold_scores["influence_late_to_early"][fold]
        - time_fold_scores["raw_late_to_early"][fold]
        for fold in range(5)
    ]
    time_directional_gap = float(
        abs(
            time_scores["influence_early_to_late"]
            - time_scores["influence_late_to_early"]
        )
    )
    deltas = {
        "confirmed_pair_ap": float(influence_ap - raw_ap),
        "worst_fold_pair_ap": float(min(influence_fold_ap) - min(raw_fold_ap)),
        "known_positive_above_unlabeled_p95_rate": float(
            influence_separation["known_positive_above_unlabeled_p95_rate"]
            - raw_separation["known_positive_above_unlabeled_p95_rate"]
        ),
        "unlabeled_to_evaluation_score_ks_absolute": float(influence_ks - raw_ks),
        "early_to_late": early_late_delta,
        "late_to_early": late_early_delta,
        "worst_time_fold": float(min(time_fold_deltas)),
        "time_directional_gap": time_directional_gap,
    }
    worst_influence_transfer = float(
        min(
            time_scores["influence_early_to_late"],
            time_scores["influence_late_to_early"],
        )
    )
    passes = bool(
        abs(influence_feature_shared_corr) <= 0.20
        and deltas["confirmed_pair_ap"] >= 0.003
        and deltas["worst_fold_pair_ap"] >= -0.02
        and deltas["known_positive_above_unlabeled_p95_rate"] >= 0.005
        and deltas["unlabeled_to_evaluation_score_ks_absolute"] <= 0.02
        and abs(influence_score_shared_corr) <= 0.10
        and early_late_delta >= 0.01
        and late_early_delta >= 0.01
        and min(time_fold_deltas) >= -0.02
        and worst_influence_transfer >= 0.84
        and time_directional_gap <= 0.05
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id", FEATURE]].copy()
    labelled_receipt["fold"] = folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["influence_oof"] = influence_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id", FEATURE]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["influence_oof_score"] = influence_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc", FEATURE]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["influence_score"] = influence_evaluation_scores
    time_receipt = early[["pair_id", "label", "label_status", "table_id", FEATURE]].rename(
        columns={FEATURE: f"{FEATURE}_early"}
    )
    time_receipt[f"{FEATURE}_late"] = late[FEATURE].to_numpy()
    time_receipt["fold"] = folds
    for name, values in time_predictions.items():
        time_receipt[name] = values
    receipt_paths = {
        "labelled": output_root / "E009_influence_labelled_oof.parquet",
        "unlabeled": output_root / "E009_influence_unlabeled_oof.parquet",
        "evaluation": output_root / "E009_influence_evaluation_scores.parquet",
        "time": output_root / "E009_influence_time_oof.parquet",
    }
    labelled_receipt.to_parquet(receipt_paths["labelled"], index=False)
    unknown_receipt.to_parquet(receipt_paths["unlabeled"], index=False)
    evaluation_receipt.to_parquet(receipt_paths["evaluation"], index=False)
    time_receipt.to_parquet(receipt_paths["time"], index=False)
    report: dict[str, object] = {
        "experiment": "E009_sequential_conditional_information_influence",
        "feature": FEATURE,
        "feature_count_added": 1,
        "feature_definition": (
            "same-street adjacent empirical conditional MI, n/(n+50) shrinkage, "
            "outsider-max net influence, symmetric direction minimum"
        ),
        "source": "https://proceedings.mlr.press/v180/bonjour22a.html",
        "source_scope_warning": "paper validates synthetic three-player RPS and Leduc only",
        "unknown_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "fold_boundary": "whole_table_equals_whole_pool",
        "cached_baseline_source": "E008 raw receipts; identical E001 features/folds/seeds",
        "feature_diagnostics": {
            "evaluation_shared_hands_spearman": influence_feature_shared_corr,
            "confirmed_nonzero": int((confirmed[FEATURE] != 0).sum()),
            "unlabeled_nonzero": int((unknown[FEATURE] != 0).sum()),
            "evaluation_nonzero": int((evaluation[FEATURE] != 0).sum()),
        },
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            **raw_separation,
        },
        "influence": {
            "confirmed_pair_ap": float(influence_ap),
            "fold_pair_ap": influence_fold_ap,
            "unlabeled_to_evaluation_score_ks": influence_ks,
            "evaluation_score_shared_hands_spearman": influence_score_shared_corr,
            **influence_separation,
        },
        "time_scores": time_scores,
        "time_fold_scores": time_fold_scores,
        "deltas": {**deltas, "worst_influence_transfer": worst_influence_transfer},
        "preregistered_gate": {
            "feature_evaluation_shared_hands_abs_spearman_max": 0.20,
            "confirmed_pair_ap_delta_min": 0.003,
            "worst_fold_pair_ap_delta_min": -0.02,
            "known_positive_above_unlabeled_p95_rate_delta_min": 0.005,
            "unlabeled_to_evaluation_score_ks_absolute_delta_max": 0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "early_to_late_delta_min": 0.01,
            "late_to_early_delta_min": 0.01,
            "worst_time_fold_delta_min": -0.02,
            "worst_influence_transfer_ap_min": 0.84,
            "time_directional_gap_max": 0.05,
            "passes": passes,
        },
        "transition_artifact": str(transition_path),
        "transition_sha256": _sha256(transition_path),
    }
    for name, path in feature_paths.items():
        report[f"{name}_feature_artifact"] = str(path)
        report[f"{name}_feature_sha256"] = _sha256(path)
    for name, path in receipt_paths.items():
        report[f"{name}_receipt_artifact"] = str(path)
        report[f"{name}_receipt_sha256"] = _sha256(path)
    report_path = output_root / "E009_influence_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_influence_validation(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
