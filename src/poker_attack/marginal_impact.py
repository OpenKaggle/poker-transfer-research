"""Outcome-conditioned marginal-impact ablation with strict pool/time checks."""

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


FEATURE = "marginal_impact"


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


def build_response_outcomes(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Create one label-free realized outcome per hand/aggressor/responder edge."""

    data_root = Path(data_dir)
    output_root = Path(work_dir)
    output = output_root / "E008_response_outcomes.parquet"
    if output.exists() and not force:
        return output

    paths = input_paths(data_root)
    prepared = output_root / "public_pu_prepared_v2"
    actions = (
        pl.scan_parquet(prepared / "action_context.parquet")
        .filter(
            (pl.col("to_call") > 0)
            & pl.col("last_aggressor").is_not_null()
            & (pl.col("last_aggressor") != pl.col("player_id"))
        )
        .select(
            "hand_id",
            pl.col("last_aggressor").alias("aggressor"),
            pl.col("player_id").alias("responder"),
        )
        .unique(["hand_id", "aggressor", "responder"])
    )
    responder_results = pl.scan_parquet(
        prepared / "player_hand_features.parquet"
    ).select(
        "hand_id",
        pl.col("player_id").alias("responder"),
        pl.col("net_bb").clip(-20.0, 20.0).alias("responder_net_bb"),
    )
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
    result = (
        actions.join(
            responder_results,
            on=["hand_id", "responder"],
            how="inner",
        )
        .join(hands, on="hand_id", how="inner")
        .select(
            "hand_id",
            "phase",
            "phase_progress",
            "aggressor",
            "responder",
            "responder_net_bb",
        )
    )
    temp = output.with_suffix(".tmp.parquet")
    result.sink_parquet(temp)
    temp.replace(output)
    return output


def aggregate_marginal_impact(
    response_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path,
    *,
    phase: str,
    phase_half: str | None = None,
    force: bool = False,
) -> Path:
    """Aggregate the preregistered symmetric marginal-impact scalar."""

    output = Path(output_path)
    if output.exists() and not force:
        return output
    responses = pl.scan_parquet(response_path).filter(pl.col("phase") == phase)
    if phase_half == "early":
        responses = responses.filter(pl.col("phase_progress") <= 0.5)
    elif phase_half == "late":
        responses = responses.filter(pl.col("phase_progress") > 0.5)
    elif phase_half is not None:
        raise ValueError(f"unknown phase_half: {phase_half}")

    edges = responses.group_by(["aggressor", "responder"]).agg(
        pl.len().alias("edge_count"),
        pl.col("responder_net_bb").sum().alias("edge_sum"),
    )
    totals = edges.group_by("aggressor").agg(
        pl.col("edge_count").sum().alias("total_count"),
        pl.col("edge_sum").sum().alias("total_sum"),
    )
    directed = (
        edges.join(totals, on="aggressor", how="left")
        .with_columns(
            (pl.col("total_count") - pl.col("edge_count")).alias("other_count"),
            (pl.col("total_sum") - pl.col("edge_sum")).alias("other_sum"),
        )
        .with_columns(
            pl.when(pl.col("other_count") > 0)
            .then(
                (
                    pl.col("edge_sum")
                    - pl.col("edge_count")
                    * pl.col("other_sum")
                    / pl.col("other_count")
                )
                / (pl.col("edge_count") + 10.0)
            )
            .otherwise(0.0)
            .alias("directed_marginal_impact")
        )
        .select("aggressor", "responder", "directed_marginal_impact")
        .collect(engine="streaming")
    )
    impact = {
        (str(row[0]), str(row[1])): float(row[2])
        for row in directed.iter_rows()
    }
    pairs = pd.read_parquet(pairs_path, columns=["pair_id", "player_1", "player_2"])
    forward = np.fromiter(
        (
            impact.get((str(left), str(right)), 0.0)
            for left, right in zip(pairs["player_1"], pairs["player_2"])
        ),
        dtype=np.float64,
        count=len(pairs),
    )
    reverse = np.fromiter(
        (
            impact.get((str(right), str(left)), 0.0)
            for left, right in zip(pairs["player_1"], pairs["player_2"])
        ),
        dtype=np.float64,
        count=len(pairs),
    )
    result = pd.DataFrame(
        {"pair_id": pairs["pair_id"].astype(str), FEATURE: forward + reverse}
    )
    result.to_parquet(output, index=False)
    return output


def _full_frames(
    data_root: Path,
    output_root: Path,
    development_impact: Path,
    evaluation_impact: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    confirmed = (
        pd.read_parquet(output_root / "development_pair_features.parquet")
        .merge(
            labels[["pair_id", "label", "label_status", "behavior_family"]],
            on="pair_id",
            validate="one_to_one",
        )
        .merge(pd.read_parquet(development_impact), on="pair_id", validate="one_to_one")
    )
    evaluation = pd.read_parquet(
        output_root / "evaluation_pair_features.parquet"
    ).merge(pd.read_parquet(evaluation_impact), on="pair_id", validate="one_to_one")

    public_pairs = pd.read_parquet(output_root / "public_pu_prepared_v2" / "dev_pairs.parquet")
    public_unknown = _canonical_players(public_pairs[public_pairs["is_pu"]])
    owned_pairs = _canonical_players(pd.read_parquet(output_root / "pu_unknown_pairs.parquet"))
    mapping = public_unknown[["pair_id", "_p_low", "_p_high"]].merge(
        owned_pairs[["pair_id", "_p_low", "_p_high"]].rename(
            columns={"pair_id": "owned_pair_id"}
        ),
        on=["_p_low", "_p_high"],
        validate="one_to_one",
    )
    if len(mapping) != len(public_unknown):
        raise ValueError("not every public PU pair maps to the owned unlabeled universe")
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
        .merge(pd.read_parquet(development_impact), on="pair_id", validate="one_to_one")
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
        raise ValueError("E008 must keep unknown targets latent")
    return confirmed, unknown, evaluation


def _time_frame(
    output_root: Path,
    labels: pd.DataFrame,
    half: str,
    impact_path: Path,
    pool_lookup: pd.Series,
) -> pd.DataFrame:
    base = pd.read_parquet(output_root / f"E007_{half}_base_features.parquet")
    frame = base.merge(
        labels[["pair_id", "label", "label_status", "behavior_family"]],
        on="pair_id",
        validate="one_to_one",
    ).merge(pd.read_parquet(impact_path), on="pair_id", validate="one_to_one")
    frame["table_id"] = frame["table_id"].fillna(frame["pair_id"].map(pool_lookup))
    if frame["table_id"].isna().any():
        raise ValueError("every time-split pair must retain its pool boundary")
    return frame


def run_marginal_impact_validation(
    data_dir: str | Path,
    work_dir: str | Path,
    *,
    seed: int = 7401,
) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    prepared = output_root / "public_pu_prepared_v2"
    response_path = build_response_outcomes(data_root, output_root)
    impact_paths = {
        "development": aggregate_marginal_impact(
            response_path,
            prepared / "dev_pairs.parquet",
            output_root / "E008_marginal_impact_development.parquet",
            phase="development",
        ),
        "evaluation": aggregate_marginal_impact(
            response_path,
            prepared / "eval_pairs.parquet",
            output_root / "E008_marginal_impact_evaluation.parquet",
            phase="evaluation",
        ),
        "early": aggregate_marginal_impact(
            response_path,
            prepared / "dev_pairs.parquet",
            output_root / "E008_marginal_impact_early.parquet",
            phase="development",
            phase_half="early",
        ),
        "late": aggregate_marginal_impact(
            response_path,
            prepared / "dev_pairs.parquet",
            output_root / "E008_marginal_impact_late.parquet",
            phase="development",
            phase_half="late",
        ),
    }
    confirmed, unknown, evaluation = _full_frames(
        data_root,
        output_root,
        impact_paths["development"],
        impact_paths["evaluation"],
    )
    base_features = _numeric_columns(
        confirmed.drop(columns=[FEATURE, "table_id"], errors="ignore")
    )
    augmented_features = base_features + [FEATURE]
    raw_confirmed = _matrix(confirmed, base_features)
    raw_unknown = _matrix(unknown, base_features)
    raw_evaluation = _matrix(evaluation, base_features)
    augmented_confirmed = _matrix(confirmed, augmented_features)
    augmented_unknown = _matrix(unknown, augmented_features)
    augmented_evaluation = _matrix(evaluation, augmented_features)
    y = confirmed["label"].to_numpy(dtype=np.int8)
    folds, unknown_folds = _assign_folds(confirmed, unknown, seed)

    raw_oof = np.zeros(len(confirmed), dtype=np.float32)
    marginal_oof = np.zeros(len(confirmed), dtype=np.float32)
    raw_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    marginal_unknown_scores = np.zeros(len(unknown), dtype=np.float32)
    raw_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    marginal_evaluation_scores = np.zeros(len(evaluation), dtype=np.float32)
    raw_fold_ap: list[float] = []
    marginal_fold_ap: list[float] = []
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        unknown_valid = np.flatnonzero(unknown_folds == fold)
        raw_model = _new_model(seed + fold)
        raw_model.fit(raw_confirmed[train], y[train])
        raw_oof[valid] = raw_model.predict_proba(raw_confirmed[valid])[:, 1]
        raw_unknown_scores[unknown_valid] = raw_model.predict_proba(raw_unknown[unknown_valid])[:, 1]
        raw_evaluation_scores += raw_model.predict_proba(raw_evaluation)[:, 1] / 5

        marginal_model = _new_model(seed + 100 + fold)
        marginal_model.fit(augmented_confirmed[train], y[train])
        marginal_oof[valid] = marginal_model.predict_proba(augmented_confirmed[valid])[:, 1]
        marginal_unknown_scores[unknown_valid] = marginal_model.predict_proba(
            augmented_unknown[unknown_valid]
        )[:, 1]
        marginal_evaluation_scores += marginal_model.predict_proba(
            augmented_evaluation
        )[:, 1] / 5
        raw_fold_ap.append(average_precision(y[valid], raw_oof[valid]))
        marginal_fold_ap.append(average_precision(y[valid], marginal_oof[valid]))

    raw_ap = average_precision(y, raw_oof)
    marginal_ap = average_precision(y, marginal_oof)
    raw_separation = _positive_percentiles(confirmed, raw_oof, unknown, raw_unknown_scores)
    marginal_separation = _positive_percentiles(
        confirmed, marginal_oof, unknown, marginal_unknown_scores
    )
    raw_ks = float(ks_2samp(raw_unknown_scores, raw_evaluation_scores).statistic)
    marginal_ks = float(
        ks_2samp(marginal_unknown_scores, marginal_evaluation_scores).statistic
    )
    raw_shared_corr = float(
        spearmanr(evaluation["shared_hands_calc"], raw_evaluation_scores).statistic
    )
    marginal_shared_corr = float(
        spearmanr(
            evaluation["shared_hands_calc"], marginal_evaluation_scores
        ).statistic
    )

    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    pool_lookup = confirmed.set_index("pair_id")["table_id"]
    early = _time_frame(
        output_root, labels, "early", impact_paths["early"], pool_lookup
    )
    late = _time_frame(
        output_root, labels, "late", impact_paths["late"], pool_lookup
    )
    confirmed_order = confirmed["pair_id"].astype(str).tolist()
    early = early.set_index("pair_id").loc[confirmed_order].reset_index()
    late = late.set_index("pair_id").loc[confirmed_order].reset_index()
    if not np.array_equal(early["label"].to_numpy(dtype=np.int8), y):
        raise ValueError("time-split labels do not align with full-period folds")
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
    time_augmented_features = time_base_features + [FEATURE]
    early_raw = _matrix(early, time_base_features)
    late_raw = _matrix(late, time_base_features)
    early_augmented = _matrix(early, time_augmented_features)
    late_augmented = _matrix(late, time_augmented_features)
    time_names = (
        "raw_early_to_late",
        "raw_late_to_early",
        "marginal_early_to_late",
        "marginal_late_to_early",
    )
    time_predictions = {
        name: np.zeros(len(early), dtype=np.float32) for name in time_names
    }
    time_fold_scores = {name: [] for name in time_names}
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        raw_early_model = _new_model(seed + 200 + fold)
        raw_early_model.fit(early_raw[train], y[train])
        time_predictions["raw_early_to_late"][valid] = raw_early_model.predict_proba(
            late_raw[valid]
        )[:, 1]
        raw_late_model = _new_model(seed + 220 + fold)
        raw_late_model.fit(late_raw[train], y[train])
        time_predictions["raw_late_to_early"][valid] = raw_late_model.predict_proba(
            early_raw[valid]
        )[:, 1]
        marginal_early_model = _new_model(seed + 240 + fold)
        marginal_early_model.fit(early_augmented[train], y[train])
        time_predictions["marginal_early_to_late"][valid] = (
            marginal_early_model.predict_proba(late_augmented[valid])[:, 1]
        )
        marginal_late_model = _new_model(seed + 260 + fold)
        marginal_late_model.fit(late_augmented[train], y[train])
        time_predictions["marginal_late_to_early"][valid] = (
            marginal_late_model.predict_proba(early_augmented[valid])[:, 1]
        )
        for name in time_names:
            time_fold_scores[name].append(
                average_precision(y[valid], time_predictions[name][valid])
            )
    time_scores = {
        name: average_precision(y, values) for name, values in time_predictions.items()
    }
    early_late_delta = float(
        time_scores["marginal_early_to_late"] - time_scores["raw_early_to_late"]
    )
    late_early_delta = float(
        time_scores["marginal_late_to_early"] - time_scores["raw_late_to_early"]
    )
    time_fold_deltas = [
        time_fold_scores["marginal_early_to_late"][fold]
        - time_fold_scores["raw_early_to_late"][fold]
        for fold in range(5)
    ] + [
        time_fold_scores["marginal_late_to_early"][fold]
        - time_fold_scores["raw_late_to_early"][fold]
        for fold in range(5)
    ]
    time_directional_gap = float(
        abs(
            time_scores["marginal_early_to_late"]
            - time_scores["marginal_late_to_early"]
        )
    )

    deltas = {
        "confirmed_pair_ap": float(marginal_ap - raw_ap),
        "worst_fold_pair_ap": float(min(marginal_fold_ap) - min(raw_fold_ap)),
        "known_positive_above_unlabeled_p95_rate": float(
            marginal_separation["known_positive_above_unlabeled_p95_rate"]
            - raw_separation["known_positive_above_unlabeled_p95_rate"]
        ),
        "unlabeled_to_evaluation_score_ks_absolute": float(marginal_ks - raw_ks),
        "early_to_late": early_late_delta,
        "late_to_early": late_early_delta,
        "worst_time_fold": float(min(time_fold_deltas)),
        "time_directional_gap": time_directional_gap,
    }
    passes = bool(
        deltas["confirmed_pair_ap"] >= -0.005
        and deltas["worst_fold_pair_ap"] >= -0.02
        and deltas["known_positive_above_unlabeled_p95_rate"] >= 0.0
        and deltas["unlabeled_to_evaluation_score_ks_absolute"] <= 0.02
        and abs(marginal_shared_corr) <= 0.10
        and early_late_delta >= 0.01
        and late_early_delta >= 0.01
        and min(time_fold_deltas) >= -0.02
        and time_directional_gap <= 0.05
    )

    labelled_receipt = confirmed[["pair_id", "label", "label_status", "table_id", FEATURE]].copy()
    labelled_receipt["fold"] = folds
    labelled_receipt["raw_oof"] = raw_oof
    labelled_receipt["marginal_oof"] = marginal_oof
    unknown_receipt = unknown[["pair_id", "label_status", "training_role", "table_id", FEATURE]].copy()
    unknown_receipt["fold"] = unknown_folds
    unknown_receipt["raw_oof_score"] = raw_unknown_scores
    unknown_receipt["marginal_oof_score"] = marginal_unknown_scores
    evaluation_receipt = evaluation[["pair_id", "table_id", "shared_hands_calc", FEATURE]].copy()
    evaluation_receipt["raw_score"] = raw_evaluation_scores
    evaluation_receipt["marginal_score"] = marginal_evaluation_scores
    time_receipt = early[["pair_id", "label", "label_status", "table_id", FEATURE]].rename(
        columns={FEATURE: f"{FEATURE}_early"}
    )
    time_receipt[f"{FEATURE}_late"] = late[FEATURE].to_numpy()
    time_receipt["fold"] = folds
    for name, values in time_predictions.items():
        time_receipt[name] = values
    receipt_paths = {
        "labelled": output_root / "E008_marginal_impact_labelled_oof.parquet",
        "unlabeled": output_root / "E008_marginal_impact_unlabeled_oof.parquet",
        "evaluation": output_root / "E008_marginal_impact_evaluation_scores.parquet",
        "time": output_root / "E008_marginal_impact_time_oof.parquet",
    }
    labelled_receipt.to_parquet(receipt_paths["labelled"], index=False)
    unknown_receipt.to_parquet(receipt_paths["unlabeled"], index=False)
    evaluation_receipt.to_parquet(receipt_paths["evaluation"], index=False)
    time_receipt.to_parquet(receipt_paths["time"], index=False)

    report: dict[str, object] = {
        "experiment": "E008_outcome_conditioned_marginal_impact",
        "feature": FEATURE,
        "feature_count_added": 1,
        "feature_definition": (
            "winsorized responder net BB edge residual versus aggressor leave-one-responder-out mean; "
            "edge residual denominator count+10; symmetric direction sum"
        ),
        "unknown_pairs": int(len(unknown)),
        "unknown_ground_truth_labels_assigned": 0,
        "fold_boundary": "whole_table_equals_whole_pool",
        "raw": {
            "confirmed_pair_ap": float(raw_ap),
            "fold_pair_ap": raw_fold_ap,
            "unlabeled_to_evaluation_score_ks": raw_ks,
            "evaluation_score_shared_hands_spearman": raw_shared_corr,
            **raw_separation,
        },
        "marginal_impact": {
            "confirmed_pair_ap": float(marginal_ap),
            "fold_pair_ap": marginal_fold_ap,
            "unlabeled_to_evaluation_score_ks": marginal_ks,
            "evaluation_score_shared_hands_spearman": marginal_shared_corr,
            **marginal_separation,
        },
        "time_scores": time_scores,
        "time_fold_scores": time_fold_scores,
        "deltas": deltas,
        "preregistered_gate": {
            "confirmed_pair_ap_delta_min": -0.005,
            "worst_fold_pair_ap_delta_min": -0.02,
            "known_positive_above_unlabeled_p95_rate_delta_min": 0.0,
            "unlabeled_to_evaluation_score_ks_absolute_delta_max": 0.02,
            "evaluation_score_shared_hands_abs_spearman_max": 0.10,
            "early_to_late_delta_min": 0.01,
            "late_to_early_delta_min": 0.01,
            "worst_time_fold_delta_min": -0.02,
            "time_directional_gap_max": 0.05,
            "passes": passes,
        },
        "response_artifact": str(response_path),
        "response_sha256": _sha256(response_path),
    }
    for name, path in impact_paths.items():
        report[f"{name}_impact_artifact"] = str(path)
        report[f"{name}_impact_sha256"] = _sha256(path)
    for name, path in receipt_paths.items():
        report[f"{name}_receipt_artifact"] = str(path)
        report[f"{name}_receipt_sha256"] = _sha256(path)
    report_path = output_root / "E008_marginal_impact_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(
        json.dumps(
            run_marginal_impact_validation(args.data_dir, args.work_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
