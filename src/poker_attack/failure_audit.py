"""Cross-experiment residual audit for E007 through E011."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metric import average_precision
from .schema import TARGET_BEHAVIORS, input_paths


STATIC_RECEIPTS = {
    "interaction": ("E007_interaction_labelled_oof.parquet", "interaction_oof"),
    "marginal": ("E008_marginal_impact_labelled_oof.parquet", "marginal_oof"),
    "influence": ("E009_influence_labelled_oof.parquet", "influence_oof"),
    "pairwise": ("E010_pairwise_labelled_oof.parquet", "pairwise_oof"),
    "hard_negative": ("E011_hard_negative_labelled_oof.parquet", "hard_negative_oof"),
}

TIME_RECEIPTS = {
    "interaction": (
        "E007_time_stability_oof.parquet",
        "interaction_early_to_late",
        "interaction_late_to_early",
    ),
    "marginal": (
        "E008_marginal_impact_time_oof.parquet",
        "marginal_early_to_late",
        "marginal_late_to_early",
    ),
    "influence": (
        "E009_influence_time_oof.parquet",
        "influence_early_to_late",
        "influence_late_to_early",
    ),
    "pairwise": (
        "E010_pairwise_time_oof.parquet",
        "pairwise_early_to_late",
        "pairwise_late_to_early",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_ap(labels: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    negative = labels["label"].to_numpy(dtype=np.int8) == 0
    for family in TARGET_BEHAVIORS:
        positive = labels["behavior_family"].astype(str).to_numpy() == family
        keep = negative | positive
        result[family] = average_precision(positive[keep].astype(np.int8), scores[keep])
    return result


def _pool_ap(labels: pd.DataFrame, score_column: str) -> pd.Series:
    values: dict[str, float] = {}
    for table, group in labels.groupby("table_id", sort=False):
        y = group["label"].to_numpy(dtype=np.int8)
        if len(np.unique(y)) == 2:
            values[str(table)] = average_precision(y, group[score_column].to_numpy())
    return pd.Series(values, dtype=float)


def run_failure_audit(data_dir: str | Path, work_dir: str | Path) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    labels = pd.read_csv(input_paths(data_root)["development_labels"])[
        ["pair_id", "label", "label_status", "behavior_family"]
    ]
    base = pd.read_parquet(
        output_root / "E008_marginal_impact_labelled_oof.parquet",
        columns=["pair_id", "table_id", "fold", "raw_oof"],
    ).merge(labels, on="pair_id", validate="one_to_one")
    for name, (filename, column) in STATIC_RECEIPTS.items():
        receipt = pd.read_parquet(output_root / filename, columns=["pair_id", column])
        base = base.merge(
            receipt.rename(columns={column: name}), on="pair_id", validate="one_to_one"
        )

    y = base["label"].to_numpy(dtype=np.int8)
    positive_index = np.flatnonzero(y == 1)
    negative_index = np.flatnonzero(y == 0)
    raw_positive = base.iloc[positive_index]["raw_oof"].to_numpy()
    raw_negative = base.iloc[negative_index]["raw_oof"].to_numpy()
    raw_inversions = raw_positive[:, None] <= raw_negative[None, :]
    per_positive = pd.DataFrame(
        {
            "pair_id": base.iloc[positive_index]["pair_id"].to_numpy(),
            "table_id": base.iloc[positive_index]["table_id"].to_numpy(),
            "behavior_family": base.iloc[positive_index]["behavior_family"].to_numpy(),
            "raw_score": raw_positive,
            "raw_negatives_outranking": raw_inversions.sum(axis=1),
        }
    )

    static: dict[str, object] = {
        "raw": {
            "pair_ap": average_precision(y, base["raw_oof"].to_numpy()),
            "family_ap": _family_ap(base, base["raw_oof"].to_numpy()),
        }
    }
    raw_pool = _pool_ap(base, "raw_oof")
    inversion_summary: dict[str, object] = {}
    for name in STATIC_RECEIPTS:
        score = base[name].to_numpy()
        candidate_inversions = (
            score[positive_index, None] <= score[negative_index][None, :]
        )
        fixed = raw_inversions & ~candidate_inversions
        new = ~raw_inversions & candidate_inversions
        candidate_pool = _pool_ap(base, name)
        common = raw_pool.index.intersection(candidate_pool.index)
        pool_delta = candidate_pool.loc[common] - raw_pool.loc[common]
        static[name] = {
            "pair_ap": average_precision(y, score),
            "family_ap": _family_ap(base, score),
            "positive_score_delta_mean_by_family": {
                family: float(
                    (base.loc[base["behavior_family"] == family, name]
                    - base.loc[base["behavior_family"] == family, "raw_oof"]).mean()
                )
                for family in TARGET_BEHAVIORS
            },
            "score_delta_raw_spearman": float(
                spearmanr(score - base["raw_oof"].to_numpy(), base["raw_oof"]).statistic
            ),
            "mixed_pool_ap": {
                "pools": int(len(common)),
                "mean_delta": float(pool_delta.mean()),
                "median_delta": float(pool_delta.median()),
                "wins": int((pool_delta > 1e-12).sum()),
                "ties": int((pool_delta.abs() <= 1e-12).sum()),
                "losses": int((pool_delta < -1e-12).sum()),
            },
        }
        inversion_summary[name] = {
            "raw_inversions": int(raw_inversions.sum()),
            "remaining_inversions": int(candidate_inversions.sum()),
            "raw_inversions_fixed": int(fixed.sum()),
            "new_inversions_created": int(new.sum()),
            "net_inversions_removed": int(raw_inversions.sum() - candidate_inversions.sum()),
            "positive_pairs_with_net_fewer_inversions": int(
                (candidate_inversions.sum(axis=1) < raw_inversions.sum(axis=1)).sum()
            ),
            "positive_pairs_with_net_more_inversions": int(
                (candidate_inversions.sum(axis=1) > raw_inversions.sum(axis=1)).sum()
            ),
        }
        per_positive[f"{name}_score"] = score[positive_index]
        per_positive[f"{name}_negatives_outranking"] = candidate_inversions.sum(axis=1)
        per_positive[f"{name}_net_inversions_removed"] = (
            raw_inversions.sum(axis=1) - candidate_inversions.sum(axis=1)
        )

    candidate_names = list(STATIC_RECEIPTS)
    delta_matrix = np.column_stack(
        [base[name].to_numpy() - base["raw_oof"].to_numpy() for name in candidate_names]
    )
    static_delta_correlation = pd.DataFrame(
        delta_matrix, columns=candidate_names
    ).corr(method="spearman")

    time: dict[str, object] = {}
    for name, (filename, early_column, late_column) in TIME_RECEIPTS.items():
        receipt = pd.read_parquet(output_root / filename)
        joined = labels.merge(receipt, on=["pair_id", "label", "label_status"], validate="one_to_one")
        joined_y = joined["label"].to_numpy(dtype=np.int8)
        directions = {}
        for direction, raw_column, candidate_column in (
            ("early_to_late", "raw_early_to_late", early_column),
            ("late_to_early", "raw_late_to_early", late_column),
        ):
            raw_score = joined[raw_column].to_numpy()
            candidate_score = joined[candidate_column].to_numpy()
            directions[direction] = {
                "raw_ap": average_precision(joined_y, raw_score),
                "candidate_ap": average_precision(joined_y, candidate_score),
                "delta": float(
                    average_precision(joined_y, candidate_score)
                    - average_precision(joined_y, raw_score)
                ),
                "raw_family_ap": _family_ap(joined, raw_score),
                "candidate_family_ap": _family_ap(joined, candidate_score),
            }
        time[name] = directions

    hard_positive_path = output_root / "E007_E011_hard_positive_residuals.csv"
    per_positive.sort_values(
        ["raw_negatives_outranking", "pair_id"], ascending=[False, True]
    ).to_csv(hard_positive_path, index=False)
    delta_corr_path = output_root / "E007_E011_score_delta_spearman.csv"
    static_delta_correlation.to_csv(delta_corr_path)
    report: dict[str, object] = {
        "audit": "E007_E011_common_residuals",
        "confirmed_pairs": int(len(base)),
        "confirmed_positives": int(y.sum()),
        "confirmed_negatives": int((y == 0).sum()),
        "mixed_pools": int(len(raw_pool)),
        "static": static,
        "inversions": inversion_summary,
        "static_score_delta_spearman": static_delta_correlation.to_dict(),
        "time": time,
        "hard_positive_artifact": str(hard_positive_path),
        "hard_positive_sha256": _sha256(hard_positive_path),
        "delta_correlation_artifact": str(delta_corr_path),
        "delta_correlation_sha256": _sha256(delta_corr_path),
    }
    report_path = output_root / "E007_E011_failure_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_failure_audit(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
