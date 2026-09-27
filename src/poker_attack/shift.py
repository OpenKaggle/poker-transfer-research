"""Feature-distribution audit across confirmed, unlabeled, and evaluation pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from .metric import average_precision
from .model import _numeric_columns
from .schema import input_paths


def _standardized_difference(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.sqrt(0.5 * (np.var(left) + np.var(right)))
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float((np.mean(left) - np.mean(right)) / denominator)


def run_shift_audit(data_dir: str | Path, work_dir: str | Path) -> dict[str, object]:
    data_root = Path(data_dir)
    output_root = Path(work_dir)
    labels = pd.read_csv(input_paths(data_root)["development_labels"])
    confirmed = pd.read_parquet(output_root / "development_pair_features.parquet").merge(
        labels[["pair_id", "label"]], on="pair_id", validate="one_to_one"
    )
    unknown = pd.read_parquet(output_root / "pu_unknown_features.parquet")
    evaluation = pd.read_parquet(output_root / "evaluation_pair_features.parquet")
    submission = pd.read_csv("submissions/poker_residual_v1.csv", usecols=["pair_id", "risk_score"])
    evaluation = evaluation.merge(submission, on="pair_id", validate="one_to_one")
    features = _numeric_columns(confirmed.drop(columns=["table_id"], errors="ignore"))
    positive = confirmed[confirmed["label"] == 1]
    negative = confirmed[confirmed["label"] == 0]
    y = confirmed["label"].to_numpy(dtype=np.int8)
    rows: list[dict[str, object]] = []

    for feature in features:
        pos = positive[feature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
        neg = negative[feature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
        unk = unknown[feature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
        eva = evaluation[feature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
        all_confirmed = confirmed[feature].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
        if np.unique(all_confirmed).size > 1:
            univariate_ap = max(
                average_precision(y, all_confirmed),
                average_precision(y, -all_confirmed),
            )
        else:
            univariate_ap = float(y.mean())
        correlation = spearmanr(eva, evaluation["risk_score"].to_numpy(float)).statistic
        rows.append(
            {
                "feature": feature,
                "positive_mean": float(np.mean(pos)),
                "confirmed_negative_mean": float(np.mean(neg)),
                "unlabeled_mean": float(np.mean(unk)),
                "evaluation_mean": float(np.mean(eva)),
                "positive_vs_negative_smd": _standardized_difference(pos, neg),
                "negative_vs_unlabeled_smd": _standardized_difference(neg, unk),
                "negative_vs_evaluation_smd": _standardized_difference(neg, eva),
                "unlabeled_vs_evaluation_smd": _standardized_difference(unk, eva),
                "negative_vs_unlabeled_ks": float(ks_2samp(neg, unk).statistic),
                "negative_vs_evaluation_ks": float(ks_2samp(neg, eva).statistic),
                "unlabeled_vs_evaluation_ks": float(ks_2samp(unk, eva).statistic),
                "confirmed_univariate_ap_best_direction": float(univariate_ap),
                "evaluation_risk_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        "negative_vs_evaluation_ks", ascending=False, kind="stable"
    )
    csv_path = output_root / "E001_feature_shift.csv"
    audit.to_csv(csv_path, index=False)
    top_shift = audit.head(15)[
        [
            "feature",
            "negative_vs_evaluation_ks",
            "negative_vs_unlabeled_ks",
            "unlabeled_vs_evaluation_ks",
            "evaluation_risk_spearman",
        ]
    ].to_dict("records")
    top_risk = (
        audit.assign(abs_risk_correlation=audit["evaluation_risk_spearman"].abs())
        .sort_values("abs_risk_correlation", ascending=False, kind="stable")
        .head(15)[
            [
                "feature",
                "evaluation_risk_spearman",
                "negative_vs_evaluation_ks",
                "confirmed_univariate_ap_best_direction",
            ]
        ]
        .to_dict("records")
    )
    report: dict[str, object] = {
        "feature_count": int(len(features)),
        "confirmed_positive_pairs": int(len(positive)),
        "confirmed_negative_pairs": int(len(negative)),
        "unlabeled_pairs": int(len(unknown)),
        "evaluation_pairs": int(len(evaluation)),
        "top_confirmed_negative_to_evaluation_shifts": top_shift,
        "top_E001_risk_correlations": top_risk,
        "audit_csv": str(csv_path),
    }
    json_path = output_root / "E001_feature_shift_summary.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    args = parser.parse_args()
    print(json.dumps(run_shift_audit(args.data_dir, args.work_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
