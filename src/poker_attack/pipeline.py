"""Command-line pipeline for the Poker competition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import audit_dataset, validate_submission, write_report
from .evidence_model import train_evidence_models, rank_evidence_with_models
from .features import (
    aggregate_actions_parquet,
    aggregate_pair_features,
    build_pair_hands,
    build_player_hands,
)
from .model import train_pair_models
from .schema import EVIDENCE_COLUMNS, NO_EVIDENCE, SUBMISSION_COLUMNS, input_paths


def _load_or_build(path: Path, builder) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    frame = builder()
    frame.to_parquet(path, index=False)
    return frame


def _minimal_submission(evaluation: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    prediction = evaluation[["pair_id", "shared_hands"]].copy()
    ranks = prediction["shared_hands"].rank(method="average", pct=True)
    prediction["risk_score"] = ranks.clip(1e-6, 1 - 1e-6)
    prediction["predicted_behavior"] = "none"
    for column in EVIDENCE_COLUMNS:
        prediction[column] = NO_EVIDENCE
    return sample[["pair_id"]].merge(
        prediction[list(SUBMISSION_COLUMNS)], on="pair_id", how="left", validate="one_to_one"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = audit_dataset(data_dir)
    report_path = work_dir / "audit.json"
    write_report(report, report_path)
    if args.audit_only:
        return {"audit": str(report_path), "status": "audit_complete"}

    paths = input_paths(data_dir)
    evaluation = pd.read_csv(paths["evaluation_pairs"])
    sample = pd.read_csv(paths["sample_submission"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.minimal:
        submission = _minimal_submission(evaluation, sample)
        validate_submission(submission, evaluation["pair_id"])
        submission.to_csv(output, index=False)
        return {"audit": str(report_path), "submission": str(output), "status": "minimal_complete"}

    players = pd.read_parquet(paths["players"])
    hands = pd.read_parquet(paths["hands"])
    seat_columns = [
        "hand_id",
        "player_id",
        "seat_no",
        "hole_card_1",
        "hole_card_2",
        "starting_stack",
        "total_contribution",
        "net_chips",
        "folded",
        "went_to_showdown",
        "won_share",
    ]
    seats = pd.read_parquet(paths["seats"], columns=seat_columns)
    labels = pd.read_csv(paths["development_labels"])
    development_evidence = pd.read_csv(paths["development_evidence"])

    actions = _load_or_build(
        work_dir / "action_by_player_hand.parquet",
        lambda: aggregate_actions_parquet(paths["actions"], hands, batch_size=args.action_batch_size),
    )
    player_hands = _load_or_build(
        work_dir / "player_hands.parquet", lambda: build_player_hands(seats, hands, actions)
    )
    development_pair_hands = _load_or_build(
        work_dir / "development_pair_hands.parquet",
        lambda: build_pair_hands(
            player_hands,
            labels[["pair_id", "player_1", "player_2"]],
            "development",
        ),
    )
    evaluation_pair_hands = _load_or_build(
        work_dir / "evaluation_pair_hands.parquet",
        lambda: build_pair_hands(player_hands, evaluation, "evaluation"),
    )
    development_features = _load_or_build(
        work_dir / "development_pair_features.parquet",
        lambda: aggregate_pair_features(development_pair_hands, labels, players),
    )
    evaluation_features = _load_or_build(
        work_dir / "evaluation_pair_features.parquet",
        lambda: aggregate_pair_features(evaluation_pair_hands, evaluation, players),
    )

    model_result = train_pair_models(development_features, labels, evaluation_features)
    evidence_model = train_evidence_models(
        development_pair_hands,
        labels,
        development_evidence,
    )
    evidence = rank_evidence_with_models(
        evaluation_pair_hands,
        model_result.predictions,
        evidence_model,
    )
    submission = (
        sample[["pair_id"]]
        .merge(model_result.predictions, on="pair_id", how="left", validate="one_to_one")
        .merge(evidence, on="pair_id", how="left", validate="one_to_one")
    )
    for column in EVIDENCE_COLUMNS:
        if column not in submission:
            submission[column] = NO_EVIDENCE
        submission[column] = submission[column].fillna(NO_EVIDENCE).astype(str)
    submission = submission[list(SUBMISSION_COLUMNS)]
    validate_submission(submission, evaluation["pair_id"])
    submission.to_csv(output, index=False)

    diagnostics = {
        **model_result.diagnostics,
        "evidence_validation": evidence_model.diagnostics,
        "submission": str(output),
        "submission_rows": int(len(submission)),
        "risk_min": float(submission["risk_score"].min()),
        "risk_max": float(submission["risk_score"].max()),
    }
    diagnostic_path = work_dir / "model_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return {"audit": str(report_path), "diagnostics": str(diagnostic_path), **diagnostics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    parser.add_argument("--output", default="submissions/poker_residual_v1.csv")
    parser.add_argument("--action-batch-size", type=int, default=1_000_000)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
