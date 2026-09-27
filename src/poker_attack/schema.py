"""Official table and submission schemas used by the local guardrails."""

from __future__ import annotations

from pathlib import Path

COMPETITION = "detect-suspicious-value-transfers-in-poker"

FILES = {
    "players": "players.parquet",
    "hands": "hands.parquet",
    "seats": "seats.parquet",
    "actions": "actions.parquet",
    "development_labels": "development_labels.csv",
    "development_evidence": "development_evidence.csv",
    "evaluation_pairs": "evaluation_pairs.csv",
    "sample_submission": "sample_submission.csv",
}

REQUIRED_COLUMNS = {
    "players": {
        "player_id",
        "account_age_days",
        "experience_hands_bucket",
        "preferred_stake",
        "region_bucket",
        "client_family",
    },
    "hands": {
        "hand_id",
        "phase",
        "table_id",
        "started_at",
        "big_blind",
        "final_pot",
    },
    "seats": {
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
    },
    "actions": {
        "hand_id",
        "action_no",
        "street",
        "player_id",
        "action",
        "amount",
        "to_call",
        "pot_before",
        "players_active",
    },
    "development_labels": {
        "pair_id",
        "player_1",
        "player_2",
        "label",
        "behavior_family",
    },
    "development_evidence": {"pair_id", "hand_id"},
    "evaluation_pairs": {"pair_id", "player_1", "player_2", "shared_hands"},
    "sample_submission": {
        "pair_id",
        "risk_score",
        "predicted_behavior",
        "evidence_hand_1",
        "evidence_hand_2",
        "evidence_hand_3",
        "evidence_hand_4",
        "evidence_hand_5",
    },
}

EVIDENCE_COLUMNS = tuple(f"evidence_hand_{i}" for i in range(1, 6))
SUBMISSION_COLUMNS = (
    "pair_id",
    "risk_score",
    "predicted_behavior",
    *EVIDENCE_COLUMNS,
)
ALLOWED_BEHAVIORS = {
    "none",
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
    "other_coordination",
}
TARGET_BEHAVIORS = (
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
)
NO_EVIDENCE = "NO_EVIDENCE"


def input_paths(data_dir: Path) -> dict[str, Path]:
    return {name: data_dir / filename for name, filename in FILES.items()}

