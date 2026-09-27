import tempfile
import unittest
from pathlib import Path

import pandas as pd

from poker_attack.audit import DataAuditError, validate_evidence_membership
from poker_attack.features import (
    aggregate_actions_parquet,
    aggregate_pair_features,
    build_pair_hands,
    build_player_hands,
    rank_evidence,
)


class FeatureTest(unittest.TestCase):
    def setUp(self):
        players = [f"p{i}" for i in range(1, 7)]
        self.players = pd.DataFrame(
            {
                "player_id": players,
                "account_age_days": [100, 110, 200, 210, 300, 310],
                "experience_hands_bucket": ["a", "a", "b", "b", "c", "c"],
                "preferred_stake": ["low"] * 6,
                "region_bucket": ["r1", "r1", "r2", "r2", "r3", "r3"],
                "client_family": ["web"] * 6,
            }
        )
        self.hands = pd.DataFrame(
            {
                "hand_id": ["h1", "h2"],
                "phase": ["development", "development"],
                "table_id": ["t1", "t1"],
                "started_at": [1, 2],
                "big_blind": [2.0, 2.0],
                "final_pot": [100.0, 40.0],
            }
        )
        rows = []
        for hand_id in self.hands["hand_id"]:
            for seat_no, player in enumerate(players):
                net = 0.0
                if hand_id == "h1" and player == "p1":
                    net = -40.0
                if hand_id == "h1" and player == "p2":
                    net = 40.0
                rows.append(
                    {
                        "hand_id": hand_id,
                        "player_id": player,
                        "seat_no": seat_no,
                        "hole_card_1": "As",
                        "hole_card_2": "Kh",
                        "starting_stack": 200.0,
                        "total_contribution": 20.0,
                        "net_chips": net,
                        "folded": False,
                        "went_to_showdown": player in {"p1", "p2"},
                        "won_share": float(net > 0),
                    }
                )
        self.seats = pd.DataFrame(rows)
        self.actions = pd.DataFrame(
            [
                {"hand_id": hand, "player_id": player, "street": "preflop", "action": "call", "amount": 2.0, "to_call": 2.0, "pot_before": 4.0, "action_no": 1, "players_active": 6}
                for hand in ["h1", "h2"]
                for player in players
            ]
        )

    def test_pair_features_and_evidence_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            actions_path = Path(tmp) / "actions.parquet"
            self.actions.to_parquet(actions_path, index=False)
            action_features = aggregate_actions_parquet(actions_path, self.hands, batch_size=5)
        player_hands = build_player_hands(self.seats, self.hands, action_features)
        pairs = pd.DataFrame(
            {"pair_id": ["pair12"], "player_1": ["p1"], "player_2": ["p2"], "shared_hands": [2]}
        )
        pair_hands = build_pair_hands(player_hands, pairs, "development")
        self.assertEqual(len(pair_hands), 2)
        h1 = pair_hands.set_index("hand_id").loc["h1"]
        self.assertEqual(h1["transfer_1_to_2_bb"], 20.0)
        self.assertGreater(h1["directed_signal"], pair_hands.set_index("hand_id").loc["h2", "directed_signal"])

        aggregate = aggregate_pair_features(pair_hands, pairs, self.players)
        self.assertEqual(int(aggregate.loc[0, "shared_hands_calc"]), 2)
        self.assertGreater(float(aggregate.loc[0, "dominant_flow_bb"]), 0)

        prediction = pd.DataFrame(
            {"pair_id": ["pair12"], "risk_score": [0.9], "predicted_behavior": ["directed_transfer"]}
        )
        evidence = rank_evidence(pair_hands, prediction)
        self.assertEqual(evidence.loc[0, "evidence_hand_1"], "h1")

    def test_evidence_membership_guard(self):
        pair_hands = pd.DataFrame(
            {
                "pair_id": ["p", "p", "p", "p", "p"],
                "hand_id": ["h1", "h2", "h3", "h4", "h5"],
            }
        )
        submission = pd.DataFrame(
            {
                "pair_id": ["p"],
                **{f"evidence_hand_{i}": [f"h{i}"] for i in range(1, 6)},
            }
        )
        report = validate_evidence_membership(submission, pair_hands)
        self.assertEqual(report["legal_evidence_hands"], 5)
        submission.loc[0, "evidence_hand_5"] = "not_shared"
        with self.assertRaises(DataAuditError):
            validate_evidence_membership(submission, pair_hands)


if __name__ == "__main__":
    unittest.main()
