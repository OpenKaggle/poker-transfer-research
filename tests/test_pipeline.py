import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from poker_attack.audit import validate_submission
from poker_attack.pipeline import run


class PipelineIntegrationTest(unittest.TestCase):
    def test_end_to_end_on_six_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            work = root / "work"
            data.mkdir()

            player_rows = []
            hand_rows = []
            seat_rows = []
            action_rows = []
            label_rows = []
            evidence_rows = []
            evaluation_rows = []
            families = ["directed_transfer", "soft_play", "coordinated_isolation"]
            action_no = 0
            for pool in range(6):
                player_ids = [f"p{pool}_{i}" for i in range(6)]
                for i, player_id in enumerate(player_ids):
                    player_rows.append(
                        {
                            "player_id": player_id,
                            "account_age_days": 100 + pool * 10 + i,
                            "experience_hands_bucket": f"e{i % 2}",
                            "preferred_stake": "low",
                            "region_bucket": f"r{pool % 2}",
                            "client_family": "web",
                        }
                    )

                positive_pair = f"dev_pos_{pool}"
                negative_pair = f"dev_neg_{pool}"
                label_rows.extend(
                    [
                        {
                            "pair_id": positive_pair,
                            "player_1": player_ids[0],
                            "player_2": player_ids[1],
                            "label": 1,
                            "label_status": "confirmed_target",
                            "behavior_family": families[pool % 3],
                        },
                        {
                            "pair_id": negative_pair,
                            "player_1": player_ids[2],
                            "player_2": player_ids[3],
                            "label": 0,
                            "label_status": "confirmed_non_target",
                            "behavior_family": "none",
                        },
                    ]
                )
                evaluation_rows.append(
                    {
                        "pair_id": f"eval_{pool}",
                        "player_1": player_ids[4],
                        "player_2": player_ids[5],
                        "shared_hands": 3,
                    }
                )

                for phase, count in (("development", 4), ("evaluation", 3)):
                    for number in range(count):
                        hand_id = f"{pool}_{phase}_{number}"
                        hand_rows.append(
                            {
                                "hand_id": hand_id,
                                "phase": phase,
                                "table_id": f"table_{pool}",
                                "started_at": number,
                                "big_blind": 2.0,
                                "final_pot": 60.0,
                            }
                        )
                        if phase == "development" and number == 0:
                            evidence_rows.append(
                                {
                                    "pair_id": positive_pair,
                                    "hand_id": hand_id,
                                    "behavior_family": families[pool % 3],
                                    "evidence_rank": 1,
                                }
                            )
                        for seat_no, player_id in enumerate(player_ids):
                            net = 0.0
                            if phase == "development" and number == 0 and seat_no == 0:
                                net = -30.0
                            if phase == "development" and number == 0 and seat_no == 1:
                                net = 30.0
                            seat_rows.append(
                                {
                                    "hand_id": hand_id,
                                    "player_id": player_id,
                                    "seat_no": seat_no,
                                    "hole_card_1": "As",
                                    "hole_card_2": "Kh",
                                    "starting_stack": 200.0,
                                    "total_contribution": 10.0,
                                    "net_chips": net,
                                    "folded": False,
                                    "went_to_showdown": seat_no in {0, 1, 4, 5},
                                    "won_share": float(net > 0),
                                }
                            )
                            action_no += 1
                            action_rows.append(
                                {
                                    "hand_id": hand_id,
                                    "action_no": action_no,
                                    "street": "preflop",
                                    "player_id": player_id,
                                    "action": "call" if seat_no % 2 else "raise",
                                    "amount": 4.0,
                                    "to_call": 2.0,
                                    "pot_before": 6.0,
                                    "players_active": 6,
                                }
                            )

            pd.DataFrame(player_rows).to_parquet(data / "players.parquet", index=False)
            pd.DataFrame(hand_rows).to_parquet(data / "hands.parquet", index=False)
            pd.DataFrame(seat_rows).to_parquet(data / "seats.parquet", index=False)
            pd.DataFrame(action_rows).to_parquet(data / "actions.parquet", index=False)
            pd.DataFrame(label_rows).to_csv(data / "development_labels.csv", index=False)
            pd.DataFrame(evidence_rows).to_csv(data / "development_evidence.csv", index=False)
            evaluation = pd.DataFrame(evaluation_rows)
            evaluation.to_csv(data / "evaluation_pairs.csv", index=False)
            sample = evaluation[["pair_id"]].copy()
            sample["risk_score"] = 0.5
            sample["predicted_behavior"] = "none"
            for index in range(1, 6):
                sample[f"evidence_hand_{index}"] = "NO_EVIDENCE"
            sample.to_csv(data / "sample_submission.csv", index=False)

            output = root / "submission.csv"
            result = run(
                argparse.Namespace(
                    data_dir=str(data),
                    work_dir=str(work),
                    output=str(output),
                    action_batch_size=13,
                    audit_only=False,
                    minimal=False,
                )
            )
            submission = pd.read_csv(output)
            validate_submission(submission, evaluation["pair_id"])
            self.assertEqual(len(submission), 6)
            self.assertTrue(submission["evidence_hand_1"].str.contains("evaluation").all())
            self.assertIn("cv_pair_ap", result)


if __name__ == "__main__":
    unittest.main()
