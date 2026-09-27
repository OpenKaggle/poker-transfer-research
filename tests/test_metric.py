import unittest

import pandas as pd

from poker_attack.audit import DataAuditError
from poker_attack.metric import competition_score


class MetricTest(unittest.TestCase):
    def setUp(self):
        self.solution = pd.DataFrame(
            {
                "pair_id": ["a", "b", "c", "d"],
                "risk_score": [1, 1, 1, 0],
                "predicted_behavior": [
                    "directed_transfer",
                    "soft_play",
                    "coordinated_isolation",
                    "none",
                ],
                "evidence_hand_1": ["h1", "h2", "h3", "NO_EVIDENCE"],
                "evidence_hand_2": ["NO_EVIDENCE"] * 4,
                "evidence_hand_3": ["NO_EVIDENCE"] * 4,
                "evidence_hand_4": ["NO_EVIDENCE"] * 4,
                "evidence_hand_5": ["NO_EVIDENCE"] * 4,
            }
        )

    def test_perfect_score(self):
        submission = self.solution.copy()
        submission["risk_score"] = [0.99, 0.98, 0.97, 0.01]
        result = competition_score(self.solution, submission)
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertAlmostEqual(result["pair_ap"], 1.0)
        self.assertAlmostEqual(result["evidence_map5"], 1.0)
        self.assertAlmostEqual(result["behavior_map"], 1.0)

    def test_duplicate_evidence_is_rejected(self):
        submission = self.solution.copy()
        submission.loc[0, "evidence_hand_2"] = "h1"
        with self.assertRaises(DataAuditError):
            competition_score(self.solution, submission)

    def test_zero_family_scores_follow_stable_pair_order(self):
        submission = self.solution.copy()
        submission["risk_score"] = [0.9, 0.8, 0.7, 0.1]
        submission.loc[submission["pair_id"] == "c", "predicted_behavior"] = "none"
        result = competition_score(self.solution, submission)
        # The official metric still ranks an all-zero family score by stable
        # pair order, so the third true row receives AP 1/3 rather than zero.
        self.assertAlmostEqual(result["behavior_map"], 7 / 9)


if __name__ == "__main__":
    unittest.main()
