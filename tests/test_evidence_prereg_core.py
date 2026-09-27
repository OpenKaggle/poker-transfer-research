import itertools
import unittest

import numpy as np
import pandas as pd

from poker_attack.evidence_prereg_core import (
    B001_FULL_FEATURES,
    B001_TIME_FEATURES,
    BOOTSTRAP_DRAWS,
    C_GRID,
    DRAFT_SHA256,
    FAMILIES,
    FEATURES,
    INNER_SPLIT_BASE_SEED,
    PREREG_STATIC_AUDIT_SHA256,
    SENSITIVITY_INTERVAL_NAME,
    WEIGHT_STREAM_NO_HEADER_SHA256,
    WEIGHT_STREAM_SHA256,
    BayesianClusterWeightPlan,
    FrozenHandTransform,
    PreregistrationError,
    assign_inner_folds,
    build_pairwise_preferences,
    choose_c_exact_tie_smaller,
    deterministic_top_five,
    expected_random_tie_ap5,
    fit_evidence_ranker,
    fit_nested_behavior_routes,
    frozen_pool_weight_stream_sha256,
    interaction_matrix,
)


def _feature_frame(rows: int) -> pd.DataFrame:
    values = np.arange(rows * len(FEATURES), dtype=float).reshape(rows, -1)
    return pd.DataFrame(values, columns=FEATURES)


def _ordinary_ap5(relevance: tuple[int, ...]) -> float:
    total = sum(relevance)
    if total == 0:
        return 0.0
    ranked = np.asarray(relevance[:5], dtype=int)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / min(total, 5))


def _binary_orders(size: int, relevant: int):
    for positions in itertools.combinations(range(size), relevant):
        result = [0] * size
        for position in positions:
            result[position] = 1
        yield tuple(result)


class FrozenConstantsTest(unittest.TestCase):
    def test_hashes_dimensions_and_grid_are_exact(self):
        self.assertEqual(
            DRAFT_SHA256,
            "54776c546377da214626828f71deed383302ceee95142073da7b031a0d1723e5",
        )
        self.assertEqual(
            PREREG_STATIC_AUDIT_SHA256,
            "a45c55aae9cfac2b4caafb99511815b38b4da9c87c0b431edb985d554e117b91",
        )
        self.assertEqual(len(FEATURES), 20)
        self.assertEqual(len(B001_FULL_FEATURES), 95)
        self.assertEqual(len(B001_TIME_FEATURES), 77)
        self.assertEqual(C_GRID, (0.01, 0.1, 1.0, 10.0))
        self.assertEqual(INNER_SPLIT_BASE_SEED, 8803)
        self.assertEqual(BOOTSTRAP_DRAWS, 5000)


class FrozenTransformTest(unittest.TestCase):
    def test_sequential_transform_is_finite_and_has_frozen_order(self):
        frame = _feature_frame(8)
        frame.loc[0, FEATURES[0]] = np.nan
        frame.loc[1, FEATURES[1]] = np.inf
        frame.loc[:, FEATURES[2]] = 7.0
        transform = FrozenHandTransform.fit(frame)
        result = transform.transform(frame)
        self.assertEqual(result.shape, (8, 20))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(transform.feature_names, FEATURES)
        self.assertEqual(transform.scale_iqr[2], 1.0)

    def test_changed_feature_order_is_rejected(self):
        with self.assertRaises(PreregistrationError):
            FrozenHandTransform.fit(_feature_frame(4), FEATURES[::-1])


class PairwiseRankerTest(unittest.TestCase):
    def test_interactions_have_exact_blocks(self):
        base = np.arange(60, dtype=float).reshape(3, 20)
        result = interaction_matrix(base, FAMILIES)
        self.assertEqual(result.shape, (3, 80))
        np.testing.assert_array_equal(result[:, :20], base)
        for row, family in enumerate(FAMILIES):
            for index, candidate in enumerate(FAMILIES):
                block = result[row, 20 + index * 20 : 40 + index * 20]
                np.testing.assert_array_equal(
                    block, base[row] if candidate == family else np.zeros(20)
                )

    def test_preferences_are_symmetric_and_query_weighted(self):
        base = np.arange(100, dtype=float).reshape(5, 20)
        preferences = build_pairwise_preferences(
            base,
            ["q1", "q1", "q1", "q1", "q2"],
            [True, True, False, False, True],
        )
        self.assertEqual(preferences.matrix.shape, (8, 20))
        self.assertEqual(int(preferences.target.sum()), 4)
        np.testing.assert_array_equal(preferences.matrix[:4], -preferences.matrix[4:])
        self.assertAlmostEqual(float(preferences.sample_weight.sum()), 1.0)
        receipt = preferences.query_receipt.set_index("pair_id")
        self.assertEqual(int(receipt.loc["q1", "preference_pairs"]), 4)
        self.assertEqual(int(receipt.loc["q2", "preference_pairs"]), 0)
        self.assertEqual(float(receipt.loc["q2", "total_symmetric_weight"]), 0.0)

    def test_synthetic_ranker_has_frozen_configuration(self):
        rng = np.random.default_rng(91)
        base = rng.normal(size=(12, 20))
        pair_ids = np.repeat(["a", "b", "c"], 4)
        evidence = np.tile([True, False, False, False], 3)
        preferences = build_pairwise_preferences(base, pair_ids, evidence)
        model = fit_evidence_ranker(preferences, 0.1)
        self.assertEqual(model.coef_.shape, (1, 20))
        self.assertFalse(model.fit_intercept)
        self.assertEqual(model.solver, "lbfgs")


class RandomTieMetricTest(unittest.TestCase):
    def test_closed_form_matches_all_55_small_block_count_cases(self):
        cases = 0
        for total_size in range(1, 5):
            block_shapes = [(total_size,)] + [
                (left, total_size - left) for left in range(1, total_size)
            ]
            for shape in block_shapes:
                for relevant_counts in itertools.product(
                    *(range(size + 1) for size in shape)
                ):
                    cases += 1
                    block_orders = [
                        tuple(_binary_orders(size, relevant))
                        for size, relevant in zip(shape, relevant_counts)
                    ]
                    observed = []
                    for chosen in itertools.product(*block_orders):
                        relevance = tuple(itertools.chain.from_iterable(chosen))
                        observed.append(_ordinary_ap5(relevance))
                    representative = tuple(
                        itertools.chain.from_iterable(block[0] for block in block_orders)
                    )
                    scores = tuple(
                        score
                        for score, size in zip(range(len(shape), 0, -1), shape)
                        for _ in range(size)
                    )
                    self.assertAlmostEqual(
                        expected_random_tie_ap5(representative, scores),
                        float(np.mean(observed)),
                        places=14,
                    )
        self.assertEqual(cases, 55)

    def test_top_five_uses_timestamp_and_rejects_unresolved_tie(self):
        frame = pd.DataFrame(
            {
                "pair_id": ["q"] * 6,
                "hand_id": ["h5", "h4", "h3", "h2", "h1", "h0"],
                "started_at": [5, 4, 3, 2, 1, 0],
                "score": [1, 1, 1, 1, 1, 0],
            }
        )
        top = deterministic_top_five(frame, "score")
        self.assertEqual(top["hand_id"].tolist(), ["h1", "h2", "h3", "h4", "h5"])
        frame.loc[0, "started_at"] = 4
        with self.assertRaises(PreregistrationError):
            deterministic_top_five(frame, "score")


class NestedRoutingTest(unittest.TestCase):
    def _pair_data(self):
        rows = []
        feature_rows = []
        for index in range(45):
            family = FAMILIES[index % 3]
            pair_id = f"q{index}"
            table_id = f"pool{index}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "table_id": table_id,
                    "behavior_family": family,
                    "outer_fold": index % 5,
                }
            )
            feature = {
                column: float((index + offset) % 11)
                for offset, column in enumerate(B001_FULL_FEATURES)
            }
            feature.update({"pair_id": pair_id, "table_id": table_id})
            feature_rows.append(feature)
        return pd.DataFrame(rows), pd.DataFrame(feature_rows)

    def test_inner_split_is_pool_held_out_and_router_is_cross_fitted(self):
        pairs, features = self._pair_data()
        first = assign_inner_folds(pairs, 0)
        second = assign_inner_folds(pairs, 0)
        np.testing.assert_array_equal(first["inner_fold"], second["inner_fold"])
        self.assertEqual(first.groupby("table_id")["inner_fold"].nunique().max(), 1)
        receipt = fit_nested_behavior_routes(
            first, features, features, view="full", outer_fold=0
        )
        self.assertEqual(len(receipt), len(first))
        self.assertNotIn("behavior_family", receipt.columns)
        self.assertTrue(set(receipt["predicted_family"]).issubset(FAMILIES))
        self.assertEqual(receipt["inner_fold"].nunique(), 3)


class BayesianSensitivityTest(unittest.TestCase):
    def _meta(self, view: str) -> pd.DataFrame:
        rows = []
        for fold in range(5):
            for index, family in enumerate(FAMILIES):
                for duplicate in range(2):
                    rows.append(
                        {
                            "pair_id": f"{view}_{fold}_{index}_{duplicate}",
                            "table_id": f"{view}_pool_{fold}_{duplicate}",
                            "outer_fold": fold,
                            "behavior_family": family,
                        }
                    )
        return pd.DataFrame(rows)

    def test_weights_are_positive_shared_and_sensitivity_semantics_are_exact(self):
        plans = {
            view: BayesianClusterWeightPlan.create(self._meta(view), view)
            for view in ("full", "early_to_late", "late_to_early")
        }
        plan = plans["full"]
        self.assertTrue((plan.query_weights > 0).all())
        # All three families sharing one pool receive the same draw weight.
        np.testing.assert_array_equal(
            plan.query_weights[:, 0], plan.query_weights[:, 2]
        )
        deltas = self._meta("full")
        deltas["delta"] = 0.0
        summary = plan.summarize(deltas)
        self.assertEqual(summary["interval_name"], SENSITIVITY_INTERVAL_NAME)
        self.assertFalse(summary["confidence_claimed"])
        self.assertFalse(summary["coverage_claimed"])
        self.assertTrue(
            summary["paired_query_level_delta"]["formed_before_weighting"]
        )
        self.assertEqual(
            summary["query_weighted_overall"]["sensitivity_lower_05"], 0.0
        )
        with_headers = frozen_pool_weight_stream_sha256(plans)
        without_headers = frozen_pool_weight_stream_sha256(
            plans, include_headers=False
        )
        self.assertEqual(len(with_headers), 64)
        self.assertEqual(len(without_headers), 64)
        self.assertNotEqual(with_headers, without_headers)

    def test_c_ties_choose_smaller_exactly(self):
        scores = {0.01: 0.4, 0.1: 0.5, 1.0: 0.5, 10.0: 0.3}
        self.assertEqual(choose_c_exact_tie_smaller(scores), 0.1)


if __name__ == "__main__":
    unittest.main()
