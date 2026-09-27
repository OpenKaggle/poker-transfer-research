import itertools
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from poker_attack.evidence_prereg_v4_1_core import (
    B001_FULL_FEATURES,
    B001_TIME_FEATURES,
    BOOTSTRAP_DRAWS,
    C_GRID,
    FAMILIES,
    FEATURES,
    V4_COVERAGE_CONTRACT_SHA256,
    V4_DRAFT_SHA256,
    V4_METHOD_AUDIT_SHA256,
    BayesianClusterWeightPlanV4,
    CoverageBookV4,
    CoverageTupleV4,
    FrozenHandTransformV4,
    V4InvariantError,
    assign_inner_folds_v4,
    build_pairwise_preferences_v4,
    deterministic_top_five_v4,
    expected_random_tie_ap5_v4,
    fit_evidence_ranker_v4,
    fit_nested_behavior_routes_v4,
    frozen_stage_weight_stream_sha256_v4,
    interaction_matrix_v4,
    metric_tree_v4,
    observed_coverage_v4,
    paired_delta_frame_v4,
    select_c_no_family_v4,
    sensitivity_tree_v4,
)


ROOT = Path(__file__).resolve().parents[2]


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


def _selection_frame(values: list[float]) -> pd.DataFrame:
    size = len(values)
    return pd.DataFrame(
        {
            "pair_id": [f"q{i}" for i in range(size)],
            "table_id": [f"p{i}" for i in range(size)],
            "outer_fold": [i % 5 for i in range(size)],
            "query_ap5": values,
            "candidate_hands": [6] * size,
            "evidence_hands": [1] * size,
        }
    )


def _mini_reporting_frame() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for family in FAMILIES:
            key = f"{fold}_{family}"
            rows.append(
                {
                    "pair_id": f"q_{key}",
                    "hand_id": f"h_{key}",
                    "table_id": f"p_{key}",
                    "outer_fold": fold,
                    "behavior_family": family,
                    "is_evidence": True,
                }
            )
    return pd.DataFrame(rows)


def _mini_book() -> CoverageBookV4:
    cells = {}
    for view in ("full", "early_to_late", "late_to_early"):
        cells[(view, "overall", "overall")] = CoverageTupleV4(15, 15, 15, 15)
        for family in FAMILIES:
            cells[(view, "family", family)] = CoverageTupleV4(5, 5, 5, 5)
        for fold in range(5):
            cells[(view, "outer_fold", str(fold))] = CoverageTupleV4(3, 3, 3, 3)
            for family in FAMILIES:
                cells[(view, "outer_fold_family", f"{fold}/{family}")] = (
                    CoverageTupleV4(1, 1, 1, 1)
                )
    return CoverageBookV4(cells)


class FrozenConstantsV4Test(unittest.TestCase):
    def test_hashes_dimensions_and_grid_are_exact(self):
        self.assertEqual(
            V4_DRAFT_SHA256,
            "3a9f20936f4173f72cb4ff867d296eb2945b43ca40cac0ee587b7fb4b83da22c",
        )
        self.assertEqual(
            V4_COVERAGE_CONTRACT_SHA256,
            "4e821092161be0b0c80892af41ed61718e0a4930562b748a826da8a8bf9001e6",
        )
        self.assertEqual(
            V4_METHOD_AUDIT_SHA256,
            "7ab04d40cff489323ee85f27484b93d57e6efdce06b09a78eeccc9cd0d22e913",
        )
        self.assertEqual(len(FEATURES), 20)
        self.assertEqual(len(B001_FULL_FEATURES), 95)
        self.assertEqual(len(B001_TIME_FEATURES), 77)
        self.assertEqual(C_GRID, (0.01, 0.1, 1.0, 10.0))
        self.assertEqual(BOOTSTRAP_DRAWS, 5000)

    def test_actual_design_coverage_contract_has_exactly_72_cells(self):
        contract = json.loads(
            (ROOT / "poker/work/EVP_v4_metric_coverage_contract.json").read_text()
        )
        book = CoverageBookV4.from_mapping(contract)
        self.assertEqual(len(book.cells), 72)
        self.assertEqual(
            book.expected("early_to_late", "outer_fold_family", "0/coordinated_isolation"),
            CoverageTupleV4(7, 465, 17, 6),
        )
        self.assertEqual(
            book.expected("early_to_late", "outer_fold_family", "4/coordinated_isolation"),
            CoverageTupleV4(7, 458, 18, 7),
        )


class TransformAndPairwiseV4Test(unittest.TestCase):
    def test_transform_interactions_and_ranker_are_frozen(self):
        frame = _feature_frame(12)
        frame.loc[0, FEATURES[0]] = np.nan
        frame.loc[1, FEATURES[1]] = np.inf
        frame.loc[:, FEATURES[2]] = 7.0
        transform = FrozenHandTransformV4.fit(frame)
        base = transform.transform(frame)
        self.assertEqual(base.shape, (12, 20))
        self.assertTrue(np.isfinite(base).all())
        self.assertEqual(transform.scale_iqr[2], 1.0)
        routes = np.asarray(FAMILIES * 4)
        interaction = interaction_matrix_v4(base, routes)
        self.assertEqual(interaction.shape, (12, 80))
        preferences = build_pairwise_preferences_v4(
            interaction,
            np.repeat(["a", "b", "c"], 4),
            np.tile([True, False, False, False], 3),
        )
        model = fit_evidence_ranker_v4(preferences, 0.1)
        self.assertEqual(model.coef_.shape, (1, 80))
        self.assertFalse(model.fit_intercept)

    def test_zero_preference_query_is_retained_and_unjudged_is_not_negative(self):
        base = np.arange(100, dtype=float).reshape(5, 20)
        preferences = build_pairwise_preferences_v4(
            base,
            ["q1", "q1", "q1", "q1", "q2"],
            [True, True, False, False, True],
        )
        receipt = preferences.query_receipt.set_index("pair_id")
        self.assertEqual(int(receipt.loc["q1", "preference_pairs"]), 4)
        self.assertEqual(int(receipt.loc["q2", "preference_pairs"]), 0)
        self.assertFalse(bool(receipt["unjudged_is_ground_truth_negative"].any()))
        self.assertAlmostEqual(float(preferences.sample_weight.sum()), 1.0)


class RandomTieV4Test(unittest.TestCase):
    def test_closed_form_matches_all_55_small_block_count_cases(self):
        cases = 0
        for total_size in range(1, 5):
            shapes = [(total_size,)] + [
                (left, total_size - left) for left in range(1, total_size)
            ]
            for shape in shapes:
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
                        observed.append(
                            _ordinary_ap5(tuple(itertools.chain.from_iterable(chosen)))
                        )
                    representative = tuple(
                        itertools.chain.from_iterable(block[0] for block in block_orders)
                    )
                    scores = tuple(
                        score
                        for score, size in zip(range(len(shape), 0, -1), shape)
                        for _ in range(size)
                    )
                    self.assertAlmostEqual(
                        expected_random_tie_ap5_v4(representative, scores),
                        float(np.mean(observed)),
                        places=14,
                    )
        self.assertEqual(cases, 55)

    def test_top_five_uses_timestamp_and_rejects_family_or_unresolved_tie(self):
        frame = pd.DataFrame(
            {
                "pair_id": ["q"] * 6,
                "hand_id": ["h5", "h4", "h3", "h2", "h1", "h0"],
                "started_at": [5, 4, 3, 2, 1, 0],
                "score": [1, 1, 1, 1, 1, 0],
            }
        )
        top = deterministic_top_five_v4(frame, "score")
        self.assertEqual(top["hand_id"].tolist(), ["h1", "h2", "h3", "h4", "h5"])
        with self.assertRaises(V4InvariantError):
            deterministic_top_five_v4(
                frame.assign(behavior_family="directed_transfer"), "score"
            )
        frame.loc[0, "started_at"] = 4
        with self.assertRaises(V4InvariantError):
            deterministic_top_five_v4(frame, "score")


class NoFamilySelectionV4Test(unittest.TestCase):
    def test_query_mean_selects_c_and_exact_tie_chooses_smaller(self):
        frames = {
            0.01: _selection_frame([0.2, 0.4]),
            0.1: _selection_frame([0.5, 0.5]),
            1.0: _selection_frame([0.4, 0.6]),
            10.0: _selection_frame([0.1, 0.2]),
        }
        selected, means = select_c_no_family_v4(frames)
        self.assertEqual(selected, 0.1)
        self.assertEqual(means[0.1], means[1.0])

    def test_any_family_label_behavior_or_route_field_is_rejected(self):
        for column in (
            "behavior_family",
            "family_target",
            "label",
            "predicted_route",
        ):
            frames = {
                c_value: _selection_frame([0.2, 0.4]).assign(**{column: "x"})
                for c_value in C_GRID
            }
            with self.subTest(column=column), self.assertRaises(V4InvariantError):
                select_c_no_family_v4(frames)


class NestedRoutingV4Test(unittest.TestCase):
    def test_router_is_cross_fitted_and_receipt_has_no_true_family(self):
        pairs = []
        features = []
        for index in range(45):
            family = FAMILIES[index % 3]
            pair_id = f"q{index}"
            table_id = f"pool{index}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "table_id": table_id,
                    "behavior_family": family,
                    "outer_fold": index % 5,
                }
            )
            row = {
                column: float((index + offset) % 11)
                for offset, column in enumerate(B001_FULL_FEATURES)
            }
            row.update({"pair_id": pair_id, "table_id": table_id})
            features.append(row)
        pairs = pd.DataFrame(pairs)
        features = pd.DataFrame(features)
        assignments = assign_inner_folds_v4(pairs, 0)
        receipt = fit_nested_behavior_routes_v4(
            assignments, features, features, view="full", outer_fold=0
        )
        self.assertEqual(len(receipt), len(assignments))
        self.assertNotIn("behavior_family", receipt)
        self.assertEqual(receipt["inner_fold"].nunique(), 3)
        self.assertTrue(set(receipt["predicted_family"]).issubset(FAMILIES))


class CoverageAndSensitivityV4Test(unittest.TestCase):
    def test_every_metric_node_embeds_local_coverage(self):
        reporting = _mini_reporting_frame()
        query = reporting[
            ["pair_id", "outer_fold", "behavior_family"]
        ].copy()
        query["query_ap5"] = np.linspace(0.1, 0.9, len(query))
        tree = metric_tree_v4(query, reporting, "full", _mini_book())
        self.assertEqual(tree["overall"]["coverage"]["eligible_queries"], 15)
        self.assertEqual(
            tree["outer_fold_family"]["0/directed_transfer"]["coverage"],
            CoverageTupleV4(1, 1, 1, 1).as_dict(),
        )
        self.assertIn("coverage", tree["unweighted_macro_family"])

    def test_exact_coverage_mismatch_stops_before_metric(self):
        frame = _mini_reporting_frame()
        book = _mini_book()
        self.assertEqual(observed_coverage_v4(frame).eligible_queries, 15)
        with self.assertRaises(V4InvariantError):
            book.validate(
                "full",
                "overall",
                "overall",
                CoverageTupleV4(14, 15, 15, 15),
            )

    def test_paired_weight_sensitivity_shares_draws_and_embeds_coverage(self):
        reporting = _mini_reporting_frame()
        meta = reporting[["pair_id", "table_id", "outer_fold"]].copy()
        plan = BayesianClusterWeightPlanV4.create(meta, "full")
        self.assertTrue((plan.query_weights > 0).all())
        stream = frozen_stage_weight_stream_sha256_v4(
            {"full": plan}, ("full",)
        )
        no_header = frozen_stage_weight_stream_sha256_v4(
            {"full": plan}, ("full",), include_headers=False
        )
        self.assertEqual(len(stream), 64)
        self.assertNotEqual(stream, no_header)
        method_a = meta.copy()
        method_b = meta.copy()
        method_a["query_ap5"] = np.linspace(0.2, 0.8, len(meta))
        method_b["query_ap5"] = method_a["query_ap5"] - 0.05
        delta = paired_delta_frame_v4(method_a, method_b)
        summary = sensitivity_tree_v4(
            delta, reporting, "full", plan, _mini_book()
        )
        self.assertFalse(summary["confidence_claimed"])
        self.assertFalse(summary["coverage_claimed"])
        self.assertAlmostEqual(summary["overall"]["point_delta"], 0.05)
        self.assertIn("coverage", summary["family"][FAMILIES[0]])


if __name__ == "__main__":
    unittest.main()
