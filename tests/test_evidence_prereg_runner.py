import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import poker_attack.evidence_prereg as evidence_runner
from poker_attack.evidence_prereg import (
    CONSUMPTION_FILENAME,
    IMPLEMENTATION_ONLY_STATUS,
    InputBundle,
    VIEW_SPECS,
    _fit_outer_and_score,
    _gate_results,
    _select_c,
    run_authorized_validation,
)
from poker_attack.evidence_prereg_core import (
    B001_FULL_FEATURES,
    B001_TIME_FEATURES,
    C_GRID,
    DRAFT_SHA256,
    FAMILIES,
    FEATURES,
    PREREG_STATIC_AUDIT_SHA256,
    assign_inner_folds,
    fit_nested_behavior_routes,
)


class ExecutionLockTest(unittest.TestCase):
    def test_missing_authorization_stops_before_data_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            with self.assertRaises(PermissionError):
                run_authorized_validation(
                    root / "nonexistent_data",
                    work,
                    root / "missing_authorization.json",
                )
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())

    def test_valid_synthetic_permit_is_consumed_before_missing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            core_path = Path(evidence_runner.__file__).with_name(
                "evidence_prereg_core.py"
            )
            runner_path = Path(evidence_runner.__file__)
            core_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()
            runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest()
            implementation_audit = root / "implementation_audit.json"
            implementation_audit.write_text(
                json.dumps(
                    {
                        "status": "PASS_IMPLEMENTATION_PRESENT_UNEXECUTED",
                        "binding": {
                            "draft_sha256": DRAFT_SHA256,
                            "prereg_static_audit_sha256": PREREG_STATIC_AUDIT_SHA256,
                            "core_implementation_sha256": core_hash,
                            "runner_implementation_sha256": runner_hash,
                        },
                        "controls": {
                            "real_model_fit_performed": False,
                            "aggregate_performance_viewed": False,
                            "evaluation_rows_loaded": False,
                            "evaluation_scored": False,
                            "experiment_number_assigned": False,
                            "submission_created_or_modified": False,
                            "preregistration_executed": False,
                        },
                    }
                )
            )
            audit_hash = hashlib.sha256(
                implementation_audit.read_bytes()
            ).hexdigest()
            authorization = root / "result_authorization.json"
            authorization.write_text(
                json.dumps(
                    {
                        "status": "RESULT_BEARING_EXECUTION_APPROVED",
                        "binding": {
                            "draft_sha256": DRAFT_SHA256,
                            "prereg_static_audit_sha256": PREREG_STATIC_AUDIT_SHA256,
                            "core_implementation_sha256": core_hash,
                            "runner_implementation_sha256": runner_hash,
                            "implementation_static_audit_path": str(
                                implementation_audit
                            ),
                            "implementation_static_audit_sha256": audit_hash,
                        },
                        "permissions": {
                            "fit_preregistered_nested_b001_and_evidence_models_once": True,
                            "view_development_evidence_performance_once": True,
                            "load_evaluation_rows": False,
                            "score_evaluation_rows": False,
                            "create_or_modify_submission": False,
                            "assign_experiment_number": False,
                            "rerun_after_any_stop_or_failure": False,
                        },
                    }
                )
            )
            with self.assertRaises(FileNotFoundError):
                run_authorized_validation(
                    root / "nonexistent_data", work, authorization
                )
            self.assertTrue((work / CONSUMPTION_FILENAME).exists())


class SyntheticExecutorIntegrationTest(unittest.TestCase):
    def _bundle(self) -> InputBundle:
        pair_rows = []
        hand_rows = []
        full_feature_rows = []
        time_feature_rows = []
        rng = np.random.default_rng(728)
        for index in range(45):
            pair_id = f"q{index}"
            table_id = f"pool{index}"
            family = FAMILIES[index % 3]
            outer_fold = index % 5
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "table_id": table_id,
                    "behavior_family": family,
                    "outer_fold": outer_fold,
                    "predicted_family": family,
                    "early_to_late__predicted_family": family,
                    "late_to_early__predicted_family": family,
                }
            )
            full_features = {
                column: float((index + position) % 13)
                for position, column in enumerate(B001_FULL_FEATURES)
            }
            full_features.update({"pair_id": pair_id, "table_id": table_id})
            full_feature_rows.append(full_features)
            time_features = {
                column: float((index * 2 + position) % 17)
                for position, column in enumerate(B001_TIME_FEATURES)
            }
            time_features.update({"pair_id": pair_id, "table_id": table_id})
            time_feature_rows.append(time_features)
            for hand in range(4):
                features = rng.normal(size=len(FEATURES))
                if hand == 0:
                    features += 1.5
                row = {
                    "pair_id": pair_id,
                    "hand_id": f"h{index}_{hand}",
                    "started_at": index * 10 + hand,
                    "table_id": table_id,
                    "outer_fold": outer_fold,
                    "behavior_family": family,
                    "phase_progress": 0.2 + hand * 0.2,
                    "is_evidence": hand == 0,
                    "predicted_family": family,
                    "early_to_late__predicted_family": family,
                    "late_to_early__predicted_family": family,
                    "directed_signal": float(features[0]),
                    "soft_signal": float(features[1]),
                    "isolation_signal": float(features[2]),
                }
                row.update(dict(zip(FEATURES, features)))
                hand_rows.append(row)
        pair_meta = pd.DataFrame(pair_rows)
        full = pd.DataFrame(full_feature_rows)
        time = pd.DataFrame(time_feature_rows)
        return InputBundle(
            pair_meta=pair_meta,
            positive_hands=pd.DataFrame(hand_rows),
            full_pair_features=full,
            early_pair_features=time,
            late_pair_features=time.copy(),
            source_hashes={},
        )

    def test_nested_selection_and_outer_scoring_compose_on_synthetic_data(self):
        bundle = self._bundle()
        spec = VIEW_SPECS[0]
        assignments = assign_inner_folds(bundle.pair_meta, 0)
        routes = fit_nested_behavior_routes(
            assignments,
            bundle.full_pair_features,
            bundle.full_pair_features,
            view="full",
            outer_fold=0,
        )
        selected_c, inner_scores = _select_c(
            bundle, spec, assignments, routes
        )
        self.assertIn(selected_c, C_GRID)
        self.assertEqual(set(inner_scores), set(C_GRID))
        receipt, diagnostics = _fit_outer_and_score(
            bundle, spec, 0, selected_c
        )
        self.assertEqual(len(receipt), 9 * 4)
        self.assertTrue(receipt["eligible_query"].all())
        self.assertTrue(receipt[["h0_score", "l0_score", "l1_score"]].notna().all().all())
        self.assertEqual(diagnostics["validation_queries_eligible"], 9)
        self.assertEqual(diagnostics["unknown_pairs_used"], 0)

    def test_gate_order_stops_at_primary_without_rescue(self):
        def sensitivity(overall, lower, macro, family_delta, fold_delta):
            return {
                "query_weighted_overall": {
                    "point_delta": overall,
                    "sensitivity_lower_05": lower,
                },
                "unweighted_macro_family_mean": {"point_delta": macro},
                "family_weighted_mean": {
                    family: {"point_delta": family_delta} for family in FAMILIES
                },
                "outer_fold_point_delta": {
                    str(fold): fold_delta for fold in range(5)
                },
            }

        results = {
            "full": {
                "sensitivity": {
                    "l1_minus_h0": sensitivity(0.039, 0.01, 0.03, 0.01, 0.01),
                    "l1_minus_l0": sensitivity(0.01, 0.001, 0.006, 0.001, 0.001),
                }
            },
            "early_to_late": {
                "sensitivity": {
                    "l1_minus_h0": sensitivity(0.03, 0.01, 0.021, 0.001, 0.001)
                }
            },
            "late_to_early": {
                "sensitivity": {
                    "l1_minus_h0": sensitivity(0.03, 0.01, 0.021, 0.001, 0.001)
                }
            },
        }
        gates = _gate_results(results)
        self.assertFalse(gates["all_gates_passed"])
        self.assertEqual(gates["first_failed_gate"], "primary_full_period")
        self.assertFalse(gates["rescue_or_rerun_allowed"])

    def test_implementation_only_receipt_cannot_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            authorization = root / "implementation_only.json"
            authorization.write_text(
                json.dumps(
                    {
                        "status": IMPLEMENTATION_ONLY_STATUS,
                        "binding": {
                            "draft_sha256": DRAFT_SHA256,
                            "prereg_static_audit_sha256": PREREG_STATIC_AUDIT_SHA256,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(PermissionError, "implementation-only"):
                run_authorized_validation(
                    root / "nonexistent_data", work, authorization
                )
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())

    def test_wrong_binding_stops_without_consuming(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            authorization = root / "wrong_binding.json"
            authorization.write_text(
                json.dumps(
                    {
                        "status": "RESULT_BEARING_EXECUTION_APPROVED",
                        "binding": {
                            "draft_sha256": "0" * 64,
                            "prereg_static_audit_sha256": PREREG_STATIC_AUDIT_SHA256,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(PermissionError, "different preregistration"):
                run_authorized_validation(
                    root / "nonexistent_data", work, authorization
                )
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
