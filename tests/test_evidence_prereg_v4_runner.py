import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import poker_attack.evidence_prereg_v4 as runner
from poker_attack.evidence_prereg_v4 import (
    CANONICAL_RELATIVE_PATHS,
    CONSUMPTION_FILENAME,
    FULL_STAGE_SOURCE_NAMES,
    FULL_SPEC,
    IMPLEMENTATION_ONLY_STATUS,
    RESULT_EXECUTION_STATUS,
    TIME_STAGE_SOURCE_NAMES,
    FullBundleV4,
    _fit_outer_l0_from_state_v4,
    _fit_outer_l1_h0_v4,
    _persist_nested_routes_v4,
    _select_c_for_outer_v4,
    run_authorized_validation_v4,
)
from poker_attack.evidence_prereg_v4_core import (
    B001_FULL_FEATURES,
    C_GRID,
    FAMILIES,
    FEATURES,
    HISTORICAL_CONTAINMENT_SHA256,
    REJECTED_V3_CORE_SHA256,
    REJECTED_V3_RUNNER_SHA256,
    V4_COVERAGE_CONTRACT_SHA256,
    V4_DRAFT_SHA256,
    V4_METHOD_AUDIT_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceSeparationV4Test(unittest.TestCase):
    def test_v4_never_imports_rejected_v3_modules(self):
        for path in (
            Path(runner.__file__),
            Path(runner.__file__).with_name("evidence_prereg_v4_core.py"),
        ):
            tree = ast.parse(path.read_text())
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            self.assertNotIn("poker_attack.evidence_prereg", imported)
            self.assertNotIn("poker_attack.evidence_prereg_core", imported)
            self.assertNotIn("evidence_prereg", imported)
            self.assertNotIn("evidence_prereg_core", imported)

    def test_full_stage_source_allowlist_excludes_every_time_source(self):
        self.assertTrue(set(TIME_STAGE_SOURCE_NAMES).isdisjoint(FULL_STAGE_SOURCE_NAMES))
        self.assertNotIn("b001_time_oof", FULL_STAGE_SOURCE_NAMES)
        self.assertNotIn("b001_early_features", FULL_STAGE_SOURCE_NAMES)
        self.assertNotIn("b001_late_features", FULL_STAGE_SOURCE_NAMES)

    def test_time_loader_is_called_only_inside_stage_3(self):
        source = Path(runner.__file__).read_text()
        tree = ast.parse(source)
        callers = []
        for function in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_load_time_inputs_v4"
                for call in ast.walk(function)
            ):
                callers.append(function.name)
        self.assertEqual(callers, ["_run_stage_3_time_v4"])


class AuthorizationLockV4Test(unittest.TestCase):
    def test_missing_authorization_stops_without_consuming(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            with self.assertRaises(PermissionError):
                run_authorized_validation_v4(
                    root / "no_data", work, root / "missing.json"
                )
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())

    def test_implementation_only_stops_before_any_canonical_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            malicious = root / "must_not_be_opened.json"
            authorization = root / "implementation_only.json"
            authorization.write_text(
                json.dumps(
                    {
                        "status": IMPLEMENTATION_ONLY_STATUS,
                        "binding": {"implementation_static_audit_path": str(malicious)},
                    }
                )
            )
            with self.assertRaisesRegex(PermissionError, "implementation-only"):
                run_authorized_validation_v4(root / "no_data", work, authorization)
            self.assertFalse(malicious.exists())
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())

    def _canonical_project(self, root: Path):
        for key in (
            "draft",
            "coverage",
            "method_audit",
            "historical_containment",
            "v3_rejection",
        ):
            source = ROOT / CANONICAL_RELATIVE_PATHS[key]
            target = root / CANONICAL_RELATIVE_PATHS[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        core_sha = _sha(Path(runner.__file__).with_name("evidence_prereg_v4_core.py"))
        runner_sha = _sha(Path(runner.__file__))
        required_binding = {
            "v4_draft_sha256": V4_DRAFT_SHA256,
            "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
            "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
            "historical_containment_sha256": HISTORICAL_CONTAINMENT_SHA256,
            "v4_core_sha256": core_sha,
            "v4_runner_sha256": runner_sha,
        }
        controls = {
            "real_development_data_read": False,
            "real_model_fit_performed": False,
            "aggregate_performance_viewed": False,
            "evaluation_rows_loaded": False,
            "evaluation_scored": False,
            "experiment_number_assigned": False,
            "candidate_promoted": False,
            "submission_created_or_modified": False,
            "preregistration_executed": False,
        }
        containment = root / CANONICAL_RELATIVE_PATHS["v4_containment"]
        containment.write_text(
            json.dumps(
                {
                    "status": "PASS_V4_IMPLEMENTATION_PRESENT_UNEXECUTED",
                    "binding": required_binding,
                    "controls": controls,
                },
                sort_keys=True,
            )
        )
        source_audit = root / CANONICAL_RELATIVE_PATHS["v4_source_audit"]
        source_audit.write_text(
            json.dumps(
                {
                    "status": "PASS_V4_SOURCE_AUDIT_UNEXECUTED",
                    "binding": required_binding,
                    "controls": controls,
                },
                sort_keys=True,
            )
        )
        exact_binding = {
            "v4_draft_sha256": V4_DRAFT_SHA256,
            "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
            "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
            "historical_containment_path": CANONICAL_RELATIVE_PATHS[
                "historical_containment"
            ],
            "historical_containment_sha256": HISTORICAL_CONTAINMENT_SHA256,
            "rejected_v3_core_sha256": REJECTED_V3_CORE_SHA256,
            "rejected_v3_runner_sha256": REJECTED_V3_RUNNER_SHA256,
            "v4_containment_path": CANONICAL_RELATIVE_PATHS["v4_containment"],
            "v4_containment_sha256": _sha(containment),
            "v4_source_audit_path": CANONICAL_RELATIVE_PATHS["v4_source_audit"],
            "v4_source_audit_sha256": _sha(source_audit),
            "v4_core_sha256": core_sha,
            "v4_runner_sha256": runner_sha,
        }
        return exact_binding

    def test_valid_synthetic_documents_consume_before_missing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = self._canonical_project(root)
            work = root / "poker/work"
            authorization = root / "result_authorization.json"
            authorization.write_text(
                json.dumps(
                    {
                        "status": RESULT_EXECUTION_STATUS,
                        "binding": binding,
                        "permissions": {
                            "run_preregistered_development_stages_once": True,
                            "load_or_score_evaluation": False,
                            "assign_experiment_number": False,
                            "promote_candidate": False,
                            "create_or_modify_submission": False,
                            "rerun_after_stop_or_failure": False,
                        },
                    }
                )
            )
            with patch.object(runner, "_project_root", return_value=root):
                with self.assertRaises(FileNotFoundError):
                    run_authorized_validation_v4(root / "no_data", work, authorization)
            self.assertTrue((work / CONSUMPTION_FILENAME).exists())

    def test_extra_caller_selected_audit_path_is_rejected_without_dereference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = self._canonical_project(root)
            malicious = root / "caller_selected.json"
            binding["implementation_static_audit_path"] = str(malicious)
            work = root / "poker/work"
            authorization = root / "result_authorization.json"
            authorization.write_text(
                json.dumps(
                    {
                        "status": RESULT_EXECUTION_STATUS,
                        "binding": binding,
                        "permissions": {
                            "run_preregistered_development_stages_once": True,
                            "load_or_score_evaluation": False,
                            "assign_experiment_number": False,
                            "promote_candidate": False,
                            "create_or_modify_submission": False,
                            "rerun_after_stop_or_failure": False,
                        },
                    }
                )
            )
            with patch.object(runner, "_project_root", return_value=root):
                with self.assertRaisesRegex(PermissionError, "exact canonical"):
                    run_authorized_validation_v4(root / "no_data", work, authorization)
            self.assertFalse(malicious.exists())
            self.assertFalse((work / CONSUMPTION_FILENAME).exists())


class SequentialBarrierV4Test(unittest.TestCase):
    def _run_with_mocked_stages(self, stage_1, stage_2=None, stage_3=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            authorization = root / "authorization.json"
            authorization.write_text("{}")
            with (
                patch.object(runner, "_validate_result_execution_authorization_v4", return_value={"status": RESULT_EXECUTION_STATUS}),
                patch.object(runner, "_load_coverage_book_canonical", return_value=MagicMock()),
                patch.object(runner, "_load_full_inputs_v4", return_value=MagicMock()),
                patch.object(runner, "_run_stage_1_full_primary_v4", return_value=stage_1) as first,
                patch.object(runner, "_run_stage_2_attribution_v4", return_value=stage_2) as second,
                patch.object(runner, "_run_stage_3_time_v4", return_value=stage_3) as third,
            ):
                result = run_authorized_validation_v4(root / "data", work, authorization)
                return result, first.call_count, second.call_count, third.call_count

    def test_primary_failure_never_calls_attribution_or_time(self):
        stopped = SimpleNamespace(passed=False, report={"status": "STOP_PRIMARY"})
        result, first, second, third = self._run_with_mocked_stages(stopped)
        self.assertEqual(result["status"], "STOP_PRIMARY")
        self.assertEqual((first, second, third), (1, 0, 0))

    def test_attribution_failure_never_calls_time(self):
        stage_1 = SimpleNamespace(passed=True, report={"status": "PASS_STAGE_1"})
        stage_2 = SimpleNamespace(passed=False, report={"status": "STOP_ATTRIBUTION"})
        result, first, second, third = self._run_with_mocked_stages(stage_1, stage_2)
        self.assertEqual(result["status"], "STOP_ATTRIBUTION")
        self.assertEqual((first, second, third), (1, 1, 0))


class SyntheticModelCompositionV4Test(unittest.TestCase):
    def _bundle(self) -> FullBundleV4:
        pair_rows = []
        hand_rows = []
        feature_rows = []
        rng = np.random.default_rng(4804)
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
                }
            )
            feature = {
                column: float((index + offset) % 13)
                for offset, column in enumerate(B001_FULL_FEATURES)
            }
            feature.update({"pair_id": pair_id, "table_id": table_id})
            feature_rows.append(feature)
            for hand in range(4):
                values = rng.normal(size=len(FEATURES))
                if hand == 0:
                    values += 1.25
                row = {
                    "pair_id": pair_id,
                    "hand_id": f"h{index}_{hand}",
                    "started_at": index * 10 + hand,
                    "table_id": table_id,
                    "outer_fold": outer_fold,
                    "is_evidence": hand == 0,
                    "directed_signal": float(values[0]),
                    "soft_signal": float(values[1]),
                    "isolation_signal": float(values[2]),
                }
                row.update(dict(zip(FEATURES, values)))
                hand_rows.append(row)
        return FullBundleV4(
            pair_meta=pd.DataFrame(pair_rows),
            hands=pd.DataFrame(hand_rows),
            full_pair_features=pd.DataFrame(feature_rows),
            source_hashes={},
        )

    def test_full_nested_selection_l1_h0_and_l0_compose_without_heldout_family(self):
        bundle = self._bundle()
        with tempfile.TemporaryDirectory() as temporary:
            route_dir = Path(temporary) / "routes"
            routes, receipt = _persist_nested_routes_v4(
                bundle, None, FULL_SPEC, 0, route_dir
            )
            self.assertFalse(receipt["true_family_persisted"])
            selected_c, means, selection = _select_c_for_outer_v4(
                bundle, None, FULL_SPEC, 0, routes
            )
            self.assertIn(selected_c, C_GRID)
            self.assertEqual(set(means), set(C_GRID))
            self.assertFalse(bool(selection["family_field_used"].iloc[0]))
            score, state, diagnostics = _fit_outer_l1_h0_v4(
                bundle, None, FULL_SPEC, 0, selected_c
            )
            self.assertNotIn("behavior_family", score)
            self.assertTrue(score[["h0_score", "l1_score"]].notna().all().all())
            self.assertFalse(diagnostics["held_out_true_family_used_for_score"])
            l0, l0_diagnostics = _fit_outer_l0_from_state_v4(state)
            self.assertNotIn("behavior_family", l0)
            self.assertTrue(l0["l0_score"].notna().all())
            self.assertTrue(l0_diagnostics["matched_c"])


if __name__ == "__main__":
    unittest.main()
