"""Source-only implementation audit for EVP-DRAFT-04.1.

Reads versioned design/source/test files only. It never opens competition data,
fits a model, or computes a competition metric.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import poker_attack.evidence_prereg_v4_1 as runner
from poker_attack.evidence_prereg_v4_1 import (
    CANONICAL_RELATIVE_PATHS,
    EXCEPTION_STOP_FILENAME,
    ExecutionStateV4_1,
    STAGE_ARTIFACTS,
)
from poker_attack.evidence_prereg_v4_1_core import (
    FROZEN_RUNTIME_VERSIONS,
    FROZEN_SCHEMA_SHA256,
    V4_1_DELTA_SHA256,
    V4_COVERAGE_CONTRACT_SHA256,
    V4_DRAFT_SHA256,
    V4_METHOD_AUDIT_SHA256,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_source(source: str, tree: ast.Module, name: str) -> str:
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, node)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    work = root / "poker/work"
    core = root / "poker/src/poker_attack/evidence_prereg_v4_1_core.py"
    runner_path = root / "poker/src/poker_attack/evidence_prereg_v4_1.py"
    core_test = root / "poker/tests/test_evidence_prereg_v4_1_core.py"
    runner_test = root / "poker/tests/test_evidence_prereg_v4_1_runner.py"

    assert sha256(root / CANONICAL_RELATIVE_PATHS["base_draft"]) == V4_DRAFT_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["delta"]) == V4_1_DELTA_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["coverage"]) == V4_COVERAGE_CONTRACT_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["method_audit"]) == V4_METHOD_AUDIT_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["schema"]) == FROZEN_SCHEMA_SHA256
    assert FROZEN_RUNTIME_VERSIONS == {
        "python": "3.13.2", "numpy": "2.4.6", "pandas": "3.0.3",
        "scikit-learn": "1.8.0", "scipy": "1.17.1", "joblib": "1.5.3",
        "threadpoolctl": "3.6.0", "pyarrow": "24.0.0",
    }

    frozen_v4 = {
        "poker/src/poker_attack/evidence_prereg_v4_core.py": "1d02aa88b4a690aa088ea1dcdfda32ead25acb2e9236ccb7457c17e0a484a62d",
        "poker/src/poker_attack/evidence_prereg_v4.py": "4737f3c4983b37042d30d801dc0407625d7dc95a155a97bcb9f92417cfd59098",
        "poker/tests/test_evidence_prereg_v4_core.py": "fa1df57317f09dd2a6de09bfcc9b1d2fc09a1ed7d53351e8fdc0cfdab29d5304",
        "poker/tests/test_evidence_prereg_v4_runner.py": "bbb3a6ca2c8dc73010aaa68e994dce3f9292569e02c3ea6bbf24e3a068c42224",
        "poker/tests/static_audit_evidence_prereg_v4.py": "3c6b77d9bc15a92ea8a3874d4af99770f1cab31696564a7bedbd02f008703711",
        "poker/work/EVP_v4_implementation_containment_audit.json": "f8a2e27b4e41044f165badd05bc4f96d4883d6a7204d1f59f02a18292ac9e232",
    }
    for relative, expected in frozen_v4.items():
        assert sha256(root / relative) == expected

    runner_source = runner_path.read_text(encoding="utf-8")
    core_source = core.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    ast.parse(core_source)

    run = function_source(runner_source, runner_tree, "run_authorized_validation_v4_1")
    assert run.index("_pre_data_runtime_and_dependency_guard_v4_1") < run.index(
        "_validate_result_execution_authorization_v4"
    ) < run.index("_load_full_inputs_v4")
    assert "except BaseException as error" in run
    assert "_write_exception_terminal_v4_1" in run

    publisher = function_source(runner_source, runner_tree, "_publish_exclusive_v4_1")
    assert "os.link(temporary, path)" in publisher
    assert ".replace(" not in publisher
    assert "FileExistsError" in publisher

    stage_1 = function_source(runner_source, runner_tree, "_run_stage_1_full_primary_v4")
    assert stage_1.index("selection_binding = _raw_json_receipt_v4_1") < stage_1.index("metric_tree_v4")
    assert stage_1.index("pairwise_binding = _raw_parquet_receipt_v4_1") < stage_1.index("metric_tree_v4")
    assert stage_1.index("score_sha256 = _sha256(score_path)") < stage_1.index("_persist_reporting_family_v4")
    assert stage_1.index("top_sha256 = _sha256(top_path)") < stage_1.index("_persist_reporting_family_v4")

    stage_2 = function_source(runner_source, runner_tree, "_run_stage_2_attribution_v4")
    assert "_reload_stage_1_states_v4_1" in stage_2
    assert "stage_1.states" not in stage_2
    reloader = function_source(runner_source, runner_tree, "_reload_stage_1_states_v4_1")
    assert "_read_frozen_json_receipt_v4_1" in reloader
    assert "FrozenHandTransformV4.from_receipt" in reloader

    stage_3 = function_source(runner_source, runner_tree, "_run_stage_3_time_v4")
    assert stage_3.index("eligibility_binding = _raw_parquet_receipt_v4_1") < stage_3.index("_fit_outer_l1_h0_v4")
    assert stage_3.index("pairwise_binding = _raw_parquet_receipt_v4_1") < stage_3.index("metric_tree_v4")
    assert stage_3.index("top_sha256 = _sha256(top_path)") < stage_3.index("_join_reporting_family_v4")

    structural = function_source(runner_source, runner_tree, "_validate_structural_coverage_v4")
    readiness = function_source(runner_source, runner_tree, "_validate_pairwise_readiness_v4")
    assert "behavior_family" not in structural
    assert "behavior_family" not in readiness
    assert "_validate_post_reporting_zero_preference_family_v4" in stage_3

    required_controls = {
        f"stage_{stage}_{field}"
        for stage, fields in {
            1: ("development_data_access", "route_fit", "l1_model_fit", "l0_model_fit", "score_serialization", "aggregate_metric_computation", "family_reporting_join", "gate_evaluated", "gate_passed"),
            2: ("new_data_source_access", "frozen_receipt_reload", "route_fit", "l1_model_fit", "l0_model_fit", "score_serialization", "aggregate_metric_computation", "family_reporting_join", "gate_evaluated", "gate_passed"),
            3: ("time_data_access", "route_fit", "l1_model_fit", "l0_model_fit", "score_serialization", "aggregate_metric_computation", "family_reporting_join", "gate_evaluated", "gate_passed"),
        }.items()
        for field in fields
    }
    required_controls.update({
        "real_model_fit_performed", "aggregate_performance_viewed",
        "evaluation_rows_loaded", "evaluation_scored", "unknown_labels_assigned",
        "experiment_number_assigned", "candidate_promoted",
        "submission_created_or_modified", "rerun_allowed",
    })
    controls = ExecutionStateV4_1().terminal_controls()
    assert required_controls.issubset(controls)
    assert all(type(value) is bool for value in controls.values())

    forbidden = (
        "evaluation_pairs.csv", "sample_submission.csv", "submission.csv",
        "kaggle competitions submit", "subprocess.", "requests.", "urllib.",
    )
    assert all(token not in runner_source for token in forbidden)
    assert all(token not in core_source for token in forbidden)
    assert not (work / runner.CONSUMPTION_FILENAME).exists()
    assert not (work / EXCEPTION_STOP_FILENAME).exists()
    for names in STAGE_ARTIFACTS.values():
        for name in names:
            assert not (work / name).exists()
    assert not (root / CANONICAL_RELATIVE_PATHS["v4_1_source_audit"]).exists()

    print(json.dumps({
        "status": "PASS_V4_1_IMPLEMENTATION_PRESENT_UNEXECUTED",
        "binding": {
            "v4_draft_sha256": V4_DRAFT_SHA256,
            "v4_1_delta_sha256": V4_1_DELTA_SHA256,
            "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
            "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
            "schema_sha256": FROZEN_SCHEMA_SHA256,
            "v4_1_core_sha256": sha256(core),
            "v4_1_runner_sha256": sha256(runner_path),
            "v4_1_core_tests_sha256": sha256(core_test),
            "v4_1_runner_tests_sha256": sha256(runner_test),
        },
        "repairs": {
            "pre_data_runtime_fail_closed": True,
            "independent_pre_aggregate_raw_receipts": True,
            "stage_2_frozen_receipt_reload_only": True,
            "append_only_and_exception_terminal_stop": True,
            "complete_boolean_terminal_controls": True,
            "post_hash_reporting_family_only": True,
        },
        "controls": {
            "real_development_data_read": False,
            "real_model_fit_performed": False,
            "aggregate_performance_viewed": False,
            "evaluation_rows_loaded": False,
            "evaluation_scored": False,
            "experiment_number_assigned": False,
            "candidate_promoted": False,
            "submission_created_or_modified": False,
            "preregistration_executed": False,
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
