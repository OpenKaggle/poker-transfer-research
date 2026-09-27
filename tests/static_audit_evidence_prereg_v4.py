"""Source-only audit for the implementation-only EVP-DRAFT-04 bundle.

This script reads design/source/test files only. It never opens development,
evidence, pair-hand, time, evaluation, or submission data and never fits a
model or computes a performance metric.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from poker_attack.evidence_prereg_v4 import (
    CANONICAL_RELATIVE_PATHS,
    CONSUMPTION_FILENAME,
    FULL_STAGE_SOURCE_NAMES,
    STAGE_ARTIFACTS,
    TIME_STAGE_SOURCE_NAMES,
)
from poker_attack.evidence_prereg_v4_core import (
    C_GRID,
    HISTORICAL_CONTAINMENT_SHA256,
    REJECTED_V3_CORE_SHA256,
    REJECTED_V3_RUNNER_SHA256,
    STAGE_WEIGHT_STREAM_NO_HEADER_SHA256,
    STAGE_WEIGHT_STREAM_SHA256,
    V4_COVERAGE_CONTRACT_SHA256,
    V4_DRAFT_SHA256,
    V4_METHOD_AUDIT_SHA256,
    CoverageBookV4,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def direct_calls(node: ast.FunctionDef) -> list[str]:
    return [
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    work = root / "poker/work"
    core = root / "poker/src/poker_attack/evidence_prereg_v4_core.py"
    runner = root / "poker/src/poker_attack/evidence_prereg_v4.py"
    core_test = root / "poker/tests/test_evidence_prereg_v4_core.py"
    runner_test = root / "poker/tests/test_evidence_prereg_v4_runner.py"
    draft = root / CANONICAL_RELATIVE_PATHS["draft"]
    coverage_path = root / CANONICAL_RELATIVE_PATHS["coverage"]
    method_audit = root / CANONICAL_RELATIVE_PATHS["method_audit"]
    historical_containment = root / CANONICAL_RELATIVE_PATHS[
        "historical_containment"
    ]
    v3_core = root / "poker/src/poker_attack/evidence_prereg_core.py"
    v3_runner = root / "poker/src/poker_attack/evidence_prereg.py"
    implementation_authorization = (
        root / "poker/work/EVP_v4_implementation_authorization.json"
    )

    assert sha256(draft) == V4_DRAFT_SHA256
    assert sha256(coverage_path) == V4_COVERAGE_CONTRACT_SHA256
    assert sha256(method_audit) == V4_METHOD_AUDIT_SHA256
    assert sha256(historical_containment) == HISTORICAL_CONTAINMENT_SHA256
    assert sha256(v3_core) == REJECTED_V3_CORE_SHA256
    assert sha256(v3_runner) == REJECTED_V3_RUNNER_SHA256
    authorization = json.loads(implementation_authorization.read_text())
    assert authorization["decision"] == "V4_IMPLEMENTATION_ONLY_APPROVED"
    assert authorization["binding"]["v4_draft_sha256"] == V4_DRAFT_SHA256
    assert (
        authorization["binding"]["coverage_contract_sha256"]
        == V4_COVERAGE_CONTRACT_SHA256
    )
    assert authorization["binding"]["method_audit_sha256"] == V4_METHOD_AUDIT_SHA256
    assert authorization["binding"]["method_verdict"] == "PASS_METHOD_SPEC_ONLY"
    assert authorization["prohibited"]["result_bearing_execution"] is True

    coverage = CoverageBookV4.from_mapping(
        json.loads(coverage_path.read_text(encoding="utf-8"))
    )
    assert len(coverage.cells) == 72
    assert coverage.expected(
        "early_to_late", "outer_fold_family", "0/coordinated_isolation"
    ).as_dict() == {
        "eligible_queries": 7,
        "candidate_hands": 465,
        "evidence_hands": 17,
        "eligible_pools": 6,
    }
    assert coverage.expected(
        "early_to_late", "outer_fold_family", "4/coordinated_isolation"
    ).eligible_queries == 7

    runner_source = runner.read_text(encoding="utf-8")
    core_source = core.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    core_tree = ast.parse(core_source)
    imported_modules = {
        node.module
        for tree in (runner_tree, core_tree)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "poker_attack.evidence_prereg" not in imported_modules
    assert "poker_attack.evidence_prereg_core" not in imported_modules
    assert "evidence_prereg" not in imported_modules
    assert "evidence_prereg_core" not in imported_modules

    forbidden_runtime_inputs = (
        "evaluation_pairs.csv",
        "evaluation_pair_features",
        "evaluation_pair_hands",
        "sample_submission.csv",
        "submission.csv",
        "kaggle competitions submit",
    )
    assert all(token not in runner_source for token in forbidden_runtime_inputs)
    assert all(token not in core_source for token in forbidden_runtime_inputs)
    assert set(FULL_STAGE_SOURCE_NAMES).isdisjoint(TIME_STAGE_SOURCE_NAMES)

    auth = function(runner_tree, "_validate_result_execution_authorization_v4")
    auth_source = ast.get_source_segment(runner_source, auth)
    assert "_canonical_path(\"v4_containment\")" in auth_source
    assert "_canonical_path(\"v4_source_audit\")" in auth_source
    assert "authorization.get(\"binding\") != exact_binding" in auth_source
    assert "Path(binding" not in auth_source
    assert "open(binding" not in auth_source
    assert "resolve(binding" not in auth_source

    run = function(runner_tree, "run_authorized_validation_v4")
    run_calls = direct_calls(run)
    ordered = [
        "_validate_result_execution_authorization_v4",
        "_load_coverage_book_canonical",
        "_load_full_inputs_v4",
        "_run_stage_1_full_primary_v4",
        "_run_stage_2_attribution_v4",
        "_run_stage_3_time_v4",
    ]
    assert all(name in run_calls for name in ordered)
    assert [run_calls.index(name) for name in ordered] == sorted(
        run_calls.index(name) for name in ordered
    )
    stage_3 = function(runner_tree, "_run_stage_3_time_v4")
    callers = [
        node.name
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef)
        and "_load_time_inputs_v4" in direct_calls(node)
    ]
    assert callers == [stage_3.name]
    family_join_callers = [
        node.name
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef)
        and "_join_reporting_family_v4" in direct_calls(node)
    ]
    assert set(family_join_callers) == {
        "_persist_reporting_family_v4",
        "_run_stage_3_time_v4",
    }
    assert "_load_full_inputs_v4" not in family_join_callers
    assert "_load_time_inputs_v4" not in family_join_callers
    full_loader = function(runner_tree, "_load_full_inputs_v4")
    full_loader_source = ast.get_source_segment(runner_source, full_loader)
    assert '"phase_progress",' not in full_loader_source
    assert "b001_time_oof" not in full_loader_source
    assert "b001_early_features" not in full_loader_source
    assert "b001_late_features" not in full_loader_source

    selection = function(runner_tree, "_select_c_for_outer_v4")
    selection_source = ast.get_source_segment(runner_source, selection)
    assert "select_c_no_family_v4" in selection_source
    assert "macro_family" not in selection_source
    assert "behavior_family" not in selection_source
    assert C_GRID == (0.01, 0.1, 1.0, 10.0)

    stage_1 = function(runner_tree, "_run_stage_1_full_primary_v4")
    stage_1_source = ast.get_source_segment(runner_source, stage_1)
    assert "_fit_outer_l0_from_state_v4" not in stage_1_source
    assert "_assert_paths_absent(_later_stage_paths(work_dir, 1))" in stage_1_source
    stage_2 = function(runner_tree, "_run_stage_2_attribution_v4")
    stage_2_source = ast.get_source_segment(runner_source, stage_2)
    assert "_load_time_inputs_v4" not in stage_2_source
    assert "_assert_paths_absent(_later_stage_paths(work_dir, 2))" in stage_2_source

    assert STAGE_WEIGHT_STREAM_SHA256[("full",)] == (
        "a00923786962622bacfd63fb211670ef14eb657a4a96e5411274ff4034777a2e"
    )
    assert STAGE_WEIGHT_STREAM_NO_HEADER_SHA256[("full",)] == (
        "e57b0a7b15873641e429a964d79075a0f56e997b675caab8e9a2a7f40ab1f16d"
    )

    assert not (work / CONSUMPTION_FILENAME).exists()
    for names in STAGE_ARTIFACTS.values():
        for name in names:
            assert not (work / name).exists()
    v4_containment_path = root / CANONICAL_RELATIVE_PATHS["v4_containment"]
    if v4_containment_path.exists():
        v4_containment = json.loads(v4_containment_path.read_text(encoding="utf-8"))
        assert v4_containment["status"] == (
            "PASS_V4_IMPLEMENTATION_PRESENT_UNEXECUTED"
        )
        assert v4_containment["binding"]["v4_core_sha256"] == sha256(core)
        assert v4_containment["binding"]["v4_runner_sha256"] == sha256(runner)
    assert not (root / CANONICAL_RELATIVE_PATHS["v4_source_audit"]).exists()

    print(
        json.dumps(
            {
                "status": "PASS_V4_IMPLEMENTATION_PRESENT_UNEXECUTED",
                "binding": {
                    "v4_draft_sha256": V4_DRAFT_SHA256,
                    "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
                    "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
                    "historical_containment_sha256": HISTORICAL_CONTAINMENT_SHA256,
                    "rejected_v3_core_sha256": sha256(v3_core),
                    "rejected_v3_runner_sha256": sha256(v3_runner),
                    "v4_core_sha256": sha256(core),
                    "v4_runner_sha256": sha256(runner),
                    "v4_core_tests_sha256": sha256(core_test),
                    "v4_runner_tests_sha256": sha256(runner_test),
                    "implementation_authorization_sha256": sha256(
                        implementation_authorization
                    ),
                },
                "audit": {
                    "new_v4_files_only": True,
                    "canonical_path_constructed_by_executor": True,
                    "caller_selected_audit_path_dereferenced": False,
                    "c_selection_family_grouping": False,
                    "strict_stage_order": [0, 1, 2, 3],
                    "time_loader_callers": callers,
                    "post_score_family_join_callers": sorted(family_join_callers),
                    "metric_local_coverage_cells": len(coverage.cells),
                    "canonical_v4_containment_exists": v4_containment_path.exists(),
                    "full_stage_time_source_intersection": 0,
                    "stage_local_full_weight_hash_frozen": True,
                },
                "controls": {
                    "real_development_data_read": False,
                    "real_model_fit_performed": False,
                    "aggregate_performance_viewed": False,
                    "result_outputs_exist": False,
                    "one_run_consumption_receipt_exists": False,
                    "evaluation_rows_loaded": False,
                    "evaluation_scored": False,
                    "experiment_number_assigned": False,
                    "candidate_promoted": False,
                    "submission_created_or_modified": False,
                    "preregistration_executed": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
