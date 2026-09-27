"""Source-only implementation check for the three EVP-DRAFT-04.2 repairs.

This script reads versioned design/source/test files only. It does not open
competition data, fit a model, calculate a metric, or issue an audit verdict.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import poker_attack.evidence_prereg_v4_2 as runner
from poker_attack.evidence_prereg_v4_2 import (
    CANONICAL_RELATIVE_PATHS,
    CANONICAL_RESULT_ROOT_RELATIVE,
    CONSUMPTION_CLAIM_FILENAME,
    CONSUMPTION_FILENAME,
    EXCEPTION_STOP_FILENAME,
    ExecutionStateV4_2,
    STAGE_ARTIFACTS,
)
from poker_attack.evidence_prereg_v4_2_core import (
    FROZEN_RUNTIME_VERSIONS,
    FROZEN_SCHEMA_SHA256,
    V4_1_DELTA_SHA256,
    V4_2_DELTA_SHA256,
    V4_2_RESULT_ROOT_BINDING_SHA256,
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
    work = root / CANONICAL_RESULT_ROOT_RELATIVE
    core = root / "poker/src/poker_attack/evidence_prereg_v4_2_core.py"
    runner_path = root / "poker/src/poker_attack/evidence_prereg_v4_2.py"
    core_test = root / "poker/tests/test_evidence_prereg_v4_2_core.py"
    runner_test = root / "poker/tests/test_evidence_prereg_v4_2_runner.py"

    assert sha256(root / CANONICAL_RELATIVE_PATHS["base_draft"]) == V4_DRAFT_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["v4_1_delta"]) == V4_1_DELTA_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["delta"]) == V4_2_DELTA_SHA256
    assert (
        sha256(root / CANONICAL_RELATIVE_PATHS["result_root_binding"])
        == V4_2_RESULT_ROOT_BINDING_SHA256
    )
    assert sha256(root / CANONICAL_RELATIVE_PATHS["coverage"]) == V4_COVERAGE_CONTRACT_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["method_audit"]) == V4_METHOD_AUDIT_SHA256
    assert sha256(root / CANONICAL_RELATIVE_PATHS["schema"]) == FROZEN_SCHEMA_SHA256
    assert FROZEN_RUNTIME_VERSIONS == {
        "python": "3.13.2",
        "numpy": "2.4.6",
        "pandas": "3.0.3",
        "scikit-learn": "1.8.0",
        "scipy": "1.17.1",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
        "pyarrow": "24.0.0",
    }

    frozen_predecessors = {
        "poker/src/poker_attack/evidence_prereg_v4_core.py": "1d02aa88b4a690aa088ea1dcdfda32ead25acb2e9236ccb7457c17e0a484a62d",
        "poker/src/poker_attack/evidence_prereg_v4.py": "4737f3c4983b37042d30d801dc0407625d7dc95a155a97bcb9f92417cfd59098",
        "poker/tests/test_evidence_prereg_v4_core.py": "fa1df57317f09dd2a6de09bfcc9b1d2fc09a1ed7d53351e8fdc0cfdab29d5304",
        "poker/tests/test_evidence_prereg_v4_runner.py": "bbb3a6ca2c8dc73010aaa68e994dce3f9292569e02c3ea6bbf24e3a068c42224",
        "poker/tests/static_audit_evidence_prereg_v4.py": "3c6b77d9bc15a92ea8a3874d4af99770f1cab31696564a7bedbd02f008703711",
        "poker/work/EVP_v4_implementation_containment_audit.json": "f8a2e27b4e41044f165badd05bc4f96d4883d6a7204d1f59f02a18292ac9e232",
        "poker/EVIDENCE_ONLY_VALIDATION_PREREG_V4_1_DELTA.md": "285a2b85beb03abf4419c4e3816bd3c0f0af68ebb64eec4c01a1fed9a3db4279",
        "poker/src/poker_attack/evidence_prereg_v4_1_core.py": "d20d2b3c5fd537cd827e2817c911860d605687bc84eb33bfcce2c9340534b67c",
        "poker/src/poker_attack/evidence_prereg_v4_1.py": "8deb18fe3f688d2bc4c02f7e7a25ae97acfc9558ba4f3001853b7f0af4ab9175",
        "poker/tests/test_evidence_prereg_v4_1_core.py": "96ff3061d3668cd7d9b6c5b17ce5a419d2fe459f6e83ab7dd1df04b7b6df33c7",
        "poker/tests/test_evidence_prereg_v4_1_runner.py": "1b8252f486cdfa839b9a79399200f5a07f81de562d0ea282fd9077ffa1386905",
        "poker/tests/static_audit_evidence_prereg_v4_1.py": "2148b28d4837b06bfeb1d9ea0fa4f11abb9fae2ced41289af5619ffd275dabef",
        "poker/work/EVP_v4_1_implementation_containment_audit.json": "144800fbb7bda2c7f3dc5bd52e6f4fa3a43834815bb302ed3561e319108d3adb",
    }
    for relative, expected in frozen_predecessors.items():
        assert sha256(root / relative) == expected

    runner_source = runner_path.read_text(encoding="utf-8")
    core_source = core.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    ast.parse(core_source)

    canonical = function_source(
        runner_source, runner_tree, "_canonical_result_root_v4_2"
    )
    assert "requested.is_absolute()" in canonical
    assert "requested != expected" in canonical
    assert "requested.resolve(strict=True) != expected" in canonical
    assert "expected.resolve(strict=True) != expected" in canonical
    assert "_read_canonical_json(" in canonical

    run = function_source(runner_source, runner_tree, "run_authorized_validation_v4_2")
    assert run.index("_canonical_result_root_v4_2") < run.index(
        "_pre_data_runtime_and_dependency_guard_v4_2"
    ) < run.index("_validate_result_execution_authorization_v4_2") < run.index(
        "_load_full_inputs_v4"
    )
    assert "CONSUMPTION_CLAIM_FILENAME" in run
    assert "CONSUMPTION_FILENAME" in run
    assert "_write_exception_terminal_v4_2" in run

    consume = function_source(runner_source, runner_tree, "_consume_authorization_v4_2")
    assert "canonical = _canonical_result_root_v4_2(work_dir)" in consume
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in consume
    assert "claim_path.exists() or receipt_path.exists()" in consume
    assert consume.index("_write_complete_claim_v4_2") < consume.index(
        "_atomic_json(receipt_path, receipt)"
    )
    complete = function_source(runner_source, runner_tree, "_write_complete_claim_v4_2")
    assert "while remaining" in complete
    assert "os.write(descriptor, remaining)" in complete
    assert "os.fsync(descriptor)" in complete

    terminal = function_source(
        runner_source, runner_tree, "_write_exception_terminal_v4_2"
    )
    safe = function_source(
        runner_source, runner_tree, "_safe_consumption_state_binding_v4_2"
    )
    assert "_safe_consumption_state_binding_v4_2" in terminal
    assert "json.loads" not in safe
    assert "_atomic_json(" in terminal

    publisher = function_source(runner_source, runner_tree, "_publish_exclusive_v4_2")
    assert "os.link(temporary, path)" in publisher
    assert ".replace(" not in publisher
    assert "_fsync_directory_v4_2(path.parent)" in publisher

    stage_2 = function_source(runner_source, runner_tree, "_run_stage_2_attribution_v4")
    assert stage_2.index("stage_2_development_data_access = True") < stage_2.index(
        "_reload_stage_1_states_v4_2"
    )
    assert stage_2.index("_fit_outer_l0_from_state_v4(state)") < stage_2.index(
        "stage_2_model_fit = True"
    )
    assert stage_2.index("sensitivity = sensitivity_tree_v4") < stage_2.index(
        "stage_2_aggregate_performance_viewed = True"
    )
    assert stage_2.index("top_sha256 = _sha256(top_path)") < stage_2.index(
        "_query_scores_with_reporting_family_v4"
    )

    controls = ExecutionStateV4_2().terminal_controls()
    required = {
        "stage_2_development_data_access",
        "stage_2_model_fit",
        "stage_2_aggregate_performance_viewed",
        "real_model_fit_performed",
        "aggregate_performance_viewed",
        "evaluation_rows_loaded",
        "evaluation_scored",
        "unknown_labels_assigned",
        "experiment_number_assigned",
        "candidate_promoted",
        "submission_created_or_modified",
        "rerun_allowed",
    }
    assert required.issubset(controls)
    assert all(type(value) is bool for value in controls.values())

    forbidden = (
        "evaluation_pairs.csv",
        "sample_submission.csv",
        "submission.csv",
        "kaggle competitions submit",
        "subprocess.",
        "requests.",
        "urllib.",
    )
    assert all(token not in runner_source for token in forbidden)
    assert all(token not in core_source for token in forbidden)
    for filename in (
        CONSUMPTION_CLAIM_FILENAME,
        CONSUMPTION_FILENAME,
        EXCEPTION_STOP_FILENAME,
    ):
        assert not (work / filename).exists()
    for names in STAGE_ARTIFACTS.values():
        for name in names:
            assert not (work / name).exists()
    assert not (root / CANONICAL_RELATIVE_PATHS["v4_2_source_audit"]).exists()

    print(
        json.dumps(
            {
                "status": "PASS_V4_2_IMPLEMENTATION_PRESENT_UNEXECUTED",
                "binding": {
                    "v4_draft_sha256": V4_DRAFT_SHA256,
                    "v4_1_delta_sha256": V4_1_DELTA_SHA256,
                    "v4_2_delta_sha256": V4_2_DELTA_SHA256,
                    "v4_2_result_root_binding_sha256": V4_2_RESULT_ROOT_BINDING_SHA256,
                    "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
                    "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
                    "schema_sha256": FROZEN_SCHEMA_SHA256,
                    "v4_2_core_sha256": sha256(core),
                    "v4_2_runner_sha256": sha256(runner_path),
                    "v4_2_core_tests_sha256": sha256(core_test),
                    "v4_2_runner_tests_sha256": sha256(runner_test),
                },
                "repairs": {
                    "canonical_exact_result_root": True,
                    "authorization_global_single_consumption_point": True,
                    "durable_exclusive_claim": True,
                    "partial_claim_fail_closed": True,
                    "exception_terminal_tolerates_partial_consumption": True,
                    "stage_2_exact_method_controls": True,
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
