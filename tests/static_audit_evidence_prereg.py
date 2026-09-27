"""Non-result static audit for the implementation-only EVP-DRAFT-03 bundle."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from poker_attack.evidence_prereg import (
    CONSUMPTION_FILENAME,
    SOURCE_SHA256,
    VIEW_SPECS,
    _evidence_bearing,
    _load_and_validate_inputs,
    _scope,
    _validate_output_terminology_contract,
)
from poker_attack.evidence_prereg_core import (
    B001_FULL_FEATURES,
    B001_TIME_FEATURES,
    BOOTSTRAP_BASE_SEED,
    BOOTSTRAP_DRAWS,
    C_GRID,
    DRAFT_SHA256,
    FEATURES,
    FROZEN_SKLEARN_VERSION,
    INNER_FOLDS,
    INNER_SPLIT_BASE_SEED,
    OUTER_FOLDS,
    PREREG_STATIC_AUDIT_SHA256,
    ROUTER_SEED_BASE,
    WEIGHT_STREAM_BIT_GENERATOR,
    WEIGHT_STREAM_NO_HEADER_SHA256,
    WEIGHT_STREAM_NUMPY_VERSION,
    WEIGHT_STREAM_SHA256,
    BayesianClusterWeightPlan,
    frozen_pool_weight_stream_sha256,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call_line(function: ast.FunctionDef, name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    if len(lines) != 1:
        raise AssertionError(f"expected one call to {name}, found {lines}")
    return lines[0]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    core_path = root / "poker/src/poker_attack/evidence_prereg_core.py"
    runner_path = root / "poker/src/poker_attack/evidence_prereg.py"
    core_test_path = root / "poker/tests/test_evidence_prereg_core.py"
    runner_test_path = root / "poker/tests/test_evidence_prereg_runner.py"
    draft_path = root / "poker/EVIDENCE_ONLY_VALIDATION_PREREG_DRAFT_V3.md"
    prereg_audit_path = root / "poker/work/EVP_prereg_v3_static_audit.json"
    work_dir = root / "poker/work"

    assert sha256(draft_path) == DRAFT_SHA256
    assert sha256(prereg_audit_path) == PREREG_STATIC_AUDIT_SHA256
    assert len(FEATURES) == 20
    assert len(B001_FULL_FEATURES) == 95
    assert len(B001_TIME_FEATURES) == 77
    assert C_GRID == (0.01, 0.1, 1.0, 10.0)
    assert OUTER_FOLDS == 5
    assert INNER_FOLDS == 3
    assert INNER_SPLIT_BASE_SEED == 8803
    assert ROUTER_SEED_BASE == {
        "full": 18401,
        "early_to_late": 19401,
        "late_to_early": 20401,
    }
    assert BOOTSTRAP_DRAWS == 5000
    assert BOOTSTRAP_BASE_SEED == 12673
    assert WEIGHT_STREAM_NUMPY_VERSION == "2.4.6"
    assert WEIGHT_STREAM_BIT_GENERATOR == "PCG64"
    assert FROZEN_SKLEARN_VERSION == "1.8.0"

    runner_source = runner_path.read_text(encoding="utf-8")
    core_source = core_path.read_text(encoding="utf-8")
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
    terminology = _validate_output_terminology_contract()
    assert terminology["interval_name"] == (
        "90% paired Bayesian cluster-weight sensitivity interval"
    )
    assert terminology["confidence_claimed"] is False
    assert terminology["coverage_claimed"] is False
    tree = ast.parse(runner_source)
    runner_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_authorized_validation"
    )
    authorization_line = call_line(
        runner_function, "_validate_result_execution_authorization"
    )
    data_load_line = call_line(runner_function, "_load_and_validate_inputs")
    assert authorization_line < data_load_line

    assert not (work_dir / CONSUMPTION_FILENAME).exists()
    for forbidden_output in (
        "EVP_validation_result.json",
        "EVP_outer_oof_per_hand.parquet",
        "EVP_outer_oof_top_five.parquet",
    ):
        assert not (work_dir / forbidden_output).exists()
    assert not (work_dir / "EVP_nested_routes").exists()

    bundle = _load_and_validate_inputs(
        root / "data/raw/poker_comp", work_dir
    )
    assert bundle.source_hashes == SOURCE_SHA256
    plans = {}
    view_counts = {}
    minimum_denominators = {}
    for spec in VIEW_SPECS:
        target = _evidence_bearing(_scope(bundle.positive_hands, spec.target_half))
        meta = target[
            ["pair_id", "table_id", "outer_fold", "behavior_family"]
        ].drop_duplicates("pair_id", keep="first").reset_index(drop=True)
        plan = BayesianClusterWeightPlan.create(meta, spec.name)
        plans[spec.name] = plan
        view_counts[spec.name] = {
            "eligible_queries": int(len(meta)),
            "eligible_pools": int(meta["table_id"].nunique()),
            "pool_counts_by_outer_fold": [
                len(plan.fold_pool_keys[fold]) for fold in range(5)
            ],
        }
        minimum_denominators[spec.name] = float(
            min(
                plan.query_weights[
                    :,
                    (
                        (meta["outer_fold"] == fold)
                        & (meta["behavior_family"].astype(str) == family)
                    ).to_numpy(),
                ]
                .sum(axis=1)
                .min()
                for fold in range(5)
                for family in (
                    "directed_transfer",
                    "soft_play",
                    "coordinated_isolation",
                )
            )
        )
    with_headers = frozen_pool_weight_stream_sha256(plans)
    without_headers = frozen_pool_weight_stream_sha256(
        plans, include_headers=False
    )
    assert with_headers == WEIGHT_STREAM_SHA256
    assert without_headers == WEIGHT_STREAM_NO_HEADER_SHA256
    assert minimum_denominators == {
        "full": 6.143430345838801,
        "early_to_late": 1.0232903325325067,
        "late_to_early": 4.9478382335473885,
    }

    print(
        json.dumps(
            {
                "status": "PASS_IMPLEMENTATION_PRESENT_UNEXECUTED",
                "binding": {
                    "draft_sha256": DRAFT_SHA256,
                    "prereg_static_audit_sha256": PREREG_STATIC_AUDIT_SHA256,
                    "core_implementation_sha256": sha256(core_path),
                    "runner_implementation_sha256": sha256(runner_path),
                    "core_tests_sha256": sha256(core_test_path),
                    "runner_tests_sha256": sha256(runner_test_path),
                },
                "frozen_constants": {
                    "evidence_features": len(FEATURES),
                    "b001_full_features": len(B001_FULL_FEATURES),
                    "b001_time_features": len(B001_TIME_FEATURES),
                    "c_grid": list(C_GRID),
                    "outer_folds": OUTER_FOLDS,
                    "inner_folds": INNER_FOLDS,
                    "inner_split_base_seed": INNER_SPLIT_BASE_SEED,
                    "router_seed_base": ROUTER_SEED_BASE,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
                    "numpy_version": WEIGHT_STREAM_NUMPY_VERSION,
                    "bit_generator": WEIGHT_STREAM_BIT_GENERATOR,
                    "scikit_learn_version": FROZEN_SKLEARN_VERSION,
                },
                "weight_stream": {
                    "with_view_and_raw_fold_byte_headers_sha256": with_headers,
                    "without_headers_negative_control_sha256": without_headers,
                    "historical_hash_does_not_bind_pool_ids_or_shapes": True,
                    "view_counts": view_counts,
                    "minimum_denominators": minimum_denominators,
                },
                "execution_lock": {
                    "authorization_checked_before_data_load": True,
                    "implementation_only_receipt_rejected_by_tests": True,
                    "one_run_consumption_receipt_exists": False,
                    "result_outputs_exist": False,
                },
                "terminology_contract": terminology,
                "controls": {
                    "implementation_created": True,
                    "real_model_fit_performed": False,
                    "aggregate_performance_viewed": False,
                    "evaluation_rows_loaded": False,
                    "evaluation_scored": False,
                    "experiment_number_assigned": False,
                    "submission_created_or_modified": False,
                    "unknown_ground_truth_labels_assigned": 0,
                    "preregistration_executed": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
