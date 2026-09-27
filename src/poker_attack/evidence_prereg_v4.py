"""Hash-locked, strictly staged executor for EVP-DRAFT-04.

The file is a new v4 implementation.  It does not import the rejected v3
executor or core.  No real execution is currently authorized; the guard also
requires future canonical v4 containment and source-audit receipts that do not
yet exist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .evidence_prereg_v4_core import (
    B001_FULL_FEATURES,
    B001_TIME_FEATURES,
    C_GRID,
    FAMILIES,
    FEATURES,
    HISTORICAL_CONTAINMENT_SHA256,
    H0_SIGNAL_BY_FAMILY,
    OUTER_FOLDS,
    REJECTED_V3_CORE_SHA256,
    REJECTED_V3_RUNNER_SHA256,
    STAGE_WEIGHT_STREAM_NO_HEADER_SHA256,
    STAGE_WEIGHT_STREAM_SHA256,
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
    fit_evidence_ranker_v4,
    fit_nested_behavior_routes_v4,
    frozen_stage_weight_stream_sha256_v4,
    heuristic_scores_v4,
    interaction_matrix_v4,
    metric_tree_v4,
    paired_delta_frame_v4,
    query_ap_frame_v4,
    select_c_no_family_v4,
    sensitivity_tree_v4,
    validate_all_metric_coverage_v4,
)
from .schema import input_paths


IMPLEMENTATION_ONLY_STATUS = "V4_IMPLEMENTATION_ONLY_APPROVED"
RESULT_EXECUTION_STATUS = "RESULT_BEARING_EXECUTION_APPROVED"
CONSUMPTION_FILENAME = "EVP_v4_result_execution_consumed.json"

CANONICAL_RELATIVE_PATHS = {
    "draft": "poker/EVIDENCE_ONLY_VALIDATION_PREREG_DRAFT_V4.md",
    "coverage": "poker/work/EVP_v4_metric_coverage_contract.json",
    "method_audit": "poker/work/EVP_v4_independent_method_audit.json",
    "historical_containment": "poker/work/EVP_implementation_containment_audit.json",
    "v3_rejection": "poker/work/EVP_v3_source_audit_rejection.json",
    "v4_containment": "poker/work/EVP_v4_implementation_containment_audit.json",
    "v4_source_audit": "poker/work/EVP_v4_source_audit.json",
}

STATIC_CANONICAL_SHA256 = {
    "draft": V4_DRAFT_SHA256,
    "coverage": V4_COVERAGE_CONTRACT_SHA256,
    "method_audit": V4_METHOD_AUDIT_SHA256,
    "historical_containment": HISTORICAL_CONTAINMENT_SHA256,
    "v3_rejection": "90bb142996798652cc64adb863c93ff845452fb56363500cab969b68de213a1d",
}

SOURCE_SHA256 = {
    "development_labels": "40fcfa99d67db2ccbc37c56d4ef858676f1a677128db873de0e94bcea15f62a9",
    "development_evidence": "f5889ee490b7dd0d03341b5505a1789071558bd5a5d1a31d61f25541460d33ba",
    "development_pair_hands": "cf197d2bf8b8242b83515b6db1a5a0c73c214ed91fc46fce9205014f07d265af",
    "b001_full_oof": "71d7ce45170267060f7a1a6ae3e4a78a425d37bc4b4d88d392d8fe08b8baee36",
    "b001_time_oof": "4f85a92ab6e72f3c71dd4d14b7f5ba57e621bd87c08a41955576d7d612273147",
    "b001_behavior_implementation": "4cae28dba762ba301bb58d0c75fa5533bd8d5e979a52764732b506e831a87b49",
    "b001_full_features": "756d2e1e1b9d031755534ff975023a5f958ab36c53ec342b71c9f03f6448dc8b",
    "b001_early_features": "0622e8f2b59cf1fb7b9109c4b9e536d5f5773d0bdee630b361134d9116db471d",
    "b001_late_features": "943fbf666fc19ac2c1fb086e7c32f94df4ead0dcc8b65e85b9feade85cd22e1a",
    "hands": "82c9ad1c01cae6a68b28e194b9dd0bad9ca09f94628b429c0e58e970cc29eecd",
    "official_metric": "3cb11be5c999ada91aa91002f18aabc0c1c48548f1b6aedf3939f31fcd511c5c",
    "ev000_implementation": "7518ed8731fa128cf66abee99dc95dc639673dfd0c025735ad4bdb028e09a49d",
    "ev000_report": "27a8245e99419fce163e5d90843cd2585a1273a467d7841433dea732384a4726",
}

FULL_STAGE_SOURCE_NAMES = (
    "development_labels",
    "development_evidence",
    "development_pair_hands",
    "b001_full_oof",
    "b001_behavior_implementation",
    "b001_full_features",
    "hands",
    "official_metric",
    "ev000_implementation",
    "ev000_report",
)
TIME_STAGE_SOURCE_NAMES = (
    "b001_time_oof",
    "b001_early_features",
    "b001_late_features",
)

EXPECTED_PAIRWISE_READINESS = {
    "full": {
        "evidence_bearing_pairs": 372,
        "pairwise_trainable_pairs": 372,
        "candidate_minimum": 57,
        "unjudged_minimum": 52,
        "preference_pairs": 212888,
        "symmetric_rows": 425776,
    },
    "early": {
        "evidence_bearing_pairs": 347,
        "pairwise_trainable_pairs": 346,
        "candidate_minimum": 1,
        "unjudged_minimum": 0,
        "preference_pairs": 82266,
        "symmetric_rows": 164532,
    },
    "late": {
        "evidence_bearing_pairs": 264,
        "pairwise_trainable_pairs": 264,
        "candidate_minimum": 10,
        "unjudged_minimum": 9,
        "preference_pairs": 42763,
        "symmetric_rows": 85526,
    },
}

EXPECTED_H0_TIE_STRUCTURE = {
    "full": (372, 4522, 30994, 65, 0),
    "early_to_late": (370, 2580, 14162, 70, 0),
    "late_to_early": (372, 2635, 14797, 70, 0),
}

STAGE_ARTIFACTS = {
    1: (
        "EVP_v4_stage_1_full_routes",
        "EVP_v4_stage_1_full_scores.parquet",
        "EVP_v4_stage_1_full_top_five.parquet",
        "EVP_v4_stage_1_reporting_family.parquet",
        "EVP_v4_stage_1_primary.json",
    ),
    2: (
        "EVP_v4_stage_2_l0_scores.parquet",
        "EVP_v4_stage_2_l0_top_five.parquet",
        "EVP_v4_stage_2_attribution.json",
    ),
    3: (
        "EVP_v4_stage_3_time_routes",
        "EVP_v4_stage_3_time_scores.parquet",
        "EVP_v4_stage_3_time_top_five.parquet",
        "EVP_v4_stage_3_reporting_family.parquet",
        "EVP_v4_stage_3_bidirectional.json",
    ),
}


@dataclass(frozen=True)
class ViewSpecV4:
    name: str
    source_half: str
    target_half: str
    outer_route_column: str


FULL_SPEC = ViewSpecV4("full", "full", "full", "predicted_family")
TIME_SPECS = (
    ViewSpecV4(
        "early_to_late",
        "early",
        "late",
        "early_to_late__predicted_family",
    ),
    ViewSpecV4(
        "late_to_early",
        "late",
        "early",
        "late_to_early__predicted_family",
    ),
)


@dataclass
class FullBundleV4:
    pair_meta: pd.DataFrame
    hands: pd.DataFrame
    full_pair_features: pd.DataFrame
    source_hashes: dict[str, str]


@dataclass
class TimeBundleV4:
    hands: pd.DataFrame
    time_routes: pd.DataFrame
    early_pair_features: pd.DataFrame
    late_pair_features: pd.DataFrame
    source_hashes: dict[str, str]


@dataclass
class OuterFitStateV4:
    spec: ViewSpecV4
    outer_fold: int
    selected_c: float
    transform: FrozenHandTransformV4
    source_train: pd.DataFrame
    source_base: np.ndarray
    target_fold: pd.DataFrame
    target_valid: pd.DataFrame
    eligible: np.ndarray
    target_routes: np.ndarray


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(key: str) -> Path:
    if key not in CANONICAL_RELATIVE_PATHS:
        raise V4InvariantError(f"unknown canonical artifact key: {key}")
    root = _project_root().resolve()
    expected = root / CANONICAL_RELATIVE_PATHS[key]
    resolved = expected.resolve(strict=True)
    if resolved != expected:
        raise PermissionError(f"canonical artifact is a symlink or escaped: {key}")
    return resolved


def _read_canonical_json(key: str, expected_sha256: str) -> dict[str, object]:
    path = _canonical_path(key)
    if _sha256(path) != expected_sha256:
        raise PermissionError(f"canonical artifact hash changed: {key}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _all_result_paths(work_dir: Path) -> list[Path]:
    return [
        work_dir / name
        for stage in (1, 2, 3)
        for name in STAGE_ARTIFACTS[stage]
    ]


def _later_stage_paths(work_dir: Path, after_stage: int) -> list[Path]:
    return [
        work_dir / name
        for stage in range(after_stage + 1, 4)
        for name in STAGE_ARTIFACTS[stage]
    ]


def _assert_paths_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise PermissionError(f"prohibited pre-existing downstream artifact: {existing}")


def _consumption_receipt_binding_v4(work_dir: Path) -> dict[str, str]:
    path = work_dir / CONSUMPTION_FILENAME
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "V4_RESULT_EXECUTION_AUTHORIZATION_CONSUMED":
        raise PermissionError("the v4 consumption receipt status changed")
    return {
        "consumption_receipt": str(path),
        "consumption_receipt_sha256": _sha256(path),
        "authorization_sha256": str(receipt["authorization_sha256"]),
    }


def _validate_result_execution_authorization_v4(
    authorization_path: Path,
    work_dir: Path,
) -> dict[str, object]:
    """Validate all canonical bindings before any development source is opened."""

    if not authorization_path.is_file():
        raise PermissionError("a result-bearing v4 authorization is required")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") == IMPLEMENTATION_ONLY_STATUS:
        raise PermissionError("v4 implementation-only approval cannot run real data")
    if authorization.get("status") != RESULT_EXECUTION_STATUS:
        raise PermissionError("v4 result-bearing execution is not approved")

    for key, expected_sha256 in STATIC_CANONICAL_SHA256.items():
        if key == "draft":
            if _sha256(_canonical_path(key)) != expected_sha256:
                raise PermissionError("v4 draft hash changed")
        else:
            _read_canonical_json(key, expected_sha256)

    historical = _read_canonical_json(
        "historical_containment", HISTORICAL_CONTAINMENT_SHA256
    )
    if historical.get("status") != "PASS_IMPLEMENTATION_PRESENT_UNEXECUTED":
        raise PermissionError("historical containment status changed")
    method_audit = _read_canonical_json("method_audit", V4_METHOD_AUDIT_SHA256)
    if method_audit.get("status") != "PASS_METHOD_SPEC_ONLY":
        raise PermissionError("v4 method audit is not PASS_METHOD_SPEC_ONLY")

    # These paths are constants constructed by this executor. No path value
    # from the caller's authorization is ever dereferenced.
    v4_containment_path = _canonical_path("v4_containment")
    v4_source_audit_path = _canonical_path("v4_source_audit")
    v4_containment_sha256 = _sha256(v4_containment_path)
    v4_source_audit_sha256 = _sha256(v4_source_audit_path)
    v4_containment = json.loads(v4_containment_path.read_text(encoding="utf-8"))
    v4_source_audit = json.loads(v4_source_audit_path.read_text(encoding="utf-8"))
    if v4_containment.get("status") != "PASS_V4_IMPLEMENTATION_PRESENT_UNEXECUTED":
        raise PermissionError("canonical v4 containment is not an unexecuted PASS")
    if v4_source_audit.get("status") != "PASS_V4_SOURCE_AUDIT_UNEXECUTED":
        raise PermissionError("canonical v4 source audit is not an unexecuted PASS")

    core_path = Path(__file__).with_name("evidence_prereg_v4_core.py")
    runner_path = Path(__file__)
    core_sha256 = _sha256(core_path)
    runner_sha256 = _sha256(runner_path)
    required_receipt_binding = {
        "v4_draft_sha256": V4_DRAFT_SHA256,
        "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
        "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
        "historical_containment_sha256": HISTORICAL_CONTAINMENT_SHA256,
        "v4_core_sha256": core_sha256,
        "v4_runner_sha256": runner_sha256,
    }
    for name, receipt in (
        ("v4 containment", v4_containment),
        ("v4 source audit", v4_source_audit),
    ):
        binding = receipt.get("binding", {})
        if any(binding.get(key) != value for key, value in required_receipt_binding.items()):
            raise PermissionError(f"{name} does not bind the current v4 source bundle")
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
        "v4_containment_sha256": v4_containment_sha256,
        "v4_source_audit_path": CANONICAL_RELATIVE_PATHS["v4_source_audit"],
        "v4_source_audit_sha256": v4_source_audit_sha256,
        "v4_core_sha256": core_sha256,
        "v4_runner_sha256": runner_sha256,
    }
    if authorization.get("binding") != exact_binding:
        raise PermissionError(
            "authorization binding is not the exact canonical v4 binding"
        )
    exact_permissions = {
        "run_preregistered_development_stages_once": True,
        "load_or_score_evaluation": False,
        "assign_experiment_number": False,
        "promote_candidate": False,
        "create_or_modify_submission": False,
        "rerun_after_stop_or_failure": False,
    }
    if authorization.get("permissions") != exact_permissions:
        raise PermissionError("authorization permissions are not the narrow v4 set")

    for receipt in (v4_containment, v4_source_audit):
        controls = receipt.get("controls", {})
        for field in (
            "real_development_data_read",
            "real_model_fit_performed",
            "aggregate_performance_viewed",
            "evaluation_rows_loaded",
            "evaluation_scored",
            "experiment_number_assigned",
            "candidate_promoted",
            "submission_created_or_modified",
            "preregistration_executed",
        ):
            if controls.get(field) is not False:
                raise PermissionError(f"unexecuted receipt control is not false: {field}")

    _assert_paths_absent(_all_result_paths(work_dir))
    consumed_path = work_dir / CONSUMPTION_FILENAME
    try:
        descriptor = os.open(
            consumed_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise PermissionError("the one-run v4 authorization was already consumed") from error
    consumed = {
        "status": "V4_RESULT_EXECUTION_AUTHORIZATION_CONSUMED",
        "authorization_path": str(authorization_path),
        "authorization_sha256": _sha256(authorization_path),
        "binding": exact_binding,
        "rerun_allowed": False,
        "evaluation_allowed": False,
        "submission_allowed": False,
        "stage_1_started": False,
        "stage_2_started": False,
        "stage_3_started": False,
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(consumed, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return authorization


def _source_paths(data_dir: Path, work_dir: Path) -> dict[str, Path]:
    inputs = input_paths(data_dir)
    root = _project_root()
    return {
        "development_labels": inputs["development_labels"],
        "development_evidence": inputs["development_evidence"],
        "development_pair_hands": work_dir / "development_pair_hands.parquet",
        "b001_full_oof": work_dir / "B001_behavior_oof.parquet",
        "b001_time_oof": work_dir / "B001_behavior_time_oof.parquet",
        "b001_behavior_implementation": root
        / "poker/src/poker_attack/behavior_audit.py",
        "b001_full_features": work_dir / "development_pair_features.parquet",
        "b001_early_features": work_dir / "E007_early_base_features.parquet",
        "b001_late_features": work_dir / "E007_late_base_features.parquet",
        "hands": inputs["hands"],
        "official_metric": root
        / "poker/reference/official_metric/slash-poker-competition-metric.ipynb",
        "ev000_implementation": root
        / "poker/src/poker_attack/evidence_feasibility.py",
        "ev000_report": work_dir / "EV000_evidence_feasibility.json",
    }


def _hash_stage_sources(
    paths: Mapping[str, Path],
    allowed_names: Sequence[str],
) -> dict[str, str]:
    observed = {name: _sha256(paths[name]) for name in allowed_names}
    expected = {name: SOURCE_SHA256[name] for name in allowed_names}
    if observed != expected:
        raise V4InvariantError("a frozen stage source hash changed")
    return observed


def _load_coverage_book_canonical() -> CoverageBookV4:
    contract = _read_canonical_json("coverage", V4_COVERAGE_CONTRACT_SHA256)
    return CoverageBookV4.from_mapping(contract)


def _load_full_inputs_v4(
    data_dir: Path,
    work_dir: Path,
    coverage_book: CoverageBookV4,
) -> FullBundleV4:
    """Stage-1 loader. It does not open time OOF/features or phase_progress."""

    paths = _source_paths(data_dir, work_dir)
    observed_hashes = _hash_stage_sources(paths, FULL_STAGE_SOURCE_NAMES)
    labels = pd.read_csv(paths["development_labels"])
    expected_status = np.where(
        labels["label"].to_numpy(dtype=int) == 1,
        "confirmed_target",
        "confirmed_non_target",
    )
    if not np.array_equal(labels["label_status"].astype(str), expected_status):
        raise V4InvariantError("labels contain a non-confirmed or mismatched status")
    positives = labels[
        (labels["label"] == 1)
        & (labels["label_status"].astype(str) == "confirmed_target")
    ][["pair_id", "behavior_family"]].copy()
    if len(positives) != 372 or positives["pair_id"].duplicated().any():
        raise V4InvariantError("confirmed-target population changed")
    if not set(positives["behavior_family"].astype(str)).issubset(FAMILIES):
        raise V4InvariantError("confirmed target has an invalid family")

    full_route = pd.read_parquet(
        paths["b001_full_oof"],
        columns=["pair_id", "table_id", "fold", "predicted_family"],
    ).rename(columns={"fold": "outer_fold"})
    pair_meta = positives.merge(
        full_route, on="pair_id", how="left", validate="one_to_one"
    )
    if pair_meta[["table_id", "outer_fold", "predicted_family"]].isna().any().any():
        raise V4InvariantError("a target lacks a frozen full route")
    if pair_meta.groupby("table_id")["outer_fold"].nunique().max() != 1:
        raise V4InvariantError("a table pool crosses outer folds")

    # phase_progress is intentionally absent from this read.
    pair_hands = pd.read_parquet(
        paths["development_pair_hands"],
        columns=[
            "pair_id",
            "hand_id",
            "table_id",
            *FEATURES,
            *H0_SIGNAL_BY_FAMILY.values(),
        ],
    )
    hands = pair_hands[
        pair_hands["pair_id"].astype(str).isin(pair_meta["pair_id"].astype(str))
    ].copy()
    if len(hands) != 45129 or hands[["pair_id", "hand_id"]].duplicated().any():
        raise V4InvariantError("full candidate-hand population changed")
    values = hands.loc[:, FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise V4InvariantError("a frozen evidence feature is non-finite")

    evidence = pd.read_csv(paths["development_evidence"])
    if len(evidence) != 1817 or evidence[["pair_id", "hand_id"]].duplicated().any():
        raise V4InvariantError("organizer evidence population changed")
    if evidence[["pair_id", "evidence_rank"]].duplicated().any() or not set(
        evidence["evidence_rank"].astype(int)
    ).issubset(range(1, 6)):
        raise V4InvariantError("organizer evidence rank changed")
    evidence_family = evidence.merge(
        positives,
        on="pair_id",
        how="left",
        validate="many_to_one",
        suffixes=("_evidence", "_label"),
    )
    if evidence_family["behavior_family_label"].isna().any() or not np.array_equal(
        evidence_family["behavior_family_evidence"].astype(str),
        evidence_family["behavior_family_label"].astype(str),
    ):
        raise V4InvariantError("organizer evidence family differs from confirmed label")
    evidence_keys = pd.MultiIndex.from_frame(evidence[["pair_id", "hand_id"]].astype(str))
    hand_keys = pd.MultiIndex.from_frame(hands[["pair_id", "hand_id"]].astype(str))
    hands["is_evidence"] = hand_keys.isin(evidence_keys)
    if int(hands["is_evidence"].sum()) != 1817:
        raise V4InvariantError("organizer evidence is not a legal shared hand")

    hand_time = pd.read_parquet(paths["hands"], columns=["hand_id", "started_at"])
    if hand_time["hand_id"].duplicated().any():
        raise V4InvariantError("hands source has duplicate hand_id")
    hands = hands.merge(hand_time, on="hand_id", how="left", validate="many_to_one")
    hands = hands.merge(
        pair_meta[["pair_id", "table_id", "outer_fold"]],
        on="pair_id",
        how="left",
        validate="many_to_one",
        suffixes=("_hand", "_pair"),
    )
    if hands["started_at"].isna().any() or not np.array_equal(
        hands["table_id_hand"].astype(str), hands["table_id_pair"].astype(str)
    ):
        raise V4InvariantError("full hand timestamp or pool join changed")
    hands = hands.rename(columns={"table_id_pair": "table_id"}).drop(
        columns=["table_id_hand"]
    )
    if "behavior_family" in hands or "phase_progress" in hands:
        raise RuntimeError("held-out family or phase entered the Stage-1 hand frame")

    full_features = pd.read_parquet(paths["b001_full_features"])
    if any(
        column not in full_features
        for column in ("pair_id", "table_id", *B001_FULL_FEATURES)
    ):
        raise V4InvariantError("the frozen full B001 feature schema changed")

    bundle = FullBundleV4(pair_meta, hands, full_features, observed_hashes)
    _validate_structural_coverage_v4(
        bundle.hands, bundle.pair_meta, "full", coverage_book
    )
    _validate_pairwise_readiness_v4(bundle.hands, "full", bundle.pair_meta)
    return bundle


def _load_time_inputs_v4(
    data_dir: Path,
    work_dir: Path,
    full: FullBundleV4,
    coverage_book: CoverageBookV4,
) -> TimeBundleV4:
    """Stage-3-only loader. Calling this before Stage-2 PASS is prohibited."""

    paths = _source_paths(data_dir, work_dir)
    observed_hashes = _hash_stage_sources(paths, TIME_STAGE_SOURCE_NAMES)
    phase = pd.read_parquet(
        paths["development_pair_hands"],
        columns=["pair_id", "hand_id", "phase_progress"],
    )
    if phase[["pair_id", "hand_id"]].duplicated().any():
        raise V4InvariantError("phase source has duplicate pair-hand keys")
    hands = full.hands.merge(
        phase, on=["pair_id", "hand_id"], how="left", validate="one_to_one"
    )
    if hands["phase_progress"].isna().any():
        raise V4InvariantError("a Stage-3 hand lacks phase_progress")
    time_routes = pd.read_parquet(
        paths["b001_time_oof"],
        columns=[
            "pair_id",
            "fold",
            "early_to_late__predicted_family",
            "late_to_early__predicted_family",
        ],
    ).rename(columns={"fold": "outer_fold"})
    if time_routes["pair_id"].duplicated().any():
        raise V4InvariantError("time route receipt has duplicate pairs")
    route_check = full.pair_meta[["pair_id", "outer_fold"]].merge(
        time_routes,
        on="pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("_full", "_time"),
    )
    if route_check.isna().any().any() or not np.array_equal(
        route_check["outer_fold_full"].to_numpy(int),
        route_check["outer_fold_time"].to_numpy(int),
    ):
        raise V4InvariantError("time route membership or outer fold changed")
    for column in (
        "early_to_late__predicted_family",
        "late_to_early__predicted_family",
    ):
        if not set(route_check[column].astype(str)).issubset(FAMILIES):
            raise V4InvariantError("time route emitted an invalid family")
    early_features = pd.read_parquet(paths["b001_early_features"])
    late_features = pd.read_parquet(paths["b001_late_features"])
    for frame in (early_features, late_features):
        if any(
            column not in frame
            for column in ("pair_id", "table_id", *B001_TIME_FEATURES)
        ):
            raise V4InvariantError("the frozen time B001 feature schema changed")
    bundle = TimeBundleV4(
        hands, time_routes, early_features, late_features, observed_hashes
    )
    _validate_pairwise_readiness_v4(
        _scope_v4(bundle.hands, "early"), "early", full.pair_meta
    )
    _validate_pairwise_readiness_v4(
        _scope_v4(bundle.hands, "late"), "late", full.pair_meta
    )
    for spec in TIME_SPECS:
        _validate_structural_coverage_v4(
            _evidence_bearing_v4(_scope_v4(bundle.hands, spec.target_half)),
            full.pair_meta,
            spec.name,
            coverage_book,
        )
    return bundle


def _scope_v4(frame: pd.DataFrame, half: str) -> pd.DataFrame:
    if half == "full":
        return frame.copy()
    if "phase_progress" not in frame:
        raise V4InvariantError("time scope requested before Stage-3 phase access")
    if half == "early":
        return frame[frame["phase_progress"] <= 0.5].copy()
    if half == "late":
        return frame[frame["phase_progress"] > 0.5].copy()
    raise V4InvariantError(f"unknown hand scope: {half}")


def _evidence_bearing_v4(frame: pd.DataFrame) -> pd.DataFrame:
    bearing = frame.groupby("pair_id", sort=False)["is_evidence"].transform("any")
    return frame[bearing].copy()


def _validate_structural_coverage_v4(
    hands: pd.DataFrame,
    pair_meta: pd.DataFrame,
    view: str,
    book: CoverageBookV4,
) -> None:
    """Validate coverage without joining held-out family onto scoring hands."""

    stats = hands.groupby(
        ["pair_id", "table_id", "outer_fold"], sort=False
    ).agg(
        candidate_hands=("hand_id", "size"),
        evidence_hands=("is_evidence", "sum"),
    ).reset_index()
    stats = stats.merge(
        pair_meta[["pair_id", "behavior_family"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if stats["behavior_family"].isna().any():
        raise V4InvariantError("structural coverage family is missing")

    def coverage(part: pd.DataFrame) -> CoverageTupleV4:
        if part.empty:
            raise V4InvariantError("a structural coverage cell is empty")
        return CoverageTupleV4(
            eligible_queries=int(len(part)),
            candidate_hands=int(part["candidate_hands"].sum()),
            evidence_hands=int(part["evidence_hands"].sum()),
            eligible_pools=int(part["table_id"].astype(str).nunique()),
        )

    book.validate(view, "overall", "overall", coverage(stats))
    for family in FAMILIES:
        family_part = stats[stats["behavior_family"].astype(str) == family]
        book.validate(view, "family", family, coverage(family_part))
    for fold in range(OUTER_FOLDS):
        fold_part = stats[stats["outer_fold"].astype(int) == fold]
        book.validate(view, "outer_fold", str(fold), coverage(fold_part))
        for family in FAMILIES:
            key = f"{fold}/{family}"
            cell = fold_part[
                fold_part["behavior_family"].astype(str) == family
            ]
            book.validate(view, "outer_fold_family", key, coverage(cell))


def _validate_pairwise_readiness_v4(
    frame: pd.DataFrame,
    half: str,
    pair_meta: pd.DataFrame,
) -> None:
    stats = frame.groupby("pair_id", sort=False).agg(
        candidate_hands=("hand_id", "size"),
        evidence_hands=("is_evidence", "sum"),
    )
    eligible = stats["evidence_hands"] > 0
    unjudged = stats["candidate_hands"] - stats["evidence_hands"]
    preferences = stats["evidence_hands"] * unjudged
    observed = {
        "evidence_bearing_pairs": int(eligible.sum()),
        "pairwise_trainable_pairs": int((eligible & (unjudged > 0)).sum()),
        "candidate_minimum": int(stats.loc[eligible, "candidate_hands"].min()),
        "unjudged_minimum": int(unjudged[eligible].min()),
        "preference_pairs": int(preferences[eligible].sum()),
        "symmetric_rows": int(2 * preferences[eligible].sum()),
    }
    if observed != EXPECTED_PAIRWISE_READINESS[half]:
        raise V4InvariantError(f"pairwise readiness changed: {half}")
    zero_ids = set(stats.index[eligible & (unjudged == 0)].astype(str))
    if half == "early":
        if len(zero_ids) != 1:
            raise V4InvariantError("the frozen early zero-preference query changed")
        zero = pair_meta[pair_meta["pair_id"].astype(str).isin(zero_ids)]
        if (
            len(zero) != 1
            or str(zero.iloc[0]["behavior_family"]) != "directed_transfer"
            or int(zero.iloc[0]["outer_fold"]) != 3
        ):
            raise V4InvariantError("the early zero-preference cell changed")
    elif zero_ids:
        raise V4InvariantError(f"unexpected zero-preference query in {half}")


def _h0_tie_counts_v4(
    frame: pd.DataFrame,
    route_table: pd.DataFrame,
) -> tuple[int, int, int, int, int]:
    routes = _route_for_hands_v4(frame, route_table)
    scores = heuristic_scores_v4(frame, routes)
    work = frame[["pair_id", "started_at"]].copy()
    work["score"] = scores
    sizes = (
        work.groupby(["pair_id", "score"], sort=False)
        .size()
        .rename("tie_size")
        .reset_index()
    )
    ties = sizes[sizes["tie_size"] > 1]
    tied_rows = work.merge(
        ties[["pair_id", "score"]],
        on=["pair_id", "score"],
        how="inner",
        validate="many_to_many",
    )
    duplicate_timestamps = int(
        tied_rows.duplicated(
            ["pair_id", "score", "started_at"], keep=False
        ).sum()
    )
    ordered = work.sort_values(
        ["pair_id", "score"], ascending=[True, False], kind="stable"
    )
    ordered["position"] = ordered.groupby("pair_id", sort=False).cumcount() + 1
    bounds = (
        ordered.groupby(["pair_id", "score"], sort=False)["position"]
        .agg(["min", "max"])
        .reset_index()
    )
    crossings = int(
        ((bounds["min"] <= 5) & (bounds["max"] >= 5) & (bounds["min"] != bounds["max"])).sum()
    )
    return (
        int(work["pair_id"].nunique()),
        int(len(ties)),
        int(len(tied_rows)),
        crossings,
        duplicate_timestamps,
    )


def _validate_h0_tie_structure_v4(
    frame: pd.DataFrame,
    route_table: pd.DataFrame,
    view: str,
) -> None:
    if _h0_tie_counts_v4(frame, route_table) != EXPECTED_H0_TIE_STRUCTURE[view]:
        raise V4InvariantError(f"H0 tie structure changed: {view}")


def _join_reporting_family_v4(
    hands: pd.DataFrame,
    pair_meta: pd.DataFrame,
) -> pd.DataFrame:
    if "behavior_family" in hands:
        raise V4InvariantError("reporting family was joined before the freeze barrier")
    result = hands.merge(
        pair_meta[["pair_id", "behavior_family"]],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )
    if result["behavior_family"].isna().any():
        raise V4InvariantError("post-score reporting family is missing")
    return result


def _pair_features_for_spec_v4(
    full: FullBundleV4,
    time: TimeBundleV4 | None,
    spec: ViewSpecV4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.name == "full":
        return full.full_pair_features, full.full_pair_features
    if time is None:
        raise V4InvariantError("time pair features requested before Stage 3")
    if spec.name == "early_to_late":
        return time.early_pair_features, time.late_pair_features
    if spec.name == "late_to_early":
        return time.late_pair_features, time.early_pair_features
    raise V4InvariantError(f"unknown view: {spec.name}")


def _hands_for_spec_v4(
    full: FullBundleV4,
    time: TimeBundleV4 | None,
    spec: ViewSpecV4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = full.hands if spec.name == "full" else time.hands if time else None
    if frame is None:
        raise V4InvariantError("time hands requested before Stage 3")
    return _scope_v4(frame, spec.source_half), _scope_v4(frame, spec.target_half)


def _outer_route_table_v4(
    full: FullBundleV4,
    time: TimeBundleV4 | None,
    spec: ViewSpecV4,
) -> pd.DataFrame:
    if spec.name == "full":
        return full.pair_meta[["pair_id", "predicted_family"]].copy()
    if time is None:
        raise V4InvariantError("time outer route requested before Stage 3")
    return time.time_routes[["pair_id", spec.outer_route_column]].rename(
        columns={spec.outer_route_column: "predicted_family"}
    )


def _route_for_hands_v4(
    hands: pd.DataFrame,
    routes: pd.DataFrame,
) -> np.ndarray:
    if routes["pair_id"].astype(str).duplicated().any():
        raise V4InvariantError("route receipt has duplicate pairs")
    mapping = routes.set_index(routes["pair_id"].astype(str))["predicted_family"]
    result = hands["pair_id"].astype(str).map(mapping)
    if result.isna().any() or not set(result.astype(str)).issubset(FAMILIES):
        raise V4InvariantError("a scored hand lacks a predicted-family route")
    return result.astype(str).to_numpy()


def _training_family_for_hands_v4(
    hands: pd.DataFrame,
    pair_meta: pd.DataFrame,
) -> np.ndarray:
    mapping = pair_meta.set_index(pair_meta["pair_id"].astype(str))["behavior_family"]
    result = hands["pair_id"].astype(str).map(mapping)
    if result.isna().any() or not set(result.astype(str)).issubset(FAMILIES):
        raise V4InvariantError("a fitting hand lacks a permitted training family")
    return result.astype(str).to_numpy()


def _score_evidence_model_v4(
    model,
    transform: FrozenHandTransformV4,
    hands: pd.DataFrame,
    routes: Sequence[object] | None,
) -> np.ndarray:
    base = transform.transform(hands)
    matrix = base if routes is None else interaction_matrix_v4(base, routes)
    scores = model.decision_function(matrix)
    if not np.isfinite(scores).all():
        raise V4InvariantError("learned evidence score is non-finite")
    return np.asarray(scores, dtype=np.float64)


def _persist_nested_routes_v4(
    full: FullBundleV4,
    time: TimeBundleV4 | None,
    spec: ViewSpecV4,
    outer_fold: int,
    route_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    assignments = assign_inner_folds_v4(full.pair_meta, outer_fold)
    source_features, target_features = _pair_features_for_spec_v4(full, time, spec)
    routes = fit_nested_behavior_routes_v4(
        assignments,
        source_features,
        target_features,
        view=spec.name,
        outer_fold=outer_fold,
    )
    route_dir.mkdir(parents=True, exist_ok=True)
    path = route_dir / f"{spec.name}__outer_{outer_fold}.parquet"
    _atomic_parquet(path, routes)
    if "behavior_family" in routes:
        raise RuntimeError("true family leaked into a nested route receipt")
    return routes, {
        "view": spec.name,
        "outer_fold": int(outer_fold),
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(routes)),
        "true_family_persisted": False,
        "evidence_model_fit_before_route_hash": False,
    }


def _select_c_for_outer_v4(
    full: FullBundleV4,
    time: TimeBundleV4 | None,
    spec: ViewSpecV4,
    outer_fold: int,
    inner_routes: pd.DataFrame,
) -> tuple[float, dict[float, float], pd.DataFrame]:
    assignments = assign_inner_folds_v4(full.pair_meta, outer_fold)
    source_all, target_all = _hands_for_spec_v4(full, time, spec)
    source = _evidence_bearing_v4(source_all)
    target = _evidence_bearing_v4(target_all)
    scores_by_c: dict[float, list[pd.DataFrame]] = {
        c_value: [] for c_value in C_GRID
    }
    for inner_fold in range(3):
        train_ids = set(
            assignments.loc[
                assignments["inner_fold"] != inner_fold, "pair_id"
            ].astype(str)
        )
        valid_ids = set(
            assignments.loc[
                assignments["inner_fold"] == inner_fold, "pair_id"
            ].astype(str)
        )
        train_hands = source[source["pair_id"].astype(str).isin(train_ids)].copy()
        valid_hands = target[target["pair_id"].astype(str).isin(valid_ids)].copy()
        if train_hands.empty or valid_hands.empty:
            raise V4InvariantError("an inner evidence cell is empty")
        if set(train_hands["table_id"].astype(str)) & set(
            valid_hands["table_id"].astype(str)
        ):
            raise V4InvariantError("inner evidence train/validation pools overlap")
        transform = FrozenHandTransformV4.fit(train_hands)
        train_routes = _training_family_for_hands_v4(train_hands, full.pair_meta)
        preferences = build_pairwise_preferences_v4(
            interaction_matrix_v4(transform.transform(train_hands), train_routes),
            train_hands["pair_id"],
            train_hands["is_evidence"],
        )
        valid_routes = _route_for_hands_v4(valid_hands, inner_routes)
        valid_matrix = interaction_matrix_v4(
            transform.transform(valid_hands), valid_routes
        )
        for c_value in C_GRID:
            model = fit_evidence_ranker_v4(preferences, c_value)
            score = model.decision_function(valid_matrix)
            if not np.isfinite(score).all():
                raise V4InvariantError("inner evidence score is non-finite")
            scored = valid_hands.copy()
            scored["l1_score"] = score
            if any("family" in column.lower() for column in scored.columns):
                raise RuntimeError("held-out family entered C-selection scoring")
            scores_by_c[c_value].append(query_ap_frame_v4(scored, "l1_score"))
    pooled = {
        c_value: pd.concat(parts, ignore_index=True)
        for c_value, parts in scores_by_c.items()
    }
    expected_pairs = set(assignments["pair_id"].astype(str)) & set(
        target.loc[target["is_evidence"], "pair_id"].astype(str)
    )
    for frame in pooled.values():
        if set(frame["pair_id"].astype(str)) != expected_pairs:
            raise V4InvariantError("inner C-selection query coverage changed")
    selected_c, mean_by_c = select_c_no_family_v4(pooled)
    receipt = pd.DataFrame(
        {
            "view": [spec.name],
            "outer_fold": [outer_fold],
            "selected_c": [selected_c],
            "selection_statistic": ["unweighted_query_mean_random_tie_ap5"],
            "family_field_used": [False],
            **{f"mean_ap5_c_{c_value}": [mean_by_c[c_value]] for c_value in C_GRID},
        }
    )
    if any("family" in column.lower() and column != "family_field_used" for column in receipt):
        raise RuntimeError("family result entered C-selection receipt")
    return selected_c, mean_by_c, receipt


def _fit_outer_l1_h0_v4(
    full: FullBundleV4,
    time: TimeBundleV4 | None,
    spec: ViewSpecV4,
    outer_fold: int,
    selected_c: float,
) -> tuple[pd.DataFrame, OuterFitStateV4, dict[str, object]]:
    source_all, target_all = _hands_for_spec_v4(full, time, spec)
    train_ids = set(
        full.pair_meta.loc[
            full.pair_meta["outer_fold"] != outer_fold, "pair_id"
        ].astype(str)
    )
    valid_ids = set(
        full.pair_meta.loc[
            full.pair_meta["outer_fold"] == outer_fold, "pair_id"
        ].astype(str)
    )
    source_train = _evidence_bearing_v4(
        source_all[source_all["pair_id"].astype(str).isin(train_ids)]
    )
    target_fold = target_all[target_all["pair_id"].astype(str).isin(valid_ids)].copy()
    eligible_series = target_fold.groupby("pair_id", sort=False)[
        "is_evidence"
    ].transform("any")
    eligible = eligible_series.to_numpy(dtype=bool)
    target_valid = target_fold[eligible].copy()
    if source_train.empty or target_valid.empty:
        raise V4InvariantError("an outer evidence cell is empty")
    if set(source_train["table_id"].astype(str)) & set(
        target_fold["table_id"].astype(str)
    ):
        raise V4InvariantError("outer evidence train/validation pools overlap")

    transform = FrozenHandTransformV4.fit(source_train)
    source_base = transform.transform(source_train)
    training_routes = _training_family_for_hands_v4(source_train, full.pair_meta)
    preferences = build_pairwise_preferences_v4(
        interaction_matrix_v4(source_base, training_routes),
        source_train["pair_id"],
        source_train["is_evidence"],
    )
    model = fit_evidence_ranker_v4(preferences, selected_c)
    target_routes = _route_for_hands_v4(
        target_fold, _outer_route_table_v4(full, time, spec)
    )
    valid_routes = target_routes[eligible]
    h0 = heuristic_scores_v4(target_valid, valid_routes)
    l1 = _score_evidence_model_v4(model, transform, target_valid, valid_routes)

    receipt_columns = [
        "pair_id",
        "hand_id",
        "started_at",
        "table_id",
        "outer_fold",
        "is_evidence",
    ]
    if "phase_progress" in target_fold:
        receipt_columns.append("phase_progress")
    receipt = target_fold[receipt_columns].copy()
    receipt["view"] = spec.name
    receipt["eligible_query"] = eligible
    receipt["predicted_route"] = target_routes
    receipt["h0_score"] = np.nan
    receipt["l1_score"] = np.nan
    receipt.loc[eligible_series, "h0_score"] = h0
    receipt.loc[eligible_series, "l1_score"] = l1
    if any(column == "behavior_family" for column in receipt):
        raise RuntimeError("held-out family entered the pre-freeze score receipt")
    state = OuterFitStateV4(
        spec=spec,
        outer_fold=outer_fold,
        selected_c=selected_c,
        transform=transform,
        source_train=source_train,
        source_base=source_base,
        target_fold=target_fold,
        target_valid=target_valid,
        eligible=eligible,
        target_routes=target_routes,
    )
    diagnostics = {
        "view": spec.name,
        "outer_fold": outer_fold,
        "selected_c": selected_c,
        "training_queries_evidence_bearing": int(source_train["pair_id"].nunique()),
        "training_queries_pairwise_trainable": int(
            (preferences.query_receipt["preference_pairs"] > 0).sum()
        ),
        "training_preference_pairs": int(
            preferences.query_receipt["preference_pairs"].sum()
        ),
        "validation_queries_eligible": int(target_valid["pair_id"].nunique()),
        "transform": transform.receipt(),
        "unknown_pairs_used": 0,
        "confirmed_non_target_pairs_used": 0,
        "held_out_true_family_used_for_score": False,
    }
    return receipt, state, diagnostics


def _fit_outer_l0_from_state_v4(
    state: OuterFitStateV4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    preferences = build_pairwise_preferences_v4(
        state.source_base,
        state.source_train["pair_id"],
        state.source_train["is_evidence"],
    )
    model = fit_evidence_ranker_v4(preferences, state.selected_c)
    l0 = _score_evidence_model_v4(
        model, state.transform, state.target_valid, None
    )
    columns = [
        "pair_id",
        "hand_id",
        "started_at",
        "table_id",
        "outer_fold",
        "is_evidence",
    ]
    if "phase_progress" in state.target_fold:
        columns.append("phase_progress")
    receipt = state.target_fold[columns].copy()
    receipt["view"] = state.spec.name
    receipt["eligible_query"] = state.eligible
    receipt["l0_score"] = np.nan
    receipt.loc[state.target_fold.index[state.eligible], "l0_score"] = l0
    return receipt, {
        "view": state.spec.name,
        "outer_fold": state.outer_fold,
        "matched_l1_c": state.selected_c,
        "matched_c": True,
        "held_out_true_family_used_for_score": False,
    }


def _eligible_score_frame_v4(
    per_hand: pd.DataFrame,
    view: str,
    score_columns: Sequence[str],
) -> pd.DataFrame:
    frame = per_hand[(per_hand["view"] == view) & per_hand["eligible_query"]].copy()
    required = [
        "pair_id",
        "hand_id",
        "started_at",
        "table_id",
        "outer_fold",
        "is_evidence",
        *score_columns,
    ]
    result = frame.loc[:, required]
    if any("family" in column.lower() for column in result):
        raise RuntimeError("held-out family entered the eligible score frame")
    return result


def _top_five_receipt_v4(
    eligible: pd.DataFrame,
    view: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    parts = []
    relevance = eligible[["pair_id", "hand_id", "is_evidence"]]
    for method in methods:
        score_column = f"{method}_score"
        top = deterministic_top_five_v4(
            eligible[["pair_id", "hand_id", "started_at", score_column]],
            score_column,
        )
        top = top.merge(
            relevance,
            on=["pair_id", "hand_id"],
            how="left",
            validate="one_to_one",
        )
        top["view"] = view
        top["method"] = method
        parts.append(top)
    return pd.concat(parts, ignore_index=True)


def _query_scores_with_reporting_family_v4(
    eligible: pd.DataFrame,
    score_column: str,
    pair_meta: pd.DataFrame,
) -> pd.DataFrame:
    # query_ap_frame_v4 rejects any family column, enforcing the score-freeze
    # side of the reporting join barrier.
    query = query_ap_frame_v4(eligible, score_column)
    result = query.merge(
        pair_meta[["pair_id", "behavior_family"]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if result["behavior_family"].isna().any():
        raise V4InvariantError("post-freeze query family join is incomplete")
    return result


def _query_meta_for_plan_v4(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["pair_id", "table_id", "outer_fold"]
    grouped = frame.groupby("pair_id", sort=False)[["table_id", "outer_fold"]]
    if grouped.nunique(dropna=False).to_numpy().max() != 1:
        raise V4InvariantError("query metadata is not one-to-one")
    return frame[columns].drop_duplicates("pair_id", keep="first").reset_index(drop=True)


def _gate_primary_v4(sensitivity: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "overall_delta_at_least_0_040": sensitivity["overall"]["point_delta"] >= 0.040,
        "lower_sensitivity_endpoint_above_zero": sensitivity["overall"][
            "sensitivity_lower_05"
        ]
        > 0.0,
        "all_family_point_deltas_nonnegative": all(
            sensitivity["family"][family]["point_delta"] >= 0.0
            for family in FAMILIES
        ),
        "at_least_four_outer_folds_positive": sum(
            sensitivity["outer_fold"][str(fold)]["point_delta"] > 0.0
            for fold in range(OUTER_FOLDS)
        )
        >= 4,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _gate_attribution_v4(sensitivity: Mapping[str, object]) -> dict[str, object]:
    macro = float(
        np.mean(
            [sensitivity["family"][family]["point_delta"] for family in FAMILIES]
        )
    )
    checks = {
        "macro_delta_at_least_0_005": macro >= 0.005,
        "coordinated_isolation_delta_nonnegative": sensitivity["family"][
            "coordinated_isolation"
        ]["point_delta"]
        >= 0.0,
    }
    return {
        "checks": checks,
        "unweighted_macro_family_point_delta": macro,
        "passed": all(checks.values()),
    }


def _gate_time_direction_v4(sensitivity: Mapping[str, object]) -> dict[str, object]:
    macro = float(
        np.mean(
            [sensitivity["family"][family]["point_delta"] for family in FAMILIES]
        )
    )
    checks = {
        "macro_delta_at_least_0_020": macro >= 0.020,
        "overall_delta_positive": sensitivity["overall"]["point_delta"] > 0.0,
        "lower_sensitivity_endpoint_above_zero": sensitivity["overall"][
            "sensitivity_lower_05"
        ]
        > 0.0,
        "all_direction_family_deltas_nonnegative": all(
            sensitivity["family"][family]["point_delta"] >= 0.0
            for family in FAMILIES
        ),
        "at_least_four_outer_folds_positive": sum(
            sensitivity["outer_fold"][str(fold)]["point_delta"] > 0.0
            for fold in range(OUTER_FOLDS)
        )
        >= 4,
    }
    return {
        "checks": checks,
        "unweighted_macro_family_point_delta": macro,
        "passed": all(checks.values()),
    }


@dataclass
class Stage1OutcomeV4:
    passed: bool
    report: dict[str, object]
    report_sha256: str
    states: list[OuterFitStateV4]
    per_hand: pd.DataFrame
    reporting_hands: pd.DataFrame
    query_scores: dict[str, pd.DataFrame]
    plan: BayesianClusterWeightPlanV4


@dataclass
class Stage2OutcomeV4:
    passed: bool
    report: dict[str, object]
    report_sha256: str


def _persist_reporting_family_v4(
    eligible_scores: pd.DataFrame,
    pair_meta: pd.DataFrame,
    path: Path,
) -> tuple[pd.DataFrame, str]:
    reporting = _join_reporting_family_v4(eligible_scores, pair_meta)
    receipt = reporting[
        ["pair_id", "hand_id", "table_id", "outer_fold", "behavior_family"]
    ].copy()
    _atomic_parquet(path, receipt)
    return reporting, _sha256(path)


def _run_stage_1_full_primary_v4(
    full: FullBundleV4,
    work_dir: Path,
    coverage_book: CoverageBookV4,
) -> Stage1OutcomeV4:
    _assert_paths_absent(_later_stage_paths(work_dir, 1))
    _validate_h0_tie_structure_v4(
        full.hands,
        full.pair_meta[["pair_id", "predicted_family"]],
        "full",
    )
    eligible_unscored = _evidence_bearing_v4(full.hands)
    plan = BayesianClusterWeightPlanV4.create(
        _query_meta_for_plan_v4(eligible_unscored), "full"
    )
    plans = {"full": plan}
    weight_hash = frozen_stage_weight_stream_sha256_v4(plans, ("full",))
    no_header_hash = frozen_stage_weight_stream_sha256_v4(
        plans, ("full",), include_headers=False
    )
    if weight_hash != STAGE_WEIGHT_STREAM_SHA256[("full",)]:
        raise V4InvariantError("Stage-1 full weight stream changed")
    if no_header_hash != STAGE_WEIGHT_STREAM_NO_HEADER_SHA256[("full",)]:
        raise V4InvariantError("Stage-1 no-header weight stream changed")

    route_dir = work_dir / STAGE_ARTIFACTS[1][0]
    score_parts = []
    states = []
    route_receipts = []
    selection_receipts = []
    fit_receipts = []
    for outer_fold in range(OUTER_FOLDS):
        routes, route_receipt = _persist_nested_routes_v4(
            full, None, FULL_SPEC, outer_fold, route_dir
        )
        route_receipts.append(route_receipt)
        selected_c, mean_by_c, selection = _select_c_for_outer_v4(
            full, None, FULL_SPEC, outer_fold, routes
        )
        selection_receipts.append(selection.iloc[0].to_dict())
        score_receipt, state, fit_receipt = _fit_outer_l1_h0_v4(
            full, None, FULL_SPEC, outer_fold, selected_c
        )
        if fit_receipt["held_out_true_family_used_for_score"] is not False:
            raise RuntimeError("held-out family score control changed")
        score_parts.append(score_receipt)
        states.append(state)
        fit_receipts.append(fit_receipt)
        if set(mean_by_c) != set(C_GRID):
            raise RuntimeError("C-selection receipt grid changed")

    per_hand = pd.concat(score_parts, ignore_index=True)
    score_path = work_dir / STAGE_ARTIFACTS[1][1]
    _atomic_parquet(score_path, per_hand)
    score_sha256 = _sha256(score_path)
    eligible = _eligible_score_frame_v4(per_hand, "full", ("h0_score", "l1_score"))
    top_five = _top_five_receipt_v4(eligible, "full", ("h0", "l1"))
    top_path = work_dir / STAGE_ARTIFACTS[1][2]
    _atomic_parquet(top_path, top_five)
    top_sha256 = _sha256(top_path)

    # The family join occurs only after both pre-family score receipts have
    # immutable hashes.
    reporting_path = work_dir / STAGE_ARTIFACTS[1][3]
    reporting, reporting_sha256 = _persist_reporting_family_v4(
        eligible, full.pair_meta, reporting_path
    )
    query_scores = {
        method: _query_scores_with_reporting_family_v4(
            eligible, f"{method}_score", full.pair_meta
        )
        for method in ("h0", "l1")
    }
    methods = {
        method: metric_tree_v4(
            query_scores[method], reporting, "full", coverage_book
        )
        for method in ("h0", "l1")
    }
    delta = paired_delta_frame_v4(query_scores["l1"], query_scores["h0"])
    sensitivity = sensitivity_tree_v4(
        delta, reporting, "full", plan, coverage_book
    )
    gate = _gate_primary_v4(sensitivity)
    report = {
        "study": "EVP-DRAFT-04",
        "stage": 1,
        "stage_name": "full_primary",
        "status": "PASS_STAGE_1_PRIMARY"
        if gate["passed"]
        else "STOP_PRIMARY_UNEXECUTED_LATER_STAGES",
        "binding": {
            **_consumption_receipt_binding_v4(work_dir),
            "v4_draft_sha256": V4_DRAFT_SHA256,
            "coverage_contract_sha256": V4_COVERAGE_CONTRACT_SHA256,
            "method_audit_sha256": V4_METHOD_AUDIT_SHA256,
            "full_stage_source_sha256": full.source_hashes,
            "full_weight_stream_sha256": weight_hash,
            "full_weight_stream_no_header_sha256": no_header_hash,
            "nested_routes": route_receipts,
            "score_receipt": str(score_path),
            "score_receipt_sha256": score_sha256,
            "top_five_receipt": str(top_path),
            "top_five_receipt_sha256": top_sha256,
            "reporting_family_receipt": str(reporting_path),
            "reporting_family_receipt_sha256": reporting_sha256,
            "score_and_top_five_hashed_before_reporting_family_join": True,
        },
        "selection": selection_receipts,
        "fits": fit_receipts,
        "methods": methods,
        "l1_minus_h0_sensitivity": sensitivity,
        "gate": gate,
        "controls": {
            "stage_1_full_data_read": True,
            "stage_1_l1_fit": True,
            "stage_1_l0_fit": False,
            "time_source_hashes_computed": False,
            "time_data_read": False,
            "stage_2_artifacts_exist": False,
            "stage_3_artifacts_exist": False,
            "evaluation_rows_loaded": False,
            "evaluation_scored": False,
            "experiment_number_assigned": False,
            "candidate_promoted": False,
            "submission_created_or_modified": False,
            "rerun_allowed": False,
        },
    }
    report_path = work_dir / STAGE_ARTIFACTS[1][4]
    _atomic_json(report_path, report)
    report_sha256 = _sha256(report_path)
    if not gate["passed"]:
        _assert_paths_absent(_later_stage_paths(work_dir, 1))
    return Stage1OutcomeV4(
        passed=bool(gate["passed"]),
        report=report,
        report_sha256=report_sha256,
        states=states,
        per_hand=per_hand,
        reporting_hands=reporting,
        query_scores=query_scores,
        plan=plan,
    )


def _run_stage_2_attribution_v4(
    full: FullBundleV4,
    stage_1: Stage1OutcomeV4,
    work_dir: Path,
    coverage_book: CoverageBookV4,
) -> Stage2OutcomeV4:
    if not stage_1.passed or stage_1.report.get("status") != "PASS_STAGE_1_PRIMARY":
        raise PermissionError("Stage 2 requires the exact Stage-1 PASS")
    stage_1_path = work_dir / STAGE_ARTIFACTS[1][4]
    if _sha256(stage_1_path) != stage_1.report_sha256:
        raise PermissionError("Stage-1 PASS receipt changed")
    _assert_paths_absent(_later_stage_paths(work_dir, 2))
    parts = []
    fit_receipts = []
    for state in stage_1.states:
        receipt, fit_receipt = _fit_outer_l0_from_state_v4(state)
        parts.append(receipt)
        fit_receipts.append(fit_receipt)
    per_hand = pd.concat(parts, ignore_index=True)
    score_path = work_dir / STAGE_ARTIFACTS[2][0]
    _atomic_parquet(score_path, per_hand)
    score_sha256 = _sha256(score_path)
    eligible = _eligible_score_frame_v4(per_hand, "full", ("l0_score",))
    top_five = _top_five_receipt_v4(eligible, "full", ("l0",))
    top_path = work_dir / STAGE_ARTIFACTS[2][1]
    _atomic_parquet(top_path, top_five)
    top_sha256 = _sha256(top_path)
    query_l0 = _query_scores_with_reporting_family_v4(
        eligible, "l0_score", full.pair_meta
    )
    method = metric_tree_v4(query_l0, stage_1.reporting_hands, "full", coverage_book)
    delta = paired_delta_frame_v4(stage_1.query_scores["l1"], query_l0)
    sensitivity = sensitivity_tree_v4(
        delta,
        stage_1.reporting_hands,
        "full",
        stage_1.plan,
        coverage_book,
    )
    gate = _gate_attribution_v4(sensitivity)
    report = {
        "study": "EVP-DRAFT-04",
        "stage": 2,
        "stage_name": "matched_c_full_attribution",
        "status": "PASS_STAGE_2_ATTRIBUTION"
        if gate["passed"]
        else "STOP_ATTRIBUTION_UNEXECUTED_TIME",
        "prior_stage_1_sha256": stage_1.report_sha256,
        "binding": {
            "l0_score_receipt": str(score_path),
            "l0_score_receipt_sha256": score_sha256,
            "l0_top_five_receipt": str(top_path),
            "l0_top_five_receipt_sha256": top_sha256,
        },
        "fits": fit_receipts,
        "l0_method": method,
        "l1_minus_l0_sensitivity": sensitivity,
        "gate": gate,
        "controls": {
            "stage_2_l0_fit": True,
            "matched_l1_c": True,
            "time_source_hashes_computed": False,
            "time_data_read": False,
            "stage_3_artifacts_exist": False,
            "evaluation_rows_loaded": False,
            "evaluation_scored": False,
            "experiment_number_assigned": False,
            "candidate_promoted": False,
            "submission_created_or_modified": False,
            "rerun_allowed": False,
        },
    }
    report_path = work_dir / STAGE_ARTIFACTS[2][2]
    _atomic_json(report_path, report)
    report_sha256 = _sha256(report_path)
    if not gate["passed"]:
        _assert_paths_absent(_later_stage_paths(work_dir, 2))
    return Stage2OutcomeV4(
        passed=bool(gate["passed"]),
        report=report,
        report_sha256=report_sha256,
    )


def _run_stage_3_time_v4(
    data_dir: Path,
    full: FullBundleV4,
    stage_2: Stage2OutcomeV4,
    work_dir: Path,
    coverage_book: CoverageBookV4,
) -> dict[str, object]:
    if not stage_2.passed or stage_2.report.get("status") != "PASS_STAGE_2_ATTRIBUTION":
        raise PermissionError("Stage 3 requires the exact Stage-2 PASS")
    stage_2_path = work_dir / STAGE_ARTIFACTS[2][2]
    if _sha256(stage_2_path) != stage_2.report_sha256:
        raise PermissionError("Stage-2 PASS receipt changed")

    # This is the first call that can hash/open time sources or phase_progress.
    time = _load_time_inputs_v4(data_dir, work_dir, full, coverage_book)
    for spec in TIME_SPECS:
        _, target = _hands_for_spec_v4(full, time, spec)
        _validate_h0_tie_structure_v4(
            target,
            _outer_route_table_v4(full, time, spec),
            spec.name,
        )
    plans = {}
    for spec in TIME_SPECS:
        _, target = _hands_for_spec_v4(full, time, spec)
        eligible = _evidence_bearing_v4(target)
        plans[spec.name] = BayesianClusterWeightPlanV4.create(
            _query_meta_for_plan_v4(eligible), spec.name
        )
    for views in (
        ("early_to_late",),
        ("late_to_early",),
        ("early_to_late", "late_to_early"),
    ):
        observed = frozen_stage_weight_stream_sha256_v4(plans, views)
        negative = frozen_stage_weight_stream_sha256_v4(
            plans, views, include_headers=False
        )
        if observed != STAGE_WEIGHT_STREAM_SHA256[views]:
            raise V4InvariantError(f"Stage-3 weight stream changed: {views}")
        if negative != STAGE_WEIGHT_STREAM_NO_HEADER_SHA256[views]:
            raise V4InvariantError(f"Stage-3 no-header stream changed: {views}")

    route_dir = work_dir / STAGE_ARTIFACTS[3][0]
    score_parts = []
    route_receipts = []
    selection_receipts = []
    fit_receipts = []
    for spec in TIME_SPECS:
        for outer_fold in range(OUTER_FOLDS):
            routes, route_receipt = _persist_nested_routes_v4(
                full, time, spec, outer_fold, route_dir
            )
            route_receipts.append(route_receipt)
            selected_c, _, selection = _select_c_for_outer_v4(
                full, time, spec, outer_fold, routes
            )
            selection_receipts.append(selection.iloc[0].to_dict())
            receipt, state, l1_fit = _fit_outer_l1_h0_v4(
                full, time, spec, outer_fold, selected_c
            )
            l0_receipt, l0_fit = _fit_outer_l0_from_state_v4(state)
            keys = ["pair_id", "hand_id"]
            if not np.array_equal(
                receipt[keys].astype(str).to_numpy(),
                l0_receipt[keys].astype(str).to_numpy(),
            ):
                raise RuntimeError("time L0 and L1 score receipts changed row order")
            receipt["l0_score"] = l0_receipt["l0_score"].to_numpy()
            score_parts.append(receipt)
            fit_receipts.extend((l1_fit, l0_fit))

    per_hand = pd.concat(score_parts, ignore_index=True)
    per_hand["time_half"] = np.where(
        per_hand["phase_progress"] <= 0.5, "early", "late"
    )
    score_path = work_dir / STAGE_ARTIFACTS[3][1]
    _atomic_parquet(score_path, per_hand)
    score_sha256 = _sha256(score_path)
    top_parts = []
    eligible_by_view = {}
    for spec in TIME_SPECS:
        eligible = _eligible_score_frame_v4(
            per_hand, spec.name, ("h0_score", "l0_score", "l1_score")
        )
        eligible_by_view[spec.name] = eligible
        top_parts.append(
            _top_five_receipt_v4(eligible, spec.name, ("h0", "l0", "l1"))
        )
    top_five = pd.concat(top_parts, ignore_index=True)
    top_path = work_dir / STAGE_ARTIFACTS[3][2]
    _atomic_parquet(top_path, top_five)
    top_sha256 = _sha256(top_path)

    reporting_parts = []
    for spec in TIME_SPECS:
        reporting_parts.append(
            _join_reporting_family_v4(eligible_by_view[spec.name], full.pair_meta)
        )
    reporting_all = pd.concat(reporting_parts, ignore_index=True)
    reporting_path = work_dir / STAGE_ARTIFACTS[3][3]
    _atomic_parquet(
        reporting_path,
        reporting_all[
            ["pair_id", "hand_id", "table_id", "outer_fold", "behavior_family"]
        ].assign(view=np.repeat([spec.name for spec in TIME_SPECS], [len(x) for x in reporting_parts])),
    )
    reporting_sha256 = _sha256(reporting_path)

    results = {}
    gates = {}
    for spec, reporting in zip(TIME_SPECS, reporting_parts):
        eligible = eligible_by_view[spec.name]
        queries = {
            method: _query_scores_with_reporting_family_v4(
                eligible, f"{method}_score", full.pair_meta
            )
            for method in ("h0", "l0", "l1")
        }
        methods = {
            method: metric_tree_v4(
                queries[method], reporting, spec.name, coverage_book
            )
            for method in ("h0", "l0", "l1")
        }
        contrasts = {}
        for name, left, right in (
            ("l1_minus_h0", "l1", "h0"),
            ("l1_minus_l0", "l1", "l0"),
        ):
            delta = paired_delta_frame_v4(queries[left], queries[right])
            contrasts[name] = sensitivity_tree_v4(
                delta, reporting, spec.name, plans[spec.name], coverage_book
            )
        gate = _gate_time_direction_v4(contrasts["l1_minus_h0"])
        results[spec.name] = {"methods": methods, "sensitivity": contrasts}
        gates[spec.name] = gate
    all_passed = all(gate["passed"] for gate in gates.values())
    report = {
        "study": "EVP-DRAFT-04",
        "stage": 3,
        "stage_name": "bidirectional_time",
        "status": "PASS_ALL_DEVELOPMENT_GATES"
        if all_passed
        else "STOP_FAILED_BIDIRECTIONAL_TIME_GATE",
        "prior_stage_2_sha256": stage_2.report_sha256,
        "binding": {
            "time_stage_source_sha256": time.source_hashes,
            "stage_weight_stream_sha256": {
                "+".join(views): STAGE_WEIGHT_STREAM_SHA256[views]
                for views in (
                    ("early_to_late",),
                    ("late_to_early",),
                    ("early_to_late", "late_to_early"),
                )
            },
            "nested_routes": route_receipts,
            "score_receipt": str(score_path),
            "score_receipt_sha256": score_sha256,
            "top_five_receipt": str(top_path),
            "top_five_receipt_sha256": top_sha256,
            "reporting_family_receipt": str(reporting_path),
            "reporting_family_receipt_sha256": reporting_sha256,
            "score_and_top_five_hashed_before_reporting_family_join": True,
        },
        "selection": selection_receipts,
        "fits": fit_receipts,
        "results": results,
        "gates": gates,
        "all_gates_passed": all_passed,
        "controls": {
            "stage_3_time_data_read": True,
            "time_scores_hashed_before_aggregate_metrics": True,
            "unknown_ground_truth_labels_assigned": 0,
            "unknown_pairs_used": 0,
            "confirmed_non_target_pairs_used": 0,
            "evaluation_rows_loaded": False,
            "evaluation_scored": False,
            "experiment_number_assigned": False,
            "candidate_promoted": False,
            "submission_created_or_modified": False,
            "rerun_allowed": False,
        },
    }
    report_path = work_dir / STAGE_ARTIFACTS[3][4]
    _atomic_json(report_path, report)
    return report


def run_authorized_validation_v4(
    data_dir: str | Path,
    work_dir: str | Path,
    authorization_path: str | Path,
) -> dict[str, object]:
    """Consume one future authorization and execute the frozen state machine."""

    data_root = Path(data_dir)
    output_root = Path(work_dir)
    if not output_root.is_dir():
        raise PermissionError("the pre-existing audited work directory is required")
    _validate_result_execution_authorization_v4(
        Path(authorization_path), output_root
    )
    coverage_book = _load_coverage_book_canonical()
    full = _load_full_inputs_v4(data_root, output_root, coverage_book)
    stage_1 = _run_stage_1_full_primary_v4(full, output_root, coverage_book)
    if not stage_1.passed:
        return stage_1.report
    stage_2 = _run_stage_2_attribution_v4(
        full, stage_1, output_root, coverage_book
    )
    if not stage_2.passed:
        return stage_2.report
    return _run_stage_3_time_v4(
        data_root, full, stage_2, output_root, coverage_book
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    parser.add_argument("--authorization", required=True)
    arguments = parser.parse_args()
    result = run_authorized_validation_v4(
        arguments.data_dir, arguments.work_dir, arguments.authorization
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
