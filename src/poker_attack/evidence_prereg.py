"""Hash-locked executor for the EVP-DRAFT-03 evidence-only validation.

The executor refuses to read competition data unless a later supervisory
receipt explicitly binds the preregistration, this implementation bundle, and
its static audit.  The current implementation-only approval is intentionally
insufficient to call :func:`run_authorized_validation`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .evidence_prereg_core import (
    B001_FULL_FEATURES,
    B001_TIME_FEATURES,
    C_GRID,
    DRAFT_SHA256,
    FAMILIES,
    FEATURES,
    H0_SIGNAL_BY_FAMILY,
    OUTER_FOLDS,
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
    fit_evidence_ranker,
    fit_nested_behavior_routes,
    frozen_pool_weight_stream_sha256,
    heuristic_scores,
    interaction_matrix,
    macro_family_mean,
    paired_delta_frame,
    query_ap_frame,
)
from .schema import input_paths


RESULT_EXECUTION_STATUS = "RESULT_BEARING_EXECUTION_APPROVED"
IMPLEMENTATION_ONLY_STATUS = "IMPLEMENTATION_ONLY_APPROVED"
CONSUMPTION_FILENAME = "EVP_result_execution_consumed.json"

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

EXPECTED_FAMILY_TOTALS = {
    "directed_transfer": (148, 18150, 725),
    "soft_play": (132, 15699, 632),
    "coordinated_isolation": (92, 11280, 460),
}
EXPECTED_FULL_VALID_PAIRS = {
    "directed_transfer": (30, 29, 30, 30, 29),
    "soft_play": (26, 26, 27, 27, 26),
    "coordinated_isolation": (18, 18, 19, 18, 19),
}
EXPECTED_FULL_EVIDENCE = {
    "directed_transfer": (148, 142, 149, 147, 139),
    "soft_play": (121, 128, 124, 132, 127),
    "coordinated_isolation": (90, 90, 95, 90, 95),
}
EXPECTED_TIME_TARGET_PAIRS = {
    "early_to_late": {
        "directed_transfer": (23, 23, 22, 19, 24),
        "soft_play": (20, 21, 20, 22, 23),
        "coordinated_isolation": (7, 11, 12, 10, 7),
    },
    "late_to_early": {
        "directed_transfer": (28, 28, 26, 27, 27),
        "soft_play": (24, 24, 26, 27, 23),
        "coordinated_isolation": (17, 15, 19, 18, 18),
    },
}
EXPECTED_TIME_TARGET_EVIDENCE = {
    "early_to_late": {
        "directed_transfer": (49, 58, 55, 42, 60),
        "soft_play": (45, 50, 39, 42, 54),
        "coordinated_isolation": (17, 36, 28, 24, 18),
    },
    "late_to_early": {
        "directed_transfer": (99, 84, 94, 105, 79),
        "soft_play": (76, 78, 85, 90, 73),
        "coordinated_isolation": (73, 54, 67, 66, 77),
    },
}
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


@dataclass(frozen=True)
class ViewSpec:
    name: str
    source_half: str
    target_half: str
    outer_route_column: str


VIEW_SPECS = (
    ViewSpec("full", "full", "full", "predicted_family"),
    ViewSpec(
        "early_to_late",
        "early",
        "late",
        "early_to_late__predicted_family",
    ),
    ViewSpec(
        "late_to_early",
        "late",
        "early",
        "late_to_early__predicted_family",
    ),
)


@dataclass
class InputBundle:
    pair_meta: pd.DataFrame
    positive_hands: pd.DataFrame
    full_pair_features: pd.DataFrame
    early_pair_features: pd.DataFrame
    late_pair_features: pd.DataFrame
    source_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_result_execution_authorization(
    authorization_path: Path,
    work_dir: Path,
) -> dict[str, object]:
    """Validate permission before any competition-data path is opened."""

    if not authorization_path.is_file():
        raise PermissionError("a result-bearing supervisory receipt is required")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") == IMPLEMENTATION_ONLY_STATUS:
        raise PermissionError("implementation-only approval cannot run real data")
    if authorization.get("status") != RESULT_EXECUTION_STATUS:
        raise PermissionError("result-bearing execution is not approved")
    binding = authorization.get("binding", {})
    if binding.get("draft_sha256") != DRAFT_SHA256:
        raise PermissionError("authorization binds a different preregistration")
    if binding.get("prereg_static_audit_sha256") != PREREG_STATIC_AUDIT_SHA256:
        raise PermissionError("authorization binds a different prereg static audit")
    project_root = _project_root()
    draft_path = project_root / "poker/EVIDENCE_ONLY_VALIDATION_PREREG_DRAFT_V3.md"
    prereg_audit_path = project_root / "poker/work/EVP_prereg_v3_static_audit.json"
    if _sha256(draft_path) != DRAFT_SHA256:
        raise PermissionError("the bound preregistration document changed")
    if _sha256(prereg_audit_path) != PREREG_STATIC_AUDIT_SHA256:
        raise PermissionError("the bound preregistration static audit changed")
    core_path = Path(__file__).with_name("evidence_prereg_core.py")
    runner_path = Path(__file__)
    core_sha256 = _sha256(core_path)
    runner_sha256 = _sha256(runner_path)
    if binding.get("core_implementation_sha256") != core_sha256:
        raise PermissionError("authorization does not bind the current core source")
    if binding.get("runner_implementation_sha256") != runner_sha256:
        raise PermissionError("authorization does not bind the current runner source")
    audit_path_text = binding.get("implementation_static_audit_path")
    if not isinstance(audit_path_text, str):
        raise PermissionError("implementation static-audit path is absent")
    audit_path = (_project_root() / audit_path_text).resolve()
    if not audit_path.is_file() or binding.get(
        "implementation_static_audit_sha256"
    ) != _sha256(audit_path):
        raise PermissionError("implementation static audit is absent or changed")
    implementation_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if implementation_audit.get("status") != "PASS_IMPLEMENTATION_PRESENT_UNEXECUTED":
        raise PermissionError("implementation static audit is not a current PASS")
    audit_binding = implementation_audit.get("binding", {})
    if audit_binding.get("draft_sha256") != DRAFT_SHA256 or audit_binding.get(
        "prereg_static_audit_sha256"
    ) != PREREG_STATIC_AUDIT_SHA256:
        raise PermissionError("implementation audit binds a different preregistration")
    if audit_binding.get("core_implementation_sha256") != core_sha256 or audit_binding.get(
        "runner_implementation_sha256"
    ) != runner_sha256:
        raise PermissionError("implementation audit does not bind current source")
    audit_controls = implementation_audit.get("controls", {})
    for field in (
        "real_model_fit_performed",
        "aggregate_performance_viewed",
        "evaluation_rows_loaded",
        "evaluation_scored",
        "experiment_number_assigned",
        "submission_created_or_modified",
        "preregistration_executed",
    ):
        if audit_controls.get(field) is not False:
            raise PermissionError(f"implementation audit control is not false: {field}")
    permissions = authorization.get("permissions", {})
    exact_permissions = {
        "fit_preregistered_nested_b001_and_evidence_models_once": True,
        "view_development_evidence_performance_once": True,
        "load_evaluation_rows": False,
        "score_evaluation_rows": False,
        "create_or_modify_submission": False,
        "assign_experiment_number": False,
        "rerun_after_any_stop_or_failure": False,
    }
    if permissions != exact_permissions:
        raise PermissionError("authorization permissions are not the frozen narrow set")
    consumed_path = work_dir / CONSUMPTION_FILENAME
    preexisting_outputs = [
        work_dir / "EVP_validation_result.json",
        work_dir / "EVP_outer_oof_per_hand.parquet",
        work_dir / "EVP_outer_oof_top_five.parquet",
        work_dir / "EVP_nested_routes",
    ]
    if any(path.exists() for path in preexisting_outputs):
        raise PermissionError("a preregistered result artifact already exists")
    try:
        descriptor = os.open(
            consumed_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise PermissionError("the one-run authorization was already consumed") from error
    consumed = {
        "status": "RESULT_BEARING_EXECUTION_AUTHORIZATION_CONSUMED",
        "authorization_path": str(authorization_path),
        "authorization_sha256": _sha256(authorization_path),
        "draft_sha256": DRAFT_SHA256,
        "core_implementation_sha256": core_sha256,
        "runner_implementation_sha256": runner_sha256,
        "rerun_allowed": False,
        "evaluation_allowed": False,
        "submission_allowed": False,
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


def _validate_output_terminology_contract() -> dict[str, object]:
    contract = {
        "interval_name": SENSITIVITY_INTERVAL_NAME,
        "endpoint_fields": ["sensitivity_lower_05", "sensitivity_upper_95"],
        "interval_semantics": (
            "paired_pool_weight_sensitivity_not_confidence_or_coverage"
        ),
        "confidence_claimed": False,
        "credible_claimed": False,
        "significance_claimed": False,
        "posterior_probability_claimed": False,
        "coverage_claimed": False,
        "estimand_fields": [
            "query_weighted_overall",
            "family_weighted_mean",
            "unweighted_macro_family_mean",
            "paired_query_level_delta",
        ],
    }
    if contract["interval_name"] != (
        "90% paired Bayesian cluster-weight sensitivity interval"
    ):
        raise PreregistrationError("the frozen sensitivity interval name changed")
    if any(
        contract[field] is not False
        for field in (
            "confidence_claimed",
            "credible_claimed",
            "significance_claimed",
            "posterior_probability_claimed",
            "coverage_claimed",
        )
    ):
        raise PreregistrationError("an inferential claim is prohibited")
    return contract


def _scope(frame: pd.DataFrame, half: str) -> pd.DataFrame:
    if half == "full":
        return frame.copy()
    if half == "early":
        return frame[frame["phase_progress"] <= 0.5].copy()
    if half == "late":
        return frame[frame["phase_progress"] > 0.5].copy()
    raise PreregistrationError(f"unknown hand scope: {half}")


def _evidence_bearing(frame: pd.DataFrame) -> pd.DataFrame:
    bearing = frame.groupby("pair_id", sort=False)["is_evidence"].transform("any")
    return frame[bearing].copy()


def _load_and_validate_inputs(data_dir: Path, work_dir: Path) -> InputBundle:
    paths = _source_paths(data_dir, work_dir)
    observed_hashes = {name: _sha256(path) for name, path in paths.items()}
    if observed_hashes != SOURCE_SHA256:
        differences = {
            name: {"expected": SOURCE_SHA256[name], "observed": observed_hashes.get(name)}
            for name in SOURCE_SHA256
            if observed_hashes.get(name) != SOURCE_SHA256[name]
        }
        raise PreregistrationError(f"a frozen source changed: {differences}")

    labels = pd.read_csv(paths["development_labels"])
    expected_status = np.where(
        labels["label"].to_numpy(dtype=int) == 1,
        "confirmed_target",
        "confirmed_non_target",
    )
    if not np.array_equal(labels["label_status"].astype(str), expected_status):
        raise PreregistrationError("labels include a non-confirmed or mismatched status")
    positives = labels[
        (labels["label"] == 1)
        & (labels["label_status"].astype(str) == "confirmed_target")
    ][["pair_id", "behavior_family"]].copy()
    if len(positives) != 372 or positives["pair_id"].duplicated().any():
        raise PreregistrationError("confirmed-target population changed")
    if not set(positives["behavior_family"].astype(str)).issubset(FAMILIES):
        raise PreregistrationError("confirmed target has an invalid family")

    full_route = pd.read_parquet(
        paths["b001_full_oof"],
        columns=["pair_id", "table_id", "fold", "predicted_family"],
    ).rename(columns={"fold": "outer_fold"})
    time_route = pd.read_parquet(
        paths["b001_time_oof"],
        columns=[
            "pair_id",
            "fold",
            "early_to_late__predicted_family",
            "late_to_early__predicted_family",
        ],
    ).rename(columns={"fold": "outer_fold"})
    pair_meta = positives.merge(
        full_route, on="pair_id", how="left", validate="one_to_one"
    ).merge(
        time_route,
        on=["pair_id", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    route_columns = [spec.outer_route_column for spec in VIEW_SPECS]
    if pair_meta[["table_id", "outer_fold", *route_columns]].isna().any().any():
        raise PreregistrationError("a target lacks a frozen fold or B001 route")
    if set(pair_meta["outer_fold"].astype(int)) != set(range(OUTER_FOLDS)):
        raise PreregistrationError("outer folds are incomplete")
    if pair_meta.groupby("table_id")["outer_fold"].nunique().max() != 1:
        raise PreregistrationError("a table_id pool crosses outer folds")
    for column in route_columns:
        if not set(pair_meta[column].astype(str)).issubset(FAMILIES):
            raise PreregistrationError(f"invalid frozen B001 route: {column}")

    pair_hands = pd.read_parquet(
        paths["development_pair_hands"],
        columns=[
            "pair_id",
            "hand_id",
            "table_id",
            "phase_progress",
            *FEATURES,
            *H0_SIGNAL_BY_FAMILY.values(),
        ],
    )
    positive_hands = pair_hands[
        pair_hands["pair_id"].astype(str).isin(pair_meta["pair_id"].astype(str))
    ].copy()
    if len(positive_hands) != 45129:
        raise PreregistrationError("confirmed-target candidate-hand count changed")
    if positive_hands[["pair_id", "hand_id"]].duplicated().any():
        raise PreregistrationError("duplicate confirmed-target pair-hand key")
    values = positive_hands.loc[:, FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PreregistrationError("frozen evidence features are missing or non-finite")

    evidence = pd.read_csv(paths["development_evidence"])
    if len(evidence) != 1817 or evidence[["pair_id", "hand_id"]].duplicated().any():
        raise PreregistrationError("organizer evidence count or uniqueness changed")
    if evidence[["pair_id", "evidence_rank"]].duplicated().any() or not set(
        evidence["evidence_rank"].astype(int)
    ).issubset(range(1, 6)):
        raise PreregistrationError("organizer evidence_rank is invalid")
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
        raise PreregistrationError("evidence family differs from confirmed label")
    evidence_keys = pd.MultiIndex.from_frame(
        evidence[["pair_id", "hand_id"]].astype(str)
    )
    candidate_keys = pd.MultiIndex.from_frame(
        positive_hands[["pair_id", "hand_id"]].astype(str)
    )
    positive_hands["is_evidence"] = candidate_keys.isin(evidence_keys)
    if int(positive_hands["is_evidence"].sum()) != 1817:
        raise PreregistrationError("an organizer evidence hand is not a legal shared hand")

    hand_time = pd.read_parquet(paths["hands"], columns=["hand_id", "started_at"])
    if hand_time["hand_id"].duplicated().any():
        raise PreregistrationError("hands source has duplicate hand_id")
    positive_hands = positive_hands.merge(
        hand_time, on="hand_id", how="left", validate="many_to_one"
    ).merge(pair_meta, on="pair_id", how="left", validate="many_to_one", suffixes=("_hand", "_pair"))
    if positive_hands["started_at"].isna().any():
        raise PreregistrationError("a legal shared hand lacks started_at")
    if not np.array_equal(
        positive_hands["table_id_hand"].astype(str),
        positive_hands["table_id_pair"].astype(str),
    ):
        raise PreregistrationError("pair-hand table_id differs from B001 pool")
    positive_hands = positive_hands.rename(columns={"table_id_pair": "table_id"}).drop(
        columns=["table_id_hand"]
    )

    full_features = pd.read_parquet(paths["b001_full_features"])
    early_features = pd.read_parquet(paths["b001_early_features"])
    late_features = pd.read_parquet(paths["b001_late_features"])
    for frame, expected in (
        (full_features, B001_FULL_FEATURES),
        (early_features, B001_TIME_FEATURES),
        (late_features, B001_TIME_FEATURES),
    ):
        if any(column not in frame for column in ("pair_id", "table_id", *expected)):
            raise PreregistrationError("a frozen B001 pair-feature schema changed")

    bundle = InputBundle(
        pair_meta=pair_meta.rename(columns={"outer_fold": "outer_fold"}),
        positive_hands=positive_hands,
        full_pair_features=full_features,
        early_pair_features=early_features,
        late_pair_features=late_features,
        source_hashes=observed_hashes,
    )
    _validate_coverage(bundle)
    _validate_h0_tie_structure(bundle)
    return bundle


def _validate_coverage(bundle: InputBundle) -> None:
    hands = bundle.positive_hands
    stats = hands.groupby(
        ["pair_id", "behavior_family", "outer_fold", "table_id"], sort=False
    ).agg(
        candidate_hands=("hand_id", "size"),
        evidence_hands=("is_evidence", "sum"),
        early_candidate_hands=("phase_progress", lambda value: int((value <= 0.5).sum())),
        early_evidence=("is_evidence", lambda value: 0),
    ).reset_index()
    early_evidence = (
        hands[hands["phase_progress"] <= 0.5]
        .groupby("pair_id")["is_evidence"]
        .sum()
    )
    stats["early_evidence"] = stats["pair_id"].map(early_evidence).fillna(0).astype(int)
    stats["late_candidate_hands"] = stats["candidate_hands"] - stats["early_candidate_hands"]
    stats["late_evidence"] = stats["evidence_hands"] - stats["early_evidence"]
    if len(stats) != 372 or not stats["evidence_hands"].between(3, 5).all():
        raise PreregistrationError("confirmed-target query coverage changed")

    for family in FAMILIES:
        part = stats[stats["behavior_family"].astype(str) == family]
        observed = (
            int(len(part)),
            int(part["candidate_hands"].sum()),
            int(part["evidence_hands"].sum()),
        )
        if observed != EXPECTED_FAMILY_TOTALS[family]:
            raise PreregistrationError(f"full family totals changed: {family}")
        valid_counts = tuple(
            int((part["outer_fold"] == fold).sum()) for fold in range(OUTER_FOLDS)
        )
        evidence_counts = tuple(
            int(part.loc[part["outer_fold"] == fold, "evidence_hands"].sum())
            for fold in range(OUTER_FOLDS)
        )
        if valid_counts != EXPECTED_FULL_VALID_PAIRS[family] or evidence_counts != EXPECTED_FULL_EVIDENCE[family]:
            raise PreregistrationError(f"full family/fold coverage changed: {family}")

    half_columns = {
        "early": ("early_candidate_hands", "early_evidence"),
        "late": ("late_candidate_hands", "late_evidence"),
    }
    for direction, target_half in (("early_to_late", "late"), ("late_to_early", "early")):
        candidate_column, evidence_column = half_columns[target_half]
        for family in FAMILIES:
            part = stats[stats["behavior_family"].astype(str) == family]
            observed_pairs = tuple(
                int(
                    (
                        (part["outer_fold"] == fold)
                        & (part[evidence_column] > 0)
                    ).sum()
                )
                for fold in range(OUTER_FOLDS)
            )
            observed_evidence = tuple(
                int(
                    part.loc[
                        (part["outer_fold"] == fold)
                        & (part[evidence_column] > 0),
                        evidence_column,
                    ].sum()
                )
                for fold in range(OUTER_FOLDS)
            )
            if observed_pairs != EXPECTED_TIME_TARGET_PAIRS[direction][family] or observed_evidence != EXPECTED_TIME_TARGET_EVIDENCE[direction][family]:
                raise PreregistrationError(f"time target coverage changed: {direction}/{family}")
            if candidate_column not in part:
                raise RuntimeError("time candidate column disappeared")

    for half in ("full", "early", "late"):
        if half == "full":
            candidates = stats["candidate_hands"]
            evidence = stats["evidence_hands"]
        else:
            candidates = stats[f"{half}_candidate_hands"]
            evidence = stats[f"{half}_evidence"]
        eligible = evidence > 0
        unjudged = candidates - evidence
        preferences = evidence * unjudged
        observed_readiness = {
            "evidence_bearing_pairs": int(eligible.sum()),
            "pairwise_trainable_pairs": int((eligible & (unjudged > 0)).sum()),
            "candidate_minimum": int(candidates[eligible].min()),
            "unjudged_minimum": int(unjudged[eligible].min()),
            "preference_pairs": int(preferences[eligible].sum()),
            "symmetric_rows": int(2 * preferences[eligible].sum()),
        }
        if observed_readiness != EXPECTED_PAIRWISE_READINESS[half]:
            raise PreregistrationError(f"pairwise readiness changed: {half}")
        zero = stats[eligible & (unjudged == 0)]
        if half == "early":
            if len(zero) != 1 or str(zero.iloc[0]["behavior_family"]) != "directed_transfer" or int(zero.iloc[0]["outer_fold"]) != 3:
                raise PreregistrationError("the frozen early zero-preference query changed")
        elif len(zero) != 0:
            raise PreregistrationError(f"unexpected zero-preference query in {half}")


def _h0_tie_counts(frame: pd.DataFrame, route_column: str) -> tuple[int, ...]:
    routes = frame[route_column].astype(str).to_numpy()
    scores = heuristic_scores(frame, routes)
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
        (
            (bounds["min"] <= 5)
            & (bounds["max"] >= 5)
            & (bounds["min"] != bounds["max"])
        ).sum()
    )
    return (
        int(work["pair_id"].nunique()),
        int(len(ties)),
        int(len(tied_rows)),
        crossings,
        duplicate_timestamps,
    )


def _validate_h0_tie_structure(bundle: InputBundle) -> None:
    hands = bundle.positive_hands
    scopes = {
        "full": (hands, "predicted_family"),
        "early_to_late": (
            hands[hands["phase_progress"] > 0.5],
            "early_to_late__predicted_family",
        ),
        "late_to_early": (
            hands[hands["phase_progress"] <= 0.5],
            "late_to_early__predicted_family",
        ),
    }
    for view, (frame, route_column) in scopes.items():
        observed = _h0_tie_counts(frame, route_column)
        if observed != EXPECTED_H0_TIE_STRUCTURE[view]:
            raise PreregistrationError(f"H0 tie structure changed: {view}")


def _pair_features_for_view(
    bundle: InputBundle,
    spec: ViewSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.name == "full":
        return bundle.full_pair_features, bundle.full_pair_features
    if spec.name == "early_to_late":
        return bundle.early_pair_features, bundle.late_pair_features
    return bundle.late_pair_features, bundle.early_pair_features


def _route_for_hands(
    hands: pd.DataFrame,
    routes: pd.DataFrame,
    route_column: str = "predicted_family",
) -> np.ndarray:
    if routes["pair_id"].astype(str).duplicated().any():
        raise PreregistrationError("route receipt has duplicate pairs")
    mapping = routes.set_index(routes["pair_id"].astype(str))[route_column]
    result = hands["pair_id"].astype(str).map(mapping)
    if result.isna().any() or not set(result.astype(str)).issubset(FAMILIES):
        raise PreregistrationError("a scored hand lacks a predicted-family route")
    return result.astype(str).to_numpy()


def _score_ranker(
    model,
    transform: FrozenHandTransform,
    hands: pd.DataFrame,
    routes: Sequence[object] | None,
) -> np.ndarray:
    base = transform.transform(hands)
    matrix = base if routes is None else interaction_matrix(base, routes)
    result = model.decision_function(matrix)
    if not np.isfinite(result).all():
        raise PreregistrationError("learned evidence score is non-finite")
    return np.asarray(result, dtype=np.float64)


def _persist_nested_routes(
    bundle: InputBundle,
    spec: ViewSpec,
    outer_fold: int,
    route_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    assignments = assign_inner_folds(bundle.pair_meta, outer_fold)
    source_features, target_features = _pair_features_for_view(bundle, spec)
    routes = fit_nested_behavior_routes(
        assignments,
        source_features,
        target_features,
        view=spec.name,
        outer_fold=outer_fold,
    )
    route_dir.mkdir(parents=True, exist_ok=True)
    path = route_dir / f"{spec.name}__outer_{outer_fold}.parquet"
    _atomic_parquet(path, routes)
    receipt = {
        "view": spec.name,
        "outer_fold": int(outer_fold),
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(routes)),
        "inner_folds": int(routes["inner_fold"].nunique()),
        "true_family_column_persisted": "behavior_family" in routes,
        "fit_evidence_model_before_receipt_hash": False,
    }
    return routes, receipt


def _select_c(
    bundle: InputBundle,
    spec: ViewSpec,
    assignments: pd.DataFrame,
    inner_routes: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    source = _evidence_bearing(_scope(bundle.positive_hands, spec.source_half))
    target = _evidence_bearing(_scope(bundle.positive_hands, spec.target_half))
    scores_by_c: dict[float, list[pd.DataFrame]] = {value: [] for value in C_GRID}

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
            raise PreregistrationError("a nested evidence train/validation cell is empty")
        if set(train_hands["table_id"].astype(str)) & set(
            valid_hands["table_id"].astype(str)
        ):
            raise PreregistrationError("nested evidence train/validation pools overlap")
        transform = FrozenHandTransform.fit(train_hands)
        train_base = transform.transform(train_hands)
        train_matrix = interaction_matrix(
            train_base, train_hands["behavior_family"].astype(str)
        )
        preferences = build_pairwise_preferences(
            train_matrix,
            train_hands["pair_id"],
            train_hands["is_evidence"],
        )
        valid_routes = _route_for_hands(valid_hands, inner_routes)
        valid_matrix = interaction_matrix(
            transform.transform(valid_hands), valid_routes
        )
        for c_value in C_GRID:
            model = fit_evidence_ranker(preferences, c_value)
            score = model.decision_function(valid_matrix)
            if not np.isfinite(score).all():
                raise PreregistrationError("inner evidence score is non-finite")
            scored = valid_hands.copy()
            scored["l1_score"] = score
            scores_by_c[c_value].append(query_ap_frame(scored, "l1_score"))

    macro_scores: dict[float, float] = {}
    for c_value in C_GRID:
        pooled = pd.concat(scores_by_c[c_value], ignore_index=True)
        expected_pairs = set(assignments["pair_id"].astype(str))
        observed_pairs = set(pooled["pair_id"].astype(str))
        target_scope = _scope(bundle.positive_hands, spec.target_half)
        evidence_bearing = set(
            target_scope.loc[target_scope["is_evidence"], "pair_id"].astype(str)
        )
        if observed_pairs != expected_pairs & evidence_bearing:
            raise PreregistrationError("nested C-selection query coverage changed")
        macro_scores[c_value] = macro_family_mean(pooled, "query_ap5")
    return choose_c_exact_tie_smaller(macro_scores), macro_scores


def _fit_outer_and_score(
    bundle: InputBundle,
    spec: ViewSpec,
    outer_fold: int,
    selected_c: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_all = _scope(bundle.positive_hands, spec.source_half)
    target_all = _scope(bundle.positive_hands, spec.target_half)
    outer_train_ids = set(
        bundle.pair_meta.loc[
            bundle.pair_meta["outer_fold"] != outer_fold, "pair_id"
        ].astype(str)
    )
    outer_valid_ids = set(
        bundle.pair_meta.loc[
            bundle.pair_meta["outer_fold"] == outer_fold, "pair_id"
        ].astype(str)
    )
    source_train = _evidence_bearing(
        source_all[source_all["pair_id"].astype(str).isin(outer_train_ids)]
    )
    target_fold = target_all[
        target_all["pair_id"].astype(str).isin(outer_valid_ids)
    ].copy()
    eligible = target_fold.groupby("pair_id", sort=False)["is_evidence"].transform(
        "any"
    )
    target_valid = target_fold[eligible].copy()
    if source_train.empty or target_valid.empty:
        raise PreregistrationError("an outer evidence train/validation cell is empty")
    if set(source_train["table_id"].astype(str)) & set(
        target_fold["table_id"].astype(str)
    ):
        raise PreregistrationError("outer evidence train/validation pools overlap")

    transform = FrozenHandTransform.fit(source_train)
    source_base = transform.transform(source_train)
    l1_preferences = build_pairwise_preferences(
        interaction_matrix(
            source_base, source_train["behavior_family"].astype(str)
        ),
        source_train["pair_id"],
        source_train["is_evidence"],
    )
    l0_preferences = build_pairwise_preferences(
        source_base,
        source_train["pair_id"],
        source_train["is_evidence"],
    )
    l1_model = fit_evidence_ranker(l1_preferences, selected_c)
    l0_model = fit_evidence_ranker(l0_preferences, selected_c)

    route_table = bundle.pair_meta[
        ["pair_id", spec.outer_route_column]
    ].rename(columns={spec.outer_route_column: "predicted_family"})
    target_routes = _route_for_hands(target_fold, route_table)
    valid_routes = target_routes[eligible.to_numpy()]
    h0_score = heuristic_scores(target_valid, valid_routes)
    l0_score = _score_ranker(l0_model, transform, target_valid, None)
    l1_score = _score_ranker(l1_model, transform, target_valid, valid_routes)

    receipt = target_fold[
        [
            "pair_id",
            "hand_id",
            "started_at",
            "table_id",
            "outer_fold",
            "behavior_family",
            "phase_progress",
            "is_evidence",
        ]
    ].copy()
    receipt["view"] = spec.name
    receipt["eligible_query"] = eligible.to_numpy(dtype=bool)
    receipt["predicted_route"] = target_routes
    receipt["h0_score"] = np.nan
    receipt["l0_score"] = np.nan
    receipt["l1_score"] = np.nan
    receipt.loc[eligible, "h0_score"] = h0_score
    receipt.loc[eligible, "l0_score"] = l0_score
    receipt.loc[eligible, "l1_score"] = l1_score
    receipt["time_half"] = np.where(
        receipt["phase_progress"] <= 0.5, "early", "late"
    )
    diagnostics = {
        "view": spec.name,
        "outer_fold": int(outer_fold),
        "selected_c": float(selected_c),
        "training_queries_evidence_bearing": int(
            source_train["pair_id"].nunique()
        ),
        "training_queries_pairwise_trainable": int(
            (l1_preferences.query_receipt["preference_pairs"] > 0).sum()
        ),
        "training_preference_pairs": int(
            l1_preferences.query_receipt["preference_pairs"].sum()
        ),
        "validation_queries_eligible": int(target_valid["pair_id"].nunique()),
        "validation_queries_ineligible": int(
            target_fold.loc[~eligible, "pair_id"].nunique()
        ),
        "transform": transform.receipt(),
        "matched_l0_l1_c": True,
        "unknown_pairs_used": 0,
        "confirmed_non_target_pairs_used": 0,
    }
    return receipt, diagnostics


def _point_summary(query_scores: pd.DataFrame) -> dict[str, object]:
    family = {
        name: float(
            query_scores.loc[
                query_scores["behavior_family"].astype(str) == name,
                "query_ap5",
            ].mean()
        )
        for name in FAMILIES
    }
    return {
        "query_weighted_overall_map5": float(query_scores["query_ap5"].mean()),
        "family_map5": family,
        "unweighted_macro_family_map5": float(np.mean(list(family.values()))),
        "outer_fold_map5": {
            str(fold): float(
                query_scores.loc[
                    query_scores["outer_fold"] == fold, "query_ap5"
                ].mean()
            )
            for fold in range(OUTER_FOLDS)
        },
        "outer_fold_family_map5": {
            str(fold): {
                family_name: float(
                    query_scores.loc[
                        (query_scores["outer_fold"] == fold)
                        & (
                            query_scores["behavior_family"].astype(str)
                            == family_name
                        ),
                        "query_ap5",
                    ].mean()
                )
                for family_name in FAMILIES
            }
            for fold in range(OUTER_FOLDS)
        },
        "eligible_queries": int(len(query_scores)),
        "candidate_hands": int(query_scores["candidate_hands"].sum()),
        "evidence_hands": int(query_scores["evidence_hands"].sum()),
    }


def _ordered_for_plan(
    frame: pd.DataFrame,
    plan: BayesianClusterWeightPlan,
) -> pd.DataFrame:
    indexed = frame.assign(pair_id=frame["pair_id"].astype(str)).set_index("pair_id")
    if set(indexed.index) != set(plan.query_keys):
        raise PreregistrationError("metric queries differ from the weight-plan universe")
    return indexed.loc[list(plan.query_keys)].reset_index()


def _gate_results(results: Mapping[str, object]) -> dict[str, object]:
    full = results["full"]
    primary = full["sensitivity"]["l1_minus_h0"]
    primary_family = primary["family_weighted_mean"]
    primary_folds = primary["outer_fold_point_delta"]
    primary_checks = {
        "overall_delta_at_least_0_040": primary["query_weighted_overall"][
            "point_delta"
        ]
        >= 0.040,
        "lower_sensitivity_bound_above_zero": primary[
            "query_weighted_overall"
        ]["sensitivity_lower_05"]
        > 0.0,
        "all_family_deltas_nonnegative": all(
            primary_family[family]["point_delta"] >= 0.0 for family in FAMILIES
        ),
        "at_least_four_outer_folds_positive": sum(
            value > 0.0 for value in primary_folds.values()
        )
        >= 4,
    }
    attribution = full["sensitivity"]["l1_minus_l0"]
    attribution_checks = {
        "macro_delta_at_least_0_005": attribution[
            "unweighted_macro_family_mean"
        ]["point_delta"]
        >= 0.005,
        "coordinated_isolation_delta_nonnegative": attribution[
            "family_weighted_mean"
        ]["coordinated_isolation"]["point_delta"]
        >= 0.0,
    }
    time_checks: dict[str, object] = {}
    for view in ("early_to_late", "late_to_early"):
        sensitivity = results[view]["sensitivity"]["l1_minus_h0"]
        checks = {
            "macro_delta_at_least_0_020": sensitivity[
                "unweighted_macro_family_mean"
            ]["point_delta"]
            >= 0.020,
            "overall_delta_positive": sensitivity["query_weighted_overall"][
                "point_delta"
            ]
            > 0.0,
            "lower_sensitivity_bound_above_zero": sensitivity[
                "query_weighted_overall"
            ]["sensitivity_lower_05"]
            > 0.0,
            "all_direction_family_deltas_nonnegative": all(
                sensitivity["family_weighted_mean"][family]["point_delta"] >= 0.0
                for family in FAMILIES
            ),
            "at_least_four_outer_folds_positive": sum(
                value > 0.0
                for value in sensitivity["outer_fold_point_delta"].values()
            )
            >= 4,
        }
        time_checks[view] = {
            "checks": checks,
            "passed": all(checks.values()),
        }

    ordered = [
        ("primary_full_period", primary_checks),
        ("family_conditioning_attribution", attribution_checks),
        (
            "bidirectional_time",
            {
                view: bool(time_checks[view]["passed"])
                for view in ("early_to_late", "late_to_early")
            },
        ),
    ]
    first_failure = None
    for name, checks in ordered:
        if not all(checks.values()):
            first_failure = name
            break
    return {
        "integrity_and_coverage": "passed_before_model_fit",
        "primary_full_period": {
            "checks": primary_checks,
            "passed": all(primary_checks.values()),
        },
        "family_conditioning_attribution": {
            "checks": attribution_checks,
            "passed": all(attribution_checks.values()),
        },
        "bidirectional_time": time_checks,
        "all_gates_passed": first_failure is None,
        "first_failed_gate": first_failure,
        "rescue_or_rerun_allowed": False,
    }


def run_authorized_validation(
    data_dir: str | Path,
    work_dir: str | Path,
    authorization_path: str | Path,
) -> dict[str, object]:
    """Consume one later authorization and run the single preregistered study."""

    data_root = Path(data_dir)
    output_root = Path(work_dir)
    if not output_root.is_dir():
        raise PermissionError("the pre-existing audited work directory is required")
    authorization = _validate_result_execution_authorization(
        Path(authorization_path), output_root
    )
    terminology_contract = _validate_output_terminology_contract()
    bundle = _load_and_validate_inputs(data_root, output_root)

    plans: dict[str, BayesianClusterWeightPlan] = {}
    for spec in VIEW_SPECS:
        target = _evidence_bearing(_scope(bundle.positive_hands, spec.target_half))
        metadata_rows = target[
            ["pair_id", "table_id", "outer_fold", "behavior_family"]
        ]
        if (
            metadata_rows.groupby("pair_id", sort=False)[
                ["table_id", "outer_fold", "behavior_family"]
            ]
            .nunique(dropna=False)
            .to_numpy()
            .max()
            != 1
        ):
            raise PreregistrationError("eligible query metadata is not one-to-one")
        meta = metadata_rows.drop_duplicates("pair_id", keep="first")
        plans[spec.name] = BayesianClusterWeightPlan.create(
            meta.reset_index(drop=True), spec.name
        )
    weight_hash = frozen_pool_weight_stream_sha256(plans)
    no_header_hash = frozen_pool_weight_stream_sha256(
        plans, include_headers=False
    )
    if weight_hash != WEIGHT_STREAM_SHA256:
        raise PreregistrationError("frozen weight-stream hash did not reproduce")
    if no_header_hash != WEIGHT_STREAM_NO_HEADER_SHA256:
        raise PreregistrationError("no-header weight-stream negative control changed")

    route_dir = output_root / "EVP_nested_routes"
    per_hand_parts: list[pd.DataFrame] = []
    fit_receipts: list[dict[str, object]] = []
    route_receipts: list[dict[str, object]] = []
    selection_receipts: list[dict[str, object]] = []
    for spec in VIEW_SPECS:
        for outer_fold in range(OUTER_FOLDS):
            routes, route_receipt = _persist_nested_routes(
                bundle, spec, outer_fold, route_dir
            )
            route_receipts.append(route_receipt)
            assignments = assign_inner_folds(bundle.pair_meta, outer_fold)
            selected_c, macro_by_c = _select_c(
                bundle, spec, assignments, routes
            )
            selection_receipts.append(
                {
                    "view": spec.name,
                    "outer_fold": int(outer_fold),
                    "selected_c": selected_c,
                    "inner_macro_family_random_tie_map5_by_c": {
                        str(key): value for key, value in macro_by_c.items()
                    },
                    "exact_tie_chooses_smaller_c": True,
                }
            )
            fold_scores, fit_receipt = _fit_outer_and_score(
                bundle, spec, outer_fold, selected_c
            )
            per_hand_parts.append(fold_scores)
            fit_receipts.append(fit_receipt)

    per_hand = pd.concat(per_hand_parts, ignore_index=True)
    required_receipt_columns = [
        "view",
        "pair_id",
        "hand_id",
        "started_at",
        "table_id",
        "outer_fold",
        "eligible_query",
        "predicted_route",
        "h0_score",
        "l0_score",
        "l1_score",
        "is_evidence",
        "time_half",
        "behavior_family",
    ]
    per_hand = per_hand.loc[:, required_receipt_columns]
    per_hand_path = output_root / "EVP_outer_oof_per_hand.parquet"
    _atomic_parquet(per_hand_path, per_hand)
    per_hand_sha256 = _sha256(per_hand_path)

    top_parts: list[pd.DataFrame] = []
    for view in (spec.name for spec in VIEW_SPECS):
        eligible = per_hand[
            (per_hand["view"] == view) & per_hand["eligible_query"]
        ]
        for method in ("h0", "l0", "l1"):
            score_column = f"{method}_score"
            top = deterministic_top_five(eligible, score_column)
            top["view"] = view
            top["method"] = method
            relevance = eligible[["pair_id", "hand_id", "is_evidence"]]
            top = top.merge(
                relevance,
                on=["pair_id", "hand_id"],
                how="left",
                validate="one_to_one",
            )
            top_parts.append(top)
    top_five = pd.concat(top_parts, ignore_index=True)
    top_five_path = output_root / "EVP_outer_oof_top_five.parquet"
    _atomic_parquet(top_five_path, top_five)
    top_five_sha256 = _sha256(top_five_path)

    # Aggregate performance is first computed only after both score receipts
    # and all deterministic top-five materializations have immutable hashes.
    results: dict[str, object] = {}
    for spec in VIEW_SPECS:
        eligible = per_hand[
            (per_hand["view"] == spec.name) & per_hand["eligible_query"]
        ].copy()
        query_scores = {
            method: query_ap_frame(eligible, f"{method}_score")
            for method in ("h0", "l0", "l1")
        }
        contrasts = {}
        for name, left, right in (
            ("l1_minus_h0", "l1", "h0"),
            ("l1_minus_l0", "l1", "l0"),
        ):
            delta = paired_delta_frame(query_scores[left], query_scores[right])
            delta = _ordered_for_plan(delta, plans[spec.name])
            contrasts[name] = plans[spec.name].summarize(delta)
        results[spec.name] = {
            "methods": {
                method: _point_summary(query_scores[method])
                for method in ("h0", "l0", "l1")
            },
            "sensitivity": contrasts,
        }

    gates = _gate_results(results)
    report = {
        "study": "EVP-DRAFT-03",
        "status": "PASS_ALL_DEVELOPMENT_GATES"
        if gates["all_gates_passed"]
        else "STOP_FAILED_FROZEN_DEVELOPMENT_GATE",
        "authorization_sha256": _sha256(Path(authorization_path)),
        "authorization_status": authorization["status"],
        "binding": {
            "draft_sha256": DRAFT_SHA256,
            "prereg_static_audit_sha256": PREREG_STATIC_AUDIT_SHA256,
            "source_sha256": bundle.source_hashes,
            "weight_stream_sha256": weight_hash,
            "weight_stream_no_header_negative_control_sha256": no_header_hash,
            "per_hand_oof_receipt": str(per_hand_path),
            "per_hand_oof_receipt_sha256": per_hand_sha256,
            "top_five_receipt": str(top_five_path),
            "top_five_receipt_sha256": top_five_sha256,
            "nested_route_receipts": route_receipts,
        },
        "selection": selection_receipts,
        "fits": fit_receipts,
        "results": results,
        "gates": gates,
        "interval_name": SENSITIVITY_INTERVAL_NAME,
        "interval_semantics": "paired_pool_weight_sensitivity_not_confidence_or_coverage",
        "terminology_contract": terminology_contract,
        "controls": {
            "model_fit_performed": True,
            "aggregate_performance_viewed": True,
            "unknown_ground_truth_labels_assigned": 0,
            "unknown_pairs_used": 0,
            "confirmed_non_target_pairs_used": 0,
            "evaluation_rows_loaded": False,
            "evaluation_scored": False,
            "experiment_number_assigned": False,
            "submission_created_or_modified": False,
            "rerun_allowed": False,
            "candidate_promotion_authorized": False,
        },
    }
    report_path = output_root / "EVP_validation_result.json"
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", default="poker/work")
    parser.add_argument("--authorization", required=True)
    arguments = parser.parse_args()
    report = run_authorized_validation(
        arguments.data_dir,
        arguments.work_dir,
        arguments.authorization,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
