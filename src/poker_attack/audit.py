"""Input-table and submission guardrails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .schema import (
    ALLOWED_BEHAVIORS,
    EVIDENCE_COLUMNS,
    NO_EVIDENCE,
    REQUIRED_COLUMNS,
    SUBMISSION_COLUMNS,
    input_paths,
)


class DataAuditError(ValueError):
    """Raised when competition inputs or a submission violate hard constraints."""


def _columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_csv(path, nrows=0).columns)


def _row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    return int(sum(len(chunk) for chunk in pd.read_csv(path, chunksize=250_000)))


def audit_dataset(data_dir: str | Path) -> dict[str, Any]:
    """Validate the official eight-file release without loading large parquet tables."""

    root = Path(data_dir)
    paths = input_paths(root)
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise DataAuditError("Missing competition files: " + ", ".join(missing_files))

    report: dict[str, Any] = {"data_dir": str(root.resolve()), "tables": {}}
    for name, path in paths.items():
        columns = _columns(path)
        missing = sorted(REQUIRED_COLUMNS[name] - set(columns))
        if missing:
            raise DataAuditError(f"{path.name} is missing columns: {missing}")
        report["tables"][name] = {
            "path": str(path),
            "rows": _row_count(path),
            "columns": columns,
        }

    evaluation = pd.read_csv(paths["evaluation_pairs"], usecols=["pair_id"])
    sample = pd.read_csv(paths["sample_submission"])
    if evaluation["pair_id"].duplicated().any():
        raise DataAuditError("evaluation_pairs.csv contains duplicate pair_id values")
    validate_submission(sample, evaluation["pair_id"], permit_placeholder_scores=True)
    report["pair_coverage_ok"] = True
    return report


def validate_submission(
    submission: pd.DataFrame,
    expected_pair_ids: pd.Series,
    *,
    permit_placeholder_scores: bool = False,
) -> None:
    """Apply the public metric's schema rules before a costly Kaggle submission."""

    missing = sorted(set(SUBMISSION_COLUMNS) - set(submission.columns))
    if missing:
        raise DataAuditError(f"submission is missing columns: {missing}")
    if submission["pair_id"].duplicated().any():
        raise DataAuditError("submission pair_id values must be unique")

    expected = set(expected_pair_ids.astype(str))
    received = set(submission["pair_id"].astype(str))
    if expected != received:
        raise DataAuditError(
            f"pair_id coverage mismatch: {len(expected - received)} missing and "
            f"{len(received - expected)} extra"
        )

    risk = pd.to_numeric(submission["risk_score"], errors="coerce")
    if risk.isna().any() or (not permit_placeholder_scores and not risk.between(0, 1).all()):
        raise DataAuditError("risk_score must be numeric and between 0 and 1")
    if permit_placeholder_scores and not risk.dropna().between(0, 1).all():
        raise DataAuditError("sample risk_score values must be between 0 and 1")

    behavior = submission["predicted_behavior"].astype(str)
    invalid = sorted(set(behavior) - ALLOWED_BEHAVIORS)
    if invalid:
        raise DataAuditError(f"invalid predicted_behavior values: {invalid}")

    if submission.loc[:, EVIDENCE_COLUMNS].isna().any(axis=None):
        raise DataAuditError("submission evidence cells must not be empty")
    for row in submission.loc[:, EVIDENCE_COLUMNS].itertuples(index=False, name=None):
        hands = [str(value).strip() for value in row if str(value).strip() != NO_EVIDENCE]
        if len(hands) != len(set(hands)):
            raise DataAuditError("evidence hand IDs must not repeat within a pair")


def validate_evidence_membership(
    submission: pd.DataFrame,
    pair_hands: pd.DataFrame,
) -> dict[str, int]:
    """Verify every submitted hand belongs to that evaluation pair.

    Hashes make the 9M-row candidate scan inexpensive; all hash matches are
    subsequently checked by an exact merge, so hash collisions cannot make an
    invalid hand pass this guardrail.
    """

    long = submission[["pair_id", *EVIDENCE_COLUMNS]].melt(
        id_vars="pair_id", value_name="hand_id"
    )
    long["pair_id"] = long["pair_id"].astype(str)
    long["hand_id"] = long["hand_id"].astype(str).str.strip()
    long = long[long["hand_id"] != NO_EVIDENCE][["pair_id", "hand_id"]]
    if long.duplicated().any():
        raise DataAuditError("submitted pair-hand evidence keys must be unique")

    candidates = pair_hands[["pair_id", "hand_id"]].astype(str)
    required_hash = pd.util.hash_pandas_object(long, index=False).to_numpy()
    candidate_hash = pd.util.hash_pandas_object(candidates, index=False).to_numpy()
    candidate_matches = candidates.loc[np.isin(candidate_hash, required_hash)].drop_duplicates()
    exact = long.merge(
        candidate_matches,
        on=["pair_id", "hand_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    invalid = exact[exact["_merge"] != "both"]
    if not invalid.empty:
        example = invalid.iloc[0]
        raise DataAuditError(
            "submitted evidence is not a shared evaluation hand: "
            f"pair_id={example['pair_id']} hand_id={example['hand_id']}"
        )
    return {
        "submitted_evidence_hands": int(len(long)),
        "legal_evidence_hands": int(len(exact)),
        "pairs_with_five_evidence_hands": int(
            long.groupby("pair_id", sort=False).size().eq(5).sum()
        ),
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
