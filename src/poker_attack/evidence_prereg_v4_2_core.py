"""Standalone frozen mechanics for EVP-DRAFT-04.2.

This module is intentionally independent of the rejected v3 implementation.
Importing it performs no file I/O, fits no model, and computes no competition
result.  Real-data orchestration, if separately authorized in the future, is
owned by :mod:`poker_attack.evidence_prereg_v4_2`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .schema import TARGET_BEHAVIORS


class V4InvariantError(ValueError):
    """Raised whenever a frozen v4 invariant is violated."""


V4_DRAFT_SHA256 = "3a9f20936f4173f72cb4ff867d296eb2945b43ca40cac0ee587b7fb4b83da22c"
V4_COVERAGE_CONTRACT_SHA256 = (
    "4e821092161be0b0c80892af41ed61718e0a4930562b748a826da8a8bf9001e6"
)
V4_METHOD_AUDIT_SHA256 = (
    "7ab04d40cff489323ee85f27484b93d57e6efdce06b09a78eeccc9cd0d22e913"
)
V4_1_DELTA_SHA256 = (
    "285a2b85beb03abf4419c4e3816bd3c0f0af68ebb64eec4c01a1fed9a3db4279"
)
V4_2_DELTA_SHA256 = (
    "55ac6be4f0a98e1d9f4a00733d32a830803dfdd639c18fc9a46124792aa1a812"
)
V4_2_RESULT_ROOT_BINDING_SHA256 = (
    "b1ae8de05864b87a9f1d3c8b089495f01de41a9d090aef803498e094f7a51910"
)
HISTORICAL_CONTAINMENT_SHA256 = (
    "f04b10963b9e466cda2cb819a668344464c7efd7ead98ebb772b6879b1e5f75c"
)
REJECTED_V3_CORE_SHA256 = (
    "5871093ce9df04c7370db9c462fc85f77d15b8ebfb0e0dc65e65d7808c54005e"
)
REJECTED_V3_RUNNER_SHA256 = (
    "5a0f5aec85d2bad1ea852c4782200316a57704aa28edc8a99e7fb27b2c7a5fad"
)

FAMILIES = tuple(TARGET_BEHAVIORS)
if FAMILIES != (
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
):
    raise RuntimeError("TARGET_BEHAVIORS no longer matches EVP-DRAFT-04")

FEATURES = (
    "pair_contribution_bb",
    "contribution_gap_bb",
    "net_gap_bb",
    "max_win_bb",
    "max_loss_bb",
    "transfer_1_to_2_bb",
    "transfer_2_to_1_bb",
    "transfer_any_bb",
    "both_showdown",
    "one_folded",
    "pair_aggression",
    "pair_passive",
    "expected_aggression",
    "expected_passive",
    "aggression_residual",
    "passive_residual",
    "pair_pot_share",
    "pair_raises",
    "pair_folds",
    "max_hole_strength",
)

H0_SIGNAL_BY_FAMILY = {
    "directed_transfer": "directed_signal",
    "soft_play": "soft_signal",
    "coordinated_isolation": "isolation_signal",
}

C_GRID = (0.01, 0.1, 1.0, 10.0)
OUTER_FOLDS = 5
INNER_FOLDS = 3
INNER_SPLIT_BASE_SEED = 8803
ROUTER_SEED_BASE = {
    "full": 18401,
    "early_to_late": 19401,
    "late_to_early": 20401,
}
VIEW_INDEX = {"full": 0, "early_to_late": 1, "late_to_early": 2}
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_BASE_SEED = 12673
WEIGHT_STREAM_NUMPY_VERSION = "2.4.6"
WEIGHT_STREAM_BIT_GENERATOR = "PCG64"
FROZEN_SKLEARN_VERSION = "1.8.0"
FROZEN_RUNTIME_VERSIONS = {
    "python": "3.13.2",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "pyarrow": "24.0.0",
}
FROZEN_SCHEMA_SHA256 = (
    "dee6eeac75fa1cd202b1656d22c394049c9913993d6efba4c493c651a3fc3ee4"
)
SENSITIVITY_INTERVAL_NAME = (
    "90% paired Bayesian cluster-weight sensitivity interval"
)

STAGE_WEIGHT_STREAM_SHA256 = {
    ("full",): "a00923786962622bacfd63fb211670ef14eb657a4a96e5411274ff4034777a2e",
    ("early_to_late",): "4b009fee27edc3e662fe9cda00bcc9ca5ad84e1d16c7b4d253ce2f9d1dd39471",
    ("late_to_early",): "938473e74b7609cccea365606f1f468ecd7958782704229ba8f40cd31f057cf0",
    ("early_to_late", "late_to_early"): (
        "380969a4fa506e7cd9734be8ba5b0e55fc9d3a7ebd82864762e43ec36a7468fe"
    ),
    ("full", "early_to_late", "late_to_early"): (
        "c833ac7d6dd02d2009f41f3566667dcf37e59071646cc1afe9929395b601975c"
    ),
}
STAGE_WEIGHT_STREAM_NO_HEADER_SHA256 = {
    ("full",): "e57b0a7b15873641e429a964d79075a0f56e997b675caab8e9a2a7f40ab1f16d",
    ("early_to_late",): "5afbd956bc156cd2b35f0270ba5637f8fbd4b6a973b6c2c4465b721987ae856f",
    ("late_to_early",): "ee671cf04e877e747254b61c669013fc41957fb91dd08e26dd5c67f3498a68fc",
    ("early_to_late", "late_to_early"): (
        "4bad16bff34f63c0caa336b128ce2994e70b49c0ec85bd582f3b44ab213ee3b2"
    ),
    ("full", "early_to_late", "late_to_early"): (
        "08ee7564ab1c5aa7736c1fc541a006f6c5181b2638088a6b91d24e034a222e1a"
    ),
}

_PAIR_AGG_FIELDS = (
    "phase_progress",
    *FEATURES,
    "directed_signal",
    "soft_signal",
    "isolation_signal",
)
_TIME_STABLE_PAIR_AGG_FIELDS = tuple(
    field for field in _PAIR_AGG_FIELDS if field != "phase_progress"
)
_PAIR_STATIC_FIELDS = (
    "shared_hands_calc",
    "flow_asymmetry",
    "dominant_flow_bb",
)
_PLAYER_COMPARISON_FIELDS = (
    "account_age_gap",
    "same_experience_hands_bucket",
    "same_preferred_stake",
    "same_region_bucket",
    "same_client_family",
)
_TEMPORAL_SIGNAL_FIELDS = tuple(
    f"{signal}__t{quartile}"
    for signal in ("directed_signal", "soft_signal", "isolation_signal")
    for quartile in (3, 2, 1, 0)
)
_TEMPORAL_RANGE_FIELDS = (
    "directed_signal__temporal_range",
    "soft_signal__temporal_range",
    "isolation_signal__temporal_range",
)
B001_FULL_FEATURES = (
    *(
        f"{field}__{aggregate}"
        for aggregate in ("mean", "max", "p95")
        for field in _PAIR_AGG_FIELDS
    ),
    *_PAIR_STATIC_FIELDS,
    *_TEMPORAL_SIGNAL_FIELDS,
    *_TEMPORAL_RANGE_FIELDS,
    *_PLAYER_COMPARISON_FIELDS,
)
B001_TIME_FEATURES = (
    *(
        f"{field}__{aggregate}"
        for aggregate in ("mean", "max", "p95")
        for field in _TIME_STABLE_PAIR_AGG_FIELDS
    ),
    *_PAIR_STATIC_FIELDS,
    *_PLAYER_COMPARISON_FIELDS,
)
if len(B001_FULL_FEATURES) != 95 or len(B001_TIME_FEATURES) != 77:
    raise RuntimeError("frozen B001 feature dimensions changed")

PROHIBITED_MODEL_FEATURES = {
    "pair_id",
    "hand_id",
    "player_1",
    "player_2",
    "table_id",
    "table_id_1",
    "table_id_2",
    "phase_progress",
    "label",
    "label_status",
    "behavior_family",
    "fold",
    "outer_fold",
    "inner_fold",
    "evidence_rank",
    "is_evidence",
    "started_at",
}


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise V4InvariantError(f"missing frozen feature columns: {missing}")
    for column in columns:
        if column in PROHIBITED_MODEL_FEATURES or column.endswith("_id"):
            raise V4InvariantError(f"prohibited model feature: {column}")
    try:
        values = frame.loc[:, list(columns)].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise V4InvariantError("a frozen feature is not numeric") from error
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass(frozen=True)
class FrozenHandTransformV4:
    feature_names: tuple[str, ...]
    impute_median: np.ndarray
    lower_01: np.ndarray
    upper_99: np.ndarray
    scale_median: np.ndarray
    scale_iqr: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: Sequence[str] = FEATURES,
    ) -> "FrozenHandTransformV4":
        names = tuple(feature_names)
        if names != FEATURES:
            raise V4InvariantError("the v4 evidence feature order changed")
        values = _numeric_matrix(frame, names)
        if len(values) == 0:
            raise V4InvariantError("cannot fit preprocessing on zero hands")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            impute = np.nanmedian(values, axis=0)
        if not np.isfinite(impute).all():
            raise V4InvariantError("a training feature has no finite value")
        imputed = np.where(np.isnan(values), impute, values)
        lower = np.quantile(imputed, 0.01, axis=0, method="linear")
        upper = np.quantile(imputed, 0.99, axis=0, method="linear")
        clipped = np.clip(imputed, lower, upper)
        center = np.median(clipped, axis=0)
        q25, q75 = np.quantile(clipped, [0.25, 0.75], axis=0, method="linear")
        iqr = np.where(q75 - q25 == 0.0, 1.0, q75 - q25)
        if not all(np.isfinite(x).all() for x in (lower, upper, center, iqr)):
            raise V4InvariantError("non-finite preprocessing statistic")
        return cls(names, impute, lower, upper, center, iqr)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_matrix(frame, self.feature_names)
        imputed = np.where(np.isnan(values), self.impute_median, values)
        result = (
            np.clip(imputed, self.lower_01, self.upper_99) - self.scale_median
        ) / self.scale_iqr
        if not np.isfinite(result).all():
            raise V4InvariantError("preprocessing produced a non-finite value")
        return result

    def receipt(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "impute_median": self.impute_median.tolist(),
            "winsor_lower_01": self.lower_01.tolist(),
            "winsor_upper_99": self.upper_99.tolist(),
            "scale_median_after_winsorization": self.scale_median.tolist(),
            "scale_iqr_after_winsorization": self.scale_iqr.tolist(),
            "zero_iqr_replacement": 1.0,
            "quantile_method": "linear",
        }

    @classmethod
    def from_receipt(cls, receipt: Mapping[str, object]) -> "FrozenHandTransformV4":
        expected_keys = {
            "feature_names",
            "impute_median",
            "winsor_lower_01",
            "winsor_upper_99",
            "scale_median_after_winsorization",
            "scale_iqr_after_winsorization",
            "zero_iqr_replacement",
            "quantile_method",
        }
        if set(receipt) != expected_keys:
            raise V4InvariantError("frozen transform receipt schema changed")
        names = tuple(str(value) for value in receipt["feature_names"])
        if names != FEATURES:
            raise V4InvariantError("frozen transform feature order changed")
        if receipt["zero_iqr_replacement"] != 1.0:
            raise V4InvariantError("frozen zero-IQR rule changed")
        if receipt["quantile_method"] != "linear":
            raise V4InvariantError("frozen quantile method changed")
        arrays = tuple(
            np.asarray(receipt[key], dtype=np.float64)
            for key in (
                "impute_median",
                "winsor_lower_01",
                "winsor_upper_99",
                "scale_median_after_winsorization",
                "scale_iqr_after_winsorization",
            )
        )
        if any(array.shape != (len(FEATURES),) for array in arrays):
            raise V4InvariantError("frozen transform receipt dimensions changed")
        if any(not np.isfinite(array).all() for array in arrays):
            raise V4InvariantError("frozen transform receipt is non-finite")
        if np.any(arrays[4] <= 0):
            raise V4InvariantError("frozen transform receipt has nonpositive IQR")
        return cls(names, *arrays)


def interaction_matrix_v4(
    base: np.ndarray,
    families: Sequence[object],
) -> np.ndarray:
    values = np.asarray(base, dtype=np.float64)
    routes = np.asarray(families, dtype=str)
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise V4InvariantError("the base evidence matrix must have 20 columns")
    if len(routes) != len(values) or not set(routes).issubset(FAMILIES):
        raise V4InvariantError("invalid or missing behavior route")
    result = np.concatenate(
        [values, *(values * (routes == family)[:, None] for family in FAMILIES)],
        axis=1,
    )
    if result.shape[1] != 80:
        raise RuntimeError("L1 interaction dimension is not 80")
    return result


@dataclass(frozen=True)
class PairwisePreferencesV4:
    matrix: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray
    query_receipt: pd.DataFrame


def build_pairwise_preferences_v4(
    matrix: np.ndarray,
    pair_ids: Sequence[object],
    is_evidence: Sequence[object],
) -> PairwisePreferencesV4:
    values = np.asarray(matrix, dtype=np.float64)
    ids = np.asarray(pair_ids, dtype=str)
    relevant = np.asarray(is_evidence, dtype=bool)
    if values.ndim != 2 or len(values) != len(ids) or len(ids) != len(relevant):
        raise V4InvariantError("misaligned pairwise inputs")
    if not np.isfinite(values).all():
        raise V4InvariantError("pairwise matrix contains non-finite values")
    differences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    receipts: list[dict[str, object]] = []
    for pair_id in pd.unique(ids):
        positions = np.flatnonzero(ids == pair_id)
        positive = positions[relevant[positions]]
        unjudged = positions[~relevant[positions]]
        if len(positive) == 0:
            raise V4InvariantError("a fitting query has no organizer evidence hand")
        count = len(positive) * len(unjudged)
        receipts.append(
            {
                "pair_id": str(pair_id),
                "candidate_hands": int(len(positions)),
                "evidence_hands": int(len(positive)),
                "unjudged_hands": int(len(unjudged)),
                "preference_pairs": int(count),
                "symmetric_rows": int(2 * count),
                "total_symmetric_weight": 1.0 if count else 0.0,
                "unjudged_is_ground_truth_negative": False,
            }
        )
        if count == 0:
            continue
        forward = (
            values[positive, None, :] - values[None, unjudged, :]
        ).reshape(count, values.shape[1])
        differences.extend((forward, -forward))
        targets.extend(
            (
                np.ones(count, dtype=np.int8),
                np.zeros(count, dtype=np.int8),
            )
        )
        weights.extend(
            (
                np.full(count, 0.5 / count),
                np.full(count, 0.5 / count),
            )
        )
    if not differences:
        raise V4InvariantError("no trainable pairwise preference exists")
    result = PairwisePreferencesV4(
        matrix=np.concatenate(differences),
        target=np.concatenate(targets),
        sample_weight=np.concatenate(weights),
        query_receipt=pd.DataFrame(receipts),
    )
    trainable = int((result.query_receipt["preference_pairs"] > 0).sum())
    if not np.isclose(result.sample_weight.sum(), trainable, atol=1e-12):
        raise RuntimeError("pairwise weights do not sum to one per trainable query")
    return result


def new_evidence_ranker_v4(c_value: float) -> LogisticRegression:
    if sklearn.__version__ != FROZEN_SKLEARN_VERSION:
        raise V4InvariantError("scikit-learn version differs from the frozen runtime")
    if float(c_value) not in C_GRID:
        raise V4InvariantError(f"C is outside the frozen grid: {c_value}")
    return LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-8,
    )


def fit_evidence_ranker_v4(
    preferences: PairwisePreferencesV4,
    c_value: float,
) -> LogisticRegression:
    model = new_evidence_ranker_v4(c_value)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            preferences.matrix,
            preferences.target,
            sample_weight=preferences.sample_weight,
        )
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise V4InvariantError("the frozen lbfgs fit did not converge")
    if not np.isfinite(model.coef_).all():
        raise V4InvariantError("the evidence ranker has non-finite coefficients")
    return model


def heuristic_scores_v4(
    frame: pd.DataFrame,
    predicted_families: Sequence[object],
) -> np.ndarray:
    routes = np.asarray(predicted_families, dtype=str)
    if len(routes) != len(frame) or not set(routes).issubset(FAMILIES):
        raise V4InvariantError("H0 received an invalid predicted-family route")
    missing = [column for column in H0_SIGNAL_BY_FAMILY.values() if column not in frame]
    if missing:
        raise V4InvariantError(f"H0 signal columns are missing: {missing}")
    scores = np.empty(len(frame), dtype=np.float64)
    for family, column in H0_SIGNAL_BY_FAMILY.items():
        mask = routes == family
        scores[mask] = pd.to_numeric(
            frame.loc[mask, column], errors="raise"
        ).to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise V4InvariantError("H0 produced a non-finite score")
    return scores


def expected_random_tie_ap5_v4(
    relevance: Sequence[object],
    scores: Sequence[object],
) -> float:
    y = np.asarray(relevance, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.ndim != 1 or len(y) != len(values) or len(y) == 0:
        raise V4InvariantError("invalid query metric inputs")
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(values).all():
        raise V4InvariantError("metric relevance must be binary and scores finite")
    total_relevant = int(y.sum())
    if total_relevant == 0:
        return 0.0
    denominator = min(total_relevant, 5)
    positions_before = 0
    relevant_before = 0
    numerator = 0.0
    for score in np.unique(values)[::-1]:
        block = y[values == score]
        n = int(len(block))
        r = int(block.sum())
        positions_used = min(n, max(0, 5 - positions_before))
        for offset in range(1, positions_used + 1):
            if n == 1:
                contribution = (
                    float(relevant_before + 1) / float(positions_before + 1)
                    if r == 1
                    else 0.0
                )
            else:
                contribution = (
                    (r / n)
                    * (
                        relevant_before
                        + 1
                        + (offset - 1) * (r - 1) / (n - 1)
                    )
                    / (positions_before + offset)
                )
            numerator += contribution
        positions_before += n
        relevant_before += r
        if positions_before >= 5:
            break
    return float(numerator / denominator)


def query_ap_frame_v4(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {
        "pair_id",
        "table_id",
        "outer_fold",
        "is_evidence",
        score_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V4InvariantError(f"query metric columns missing: {missing}")
    if any("family" in column.lower() for column in frame.columns):
        raise V4InvariantError(
            "held-out family must not enter the pre-freeze query metric frame"
        )
    rows = []
    for pair_id, part in frame.groupby("pair_id", sort=False):
        for column in ("table_id", "outer_fold"):
            if part[column].nunique(dropna=False) != 1:
                raise V4InvariantError(f"query has multiple {column} values")
        if int(part["is_evidence"].sum()) == 0:
            raise V4InvariantError("an eligible query has no organizer evidence")
        rows.append(
            {
                "pair_id": str(pair_id),
                "table_id": str(part["table_id"].iloc[0]),
                "outer_fold": int(part["outer_fold"].iloc[0]),
                "query_ap5": expected_random_tie_ap5_v4(
                    part["is_evidence"], part[score_column]
                ),
                "candidate_hands": int(len(part)),
                "evidence_hands": int(part["is_evidence"].sum()),
            }
        )
    if not rows:
        raise V4InvariantError("no eligible query metric exists")
    return pd.DataFrame(rows)


def select_c_no_family_v4(
    query_scores_by_c: Mapping[float, pd.DataFrame],
) -> tuple[float, dict[float, float]]:
    """Select C by pooled query mean AP@5 with no family information."""

    if set(float(key) for key in query_scores_by_c) != set(C_GRID):
        raise V4InvariantError("C selection did not evaluate the exact frozen grid")
    reference_keys: tuple[str, ...] | None = None
    means: dict[float, float] = {}
    forbidden_fragments = ("family", "behavior", "label", "route")
    for c_value in C_GRID:
        frame = query_scores_by_c[c_value]
        if any(
            fragment in column.lower()
            for column in frame.columns
            for fragment in forbidden_fragments
        ):
            raise V4InvariantError("family/label/route entered C selection")
        if set(frame.columns) != {
            "pair_id",
            "table_id",
            "outer_fold",
            "query_ap5",
            "candidate_hands",
            "evidence_hands",
        }:
            raise V4InvariantError("C-selection frame schema changed")
        if frame["pair_id"].astype(str).duplicated().any() or frame.empty:
            raise V4InvariantError("C-selection query keys are empty or duplicated")
        keys = tuple(sorted(frame["pair_id"].astype(str)))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise V4InvariantError("C candidates do not share an identical query set")
        values = pd.to_numeric(frame["query_ap5"], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise V4InvariantError("C-selection metric is non-finite")
        means[c_value] = float(values.mean())
    best_c = C_GRID[0]
    best_score = means[best_c]
    for c_value in C_GRID[1:]:
        if means[c_value] > best_score:
            best_c = c_value
            best_score = means[c_value]
    return float(best_c), means


def deterministic_top_five_v4(
    frame: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    required = {"pair_id", "hand_id", "started_at", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V4InvariantError(f"top-five columns missing: {missing}")
    if any("family" in column.lower() for column in frame.columns):
        raise V4InvariantError("family entered pre-freeze top-five materialization")
    if frame[["pair_id", "hand_id"]].duplicated().any():
        raise V4InvariantError("duplicate pair-hand key")
    if frame["started_at"].isna().any():
        raise V4InvariantError("a candidate hand lacks started_at")
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all():
        raise V4InvariantError("top-five score is non-finite")
    work = frame.assign(_score=score)
    ties = work.groupby(["pair_id", "_score"], sort=False).size()
    tied_keys = ties[ties > 1].reset_index()[["pair_id", "_score"]]
    tied_rows = work.merge(
        tied_keys, on=["pair_id", "_score"], how="inner", validate="many_to_many"
    )
    if tied_rows.duplicated(
        ["pair_id", "_score", "started_at"], keep=False
    ).any():
        raise V4InvariantError(
            "an exact score tie also shares started_at; ID fallback prohibited"
        )
    ordered = work.sort_values(
        ["pair_id", "_score", "started_at"],
        ascending=[True, False, True],
        kind="stable",
    )
    top = ordered.groupby("pair_id", sort=False).head(5).copy()
    top["rank"] = top.groupby("pair_id", sort=False).cumcount() + 1
    return top[["pair_id", "rank", "hand_id", "started_at", score_column]]


def assign_inner_folds_v4(
    positive_pairs: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    required = {"pair_id", "table_id", "behavior_family", "outer_fold"}
    missing = sorted(required - set(positive_pairs.columns))
    if missing:
        raise V4InvariantError(f"inner split columns missing: {missing}")
    if outer_fold not in range(OUTER_FOLDS):
        raise V4InvariantError("outer fold is outside 0..4")
    frame = positive_pairs[positive_pairs["outer_fold"] != outer_fold].copy()
    if frame["pair_id"].duplicated().any():
        raise V4InvariantError("duplicate confirmed-target pair")
    splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=INNER_SPLIT_BASE_SEED + outer_fold,
    )
    assigned = np.full(len(frame), -1, dtype=np.int8)
    for inner_fold, (_, valid) in enumerate(
        splitter.split(frame, frame["behavior_family"], frame["table_id"])
    ):
        assigned[valid] = inner_fold
    if np.any(assigned < 0):
        raise V4InvariantError("an outer-training pair lacks an inner fold")
    frame["inner_fold"] = assigned
    if frame.groupby("table_id")["inner_fold"].nunique().max() != 1:
        raise V4InvariantError("an inner split divides a table pool")
    return frame.reset_index(drop=True)


def _b001_matrix_v4(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    missing = [column for column in feature_names if column not in frame]
    if missing:
        raise V4InvariantError(f"nested B001 features missing: {missing}")
    matrix = (
        frame.loc[:, list(feature_names)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(matrix).all():
        raise V4InvariantError("nested B001 matrix is non-finite")
    return matrix


def _new_b001_router_v4(seed: int) -> HistGradientBoostingClassifier:
    if sklearn.__version__ != FROZEN_SKLEARN_VERSION:
        raise V4InvariantError("scikit-learn version differs from the frozen runtime")
    return HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=280,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=2.5,
        early_stopping=False,
        random_state=seed,
    )


def fit_nested_behavior_routes_v4(
    assignments: pd.DataFrame,
    source_pair_features: pd.DataFrame,
    target_pair_features: pd.DataFrame,
    *,
    view: str,
    outer_fold: int,
) -> pd.DataFrame:
    if view not in VIEW_INDEX or outer_fold not in range(OUTER_FOLDS):
        raise V4InvariantError("invalid nested-routing view or outer fold")
    features = B001_FULL_FEATURES if view == "full" else B001_TIME_FEATURES
    keys = assignments[["pair_id", "table_id", "behavior_family", "inner_fold"]]
    source = keys.merge(
        source_pair_features,
        on="pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("_split", "_feature"),
    )
    target = keys[["pair_id", "table_id", "inner_fold"]].merge(
        target_pair_features,
        on="pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("_split", "_feature"),
    )
    for merged in (source, target):
        table_columns = [column for column in merged if column.startswith("table_id")]
        if len(table_columns) >= 2:
            left = merged[table_columns[0]].astype(str).to_numpy()
            if any(
                not np.array_equal(left, merged[column].astype(str).to_numpy())
                for column in table_columns[1:]
            ):
                raise V4InvariantError("B001 feature pool differs from split pool")
    source_x = _b001_matrix_v4(source, features)
    target_x = _b001_matrix_v4(target, features)
    probabilities = np.zeros((len(keys), len(FAMILIES)), dtype=np.float64)
    for inner_fold in range(INNER_FOLDS):
        train = np.flatnonzero(source["inner_fold"].to_numpy() != inner_fold)
        valid = np.flatnonzero(target["inner_fold"].to_numpy() == inner_fold)
        train_pools = set(source.iloc[train]["table_id_split"].astype(str))
        valid_pools = set(target.iloc[valid]["table_id_split"].astype(str))
        if train_pools & valid_pools:
            raise V4InvariantError("nested router train/validation pools overlap")
        family_counts = source.iloc[train]["behavior_family"].astype(str).value_counts()
        if any(int(family_counts.get(family, 0)) == 0 for family in FAMILIES):
            raise V4InvariantError("nested B001 training fold lacks a family")
        router = _new_b001_router_v4(
            ROUTER_SEED_BASE[view] + outer_fold * 10 + inner_fold
        )
        with threadpool_limits(limits=1):
            router.fit(source_x[train], source.iloc[train]["behavior_family"].astype(str))
            raw = router.predict_proba(target_x[valid])
        for source_index, family in enumerate(router.classes_):
            family_name = str(family)
            if family_name not in FAMILIES:
                raise V4InvariantError("nested B001 emitted an unknown family")
            probabilities[valid, FAMILIES.index(family_name)] = raw[:, source_index]
    if np.any(probabilities.sum(axis=1) == 0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise V4InvariantError("a nested B001 route is missing or malformed")
    receipt = keys[["pair_id", "table_id", "inner_fold"]].copy()
    receipt["outer_fold"] = int(outer_fold)
    receipt["view"] = view
    receipt["predicted_family"] = np.asarray(FAMILIES, dtype=object)[
        probabilities.argmax(axis=1)
    ]
    for index, family in enumerate(FAMILIES):
        receipt[f"family_probability__{family}"] = probabilities[:, index]
    if "behavior_family" in receipt:
        raise RuntimeError("true family leaked into nested route receipt")
    return receipt


@dataclass(frozen=True)
class CoverageTupleV4:
    eligible_queries: int
    candidate_hands: int
    evidence_hands: int
    eligible_pools: int

    @classmethod
    def from_sequence(cls, values: Sequence[object]) -> "CoverageTupleV4":
        if len(values) != 4:
            raise V4InvariantError("coverage tuple must have four values")
        result = cls(*(int(value) for value in values))
        if (
            result.eligible_queries <= 0
            or result.candidate_hands < result.eligible_queries
            or result.evidence_hands < result.eligible_queries
            or result.eligible_pools <= 0
        ):
            raise V4InvariantError("coverage tuple is malformed")
        return result

    def as_dict(self) -> dict[str, int]:
        return {
            "eligible_queries": self.eligible_queries,
            "candidate_hands": self.candidate_hands,
            "evidence_hands": self.evidence_hands,
            "eligible_pools": self.eligible_pools,
        }


@dataclass(frozen=True)
class CoverageBookV4:
    cells: Mapping[tuple[str, str, str], CoverageTupleV4]

    @classmethod
    def from_mapping(cls, contract: Mapping[str, object]) -> "CoverageBookV4":
        if contract.get("status") != "DESIGN_ONLY_NO_RESULT":
            raise V4InvariantError("coverage contract status changed")
        if contract.get("count_tuple_schema") != [
            "eligible_queries",
            "candidate_hands",
            "evidence_hands",
            "eligible_pools",
        ]:
            raise V4InvariantError("coverage tuple schema changed")
        views = contract.get("views")
        if not isinstance(views, Mapping) or set(views) != set(VIEW_INDEX):
            raise V4InvariantError("coverage views changed")
        cells: dict[tuple[str, str, str], CoverageTupleV4] = {}
        expected_levels = {
            "overall": {"overall"},
            "family": set(FAMILIES),
            "outer_fold": {str(fold) for fold in range(OUTER_FOLDS)},
            "outer_fold_family": {
                f"{fold}/{family}"
                for fold in range(OUTER_FOLDS)
                for family in FAMILIES
            },
        }
        for view in VIEW_INDEX:
            view_mapping = views[view]
            if not isinstance(view_mapping, Mapping) or set(view_mapping) != set(
                expected_levels
            ):
                raise V4InvariantError(f"coverage levels changed: {view}")
            for level, expected_keys in expected_levels.items():
                raw = view_mapping[level]
                raw_mapping = {"overall": raw} if level == "overall" else raw
                if not isinstance(raw_mapping, Mapping) or set(raw_mapping) != expected_keys:
                    raise V4InvariantError(f"coverage keys changed: {view}/{level}")
                for key, values in raw_mapping.items():
                    cells[(view, level, str(key))] = CoverageTupleV4.from_sequence(
                        values
                    )
        if len(cells) != 72:
            raise V4InvariantError("coverage contract must contain exactly 72 cells")
        floors = contract.get("stops", {})
        if floors.get("full_fold_family_minimum") != [18, 2058, 90, 14]:
            raise V4InvariantError("full low-coverage floor changed")
        if floors.get("time_fold_family_minimum") != [7, 458, 17, 6]:
            raise V4InvariantError("time low-coverage floor changed")
        for (view, level, _), coverage in cells.items():
            if level != "outer_fold_family":
                continue
            floor = (18, 2058, 90, 14) if view == "full" else (7, 458, 17, 6)
            observed = (
                coverage.eligible_queries,
                coverage.candidate_hands,
                coverage.evidence_hands,
                coverage.eligible_pools,
            )
            if any(value < minimum for value, minimum in zip(observed, floor)):
                raise V4InvariantError(f"a frozen coverage cell is below floor: {view}")
        return cls(cells)

    def expected(self, view: str, level: str, key: str) -> CoverageTupleV4:
        try:
            return self.cells[(view, level, key)]
        except KeyError as error:
            raise V4InvariantError(f"unknown coverage cell: {view}/{level}/{key}") from error

    def validate(
        self,
        view: str,
        level: str,
        key: str,
        observed: CoverageTupleV4,
    ) -> dict[str, int]:
        expected = self.expected(view, level, key)
        if observed != expected:
            raise V4InvariantError(
                f"metric-local coverage changed: {view}/{level}/{key}; "
                f"expected={expected.as_dict()} observed={observed.as_dict()}"
            )
        return observed.as_dict()


def observed_coverage_v4(frame: pd.DataFrame) -> CoverageTupleV4:
    required = {"pair_id", "hand_id", "table_id", "is_evidence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V4InvariantError(f"coverage columns missing: {missing}")
    if frame.empty or frame[["pair_id", "hand_id"]].duplicated().any():
        raise V4InvariantError("coverage frame is empty or duplicated")
    if frame.groupby("pair_id")["is_evidence"].any().eq(False).any():
        raise V4InvariantError("coverage contains a query without evidence")
    return CoverageTupleV4(
        eligible_queries=int(frame["pair_id"].astype(str).nunique()),
        candidate_hands=int(len(frame)),
        evidence_hands=int(frame["is_evidence"].sum()),
        eligible_pools=int(frame["table_id"].astype(str).nunique()),
    )


def validate_all_metric_coverage_v4(
    frame: pd.DataFrame,
    view: str,
    book: CoverageBookV4,
) -> dict[str, dict[str, dict[str, int]]]:
    required = {"behavior_family", "outer_fold"}
    if not required.issubset(frame.columns):
        raise V4InvariantError("post-score reporting metadata is incomplete")
    if not set(frame["behavior_family"].astype(str)).issubset(FAMILIES):
        raise V4InvariantError("reporting family is invalid")
    result: dict[str, dict[str, dict[str, int]]] = {
        "overall": {},
        "family": {},
        "outer_fold": {},
        "outer_fold_family": {},
    }
    result["overall"]["overall"] = book.validate(
        view, "overall", "overall", observed_coverage_v4(frame)
    )
    for family in FAMILIES:
        part = frame[frame["behavior_family"].astype(str) == family]
        result["family"][family] = book.validate(
            view, "family", family, observed_coverage_v4(part)
        )
    for fold in range(OUTER_FOLDS):
        part = frame[frame["outer_fold"].astype(int) == fold]
        result["outer_fold"][str(fold)] = book.validate(
            view, "outer_fold", str(fold), observed_coverage_v4(part)
        )
        for family in FAMILIES:
            cell = part[part["behavior_family"].astype(str) == family]
            key = f"{fold}/{family}"
            result["outer_fold_family"][key] = book.validate(
                view, "outer_fold_family", key, observed_coverage_v4(cell)
            )
    return result


def metric_tree_v4(
    query_scores: pd.DataFrame,
    reporting_hands: pd.DataFrame,
    view: str,
    book: CoverageBookV4,
) -> dict[str, object]:
    """Compute each scalar only after validating and embedding local coverage."""

    coverage = validate_all_metric_coverage_v4(reporting_hands, view, book)
    required = {"pair_id", "outer_fold", "behavior_family", "query_ap5"}
    if not required.issubset(query_scores.columns):
        raise V4InvariantError("post-freeze query scores are incomplete")
    if query_scores["pair_id"].astype(str).duplicated().any():
        raise V4InvariantError("query scores contain duplicate pairs")
    hand_pairs = set(reporting_hands["pair_id"].astype(str))
    if set(query_scores["pair_id"].astype(str)) != hand_pairs:
        raise V4InvariantError("metric query universe differs from coverage universe")

    def node(part: pd.DataFrame, local_coverage: Mapping[str, int]) -> dict[str, object]:
        values = pd.to_numeric(part["query_ap5"], errors="raise").to_numpy(float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise V4InvariantError("metric cell is empty or non-finite")
        return {"value": float(values.mean()), "coverage": dict(local_coverage)}

    result: dict[str, object] = {
        "overall": node(query_scores, coverage["overall"]["overall"]),
        "family": {},
        "outer_fold": {},
        "outer_fold_family": {},
    }
    for family in FAMILIES:
        part = query_scores[query_scores["behavior_family"].astype(str) == family]
        result["family"][family] = node(part, coverage["family"][family])
    for fold in range(OUTER_FOLDS):
        part = query_scores[query_scores["outer_fold"].astype(int) == fold]
        result["outer_fold"][str(fold)] = node(
            part, coverage["outer_fold"][str(fold)]
        )
        for family in FAMILIES:
            key = f"{fold}/{family}"
            cell = part[part["behavior_family"].astype(str) == family]
            result["outer_fold_family"][key] = node(
                cell, coverage["outer_fold_family"][key]
            )
    macro = np.mean([result["family"][family]["value"] for family in FAMILIES])
    result["unweighted_macro_family"] = {
        "value": float(macro),
        "coverage": coverage["overall"]["overall"],
    }
    return result


@dataclass(frozen=True)
class BayesianClusterWeightPlanV4:
    view: str
    query_keys: tuple[str, ...]
    pool_keys: tuple[str, ...]
    outer_folds: np.ndarray
    query_weights: np.ndarray
    fold_pool_keys: tuple[tuple[str, ...], ...]
    fold_pool_weights: tuple[np.ndarray, ...]

    @classmethod
    def create(cls, query_meta: pd.DataFrame, view: str) -> "BayesianClusterWeightPlanV4":
        if view not in VIEW_INDEX:
            raise V4InvariantError(f"unknown sensitivity view: {view}")
        if np.__version__ != WEIGHT_STREAM_NUMPY_VERSION:
            raise V4InvariantError("NumPy version differs from frozen weight runtime")
        required = {"pair_id", "table_id", "outer_fold"}
        if not required.issubset(query_meta.columns):
            raise V4InvariantError("sensitivity metadata is incomplete")
        if query_meta["pair_id"].astype(str).duplicated().any():
            raise V4InvariantError("sensitivity metadata has duplicate queries")
        frame = query_meta.reset_index(drop=True).copy()
        folds = frame["outer_fold"].to_numpy(dtype=int)
        if set(folds) != set(range(OUTER_FOLDS)):
            raise V4InvariantError("sensitivity folds are incomplete")
        weights = np.zeros((BOOTSTRAP_DRAWS, len(frame)), dtype=np.float64)
        fold_pool_keys = []
        fold_pool_weights = []
        for fold in range(OUTER_FOLDS):
            positions = np.flatnonzero(folds == fold)
            pools = tuple(sorted(frame.iloc[positions]["table_id"].astype(str).unique()))
            if not pools:
                raise V4InvariantError("a sensitivity fold has no pool")
            rng = np.random.default_rng(
                np.random.SeedSequence([BOOTSTRAP_BASE_SEED, VIEW_INDEX[view], fold])
            )
            if type(rng.bit_generator).__name__ != WEIGHT_STREAM_BIT_GENERATOR:
                raise V4InvariantError("NumPy bit generator changed")
            pool_weights = rng.exponential(1.0, size=(BOOTSTRAP_DRAWS, len(pools)))
            pool_weights /= pool_weights.mean(axis=1, keepdims=True)
            pool_index = {pool: index for index, pool in enumerate(pools)}
            columns = [
                pool_index[pool]
                for pool in frame.iloc[positions]["table_id"].astype(str)
            ]
            weights[:, positions] = pool_weights[:, columns]
            fold_pool_keys.append(pools)
            fold_pool_weights.append(pool_weights)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise RuntimeError("Bayesian cluster weights are not positive and finite")
        return cls(
            view=view,
            query_keys=tuple(frame["pair_id"].astype(str)),
            pool_keys=tuple(frame["table_id"].astype(str)),
            outer_folds=folds,
            query_weights=weights,
            fold_pool_keys=tuple(fold_pool_keys),
            fold_pool_weights=tuple(fold_pool_weights),
        )

    def validate_delta(self, deltas: pd.DataFrame) -> np.ndarray:
        required = {"pair_id", "table_id", "outer_fold", "delta"}
        if not required.issubset(deltas.columns):
            raise V4InvariantError("paired delta fields are missing")
        frame = deltas.reset_index(drop=True)
        if tuple(frame["pair_id"].astype(str)) != self.query_keys:
            raise V4InvariantError("paired-delta query order or membership changed")
        if tuple(frame["table_id"].astype(str)) != self.pool_keys:
            raise V4InvariantError("paired-delta pool membership changed")
        if not np.array_equal(frame["outer_fold"].to_numpy(int), self.outer_folds):
            raise V4InvariantError("paired-delta folds changed")
        values = pd.to_numeric(frame["delta"], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise V4InvariantError("paired delta is non-finite")
        return values

    def weighted_draws(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            raise V4InvariantError("sensitivity cell is empty")
        denominator = self.query_weights[:, mask].sum(axis=1)
        if np.any(denominator <= 0):
            raise V4InvariantError("sensitivity denominator is not positive")
        return (self.query_weights[:, mask] @ values[mask]) / denominator


def frozen_stage_weight_stream_sha256_v4(
    plans: Mapping[str, BayesianClusterWeightPlanV4],
    views: Sequence[str],
    *,
    include_headers: bool = True,
) -> str:
    view_order = tuple(views)
    if view_order not in STAGE_WEIGHT_STREAM_SHA256:
        raise V4InvariantError("unfrozen stage weight-view order")
    digest = hashlib.sha256()
    for view in view_order:
        plan = plans.get(view)
        if plan is None or plan.view != view:
            raise V4InvariantError(f"missing or mismatched weight plan: {view}")
        for fold, values in enumerate(plan.fold_pool_weights):
            if values.shape != (BOOTSTRAP_DRAWS, len(plan.fold_pool_keys[fold])):
                raise V4InvariantError("pool-weight matrix shape changed")
            if include_headers:
                digest.update(view.encode("utf-8"))
                digest.update(bytes([fold]))
            digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def paired_delta_frame_v4(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["pair_id", "table_id", "outer_fold"]
    for frame in (method_a, method_b):
        if any(column not in frame for column in (*keys, "query_ap5")):
            raise V4InvariantError("query AP frame is incomplete")
        if frame["pair_id"].astype(str).duplicated().any():
            raise V4InvariantError("query AP frame has duplicate pairs")
    joined = method_a[keys + ["query_ap5"]].merge(
        method_b[keys + ["query_ap5"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_a", "_b"),
    )
    if not (joined["_merge"] == "both").all():
        raise V4InvariantError("paired methods do not share an identical query set")
    joined["delta"] = joined["query_ap5_a"] - joined["query_ap5_b"]
    return joined[keys + ["delta"]]


def sensitivity_tree_v4(
    deltas: pd.DataFrame,
    reporting_hands: pd.DataFrame,
    view: str,
    plan: BayesianClusterWeightPlanV4,
    book: CoverageBookV4,
) -> dict[str, object]:
    coverage = validate_all_metric_coverage_v4(reporting_hands, view, book)
    meta = reporting_hands[
        ["pair_id", "behavior_family"]
    ].drop_duplicates("pair_id", keep="first")
    ordered = deltas.assign(pair_id=deltas["pair_id"].astype(str)).merge(
        meta.assign(pair_id=meta["pair_id"].astype(str)),
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    indexed = ordered.set_index("pair_id")
    if set(indexed.index) != set(plan.query_keys):
        raise V4InvariantError("delta queries differ from the weight plan")
    ordered = indexed.loc[list(plan.query_keys)].reset_index()
    values = plan.validate_delta(ordered)
    families = ordered["behavior_family"].astype(str).to_numpy()
    folds = ordered["outer_fold"].to_numpy(int)

    def node(mask: np.ndarray, local_coverage: Mapping[str, int]) -> dict[str, object]:
        draws = plan.weighted_draws(values, mask)
        lower, upper = np.quantile(draws, [0.05, 0.95], method="linear")
        return {
            "point_delta": float(values[mask].mean()),
            "sensitivity_lower_05": float(lower),
            "sensitivity_upper_95": float(upper),
            "coverage": dict(local_coverage),
        }

    result: dict[str, object] = {
        "interval_name": SENSITIVITY_INTERVAL_NAME,
        "interval_semantics": "paired_pool_weight_sensitivity_not_confidence_or_coverage",
        "confidence_claimed": False,
        "credible_claimed": False,
        "significance_claimed": False,
        "posterior_probability_claimed": False,
        "coverage_claimed": False,
        "paired_query_level_delta_formed_before_weighting": True,
        "overall": node(np.ones(len(values), dtype=bool), coverage["overall"]["overall"]),
        "family": {},
        "outer_fold": {},
        "outer_fold_family": {},
    }
    family_draws = []
    for family in FAMILIES:
        mask = families == family
        result["family"][family] = node(mask, coverage["family"][family])
        family_draws.append(plan.weighted_draws(values, mask))
    macro_draws = np.mean(np.stack(family_draws), axis=0)
    macro_lower, macro_upper = np.quantile(
        macro_draws, [0.05, 0.95], method="linear"
    )
    result["unweighted_macro_family"] = {
        "point_delta": float(
            np.mean([result["family"][family]["point_delta"] for family in FAMILIES])
        ),
        "sensitivity_lower_05": float(macro_lower),
        "sensitivity_upper_95": float(macro_upper),
        "coverage": coverage["overall"]["overall"],
    }
    for fold in range(OUTER_FOLDS):
        mask = folds == fold
        result["outer_fold"][str(fold)] = node(
            mask, coverage["outer_fold"][str(fold)]
        )
        for family in FAMILIES:
            key = f"{fold}/{family}"
            cell = mask & (families == family)
            result["outer_fold_family"][key] = node(
                cell, coverage["outer_fold_family"][key]
            )
    return result
