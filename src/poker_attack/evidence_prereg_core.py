"""Frozen mechanics for the EVP-DRAFT-03 evidence validation.

This module contains only reusable mechanics.  Importing it performs no I/O,
fits no model, and computes no competition result.  Real-data execution is
guarded in :mod:`poker_attack.evidence_prereg` by a separate, hash-bound,
one-run authorization receipt.
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


class PreregistrationError(ValueError):
    """Raised when a frozen EVP-DRAFT-03 invariant is violated."""


DRAFT_SHA256 = "54776c546377da214626828f71deed383302ceee95142073da7b031a0d1723e5"
PREREG_STATIC_AUDIT_SHA256 = (
    "a45c55aae9cfac2b4caafb99511815b38b4da9c87c0b431edb985d554e117b91"
)

FAMILIES = tuple(TARGET_BEHAVIORS)
if FAMILIES != (
    "directed_transfer",
    "soft_play",
    "coordinated_isolation",
):
    raise RuntimeError("TARGET_BEHAVIORS no longer matches EVP-DRAFT-03")

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
SENSITIVITY_INTERVAL_NAME = (
    "90% paired Bayesian cluster-weight sensitivity interval"
)
WEIGHT_STREAM_SHA256 = (
    "c833ac7d6dd02d2009f41f3566667dcf37e59071646cc1afe9929395b601975c"
)
WEIGHT_STREAM_NO_HEADER_SHA256 = (
    "08ee7564ab1c5aa7736c1fc541a006f6c5181b2638088a6b91d24e034a222e1a"
)

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
    *(f"{field}__{aggregate}" for aggregate in ("mean", "max", "p95") for field in _PAIR_AGG_FIELDS),
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

PROHIBITED_FEATURE_NAMES = {
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
    "evidence_rank",
    "is_evidence",
    "started_at",
}


def _as_finite_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise PreregistrationError(f"missing frozen feature columns: {missing}")
    for column in columns:
        if column in PROHIBITED_FEATURE_NAMES or column.endswith("_id"):
            raise PreregistrationError(f"prohibited model feature: {column}")
    try:
        values = frame.loc[:, list(columns)].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise PreregistrationError("a frozen feature is not numeric") from error
    values[~np.isfinite(values)] = np.nan
    return values


@dataclass(frozen=True)
class FrozenHandTransform:
    """Sequential train-only imputation, winsorization, and robust scaling."""

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
    ) -> "FrozenHandTransform":
        names = tuple(feature_names)
        if names != FEATURES:
            raise PreregistrationError("the evidence feature list or order changed")
        values = _as_finite_matrix(frame, names)
        if len(values) == 0:
            raise PreregistrationError("cannot fit preprocessing on zero hands")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            impute = np.nanmedian(values, axis=0)
        if not np.isfinite(impute).all():
            raise PreregistrationError("a training feature has no finite value")
        imputed = np.where(np.isnan(values), impute, values)
        lower = np.quantile(imputed, 0.01, axis=0, method="linear")
        upper = np.quantile(imputed, 0.99, axis=0, method="linear")
        clipped = np.clip(imputed, lower, upper)
        center = np.median(clipped, axis=0)
        q25, q75 = np.quantile(
            clipped, [0.25, 0.75], axis=0, method="linear"
        )
        iqr = q75 - q25
        iqr = np.where(iqr == 0.0, 1.0, iqr)
        if not all(
            np.isfinite(array).all()
            for array in (lower, upper, center, iqr)
        ):
            raise PreregistrationError("non-finite preprocessing statistic")
        return cls(names, impute, lower, upper, center, iqr)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = _as_finite_matrix(frame, self.feature_names)
        imputed = np.where(np.isnan(values), self.impute_median, values)
        clipped = np.clip(imputed, self.lower_01, self.upper_99)
        result = (clipped - self.scale_median) / self.scale_iqr
        if not np.isfinite(result).all():
            raise PreregistrationError("preprocessing produced a non-finite value")
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


def interaction_matrix(
    base: np.ndarray,
    families: Sequence[object],
) -> np.ndarray:
    """Return 20 base plus 60 family-interaction columns in frozen order."""

    values = np.asarray(base, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise PreregistrationError("the base evidence matrix must have 20 columns")
    routes = np.asarray(families, dtype=str)
    if len(routes) != len(values) or not set(routes).issubset(FAMILIES):
        raise PreregistrationError("invalid or missing behavior route")
    blocks = [values]
    for family in FAMILIES:
        blocks.append(values * (routes == family)[:, None])
    result = np.concatenate(blocks, axis=1)
    if result.shape[1] != 80:
        raise RuntimeError("L1 interaction dimension is not 80")
    return result


@dataclass(frozen=True)
class PairwisePreferences:
    matrix: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray
    query_receipt: pd.DataFrame


def build_pairwise_preferences(
    matrix: np.ndarray,
    pair_ids: Sequence[object],
    is_evidence: Sequence[object],
) -> PairwisePreferences:
    """Build symmetric evidence-over-unjudged differences.

    A zero-class row is only the reverse of a within-query preference.  It is
    never a ground-truth negative hand, pair, or unknown pair.
    """

    values = np.asarray(matrix, dtype=np.float64)
    ids = np.asarray(pair_ids, dtype=str)
    relevant = np.asarray(is_evidence, dtype=bool)
    if values.ndim != 2 or len(values) != len(ids) or len(ids) != len(relevant):
        raise PreregistrationError("misaligned pairwise inputs")
    if not np.isfinite(values).all():
        raise PreregistrationError("pairwise matrix contains non-finite values")

    differences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    receipts: list[dict[str, object]] = []
    # Pair IDs are equality keys only. Their text and ordering never enter X.
    for pair_id in pd.unique(ids):
        positions = np.flatnonzero(ids == pair_id)
        positive_positions = positions[relevant[positions]]
        unjudged_positions = positions[~relevant[positions]]
        if len(positive_positions) == 0:
            raise PreregistrationError(
                "a pairwise training query has no organizer evidence hand"
            )
        preference_count = len(positive_positions) * len(unjudged_positions)
        receipts.append(
            {
                "pair_id": pair_id,
                "candidate_hands": int(len(positions)),
                "evidence_hands": int(len(positive_positions)),
                "unjudged_hands": int(len(unjudged_positions)),
                "preference_pairs": int(preference_count),
                "symmetric_rows": int(2 * preference_count),
                "total_symmetric_weight": 1.0 if preference_count else 0.0,
            }
        )
        if preference_count == 0:
            continue
        forward = (
            values[positive_positions, None, :]
            - values[None, unjudged_positions, :]
        ).reshape(preference_count, values.shape[1])
        differences.extend((forward, -forward))
        targets.extend(
            (
                np.ones(preference_count, dtype=np.int8),
                np.zeros(preference_count, dtype=np.int8),
            )
        )
        # Both directions together give this query total weight exactly one.
        weights.extend(
            (
                np.full(preference_count, 0.5 / preference_count),
                np.full(preference_count, 0.5 / preference_count),
            )
        )
    if not differences:
        raise PreregistrationError("no trainable pairwise preference exists")
    result = PairwisePreferences(
        matrix=np.concatenate(differences, axis=0),
        target=np.concatenate(targets),
        sample_weight=np.concatenate(weights),
        query_receipt=pd.DataFrame(receipts),
    )
    expected_weight = float((result.query_receipt["preference_pairs"] > 0).sum())
    if not np.isclose(result.sample_weight.sum(), expected_weight, atol=1e-12):
        raise RuntimeError("pairwise query weights do not sum to one per query")
    return result


def new_evidence_ranker(c_value: float) -> LogisticRegression:
    if sklearn.__version__ != FROZEN_SKLEARN_VERSION:
        raise PreregistrationError(
            "scikit-learn version differs from the frozen implementation runtime"
        )
    if float(c_value) not in C_GRID:
        raise PreregistrationError(f"C is outside the frozen grid: {c_value}")
    return LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-8,
    )


def fit_evidence_ranker(
    preferences: PairwisePreferences,
    c_value: float,
) -> LogisticRegression:
    model = new_evidence_ranker(c_value)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            preferences.matrix,
            preferences.target,
            sample_weight=preferences.sample_weight,
        )
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise PreregistrationError("the frozen lbfgs fit did not converge")
    if model.fit_intercept or model.solver != "lbfgs" or model.max_iter != 2000:
        raise RuntimeError("the fitted ranker differs from frozen configuration")
    if not np.isfinite(model.coef_).all():
        raise PreregistrationError("the evidence ranker has non-finite coefficients")
    return model


def heuristic_scores(
    frame: pd.DataFrame,
    predicted_families: Sequence[object],
) -> np.ndarray:
    routes = np.asarray(predicted_families, dtype=str)
    if len(routes) != len(frame) or not set(routes).issubset(FAMILIES):
        raise PreregistrationError("H0 received an invalid predicted-family route")
    missing = [column for column in H0_SIGNAL_BY_FAMILY.values() if column not in frame]
    if missing:
        raise PreregistrationError(f"H0 signal columns are missing: {missing}")
    result = np.empty(len(frame), dtype=np.float64)
    for family, column in H0_SIGNAL_BY_FAMILY.items():
        mask = routes == family
        result[mask] = pd.to_numeric(
            frame.loc[mask, column], errors="raise"
        ).to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise PreregistrationError("H0 produced a non-finite score")
    return result


def expected_random_tie_ap5(
    relevance: Sequence[object],
    scores: Sequence[object],
) -> float:
    """Exact expected AP@5 under uniform ordering inside exact score ties."""

    y = np.asarray(relevance, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.ndim != 1 or len(y) != len(values) or len(y) == 0:
        raise PreregistrationError("invalid query metric inputs")
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(values).all():
        raise PreregistrationError("metric relevance must be binary and scores finite")
    total_relevant = int(y.sum())
    if total_relevant == 0:
        return 0.0
    denominator = min(total_relevant, 5)
    unique_scores = np.unique(values)[::-1]
    positions_before = 0
    relevant_before = 0
    numerator = 0.0
    for score in unique_scores:
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


def query_ap_frame(
    frame: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    required = {
        "pair_id",
        "table_id",
        "outer_fold",
        "behavior_family",
        "is_evidence",
        score_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PreregistrationError(f"metric receipt columns missing: {missing}")
    rows: list[dict[str, object]] = []
    for pair_id, part in frame.groupby("pair_id", sort=False):
        for column in ("table_id", "outer_fold", "behavior_family"):
            if part[column].nunique(dropna=False) != 1:
                raise PreregistrationError(f"query has multiple {column} values")
        if int(part["is_evidence"].sum()) == 0:
            raise PreregistrationError(
                "an eligible evidence query has no organizer evidence hand"
            )
        rows.append(
            {
                "pair_id": str(pair_id),
                "table_id": str(part["table_id"].iloc[0]),
                "outer_fold": int(part["outer_fold"].iloc[0]),
                "behavior_family": str(part["behavior_family"].iloc[0]),
                "query_ap5": expected_random_tie_ap5(
                    part["is_evidence"], part[score_column]
                ),
                "candidate_hands": int(len(part)),
                "evidence_hands": int(part["is_evidence"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise PreregistrationError("no eligible query metric exists")
    return result


def deterministic_top_five(
    frame: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """Materialize top five using score then gameplay timestamp only."""

    required = {"pair_id", "hand_id", "started_at", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PreregistrationError(f"top-five receipt columns missing: {missing}")
    if frame[["pair_id", "hand_id"]].duplicated().any():
        raise PreregistrationError("duplicate pair-hand key")
    if frame["started_at"].isna().any():
        raise PreregistrationError("a candidate hand lacks started_at")
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all():
        raise PreregistrationError("top-five score is non-finite")
    work = frame.assign(_score=score)
    tied = work.groupby(["pair_id", "_score"], sort=False).size()
    tied_keys = tied[tied > 1].reset_index()[["pair_id", "_score"]]
    tied_rows = work.merge(
        tied_keys, on=["pair_id", "_score"], how="inner", validate="many_to_many"
    )
    if tied_rows.duplicated(
        ["pair_id", "_score", "started_at"], keep=False
    ).any():
        raise PreregistrationError(
            "an exact score tie also shares started_at; identifier fallback prohibited"
        )
    ordered = work.sort_values(
        ["pair_id", "_score", "started_at"],
        ascending=[True, False, True],
        kind="stable",
    )
    top = ordered.groupby("pair_id", sort=False).head(5).copy()
    top["rank"] = top.groupby("pair_id", sort=False).cumcount() + 1
    return top[["pair_id", "rank", "hand_id", "started_at", score_column]]


def assign_inner_folds(
    positive_pairs: pd.DataFrame,
    outer_fold: int,
) -> pd.DataFrame:
    required = {"pair_id", "table_id", "behavior_family", "outer_fold"}
    missing = sorted(required - set(positive_pairs.columns))
    if missing:
        raise PreregistrationError(f"inner split columns missing: {missing}")
    if outer_fold not in range(OUTER_FOLDS):
        raise PreregistrationError("outer fold is outside 0..4")
    frame = positive_pairs[positive_pairs["outer_fold"] != outer_fold].copy()
    if frame["pair_id"].duplicated().any():
        raise PreregistrationError("duplicate confirmed-target pair")
    if not set(frame["behavior_family"].astype(str)).issubset(FAMILIES):
        raise PreregistrationError("inner split contains a non-target family")
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
        raise PreregistrationError("an outer-training pair lacks an inner fold")
    frame["inner_fold"] = assigned
    if frame.groupby("table_id")["inner_fold"].nunique().max() != 1:
        raise PreregistrationError("an inner split divides a table_id pool")
    return frame.reset_index(drop=True)


def _b001_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    missing = [column for column in feature_names if column not in frame]
    if missing:
        raise PreregistrationError(f"nested B001 features missing: {missing}")
    matrix = (
        frame.loc[:, list(feature_names)]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(matrix).all():
        raise PreregistrationError("nested B001 matrix is non-finite")
    return matrix


def _new_b001_router(seed: int) -> HistGradientBoostingClassifier:
    if sklearn.__version__ != FROZEN_SKLEARN_VERSION:
        raise PreregistrationError(
            "scikit-learn version differs from the frozen implementation runtime"
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=280,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=2.5,
        early_stopping=False,
        random_state=seed,
    )


def fit_nested_behavior_routes(
    assignments: pd.DataFrame,
    source_pair_features: pd.DataFrame,
    target_pair_features: pd.DataFrame,
    *,
    view: str,
    outer_fold: int,
) -> pd.DataFrame:
    """Cross-fit B001 inside one outer-training partition.

    Validation true families are never passed to the router's feature matrix;
    they remain only in ``assignments`` to define the pre-score split.
    """

    if view not in VIEW_INDEX or outer_fold not in range(OUTER_FOLDS):
        raise PreregistrationError("invalid nested-routing view or outer fold")
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
            for column in table_columns[1:]:
                if not np.array_equal(left, merged[column].astype(str).to_numpy()):
                    raise PreregistrationError("B001 feature pool differs from split pool")
    source_x = _b001_matrix(source, features)
    target_x = _b001_matrix(target, features)
    probabilities = np.zeros((len(keys), len(FAMILIES)), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    for inner_fold in range(INNER_FOLDS):
        train = np.flatnonzero(source["inner_fold"].to_numpy() != inner_fold)
        valid = np.flatnonzero(target["inner_fold"].to_numpy() == inner_fold)
        train_pools = set(source.iloc[train]["table_id_split"].astype(str))
        valid_pools = set(target.iloc[valid]["table_id_split"].astype(str))
        if train_pools & valid_pools:
            raise PreregistrationError("nested router train/validation pools overlap")
        family_counts = source.iloc[train]["behavior_family"].astype(str).value_counts()
        if any(int(family_counts.get(family, 0)) == 0 for family in FAMILIES):
            raise PreregistrationError("nested B001 training fold lacks a family")
        seed = ROUTER_SEED_BASE[view] + outer_fold * 10 + inner_fold
        router = _new_b001_router(seed)
        with threadpool_limits(limits=1):
            router.fit(source_x[train], source.iloc[train]["behavior_family"].astype(str))
            raw = router.predict_proba(target_x[valid])
        for source_index, family in enumerate(router.classes_):
            family_name = str(family)
            if family_name not in FAMILIES:
                raise PreregistrationError("nested B001 emitted an unknown family")
            probabilities[valid, FAMILIES.index(family_name)] = raw[:, source_index]
        diagnostics.append(
            {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "seed": int(seed),
                "training_pairs": int(len(train)),
                "validation_pairs": int(len(valid)),
                "training_pools": int(len(train_pools)),
                "validation_pools": int(len(valid_pools)),
            }
        )
    if np.any(probabilities.sum(axis=1) == 0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise PreregistrationError("a nested B001 route is missing or malformed")
    predicted = np.asarray(FAMILIES, dtype=object)[probabilities.argmax(axis=1)]
    receipt = keys[["pair_id", "table_id", "inner_fold"]].copy()
    receipt["outer_fold"] = int(outer_fold)
    receipt["view"] = view
    receipt["predicted_family"] = predicted
    for index, family in enumerate(FAMILIES):
        receipt[f"family_probability__{family}"] = probabilities[:, index]
    receipt.attrs["diagnostics"] = diagnostics
    return receipt


def choose_c_exact_tie_smaller(scores: Mapping[float, float]) -> float:
    if set(float(value) for value in scores) != set(C_GRID):
        raise PreregistrationError("C selection did not evaluate the frozen grid")
    if not all(np.isfinite(float(value)) for value in scores.values()):
        raise PreregistrationError("C selection score is non-finite")
    best_c = C_GRID[0]
    best_score = float(scores[best_c])
    for c_value in C_GRID[1:]:
        candidate = float(scores[c_value])
        if candidate > best_score:
            best_c = c_value
            best_score = candidate
    return float(best_c)


def macro_family_mean(query_values: pd.DataFrame, value_column: str) -> float:
    if value_column not in query_values or "behavior_family" not in query_values:
        raise PreregistrationError("macro-family inputs are missing")
    means = []
    for family in FAMILIES:
        part = pd.to_numeric(
            query_values.loc[
                query_values["behavior_family"].astype(str) == family,
                value_column,
            ],
            errors="raise",
        )
        if len(part) == 0:
            raise PreregistrationError(f"macro-family cell is empty: {family}")
        means.append(float(part.mean()))
    return float(np.mean(means))


@dataclass(frozen=True)
class BayesianClusterWeightPlan:
    """One frozen, paired set of positive pool weights for a validation view."""

    view: str
    query_keys: tuple[str, ...]
    pool_keys: tuple[str, ...]
    outer_folds: np.ndarray
    families: tuple[str, ...]
    query_weights: np.ndarray
    fold_pool_keys: tuple[tuple[str, ...], ...]
    fold_pool_weights: tuple[np.ndarray, ...]

    @classmethod
    def create(cls, query_meta: pd.DataFrame, view: str) -> "BayesianClusterWeightPlan":
        if view not in VIEW_INDEX:
            raise PreregistrationError(f"unknown sensitivity view: {view}")
        if np.__version__ != WEIGHT_STREAM_NUMPY_VERSION:
            raise PreregistrationError(
                "NumPy version differs from the frozen weight-stream runtime"
            )
        required = {"pair_id", "table_id", "outer_fold", "behavior_family"}
        missing = sorted(required - set(query_meta.columns))
        if missing:
            raise PreregistrationError(f"sensitivity metadata missing: {missing}")
        if query_meta["pair_id"].astype(str).duplicated().any():
            raise PreregistrationError("sensitivity input has duplicate queries")
        frame = query_meta.reset_index(drop=True).copy()
        folds = frame["outer_fold"].to_numpy(dtype=int)
        families = tuple(frame["behavior_family"].astype(str))
        if set(folds) != set(range(OUTER_FOLDS)) or not set(families).issubset(FAMILIES):
            raise PreregistrationError("sensitivity folds or families are incomplete")
        weights = np.zeros((BOOTSTRAP_DRAWS, len(frame)), dtype=np.float64)
        fold_pool_keys: list[tuple[str, ...]] = []
        fold_pool_weights: list[np.ndarray] = []
        for outer_fold in range(OUTER_FOLDS):
            positions = np.flatnonzero(folds == outer_fold)
            pools = tuple(sorted(frame.iloc[positions]["table_id"].astype(str).unique()))
            if not pools:
                raise PreregistrationError("a sensitivity outer fold has no pool")
            pool_index = {pool: index for index, pool in enumerate(pools)}
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [BOOTSTRAP_BASE_SEED, VIEW_INDEX[view], outer_fold]
                )
            )
            if type(rng.bit_generator).__name__ != WEIGHT_STREAM_BIT_GENERATOR:
                raise PreregistrationError(
                    "NumPy bit generator differs from the frozen weight stream"
                )
            pool_weights = rng.exponential(
                1.0, size=(BOOTSTRAP_DRAWS, len(pools))
            )
            pool_weights /= pool_weights.mean(axis=1, keepdims=True)
            columns = [
                pool_index[pool]
                for pool in frame.iloc[positions]["table_id"].astype(str)
            ]
            weights[:, positions] = pool_weights[:, columns]
            fold_pool_keys.append(pools)
            fold_pool_weights.append(pool_weights)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise RuntimeError("Bayesian cluster weights are not positive and finite")
        for outer_fold in range(OUTER_FOLDS):
            fold_mask = folds == outer_fold
            for family in FAMILIES:
                cell = fold_mask & (np.asarray(families) == family)
                if not cell.any() or np.any(weights[:, cell].sum(axis=1) <= 0):
                    raise PreregistrationError("a view/fold/family denominator is empty")
        return cls(
            view=view,
            query_keys=tuple(frame["pair_id"].astype(str)),
            pool_keys=tuple(frame["table_id"].astype(str)),
            outer_folds=folds,
            families=families,
            query_weights=weights,
            fold_pool_keys=tuple(fold_pool_keys),
            fold_pool_weights=tuple(fold_pool_weights),
        )

    def _validate_delta_frame(self, deltas: pd.DataFrame) -> np.ndarray:
        required = {"pair_id", "table_id", "outer_fold", "behavior_family", "delta"}
        missing = sorted(required - set(deltas.columns))
        if missing:
            raise PreregistrationError(f"paired-delta fields missing: {missing}")
        frame = deltas.reset_index(drop=True)
        if tuple(frame["pair_id"].astype(str)) != self.query_keys:
            raise PreregistrationError("paired-delta query order or membership changed")
        if tuple(frame["table_id"].astype(str)) != self.pool_keys:
            raise PreregistrationError("paired-delta pool membership changed")
        if not np.array_equal(frame["outer_fold"].to_numpy(int), self.outer_folds):
            raise PreregistrationError("paired-delta outer folds changed")
        if tuple(frame["behavior_family"].astype(str)) != self.families:
            raise PreregistrationError("paired-delta family membership changed")
        values = pd.to_numeric(frame["delta"], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise PreregistrationError("paired delta is non-finite")
        return values

    def summarize(self, deltas: pd.DataFrame) -> dict[str, object]:
        """Return point deltas and frozen sensitivity endpoints, never a CI."""

        values = self._validate_delta_frame(deltas)
        family_array = np.asarray(self.families)
        numerator = self.query_weights @ values
        overall_draws = numerator / self.query_weights.sum(axis=1)
        family_draws: dict[str, np.ndarray] = {}
        family_point: dict[str, float] = {}
        family_intervals: dict[str, dict[str, float]] = {}
        for family in FAMILIES:
            mask = family_array == family
            draws = (self.query_weights[:, mask] @ values[mask]) / self.query_weights[
                :, mask
            ].sum(axis=1)
            family_draws[family] = draws
            family_point[family] = float(values[mask].mean())
            lower, upper = np.quantile(draws, [0.05, 0.95], method="linear")
            family_intervals[family] = {
                "sensitivity_lower_05": float(lower),
                "sensitivity_upper_95": float(upper),
            }
        macro_draws = np.mean(np.stack(list(family_draws.values())), axis=0)
        overall_lower, overall_upper = np.quantile(
            overall_draws, [0.05, 0.95], method="linear"
        )
        macro_lower, macro_upper = np.quantile(
            macro_draws, [0.05, 0.95], method="linear"
        )
        fold_summary: dict[str, dict[str, object]] = {}
        fold_point: dict[str, float] = {}
        for fold in range(OUTER_FOLDS):
            fold_mask = self.outer_folds == fold
            fold_draws = (
                self.query_weights[:, fold_mask] @ values[fold_mask]
            ) / self.query_weights[:, fold_mask].sum(axis=1)
            fold_lower, fold_upper = np.quantile(
                fold_draws, [0.05, 0.95], method="linear"
            )
            fold_point[str(fold)] = float(values[fold_mask].mean())
            fold_families: dict[str, dict[str, float]] = {}
            for family in FAMILIES:
                cell = fold_mask & (family_array == family)
                cell_draws = (
                    self.query_weights[:, cell] @ values[cell]
                ) / self.query_weights[:, cell].sum(axis=1)
                cell_lower, cell_upper = np.quantile(
                    cell_draws, [0.05, 0.95], method="linear"
                )
                fold_families[family] = {
                    "point_delta": float(values[cell].mean()),
                    "sensitivity_lower_05": float(cell_lower),
                    "sensitivity_upper_95": float(cell_upper),
                }
            fold_summary[str(fold)] = {
                "query_weighted_overall": {
                    "point_delta": fold_point[str(fold)],
                    "sensitivity_lower_05": float(fold_lower),
                    "sensitivity_upper_95": float(fold_upper),
                },
                "family_weighted_mean": fold_families,
            }
        return {
            "interval_name": SENSITIVITY_INTERVAL_NAME,
            "interval_semantics": (
                "paired_pool_weight_sensitivity_not_confidence_or_coverage"
            ),
            "confidence_claimed": False,
            "credible_claimed": False,
            "significance_claimed": False,
            "posterior_probability_claimed": False,
            "coverage_claimed": False,
            "paired_query_level_delta": {
                "formed_before_weighting": True,
                "separate_method_means_subtracted": False,
                "queries": int(len(values)),
            },
            "query_weighted_overall": {
                "point_delta": float(values.mean()),
                "sensitivity_lower_05": float(overall_lower),
                "sensitivity_upper_95": float(overall_upper),
            },
            "family_weighted_mean": {
                family: {
                    "point_delta": family_point[family],
                    **family_intervals[family],
                }
                for family in FAMILIES
            },
            "unweighted_macro_family_mean": {
                "point_delta": float(np.mean(list(family_point.values()))),
                "sensitivity_lower_05": float(macro_lower),
                "sensitivity_upper_95": float(macro_upper),
            },
            "outer_fold_point_delta": fold_point,
            "outer_fold_sensitivity": fold_summary,
            "draws": BOOTSTRAP_DRAWS,
            "quantiles": [0.05, 0.95],
            "quantile_method": "linear",
        }


def frozen_pool_weight_stream_sha256(
    plans: Mapping[str, BayesianClusterWeightPlan],
    *,
    include_headers: bool = True,
) -> str:
    """Reproduce the historical c833... normalized pool-weight byte stream.

    Serialization is full, early_to_late, late_to_early; within each view it is
    outer folds 0..4. Each block begins with the UTF-8 view name and one raw
    fold byte (0x00..0x04), then its C-contiguous little-endian float64
    (5000, pool_count) matrix after row-wise mean-one normalization. Shape,
    separators, and pool IDs are deliberately absent because the historical
    receipt did not bind them; callers must separately verify frozen pool
    membership/counts. Setting include_headers false exists only to verify the
    frozen 08ee... no-header negative control.
    """

    if np.__version__ != WEIGHT_STREAM_NUMPY_VERSION:
        raise PreregistrationError(
            "NumPy version differs from the frozen weight-stream runtime"
        )
    digest = hashlib.sha256()
    for view in ("full", "early_to_late", "late_to_early"):
        if view not in plans or plans[view].view != view:
            raise PreregistrationError(f"missing or mismatched weight plan: {view}")
        plan = plans[view]
        if len(plan.fold_pool_weights) != OUTER_FOLDS:
            raise PreregistrationError("weight plan does not contain five folds")
        for outer_fold, values in enumerate(plan.fold_pool_weights):
            expected_columns = len(plan.fold_pool_keys[outer_fold])
            if values.shape != (BOOTSTRAP_DRAWS, expected_columns):
                raise PreregistrationError("pool-weight matrix shape changed")
            if include_headers:
                digest.update(view.encode("utf-8"))
                digest.update(bytes([outer_fold]))
            digest.update(
                np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
            )
    return digest.hexdigest()


def paired_delta_frame(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
) -> pd.DataFrame:
    """Pair query AP before weighting; never subtract separate weighted means."""

    keys = ["pair_id", "table_id", "outer_fold", "behavior_family"]
    for frame in (method_a, method_b):
        if any(column not in frame for column in (*keys, "query_ap5")):
            raise PreregistrationError("query AP frame is incomplete")
        if frame["pair_id"].astype(str).duplicated().any():
            raise PreregistrationError("query AP frame has duplicate pairs")
    joined = method_a[keys + ["query_ap5"]].merge(
        method_b[keys + ["query_ap5"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_a", "_b"),
    )
    if not (joined["_merge"] == "both").all():
        raise PreregistrationError("paired methods do not share an identical query set")
    joined["delta"] = joined["query_ap5_a"] - joined["query_ap5_b"]
    return joined[keys + ["delta"]]
