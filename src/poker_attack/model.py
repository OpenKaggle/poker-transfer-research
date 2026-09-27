"""Group-held-out pair and behavior models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .metric import average_precision
from .schema import TARGET_BEHAVIORS


@dataclass
class ModelResult:
    predictions: pd.DataFrame
    diagnostics: dict[str, object]


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    # Player and pair identifiers may be numeric in future releases, but are
    # explicitly prohibited as signals. shared_hands is gameplay exposure and
    # remains eligible.
    exclude = {"label", "pair_id", "player_1", "player_2", "table_id"}
    return [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in exclude and not column.startswith("account_age_days_")
    ]


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)


def _new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=280,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=2.5,
        early_stopping=False,
        random_state=seed,
    )


def train_pair_models(
    development_features: pd.DataFrame,
    labels: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    *,
    seed: int = 7401,
) -> ModelResult:
    labelled = development_features.merge(
        labels[["pair_id", "label", "behavior_family"]], on="pair_id", how="inner", validate="one_to_one"
    )
    labelled["label"] = pd.to_numeric(labelled["label"], errors="raise").astype(int)
    feature_columns = _numeric_columns(labelled.drop(columns=["table_id"], errors="ignore"))
    train_x = _matrix(labelled, feature_columns)
    eval_x = _matrix(evaluation_features, feature_columns)
    y = labelled["label"].to_numpy(dtype=int)
    groups = labelled["table_id"].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    folds = min(5, len(unique_groups))
    oof = np.zeros(len(labelled), dtype=float)
    risk_models: list[HistGradientBoostingClassifier] = []
    if folds >= 2 and len(np.unique(y)) == 2:
        for fold, (train_idx, valid_idx) in enumerate(GroupKFold(folds).split(train_x, y, groups)):
            model = _new_model(seed + fold)
            model.fit(train_x[train_idx], y[train_idx])
            oof[valid_idx] = model.predict_proba(train_x[valid_idx])[:, 1]
            risk_models.append(model)
    else:
        model = _new_model(seed)
        model.fit(train_x, y)
        oof[:] = model.predict_proba(train_x)[:, 1]
        risk_models.append(model)

    eval_risk = np.mean([model.predict_proba(eval_x)[:, 1] for model in risk_models], axis=0)

    positives = labelled[labelled["label"] == 1].copy()
    behavior_y = positives["behavior_family"].astype(str)
    behavior_columns = list(TARGET_BEHAVIORS)
    if len(set(behavior_y)) >= 2:
        behavior_model = _new_model(seed + 91)
        behavior_model.fit(_matrix(positives, feature_columns), behavior_y)
        raw = behavior_model.predict_proba(eval_x)
        behavior_probability = np.zeros((len(evaluation_features), len(behavior_columns)), dtype=float)
        for source_index, family in enumerate(behavior_model.classes_):
            if family in behavior_columns:
                behavior_probability[:, behavior_columns.index(family)] = raw[:, source_index]
    else:
        behavior_probability = np.full((len(evaluation_features), len(behavior_columns)), 1 / len(behavior_columns))

    family_index = behavior_probability.argmax(axis=1)
    predicted_family = np.asarray(behavior_columns, dtype=object)[family_index]

    # Only the strongest-risk tail receives a behavior label.  This makes the
    # behavior AP scores sparse while preserving Pair AP for every row.
    labelled_rate = float(y.mean()) if len(y) else 0.01
    active_fraction = float(np.clip(labelled_rate * 0.20, 0.005, 0.05))
    active_count = max(1, int(round(len(eval_risk) * active_fraction)))
    active = np.zeros(len(eval_risk), dtype=bool)
    active[np.argsort(-eval_risk, kind="stable")[:active_count]] = True
    predicted_behavior = np.where(active, predicted_family, "none")

    predictions = pd.DataFrame(
        {
            "pair_id": evaluation_features["pair_id"].to_numpy(),
            "risk_score": np.clip(eval_risk, 0, 1),
            "predicted_behavior": predicted_behavior,
            # Evidence is scored for every true positive, including positives
            # below the sparse behavior-output cutoff. Keep the best family for
            # the evidence ranker without exposing this helper in submission.csv.
            "evidence_behavior": predicted_family,
        }
    )
    diagnostics: dict[str, object] = {
        "n_labelled_pairs": int(len(labelled)),
        "n_positive_pairs": int(y.sum()),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "cv_pair_ap": float(average_precision(y, oof)),
        "active_behavior_fraction": active_fraction,
        "predicted_behavior_counts": predictions["predicted_behavior"].value_counts().to_dict(),
    }
    return ModelResult(predictions=predictions, diagnostics=diagnostics)
