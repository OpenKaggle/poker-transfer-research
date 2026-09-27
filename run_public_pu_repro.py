"""Execute the retained public PU notebook's modelling path against local data.

This runner deliberately executes selected source cells from the untouched
notebook instead of copying their contents into an untraceable derivative.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POKER = ROOT / "poker"
NOTEBOOK = (
    POKER
    / "reference"
    / "public_pu_ranker"
    / "poker-collusion-pu-aware-evidence-ranker.ipynb"
)
DATA = ROOT / "data" / "raw" / "poker_comp"
PREPARED = POKER / "work" / "public_pu_prepared_v2"
OUTPUT = ROOT / "submissions" / "public_pu_ranker_repro.csv"
REPORT = POKER / "work" / "public_pu_ranker_repro_report.json"
SOURCE_CELLS = [0, 2, 3, 29, 30, 31, 32, 34, 35, 36, 37, 39, 40]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__public_pu_repro__",
        "display": lambda *args, **kwargs: None,
    }
    replacements = {
        "/kaggle/input/competitions/detect-suspicious-value-transfers-in-poker": str(DATA),
        "/kaggle/working/prepared_v2": str(PREPARED),
        "/kaggle/working/submission.csv": str(OUTPUT),
    }
    for cell_index in SOURCE_CELLS:
        source = "".join(notebook["cells"][cell_index]["source"])
        for old, new in replacements.items():
            source = source.replace(old, new)
        print(f"PUBLIC_REPRO_CELL {cell_index} START", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell-{cell_index}", "exec"), namespace)
        print(f"PUBLIC_REPRO_CELL {cell_index} DONE", flush=True)

    sys.path.insert(0, str(POKER / "src"))
    from poker_attack.audit import validate_evidence_membership, validate_submission

    submission = pd.read_csv(
        OUTPUT, dtype={f"evidence_hand_{index}": str for index in range(1, 6)}
    )
    evaluation = pd.read_csv(DATA / "evaluation_pairs.csv")
    validate_submission(submission, evaluation["pair_id"])
    pair_hands = pd.read_parquet(
        POKER / "work" / "evaluation_pair_hands.parquet",
        columns=["pair_id", "hand_id"],
    )
    legality = validate_evidence_membership(submission, pair_hands)

    dev_pairs = namespace["dev_pairs_prep"]
    pu_rows = dev_pairs.filter(namespace["pl"].col("is_pu"))
    if pu_rows.filter(namespace["pl"].col("label").is_not_null()).height:
        raise ValueError("public PU reproduction persisted a ground-truth label on unknown rows")
    feature_columns = list(namespace["feature_cols"])
    prohibited_feature_columns = [
        column
        for column in feature_columns
        if column in {"pair_id", "player_1", "player_2", "table_id", "cv_group"}
        or column.endswith("_id")
    ]
    if prohibited_feature_columns:
        raise ValueError(f"prohibited identifier features: {prohibited_feature_columns}")

    report = {
        "source_notebook": str(NOTEBOOK),
        "source_notebook_sha256": sha256(NOTEBOOK),
        "source_kernel": "nomannic19/poker-collusion-pu-aware-evidence-ranker",
        "source_cells_executed": SOURCE_CELLS,
        "prepared_directory": str(PREPARED),
        "submission": str(OUTPUT),
        "submission_sha256": sha256(OUTPUT),
        "submission_rows": int(len(submission)),
        "risk_unique": int(submission["risk_score"].nunique()),
        "risk_min": float(submission["risk_score"].min()),
        "risk_max": float(submission["risk_score"].max()),
        "behavior_counts": submission["predicted_behavior"].value_counts().to_dict(),
        "feature_count": int(len(feature_columns)),
        "prohibited_feature_columns": prohibited_feature_columns,
        "confirmed_labelled_pair_ap": float(namespace["pair_ap_labeled"]),
        "pu_stress_pair_ap": float(namespace["pair_ap_stress"]),
        "confirmed_behavior_map": float(namespace["behavior_map_labeled"]),
        "pu_stress_behavior_map": float(namespace["behavior_map_stress"]),
        "positive_family_accuracy": float(namespace["family_accuracy"]),
        "selected_behavior_rate": float(namespace["behavior_positive_rate"]),
        "evidence_oof_map5": float(namespace["overall_evidence_map"]),
        "unknown_pairs": int(pu_rows.height),
        "unknown_ground_truth_labels_assigned": 0,
        "unknown_surrogate_role": "biased_PU_selection_background_weight_0.35",
        **legality,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
