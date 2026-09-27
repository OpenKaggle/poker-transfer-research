"""Memory-conscious poker activity feature engineering."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ACTION_FEATURES = (
    "n_actions",
    "aggressive_actions",
    "raises",
    "bets",
    "calls",
    "checks",
    "folds",
    "all_ins",
    "postflop_checks",
    "facing_pressure",
    "amount_bb_sum",
    "amount_bb_max",
    "to_call_bb_sum",
    "to_call_bb_max",
    "amount_pot_ratio_max",
)


def pair_key(player_1: pd.Series, player_2: pd.Series) -> pd.Series:
    """Stable canonical key independent of the pair's submitted orientation."""

    a = player_1.astype(str)
    b = player_2.astype(str)
    low = a.where(a <= b, b)
    high = b.where(a <= b, a)
    return low + "\x1f" + high


def card_strength(card_1: pd.Series, card_2: pd.Series) -> np.ndarray:
    rank_map = {str(i): i for i in range(2, 11)} | {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    c1 = card_1.fillna("").astype(str).str.upper()
    c2 = card_2.fillna("").astype(str).str.upper()
    r1 = c1.str[:-1].map(rank_map).fillna(0).to_numpy(dtype=float)
    r2 = c2.str[:-1].map(rank_map).fillna(0).to_numpy(dtype=float)
    paired = r1 == r2
    suited = c1.str[-1:].to_numpy() == c2.str[-1:].to_numpy()
    connected = np.maximum(0, 4 - np.abs(r1 - r2)) / 4
    return np.maximum(r1, r2) + 7 * paired + 1.5 * suited + connected


def aggregate_actions_parquet(
    path: str | Path,
    hands: pd.DataFrame,
    *,
    batch_size: int = 1_000_000,
) -> pd.DataFrame:
    """Aggregate 18M ordered actions in bounded-memory batches.

    All features here are associative counts/sums/maxima, so a hand split across
    parquet batches is combined exactly in a second aggregation.
    """

    hand_index = hands.set_index("hand_id")
    blind_map = hand_index["big_blind"]
    batches: list[pd.DataFrame] = []
    columns = [
        "hand_id",
        "player_id",
        "street",
        "action",
        "amount",
        "to_call",
        "pot_before",
    ]
    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = record_batch.to_pandas()
        blind = frame["hand_id"].map(blind_map).replace(0, np.nan).astype(float)
        action = frame["action"].astype(str).str.lower()
        street = frame["street"].astype(str).str.lower()
        amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
        to_call = pd.to_numeric(frame["to_call"], errors="coerce").fillna(0.0)
        pot = pd.to_numeric(frame["pot_before"], errors="coerce").fillna(0.0)
        aggressive = action.isin(["bet", "raise"]) | ((action == "all_in") & (amount > to_call))
        work = pd.DataFrame(
            {
                "hand_id": frame["hand_id"],
                "player_id": frame["player_id"],
                "n_actions": 1,
                "aggressive_actions": aggressive.astype(np.int8),
                "raises": (action == "raise").astype(np.int8),
                "bets": (action == "bet").astype(np.int8),
                "calls": (action == "call").astype(np.int8),
                "checks": (action == "check").astype(np.int8),
                "folds": (action == "fold").astype(np.int8),
                "all_ins": (action == "all_in").astype(np.int8),
                "postflop_checks": ((street != "preflop") & (action == "check")).astype(np.int8),
                "facing_pressure": (to_call > 0).astype(np.int8),
                "amount_bb_sum": (amount / blind).replace([np.inf, -np.inf], np.nan).fillna(0),
                "amount_bb_max": (amount / blind).replace([np.inf, -np.inf], np.nan).fillna(0),
                "to_call_bb_sum": (to_call / blind).replace([np.inf, -np.inf], np.nan).fillna(0),
                "to_call_bb_max": (to_call / blind).replace([np.inf, -np.inf], np.nan).fillna(0),
                "amount_pot_ratio_max": (amount / np.maximum(pot, blind)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 20),
            }
        )
        agg_spec = {name: "sum" for name in ACTION_FEATURES}
        for name in ("amount_bb_max", "to_call_bb_max", "amount_pot_ratio_max"):
            agg_spec[name] = "max"
        batches.append(work.groupby(["hand_id", "player_id"], sort=False).agg(agg_spec).reset_index())

    partial = pd.concat(batches, ignore_index=True)
    second_spec = {name: "sum" for name in ACTION_FEATURES}
    for name in ("amount_bb_max", "to_call_bb_max", "amount_pot_ratio_max"):
        second_spec[name] = "max"
    return partial.groupby(["hand_id", "player_id"], sort=False).agg(second_spec).reset_index()


def build_player_hands(
    seats: pd.DataFrame,
    hands: pd.DataFrame,
    action_features: pd.DataFrame,
) -> pd.DataFrame:
    hand_cols = ["hand_id", "phase", "table_id", "started_at", "big_blind", "final_pot"]
    frame = seats.merge(hands.loc[:, hand_cols], on="hand_id", how="left", validate="many_to_one")
    frame = frame.merge(action_features, on=["hand_id", "player_id"], how="left", validate="one_to_one")
    for column in ACTION_FEATURES:
        frame[column] = frame[column].fillna(0.0)
    blind = pd.to_numeric(frame["big_blind"], errors="coerce").replace(0, np.nan)
    frame["stack_bb"] = pd.to_numeric(frame["starting_stack"], errors="coerce") / blind
    frame["contribution_bb"] = pd.to_numeric(frame["total_contribution"], errors="coerce") / blind
    frame["net_bb"] = pd.to_numeric(frame["net_chips"], errors="coerce") / blind
    frame["pot_bb"] = pd.to_numeric(frame["final_pot"], errors="coerce") / blind
    frame["hole_strength"] = card_strength(frame["hole_card_1"], frame["hole_card_2"])

    hands_progress = hands.loc[:, ["hand_id", "table_id", "phase", "started_at"]].copy()
    hands_progress["phase_progress"] = hands_progress.groupby(
        ["table_id", "phase"], sort=False
    )["started_at"].rank(method="first", pct=True)
    frame = frame.drop(columns=["phase_progress"], errors="ignore").merge(
        hands_progress[["hand_id", "phase_progress"]], on="hand_id", how="left"
    )

    baseline_columns = [
        "aggressive_actions",
        "calls",
        "checks",
        "folds",
        "facing_pressure",
        "contribution_bb",
        "net_bb",
    ]
    baselines = (
        frame.groupby(["phase", "player_id"], sort=False)[baseline_columns]
        .mean()
        .add_prefix("base_")
        .reset_index()
    )
    return frame.merge(baselines, on=["phase", "player_id"], how="left", validate="many_to_one")


def _pair_membership(
    phase_seats: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Map requested pairs to their shared hands without a 72M-row self join."""

    seats = phase_seats.loc[:, ["hand_id", "seat_no", "player_id"]].sort_values(
        ["hand_id", "seat_no"], kind="stable"
    )
    counts = seats.groupby("hand_id", sort=False).size()
    if not (counts == 6).all():
        bad = int((counts != 6).sum())
        raise ValueError(f"expected six seats per hand; found {bad} malformed hands")

    hand_ids = seats["hand_id"].to_numpy().reshape(-1, 6)[:, 0]
    players = seats["player_id"].to_numpy().reshape(-1, 6)
    lookup = pairs[["pair_id", "player_1", "player_2"]].copy()
    lookup["_pair_key"] = pair_key(lookup["player_1"], lookup["player_2"])
    if lookup["_pair_key"].duplicated().any():
        raise ValueError("pair list contains the same player pair more than once")
    lookup = lookup[["_pair_key", "pair_id", "player_1", "player_2"]]

    matches: list[pd.DataFrame] = []
    for i, j in combinations(range(6), 2):
        candidate = pd.DataFrame(
            {"hand_id": hand_ids, "seat_player_a": players[:, i], "seat_player_b": players[:, j]}
        )
        candidate["_pair_key"] = pair_key(candidate["seat_player_a"], candidate["seat_player_b"])
        matched = candidate.merge(lookup, on="_pair_key", how="inner", validate="many_to_one")
        if not matched.empty:
            matches.append(matched.drop(columns="_pair_key"))
    if not matches:
        return pd.DataFrame(columns=["pair_id", "hand_id", "player_1", "player_2"])
    result = pd.concat(matches, ignore_index=True)
    return result[["pair_id", "hand_id", "player_1", "player_2"]].sort_values(
        ["pair_id", "hand_id"], kind="stable"
    )


def build_pair_hands(player_hands: pd.DataFrame, pairs: pd.DataFrame, phase: str) -> pd.DataFrame:
    phase_players = player_hands[player_hands["phase"].astype(str) == phase].copy()
    membership = _pair_membership(phase_players, pairs)
    if membership.empty:
        return membership

    player_columns = [
        "hand_id",
        "player_id",
        "table_id",
        "phase_progress",
        "pot_bb",
        "stack_bb",
        "contribution_bb",
        "net_bb",
        "folded",
        "went_to_showdown",
        "won_share",
        "hole_strength",
        *ACTION_FEATURES,
        "base_aggressive_actions",
        "base_calls",
        "base_checks",
        "base_folds",
        "base_facing_pressure",
        "base_contribution_bb",
        "base_net_bb",
    ]
    base = phase_players.loc[:, player_columns]
    left = base.rename(columns={c: f"{c}_1" for c in base.columns if c not in {"hand_id"}})
    right = base.rename(columns={c: f"{c}_2" for c in base.columns if c not in {"hand_id"}})
    frame = membership.merge(
        left,
        left_on=["hand_id", "player_1"],
        right_on=["hand_id", "player_id_1"],
        how="left",
        validate="many_to_one",
    ).drop(columns="player_id_1")
    frame = frame.merge(
        right,
        left_on=["hand_id", "player_2"],
        right_on=["hand_id", "player_id_2"],
        how="left",
        validate="many_to_one",
    ).drop(columns="player_id_2")

    frame["table_id"] = frame["table_id_1"]
    frame["phase_progress"] = frame["phase_progress_1"]
    frame["pair_contribution_bb"] = frame["contribution_bb_1"] + frame["contribution_bb_2"]
    frame["contribution_gap_bb"] = (frame["contribution_bb_1"] - frame["contribution_bb_2"]).abs()
    frame["net_gap_bb"] = (frame["net_bb_1"] - frame["net_bb_2"]).abs()
    frame["max_win_bb"] = frame[["net_bb_1", "net_bb_2"]].max(axis=1).clip(lower=0)
    frame["max_loss_bb"] = -frame[["net_bb_1", "net_bb_2"]].min(axis=1).clip(upper=0)
    frame["transfer_1_to_2_bb"] = np.minimum((-frame["net_bb_1"]).clip(lower=0), frame["net_bb_2"].clip(lower=0))
    frame["transfer_2_to_1_bb"] = np.minimum((-frame["net_bb_2"]).clip(lower=0), frame["net_bb_1"].clip(lower=0))
    frame["transfer_any_bb"] = frame[["transfer_1_to_2_bb", "transfer_2_to_1_bb"]].max(axis=1)
    frame["both_showdown"] = (frame["went_to_showdown_1"].astype(bool) & frame["went_to_showdown_2"].astype(bool)).astype(np.int8)
    frame["one_folded"] = (frame["folded_1"].astype(bool) ^ frame["folded_2"].astype(bool)).astype(np.int8)
    frame["pair_aggression"] = frame["aggressive_actions_1"] + frame["aggressive_actions_2"]
    frame["pair_passive"] = frame["calls_1"] + frame["calls_2"] + frame["checks_1"] + frame["checks_2"]
    frame["expected_aggression"] = frame["base_aggressive_actions_1"] + frame["base_aggressive_actions_2"]
    frame["expected_passive"] = frame["base_calls_1"] + frame["base_calls_2"] + frame["base_checks_1"] + frame["base_checks_2"]
    frame["aggression_residual"] = frame["pair_aggression"] - frame["expected_aggression"]
    frame["passive_residual"] = frame["pair_passive"] - frame["expected_passive"]
    frame["pair_pot_share"] = frame["pair_contribution_bb"] / (frame["pot_bb_1"] + 1e-3)
    frame["pair_raises"] = frame["raises_1"] + frame["raises_2"]
    frame["pair_folds"] = frame["folds_1"] + frame["folds_2"]
    frame["max_hole_strength"] = frame[["hole_strength_1", "hole_strength_2"]].max(axis=1)

    frame["directed_signal"] = (
        np.log1p(frame["transfer_any_bb"].clip(lower=0))
        + 0.25 * np.log1p(frame["net_gap_bb"].clip(lower=0))
        + 0.15 * frame["both_showdown"]
    )
    frame["soft_signal"] = (
        np.log1p(frame["pair_passive"].clip(lower=0))
        + 0.65 * frame["both_showdown"]
        - 0.55 * np.log1p(frame["pair_aggression"].clip(lower=0))
        + 0.35 * frame["passive_residual"]
        - 0.45 * frame["aggression_residual"]
    )
    frame["isolation_signal"] = (
        np.log1p(frame["pair_aggression"].clip(lower=0))
        + 0.35 * np.log1p(frame["pair_contribution_bb"].clip(lower=0))
        + 0.25 * frame["pair_pot_share"].clip(0, 1)
        - 0.20 * frame["both_showdown"]
    )
    return frame


def aggregate_pair_features(pair_hands: pd.DataFrame, pairs: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    identity = {"pair_id", "hand_id", "player_1", "player_2", "table_id_1", "table_id_2"}
    numeric = [
        c
        for c in pair_hands.select_dtypes(include=[np.number, "bool"]).columns
        if c not in identity and not c.endswith("_1") and not c.endswith("_2")
    ]
    group = pair_hands.groupby("pair_id", sort=False)
    mean = group[numeric].mean().add_suffix("__mean")
    maximum = group[numeric].max().add_suffix("__max")
    p95 = group[numeric].quantile(0.95).add_suffix("__p95")
    stats = pd.concat([mean, maximum, p95], axis=1)
    stats["shared_hands_calc"] = group.size()

    flows = group[["transfer_1_to_2_bb", "transfer_2_to_1_bb"]].sum()
    stats["flow_asymmetry"] = (
        (flows["transfer_1_to_2_bb"] - flows["transfer_2_to_1_bb"]).abs()
        / (flows.sum(axis=1) + 1e-3)
    )
    stats["dominant_flow_bb"] = flows.max(axis=1)

    episodic = pair_hands[["pair_id", "phase_progress", "directed_signal", "soft_signal", "isolation_signal"]].copy()
    episodic["time_bin"] = np.minimum(3, np.floor(episodic["phase_progress"].fillna(0) * 4).astype(int))
    time_means = episodic.groupby(["pair_id", "time_bin"], sort=False)[
        ["directed_signal", "soft_signal", "isolation_signal"]
    ].mean()
    wide = time_means.unstack("time_bin")
    wide.columns = [f"{signal}__t{time_bin}" for signal, time_bin in wide.columns]
    stats = stats.join(wide)
    for signal in ("directed_signal", "soft_signal", "isolation_signal"):
        columns = [c for c in wide.columns if c.startswith(f"{signal}__t")]
        if columns:
            stats[f"{signal}__temporal_range"] = stats[columns].max(axis=1) - stats[columns].min(axis=1)

    meta = players.set_index("player_id")
    base_columns = [
        column
        for column in ("pair_id", "player_1", "player_2", "shared_hands")
        if column in pairs.columns
    ]
    result = pairs.loc[:, base_columns].copy().set_index("pair_id").join(stats, how="left")
    result["table_id"] = group["table_id"].first()
    for suffix, id_column in (("1", "player_1"), ("2", "player_2")):
        ids = result[id_column]
        result[f"account_age_days_{suffix}"] = ids.map(meta["account_age_days"])
    result["account_age_gap"] = (result["account_age_days_1"] - result["account_age_days_2"]).abs()
    for column in ("experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"):
        left = result["player_1"].map(meta[column])
        right = result["player_2"].map(meta[column])
        result[f"same_{column}"] = (left == right).astype(np.int8)
    return result.reset_index()


def rank_evidence(pair_hands: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank valid shared hands using the predicted behavior's observable signal."""

    work = pair_hands.merge(
        predictions[["pair_id", "predicted_behavior"]], on="pair_id", how="inner", validate="many_to_one"
    )
    work["evidence_score"] = np.select(
        [
            work["predicted_behavior"] == "directed_transfer",
            work["predicted_behavior"] == "soft_play",
            work["predicted_behavior"] == "coordinated_isolation",
        ],
        [work["directed_signal"], work["soft_signal"], work["isolation_signal"]],
        default=work[["directed_signal", "soft_signal", "isolation_signal"]].max(axis=1),
    )
    top = work.sort_values(
        ["pair_id", "evidence_score", "hand_id"], ascending=[True, False, True], kind="stable"
    ).groupby("pair_id", sort=False).head(5)
    top["rank"] = top.groupby("pair_id", sort=False).cumcount() + 1
    return top.pivot(index="pair_id", columns="rank", values="hand_id").rename(
        columns=lambda rank: f"evidence_hand_{rank}"
    ).reset_index()
