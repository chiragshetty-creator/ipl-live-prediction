from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_match_context_dataset(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    match_meta = matches[
        ["match_id", "season", "date", "venue", "team1", "team2", "winner", "toss_winner", "toss_decision"]
    ].copy()

    inning1 = deliveries[deliveries["inning"] == 1].groupby("match_id", as_index=False)["total_runs"].sum()
    inning1 = inning1.rename(columns={"total_runs": "first_innings_score"})

    second = deliveries[deliveries["inning"] == 2].copy()
    if second.empty:
        raise ValueError("No second innings data available for win-probability training.")

    over_offset = 0 if deliveries["over"].min() == 0 else 1
    second["ball_number"] = (second["over"] - over_offset) * 6 + second["ball"].clip(upper=6)
    second = second.sort_values(["match_id", "ball_number"])
    second["cum_runs"] = second.groupby("match_id")["total_runs"].cumsum()
    second["cum_wickets"] = second.groupby("match_id")["is_wicket"].cumsum()
    second = second.merge(inning1, on="match_id", how="left")
    second = second.merge(match_meta, on="match_id", how="left")

    second["target"] = second["first_innings_score"] + 1
    second["balls_remaining"] = 120 - second["ball_number"]
    second["overs_remaining"] = second["balls_remaining"] / 6.0
    second["wickets_remaining"] = 10 - second["cum_wickets"]
    second["runs_needed"] = second["target"] - second["cum_runs"]
    second["current_run_rate"] = np.where(
        second["ball_number"] > 0,
        second["cum_runs"] / (second["ball_number"] / 6.0),
        0.0,
    )
    second["required_run_rate"] = np.where(
        second["balls_remaining"] > 0,
        second["runs_needed"] / (second["balls_remaining"] / 6.0),
        0.0,
    )
    second["won"] = (second["winner"] == second["batting_team"]).astype(int)

    second = second[
        (second["balls_remaining"] >= 0)
        & (second["runs_needed"] >= 0)
        & (second["wickets_remaining"] >= 0)
    ].copy()

    columns = [
        "match_id",
        "season",
        "date",
        "venue",
        "batting_team",
        "bowling_team",
        "target",
        "cum_runs",
        "runs_needed",
        "overs_remaining",
        "balls_remaining",
        "wickets_remaining",
        "current_run_rate",
        "required_run_rate",
        "won",
    ]
    return second[columns].rename(columns={"cum_runs": "current_score"})


def build_match_baseline_dataset(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    inning_scores = (
        deliveries.groupby(["match_id", "inning"], as_index=False)["total_runs"].sum()
        .pivot(index="match_id", columns="inning", values="total_runs")
        .reset_index()
        .rename(columns={1: "first_innings_score", 2: "second_innings_score"})
    )
    baseline = matches[
        ["match_id", "season", "venue", "team1", "team2", "winner", "toss_winner", "toss_decision"]
    ].merge(inning_scores[["match_id", "first_innings_score"]], on="match_id", how="left")
    baseline = baseline.dropna(subset=["winner", "first_innings_score"]).copy()
    baseline["team1_won"] = (baseline["winner"] == baseline["team1"]).astype(int)
    baseline["toss_winner_is_team1"] = (baseline["toss_winner"] == baseline["team1"]).astype(int)
    return baseline[
        [
            "match_id",
            "season",
            "venue",
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "toss_winner_is_team1",
            "first_innings_score",
            "team1_won",
        ]
    ]


def build_player_season_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    meta = matches[["match_id", "season"]].copy()
    base = deliveries.merge(meta, on="match_id", how="left")

    batting = (
        base.groupby(["season", "batter"], as_index=False)
        .agg(
            batting_runs=("batsman_runs", "sum"),
            batting_balls=("ball", "count"),
            dismissals=("is_wicket", "sum"),
            batting_matches=("match_id", "nunique"),
        )
        .rename(columns={"batter": "player"})
    )
    batting["batting_average"] = batting["batting_runs"] / batting["dismissals"].replace(0, np.nan)
    batting["batting_average"] = batting["batting_average"].fillna(batting["batting_runs"])
    batting["strike_rate"] = np.where(
        batting["batting_balls"] > 0,
        batting["batting_runs"] * 100.0 / batting["batting_balls"],
        0.0,
    )

    bowling = (
        base.groupby(["season", "bowler"], as_index=False)
        .agg(
            balls_bowled=("ball", "count"),
            bowling_runs=("total_runs", "sum"),
            bowling_wickets=("is_wicket", "sum"),
            bowling_matches=("match_id", "nunique"),
        )
        .rename(columns={"bowler": "player"})
    )
    bowling["overs_bowled"] = bowling["balls_bowled"] / 6.0
    bowling["economy"] = np.where(
        bowling["overs_bowled"] > 0,
        bowling["bowling_runs"] / bowling["overs_bowled"],
        0.0,
    )
    bowling["bowling_average"] = bowling["bowling_runs"] / bowling["bowling_wickets"].replace(0, np.nan)
    bowling["bowling_average"] = bowling["bowling_average"].fillna(bowling["bowling_runs"])

    merged = batting.merge(bowling, on=["season", "player"], how="outer").fillna(0)
    numeric_cols = [col for col in merged.columns if col not in {"season", "player"}]
    merged[numeric_cols] = merged[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return merged


def build_player_cluster_dataset(player_season: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        player_season.groupby("player", as_index=False)
        .agg(
            seasons=("season", "nunique"),
            batting_runs=("batting_runs", "sum"),
            batting_balls=("batting_balls", "sum"),
            dismissals=("dismissals", "sum"),
            bowling_runs=("bowling_runs", "sum"),
            balls_bowled=("balls_bowled", "sum"),
            bowling_wickets=("bowling_wickets", "sum"),
            batting_matches=("batting_matches", "sum"),
            bowling_matches=("bowling_matches", "sum"),
        )
    )
    grouped["batting_average"] = grouped["batting_runs"] / grouped["dismissals"].replace(0, np.nan)
    grouped["batting_average"] = grouped["batting_average"].fillna(grouped["batting_runs"])
    grouped["strike_rate"] = np.where(
        grouped["batting_balls"] > 0,
        grouped["batting_runs"] * 100.0 / grouped["batting_balls"],
        0.0,
    )
    grouped["overs_bowled"] = grouped["balls_bowled"] / 6.0
    grouped["economy"] = np.where(
        grouped["overs_bowled"] > 0,
        grouped["bowling_runs"] / grouped["overs_bowled"],
        0.0,
    )
    grouped["wickets_per_match"] = np.where(
        grouped["bowling_matches"] > 0,
        grouped["bowling_wickets"] / grouped["bowling_matches"],
        0.0,
    )
    grouped["runs_per_match"] = np.where(
        grouped["batting_matches"] > 0,
        grouped["batting_runs"] / grouped["batting_matches"],
        0.0,
    )
    return grouped[
        [
            "player",
            "seasons",
            "batting_runs",
            "batting_average",
            "strike_rate",
            "runs_per_match",
            "bowling_wickets",
            "economy",
            "wickets_per_match",
        ]
    ]


def label_clusters(cluster_frame: pd.DataFrame, labels: Iterable[int]) -> pd.Series:
    cluster_frame = cluster_frame.copy()
    cluster_frame["cluster"] = list(labels)
    profile = cluster_frame.groupby("cluster").agg(
        strike_rate=("strike_rate", "mean"),
        batting_average=("batting_average", "mean"),
        economy=("economy", "mean"),
        wickets=("bowling_wickets", "mean"),
        runs=("batting_runs", "mean"),
    )
    ordered = {}
    for cluster_id, row in profile.iterrows():
        if row["wickets"] >= profile["wickets"].quantile(0.75) and row["economy"] <= profile["economy"].median():
            ordered[cluster_id] = "Economy Bowlers"
        elif row["strike_rate"] >= profile["strike_rate"].quantile(0.75):
            ordered[cluster_id] = "Power Hitters"
        elif row["batting_average"] >= profile["batting_average"].median() and row["strike_rate"] < profile["strike_rate"].median():
            ordered[cluster_id] = "Anchors"
        else:
            ordered[cluster_id] = "Finishers / All-Rounders"
    return cluster_frame["cluster"].map(ordered)
