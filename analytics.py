from __future__ import annotations

import numpy as np
import pandas as pd


def build_player_dashboard_table(player_season: pd.DataFrame) -> pd.DataFrame:
    frame = player_season.copy()
    frame["batting_average"] = frame["batting_average"].round(2)
    frame["strike_rate"] = frame["strike_rate"].round(2)
    frame["economy"] = frame["economy"].round(2)
    return frame.sort_values(["season", "batting_runs"], ascending=[True, False])


def build_venue_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    inning_scores = (
        deliveries.groupby(["match_id", "inning"], as_index=False)["total_runs"].sum()
        .pivot(index="match_id", columns="inning", values="total_runs")
        .reset_index()
        .rename(columns={1: "first_innings_score", 2: "second_innings_score"})
    )
    base = matches[["match_id", "venue", "winner", "team1", "team2"]].merge(inning_scores, on="match_id", how="left")
    second_batting = deliveries[deliveries["inning"] == 2][["match_id", "batting_team"]].drop_duplicates()
    first_batting = deliveries[deliveries["inning"] == 1][["match_id", "batting_team"]].drop_duplicates()
    base = base.merge(first_batting.rename(columns={"batting_team": "first_batting_team"}), on="match_id", how="left")
    base = base.merge(second_batting.rename(columns={"batting_team": "second_batting_team"}), on="match_id", how="left")
    base["chasing_won"] = (base["winner"] == base["second_batting_team"]).astype(int)
    base["defending_won"] = (base["winner"] == base["first_batting_team"]).astype(int)

    venue = (
        base.groupby("venue", as_index=False)
        .agg(
            matches=("match_id", "count"),
            avg_first_innings_score=("first_innings_score", "mean"),
            avg_second_innings_score=("second_innings_score", "mean"),
            chase_win_rate=("chasing_won", "mean"),
            defend_win_rate=("defending_won", "mean"),
        )
        .sort_values("matches", ascending=False)
    )
    venue["avg_first_innings_score"] = venue["avg_first_innings_score"].round(2)
    venue["avg_second_innings_score"] = venue["avg_second_innings_score"].round(2)
    venue["chase_win_rate"] = (venue["chase_win_rate"] * 100).round(2)
    venue["defend_win_rate"] = (venue["defend_win_rate"] * 100).round(2)
    return venue


def build_head_to_head(matches: pd.DataFrame, deliveries: pd.DataFrame, team_a: str, team_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b))
        | ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        subset.groupby("winner", as_index=False)
        .agg(wins=("match_id", "count"))
        .sort_values("wins", ascending=False)
    )

    relevant = deliveries[deliveries["match_id"].isin(subset["match_id"])].copy()
    batting = (
        relevant.groupby("batter", as_index=False)
        .agg(runs=("batsman_runs", "sum"), balls=("ball", "count"))
        .sort_values("runs", ascending=False)
        .head(10)
    )
    batting["role"] = "Batter"
    batting = batting.rename(columns={"batter": "player"})

    bowling = (
        relevant.groupby("bowler", as_index=False)
        .agg(wickets=("is_wicket", "sum"), runs_conceded=("total_runs", "sum"))
        .sort_values(["wickets", "runs_conceded"], ascending=[False, True])
        .head(10)
    )
    bowling["role"] = "Bowler"
    bowling = bowling.rename(columns={"bowler": "player"})
    top = pd.concat([batting, bowling], ignore_index=True, sort=False).fillna(0)
    return summary, top


def leaderboard_tables(player_season: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batting = (
        player_season.groupby("player", as_index=False)
        .agg(runs=("batting_runs", "sum"), average=("batting_average", "mean"), strike_rate=("strike_rate", "mean"))
        .sort_values(["runs", "strike_rate"], ascending=[False, False])
        .head(15)
    )
    bowling = (
        player_season.groupby("player", as_index=False)
        .agg(wickets=("bowling_wickets", "sum"), economy=("economy", "mean"))
        .query("wickets > 0")
        .sort_values(["wickets", "economy"], ascending=[False, True])
        .head(15)
    )
    batting[["average", "strike_rate"]] = batting[["average", "strike_rate"]].round(2)
    bowling["economy"] = bowling["economy"].round(2)
    return batting, bowling
