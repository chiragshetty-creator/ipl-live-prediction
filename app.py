from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ipl_analytics.analytics import build_head_to_head
from ipl_analytics.data import MODELS_DIR, PROCESSED_DIR, load_processed_csv, load_raw_datasets
from ipl_analytics.models import load_classifier, load_model_metrics, predict_win_probability


st.set_page_config(page_title="IPL Sports Analytics", layout="wide")


def ensure_assets() -> None:
    required = [
        MODELS_DIR / "xgboost.joblib",
        MODELS_DIR / "model_metrics.csv",
        PROCESSED_DIR / "player_dashboard.csv",
        PROCESSED_DIR / "venue_stats.csv",
    ]
    if all(path.exists() for path in required):
        return

    import runpy

    runpy.run_path(str(PROJECT_ROOT / "scripts" / "train_models.py"), run_name="__main__")


@st.cache_data
def load_data():
    ensure_assets()
    bundle = load_raw_datasets()
    player_dashboard = load_processed_csv("player_dashboard.csv")
    venue_stats = load_processed_csv("venue_stats.csv")
    batting_lb = load_processed_csv("batting_leaderboard.csv")
    bowling_lb = load_processed_csv("bowling_leaderboard.csv")
    player_clusters = pd.read_csv(MODELS_DIR / "player_clusters.csv")
    metrics = load_model_metrics()
    summary = json.loads((PROCESSED_DIR / "summary.json").read_text(encoding="utf-8"))
    return bundle, player_dashboard, venue_stats, batting_lb, bowling_lb, player_clusters, metrics, summary


@st.cache_resource
def load_primary_model():
    ensure_assets()
    return load_classifier("xgboost")


bundle, player_dashboard, venue_stats, batting_lb, bowling_lb, player_clusters, metrics, summary = load_data()
model = load_primary_model()
teams = sorted(pd.unique(pd.concat([bundle.matches["team1"], bundle.matches["team2"]], ignore_index=True)))
venues = sorted(bundle.matches["venue"].dropna().unique())
seasons = sorted(player_dashboard["season"].astype(int).unique())

st.title("IPL Sports Analytics Platform")
st.caption("Win prediction, player segmentation, venue intelligence, and team matchup analysis across 16+ IPL seasons.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Seasons", summary["seasons"])
col2.metric("Matches", summary["matches"])
col3.metric("Deliveries", summary["deliveries"])
col4.metric("Best Model", summary["best_model"])

tab1, tab2, tab3, tab4 = st.tabs(
    ["Match Simulator", "Player Analytics", "Venue Analytics", "Team Comparison"]
)

with tab1:
    st.subheader("Live Win Probability Simulator")
    left, right = st.columns([1, 1])
    with left:
        batting_team = st.selectbox("Batting Team", teams, index=0)
        bowling_team = st.selectbox("Bowling Team", [team for team in teams if team != batting_team], index=0)
        venue = st.selectbox("Venue", venues, index=0)
        target = st.number_input("Target", min_value=80, max_value=280, value=180, step=1)
    with right:
        runs_needed = st.number_input("Runs Needed", min_value=0, max_value=280, value=60, step=1)
        overs_remaining = st.slider("Overs Remaining", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
        wickets_remaining = st.slider("Wickets Remaining", min_value=0, max_value=10, value=6, step=1)
        current_score = max(target - runs_needed, 0)
        balls_remaining = int(round(overs_remaining * 6))
        current_overs = max((120 - balls_remaining) / 6.0, 0.1)
        current_run_rate = current_score / current_overs
        required_run_rate = (runs_needed / overs_remaining) if overs_remaining > 0 else 0.0

    payload = pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "target": target,
                "current_score": current_score,
                "runs_needed": runs_needed,
                "overs_remaining": overs_remaining,
                "balls_remaining": balls_remaining,
                "wickets_remaining": wickets_remaining,
                "current_run_rate": round(current_run_rate, 2),
                "required_run_rate": round(required_run_rate, 2),
            }
        ]
    )
    win_prob = float(predict_win_probability(model, payload)[0]) * 100
    lose_prob = 100 - win_prob

    m1, m2, m3 = st.columns(3)
    m1.metric("Win Probability", f"{win_prob:.1f}%")
    m2.metric("Loss Probability", f"{lose_prob:.1f}%")
    m3.metric("Runs Needed", runs_needed)

    gauge = px.bar_polar(
        pd.DataFrame({"result": ["Win", "Loss"], "probability": [win_prob, lose_prob]}),
        r="probability",
        theta="result",
        color="result",
        color_discrete_sequence=["#2ca02c", "#d62728"],
        template="plotly_white",
    )
    gauge.update_layout(height=420, showlegend=False)
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("Model Comparison")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Player Performance Dashboard")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    season_filter = filter_col1.selectbox("Season", seasons, index=len(seasons) - 1)
    cluster_filter = filter_col2.selectbox(
        "Cluster",
        ["All"] + sorted(player_clusters["cluster_name"].dropna().unique().tolist()),
        index=0,
    )
    player_options = sorted(player_dashboard["player"].unique().tolist())
    player_focus = filter_col3.selectbox("Player Trend", player_options, index=0)

    player_view = player_dashboard[player_dashboard["season"].astype(int) == int(season_filter)].copy()
    if cluster_filter != "All":
        players_in_cluster = set(player_clusters[player_clusters["cluster_name"] == cluster_filter]["player"])
        player_view = player_view[player_view["player"].isin(players_in_cluster)]

    st.dataframe(
        player_view[
            [
                "player",
                "batting_runs",
                "batting_average",
                "strike_rate",
                "bowling_wickets",
                "economy",
            ]
        ].sort_values("batting_runs", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    fig_runs = px.scatter(
        player_view,
        x="strike_rate",
        y="batting_average",
        size="batting_runs",
        color="bowling_wickets",
        hover_name="player",
        title=f"Batting profile - {season_filter}",
        template="plotly_white",
    )
    st.plotly_chart(fig_runs, use_container_width=True)

    trend_frame = player_dashboard[player_dashboard["player"] == player_focus].copy().sort_values("season")
    trend_long = trend_frame.melt(
        id_vars=["season", "player"],
        value_vars=["batting_average", "strike_rate", "economy"],
        var_name="metric",
        value_name="value",
    )
    fig_trend = px.line(
        trend_long,
        x="season",
        y="value",
        color="metric",
        markers=True,
        title=f"Season trend - {player_focus}",
        template="plotly_white",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Player Segmentation")
    fig_clusters = px.scatter(
        player_clusters,
        x="strike_rate",
        y="economy",
        color="cluster_name",
        size="batting_runs",
        hover_name="player",
        template="plotly_white",
    )
    st.plotly_chart(fig_clusters, use_container_width=True)

with tab3:
    st.subheader("Venue Performance Heatmap")
    heatmap_df = venue_stats.melt(
        id_vars=["venue", "matches"],
        value_vars=["avg_first_innings_score", "chase_win_rate", "defend_win_rate"],
        var_name="metric",
        value_name="value",
    )
    fig_heatmap = px.density_heatmap(
        heatmap_df,
        x="metric",
        y="venue",
        z="value",
        histfunc="avg",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.dataframe(venue_stats, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Team Head-to-Head Comparison")
    hh1, hh2 = st.columns(2)
    team_a = hh1.selectbox("Team A", teams, index=0, key="team_a")
    team_b = hh2.selectbox("Team B", [team for team in teams if team != team_a], index=0, key="team_b")

    summary_df, top_df = build_head_to_head(bundle.matches, bundle.deliveries, team_a, team_b)
    if summary_df.empty:
        st.info("No historical meetings found for this team combination.")
    else:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.dataframe(top_df.sort_values(["role", "runs", "wickets"], ascending=[True, False, False]), use_container_width=True, hide_index=True)
        chart = px.bar(summary_df, x="winner", y="wins", color="winner", template="plotly_white")
        st.plotly_chart(chart, use_container_width=True)
