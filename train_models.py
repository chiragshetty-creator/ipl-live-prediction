from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from ipl_analytics.analytics import build_player_dashboard_table, build_venue_stats, leaderboard_tables
from ipl_analytics.data import PROCESSED_DIR, dataset_summary, load_raw_datasets, raw_data_exists, save_processed_csv
from ipl_analytics.features import (
    build_match_baseline_dataset,
    build_match_context_dataset,
    build_player_cluster_dataset,
    build_player_season_features,
)
from ipl_analytics.models import save_artifacts, train_classifiers, train_player_clusters


def ensure_raw_data() -> None:
    if raw_data_exists():
        return
    from generate_demo_data import generate_demo_data

    matches, deliveries = generate_demo_data()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(raw_dir / "matches.csv", index=False)
    deliveries.to_csv(raw_dir / "deliveries.csv", index=False)


def main() -> None:
    ensure_raw_data()
    bundle = load_raw_datasets()
    context = build_match_context_dataset(bundle.matches, bundle.deliveries)
    baseline = build_match_baseline_dataset(bundle.matches, bundle.deliveries)
    player_season = build_player_season_features(bundle.matches, bundle.deliveries)
    cluster_input = build_player_cluster_dataset(player_season)
    classifiers, metrics = train_classifiers(context, baseline)
    cluster_frame, kmeans_model = train_player_clusters(cluster_input)

    save_artifacts(classifiers, metrics, cluster_frame, kmeans_model)
    save_processed_csv(context, "match_context.csv")
    save_processed_csv(baseline, "match_baseline.csv")
    save_processed_csv(player_season, "player_season.csv")
    save_processed_csv(build_player_dashboard_table(player_season), "player_dashboard.csv")
    save_processed_csv(build_venue_stats(bundle.matches, bundle.deliveries), "venue_stats.csv")

    batting_lb, bowling_lb = leaderboard_tables(player_season)
    save_processed_csv(batting_lb, "batting_leaderboard.csv")
    save_processed_csv(bowling_lb, "bowling_leaderboard.csv")

    seasons, matches_count, deliveries_count = dataset_summary(bundle)
    summary = {
        "seasons": seasons,
        "matches": matches_count,
        "deliveries": deliveries_count,
        "best_model": metrics.iloc[0]["model"],
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
