from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class DatasetBundle:
    matches: pd.DataFrame
    deliveries: pd.DataFrame


MATCH_COLUMN_ALIASES = {
    "id": "match_id",
    "match_id": "match_id",
    "season": "season",
    "date": "date",
    "venue": "venue",
    "team1": "team1",
    "team2": "team2",
    "winner": "winner",
    "toss_winner": "toss_winner",
    "toss_decision": "toss_decision",
    "result": "result",
}

DELIVERY_COLUMN_ALIASES = {
    "match_id": "match_id",
    "id": "match_id",
    "inning": "inning",
    "innings": "inning",
    "over": "over",
    "ball": "ball",
    "batting_team": "batting_team",
    "bowling_team": "bowling_team",
    "batter": "batter",
    "batsman": "batter",
    "non_striker": "non_striker",
    "bowler": "bowler",
    "batsman_runs": "batsman_runs",
    "total_runs": "total_runs",
    "extra_runs": "extra_runs",
    "is_wicket": "is_wicket",
    "player_dismissed": "player_dismissed",
    "dismissal_kind": "dismissal_kind",
}


def _normalize_columns(frame: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = column.strip().lower()
        if key in alias_map:
            renamed[column] = alias_map[key]
    return frame.rename(columns=renamed)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_season(value: object) -> int:
    text = str(value).strip()
    if "/" not in text:
        return int(float(text))

    start_text, end_text = text.split("/", 1)
    start_year = int(start_text)
    if len(end_text) == 2:
        century = start_year // 100
        end_year = century * 100 + int(end_text)
        if end_year < start_year:
            end_year += 100
        return end_year
    return int(end_text)


def raw_data_exists() -> bool:
    return (RAW_DIR / "matches.csv").exists() and (RAW_DIR / "deliveries.csv").exists()


def load_raw_datasets() -> DatasetBundle:
    matches = _normalize_columns(_load_csv(RAW_DIR / "matches.csv"), MATCH_COLUMN_ALIASES)
    deliveries = _normalize_columns(_load_csv(RAW_DIR / "deliveries.csv"), DELIVERY_COLUMN_ALIASES)

    required_matches = {"match_id", "season", "date", "venue", "team1", "team2", "winner"}
    required_deliveries = {
        "match_id",
        "inning",
        "over",
        "ball",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "batsman_runs",
        "total_runs",
        "is_wicket",
    }
    missing_matches = required_matches - set(matches.columns)
    missing_deliveries = required_deliveries - set(deliveries.columns)
    if missing_matches:
        raise ValueError(f"matches.csv missing columns: {sorted(missing_matches)}")
    if missing_deliveries:
        raise ValueError(f"deliveries.csv missing columns: {sorted(missing_deliveries)}")

    matches["season"] = matches["season"].apply(_normalize_season)
    deliveries["inning"] = deliveries["inning"].astype(int)
    deliveries["over"] = deliveries["over"].astype(int)
    deliveries["ball"] = deliveries["ball"].astype(int)
    deliveries["batsman_runs"] = deliveries["batsman_runs"].fillna(0).astype(int)
    deliveries["total_runs"] = deliveries["total_runs"].fillna(0).astype(int)
    deliveries["is_wicket"] = deliveries["is_wicket"].fillna(0).astype(int)
    deliveries["extra_runs"] = deliveries.get("extra_runs", 0)
    deliveries["extra_runs"] = deliveries["extra_runs"].fillna(0).astype(int)

    if "player_dismissed" not in deliveries.columns:
        deliveries["player_dismissed"] = None
    if "dismissal_kind" not in deliveries.columns:
        deliveries["dismissal_kind"] = None
    if "toss_winner" not in matches.columns:
        matches["toss_winner"] = matches["team1"]
    if "toss_decision" not in matches.columns:
        matches["toss_decision"] = "field"
    if "result" not in matches.columns:
        matches["result"] = "normal"

    return DatasetBundle(matches=matches, deliveries=deliveries)


def load_processed_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_processed_csv(frame: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    frame.to_csv(path, index=False)
    return path


def dataset_summary(bundle: DatasetBundle) -> Tuple[int, int, int]:
    seasons = bundle.matches["season"].nunique()
    matches = len(bundle.matches)
    deliveries = len(bundle.deliveries)
    return seasons, matches, deliveries
