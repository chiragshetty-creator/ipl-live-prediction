from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
]

VENUES = [
    "Wankhede Stadium",
    "M. A. Chidambaram Stadium",
    "Eden Gardens",
    "Arun Jaitley Stadium",
    "Narendra Modi Stadium",
    "M. Chinnaswamy Stadium",
    "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium",
]

TEAM_STRENGTH = {
    "Chennai Super Kings": 1.08,
    "Mumbai Indians": 1.06,
    "Royal Challengers Bengaluru": 1.03,
    "Kolkata Knight Riders": 1.04,
    "Rajasthan Royals": 1.02,
    "Sunrisers Hyderabad": 1.01,
    "Delhi Capitals": 0.99,
    "Punjab Kings": 0.97,
}


def build_squads() -> dict[str, list[str]]:
    squads = {}
    for team in TEAMS:
        prefix = "".join(word[0] for word in team.split())
        squads[team] = [f"{prefix} Player {i:02d}" for i in range(1, 16)]
    return squads


def choose_lineup(players: list[str]) -> list[str]:
    return random.sample(players, 11)


def simulate_innings(
    match_id: int,
    inning: int,
    batting_team: str,
    bowling_team: str,
    batting_order: list[str],
    bowlers: list[str],
    base_score: int,
    chase_bias: float = 0.0,
) -> tuple[list[dict], int]:
    records = []
    striker_idx = 0
    non_striker_idx = 1
    next_batter_idx = 2
    wickets = 0
    total = 0

    for over in range(1, 21):
        over_bowler = random.choice(bowlers[:6] if len(bowlers) >= 6 else bowlers)
        for ball in range(1, 7):
            if wickets >= 10:
                break
            batting_factor = TEAM_STRENGTH[batting_team] + chase_bias
            bowling_factor = TEAM_STRENGTH[bowling_team]
            expected = base_score / 120.0 * batting_factor / bowling_factor
            probs = np.array([0.28, 0.33, 0.12, 0.14, 0.07, 0.03, 0.03])
            outcome = np.random.choice([0, 1, 2, 3, 4, 6, -1], p=probs)
            wicket = int(outcome == -1)
            batsman_runs = 0 if wicket else int(outcome)
            extras = int(np.random.rand() < 0.03)
            adjustment = np.random.normal(loc=expected - 1.3, scale=0.45)
            if not wicket:
                batsman_runs = max(0, min(6, int(round(batsman_runs + adjustment))))
                if batsman_runs == 5:
                    batsman_runs = 4
            total_runs = batsman_runs + extras
            total += total_runs

            striker = batting_order[striker_idx]
            non_striker = batting_order[non_striker_idx]
            dismissal_kind = None
            player_dismissed = None
            if wicket:
                dismissal_kind = random.choice(["caught", "bowled", "lbw", "run out"])
                player_dismissed = striker
                wickets += 1
                if next_batter_idx < len(batting_order):
                    striker_idx = next_batter_idx
                    next_batter_idx += 1
            elif batsman_runs % 2 == 1:
                striker_idx, non_striker_idx = non_striker_idx, striker_idx

            records.append(
                {
                    "match_id": match_id,
                    "inning": inning,
                    "over": over,
                    "ball": ball,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batter": batting_order[striker_idx] if wicket else striker,
                    "non_striker": non_striker,
                    "bowler": over_bowler,
                    "batsman_runs": batsman_runs,
                    "extra_runs": extras,
                    "total_runs": total_runs,
                    "is_wicket": wicket,
                    "player_dismissed": player_dismissed,
                    "dismissal_kind": dismissal_kind,
                }
            )

        striker_idx, non_striker_idx = non_striker_idx, striker_idx
        if wickets >= 10:
            break
    return records, total


def generate_demo_data(seed: int = 42, matches_per_season: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    squads = build_squads()
    matches = []
    deliveries = []
    match_id = 1
    start = date(2008, 4, 18)

    for season in range(2008, 2025):
        for _ in range(matches_per_season):
            team1, team2 = random.sample(TEAMS, 2)
            venue = random.choice(VENUES)
            toss_winner = random.choice([team1, team2])
            toss_decision = random.choice(["bat", "field"])
            first_batting = toss_winner if toss_decision == "bat" else (team2 if toss_winner == team1 else team1)
            second_batting = team2 if first_batting == team1 else team1

            lineup1 = choose_lineup(squads[team1])
            lineup2 = choose_lineup(squads[team2])
            bowling_lineup1 = lineup1[5:]
            bowling_lineup2 = lineup2[5:]

            first_base = random.randint(145, 188)
            second_base = random.randint(140, 185)
            innings1, score1 = simulate_innings(
                match_id,
                1,
                first_batting,
                second_batting,
                lineup1 if first_batting == team1 else lineup2,
                bowling_lineup2 if second_batting == team2 else bowling_lineup1,
                first_base,
            )
            chase_bias = 0.05 if score1 < 165 else -0.02
            innings2, score2 = simulate_innings(
                match_id,
                2,
                second_batting,
                first_batting,
                lineup2 if second_batting == team2 else lineup1,
                bowling_lineup1 if first_batting == team1 else bowling_lineup2,
                second_base,
                chase_bias=chase_bias,
            )
            winner = second_batting if score2 >= score1 + 1 else first_batting
            matches.append(
                {
                    "match_id": match_id,
                    "season": season,
                    "date": (start + timedelta(days=(season - 2008) * 35 + match_id)).isoformat(),
                    "venue": venue,
                    "team1": team1,
                    "team2": team2,
                    "winner": winner,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "result": "normal",
                }
            )
            deliveries.extend(innings1)
            deliveries.extend(innings2)
            match_id += 1

    return pd.DataFrame(matches), pd.DataFrame(deliveries)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches, deliveries = generate_demo_data()
    matches.to_csv(RAW_DIR / "matches.csv", index=False)
    deliveries.to_csv(RAW_DIR / "deliveries.csv", index=False)
    print(f"Generated {len(matches)} matches and {len(deliveries)} deliveries.")


if __name__ == "__main__":
    main()
