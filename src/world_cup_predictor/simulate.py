import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from world_cup_predictor.train import MODEL_PATH, TARGET_MAP_INV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRACKET_PATH  = PROJECT_ROOT / "data" / "raw" / "fifa_world_cup_2026_bracket_format.json"
GROUPS_PATH   = PROJECT_ROOT / "data" / "raw" / "wc2026_groups.json"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "matches_with_rankings.csv"


# Scoreline weights derived from FIFA World Cup / major tournament data.
_HOME_WIN_SCORELINES = [(1,0),(2,1),(2,0),(3,0),(3,1),(3,2),(4,1),(4,0)]
_HOME_WIN_WEIGHTS    = [0.261, 0.176, 0.157, 0.118, 0.118, 0.046, 0.039, 0.026]
_DRAW_SCORELINES     = [(0,0),(1,1),(2,2),(3,3)]
_DRAW_WEIGHTS        = [0.357, 0.449, 0.153, 0.041]
_AWAY_WIN_SCORELINES = [(0,1),(1,2),(0,2),(1,3),(0,3),(2,3),(0,4),(1,4)]
_AWAY_WIN_WEIGHTS    = [0.303, 0.212, 0.172, 0.091, 0.081, 0.081, 0.030, 0.020]


FEATURE_COLS = [
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_rank_missing",
    "away_rank_missing",
    "neutral",
    "tournament_importance",
    "home_recent_wins_5",
    "home_recent_draws_5",
    "home_recent_losses_5",
    "home_recent_goals_for_5",
    "home_recent_goals_against_5",
    "home_recent_opp_rank_avg_5",
    "away_recent_wins_5",
    "away_recent_draws_5",
    "away_recent_losses_5",
    "away_recent_goals_for_5",
    "away_recent_goals_against_5",
    "away_recent_opp_rank_avg_5",
    "home_recent_form_points",
    "away_recent_form_points",
    "recent_form_diff",
    "tournament",
]


def load_pipeline(model_path: Path = MODEL_PATH):
    return joblib.load(model_path)


class FastPredictor:
    """Numpy-only predictor that bypasses pandas overhead for Monte Carlo speed.

    Extracts the fitted imputer medians, scaler params, and one-hot categories
    from the sklearn pipeline once, then predicts with raw numpy arrays.
    All World Cup simulation matches share tournament="FIFA World Cup" and
    neutral=True so both are baked in at construction time.
    """

    # Numeric features in the same order the ColumnTransformer expects them.
    _NUMERIC_KEYS = [
        "home_rank", "away_rank", "rank_diff",
        "home_rank_missing", "away_rank_missing",
        "neutral", "tournament_importance",
        "home_recent_wins_5", "home_recent_draws_5", "home_recent_losses_5",
        "home_recent_goals_for_5", "home_recent_goals_against_5",
        "home_recent_opp_rank_avg_5",
        "away_recent_wins_5", "away_recent_draws_5", "away_recent_losses_5",
        "away_recent_goals_for_5", "away_recent_goals_against_5",
        "away_recent_opp_rank_avg_5",
        "home_recent_form_points", "away_recent_form_points", "recent_form_diff",
    ]

    _N_NUMERIC = 22  # number of numeric features

    def __init__(self, pipeline):
        pre = pipeline.named_steps["preprocessor"]
        num = pre.named_transformers_["num"]
        cat = pre.named_transformers_["cat"]

        self._imputer_medians = num.named_steps["imputer"].statistics_.copy()
        self._scaler_mean     = num.named_steps["scaler"].mean_.copy()
        self._scaler_scale    = num.named_steps["scaler"].scale_.copy()
        self._model           = pipeline.named_steps["model"]

        # Pre-compute the one-hot vector for "FIFA World Cup" (constant for all WC matches)
        cat_out = cat.transform([["FIFA World Cup"]])
        cat_arr = np.asarray(cat_out.todense() if hasattr(cat_out, "todense") else cat_out).flatten()

        # Pre-allocate a single reusable X buffer: [numeric(22) | cat_ohe(n)]
        self._X = np.empty((1, self._N_NUMERIC + len(cat_arr)), dtype=np.float64)
        self._X[0, self._N_NUMERIC:] = cat_arr  # cat part never changes

    def predict_proba(self, home: dict, away: dict) -> np.ndarray:
        """Return (p_home_win, p_draw, p_away_win) as a float array."""
        h_rank = home.get("rank")
        a_rank = away.get("rank")
        h_miss = 1.0 if (h_rank is None or (isinstance(h_rank, float) and np.isnan(h_rank))) else 0.0
        a_miss = 1.0 if (a_rank is None or (isinstance(a_rank, float) and np.isnan(a_rank))) else 0.0
        rank_diff = np.nan if (h_miss or a_miss) else float(a_rank) - float(h_rank)
        h_form = home.get("recent_form_points") or 0.0
        a_form = away.get("recent_form_points") or 0.0

        buf = self._X[0]
        buf[0]  = h_rank if not h_miss else np.nan
        buf[1]  = a_rank if not a_miss else np.nan
        buf[2]  = rank_diff
        buf[3]  = h_miss
        buf[4]  = a_miss
        buf[5]  = 1.0  # neutral
        buf[6]  = 5.0  # tournament_importance
        buf[7]  = home.get("recent_wins_5",          0.0)
        buf[8]  = home.get("recent_draws_5",         0.0)
        buf[9]  = home.get("recent_losses_5",        0.0)
        buf[10] = home.get("recent_goals_for_5",     0.0)
        buf[11] = home.get("recent_goals_against_5", 0.0)
        buf[12] = home.get("recent_opp_rank_avg_5") or np.nan
        buf[13] = away.get("recent_wins_5",          0.0)
        buf[14] = away.get("recent_draws_5",         0.0)
        buf[15] = away.get("recent_losses_5",        0.0)
        buf[16] = away.get("recent_goals_for_5",     0.0)
        buf[17] = away.get("recent_goals_against_5", 0.0)
        buf[18] = away.get("recent_opp_rank_avg_5") or np.nan
        buf[19] = h_form
        buf[20] = a_form
        buf[21] = h_form - a_form

        # Impute then scale (in-place on the numeric slice)
        num = buf[:self._N_NUMERIC]
        nan_mask = np.isnan(num)
        num[nan_mask] = self._imputer_medians[nan_mask]
        num -= self._scaler_mean
        num /= self._scaler_scale

        return self._model.predict_proba(self._X)[0]


def predict_match(
    home_team: str,
    away_team: str,
    team_states: dict,
    pipeline,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    fast: "FastPredictor | None" = None,
) -> dict:
    """Return H/D/A probabilities for a single match.

    Builds the feature row directly from pre-computed team states rather than
    rerunning rolling form over the full match history.

    Pass fast=<FastPredictor> to bypass pandas entirely (used by monte_carlo).
    """
    home = team_states.get(home_team, {})
    away = team_states.get(away_team, {})

    if fast is not None:
        proba = fast.predict_proba(home, away)
        return {
            "p_home_win": float(proba[0]),
            "p_draw":     float(proba[1]),
            "p_away_win": float(proba[2]),
        }

    home_rank = home.get("rank")
    away_rank = away.get("rank")

    home_rank_missing = float(home_rank is None or pd.isna(home_rank))
    away_rank_missing = float(away_rank is None or pd.isna(away_rank))

    if home_rank_missing or away_rank_missing:
        rank_diff = np.nan
    else:
        rank_diff = float(away_rank) - float(home_rank)

    home_form = home.get("recent_form_points") or 0.0
    away_form = away.get("recent_form_points") or 0.0

    row = {
        "home_rank": home_rank,
        "away_rank": away_rank,
        "rank_diff": rank_diff,
        "home_rank_missing": home_rank_missing,
        "away_rank_missing": away_rank_missing,
        "neutral": float(neutral),
        "tournament_importance": 5.0,
        "home_recent_wins_5": home.get("recent_wins_5", 0.0),
        "home_recent_draws_5": home.get("recent_draws_5", 0.0),
        "home_recent_losses_5": home.get("recent_losses_5", 0.0),
        "home_recent_goals_for_5": home.get("recent_goals_for_5", 0.0),
        "home_recent_goals_against_5": home.get("recent_goals_against_5", 0.0),
        "home_recent_opp_rank_avg_5": home.get("recent_opp_rank_avg_5"),
        "away_recent_wins_5": away.get("recent_wins_5", 0.0),
        "away_recent_draws_5": away.get("recent_draws_5", 0.0),
        "away_recent_losses_5": away.get("recent_losses_5", 0.0),
        "away_recent_goals_for_5": away.get("recent_goals_for_5", 0.0),
        "away_recent_goals_against_5": away.get("recent_goals_against_5", 0.0),
        "away_recent_opp_rank_avg_5": away.get("recent_opp_rank_avg_5"),
        "home_recent_form_points": home_form,
        "away_recent_form_points": away_form,
        "recent_form_diff": home_form - away_form,
        "tournament": tournament,
    }

    proba = pipeline.predict_proba(pd.DataFrame([row])[FEATURE_COLS])[0]

    return {
        "p_home_win": float(proba[0]),
        "p_draw": float(proba[1]),
        "p_away_win": float(proba[2]),
    }


def sample_outcome(probs: dict, knockout: bool = False) -> str:
    """Sample H/D/A from probabilities.

    In knockout mode the draw probability is split equally between H and A,
    simulating extra time and penalties as a 50/50 coin flip.
    """
    if knockout:
        p_h = probs["p_home_win"] + probs["p_draw"] / 2
        p_a = probs["p_away_win"] + probs["p_draw"] / 2
        return np.random.choice(["H", "A"], p=[p_h, p_a])
    return np.random.choice(
        ["H", "D", "A"],
        p=[probs["p_home_win"], probs["p_draw"], probs["p_away_win"]],
    )


def sample_scoreline(outcome: str) -> tuple[int, int]:
    """Sample a (home_goals, away_goals) scoreline consistent with outcome.

    Weights are derived from FIFA World Cup and major tournament data.
    Weights don't need to sum to 1 — numpy normalises them.
    """
    if outcome == "H":
        scorelines, weights = _HOME_WIN_SCORELINES, _HOME_WIN_WEIGHTS
    elif outcome == "D":
        scorelines, weights = _DRAW_SCORELINES, _DRAW_WEIGHTS
    else:
        scorelines, weights = _AWAY_WIN_SCORELINES, _AWAY_WIN_WEIGHTS
    w = np.array(weights, dtype=float)
    idx = np.random.choice(len(scorelines), p=w / w.sum())
    return scorelines[idx]


def simulate_group(
    group_teams: list[str],
    team_states: dict,
    pipeline,
    fast: "FastPredictor | None" = None,
) -> tuple[list[dict], list[dict]]:
    """Simulate all round-robin matches for a group of 4 teams.

    Returns standings as a list of dicts sorted 1st→4th (pts → gd → gf →
    random tiebreak), and a match log with scorelines and probabilities.
    Each standings entry: {"team": str, "pts": int, "gd": int, "gf": int,
                           "ga": int, "w": int, "d": int, "l": int}
    """
    records = {
        t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0}
        for t in group_teams
    }
    match_log = []

    for home_team, away_team in combinations(group_teams, 2):
        probs = predict_match(home_team, away_team, team_states, pipeline, neutral=True, fast=fast)
        outcome = sample_outcome(probs, knockout=False)
        home_goals, away_goals = sample_scoreline(outcome)

        h, a = records[home_team], records[away_team]
        if outcome == "H":
            h["pts"] += 3; h["w"] += 1; a["l"] += 1
        elif outcome == "D":
            h["pts"] += 1; h["d"] += 1; a["pts"] += 1; a["d"] += 1
        else:
            a["pts"] += 3; a["w"] += 1; h["l"] += 1
        h["gf"] += home_goals; h["ga"] += away_goals; h["gd"] += home_goals - away_goals
        a["gf"] += away_goals; a["ga"] += home_goals; a["gd"] += away_goals - home_goals

        match_log.append({
            "home_team": home_team, "away_team": away_team,
            "home_goals": home_goals, "away_goals": away_goals,
            "outcome": outcome,
            "p_home_win": probs["p_home_win"],
            "p_draw": probs["p_draw"],
            "p_away_win": probs["p_away_win"],
        })

    standings = sorted(
        records.values(),
        key=lambda r: (r["pts"], r["gd"], r["gf"], np.random.random()),
        reverse=True,
    )
    return standings, match_log


# ---------------------------------------------------------------------------
# Step 6 — Third-place qualification
# ---------------------------------------------------------------------------

def get_best_third_place(
    all_standings: dict[str, list[dict]],
) -> tuple[list[str], list[str]]:
    """Return the 8 best 3rd-place teams and their source groups.

    Ranked by pts → gd → gf with a random tiebreak (simulates draw of lots).
    Returns (team_names, group_letters) so the bracket builder can avoid
    pairing a 3rd-place team against their own group in the R32.
    """
    thirds = [
        {"team": standings[2]["team"], "group": group,
         "pts": standings[2]["pts"], "gd": standings[2]["gd"], "gf": standings[2]["gf"]}
        for group, standings in all_standings.items()
    ]
    thirds.sort(key=lambda r: (r["pts"], r["gd"], r["gf"], np.random.random()), reverse=True)
    top8 = thirds[:8]
    return [r["team"] for r in top8], [r["group"] for r in top8]


# ---------------------------------------------------------------------------
# Step 7 — Knockout bracket driven by the official FIFA 2026 bracket JSON
# ---------------------------------------------------------------------------

def simulate_bracket(
    all_standings: dict[str, pd.DataFrame],
    team_states: dict,
    pipeline,
    bracket_path: Path = BRACKET_PATH,
    _bracket_cache: dict | None = None,
    fast: "FastPredictor | None" = None,
) -> dict:
    """Simulate the full knockout bracket using the official FIFA 2026 bracket JSON.

    Slot resolution rules:
      "1A"          → winner of Group A
      "2B"          → runner-up of Group B
      "M74_away"    → best available 3rd-place team eligible for that slot
      "W73"         → winner of match 73
      "L101"        → loser of match 101

    Third-place teams are assigned greedily: for each slot the best available
    (pts → gd → gf) 3rd-place team whose group appears in the slot's eligible
    list is chosen.

    Pass _bracket_cache to avoid re-reading the JSON file on every Monte Carlo run.
    """
    if _bracket_cache is not None:
        bracket = _bracket_cache
    else:
        with open(bracket_path) as f:
            bracket = json.load(f)

    # Build sorted pool of 3rd-place teams: [{"team": ..., "group": ...}, ...]
    best_thirds, thirds_groups = get_best_third_place(all_standings)
    thirds_pool = [{"team": t, "group": g} for t, g in zip(best_thirds, thirds_groups)]
    third_place_slots = bracket["third_place_slots"]

    # Match results: {match_number: {"winner": str, "loser": str}}
    match_results: dict[int, dict] = {}

    def resolve(slot: str) -> str:
        """Turn a slot string into a concrete team name."""
        if slot[0].isdigit() and len(slot) == 2:
            # e.g. "1A", "2B" — standings is now a list of dicts (0-indexed)
            pos = int(slot[0]) - 1
            group = slot[1]
            return all_standings[group][pos]["team"]
        if slot.startswith("W"):
            return match_results[int(slot[1:])]["winner"]
        if slot.startswith("L"):
            return match_results[int(slot[1:])]["loser"]
        if slot in third_place_slots:
            # Best available 3rd-place team from the eligible groups
            eligible = third_place_slots[slot]
            for i, entry in enumerate(thirds_pool):
                if entry["group"] in eligible:
                    return thirds_pool.pop(i)["team"]
            # Fallback: take the best remaining regardless of group
            return thirds_pool.pop(0)["team"]
        raise ValueError(f"Cannot resolve bracket slot: {slot!r}")

    def play(match_num: int, home_slot: str, away_slot: str) -> str:
        home = resolve(home_slot)
        away = resolve(away_slot)
        probs = predict_match(home, away, team_states, pipeline, neutral=True, fast=fast)
        outcome = sample_outcome(probs, knockout=True)
        winner, loser = (home, away) if outcome == "H" else (away, home)
        match_results[match_num] = {"winner": winner, "loser": loser,
                                     "home": home, "away": away}
        return winner

    ko = bracket["knockout_stage"]
    for stage in ("round_of_32", "round_of_16", "quarter_finals",
                   "semi_finals", "third_place_match", "final"):
        for m in ko[stage]:
            play(m["match"], m["home"], m["away"])

    final_num        = ko["final"][0]["match"]
    sf1_num, sf2_num = ko["semi_finals"][0]["match"], ko["semi_finals"][1]["match"]
    tp_num           = ko["third_place_match"][0]["match"]

    return {
        "champion":     match_results[final_num]["winner"],
        "runner_up":    match_results[final_num]["loser"],
        "third_place":  match_results[tp_num]["winner"],
        "fourth_place": match_results[tp_num]["loser"],
        "semi_finalists": [
            match_results[sf1_num]["home"], match_results[sf1_num]["away"],
            match_results[sf2_num]["home"], match_results[sf2_num]["away"],
        ],
        "match_results": match_results,
    }


# ---------------------------------------------------------------------------
# Step 8 — simulate_tournament: wire everything into one call
# ---------------------------------------------------------------------------

def simulate_tournament(
    groups_path: Path = GROUPS_PATH,
    processed_path: Path = PROCESSED_PATH,
    model_path: Path = MODEL_PATH,
    bracket_path: Path = BRACKET_PATH,
) -> dict:
    """Run one full simulation of the 2026 FIFA World Cup.

    Loads all data and the model fresh on each call so it is safe to use
    inside a Monte Carlo loop without shared mutable state.

    Returns the bracket result dict from simulate_bracket plus the full
    group standings for inspection.
    """
    from world_cup_predictor.features import get_current_team_states

    with open(groups_path) as f:
        groups = json.load(f)["groups"]

    df = pd.read_csv(processed_path, parse_dates=["date"])
    team_states = get_current_team_states(df)
    pipeline = load_pipeline(model_path)

    all_standings: dict[str, list[dict]] = {}
    for letter, teams in groups.items():
        standings, _ = simulate_group(teams, team_states, pipeline)
        all_standings[letter] = standings

    with open(bracket_path) as f:
        bracket = json.load(f)

    result = simulate_bracket(all_standings, team_states, pipeline,
                               _bracket_cache=bracket)
    result["group_standings"] = all_standings
    return result


# ---------------------------------------------------------------------------
# Step 9 — Monte Carlo wrapper
# ---------------------------------------------------------------------------

def monte_carlo(
    n: int = 10_000,
    seed: int | None = None,
    groups_path: Path = GROUPS_PATH,
    processed_path: Path = PROCESSED_PATH,
    model_path: Path = MODEL_PATH,
    bracket_path: Path = BRACKET_PATH,
) -> pd.DataFrame:
    """Run the tournament n times and return win-probability estimates.

    Tracks four finish levels per team:
      - champion        (finished 1st)
      - final           (reached the final, win or lose)
      - semi_final      (reached the semi-finals)
      - quarter_final   (reached the quarter-finals, inferred from R16 losers)

    Returns a DataFrame sorted by champion probability, with columns:
      team, champion_pct, final_pct, semi_final_pct, quarter_final_pct
    """
    if seed is not None:
        np.random.seed(seed)

    from world_cup_predictor.features import get_current_team_states

    # Load shared resources once — only the random sampling differs per run.
    with open(groups_path) as f:
        groups = json.load(f)["groups"]

    df = pd.read_csv(processed_path, parse_dates=["date"])
    team_states = get_current_team_states(df)
    pipeline = load_pipeline(model_path)

    with open(bracket_path) as f:
        bracket = json.load(f)

    fast = FastPredictor(pipeline)

    ko = bracket["knockout_stage"]
    qf_match_nums = [m["match"] for m in ko["quarter_finals"]]
    sf_match_nums = [m["match"] for m in ko["semi_finals"]]

    counts: dict[str, dict[str, int]] = {}

    def _add(team: str, level: str) -> None:
        counts.setdefault(team, {"champion": 0, "final": 0,
                                  "semi_final": 0, "quarter_final": 0})
        counts[team][level] += 1

    for _ in range(n):
        all_standings: dict[str, pd.DataFrame] = {}
        for letter, teams in groups.items():
            standings, _ = simulate_group(teams, team_states, pipeline, fast=fast)
            all_standings[letter] = standings

        result = simulate_bracket(
            all_standings, team_states, pipeline, _bracket_cache=bracket, fast=fast
        )
        mr = result["match_results"]

        _add(result["champion"], "champion")

        for team in (result["champion"], result["runner_up"]):
            _add(team, "final")

        for team in result["semi_finalists"]:
            _add(team, "semi_final")

        for m_num in qf_match_nums:
            for team in (mr[m_num]["home"], mr[m_num]["away"]):
                _add(team, "quarter_final")

    rows = []
    for team, c in counts.items():
        rows.append({
            "team":             team,
            "champion_pct":     round(c["champion"]      / n * 100, 2),
            "final_pct":        round(c["final"]         / n * 100, 2),
            "semi_final_pct":   round(c["semi_final"]    / n * 100, 2),
            "quarter_final_pct":round(c["quarter_final"] / n * 100, 2),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("champion_pct", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import time
    print("Running single tournament simulation...")
    result = simulate_tournament()
    print(f"Champion:   {result['champion']}")
    print(f"Runner-up:  {result['runner_up']}")
    print(f"3rd place:  {result['third_place']}")
    print()
    print("Running Monte Carlo (1 000 simulations)...")
    t0 = time.time()
    df_mc = monte_carlo(n=1_000, seed=42)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    print(df_mc.head(10).to_string(index=False))
