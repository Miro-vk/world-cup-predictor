from pathlib import Path

import pandas as pd

from world_cup_predictor.clean_data import clean_rankings, clean_results
from world_cup_predictor.load_data import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def add_recent_form_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
	"""Add last-N team form features from prior matches only.

	Features include wins/draws/losses, goals for/against, average opponent rank,
	and form points for both home and away teams.
	"""
	df = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True).copy()
	df["match_id"] = df.index

	home_history = pd.DataFrame(
		{
			"match_id": df["match_id"],
			"date": df["date"],
			"team": df["home_team"],
			"gf": df["home_score"],
			"ga": df["away_score"],
			"opponent_rank": df["away_rank"],
			"side": "home",
		}
	)

	away_history = pd.DataFrame(
		{
			"match_id": df["match_id"],
			"date": df["date"],
			"team": df["away_team"],
			"gf": df["away_score"],
			"ga": df["home_score"],
			"opponent_rank": df["home_rank"],
			"side": "away",
		}
	)

	team_history = pd.concat([home_history, away_history], ignore_index=True)
	# Nullable Int64 (from in-memory merge) breaks rolling — cast to float64.
	team_history["opponent_rank"] = pd.to_numeric(team_history["opponent_rank"], errors="coerce")
	team_history = team_history.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

	team_history["win"] = (team_history["gf"] > team_history["ga"]).astype(float)
	team_history["draw_game"] = (team_history["gf"] == team_history["ga"]).astype(float)
	team_history["loss"] = (team_history["gf"] < team_history["ga"]).astype(float)

	roll = team_history.groupby("team", group_keys=False)
	team_history["recent_wins_5"] = roll["win"].transform(
		lambda s: s.shift(1).rolling(window, min_periods=1).sum()
	)
	team_history["recent_draws_5"] = roll["draw_game"].transform(
		lambda s: s.shift(1).rolling(window, min_periods=1).sum()
	)
	team_history["recent_losses_5"] = roll["loss"].transform(
		lambda s: s.shift(1).rolling(window, min_periods=1).sum()
	)
	team_history["recent_goals_for_5"] = roll["gf"].transform(
		lambda s: s.shift(1).rolling(window, min_periods=1).sum()
	)
	team_history["recent_goals_against_5"] = roll["ga"].transform(
		lambda s: s.shift(1).rolling(window, min_periods=1).sum()
	)
	team_history["recent_opp_rank_avg_5"] = roll["opponent_rank"].transform(
		lambda s: s.shift(1).rolling(window, min_periods=1).mean()
	)

	team_history["recent_form_points"] = team_history["recent_wins_5"] * 3.0 + team_history["recent_draws_5"]

	feature_cols = [
		"recent_wins_5",
		"recent_draws_5",
		"recent_losses_5",
		"recent_goals_for_5",
		"recent_goals_against_5",
		"recent_opp_rank_avg_5",
		"recent_form_points",
	]

	home_features = team_history.loc[team_history["side"] == "home", ["match_id", *feature_cols]].rename(
		columns={
			"recent_wins_5": "home_recent_wins_5",
			"recent_draws_5": "home_recent_draws_5",
			"recent_losses_5": "home_recent_losses_5",
			"recent_goals_for_5": "home_recent_goals_for_5",
			"recent_goals_against_5": "home_recent_goals_against_5",
			"recent_opp_rank_avg_5": "home_recent_opp_rank_avg_5",
			"recent_form_points": "home_recent_form_points",
		}
	)

	away_features = team_history.loc[team_history["side"] == "away", ["match_id", *feature_cols]].rename(
		columns={
			"recent_wins_5": "away_recent_wins_5",
			"recent_draws_5": "away_recent_draws_5",
			"recent_losses_5": "away_recent_losses_5",
			"recent_goals_for_5": "away_recent_goals_for_5",
			"recent_goals_against_5": "away_recent_goals_against_5",
			"recent_opp_rank_avg_5": "away_recent_opp_rank_avg_5",
			"recent_form_points": "away_recent_form_points",
		}
	)

	df = df.merge(home_features, on="match_id", how="left")
	df = df.merge(away_features, on="match_id", how="left")

	numeric_cols = [
		"home_recent_wins_5",
		"home_recent_draws_5",
		"home_recent_losses_5",
		"home_recent_goals_for_5",
		"home_recent_goals_against_5",
		"home_recent_opp_rank_avg_5",
		"home_recent_form_points",
		"away_recent_wins_5",
		"away_recent_draws_5",
		"away_recent_losses_5",
		"away_recent_goals_for_5",
		"away_recent_goals_against_5",
		"away_recent_opp_rank_avg_5",
		"away_recent_form_points",
	]
	df[numeric_cols] = df[numeric_cols].fillna(0.0)
	df["recent_form_diff"] = df["home_recent_form_points"] - df["away_recent_form_points"]

	return df.drop(columns=["match_id"])


def merge_rankings_with_results(
	results: pd.DataFrame,
	rankings: pd.DataFrame,
) -> pd.DataFrame:
	"""Attach latest available ranking info to each home and away team before each match."""
	rankings_sorted = rankings.sort_values(["team", "ranking_date"]).reset_index(drop=True)

	home_merged = _merge_team_rankings(
		results=results,
		rankings=rankings_sorted,
		team_col="home_team",
		prefix="home",
	)
	merged = _merge_team_rankings(
		results=home_merged,
		rankings=rankings_sorted,
		team_col="away_team",
		prefix="away",
	)

	merged["rank_diff"] = merged["away_rank"] - merged["home_rank"]
	merged["points_diff"] = merged["home_ranking_points"] - merged["away_ranking_points"]
	merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype("int64")
	merged["draw"] = (merged["home_score"] == merged["away_score"]).astype("int64")

	return merged.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def summarize_ranking_coverage(merged: pd.DataFrame) -> dict[str, float | int]:
	"""Summarize rank availability after merging rankings and results."""
	home_missing = int(merged["home_rank"].isna().sum())
	away_missing = int(merged["away_rank"].isna().sum())
	both_missing = int((merged["home_rank"].isna() & merged["away_rank"].isna()).sum())
	both_available_pct = float((~(merged["home_rank"].isna() | merged["away_rank"].isna())).mean() * 100)

	return {
		"rows": int(len(merged)),
		"missing_home_rank": home_missing,
		"missing_away_rank": away_missing,
		"missing_both_rank": both_missing,
		"both_rank_available_pct": both_available_pct,
	}


def _merge_team_rankings(
	results: pd.DataFrame,
	rankings: pd.DataFrame,
	team_col: str,
	prefix: str,
) -> pd.DataFrame:
	merged_parts: list[pd.DataFrame] = []

	for team_name, matches in results.groupby(team_col, dropna=False):
		matches_sorted = matches.sort_values("date").copy()
		team_rankings = rankings.loc[
			rankings["team"] == team_name,
			["ranking_date", "rank", "ranking_points"],
		].sort_values("ranking_date")

		if team_rankings.empty:
			matches_sorted[f"{prefix}_ranking_date"] = pd.NaT
			matches_sorted[f"{prefix}_rank"] = pd.NA
			matches_sorted[f"{prefix}_ranking_points"] = pd.NA
			merged_parts.append(matches_sorted)
			continue

		merged_team = pd.merge_asof(
			matches_sorted,
			team_rankings,
			left_on="date",
			right_on="ranking_date",
			direction="backward",
			allow_exact_matches=True,
		)
		merged_team = merged_team.rename(
			columns={
				"ranking_date": f"{prefix}_ranking_date",
				"rank": f"{prefix}_rank",
				"ranking_points": f"{prefix}_ranking_points",
			}
		)
		merged_parts.append(merged_team)

	return pd.concat(merged_parts, ignore_index=True)


def get_current_team_states(processed_df: pd.DataFrame) -> dict:
	"""Return each team's latest rank and form snapshot from the processed match data.

	Iterates chronologically so the last appearance of each team (as home or away)
	overwrites earlier entries, leaving the most recent state in the dict.
	"""
	df = add_recent_form_features(processed_df, window=5)
	df = df.sort_values("date")

	states: dict = {}
	for _, row in df.iterrows():
		for team, prefix in [(row["home_team"], "home"), (row["away_team"], "away")]:
			states[team] = {
				"rank": row[f"{prefix}_rank"],
				"ranking_points": row[f"{prefix}_ranking_points"],
				"recent_wins_5": row[f"{prefix}_recent_wins_5"],
				"recent_draws_5": row[f"{prefix}_recent_draws_5"],
				"recent_losses_5": row[f"{prefix}_recent_losses_5"],
				"recent_goals_for_5": row[f"{prefix}_recent_goals_for_5"],
				"recent_goals_against_5": row[f"{prefix}_recent_goals_against_5"],
				"recent_opp_rank_avg_5": row[f"{prefix}_recent_opp_rank_avg_5"],
				"recent_form_points": row[f"{prefix}_recent_form_points"],
			}
	return states


def build_merged_dataset(output_path: Path | None = None) -> pd.DataFrame:
	"""Load, clean, merge rankings and results, and write a processed CSV."""
	rankings_raw, results_raw = load_raw_data()
	rankings = clean_rankings(rankings_raw)
	results = clean_results(results_raw)
	merged = merge_rankings_with_results(results, rankings)

	target = output_path or PROCESSED_DATA_DIR / "matches_with_rankings.csv"
	target.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(target, index=False)

	coverage = summarize_ranking_coverage(merged)
	print(
		"Ranking coverage | "
		f"rows={coverage['rows']} "
		f"home_missing={coverage['missing_home_rank']} "
		f"away_missing={coverage['missing_away_rank']} "
		f"both_missing={coverage['missing_both_rank']} "
		f"both_available_pct={coverage['both_rank_available_pct']:.2f}"
	)
	return merged


if __name__ == "__main__":
	dataset = build_merged_dataset()
	print(f"Created merged dataset with {len(dataset)} rows")
