from pathlib import Path

import pandas as pd

from world_cup_predictor.clean_data import clean_rankings, clean_results
from world_cup_predictor.load_data import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


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
