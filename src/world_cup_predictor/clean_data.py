import pandas as pd


def _normalize_team_name(series: pd.Series) -> pd.Series:
	return series.astype("string").str.strip()


def clean_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
	"""Clean rankings data and keep only required columns."""
	required = {"ranking_date", "team", "rank", "ranking_points"}
	missing = required - set(rankings.columns)
	if missing:
		missing_cols = ", ".join(sorted(missing))
		raise ValueError(f"Missing rankings columns: {missing_cols}")

	cleaned = rankings.loc[:, ["ranking_date", "team", "rank", "ranking_points"]].copy()
	cleaned["team"] = _normalize_team_name(cleaned["team"])
	cleaned["rank"] = pd.to_numeric(cleaned["rank"], errors="coerce")
	cleaned["ranking_points"] = pd.to_numeric(cleaned["ranking_points"], errors="coerce")
	cleaned = cleaned.dropna(subset=["ranking_date", "team", "rank"])
	cleaned["rank"] = cleaned["rank"].astype("int64")

	# Keep the most recent duplicate record per team/date if duplicates exist.
	cleaned = cleaned.drop_duplicates(subset=["ranking_date", "team"], keep="last")
	return cleaned.sort_values(["team", "ranking_date"]).reset_index(drop=True)


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
	"""Clean match results data and keep only required columns."""
	required = {
		"date",
		"home_team",
		"away_team",
		"home_score",
		"away_score",
		"tournament",
		"city",
		"country",
		"neutral",
	}
	missing = required - set(results.columns)
	if missing:
		missing_cols = ", ".join(sorted(missing))
		raise ValueError(f"Missing results columns: {missing_cols}")

	cleaned = results.loc[
		:,
		[
			"date",
			"home_team",
			"away_team",
			"home_score",
			"away_score",
			"tournament",
			"city",
			"country",
			"neutral",
		],
	].copy()

	cleaned["home_team"] = _normalize_team_name(cleaned["home_team"])
	cleaned["away_team"] = _normalize_team_name(cleaned["away_team"])
	cleaned["home_score"] = pd.to_numeric(cleaned["home_score"], errors="coerce")
	cleaned["away_score"] = pd.to_numeric(cleaned["away_score"], errors="coerce")
	cleaned = cleaned.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
	cleaned["home_score"] = cleaned["home_score"].astype("int64")
	cleaned["away_score"] = cleaned["away_score"].astype("int64")
	cleaned["neutral"] = cleaned["neutral"].astype("boolean")

	cleaned = cleaned.drop_duplicates(
		subset=["date", "home_team", "away_team", "home_score", "away_score"],
		keep="last",
	)
	return cleaned.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
