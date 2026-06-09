import pandas as pd


RANKINGS_REQUIRED_COLUMNS = {"ranking_date", "team", "rank", "ranking_points"}
RESULTS_REQUIRED_COLUMNS = {
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

# Keep aliases minimal and explicit to avoid accidental remaps.
CANONICAL_NAME_ALIASES = {
	"USA": "United States",
	"US": "United States",
	"U.S.": "United States",
	"Korea Republic": "South Korea",
	"Korea DPR": "North Korea",
	"IR Iran": "Iran",
	"Cote d'Ivoire": "Ivory Coast",
	"Czechia": "Czech Republic",
	"Curacao": "Curaçao",
}


def _normalize_team_name(series: pd.Series) -> pd.Series:
	normalized = series.astype("string").str.strip()
	return normalized.replace(CANONICAL_NAME_ALIASES)


def _normalize_country_name(series: pd.Series) -> pd.Series:
	normalized = series.astype("string").str.strip()
	return normalized.replace(CANONICAL_NAME_ALIASES)


def _coerce_boolean(series: pd.Series) -> pd.Series:
	"""Coerce common boolean encodings to pandas nullable boolean dtype."""
	mapping = {
		"true": True,
		"false": False,
		"1": True,
		"0": False,
		"yes": True,
		"no": False,
		"y": True,
		"n": False,
	}
	text = series.astype("string").str.strip().str.lower()
	coerced = text.map(mapping).astype("object")
	coerced = coerced.where(text.notna(), pd.NA)
	return pd.Series(coerced, index=series.index, dtype="boolean")


def clean_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
	"""Clean rankings data and keep only required columns."""
	missing = RANKINGS_REQUIRED_COLUMNS - set(rankings.columns)
	if missing:
		missing_cols = ", ".join(sorted(missing))
		raise ValueError(f"Missing rankings columns: {missing_cols}")

	cleaned = rankings.loc[:, ["ranking_date", "team", "rank", "ranking_points"]].copy()
	cleaned["ranking_date"] = pd.to_datetime(cleaned["ranking_date"], errors="coerce")
	cleaned["team"] = _normalize_team_name(cleaned["team"])
	cleaned["rank"] = pd.to_numeric(cleaned["rank"], errors="coerce")
	cleaned["ranking_points"] = pd.to_numeric(cleaned["ranking_points"], errors="coerce")
	cleaned = cleaned.dropna(subset=["ranking_date", "team", "rank"])

	if (cleaned["rank"] <= 0).any():
		raise ValueError("Invalid rankings data: rank must be greater than zero")

	if cleaned["team"].str.len().eq(0).any():
		raise ValueError("Invalid rankings data: team names cannot be empty")

	cleaned["rank"] = cleaned["rank"].astype("int64")

	# Keep the most recent duplicate record per team/date if duplicates exist.
	cleaned = cleaned.drop_duplicates(subset=["ranking_date", "team"], keep="last")
	return cleaned.sort_values(["team", "ranking_date"]).reset_index(drop=True)


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
	"""Clean match results data and keep only required columns."""
	missing = RESULTS_REQUIRED_COLUMNS - set(results.columns)
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

	cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
	cleaned["home_team"] = _normalize_team_name(cleaned["home_team"])
	cleaned["away_team"] = _normalize_team_name(cleaned["away_team"])
	cleaned["country"] = _normalize_country_name(cleaned["country"])
	cleaned["home_score"] = pd.to_numeric(cleaned["home_score"], errors="coerce")
	cleaned["away_score"] = pd.to_numeric(cleaned["away_score"], errors="coerce")
	cleaned = cleaned.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

	if cleaned["home_team"].eq(cleaned["away_team"]).any():
		raise ValueError("Invalid results data: home_team and away_team must be different")

	if (cleaned["home_score"] < 0).any() or (cleaned["away_score"] < 0).any():
		raise ValueError("Invalid results data: scores must be non-negative")

	cleaned["home_score"] = cleaned["home_score"].astype("int64")
	cleaned["away_score"] = cleaned["away_score"].astype("int64")
	cleaned["neutral"] = _coerce_boolean(cleaned["neutral"])
	if cleaned["neutral"].isna().any():
		raise ValueError("Invalid results data: neutral must be a boolean value")

	cleaned = cleaned.drop_duplicates(
		subset=["date", "home_team", "away_team", "home_score", "away_score"],
		keep="last",
	)
	return cleaned.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
