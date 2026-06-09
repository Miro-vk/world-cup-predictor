import pandas as pd
import pytest

from world_cup_predictor.clean_data import clean_rankings, clean_results
from world_cup_predictor.features import add_recent_form_features


def test_clean_results_coerces_boolean_and_types() -> None:
	results = pd.DataFrame(
		{
			"date": ["2024-01-01", "2024-01-02"],
			"home_team": ["USA", "Japan"],
			"away_team": ["Mexico", "Korea Republic"],
			"home_score": [2, "1"],
			"away_score": [1, "0"],
			"tournament": ["Friendly", "Friendly"],
			"city": ["CityA", "CityB"],
			"country": ["USA", "CountryB"],
			"neutral": ["FALSE", "1"],
		}
	)

	cleaned = clean_results(results)

	assert str(cleaned["date"].dtype).startswith("datetime64")
	assert cleaned["home_team"].tolist()[0] == "United States of America"
	assert cleaned["away_team"].tolist()[1] == "South Korea"
	assert cleaned["country"].tolist()[0] == "United States of America"
	assert cleaned["home_score"].dtype == "int64"
	assert cleaned["away_score"].dtype == "int64"
	assert cleaned["neutral"].dtype == "boolean"


def test_clean_results_rejects_negative_score() -> None:
	results = pd.DataFrame(
		{
			"date": ["2024-01-01"],
			"home_team": ["Brazil"],
			"away_team": ["Argentina"],
			"home_score": [-1],
			"away_score": [0],
			"tournament": ["Friendly"],
			"city": ["Rio"],
			"country": ["Brazil"],
			"neutral": [False],
		}
	)

	with pytest.raises(ValueError, match="scores must be non-negative"):
		clean_results(results)


def test_clean_results_rejects_invalid_neutral() -> None:
	results = pd.DataFrame(
		{
			"date": ["2024-01-01"],
			"home_team": ["Brazil"],
			"away_team": ["Argentina"],
			"home_score": [1],
			"away_score": [0],
			"tournament": ["Friendly"],
			"city": ["Rio"],
			"country": ["Brazil"],
			"neutral": ["maybe"],
		}
	)

	with pytest.raises(ValueError, match="neutral must be a boolean"):
		clean_results(results)


def test_clean_rankings_rejects_invalid_rank() -> None:
	rankings = pd.DataFrame(
		{
			"ranking_date": ["2024-01-01", "2024-01-01"],
			"team": ["USA", "France"],
			"rank": [0, 2],
			"ranking_points": [1500.0, 1800.0],
		}
	)

	with pytest.raises(ValueError, match="rank must be greater than zero"):
		clean_rankings(rankings)


def test_clean_rankings_normalizes_names_and_types() -> None:
	rankings = pd.DataFrame(
		{
			"ranking_date": ["2024-01-01"],
			"team": ["USA"],
			"rank": [11],
			"ranking_points": [1700.5],
		}
	)

	cleaned = clean_rankings(rankings)

	assert cleaned["team"].iloc[0] == "United States of America"
	assert cleaned["rank"].dtype == "int64"
	assert str(cleaned["ranking_date"].dtype).startswith("datetime64")


def test_recent_form_last_five_includes_wdl_goals_and_opponent_rank() -> None:
	matches = pd.DataFrame(
		{
			"date": pd.date_range("2024-01-01", periods=7, freq="D"),
			"home_team": ["A", "A", "A", "A", "A", "A", "A"],
			"away_team": ["B", "C", "D", "E", "F", "G", "H"],
			"home_score": [1, 0, 2, 3, 1, 2, 0],
			"away_score": [0, 1, 2, 0, 3, 1, 0],
			"home_rank": [5, 5, 5, 5, 5, 5, 5],
			"away_rank": [10, 20, 30, 40, 50, 60, 70],
			"tournament": ["Friendly"] * 7,
			"neutral": [False] * 7,
		}
	)

	featured = add_recent_form_features(matches, window=5)
	fixture = featured.iloc[6]

	# Last 5 games for team A before day 7 are days 2-6:
	# scores: 0-1, 2-2, 3-0, 1-3, 2-1 -> W=2, D=1, L=2
	assert fixture["home_recent_wins_5"] == 2.0
	assert fixture["home_recent_draws_5"] == 1.0
	assert fixture["home_recent_losses_5"] == 2.0
	assert fixture["home_recent_goals_for_5"] == 8.0
	assert fixture["home_recent_goals_against_5"] == 7.0
	assert fixture["home_recent_opp_rank_avg_5"] == pytest.approx((20 + 30 + 40 + 50 + 60) / 5)

	# First game for team A has no prior history.
	first_fixture = featured.iloc[0]
	assert first_fixture["home_recent_wins_5"] == 0.0
	assert first_fixture["home_recent_goals_for_5"] == 0.0
	assert first_fixture["home_recent_opp_rank_avg_5"] == 0.0
