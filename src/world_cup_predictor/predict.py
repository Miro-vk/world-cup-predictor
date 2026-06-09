from pathlib import Path

import joblib
import pandas as pd

from world_cup_predictor.train import (
	METADATA_PATH,
	MODEL_PATH,
	TARGET_MAP_INV,
	prepare_training_frame,
)


def load_model(model_path: Path = MODEL_PATH):
	return joblib.load(model_path)


def predict_outcomes(
	matches: pd.DataFrame,
	model_path: Path = MODEL_PATH,
	metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
	"""Predict 3-way probabilities for matches in a feature-compatible frame."""
	pipeline = load_model(model_path)
	metadata = joblib.load(metadata_path)
	target_map_inv = metadata.get("target_map_inv", TARGET_MAP_INV)

	prepared = prepare_training_frame(matches)
	feature_cols = [
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
	X = prepared[feature_cols]
	proba = pipeline.predict_proba(X)
	pred = pipeline.predict(X)

	result = prepared[["date", "home_team", "away_team", "tournament"]].copy()
	result["p_home_win"] = proba[:, 0]
	result["p_draw"] = proba[:, 1]
	result["p_away_win"] = proba[:, 2]
	result["pred_class"] = pred
	result["pred_outcome"] = result["pred_class"].map(target_map_inv)
	return result
