from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from world_cup_predictor.features import add_recent_form_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_with_rankings.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "match_outcome_model.joblib"
METADATA_PATH = MODELS_DIR / "match_outcome_metadata.joblib"

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_MAP_INV = {v: k for k, v in TARGET_MAP.items()}


def add_target(matches: pd.DataFrame) -> pd.DataFrame:
	"""Create 3-way target: H (home win), D (draw), A (away win)."""
	df = matches.copy()
	df["outcome"] = "D"
	df.loc[df["home_score"] > df["away_score"], "outcome"] = "H"
	df.loc[df["home_score"] < df["away_score"], "outcome"] = "A"
	df["target"] = df["outcome"].map(TARGET_MAP).astype("int64")
	return df


def prepare_training_frame(matches: pd.DataFrame) -> pd.DataFrame:
	"""Prepare training data with requested baseline features."""
	df = matches.copy()
	if "home_recent_form_points" not in df.columns or "away_recent_form_points" not in df.columns:
		df = add_recent_form_features(df, window=5)
	if "target" not in df.columns:
		df = add_target(df)
	if "recent_form_diff" not in df.columns:
		df["recent_form_diff"] = df["home_recent_form_points"] - df["away_recent_form_points"]

	# Tournament importance mapping: higher is more important.
	tournament_weights = {
		"FIFA World Cup": 5,
		"UEFA Euro": 4,
		"Copa América": 4,
		"AFC Asian Cup": 4,
		"African Cup of Nations": 4,
		"FIFA World Cup qualification": 3,
		"UEFA Nations League": 2,
		"Friendly": 1,
	}
	if "tournament_importance" not in df.columns:
		df["tournament_importance"] = df["tournament"].map(tournament_weights).fillna(2).astype(float)

	df["home_rank_missing"] = df["home_rank"].isna().astype(float)
	df["away_rank_missing"] = df["away_rank"].isna().astype(float)
	df["neutral"] = df["neutral"].astype(float)
	return df


def train_model(train_df: pd.DataFrame) -> Pipeline:
	feature_cols_numeric = [
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
	]
	feature_cols_categorical = ["tournament"]

	preprocessor = ColumnTransformer(
		transformers=[
			(
				"num",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="median")),
						("scaler", StandardScaler()),
					]
				),
				feature_cols_numeric,
			),
			(
				"cat",
				Pipeline(
					steps=[
						("imputer", SimpleImputer(strategy="most_frequent")),
						(
							"onehot",
							OneHotEncoder(handle_unknown="ignore", sparse_output=False),
						),
					]
				),
				feature_cols_categorical,
			),
		]
	)

	model = LogisticRegression(max_iter=1200)

	pipeline = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("model", model),
		]
	)

	feature_cols = feature_cols_numeric + feature_cols_categorical
	X = train_df[feature_cols]
	y = train_df["target"]
	pipeline.fit(X, y)
	return pipeline


def split_time(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
	sorted_df = df.sort_values("date").reset_index(drop=True)
	cutoff_idx = int(len(sorted_df) * train_ratio)
	return sorted_df.iloc[:cutoff_idx].copy(), sorted_df.iloc[cutoff_idx:].copy()


def run_training(data_path: Path = PROCESSED_DATA_PATH) -> dict[str, object]:
	raw = pd.read_csv(data_path, parse_dates=["date", "home_ranking_date", "away_ranking_date"])
	prepared = prepare_training_frame(raw)
	train_df, test_df = split_time(prepared, train_ratio=0.8)

	pipeline = train_model(train_df)

	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, MODEL_PATH)
	metadata = {
		"target_map": TARGET_MAP,
		"target_map_inv": TARGET_MAP_INV,
		"feature_version": "v1_home_away_rank_form_tournament",
		"train_rows": len(train_df),
		"test_rows": len(test_df),
	}
	joblib.dump(metadata, METADATA_PATH)

	return {
		"model_path": MODEL_PATH,
		"metadata_path": METADATA_PATH,
		"train_df": train_df,
		"test_df": test_df,
		"pipeline": pipeline,
	}


if __name__ == "__main__":
	outputs = run_training()
	print(f"Trained model saved to {outputs['model_path']}")
	print(f"Train rows: {len(outputs['train_df'])}, Test rows: {len(outputs['test_df'])}")
