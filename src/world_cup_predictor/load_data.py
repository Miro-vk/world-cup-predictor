from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_rankings(path: Path | None = None) -> pd.DataFrame:
	"""Load FIFA rankings data from CSV."""
	csv_path = path or RAW_DATA_DIR / "rankings.csv"
	return pd.read_csv(csv_path, parse_dates=["ranking_date"])


def load_results(path: Path | None = None) -> pd.DataFrame:
	"""Load match results data from CSV."""
	csv_path = path or RAW_DATA_DIR / "results.csv"
	return pd.read_csv(csv_path, parse_dates=["date"])


def load_raw_data(
	rankings_path: Path | None = None,
	results_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Load rankings and results datasets."""
	rankings = load_rankings(rankings_path)
	results = load_results(results_path)
	return rankings, results
