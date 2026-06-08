from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, log_loss

from world_cup_predictor.predict import predict_outcomes
from world_cup_predictor.train import PROCESSED_DATA_PATH, run_training


def evaluate_model(data_path: Path = PROCESSED_DATA_PATH) -> dict[str, float | str]:
	outputs = run_training(data_path)
	test_df = outputs["test_df"]
	preds = predict_outcomes(test_df)

	y_true = test_df["target"]
	y_pred = preds["pred_class"]
	y_proba = preds[["p_home_win", "p_draw", "p_away_win"]]

	acc = accuracy_score(y_true, y_pred)
	ll = log_loss(y_true, y_proba)
	report = classification_report(y_true, y_pred, digits=4)

	return {
		"accuracy": float(acc),
		"log_loss": float(ll),
		"classification_report": report,
	}


if __name__ == "__main__":
	metrics = evaluate_model()
	print(f"Accuracy: {metrics['accuracy']:.4f}")
	print(f"Log loss: {metrics['log_loss']:.4f}")
	print("Classification report:")
	print(metrics["classification_report"])
