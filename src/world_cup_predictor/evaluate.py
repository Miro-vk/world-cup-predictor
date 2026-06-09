from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

from world_cup_predictor.predict import predict_outcomes
from world_cup_predictor.train import PROCESSED_DATA_PATH, run_training


def evaluate_model(data_path: Path = PROCESSED_DATA_PATH) -> dict[str, float | str | list[list[float]] | list[str]]:
	outputs = run_training(data_path)
	test_df = outputs["test_df"]
	preds = predict_outcomes(test_df)

	y_true = test_df["target"]
	y_pred = preds["pred_class"]
	y_proba = preds[["p_home_win", "p_draw", "p_away_win"]]
	labels = [0, 1, 2]
	label_names = ["H", "D", "A"]

	acc = accuracy_score(y_true, y_pred)
	f1_macro = f1_score(y_true, y_pred, average="macro")
	ll = log_loss(y_true, y_proba)
	report = classification_report(y_true, y_pred, digits=4)
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

	return {
		"accuracy": float(acc),
		"macro_f1": float(f1_macro),
		"log_loss": float(ll),
		"classification_report": report,
		"confusion_matrix": cm.tolist(),
		"confusion_matrix_normalized": cm_norm.tolist(),
		"class_labels": label_names,
	}


if __name__ == "__main__":
	metrics = evaluate_model()
	print(f"Accuracy: {metrics['accuracy']:.4f}")
	print(f"Macro F1: {metrics['macro_f1']:.4f}")
	print(f"Log loss: {metrics['log_loss']:.4f}")
	print("Classification report:")
	print(metrics["classification_report"])
	print("Confusion matrix (rows=true, cols=pred; H,D,A):")
	for row in metrics["confusion_matrix"]:
		print(row)
	print("Normalized confusion matrix (rows sum to 1; H,D,A):")
	for row in metrics["confusion_matrix_normalized"]:
		print([round(float(v), 4) for v in row])
