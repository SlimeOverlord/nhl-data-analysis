import os
import warnings
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
from ift6758.data.load_datasets import load_training_and_test_sets
from ift6758.data.data_cleaning import feature_engineering_1
from xgboost import XGBClassifier


def main_xgboost_baseline():
	# Train and evaluate an XGBoost model using distance + angle features and the training set data.
	training_set, test_set = load_training_and_test_sets()
	df = feature_engineering_1(training_set)

	# Clean & filter
	df = df[df["distance_from_goal"] <= 100]
	df = df.dropna(subset=["distance_from_goal", "angle_from_goal", "is_goal"]).copy()

	y = df["is_goal"].astype(int).values
	X = df[["distance_from_goal", "angle_from_goal"]].values

	X_train, X_val, y_train, y_val = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	model = XGBClassifier()
	model.fit(X_train, y_train)

	y_proba = model.predict_proba(X_val)[:, 1]
	auc = roc_auc_score(y_val, y_proba)
	acc = accuracy_score(y_val, (y_proba > 0.5).astype(int))

	run = wandb.init(
		entity="IFT6758-2025-A09",
		project="ift6758-milestone2",
		name="xgb_distance_angle_base",
		tags=["advanced", "xgboost", "distance_angle"],
		config={"model": "XGBClassifier", "features": ["distance_from_goal", "angle_from_goal"]},
		reinit=True,
	)
	wandb.log({"AUC": auc, "accuracy": acc})
	print(f"XGB Distance+Angle — AUC: {auc:.4f} | Accuracy: {acc:.4f}")
	print("W&B run:", run.get_url())

	# Persist model locally
	models_dir = os.path.join(os.getcwd(), "models")
	os.makedirs(models_dir, exist_ok=True)
	model_path = os.path.join(models_dir, "xgb_distance_angle_model.pkl")
	import joblib
	joblib.dump(model, model_path)
	print("Model saved:", model_path)

	# Log model as W&B artifact for reproducibility
	artifact = wandb.Artifact("xgb_distance_angle_baseline", type="model", metadata={
		"features": ["distance_from_goal", "angle_from_goal"],
		"roc_auc": auc,
		"accuracy": acc,
		"framework": "xgboost",
	})
	artifact.add_file(model_path)
	wandb.log_artifact(artifact)
	print("Model artifact logged to W&B")

	figures_dir = os.path.join(os.getcwd(), "figures", "advanced_models")
	os.makedirs(figures_dir, exist_ok=True)
	sns.set_theme(style="whitegrid")
	color = sns.color_palette("Set2")[0]

	# ROC
	fpr, tpr, _ = roc_curve(y_val, y_proba)
	fig = plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, color=color, label=f"XGB (AUC={auc:.3f})")
	plt.plot([0, 1], [0, 1], "--", color="gray", label="random (AUC=0.5)")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve — XGBoost (distance+angle)")
	plt.grid(True)
	plt.legend()
	path = os.path.join(figures_dir, "xgb_roc_curves.png")
	fig.savefig(path, bbox_inches="tight")
	print("Saved:", path)
	plt.close(fig)

	# Goal rate vs percentile
	df_val = pd.DataFrame({"y_true": y_val, "y_proba": y_proba})
	df_val["percentile"] = pd.qcut(df_val["y_proba"], q=20, labels=False, duplicates="drop")
	goal_rate = df_val.groupby("percentile")["y_true"].mean().reset_index()
	goal_rate["percentile"] = goal_rate["percentile"] * 5
	goal_rate = goal_rate.sort_values("percentile", ascending=False)
	goal_rate["goal_rate_percent"] = goal_rate["y_true"] * 100
	fig = plt.figure(figsize=(7, 5))
	plt.plot(goal_rate["percentile"], goal_rate["goal_rate_percent"], color=color)
	plt.title("Goal Rate vs Shot Probability Model Percentile — XGBoost")
	plt.xlabel("Shot Probability Model Percentile")
	plt.ylabel("Goals [%]")
	plt.ylim(0, 100)
	plt.grid(True)
	plt.gca().invert_xaxis()
	path = os.path.join(figures_dir, "xgb_goal_rate_vs_percentile.png")
	fig.savefig(path, bbox_inches="tight")
	print("Saved:", path)
	plt.close(fig)

	# Cumulative % of goals
	df_sorted = df_val.sort_values("y_proba", ascending=False).reset_index(drop=True)
	df_sorted["cum_goals"] = df_sorted["y_true"].cumsum()
	total_goals = df_sorted["y_true"].sum()
	df_sorted["cum_goal_rate"] = (
		df_sorted["cum_goals"] / total_goals * 100 if total_goals > 0 else 0
	)
	df_sorted["percentile"] = 100 - (np.arange(1, len(df_sorted) + 1) / len(df_sorted) * 100)
	fig = plt.figure(figsize=(7, 5))
	plt.plot(df_sorted["percentile"], df_sorted["cum_goal_rate"], color=color)
	plt.gca().invert_xaxis()
	plt.plot([100, 0], [0, 100], "--", color="gray", label="random baseline")
	plt.xlabel("Shot Probability Model Percentile")
	plt.ylabel("Cumulative proportion of goals [%]")
	plt.title("Cumulative % of Goals — XGBoost")
	plt.grid(True)
	plt.legend()
	path = os.path.join(figures_dir, "xgb_cumulative_goals.png")
	fig.savefig(path, bbox_inches="tight")
	print("Saved:", path)
	plt.close(fig)

	# Calibration plot
	prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
	fig = plt.figure(figsize=(6, 5))
	plt.plot(prob_pred, prob_true, marker="o", color=color, label="XGB")
	plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
	plt.title("Calibration Plot — XGBoost (distance+angle)")
	plt.xlabel("Predicted probability")
	plt.ylabel("Observed frequency")
	plt.grid(True)
	plt.legend()
	path = os.path.join(figures_dir, "xgb_calibration_plot.png")
	fig.savefig(path, bbox_inches="tight")
	print("Saved:", path)
	plt.close(fig)


	wandb.finish()


if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main_xgboost_baseline()
