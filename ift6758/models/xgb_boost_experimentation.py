from ift6758.data.load_datasets import load_feature_engineered2_training_and_test_sets
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import wandb
import optuna
from optuna.integration import WeightsAndBiasesCallback  # requires optuna[wandb] or optuna + wandb installed

def main_xgboost_experimentation():
	# Initialize W&B run
	run = wandb.init(
		entity="IFT6758-2025-A09",
		project="ift6758-milestone2",
		name="xgb_best_all_features_optuna",
		tags=["advanced", "xgboost", "optuna", "all-features"],
		reinit=True,
	)
	
	train_df, test_df = load_feature_engineered2_training_and_test_sets()
	X = train_df.drop(columns=["is_goal"])
	y = train_df["is_goal"]

	# Encode categorical columns as numeric for XGBoost
	categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
	if categorical_cols:
		print(f"Encoding categorical columns: {categorical_cols}")
		for col in categorical_cols:
			X[col] = X[col].astype('category').cat.codes

	X_train, X_val, y_train, y_val = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	# Replace inf values and drop rows with NaNs in train only
	X_train = X_train.replace([np.inf, -np.inf], np.nan)
	X_train = X_train.dropna()
	y_train = y_train.loc[X_train.index]
	# Align validation similarly 
	X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna()
	y_val = y_val.loc[X_val.index]

	# Define Optuna objective function for XGBoost hyperparameter tuning
	def objective(trial):
		params = {
			# Common GridSearch-style space, now explored via Optuna
			"n_estimators": trial.suggest_int("n_estimators", 100, 400),
			"max_depth": trial.suggest_int("max_depth", 3, 7),
			"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
			"subsample": trial.suggest_float("subsample", 0.7, 1.0),
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
			"random_state": 42,
			"eval_metric": "logloss",
		}
		
		# Use StratifiedKFold for robust CV
		kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
		model = XGBClassifier(**params)
		
		# Cross-validation with ROC AUC scoring
		cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="roc_auc", n_jobs=-1)
		return cv_scores.mean()

	# Run Optuna study with W&B integration
	wandb_callback = WeightsAndBiasesCallback(wandb_kwargs={"project": "ift6758-milestone2"}, as_multirun=False)
	
	study = optuna.create_study(
		direction="maximize",
		sampler=optuna.samplers.TPESampler(seed=42),
		pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
	)
	
	print("Starting Optuna hyperparameter search...")
	study.optimize(objective, n_trials=10, callbacks=[wandb_callback], show_progress_bar=True)

	print("\nBest trial:")
	print(f"  Value (CV AUC): {study.best_trial.value:.4f}")
	print("  Params:")
	for k, v in study.best_trial.params.items():
		print(f"{k}: {v}")

	# Log best params and CV score to W&B
	wandb.log({"best_cv_auc": study.best_trial.value, "best_params": study.best_trial.params})

	# Train final model with best hyperparameters on full training set
	best_params = study.best_trial.params.copy()
	best_params.update({"random_state": 42, "eval_metric": "logloss", "use_label_encoder": False})
	best_model = XGBClassifier(**best_params)
	best_model.fit(X_train, y_train)
	y_val_proba = best_model.predict_proba(X_val)[:, 1]
	val_auc = roc_auc_score(y_val, y_val_proba)
	val_acc = accuracy_score(y_val, (y_val_proba > 0.5).astype(int))
	print(f"Validation AUC: {val_auc:.4f} | Validation Accuracy: {val_acc:.4f}")
	
	# Log validation metrics to W&B
	wandb.log({"val_auc": val_auc, "val_accuracy": val_acc})

	# Create figures directory inside advanced_models
	fig_dir = os.path.join(os.getcwd(), "figures", "advanced_models", "best_xgboost_model")
	os.makedirs(fig_dir, exist_ok=True)

	sns.set_theme(style="whitegrid")
	color = sns.color_palette("Set2")[1]

	# 1. ROC Curve
	fpr, tpr, _ = roc_curve(y_val, y_val_proba)
	plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, color=color, label=f"Best XGB (AUC={val_auc:.3f})")
	plt.plot([0, 1], [0, 1], "--", color="gray", label="random")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve — Best XGBoost")
	plt.legend()
	plt.grid(True)
	roc_path = os.path.join(fig_dir, "roc_curve.png")
	plt.savefig(roc_path, bbox_inches="tight")
	plt.close()
	print("Saved:", roc_path)

	# 2. Goal Rate vs Percentile
	df_val = pd.DataFrame({"y_true": y_val.values, "y_proba": y_val_proba})
	df_val["percentile"] = pd.qcut(df_val["y_proba"], q=20, labels=False, duplicates="drop")
	goal_rate = df_val.groupby("percentile")["y_true"].mean().reset_index()
	goal_rate["percentile"] = goal_rate["percentile"] * 5
	goal_rate = goal_rate.sort_values("percentile", ascending=False)
	goal_rate["goal_rate_percent"] = goal_rate["y_true"] * 100
	plt.figure(figsize=(7, 5))
	plt.plot(goal_rate["percentile"], goal_rate["goal_rate_percent"], color=color)
	plt.title("Goal Rate vs Shot Probability Model Percentile — Best XGBoost")
	plt.xlabel("Shot Probability Model Percentile")
	plt.ylabel("Goals [%]")
	plt.ylim(0, 100)
	plt.grid(True)
	plt.gca().invert_xaxis()
	goal_rate_path = os.path.join(fig_dir, "goal_rate_vs_percentile.png")
	plt.savefig(goal_rate_path, bbox_inches="tight")
	plt.close()
	print("Saved:", goal_rate_path)

	# 3. Cumulative % of Goals
	df_sorted = df_val.sort_values("y_proba", ascending=False).reset_index(drop=True)
	df_sorted["cum_goals"] = df_sorted["y_true"].cumsum()
	total_goals = df_sorted["y_true"].sum()
	df_sorted["cum_goal_rate"] = (df_sorted["cum_goals"] / total_goals * 100 if total_goals > 0 else 0)
	df_sorted["percentile"] = 100 - (np.arange(1, len(df_sorted) + 1) / len(df_sorted) * 100)
	plt.figure(figsize=(7, 5))
	plt.plot(df_sorted["percentile"], df_sorted["cum_goal_rate"], color=color)
	plt.gca().invert_xaxis()
	plt.plot([100, 0], [0, 100], "--", color="gray", label="random baseline")
	plt.xlabel("Shot Probability Model Percentile")
	plt.ylabel("Cumulative proportion of goals [%]")
	plt.title("Cumulative % of Goals — Best XGBoost")
	plt.grid(True)
	plt.legend()
	cumulative_path = os.path.join(fig_dir, "cumulative_goals.png")
	plt.savefig(cumulative_path, bbox_inches="tight")
	plt.close()
	print("Saved:", cumulative_path)

	# 4. Calibration Plot
	prob_true, prob_pred = calibration_curve(y_val, y_val_proba, n_bins=10)
	plt.figure(figsize=(6, 5))
	plt.plot(prob_pred, prob_true, marker="o", color=color, label="Best XGB")
	plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
	plt.title("Calibration Plot — Best XGBoost")
	plt.xlabel("Predicted probability")
	plt.ylabel("Observed frequency")
	plt.grid(True)
	plt.legend()
	calibration_path = os.path.join(fig_dir, "calibration_plot.png")
	plt.savefig(calibration_path, bbox_inches="tight")
	plt.close()
	print("Saved:", calibration_path)

	# Save model with joblib
	models_dir = os.path.join(os.getcwd(), "models")
	os.makedirs(models_dir, exist_ok=True)
	model_path = os.path.join(models_dir, "xgb_best_model.pkl")
	joblib.dump(best_model, model_path)
	print("Model saved:", model_path)

	# Upload model as W&B artifact
	artifact = wandb.Artifact("xgb_best_all_features", type="model")
	artifact.add_file(model_path)
	wandb.log_artifact(artifact)
	print("Model artifact logged to W&B")
	
	print("W&B run URL:", run.get_url())
	wandb.finish()

	return best_model, study

if __name__ == "__main__":
	main_xgboost_experimentation()


