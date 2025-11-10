from ift6758.data.load_datasets import load_feature_engineered2_training_and_test_sets
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import wandb
import optuna
from optuna.integration import WeightsAndBiasesCallback  
from sklearn.preprocessing import OrdinalEncoder



def load_and_prepare_data():
	"""
	Load feature-engineered dataset 2, encode categorical columns, 
	split into train/val, and clean inf/NaN values.
	
	Returns:
		tuple: (X_train, X_val, y_train, y_val, X_full, y_full)
			- X_train, X_val: cleaned training and validation feature sets
			- y_train, y_val: corresponding target labels
			- X_full, y_full: full dataset (before split) for reference
	"""
	print("Loading feature-engineered dataset 2...")
	train_df, test_df = load_feature_engineered2_training_and_test_sets()
	X = train_df.drop(columns=["is_goal"])
	y = train_df["is_goal"]
	
	# Train/validation split 
	X_raw = X  # keep original pre-encoding feature frame
	categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
	X_train_raw, X_val_raw, y_train, y_val = train_test_split(
		X_raw, y, test_size=0.2, random_state=42, stratify=y
	)

	# Ordinal encoding (fit only on train to avoid leakage)
	categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()
	if categorical_cols:
		print(f"Ordinal encoding: {categorical_cols}")
		encoder = OrdinalEncoder()
		X_train_raw[categorical_cols] = encoder.fit_transform(X_train_raw[categorical_cols])
		X_val_raw[categorical_cols] = encoder.transform(X_val_raw[categorical_cols])
		X_raw[categorical_cols] = encoder.transform(X_raw[categorical_cols])
		X_train, X_val, X_full = X_train_raw, X_val_raw, X_raw
	else:
		print("No categorical columns detected for encoding.")
		X_train, X_val, X_full = X_train_raw, X_val_raw, X_raw

	
	# Clean inf/NaN values if any
	X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
	y_train = y_train.loc[X_train.index]
	X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna()
	y_val = y_val.loc[X_val.index]
	X_full = X_full.replace([np.inf, -np.inf], np.nan).dropna()
	
	print(f"Dataset shape: {X_train.shape[0]} train samples, {X_val.shape[0]} val samples, {X_train.shape[1]} features")
	return X_train, X_val, y_train, y_val, X, y


def main_xgboost_experimentation():
	# Initialize W&B run
	run = wandb.init(
		entity="IFT6758-2025-A09",
		project="ift6758-milestone2",
		name="xgb_best_all_features_optuna",
		tags=["advanced", "xgboost", "optuna", "all-features"],
		reinit=True,
	)
	
	# Load and prepare data
	X_train, X_val, y_train, y_val, X, y = load_and_prepare_data()

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
	study.optimize(objective, n_trials=30, callbacks=[wandb_callback], show_progress_bar=True)

	print("\nBest trial:")
	print(f"Value (CV AUC): {study.best_trial.value:.4f}")
	print("Params:")
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
	fig_dir = os.path.join(os.getcwd(), "figures", "advanced_models", "best_xgboost_model_all_features")
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

	# Upload model as W&B artifact with metadata
	artifact = wandb.Artifact(
		"xgb_best_all_features", 
		type="model",
		metadata={
			"n_features": X_train.shape[1],
			"val_auc": val_auc,
			"val_accuracy": val_acc,
			"best_cv_auc": study.best_trial.value,
			"best_params": study.best_trial.params,
			"n_trials": len(study.trials),
			"method": "Optuna hyperparameter tuning with all features"
		}
	)
	artifact.add_file(model_path)
	wandb.log_artifact(artifact)
	print("Model artifact 'xgb_best_all_features' logged to W&B")
	
	print("W&B run URL:", run.get_url())
	wandb.finish()

	return best_model, study


def feature_selection():
	"""
	Feature selection experiment on feature_engineered_2 dataset.
	Uses correlation analysis and SelectKBest (filter method) to identify a feature subset.
	Trains XGBoost with Optuna and compares against full-feature baseline.
	"""
	# Initialize W&B
	run = wandb.init(
		entity="IFT6758-2025-A09",
		project="ift6758-milestone2",
		name="xgb_feature_selection_optuna",
		tags=["advanced", "xgboost", "feature-selection", "optuna"],
		reinit=True,
	)
	
	# Load and prepare data
	X_train, X_val, y_train, y_val, X, y = load_and_prepare_data()

	optuna_trials = 20
	cv_splits = 5
	k = 11
	
	# Create output directory
	fig_dir = os.path.join(os.getcwd(), "figures", "advanced_models", "feature_selection")
	os.makedirs(fig_dir, exist_ok=True)
	
	# 1. Correlation analysis
	print("\n Step 1: Correlation Analysis")
	corr = X_train.corr().abs()
	upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
	
	high_corr_pairs = []
	for column in upper_tri.columns:
		high_corr = upper_tri[column][upper_tri[column] > 0.8]
		if len(high_corr) > 0:
			for idx in high_corr.index:
				high_corr_pairs.append((column, idx, upper_tri.loc[idx, column]))
	
	print(f"Found {len(high_corr_pairs)} feature pairs with correlation > 0.8")
	
	# Plot correlation matrix
	n_features = min(30, len(X.columns))
	plt.figure(figsize=(12, 10))
	sns.heatmap(
		corr.iloc[:n_features, :n_features],
		cmap="coolwarm", 
		annot=False,
	)
	plt.title(f"Feature Correlation Matrix (top {n_features} features)")
	plt.tight_layout()
	corr_path = os.path.join(fig_dir, "feature_correlation_matrix.png")
	plt.savefig(corr_path, bbox_inches="tight")
	plt.close()
	print(f"Saved: {corr_path}")
	
	wandb.log({"high_correlation_pairs": len(high_corr_pairs)})
	
	# 2. Feature selection methods
	print("\nStep 2: Feature Selection")
	
	# SelectKBest 
	selector_kbest = SelectKBest(score_func=f_classif, k=k)
	selector_kbest.fit(X_train, y_train)
	selected_kbest = X_train.columns[selector_kbest.get_support()].tolist()
	scores_kbest = pd.DataFrame({
		'feature': X_train.columns,
		'score': selector_kbest.scores_
	}).sort_values('score', ascending=False)
	
	print(f"\nSelectKBest (k={k}) selected features:")
	for feature in selected_kbest:
		print(f"  - {feature}")

	# Plot SelectKBest importance
	top_features = scores_kbest
	plt.figure(figsize=(10, 8))
	plt.barh(range(len(top_features)), top_features['score'], color=sns.color_palette("viridis", len(top_features)))
	plt.yticks(range(len(top_features)), top_features['feature'])
	plt.xlabel('F-Score')
	plt.title('Top Features — SelectKBest')
	plt.gca().invert_yaxis()
	plt.tight_layout()
	kbest_path = os.path.join(fig_dir, "feature_importance_selectkbest.png")
	plt.savefig(kbest_path)
	plt.close()
	print(f"Saved: {kbest_path}")
	
	selected_features = selected_kbest
	print(f"\nUsing SelectKBest only ({len(selected_features)} features)")
	
	wandb.log({"n_selected_features": len(selected_features), "selected_features": selected_features})
	
	# 3. Train with Optuna on selected features
	print("\nStep 3: Hyperparameter Tuning on Selected Features")
	X_train_sel = X_train[selected_features]
	X_train_sel_cv = X_train[selected_features]
	
	def objective(trial):
		params = {
			"n_estimators": trial.suggest_int("n_estimators", 100, 400),
			"max_depth": trial.suggest_int("max_depth", 3, 7),
			"learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
			"subsample": trial.suggest_float("subsample", 0.7, 1.0),
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
			"random_state": 42,
			"eval_metric": "logloss",
		}
		
		kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
		model = XGBClassifier(**params)
		cv_scores = cross_val_score(model, X_train_sel_cv, y_train, cv=kfold, scoring="roc_auc", n_jobs=-1)
		return cv_scores.mean()
	
	study = optuna.create_study(
		direction="maximize",
		sampler=optuna.samplers.TPESampler(seed=42),
		pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
	)
	
	print(f"Starting Optuna search with {len(selected_features)} features (trials={optuna_trials}, cv={cv_splits})...")
	study.optimize(objective, n_trials=optuna_trials, show_progress_bar=True)
	
	print(f"\nBest CV AUC: {study.best_trial.value:.4f}")
	print("Best params:", study.best_trial.params)
	
	# Train final model
	best_params = study.best_trial.params.copy()
	best_params.update({"random_state": 42, "eval_metric": "logloss", "use_label_encoder": False})
	model_selected = XGBClassifier(**best_params)
	model_selected.fit(X_train_sel, y_train)
	
	y_val_proba_sel = model_selected.predict_proba(X_val[selected_features])[:, 1]
	val_auc_sel = roc_auc_score(y_val, y_val_proba_sel)
	val_acc_sel = accuracy_score(y_val, (y_val_proba_sel > 0.5).astype(int))
	
	print(f"Validation AUC (selected features): {val_auc_sel:.4f}")
	print(f"Validation Accuracy (selected features): {val_acc_sel:.4f}")
	
	wandb.log({
		"best_cv_auc_selected": study.best_trial.value,
		"val_auc_selected": val_auc_sel,
		"val_accuracy_selected": val_acc_sel,
		"best_params_selected": study.best_trial.params
	})
	
	# 4. Load or train baseline full-feature model for comparison
	print("\nStep 4: Comparison with Full Feature Model")
	models_dir = os.path.join(os.getcwd(), "models")
	full_model_path = os.path.join(models_dir, "xgb_best_model.pkl")
	
	if os.path.exists(full_model_path):
		print(f"Loading existing full-feature model from {full_model_path}")
		model_full = joblib.load(full_model_path)
	else:
		print("Training baseline full-feature model...")
		model_full = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
		model_full.fit(X_train, y_train)
	
	y_val_proba_full = model_full.predict_proba(X_val)[:, 1]
	auc_full = roc_auc_score(y_val, y_val_proba_full)
	
	print(f"Validation AUC (full features): {auc_full:.4f}")
	print(f"AUC improvement: {val_auc_sel - auc_full:.4f}")
	
	# Comparison plots
	sns.set_theme(style="whitegrid")
	palette = sns.color_palette("Set2")
	
	# ROC Curve comparison
	fpr_full, tpr_full, _ = roc_curve(y_val, y_val_proba_full)
	fpr_sel, tpr_sel, _ = roc_curve(y_val, y_val_proba_sel)
	
	plt.figure(figsize=(7, 6))
	plt.plot(fpr_full, tpr_full, color=palette[0], label=f"Full features (AUC={auc_full:.3f})")
	plt.plot(fpr_sel, tpr_sel, color=palette[1], label=f"Selected features (AUC={val_auc_sel:.3f})")
	plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve — Feature Selection Comparison")
	plt.legend()
	plt.grid(True)
	roc_path = os.path.join(fig_dir, "roc_comparison.png")
	plt.savefig(roc_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"Saved: {roc_path}")
	
	# Calibration comparison
	prob_true_full, prob_pred_full = calibration_curve(y_val, y_val_proba_full, n_bins=10)
	prob_true_sel, prob_pred_sel = calibration_curve(y_val, y_val_proba_sel, n_bins=10)
	
	plt.figure(figsize=(7, 6))
	plt.plot(prob_pred_full, prob_true_full, marker="o", color=palette[0], label="Full features")
	plt.plot(prob_pred_sel, prob_true_sel, marker="s", color=palette[1], label="Selected features")
	plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
	plt.xlabel("Predicted probability")
	plt.ylabel("Observed frequency")
	plt.title("Calibration Plot — Feature Selection Comparison")
	plt.legend()
	plt.grid(True)
	cal_path = os.path.join(fig_dir, "calibration_comparison.png")
	plt.savefig(cal_path, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"Saved: {cal_path}")
	
	wandb.log({
		"val_auc_full_features": auc_full,
		"feature_reduction_ratio": len(selected_features) / X_train.shape[1]
	})
	
	# 5. Save selected-feature model
	os.makedirs(models_dir, exist_ok=True)
	model_path = os.path.join(models_dir, "xgb_feature_selected_model.pkl")
	joblib.dump(model_selected, model_path)
	print(f"\nModel saved: {model_path}")
	
	# Save feature list
	feature_list_path = os.path.join(models_dir, "selected_features.txt")
	with open(feature_list_path, "w") as f:
		f.write("\n".join(selected_features))
	print(f"Selected features saved: {feature_list_path}")
	
	# 6. Upload as W&B artifact
	artifact = wandb.Artifact(
		"xgb_feature_selected_model",
		type="model",
		metadata={
			"n_features": len(selected_features),
			"features": selected_features,
			"val_auc": val_auc_sel,
			"val_accuracy": val_acc_sel,
			"best_params": study.best_trial.params,
		"method": "SelectKBest (filter method)"
		}
	)
	artifact.add_file(model_path)
	artifact.add_file(feature_list_path)
	wandb.log_artifact(artifact)
	print("Model artifact 'xgb_feature_selected_model' logged to W&B")
	
	print(f"\nW&B run URL: {run.get_url()}")
	wandb.finish()
	
	return model_selected, selected_features, study


if __name__ == "__main__":
	main_xgboost_experimentation()
	feature_selection()
