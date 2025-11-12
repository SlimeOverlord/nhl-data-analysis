import os
import pandas as pd
import numpy as np
import wandb
from ift6758.data.load_datasets import load_feature_engineered2_training_and_test_sets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from feature_engine.encoding import MeanEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

# Global config
RANDOM_STATE = 42
WANDB_ENTITY = "IFT6758-2025-A09"
WANDB_PROJECT = "ift6758-milestone2"

print("=" * 80)
print("                PART 6: BEST MODELS (IMPROVED VERSION)")
print("=" * 80)
print("\nPerformance Improvements:")
print("  - MeanEncoder: smoothing=1 (was 'auto') - increased variance")
print("  - RecursiveFeatureElimination: Select top 10/15 features")
print("  - RandomForest: 15 trials, n_estimators 50-150, max_depth 5-15")
print("  - Target: AUC > 0.7603 (baseline)")
print("=" * 80)

# 1. Load data
print("\n" + "=" * 80)
print("Loading Feature Engineering 2 dataset...")
print("=" * 80)
train_df, test_df = load_feature_engineered2_training_and_test_sets()

print(f"Dataset loaded: {len(train_df)} samples, {len(train_df.columns)} features")
print(f"Features: {list(train_df.columns)}")

# Separate features and target
X = train_df.drop('is_goal', axis=1)
y = train_df['is_goal']

print(f"\nTarget distribution:")
print(f"  Goals: {y.sum()} ({y.mean()*100:.2f}%)")
print(f"  Non-goals: {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)")

# Convert boolean columns to int
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    print(f"\nConverting boolean columns to int: {bool_cols}")
    X[bool_cols] = X[bool_cols].astype(int)

# Clean inf/NaN
print("\nCleaning inf and NaN values...")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
nan_count = X.isna().sum().sum()
if nan_count > 0:
    initial_len = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    print(f"Dropped {initial_len - len(X)} rows with NaN values")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# 2. Feature Engineering: ONLY MeanEncoder (NO ProbeFeatureSelection)
print("\n" + "=" * 80)
print("Feature Engineering: MeanEncoder Only")
print("=" * 80)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nCategorical columns found: {categorical_cols}")

if categorical_cols:
    print(f"Applying MeanEncoder with smoothing=1 (increased variance)...")
    encoder = MeanEncoder(variables=categorical_cols, smoothing=1)
    X_train = encoder.fit_transform(X_train, y_train)
    X_val = encoder.transform(X_val)
    print(f"Encoding complete. Sample encoded values:")
    for col in categorical_cols:
        print(f"  {col}: {{'min': {X_train[col].min()}, 'max': {X_train[col].max()}, 'mean': {X_train[col].mean()}}}")

# RecursiveFeatureElimination (RFE) for feature selection
print("\n" + "=" * 80)
print("Applying RecursiveFeatureElimination (RFE)")
print("=" * 80)
print(f"Starting with {len(X_train.columns)} features")

# Scale features before RFE (important for LogisticRegression convergence)
print("\nScaling features for RFE...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

rfe_estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=10, step=1, verbose=1)
print("\nFitting RFE (this may take a minute)...")
rfe.fit(X_train_scaled_df, y_train)

selected_features = X_train.columns[rfe.support_].tolist()
rejected_features = X_train.columns[~rfe.support_].tolist()

print(f"\n✓ RFE complete!")
print(f"  Selected {len(selected_features)} features: {selected_features}")
print(f"  Rejected {len(rejected_features)} features: {rejected_features}")

# Apply feature selection to original (unscaled) data
X_train = X_train[selected_features]
X_val = X_val[selected_features]

# 3. Train XGBoost (unchanged - already fast)
def train_xgboost_optuna(X_train, y_train, X_val, y_val, selected_features):
    print("\n" + "=" * 80)
    print("Training XGBoost with Optuna Optimization")
    print("=" * 80)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="part6_xgboost_optuna_fast",
        tags=["part6", "best-models", "xgboost", "optuna", "fast", "no-feature-selection"],
        config={
            "model": "XGBoost",
            "features": selected_features,
            "n_features": len(selected_features),
            "random_state": RANDOM_STATE,
            "optimization": "optuna",
            "version": "fast"
        },
        reinit=True
    )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**params)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
        return cv_scores.mean()

    print("\nRunning Optuna optimization (40 trials)...")
    wandb_callback = WeightsAndBiasesCallback(wandb_kwargs={"project": WANDB_PROJECT}, as_multirun=False)
    study = optuna.create_study(direction="maximize", study_name="xgboost_study")
    study.optimize(objective, n_trials=40, callbacks=[wandb_callback], show_progress_bar=True)

    best_params = study.best_trial.params
    best_cv_auc = study.best_trial.value

    print(f"\nOptimization complete!")
    print(f"Best CV AUC: {best_cv_auc:.4f}")
    print(f"Best parameters: {best_params}")

    print("\nTraining final model with best parameters...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, (y_val_proba > 0.5).astype(int))

    print(f"\nValidation Results:")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")

    wandb.log({
        "best_cv_auc": best_cv_auc,
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "best_params": best_params,
        "n_features": len(selected_features)
    })

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "part6_xgboost_fast.pkl")
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")

    artifact = wandb.Artifact(
        name="part6_xgboost_model_fast",
        type="model",
        description="XGBoost model (fast version, no feature selection)",
        metadata={
            "n_features": len(selected_features),
            "features": selected_features,
            "val_auc": val_auc,
            "val_accuracy": val_accuracy,
            "best_cv_auc": best_cv_auc,
            "model_type": "XGBoost",
            "hyperparameters": best_params,
            "version": "fast"
        }
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"Model artifact logged to Wandb")
    print(f"\nWandb run URL: {run.get_url()}")
    wandb.finish()

    return final_model, best_params, val_auc, val_accuracy, y_val_proba

# 4. Train HistGradientBoosting (unchanged - already fast)
def train_histgb_optuna(X_train, y_train, X_val, y_val, selected_features):
    print("\n" + "=" * 80)
    print("Training HistGradientBoosting with Optuna Optimization")
    print("=" * 80)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="part6_histgb_optuna_fast",
        tags=["part6", "best-models", "histgb", "optuna", "fast", "no-feature-selection"],
        config={
            "model": "HistGradientBoosting",
            "features": selected_features,
            "n_features": len(selected_features),
            "random_state": RANDOM_STATE,
            "optimization": "optuna",
            "version": "fast"
        },
        reinit=True
    )

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
            'max_bins': trial.suggest_int('max_bins', 128, 255),
            'random_state': RANDOM_STATE
        }

        model = HistGradientBoostingClassifier(**params)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
        return cv_scores.mean()

    print("\nRunning Optuna optimization (40 trials)...")
    wandb_callback = WeightsAndBiasesCallback(wandb_kwargs={"project": WANDB_PROJECT}, as_multirun=False)
    study = optuna.create_study(direction="maximize", study_name="histgb_study")
    study.optimize(objective, n_trials=40, callbacks=[wandb_callback], show_progress_bar=True)

    best_params = study.best_trial.params
    best_cv_auc = study.best_trial.value

    print(f"\nOptimization complete!")
    print(f"Best CV AUC: {best_cv_auc:.4f}")
    print(f"Best parameters: {best_params}")

    print("\nTraining final model with best parameters...")
    final_model = HistGradientBoostingClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, (y_val_proba > 0.5).astype(int))

    print(f"\nValidation Results:")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")

    wandb.log({
        "best_cv_auc": best_cv_auc,
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "best_params": best_params,
        "n_features": len(selected_features)
    })

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "part6_histgb_fast.pkl")
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")

    artifact = wandb.Artifact(
        name="part6_histgb_model_fast",
        type="model",
        description="HistGradientBoosting model (fast version, no feature selection)",
        metadata={
            "n_features": len(selected_features),
            "features": selected_features,
            "val_auc": val_auc,
            "val_accuracy": val_accuracy,
            "best_cv_auc": best_cv_auc,
            "model_type": "HistGradientBoosting",
            "hyperparameters": best_params,
            "version": "fast"
        }
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"Model artifact logged to Wandb")
    print(f"\nWandb run URL: {run.get_url()}")
    wandb.finish()

    return final_model, best_params, val_auc, val_accuracy, y_val_proba

# 5. Train RandomForest (OPTIMIZED - much faster)
def train_rf_optuna_fast(X_train, y_train, X_val, y_val, selected_features):
    print("\n" + "=" * 80)
    print("Training RandomForest with Optuna Optimization (FAST VERSION)")
    print("=" * 80)
    print("Optimizations: 15 trials (was 40), n_estimators 50-150 (was 100-500), max_depth 5-15 (was 5-30)")

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="part6_rf_optuna_fast",
        tags=["part6", "best-models", "random-forest", "optuna", "fast", "no-feature-selection"],
        config={
            "model": "RandomForest",
            "features": selected_features,
            "n_features": len(selected_features),
            "random_state": RANDOM_STATE,
            "optimization": "optuna_fast",
            "n_trials": 15,
            "version": "fast"
        },
        reinit=True
    )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),  # REDUCED from 100-500
            'max_depth': trial.suggest_int('max_depth', 5, 15),          # REDUCED from 5-30
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # REDUCED from 2-20
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),     # REDUCED from 1-10
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # REMOVED None
            'bootstrap': True,  # FIXED to True
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc', n_jobs=-1)
        return cv_scores.mean()

    print("\nRunning Optuna optimization (15 trials)...")  # REDUCED from 40
    wandb_callback = WeightsAndBiasesCallback(wandb_kwargs={"project": WANDB_PROJECT}, as_multirun=False)
    study = optuna.create_study(direction="maximize", study_name="rf_study_fast")
    study.optimize(objective, n_trials=15, callbacks=[wandb_callback], show_progress_bar=True)

    best_params = study.best_trial.params
    best_cv_auc = study.best_trial.value

    print(f"\nOptimization complete!")
    print(f"Best CV AUC: {best_cv_auc:.4f}")
    print(f"Best parameters: {best_params}")

    print("\nTraining final model with best parameters...")
    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, (y_val_proba > 0.5).astype(int))

    print(f"\nValidation Results:")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")

    wandb.log({
        "best_cv_auc": best_cv_auc,
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "best_params": best_params,
        "n_features": len(selected_features)
    })

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "part6_rf_fast.pkl")
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")

    artifact = wandb.Artifact(
        name="part6_rf_model_fast",
        type="model",
        description="RandomForest model (fast version, no feature selection)",
        metadata={
            "n_features": len(selected_features),
            "features": selected_features,
            "val_auc": val_auc,
            "val_accuracy": val_accuracy,
            "best_cv_auc": best_cv_auc,
            "model_type": "RandomForest",
            "hyperparameters": best_params,
            "version": "fast"
        }
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"Model artifact logged to Wandb")
    print(f"\nWandb run URL: {run.get_url()}")
    wandb.finish()

    return final_model, best_params, val_auc, val_accuracy, y_val_proba

# 6. Train VotingClassifier
def train_voting_classifier(xgb_model, histgb_model, rf_model, X_train, y_train, X_val, y_val, selected_features):
    print("\n" + "=" * 80)
    print("Training VotingClassifier Ensemble")
    print("=" * 80)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="part6_voting_fast",
        tags=["part6", "best-models", "voting", "ensemble", "fast", "no-feature-selection"],
        config={
            "model": "VotingClassifier",
            "estimators": ["XGBoost", "HistGradientBoosting", "RandomForest"],
            "voting": "soft",
            "features": selected_features,
            "n_features": len(selected_features),
            "version": "fast"
        },
        reinit=True
    )

    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('histgb', histgb_model),
            ('rf', rf_model)
        ],
        voting='soft',
        n_jobs=-1
    )

    print("\nTraining ensemble...")
    voting_clf.fit(X_train, y_train)

    y_val_proba = voting_clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, (y_val_proba > 0.5).astype(int))

    print(f"\nValidation Results:")
    print(f"  AUC: {val_auc:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")

    wandb.log({
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "n_features": len(selected_features)
    })

    models_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(models_dir, "part6_voting_fast.pkl")
    joblib.dump(voting_clf, model_path)
    print(f"\nModel saved to: {model_path}")

    artifact = wandb.Artifact(
        name="part6_voting_model_fast",
        type="model",
        description="VotingClassifier ensemble (fast version, no feature selection)",
        metadata={
            "n_features": len(selected_features),
            "features": selected_features,
            "val_auc": val_auc,
            "val_accuracy": val_accuracy,
            "model_type": "VotingClassifier",
            "estimators": ["XGBoost", "HistGradientBoosting", "RandomForest"],
            "voting": "soft",
            "version": "fast"
        }
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"Model artifact logged to Wandb")
    print(f"\nWandb run URL: {run.get_url()}")
    wandb.finish()

    return voting_clf, val_auc, val_accuracy, y_val_proba

# Main execution
if __name__ == "__main__":
    # Train models
    xgb_model, xgb_params, xgb_auc, xgb_acc, xgb_proba = train_xgboost_optuna(
        X_train, y_train, X_val, y_val, selected_features
    )

    histgb_model, histgb_params, histgb_auc, histgb_acc, histgb_proba = train_histgb_optuna(
        X_train, y_train, X_val, y_val, selected_features
    )

    rf_model, rf_params, rf_auc, rf_acc, rf_proba = train_rf_optuna_fast(
        X_train, y_train, X_val, y_val, selected_features
    )

    voting_model, voting_auc, voting_acc, voting_proba = train_voting_classifier(
        xgb_model, histgb_model, rf_model, X_train, y_train, X_val, y_val, selected_features
    )

    # Print summary
    print("\n" + "=" * 80)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 80)
    print("\nFinal Results Summary:")
    print(f"  XGBoost:              AUC = {xgb_auc:.4f}, Accuracy = {xgb_acc:.4f}")
    print(f"  HistGradientBoosting: AUC = {histgb_auc:.4f}, Accuracy = {histgb_acc:.4f}")
    print(f"  RandomForest:         AUC = {rf_auc:.4f}, Accuracy = {rf_acc:.4f}")
    print(f"  VotingClassifier:     AUC = {voting_auc:.4f}, Accuracy = {voting_acc:.4f}")
    print(f"\nAll models saved to: {os.path.join(os.getcwd(), 'models')}")
    print(f"Features used: {len(selected_features)} (all features, no selection)")
    print("=" * 80)
