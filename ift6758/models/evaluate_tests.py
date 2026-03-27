import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.calibration import calibration_curve
import json

from ift6758.data.load_datasets import (
    load_feature_engineered2_training_and_test_sets,
    load_training_and_test_sets,
)
from ift6758.data.data_cleaning import feature_engineering_1
from xgb_boost_experimentation import load_and_prepare_data
from feature_engine.encoding import MeanEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _download_models_from_runs(entity: str, project: str, dest_dir: str):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    os.makedirs(dest_dir, exist_ok=True)
    downloaded = []
    # discover existing model files to avoid re-downloading
    existing = {}
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.endswith('.pkl') or f.endswith('.joblib'):
                existing[f] = os.path.join(root, f)
    existing_basenames = set(existing.keys())
    existing_paths = set(existing.values())
    for run in runs:
        try:
            arts = []
            try:
                arts = list(run.logged_artifacts())
            except Exception:
                arts = []

            # prefer artifacts (model artifacts)
            if arts:
                for a in arts:
                    try:
                        # try to inspect artifact files without downloading when possible
                        try:
                            art_files = [f.name for f in a.files()]
                        except Exception:
                            art_files = []

                        # if any of the artifact files already exist locally (by basename)
                        already = False
                        for fname in art_files:
                            if fname in existing_basenames:
                                downloaded.append(existing[fname])
                                already = True
                        # also skip if an existing path contains the run id or artifact name
                        art_ident = getattr(a, 'name', None) or getattr(a, 'id', None)
                        if not already and art_ident:
                            for p in existing_paths:
                                if art_ident in p:
                                    downloaded.append(p)
                                    already = True

                        if already:
                            continue

                        # otherwise download the artifact (will create subdir)
                        loc = a.download(root=dest_dir)
                        # artifact may download into a subdir; pick files
                        for rroot, _, rfiles in os.walk(loc):
                            for f in rfiles:
                                if f.endswith('.pkl') or f.endswith('.joblib'):
                                    full = os.path.join(rroot, f)
                                    downloaded.append(full)
                                    existing[f] = full
                    except Exception as e:
                        logger.warning(f"Failed to download artifact {a}: {e}")
            else:
                # fallback: look at run files for pickle artifacts
                try:
                    files = list(run.files())
                    for f in files:
                        if f.name.endswith('.pkl') or f.name.endswith('.joblib'):
                            if f.name in existing:
                                downloaded.append(existing[f.name])
                                continue
                            target = os.path.join(dest_dir, f"{run.id}_{f.name}")
                            try:
                                f.download(target_path=target, replace=True)
                                downloaded.append(target)
                                existing[f.name] = target
                            except Exception:
                                # older API may return a file-like object
                                try:
                                    f.download(root=dest_dir)
                                except Exception:
                                    logger.debug(f"Could not download file {f.name} from run {run.id}")
                except Exception:
                    logger.debug(f"No files found for run {run.id}")
        except Exception as e:
            logger.warning(f"Error processing run {run.id}: {e}")

    # deduplicate
    downloaded = list(dict.fromkeys(downloaded))
    return downloaded


def evaluate_models_on_test(models_paths, X_test: pd.DataFrame, y_test: pd.Series):
    def _sanitize_X_for_model(X: pd.DataFrame):
        """Sanitize feature DataFrame before passing to a scikit-learn model.

        - Replace inf with NaN, fill NaNs for numeric columns with median.
        - Attempt to coerce non-numeric columns to numeric; drop if impossible.
        - Clip float columns to float32-safe range to avoid overflow errors.
        Returns sanitized DataFrame.
        """
        Xs = X.copy()
        # replace infinities
        Xs = Xs.replace([np.inf, -np.inf], np.nan)

        # For numeric columns, fill NaN with median and clip to float32-friendly range
        num_cols = Xs.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            col = Xs[c]
            med = col.median(skipna=True)
            if pd.isna(med):
                med = 0.0
            Xs[c] = col.fillna(med)
            if np.issubdtype(Xs[c].dtype, np.floating):
                cap = np.finfo(np.float32).max * 0.5
                Xs[c] = Xs[c].clip(lower=-cap, upper=cap)

        # For non-numeric columns, try to coerce to numeric; if not possible, drop them
        non_num = [c for c in Xs.columns if c not in num_cols]
        for c in non_num:
            try:
                coerced = pd.to_numeric(Xs[c], errors='coerce')
                if coerced.notna().any():
                    med = coerced.median(skipna=True)
                    if pd.isna(med):
                        med = 0.0
                    Xs[c] = coerced.fillna(med)
                else:
                    # cannot coerce any value -> drop column
                    Xs.drop(columns=[c], inplace=True)
            except Exception:
                Xs.drop(columns=[c], inplace=True)

        # final check: ensure no inf or NaN remain
        Xs = Xs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xs

    results = []
    for p in models_paths:
        try:
            m = joblib.load(p)
            print(m.get_feature_names_out())
        except Exception as e:
            logger.warning(f"Failed loading model {p}: {e}")
            continue

        # predict probabilities if possible
        try:
            # sanitize X_test for safety (fix inf/NaN/huge values)
            X_safe = X_test #_sanitize_X_for_model(X_test)
            proba = m.predict_proba(X_safe)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_proba = proba[:, 1]
            else:
                y_proba = proba.ravel()
        except Exception:
            try:
                # fallback to decision_function -> map to [0,1] via logistic
                X_safe = _sanitize_X_for_model(X_test)
                df = m.decision_function(X_safe)
                # simple sigmoid
                y_proba = 1.0 / (1.0 + np.exp(-df))
            except Exception:
                try:
                    preds = m.predict(X_safe)
                    # if binary predictions, treat as probs
                    y_proba = np.asarray(preds).astype(float)
                except Exception as e:
                    logger.warning(f"Model {p} cannot produce predictions: {e}")
                    continue

        # compute AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception as e:
            logger.warning(f"Failed to compute AUC for model {p}: {e}")
            continue

        results.append({"path": p, "auc": auc, "y_proba": y_proba})

    # sort desc
    results.sort(key=lambda x: x["auc"], reverse=True)
    return results


def _prepare_test_for_model(model_path):
    """Heuristic to prepare (X_test,y_test) for a given model file.
    Uses filename patterns and sidecar feature files if present.
    """
    basename = os.path.basename(model_path).lower()
    model_base = os.path.splitext(os.path.basename(model_path))[0]

    # XGBoost experiments use feature_engineered2
    if any(k in basename for k in ("xgb", "xgboost")):
        _, test_df = load_feature_engineered2_training_and_test_sets()
        _, _, _, _, X_test, y_test = load_and_prepare_data(test=True)
        return X_test, y_test

    # Logistic/regression baselines use feature_engineering_1 on raw test_set
    if any(k in basename for k in ("logreg", "logistic", "distance", "angle", "distance_angle")):
        train_df, test_df = load_training_and_test_sets()
        df_test = feature_engineering_1(test_df)
        # pick subset based on name
        if "distance_angle" in basename or ("distance" in basename and "angle" in basename):
            cols = ["distance_from_goal", "angle_from_goal"]
        elif "distance" in basename:
            cols = ["distance_from_goal"]
        elif "angle" in basename:
            cols = ["angle_from_goal"]
        else:
            cols = [c for c in df_test.columns if c != "is_goal"]

        X_test = df_test[cols].copy()
        y_test = df_test["is_goal"].astype(int)
        return X_test, y_test

    # PART6 models (voting / part6) require the same MeanEncoder + selected features
    if "part6" in basename or "voting" in basename:
        # load feature_engineered2 dataset (same as part6 script)
        train_df, test_df = load_feature_engineered2_training_and_test_sets()
        X = train_df.drop('is_goal', axis=1)
        y = train_df['is_goal']

        X_test = test_df.drop(columns=["is_goal"]).copy()
        y_test = test_df["is_goal"].astype(int)

        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
        if bool_cols:
            print(f"\nConverting boolean columns to int: {bool_cols}")
            X[bool_cols] = X[bool_cols].astype(int)
            X_test[bool_cols] = X_test[bool_cols].astype(int)

        # Clean inf/NaN
        print("\nCleaning inf and NaN values...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            initial_len = len(X)
            X = X.dropna()
            y = y.loc[X.index]
            print(f"Dropped {initial_len - len(X)} rows with NaN values")
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_count_test = X_test.isna().sum().sum()
        if nan_count_test > 0:
            initial_len_test = len(X_test)
            X_test = X_test.dropna()
            y_test = y_test.loc[X_test.index]
            print(f"Dropped {initial_len_test - len(X_test)} rows with NaN values in test set")


        X_train = X.copy()
        y_train = y.copy()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"\nCategorical columns found: {categorical_cols}")

        if categorical_cols:
            print(f"Applying MeanEncoder with smoothing=1 (increased variance)...")
            encoder = MeanEncoder(variables=categorical_cols, smoothing=1)
            X_train = encoder.fit_transform(X_train, y_train)
            X_test = encoder.transform(X_test)
            print(f"Encoding complete. Sample encoded values:")

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        X_test_scaled = scaler.transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        rfe_estimator = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
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
        X_test = X_test[selected_features]
        return X_test, y_test
    # default fallback: feature_engineered2
    _, test_df = load_feature_engineered2_training_and_test_sets()
    X_test = test_df.drop(columns=["is_goal"]).copy()
    y_test = test_df["is_goal"].astype(int)
    return X_test, y_test


def plot_auc_results(results, out_path=None):
    if not results:
        logger.info("No results to plot")
        return None
    df = pd.DataFrame([{"model": os.path.basename(r["path"]), "auc": r["auc"]} for r in results])
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="auc", palette="Set2")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title("Model AUCs on Feature-Engineered Test Set")
    plt.tight_layout()
    if out_path:
        # include model names in filename and save as PNG
        model_names = "_".join([os.path.splitext(os.path.basename(r["path"]))[0] for r in results[:5]])
        dirpath = os.path.dirname(out_path)
        base = os.path.splitext(os.path.basename(out_path))[0]
        filename = f"{base}_{model_names}.png"
        full_path = os.path.join(dirpath, filename)
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(full_path, bbox_inches="tight", format="png")
        logger.info(f"Saved AUC plot to {full_path}")
    return plt


def plot_model_diagnostics(model_path: str, y_true: pd.Series, y_proba: np.ndarray, out_dir: str = "figures/per_model"):
    """Create ROC, calibration, goal-rate and cumulative plots for a single model and save them as PNG files.

    Saves files as: {out_dir}/{model_basename}_<chart>.png
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(model_path))[0]

    # ensure y_true is a 1d numpy array
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).ravel()

    # 1) ROC
    try:
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)
        auc = roc_auc_score(y_true_arr, y_proba_arr)
    except Exception:
        fpr, tpr, auc = None, None, None

    # 2) Calibration
    try:
        prob_true, prob_pred = calibration_curve(y_true_arr, y_proba_arr, n_bins=10)
    except Exception:
        prob_true, prob_pred = None, None

    # 3) Goal rate vs percentile
    try:
        df_val = pd.DataFrame({"y_true": y_true_arr, "y_proba": y_proba_arr})
        df_val["percentile"] = pd.qcut(df_val["y_proba"], q=20, labels=False, duplicates="drop")
        goal_rate = df_val.groupby("percentile")["y_true"].mean().reset_index()
        goal_rate["percentile"] = goal_rate["percentile"] * 5
        goal_rate = goal_rate.sort_values("percentile", ascending=False)
        goal_rate["goal_rate_percent"] = goal_rate["y_true"] * 100
    except Exception:
        goal_rate = None

    # 4) Cumulative goals
    try:
        df_sorted = pd.DataFrame({"y_true": y_true_arr, "y_proba": y_proba_arr})
        df_sorted = df_sorted.sort_values("y_proba", ascending=False).reset_index(drop=True)
        df_sorted["cum_goals"] = df_sorted["y_true"].cumsum()
        total_goals = df_sorted["y_true"].sum()
        if total_goals > 0:
            df_sorted["cum_goal_rate"] = df_sorted["cum_goals"] / total_goals * 100
        else:
            df_sorted["cum_goal_rate"] = 0
        df_sorted["percentile"] = 100 - (np.arange(1, len(df_sorted) + 1) / len(df_sorted) * 100)
    except Exception:
        df_sorted = None

    sns.set_theme(style="whitegrid")
    # Save each chart as its own PNG file
    try:
        # ROC
        roc_path = os.path.join(out_dir, f"{base}_roc.png")
        fig = plt.figure(figsize=(6, 5))
        if fpr is not None and tpr is not None:
            plt.plot(fpr, tpr, color="C0", label=f"AUC={auc:.3f}" if auc is not None else "")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {base}")
        plt.legend()
        plt.grid(True)
        plt.savefig(roc_path, bbox_inches="tight", format="png")
        plt.close(fig)

        # Calibration
        calib_path = os.path.join(out_dir, f"{base}_calibration.png")
        fig = plt.figure(figsize=(6, 5))
        if prob_true is not None and prob_pred is not None:
            plt.plot(prob_pred, prob_true, marker='o', color='C1')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title(f"Calibration - {base}")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.grid(True)
        plt.savefig(calib_path, bbox_inches="tight", format="png")
        plt.close(fig)

        # Goal rate vs percentile
        goal_path = os.path.join(out_dir, f"{base}_goal_rate.png")
        fig = plt.figure(figsize=(7, 5))
        if goal_rate is not None:
            plt.plot(goal_rate["percentile"], goal_rate["goal_rate_percent"], color='C2')
        plt.title(f"Goal Rate vs Percentile - {base}")
        plt.xlabel("Shot Probability Model Percentile")
        plt.ylabel("Goals [%]")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.savefig(goal_path, bbox_inches="tight", format="png")
        plt.close(fig)

        # Cumulative goals
        cum_path = os.path.join(out_dir, f"{base}_cumulative.png")
        fig = plt.figure(figsize=(7, 5))
        if df_sorted is not None:
            plt.plot(df_sorted["percentile"], df_sorted["cum_goal_rate"], color='C3')
            plt.gca().invert_xaxis()
        plt.plot([100, 0], [0, 100], '--', color='gray')
        plt.xlabel("Shot Probability Model Percentile")
        plt.ylabel("Cumulative proportion of goals [%]")
        plt.title(f"Cumulative % of Goals - {base}")
        plt.grid(True)
        plt.savefig(cum_path, bbox_inches="tight", format="png")
        plt.close(fig)

        logger.info(f"Saved diagnostics PNGs for model {base} -> {roc_path}, {calib_path}, {goal_path}, {cum_path}")
    except Exception as e:
        logger.warning(f"Failed to write diagnostics PNGs for {model_path}: {e}")



def main(entity="IFT6758-2025-A09", project="ift6758-milestone2", download_dir="models/wandb_downloads", plot_path="figures/advanced_models/eval_models_auc.png", interactive=True, yes_all=False):
    # Note: per-model test sets are prepared by `_prepare_test_for_model`.
    # We no longer load a global test set here to avoid unnecessary IO.

    # ask user whether to download models from W&B or use existing local files
    os.makedirs(download_dir, exist_ok=True)
    local_existing = []
    for root, _, files in os.walk(download_dir):
        for f in files:
            if f.endswith('.pkl') or f.endswith('.joblib'):
                local_existing.append(os.path.join(root, f))

    try:
        resp_download = input("Download model artifacts from W&B? [y/N]: ").strip().lower()
    except Exception:
        resp_download = 'n'

    models = []
    if resp_download in ('y', 'yes'):
        logger.info("Fetching model artifacts from W&B runs...")
        models = _download_models_from_runs(entity, project, download_dir)
        if not models:
            logger.warning("No model files found in W&B runs.")
    else:
        # user declined download; use local files if available
        if local_existing:
            logger.info(f"Using {len(local_existing)} existing model files from {download_dir}")
            models = local_existing
        else:
            # nothing local; ask again whether to download
            try:
                resp_again = input("No local models found. Download from W&B now? [y/N]: ").strip().lower()
            except Exception:
                resp_again = 'n'
            if resp_again in ('y', 'yes'):
                logger.info("Fetching model artifacts from W&B runs...")
                models = _download_models_from_runs(entity, project, download_dir)

    if not models:
        logger.warning("No model files available for evaluation. Exiting.")
        return {}

    # Optionally ask user which models to evaluate
    models_to_eval = []
    if yes_all:
        models_to_eval = models
    else:
        for m in models:
            if not interactive:
                models_to_eval.append(m)
                continue
            # prompt the user
            try:
                resp = input(f"Evaluate model {os.path.basename(m)}? [y/N]: ").strip().lower()
            except Exception:
                resp = 'n'
            if resp in ('y', 'yes'):
                models_to_eval.append(m)

    print(models_to_eval)
    if not models_to_eval:
        logger.warning("No models selected for evaluation. Exiting.")
        return {}

    # evaluate each model on its appropriate test set (prepared per-model)
    results = []
    for mpath in models_to_eval:
        try:
            X_t, y_t = _prepare_test_for_model(mpath)
        except Exception as e:
            logger.warning(f"Failed to prepare test set for {mpath}: {e}")
            continue
        logger.info(f"Evaluating model {os.path.basename(mpath)} on test set (n={len(y_t)})")
        print(X_t.columns)
        res = evaluate_models_on_test([mpath], X_t, y_t)
        if res:
            results.extend(res)
            # for each evaluated model (usually one), produce per-model diagnostics PDF
            for r in res:
                try:
                    plot_model_diagnostics(r["path"], y_t, r["y_proba"], out_dir=os.path.join("figures", "per_model"))
                except Exception as e:
                    logger.warning(f"Failed to plot diagnostics for {r['path']}: {e}")

    # plot
    plot = plot_auc_results(results, out_path=plot_path)

    # optionally log to W&B a summary run
    try:
        run = wandb.init(entity=entity, project=project, name="evaluate_deployed_models", reinit=True)
        # log table
        table = pd.DataFrame([{"model": os.path.basename(r["path"]), "auc": r["auc"]} for r in results])
        run.log({"eval_table": wandb.Table(dataframe=table)})
        if plot is not None:
            run.log({"eval/auc_bar": wandb.Image(plot_path)})
        wandb.finish()
    except Exception as e:
        logger.warning(f"Failed to log evaluation to W&B: {e}")

    # print summary
    for r in results:
        logger.info(f"Model: {r['path']} | AUC: {r['auc']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deployed models from W&B on the feature-engineered test set")
    parser.add_argument("--entity", default="IFT6758-2025-A09")
    parser.add_argument("--project", default="ift6758-milestone2")
    parser.add_argument("--download-dir", default="models/wandb_downloads")
    parser.add_argument("--plot-path", default="figures/advanced_models/eval_models_auc.png")
    parser.add_argument("--yes-all", action="store_true", help="Automatically evaluate all downloaded models without prompting")
    parser.add_argument("--no-interactive", action="store_true", help="Do not prompt; skip interactive selection (only used when not --yes-all)")
    args = parser.parse_args()

    res = main(entity=args.entity, project=args.project, download_dir=args.download_dir, plot_path=args.plot_path, interactive=not args.no_interactive, yes_all=args.yes_all)
    print("Evaluation complete. Top results:")
    for r in res[:10]:
        print(r["path"], r["auc"])
