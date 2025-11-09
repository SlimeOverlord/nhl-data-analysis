import os
import wandb
from ift6758.data.data_cleaning import feature_engineering_1
from ift6758.data.load_datasets import load_training_and_test_sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.calibration import CalibrationDisplay, calibration_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from matplotlib.colors import to_hex
'''
This script trains three simple logistic regression baseline models
to estimate the probability that a shot becomes a goal.
The three feature sets are: distance only, angle only, and distance+angle.

For each feature set the script does:
 - split the data into train/validation
 - scale features with StandardScaler
 - train LogisticRegression
 - compute probabilities on the validation set
 - compute and log metrics (AUC, accuracy) to Weights & Biases (wandb)
 - save validation predictions as a CSV and upload as a dataset artifact
 - save the trained model with joblib and upload as a model artifact

After training all runs, the script produces four diagnostic figures on
the validation set: ROC curves, goal rate vs model percentile, cumulative
goals vs percentile, and calibration plots.
'''
# 1. Load datasets and apply feature engineering
# Load the pre-processed training and test sets (helper in project)
training_set, test_set = load_training_and_test_sets()
# Apply the project's feature engineering function to the training set
df = feature_engineering_1(training_set)

# Basic cleaning: remove shots with unrealistic distance (>100)
# and drop rows missing distance, angle or target label.
df = df[df["distance_from_goal"] <= 100]
df.dropna(subset=["distance_from_goal", "angle_from_goal", "is_goal"], inplace=True)

# Target variable: 1 if the event is a goal, 0 otherwise
y = df["is_goal"]

# 2. Define the three feature sets we will use as baselines
feature_sets = {
    "distance": ["distance_from_goal"],
    "angle": ["angle_from_goal"],
    "distance_angle": ["distance_from_goal", "angle_from_goal"]
}

# Colors used for plotting each model's curve — use Seaborn Set2 palette
# We convert the palette to hex strings so matplotlib/wandb accept them
set2 = sns.color_palette("Set2", 3)
hex_colors = [to_hex(c) for c in set2]
colors = {
    "distance": hex_colors[0],       # Set2[0]
    "angle": hex_colors[1],          # Set2[1]
    "distance_angle": hex_colors[2]  # Set2[2]
}

# 3. Train the three models and save predictions in `results`
results = {}

"""
for name, features in feature_sets.items():
    X = df[features]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    y_proba = clf.predict_proba(X_val_scaled)[:, 1]
    results[name] = {"y_proba": y_proba, "clf": clf}"""

for name, features in feature_sets.items():
    # Select the columns for this feature set
    X = df[features]

    # Create a stratified train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features using training statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- Initialize a W&B run for this model ---
    # We set entity/project/name/tags/config so runs are easy to find
    run = wandb.init(
        entity="IFT6758-2025-A09",
        project="ift6758-milestone2",
        name=f"logreg_{name}_base",
        tags=["baseline", "logreg", name],
        config={
            "model": "LogisticRegression",
            "features": features,
            "random_state": 42
        },
        reinit=True  # allow multiple runs in a single script/process
    )

    # --- Train the logistic regression model ---
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Predict probabilities on the validation set (probability of class 1)
    y_proba = clf.predict_proba(X_val_scaled)[:, 1]
    results[name] = {"y_proba": y_proba, "clf": clf}

    # --- Compute validation metrics ---
    auc = roc_auc_score(y_val, y_proba)
    accuracy = accuracy_score(y_val, (y_proba > 0.5).astype(int))

    # Print some info locally to help debugging
    print(f"\nModel {name}:")
    print(f"Features used: {features}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"X_train shape: {X_train.shape}")

    # Log scalar metrics to W&B
    wandb.log({
        "AUC": auc,
        "accuracy": accuracy,
        "feature_set": name
    })

    # --- Create and log per-run diagnostic figures to W&B ---
    # We log these here (while 'run' is active) so each model's run
    # stores its own ROC, calibration, goal-rate and cumulative plots.
    # 1) ROC curve (single-run)
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    fig_roc = plt.figure()
    plt.plot(fpr, tpr, color=colors.get(name, "blue"), label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.grid(True)
    run.log({f"roc_plot_{name}": wandb.Image(fig_roc)})
    plt.close(fig_roc)

    # 2) Calibration plot (single-run) — use calibration_curve so we control colors
    fig_cal = plt.figure()
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', color=colors.get(name, "gray"), label=name)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f"Calibration - {name}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.grid(True)
    run.log({f"calibration_plot_{name}": wandb.Image(fig_cal)})
    plt.close(fig_cal)

    # 3) Goal rate vs percentile (single-run)
    df_val = pd.DataFrame({"y_true": y_val, "y_proba": y_proba})
    df_val["percentile"] = pd.qcut(df_val["y_proba"], q=20, labels=False, duplicates="drop")
    goal_rate = df_val.groupby("percentile")["y_true"].mean().reset_index()
    goal_rate["percentile"] = goal_rate["percentile"] * 5
    goal_rate = goal_rate.sort_values("percentile", ascending=False)
    goal_rate["goal_rate_percent"] = goal_rate["y_true"] * 100
    fig_goal = plt.figure()
    plt.plot(goal_rate["percentile"], goal_rate["goal_rate_percent"], color=colors.get(name, "gray"))
    plt.title(f"Goal Rate vs Percentile - {name}")
    plt.xlabel("Shot Probability Model Percentile")
    plt.ylabel("Goals [%]")
    plt.ylim(0, 100)
    plt.grid(True)
    run.log({f"goal_rate_plot_{name}": wandb.Image(fig_goal)})
    plt.close(fig_goal)

    # 4) Cumulative goals (single-run)
    df_val_sorted = df_val.sort_values("y_proba", ascending=False).reset_index(drop=True)
    df_val_sorted["cum_goals"] = df_val_sorted["y_true"].cumsum()
    total_goals = df_val_sorted["y_true"].sum()
    if total_goals > 0:
        df_val_sorted["cum_goal_rate"] = df_val_sorted["cum_goals"] / total_goals * 100
    else:
        df_val_sorted["cum_goal_rate"] = 0
    df_val_sorted["percentile"] = 100 - (np.arange(1, len(df_val_sorted) + 1) / len(df_val_sorted) * 100)
    fig_cum = plt.figure()
    plt.plot(df_val_sorted["percentile"], df_val_sorted["cum_goal_rate"], color=colors.get(name, "gray"))
    plt.gca().invert_xaxis()
    plt.title(f"Cumulative % of Goals - {name}")
    plt.xlabel("Shot Probability Model Percentile (100 → 0)")
    plt.ylabel("Cumulative proportion of goals [%]")
    plt.grid(True)
    run.log({f"cumulative_plot_{name}": wandb.Image(fig_cum)})
    plt.close(fig_cum)

    # --- Save validation predictions to CSV and upload as an artifact ---
    val_results = pd.DataFrame({
        "y_true": y_val,
        "y_proba": y_proba,
        "feature_set": name
    })
    val_csv = f"{name}_validation_predictions.csv"
    val_results.to_csv(val_csv, index=False)

    dataset_artifact = wandb.Artifact(
        name=f"{name}_validation_data",
        type="dataset",
        description=f"Validation predictions for {name} feature set"
    )
    dataset_artifact.add_file(val_csv)
    run.log_artifact(dataset_artifact)

    # --- Save model (joblib) and upload as a model artifact ---
    model_filename = f"{name}_model.pkl"
    joblib.dump(clf, model_filename)

    model_artifact = wandb.Artifact(
        name=f"{name}_model",
        type="model",
        description=f"Logistic Regression trained on {features}",
        metadata={
            "features": features,
            "validation_auc": auc,
            "validation_accuracy": accuracy,
            "model_type": "LogisticRegression",
            "feature_set": name
        }
    )
    model_artifact.add_file(model_filename)
    run.log_artifact(model_artifact)

    # Print the run URL (convenient to paste into the blog)
    print(f"\nModel {name} - Wandb run URL:", run.get_url())

    # Finish the current W&B run to flush data and allow a new run next loop
    wandb.finish()

# Ensure a local `figures/` directory exists to save the figures shown with plt.show()
figures_dir = os.path.join(os.getcwd(), "figures")
os.makedirs(figures_dir, exist_ok=True)

# Aleatory baseline
np.random.seed(42)
results["random"] = {"y_proba": np.random.uniform(0, 1, size=len(y_val))}

# 4. ROC curves
roc_data = {name: data["y_proba"] for name, data in results.items() if name != "random"}
fig_roc = plt.figure(figsize=(6,5))
for name, data in results.items():
    y_proba = data["y_proba"]
    if name != "random":
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        auc = roc_auc_score(y_val, y_proba)
        plt.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC={auc:.3f})")
        # Log ROC curve to wandb for each model
        if wandb.run is not None:
            wandb.log({f"roc_curve_{name}": wandb.Image(fig_roc)})
    else:
        plt.plot([0,1],[0,1],'--',color='gray',label="random (AUC=0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Logistic Regression Models")
plt.legend()
plt.grid(True)

# Save global ROC figure
roc_path = os.path.join(figures_dir, "roc_curves.png")
fig_roc.savefig(roc_path, bbox_inches="tight")
print(f"Saved global ROC figure to: {roc_path}")

plt.show()
plt.close(fig_roc)


# 5. Goal Rate vs Percentile  
plt.figure(figsize=(7,5))
for name, data in results.items():
    y_proba = data["y_proba"]
    # Create dataframe
    df_val = pd.DataFrame({"y_true": y_val, "y_proba": y_proba})
    # Divide into 20 percentiles
    df_val["percentile"] = pd.qcut(df_val["y_proba"], q=20, labels=False, duplicates="drop")
    # Calculate average goal rate by percentile
    goal_rate = df_val.groupby("percentile")["y_true"].mean().reset_index()
    # Scale percentiles to 0–100
    goal_rate["percentile"] = goal_rate["percentile"] * 5
    # Order descending (so 100 goes on the left)
    goal_rate = goal_rate.sort_values("percentile", ascending=False)
    # Calculate goal rate in percentage
    goal_rate["goal_rate_percent"] = goal_rate["y_true"] * 100
    # Plot
    plt.plot(goal_rate["percentile"],
             goal_rate["goal_rate_percent"],
             color=colors.get(name, "gray"),
             label=name)

# Ejes y formato igual que antes
plt.title("Goal Rate vs Shot Probability Model Percentile (Validation Set)")
plt.xlabel("Shot Probability Model Percentile")
plt.ylabel("Goals / (Shots + Goals) [%]")
plt.ylim(0, 100)
plt.grid(True)
plt.gca().invert_xaxis()   
plt.legend()

# Save the global goal-rate figure
fig_goal_global = plt.gcf()
goal_path = os.path.join(figures_dir, "goal_rate_vs_percentile.png")
fig_goal_global.savefig(goal_path, bbox_inches="tight")
print(f"Saved goal-rate figure to: {goal_path}")

plt.show()

# 6. Cumulative Goals
plt.figure(figsize=(7,5))
for name, data in results.items():
    y_proba = data["y_proba"]
    # Create DataFrame and sort by descending probability
    df_val = pd.DataFrame({"y_true": y_val, "y_proba": y_proba})
    df_val = df_val.sort_values("y_proba", ascending=False).reset_index(drop=True)
    # Calculate cumulative goals
    df_val["cum_goals"] = df_val["y_true"].cumsum()
    total_goals = df_val["y_true"].sum()
    df_val["cum_goal_rate"] = df_val["cum_goals"] / total_goals * 100
    # Create inverted X axis (so 100 goes on the left)
    df_val["percentile"] = 100 - (np.arange(1, len(df_val)+1) / len(df_val) * 100)
    # Plot
    plt.plot(df_val["percentile"], df_val["cum_goal_rate"],
             color=colors.get(name, "gray"), label=name)

plt.gca().invert_xaxis()
plt.plot([100,0],[0,100],'--',color='gray',label="random baseline")
plt.xlabel("Shot Probability Model Percentile (100 → 0)")
plt.ylabel("Cumulative proportion of goals [%]")
plt.title("Cumulative % of Goals (Validation Set)")
plt.grid(True)
plt.legend()
# Save cumulative goals figure
cum_path = os.path.join(figures_dir, "cumulative_goals.png")
plt.savefig(cum_path, bbox_inches="tight")
print(f"Saved cumulative goals figure to: {cum_path}")
plt.show()

# 7. Calibration Plot 
calib_data = {name: data["y_proba"] for name, data in results.items() if name != "random"}
fig_global_cal = plt.figure(figsize=(6,5))
ax = plt.gca()  #create a single axis for all plots
for name, data in results.items():
    if name != "random":
        prob_true, prob_pred = calibration_curve(y_val, data["y_proba"], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', color=colors.get(name, "gray"), label=name)

# Reference line for perfect calibration
plt.plot([0,1],[0,1],'--',color='gray',label="Perfect calibration")

plt.title("Calibration Plot — Logistic Regression Models")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.grid(True)
plt.legend()

# Log the global calibration plot to Wandb
if wandb.run is not None:
    wandb.log({"global_calibration_plot": wandb.Image(fig_global_cal)})

# Save the calibration plot
calib_path = os.path.join(figures_dir, "calibration_plot.png")
fig_global_cal.savefig(calib_path, bbox_inches="tight")
print(f"Saved calibration plot to: {calib_path}")

plt.show()
plt.close(fig_global_cal)
