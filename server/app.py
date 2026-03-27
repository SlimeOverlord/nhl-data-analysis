import os
import sys
import logging

import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, session

# Local imports
sys.path.append("..")  # to allow imports from parent directory
from server.wandb_utils import fetch_and_load_artifact


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="server.log",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env (if present)
load_dotenv()


def setup() -> None:
    # Fetch model from WandB project (helper handles errors and returns (None, None) on failure)
    wandb_entity = os.getenv("WANDB_ENTITY", "IFT6758-2025-A09")
    wandb_project = os.getenv("WANDB_PROJECT", "ift6758-milestone2")

    # If WANDB_API_KEY is provided in env, wandb will pick it up automatically.
    if not os.getenv("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not set in environment; WandB API calls may fail")
    logger.info("Attempting to fetch 'distance_angle_model' from WandB project")
    # By default, load the 'distance_angle_model'
    model, scaler = fetch_and_load_artifact(
        wandb_entity, wandb_project, "distance_angle_model"
    )
    if model:
        logger.info("Model loaded and assigned to module-level `model` variable")
    else:
        logger.info("No model was loaded from WandB; `model` remains None")
    return model, scaler


app = Flask(__name__)

# Secret key for session support. In production set `FLASK_SECRET_KEY` in env.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# In-memory store for loaded models: {model_name: (model, scaler)}.
# This keeps model/scaler objects in memory and avoids storing paths in session.
models_store: dict = {}

# Run initial setup to verify default model availability and cache it in memory.
default_model, default_scaler = setup()
if default_model:
    models_store["distance_angle_model"] = (default_model, default_scaler)

# Feature sets expected by each model (order matters)
MODEL_FEATURES = {
    "distance_angle_model": ["distance_from_goal", "angle_from_goal"],
    "distance_model": ["distance_from_goal"],
    "angle_model": ["angle_from_goal"],
}


@app.route("/predict", methods=["POST"])
def predict() -> dict:
    """Gives a probability if the given shot results in a goal.
    Returns:
        float: Probability of the shot resulting in a goal."""
    logging.info("Prediction request received")
    data: dict = request.get_json()
    if not data:
        logging.error("Prediction request missing input data")
        return {"error": "No input data provided"}, 400
    logging.info(f"Input data: {data}")
    try:
        # Determine which model to use for this session (fallback to default)
        model_name = session.get("model_name", "distance_angle_model")
        logging.info(f"Using model '{model_name}' for prediction")

        # Load model & scaler from the in-memory store if available.
        model_entry = models_store.get(model_name)
        if model_entry:
            model, scaler = model_entry
        else:
            # Model must be already loaded in memory; do not attempt to load here.
            logging.error(
                "Model '%s' is not loaded in memory; refusing to load on-demand",
                model_name,
            )
            return {"error": "Requested model not loaded in memory"}, 500

        # Build input features according to model's expected features
        feature_names = MODEL_FEATURES.get(
            model_name, ["distance_from_goal", "angle_from_goal"]
        )
        X_vals = []
        for fname in feature_names:
            if fname not in data:
                logging.error("Missing feature '%s' for model '%s'", fname, model_name)
                return {"error": f"Missing feature {fname}"}, 400
            X_vals.append(data[fname])

        X = np.array([X_vals])

        if scaler is not None:
            X = scaler.transform(X)
        probability = model.predict_proba(X)
        logging.info(f"Predicted probability: {probability[0][1]}")
        return {"probability": float(probability[0][1])}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": "Prediction failed"}, 500


@app.route("/logs", methods=["GET", "POST"])
def logs() -> list[str]:
    """Returns the logs of the server.
    Returns:
        list[str]: List of log entries."""
    logging.info("Logs request received")
    if not os.path.exists("server.log"):
        logging.warning("Log file does not exist")
        return []

    # Maybe we should use a logging handler that stores logs in memory?
    with open("server.log", "r") as f:
        log_entries = f.readlines()
    logging.info(f"Returning {len(log_entries)} log entries")
    return log_entries


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model() -> str:
    """Downloads the model from the model registry and change the current model.
    Returns:
        str: Confirmation message."""
    logging.info("Download registry model request received")
    name = request.json.get("name")
    if not name:
        logging.error("No model name provided in request")
        return {"error": "No model name provided"}, 400

    if name not in MODEL_FEATURES:
        logging.error("Requested model '%s' is not available", name)
        return {"error": "Requested model is not available"}, 400

    # Attempt to load the requested model (prefer local artifacts). Cache in memory.
    logging.info("Attempting to load model '%s' for session", name)
    wandb_entity = os.getenv("WANDB_ENTITY", "IFT6758-2025-A09")
    wandb_project = os.getenv("WANDB_PROJECT", "ift6758-milestone2")
    model_obj, scaler = fetch_and_load_artifact(wandb_entity, wandb_project, name)

    if not model_obj:
        logging.error("Failed to fetch model '%s' from WandB or local artifacts", name)
        return {"error": "Failed to fetch requested model"}, 500

    # Cache the loaded model/scaler in memory and set the session model name.
    models_store[name] = (model_obj, scaler)
    session["model_name"] = name
    logging.info(
        "Model '%s' is available, cached in memory, and selected for session", name
    )
    return {"selected": name}
