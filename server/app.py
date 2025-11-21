import os
import sys
import logging

import numpy as np
from dotenv import load_dotenv
from flask import Flask, request

# Local imports
sys.path.append("..")  # to allow imports from parent directory
from ift6758.data.data_cleaning import feature_engineering_1
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


def setup():
    # Fetch model from WandB project (helper handles errors and returns (None, None) on failure)
    wandb_entity = os.getenv("WANDB_ENTITY", "IFT6758-2025-A09")
    wandb_project = os.getenv("WANDB_PROJECT", "ift6758-milestone2")

    # If WANDB_API_KEY is provided in env, wandb will pick it up automatically.
    if not os.getenv("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not set in environment; WandB API calls may fail")
    logger.info("Attempting to fetch 'angle_model' from WandB project")
    model, downloaded_dir = fetch_and_load_artifact(
        wandb_entity, wandb_project, "angle_model"
    )
    logger.info(
        "Model loaded and assigned to module-level `model` variable"
        if model
        else "No model was loaded from WandB; `model` remains None"
    )
    return model


model = setup()
app = Flask(__name__)


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
        logging.info("Scaling input data")
        # TODO: use scaler, we should get it from wandb as well
        # keep only angle feature for prediction
        probability = model.predict_proba(np.array([[data["angle_from_goal"]]]))
        logging.info(f"Predicted probability: {probability[0][1]}")
        return {"probability": probability[0][1]}
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
        return "No model name provided", 400
    return "Download registry model endpoint"
