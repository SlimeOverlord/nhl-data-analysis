import os
import sys
import logging

sys.path.append("..")

from flask import Flask, request

# from ift6758 import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="server.log",
)

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict() -> float:
    """Gives a probability if the given shot results in a goal.
    Returns:
        float: Probability of the shot resulting in a goal."""
    logging.info("Prediction request received")
    data: dict = request.get_json()
    if not data:
        logging.error("Prediction request missing input data")
        return {"error": "No input data provided"}, 400
    logging.info(f"Input data: {data}")

    return 0.0


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
    return "Download registry model endpoint"
