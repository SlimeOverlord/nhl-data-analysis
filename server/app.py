from flask import Flask

app = Flask(__name__)


@app.route("/predict")
def predict():
    return "Prediction endpoint"


@app.route("/logs")
def logs():
    return "Logs endpoint"


@app.route("/download_registry_model")
def download_registry_model():
    return "Download registry model endpoint"
