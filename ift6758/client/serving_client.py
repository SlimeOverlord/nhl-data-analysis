import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        #if features is None:
        #    features = ["distance"]
        #self.features = features

        # any other potential initialization
        

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        predictions = []
        for idx, row in X.iterrows():
            response = requests.post(f"{self.base_url}/predict", json=json.loads(row.to_json()))
            prob = response.json()['probability']
            predictions.append(prob)
        
        result = X.copy()
        result['prediction'] = predictions 
        return result



    def logs(self) -> dict:
        """Get server logs"""
        response = requests.get(f"{self.base_url}/logs")
        return response.json()


    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        response = requests.post(f"{self.base_url}/download_registry_model", json={"name": model})
        return response.json()


