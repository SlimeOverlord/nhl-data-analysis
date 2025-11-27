import json
import requests
import pandas as pd
import logging
from ift6758.data.data_cleaning import *
from ift6758.client.serving_client import ServingClient  # Ajusta el import según tu estructura



logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self, game_id, serving_client):
        self.game_id = game_id
        self.service_client = serving_client
        self.seen_event_ids = set()

    def fetch_events(self):
        features = []
        # Download all events for the given game_id as a Python dictionary
        url_to_get_events = f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play"
        response = requests.get(url_to_get_events)
        data = response.json()
        # Clean and preprocess the raw event data with the previous implemented function
        cleaned_df = clean_play_by_play_data(data)
        # Only keep the new events 
        new_events = cleaned_df[~cleaned_df['event_id'].isin(self.seen_event_ids)]

        for _, event_row in new_events.iterrows():
            #obtain the features for each event
            features = feature_engineering_1(pd.DataFrame([event_row])).iloc[0].to_dict()
            # get the prediction calling the service_client (it has to be a dataframe because predict in serving_client expects a dataframe)
            prediction = self.service_client.predict(pd.DataFrame([features]))
            self.seen_event_ids.add(event_row['event_id'])
            # This line was to check everything was running fine
            print(prediction)


# To check it is working
if __name__ == "__main__":
    serving_client = ServingClient(ip="localhost", port=8000) 
    game_id = 2023020410  
    client = GameClient(game_id, serving_client)
    client.fetch_events()





