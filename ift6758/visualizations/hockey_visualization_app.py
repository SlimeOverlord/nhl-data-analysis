import streamlit as st
import pandas as pd
import os
import shutil
import numpy as np
import requests
from datetime import timedelta
from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient
from ift6758.data.data_cleaning import clean_play_by_play_data

# To avoid keeping the old cache when we refresh the page, we simply delete it and create a new empty one
if "old_cache_deleted" not in st.session_state:
    shutil.rmtree("app_cache", ignore_errors=True)
    os.makedirs("app_cache", exist_ok=True)
    st.session_state.old_cache_deleted = True

# Creating the serving client that will communicate with the Flask app
if "serving_client" not in st.session_state:
    st.session_state.serving_client = ServingClient(ip="127.0.0.1", port=5000)

# Creating the game client dictionary. The clients are indexed by game ID and model, allowing to make predictions with different models for a same game ID
if "game_clients" not in st.session_state:
    st.session_state.game_clients = {}

# Method that loads the already seen data from a cache containing csv files
def load_from_cache(game_id, model_name):
    filepath = f"app_cache/{game_id}_{model_name}_seen.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return pd.DataFrame()

# Method that writes the newly seen data to the cache, making it "seen"
def add_to_cache(game_id, model_name, new_data: pd.DataFrame):
    filepath = f"app_cache/{game_id}_{model_name}_seen.csv"
    if os.path.exists(filepath):
       new_data.to_csv(filepath, mode="a", header=False, index=False)
    else:
        new_data.to_csv(filepath, header=True, index=False)

# Method that computes from the whole data (already seen and newly seen) the features shown on the app or used to calculate those
def features_for_app(data: pd.DataFrame):
    
    # Getting the team names
    team_names = data["team_name"].unique()
    team1_name = team_names[0]
    team2_name = team_names[1]

    # Getting each team's total goals and predicted goals up to now
    goals_df = data.groupby(["team_name"])[["is_goal", "prediction"]].sum()

    team1_score = goals_df.loc[team1_name, "is_goal"].astype(int)
    team2_score = goals_df.loc[team2_name, "is_goal"].astype(int)

    team1_xG = round(goals_df.loc[team1_name, "prediction"], 1)
    team2_xG = round(goals_df.loc[team2_name, "prediction"], 1)

    # Obtaining from the last event the current period and the remaining time
    last_event = data.iloc[-1]

    period = last_event["period"]
    minutes, seconds = str.split(last_event["period_time"], ":")
    time_in_period = timedelta(minutes=int(minutes), seconds=int(seconds))

    time_left = str(timedelta(minutes=20) - time_in_period)
    time_list = str.split(time_left, sep=":")

    remaining_time = f"{time_list[1]}:{time_list[2]}"

    features = {
        "team1_name": team1_name,
        "team2_name": team2_name,
        "team1_score": team1_score,
        "team2_score": team2_score,
        "team1_xG": team1_xG,
        "team2_xG": team2_xG,
        "period": period,
        "remaining_time": remaining_time
    }
    
    return features

st.title("Hockey Visualization App")

# The sidebar containing all the options for loading the model from WandB
st.sidebar.header("Load a model from WandB")
workspace = st.sidebar.text_input("Workspace", "ift6758-milestone2")
model_name = st.sidebar.text_input("Model", "distance_angle_model")
version = st.sidebar.text_input("Version", "1")

# The "Get Model" button that will load the model
if st.sidebar.button("Get Model"):
    try:
        # Calling the serving client to "swap models" with the specified model
        response = st.session_state.serving_client.download_registry_model(workspace, model_name, version)
        if "error" in response:
            st.sidebar.error(f"Error loading the model: {response["error"]}")
        else:
            st.sidebar.write(f"Model {response["selected"]} loaded successfully!")
    except Exception as e:
        st.sidebar.write(f"An exception occurred while loading the model: {e}")

# The text zone where the game ID is entered by the user
game_id = st.text_input("Game ID", "2021020329")

# The "Ping Game" button that will make a request to the game client and update the display with the received info
if st.button("Ping game"):

    # Loading the already seen data from the corresponding CSV file in the cache
    seen_data = load_from_cache(game_id, model_name)

    # Obtaining the game client, or creating it if it doesn't already exist
    if(game_id in st.session_state.game_clients):
        if(model_name in st.session_state.game_clients[game_id]):
            game_client = st.session_state.game_clients[game_id][model_name]  
        else:
            game_client = GameClient(game_id, st.session_state.serving_client) 
    else:
        st.session_state.game_clients[game_id] = {}
        game_client = GameClient(game_id, st.session_state.serving_client)

    st.session_state.game_clients[game_id][model_name] = game_client

    try:
        # Obtaining new data from the game using the game client
        new_data = game_client.fetch_events()    
    
        # Making an API request to get all the current data from the game (already seen and unseen)
        url_to_get_events = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        response = requests.get(url_to_get_events)
        all_data = response.json()

        # Getting only the unseen cleaned data, extracting the important features and fusing it with the data returned by the game client
        cleaned_df = clean_play_by_play_data(all_data)
        new_data_df = cleaned_df.loc[len(seen_data):]
        new_data_df = new_data_df[["period", "period_time", "team_name"]]
        new_data_df = pd.concat([new_data_df, new_data], axis=1)

        # Saving the newly obtained data to the cache's corresponding CSV file
        add_to_cache(game_id, model_name, new_data_df)
    
        # Fusing the already seen and new data to compute the features for the display
        complete_data_df = pd.concat([seen_data, new_data_df], axis=0)

        # Obtaining the features for the display 
        features = features_for_app(complete_data_df)

        # Updating the display with the obtained features
        st.header(f"Game {game_id}: {features["team1_name"]} vs {features["team2_name"]}")
        st.write("")
        st.write(f"Period {features["period"]} - {features["remaining_time"]} left")
        st.write("")

        team1_col, team2_col = st.columns(2)
        with team1_col:
            st.metric(f"{features["team1_name"]} xG (actual)", f"{features["team1_xG"]} ({features["team1_score"]})", f"{round(features["team1_score"] - features["team1_xG"], 1)}")

        with team2_col:
            st.metric(f"{features["team2_name"]} xG (actual)", f"{features["team2_xG"]} ({features["team2_score"]})", f"{round(features["team2_score"] - features["team2_xG"], 1)}")

        st.header("Data used for predictions (and predictions)")
        st.write(complete_data_df.iloc[:, 3:])

    except Exception as e:
        st.write(f"An exception occurred while pinging the server: {e}")