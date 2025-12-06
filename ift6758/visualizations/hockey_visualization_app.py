import streamlit as st
import pandas as pd
import os
import shutil
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from ift6758.client.serving_client import ServingClient
from ift6758.client.game_client import GameClient
from ift6758.data.data_cleaning import clean_play_by_play_data

# To avoid keeping the old cache when we refresh the page, we simply delete it and create a new empty one
if "old_cache_deleted" not in st.session_state:
    shutil.rmtree("app_cache", ignore_errors=True)
    os.makedirs("app_cache", exist_ok=True)
    st.session_state.old_cache_deleted = True

# Use environment variable for Docker, fallback to localhost for local dev
SERVING_HOST = os.environ.get("SERVING_HOST", "127.0.0.1")

if "serving_client" not in st.session_state:
    st.session_state.serving_client = ServingClient(ip=SERVING_HOST, port=5000)

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


# BONUS FEATURE: Shot location visualization on hockey rink
def create_shot_location_chart(data: pd.DataFrame, team_names: list):
    """
    Creates a scatter plot showing shot locations on the hockey rink.
    Goals are shown as stars, shots as circles. Size represents xG probability.
    """
    if data.empty or 'standardized_x_coord' not in data.columns:
        return None
    
    # Filter out rows with missing coordinates
    plot_data = data.dropna(subset=['standardized_x_coord', 'standardized_y_coord'])
    if plot_data.empty:
        return None
    
    fig = go.Figure()
    
    # Draw simplified rink outline
    # Center line
    fig.add_shape(type="line", x0=0, y0=-42.5, x1=0, y1=42.5,
                  line=dict(color="red", width=2))
    # Blue lines
    fig.add_shape(type="line", x0=-25, y0=-42.5, x1=-25, y1=42.5,
                  line=dict(color="blue", width=2))
    fig.add_shape(type="line", x0=25, y0=-42.5, x1=25, y1=42.5,
                  line=dict(color="blue", width=2))
    # Goal lines
    fig.add_shape(type="line", x0=-89, y0=-42.5, x1=-89, y1=42.5,
                  line=dict(color="red", width=1))
    fig.add_shape(type="line", x0=89, y0=-42.5, x1=89, y1=42.5,
                  line=dict(color="red", width=1))
    # Rink outline
    fig.add_shape(type="rect", x0=-100, y0=-42.5, x1=100, y1=42.5,
                  line=dict(color="black", width=2))
    
    colors = px.colors.qualitative.Set1
    
    for i, team in enumerate(team_names):
        team_data = plot_data[plot_data['team_name'] == team]
        if team_data.empty:
            continue
        
        # Shots (non-goals)
        shots = team_data[team_data['is_goal'] == 0]
        if not shots.empty:
            fig.add_trace(go.Scatter(
                x=shots['standardized_x_coord'],
                y=shots['standardized_y_coord'],
                mode='markers',
                marker=dict(
                    size=shots['prediction'] * 30 + 5,
                    color=colors[i % len(colors)],
                    opacity=0.6,
                    symbol='circle'
                ),
                name=f"{team} Shots",
                hovertemplate=f"{team}<br>xG: %{{customdata:.3f}}<extra></extra>",
                customdata=shots['prediction']
            ))
        
        # Goals
        goals = team_data[team_data['is_goal'] == 1]
        if not goals.empty:
            fig.add_trace(go.Scatter(
                x=goals['standardized_x_coord'],
                y=goals['standardized_y_coord'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=colors[i % len(colors)],
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name=f"{team} Goals",
                hovertemplate=f"{team} GOAL<br>xG: %{{customdata:.3f}}<extra></extra>",
                customdata=goals['prediction']
            ))
    
    fig.update_layout(
        title="Shot Locations (size = xG, stars = goals)",
        xaxis=dict(range=[-100, 100], title="", showgrid=False, zeroline=False),
        yaxis=dict(range=[-45, 45], title="", showgrid=False, zeroline=False, scaleanchor="x"),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig


# BONUS FEATURE: Cumulative xG chart over time
def create_cumulative_xg_chart(data: pd.DataFrame, team_names: list):
    """
    Creates a line chart showing cumulative xG over time for each team.
    """
    if data.empty or 'prediction' not in data.columns:
        return None
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, team in enumerate(team_names):
        team_data = data[data['team_name'] == team].copy()
        if team_data.empty:
            continue
        
        team_data = team_data.reset_index(drop=True)
        team_data['cumulative_xG'] = team_data['prediction'].cumsum()
        team_data['cumulative_goals'] = team_data['is_goal'].cumsum()
        team_data['event_num'] = range(1, len(team_data) + 1)
        
        # Cumulative xG line
        fig.add_trace(go.Scatter(
            x=team_data['event_num'],
            y=team_data['cumulative_xG'],
            mode='lines',
            name=f"{team} xG",
            line=dict(color=colors[i % len(colors)], width=2)
        ))
        
        # Cumulative goals line (stepped)
        fig.add_trace(go.Scatter(
            x=team_data['event_num'],
            y=team_data['cumulative_goals'],
            mode='lines',
            name=f"{team} Goals",
            line=dict(color=colors[i % len(colors)], width=2, dash='dot')
        ))
    
    fig.update_layout(
        title="Cumulative xG vs Actual Goals",
        xaxis_title="Shot Number",
        yaxis_title="Cumulative Value",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig


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
            st.sidebar.error(f"Error loading the model: {response['error']}")
        else:
            st.sidebar.write(f"Model {response['selected']} loaded successfully!")
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
        new_data_df = new_data_df[["period", "period_time", "team_name", "standardized_x_coord", "standardized_y_coord", "event_type"]]
        new_data_df = pd.concat([new_data_df.reset_index(drop=True), new_data.reset_index(drop=True)], axis=1)

        # Saving the newly obtained data to the cache's corresponding CSV file
        add_to_cache(game_id, model_name, new_data_df)
    
        # Fusing the already seen and new data to compute the features for the display
        complete_data_df = pd.concat([seen_data, new_data_df], axis=0)

        # Store complete data in session state for bonus visualizations
        st.session_state.complete_data = complete_data_df

        # Obtaining the features for the display 
        features = features_for_app(complete_data_df)

        # Store features in session state
        st.session_state.features = features

        # Updating the display with the obtained features
        st.header(f"Game {game_id}: {features['team1_name']} vs {features['team2_name']}")
        st.write("")
        st.write(f"Period {features['period']} - {features['remaining_time']} left")
        st.write("")

        team1_col, team2_col = st.columns(2)
        with team1_col:
            st.metric(f"{features['team1_name']} xG (actual)", f"{features['team1_xG']} ({features['team1_score']})", f"{round(features['team1_score'] - features['team1_xG'], 1)}")

        with team2_col:
            st.metric(f"{features['team2_name']} xG (actual)", f"{features['team2_xG']} ({features['team2_score']})", f"{round(features['team2_score'] - features['team2_xG'], 1)}")

        st.header("Data used for predictions (and predictions)")
        st.write(complete_data_df)

        # BONUS: Display visualizations
        st.divider()
        st.header("Bonus: Interactive Visualizations")
        
        team_names = [features['team1_name'], features['team2_name']]
        
        # Shot location chart
        shot_chart = create_shot_location_chart(complete_data_df, team_names)
        if shot_chart:
            st.plotly_chart(shot_chart, use_container_width=True)
        
        # Cumulative xG chart
        xg_chart = create_cumulative_xg_chart(complete_data_df, team_names)
        if xg_chart:
            st.plotly_chart(xg_chart, use_container_width=True)

    except Exception as e:
        st.write(f"An exception occurred while pinging the server: {e}")

# BONUS SECTION DESCRIPTION
st.divider()
st.subheader("Bonus Feature Description")
st.write("""
**Added Features:**

1. **Shot Location Visualization**: An interactive scatter plot showing all shots on a simplified hockey rink diagram. 
   - Shot locations are displayed with circle markers where the size represents the xG probability
   - Goals are highlighted with star markers
   - Each team has a distinct color for easy comparison
   - Hover over any marker to see the exact xG value

2. **Cumulative xG Chart**: A line chart tracking the progression of expected goals throughout the game.
   - Solid lines show cumulative xG for each team
   - Dotted lines show actual goals scored
   - This visualization helps identify if a team is over/under-performing relative to their shot quality

These visualizations provide deeper insights into game flow and shot quality beyond the basic xG statistics,
allowing viewers to understand not just how many chances each team created, but where those chances came from
and how the game momentum shifted over time.
""")
