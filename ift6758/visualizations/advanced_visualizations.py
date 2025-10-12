import sys

sys.path.append("../..")

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from PIL import Image
from typing import Dict, List, Tuple
import pandas as pd

from ift6758.data import SeasonData
from ift6758.data.data_cleaning import clean_season_data, get_additional_features


def load_season_data(start_year: int, end_year: int) -> Dict[int, pd.DataFrame]:
    """
    Get all NHL season data for specified year range.
    
    Args:
        start_year: Starting year (inclusive)
        end_year: Ending year (exclusive)
        
    Returns:
        Dictionary mapping year to processed DataFrame
    """
    seasons_df = {}
    
    for year in range(start_year, end_year):
        season = SeasonData(year)
        season.get_data_from_api()
        seasons_df[year] = get_additional_features(clean_season_data(season))
    
    return seasons_df


def get_unique_teams(seasons_df: Dict[int, pd.DataFrame]) -> List[str]:
    """
    Extract all unique teams ids across all seasons.
    
    Args:
        seasons_df: Dictionary mapping year to DataFrame
        
    Returns:
        Sorted list of teams ids
    """
    all_teams = set()
    for df in seasons_df.values():
        # to avoid case: if a team is present in a season but not in other
        all_teams.update(df['team'].unique())
    return sorted([team for team in all_teams])


def calculate_shot_density(df: pd.DataFrame, x_range: List[float], y_range: List[float], 
                           bins: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate 2D histogram of shot density.
    
    Args:
        df: DataFrame with shot coordinates
        x_range: [min, max] for x coordinates
        y_range: [min, max] for y coordinates
        bins: [x_bins, y_bins] for histogram
        
    Returns:
        Tuple of (histogram, x_edges, y_edges)
    """
    hist, xe, ye = np.histogram2d(
        df["standardized_x_coord"],
        df["standardized_y_coord"],
        range=[x_range, y_range],
        bins=bins,
    )
    # Normalize by number of unique games
    n_games = df['game_id'].nunique()
    if n_games > 0:
        hist /= n_games # we consider all matches to be 60 minutes
        # hist /= hist.sum()  # normalize
    return hist, xe, ye


def calculate_heatmaps(seasons_df: Dict[int, pd.DataFrame], all_teams: List[str]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Calculate differential heatmaps for each year-team combination.
    
    Args:
        seasons_df: Dictionary mapping year to DataFrame
        all_teams: List of all team abbreviations
        
    Returns:
        Nested dictionary: {year: {team: heatmap_data}}
    """
    heatmap_data_by_year_team = {}
    x_range = [-100, 100]
    y_range = [-42, 42]
    bins = [200, 84]
    
    for year, df in seasons_df.items():
        heatmap_data_by_year_team[year] = {}
        
        # Calculate overall distribution for this year (normalized by number of games)
        hist_all, _, _ = calculate_shot_density(df, x_range, y_range, bins)
        smooth_all = gaussian_filter(hist_all, sigma=1)
        
        # Calculate distribution for each team
        for team in all_teams:
            filtered_df = df[df["team"] == team]
            
            if len(filtered_df) > 0:
                hist_team, _, _ = calculate_shot_density(filtered_df, x_range, y_range, bins)
                smooth_team = gaussian_filter(hist_team, sigma=1)
                
                # Store the differential heatmap data
                heatmap_data_by_year_team[year][team] = smooth_team.T - smooth_all.T
            else:
                # If no data for this team, store empty array
                heatmap_data_by_year_team[year][team] = np.zeros((84, 200))
    
    return heatmap_data_by_year_team


def create_figure_traces(heatmap_data_by_year_team: Dict[int, Dict[str, np.ndarray]], 
                         all_teams: List[str], 
                         first_year: int, 
                         first_team: str,
                         rink_img: Image.Image) -> go.Figure:
    """
    Create plotly figure with all traces for year-team combinations.
    
    Args:
        heatmap_data_by_year_team: Nested dictionary with heatmap data
        all_teams: List of teams ids
        first_year: Initial year to display
        first_team: Initial team to display
        rink_img: NHL rink background image
        
    Returns:
        Plotly Figure object with all traces
    """
    x_coords = np.linspace(-100, 100, 200)
    y_coords = np.linspace(-42, 42, 84)
    
    fig = go.Figure()
    
    fig.add_layout_image(
        {
            "source": rink_img,
            "xref": "x",
            "yref": "y",
            "x": -100,
            "y": 42,
            "sizex": 200,
            "sizey": 84,
            "sizing": "stretch",
            "opacity": 1.0,
            "layer": "below"
        }
    )
    
    # Add traces for each year-team combination
    for year in sorted(heatmap_data_by_year_team.keys()):
        for team in all_teams:
            heatmap_data = heatmap_data_by_year_team[year][team]
            visible = (year == first_year and team == first_team)
            
            # Add the heatmap
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=x_coords,
                    y=y_coords,
                    colorscale="plotly3",
                    zmid=0,
                    opacity=0.6,
                    colorbar={
                        "title": "Shots per hours difference",
                        "thickness": 20,
                        "len": 0.7
                    },
                    hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Difference: %{z:.10f}<extra></extra>",
                    visible=visible,
                    name=f"{year}_{team}_heatmap"
                )
            )
            
            # Add contour lines
            fig.add_trace(
                go.Contour(
                    z=heatmap_data,
                    x=x_coords,
                    y=y_coords,
                    showscale=False,
                    contours={
                        "showlabels": True,
                        "labelfont": {"size": 8, "color": "white"},
                    },
                    line={"color": "white", "width": 1},
                    opacity=0.8,
                    hoverinfo="skip",
                    visible=visible,
                    name=f"{year}_{team}_contour"
                )
            )
    
    return fig


def create_dropdown_buttons(heatmap_data_by_year_team: Dict[int, Dict[str, np.ndarray]], 
                            all_teams: List[str],
                            first_year: int,
                            first_team: str) -> Tuple[List[dict], List[dict]]:
    """
    Create dropdown button configurations for seasons and teams.
    
    Args:
        heatmap_data_by_year_team: Nested dictionary with heatmap data
        all_teams: List of team abbreviations
        first_year: Initial year to display
        first_team: Initial team to display
        
    Returns:
        Tuple of (season_buttons, team_buttons)
    """
    # Create dropdown menu buttons for seasons
    season_buttons = []
    for year in sorted(heatmap_data_by_year_team.keys()):
        visibility = []
        for y in sorted(heatmap_data_by_year_team.keys()):
            for team in all_teams:
                if y == year and team == first_team:
                    visibility.extend([True, True])  # Show heatmap + contour for this year and first team
                else:
                    visibility.extend([False, False])  # Hide all others
        
        season_buttons.append(
            {
                "label": f"{year} Season",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"NHL Shot Density Differential - {year} Season - {first_team}"}
                ]
            }
        )
    
    # Create dropdown menu buttons for teams
    team_buttons = []
    for team in all_teams:
        visibility = []
        for year in sorted(heatmap_data_by_year_team.keys()):
            for t in all_teams:
                if year == first_year and t == team:
                    visibility.extend([True, True])  # Show heatmap + contour for first year and this team
                else:
                    visibility.extend([False, False])  # Hide all others
        
        team_buttons.append(
            {
                "label": f"{team}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"NHL Shot Density Differential - {first_year} Season - {team}"}
                ]
            }
        )
    
    return season_buttons, team_buttons


def configure_layout(fig: go.Figure, 
                    season_buttons: List[dict], 
                    team_buttons: List[dict],
                    first_year: int,
                    first_team: str) -> None:
    """
    Configure with dropdown menus and styling.
    
    Args:
        fig: Plotly Figure object
        season_buttons: List of season dropdown button configs
        team_buttons: List of team dropdown button configs
        first_year: Initial year to display
        first_team: Initial team to display
    """
    fig.update_layout(
        title=f"NHL Shot Density Differential - {first_year} Season - {first_team}",
        xaxis={
            "range": [-100, 100],
            "title": "X Coordinate (feet)",
            "showgrid": False,
            "zeroline": False
        },
        yaxis={
            "range": [-42, 42],
            "title": "Y Coordinate (feet)",
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
            "scaleratio": 1
        },
        width=1200,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        updatemenus=[
            # Season dropdown
            {
                "active": 0,
                "buttons": season_buttons,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.11,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": "gray",
                "borderwidth": 1
            },
            # Team dropdown
            {
                "active": 0,
                "buttons": team_buttons,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.30,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": "gray",
                "borderwidth": 1
            }
        ]
    )


def main():
    """
    Main execution function for creating NHL shot density visualization.
    """
    # Load and process data
    print("Loading season data...")
    seasons_df = load_season_data(2016, 2024)
    
    # Get all teams
    print("Extracting team information...")
    all_teams = get_unique_teams(seasons_df)
    
    # Calculate heatmaps
    print("Calculating shot density heatmaps...")
    heatmap_data_by_year_team = calculate_heatmaps(seasons_df, all_teams)
    
    # Load rink image
    print("Loading rink image...")
    rink_img = Image.open("../../figures/nhl_rink.png")
    
    # Setup initial display
    first_year = 2016
    first_team = all_teams[0]
    
    # Create figure
    print("Creating interactive visualization...")
    fig = create_figure_traces(heatmap_data_by_year_team, all_teams, first_year, first_team, rink_img)
    
    # Create dropdown buttons
    season_buttons, team_buttons = create_dropdown_buttons(heatmap_data_by_year_team, all_teams, first_year, first_team)
    
    # Configure layout
    configure_layout(fig, season_buttons, team_buttons, first_year, first_team)
    
    # Display figure
    print("Displaying figure...")
    fig.show()


if __name__ == "__main__":
    main()
