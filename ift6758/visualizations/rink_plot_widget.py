import sys
sys.path.append('../..')
from ift6758.data.data_fetching import SeasonData
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path

def load_games(year, season_type="reg"):
    season = SeasonData(year)
    season.get_data_from_api()  
    
    if season_type == "reg":
        return season.reg_season_data
    else:
        return season.playoffs_data
    
def create_game_slider(game_ids):
    """Create a game slider."""
    return widgets.SelectionSlider(
        # Create a tuple in which the first element is the last 4 digits of the game_id (to be shown in the slider) and the second element is the full game_id (to be returned when selecting an option)
        options=[(int(gid[-4:]), gid) for gid in game_ids],
        value=game_ids[0],
        description="Game"
    )

def create_event_slider(game_dict):
    """Create an event slider for a game."""
    event_ids = sorted([p["eventId"] for p in game_dict["plays"]])
    return widgets.SelectionSlider(
        options=event_ids,
        value=event_ids[0],
        description="Event"
    )

def update_event_slider(change, games, game_slider, container):
    """
    It executes when the game changes. It updates the container with the new event slider.
    """
    # Extract the newly selected game_id from the change event
    new_game_id = change["new"]
    # Create a new event slider
    new_event_slider = create_event_slider(games[new_game_id])
    # Update the container with the new event slider
    container.children = [game_slider, new_event_slider]

def get_rink_path():
    project_root = Path.cwd()
    while project_root.name != "nhl-data-analysis" and project_root.parent != project_root:
        project_root = project_root.parent
    return project_root / "figures" / "nhl_rink.png"

def plot_event_on_rink(play, rink_path=get_rink_path()):
    """
    Graph an event on the hockey rink image.
    """

    # Get the details of the play to extract the coordinates and relevant data
    details = play.get("details", {})
    x, y = details.get("xCoord"), details.get("yCoord")

    if x is None or y is None:
        print("This event has no coordinates (x,y).")
        return

    rink_img = mpimg.imread(rink_path)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(rink_img, extent=[-100, 100, -42.5, 42.5])

    ax.scatter(x, y, s=300, c="dodgerblue", edgecolor="white", linewidth=2)
    ax.set_title(f"{play.get('typeDescKey', 'event')} at {play.get('timeInPeriod', '')}")
    plt.show()

    # Show event details in a table (dataframe) enhanced with html formatting
    df_details = pd.DataFrame([details]).T
    df_details.columns = ["Value"]
    df_details.index.name = "Field"

    html_table = df_details.reset_index().to_html(
    index=False,
    classes="table table-striped",
    border=0,
    justify="center")

    html = f"""
    <div style="font-family:Arial; font-size:12px; 
            width:40%; margin-left:35%; text-align:center;">
    <h4>Event details</h4>
    {html_table}
    </div>
    """

    display(HTML(html))

def callback_to_show_game_info(game_id, games):
    # Get the game details
    game_dict = games[game_id]

    home = game_dict["homeTeam"]
    away = game_dict["awayTeam"]

    # Summary of the game in HTML formatting
    html = f"""
    <div style='text-align:center; font-family:Arial; padding:30px;'>
        <h3>{home['commonName']['default']} (home) vs {away['commonName']['default']} (away)</h3>
        <p><b>Date:</b> {game_dict['startTimeUTC']} &nbsp; | &nbsp; <b>Game ID:</b> {game_dict['id']}</p>
        {"<p><b>OT</b></p>" if game_dict["otInUse"] else ""}
        <table style='margin:auto; border-collapse:collapse;'>
            <tr style='background:#f2f2f2;'>
                <th></th><th>Home</th><th>Away</th>
            </tr>
            <tr>
                <td><b>Teams</b></td>
                <td>{home['abbrev']}</td>
                <td>{away['abbrev']}</td>
            </tr>
            <tr>
                <td><b>Goals</b></td>
                <td>{home['score']}</td>
                <td>{away['score']}</td>
            </tr>
            <tr>
                <td><b>SoG</b></td>
                <td>{home['sog']}</td>
                <td>{away['sog']}</td>
            </tr>
        </table>
    </div>
    """

    display(HTML(html))

def update_output(games, game_slider, event_slider, out):
    """
    Updates the output based on the selected game and event.
    """
    with out:
        # Clear any previous output in the widget
        out.clear_output()
        # Get the selected game and event IDs
        game_id = game_slider.value
        event_id = event_slider.value

        # Show game summary 
        callback_to_show_game_info(game_id,games)

        # Find the selected play
        game_dict = games[game_id]
        play = next(p for p in game_dict["plays"] if p["eventId"] == event_id)

        # Show selected play name and time
        print("\nSelected event:", play.get("typeDescKey"), "at", play.get("timeInPeriod"))

        # Graph play
        plot_event_on_rink(play)

def on_game_change(change, games, game_slider, event_slider, out):
    """
    It executes when the game changes. It updates the options of the event slider and refreshes the view.
    """
    new_game_id = change["new"]
    new_event_ids = [p["eventId"] for p in games[new_game_id]["plays"]]

    # reset the options of the event_slider
    event_slider.options = new_event_ids
    event_slider.value = new_event_ids[0]

    # update output
    update_output(games, game_slider, event_slider, out)

def on_event_change(change, games, game_slider, event_slider, out):
    """
    It executes when the event changes, it refreshes the view.
    """
    update_output(games, game_slider, event_slider, out)

def interactive_explorer(year=2017, season_type="reg"):
    # Load games for the specified year and season type
    games = load_games(year, season_type)
    # Create sliders and output container
    game_ids = list(games.keys())
    game_slider = create_game_slider(game_ids)
    event_slider = create_event_slider(games[game_slider.value])

    out = widgets.Output()
    # Attach observers to sliders so the output updates when a selection changes
    game_slider.observe(lambda change: on_game_change(change, games, game_slider, event_slider, out),
                        names="value")
    event_slider.observe(lambda change: on_event_change(change, games, game_slider, event_slider, out),
                         names="value")

    # Show widgets and output
    display(widgets.VBox([game_slider, event_slider, out]))
    # Initialize the output
    update_output(games, game_slider, event_slider, out)

