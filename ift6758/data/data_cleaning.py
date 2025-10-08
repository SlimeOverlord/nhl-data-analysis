import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def extract_player_info(event: Dict, player_type: str) -> Optional[str]:
    """
    Extract player information from event data.
    """
    try:
        details = event.get('details', {})

        if player_type == 'shooter':
            if 'scoringPlayerId' in details:
                return f"Player_{details['scoringPlayerId']}"
            elif 'shootingPlayerId' in details:
                return f"Player_{details['shootingPlayerId']}"
            elif 'players' in event:
                for player in event['players']:
                    if player.get('playerType') in ['Scorer', 'Shooter']:
                        return player.get('player', {}).get('fullName', f"Player_{player.get('playerId', 'Unknown')}")

        elif player_type == 'goalie':
            if 'goalieInNetId' in details:
                return f"Goalie_{details['goalieInNetId']}"
            elif 'players' in event:
                for player in event['players']:
                    if player.get('playerType') == 'Goalie':
                        return player.get('player', {}).get('fullName', f"Goalie_{player.get('playerId', 'Unknown')}")

    except Exception:
        pass

    return None


def get_coordinates(event: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract x, y coordinates from event.
    """
    try:
        details = event.get('details', {})
        x_coord = details.get('xCoord')
        y_coord = details.get('yCoord')

        if x_coord is not None and y_coord is not None:
            return float(x_coord), float(y_coord)
    except Exception:
        pass

    return None, None


def standardize_coordinates(x: Optional[float], y: Optional[float],
                           period: int, team_id: int, start_defend_left_team: int, start_defend_right_team: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Standardize coordinates so team always shoots to the right goal (for visualization purposes)
    """
    if x is None or y is None:
        return x, y

    # teams switch sides in period 2
    if(period % 2 != 0 and team_id == start_defend_right_team) or (period % 2 == 0 and team_id == start_defend_left_team):
        x = -x
        y = -y

    return x, y


def determine_game_strength(event: Dict, situation_code: Optional[str] = None) -> str:

    if situation_code:
        if 'PP' in situation_code.upper():
            return 'Power Play'
        elif 'SH' in situation_code.upper():
            return 'Short Handed'

    details = event.get('details', {})
    secondary_type = details.get('secondaryType', '')

    if 'Power Play' in secondary_type:
        return 'Power Play'
    elif 'Short Handed' in secondary_type:
        return 'Short Handed'

    return 'Even'


def is_empty_net(event: Dict) -> bool:

    details = event.get('details', {})

    if details.get('emptyNet'):
        return True

    if not details.get('goalieInNetId') and event.get('typeDescKey') == 'goal':
        return True

    return False


def get_shot_type(event: Dict) -> Optional[str]:

    details = event.get('details', {})

    shot_type = details.get('shotType')
    if shot_type:
        return shot_type

    secondary_type = details.get('secondaryType')
    if secondary_type:
        return secondary_type

    return 'Unknown'

def get_starting_sides (event: Dict, home_team_id: int, away_team_id: int) -> Tuple[int, int]:    
    # Checking which side each team defends at the start of the game
    home_team_defends = ""

    # For seasons 2019 and onwards, the first element of plays has the side the home team is defending
    period_start_play = event[0]
    if "homeTeamDefendingSide" in period_start_play:
        home_team_defends = period_start_play["homeTeamDefendingSide"]

    # For seasons before 2019, we loop through the shots on goal to find one that gives us enough information to determine each side
    else:
        for play in event:
            event_type = play.get('typeDescKey', '').lower()
            if event_type not in ['shot-on-goal', 'goal']:
                continue
            else:
                x, y = get_coordinates(play)

                if x is None or y is None or (-20 < x < 20):
                    continue
                else:
                    team_id = play.get('details', {}).get('eventOwnerTeamId')
                    if x < 0:
                        if team_id == home_team_id:
                            home_team_defends = "right"
                            break
                        elif team_id == away_team_id:
                            home_team_defends = "left"
                            break
                    elif x > 0:
                        if team_id == home_team_id:
                            home_team_defends = "left"
                            break
                        elif team_id == away_team_id:
                            home_team_defends = "right"
                            break
    
    if home_team_defends == "left":
        return (home_team_id, away_team_id)
    elif home_team_defends == "right":
        return (away_team_id, home_team_id)

def clean_play_by_play_data(game_data: Dict) -> pd.DataFrame:
    """
    Convert raw play-by-play JSON data to cleaned DataFrame.
    """
    events = []

    game_id = game_data.get('id')

    away_team = game_data.get("awayTeam", {})
    away_team_id = away_team.get("id")

    home_team = game_data.get("homeTeam", {})
    home_team_id = home_team.get("id")

    # Handle both old API format (plays.allPlays) and new format (plays as list)
    plays_data = game_data.get('plays', [])
    if isinstance(plays_data, dict):
        plays = plays_data.get('allPlays', [])
    else:
        plays = plays_data

    start_defend_left_team, start_defend_right_team = get_starting_sides(plays, home_team_id, away_team_id)

    for play in plays:
        event_type = play.get('typeDescKey', '').lower()

        if event_type not in ['shot-on-goal', 'goal', 'shot', 'missed-shot']:
            continue

        # skip missed and blocked shots for now
        if event_type == 'missed-shot' or play.get('result', {}).get('event') == 'Blocked Shot':
            continue

        period_info = play.get('periodDescriptor', {})
        period = period_info.get('number', 0)
        period_time = play.get('timeInPeriod', '00:00')

        team_id = play.get('details', {}).get('eventOwnerTeamId')
        team_abbrev = None

        x, y = get_coordinates(play)

        standardized_x, standardized_y = standardize_coordinates(x, y, period, team_id, start_defend_left_team, start_defend_right_team)

        if 'team' in play:
            team_abbrev = play['team'].get('triCode')
        elif 'details' in play:
            situation = play.get('situationCode', '')
            if situation and len(situation) >= 3:
                pass

        is_goal = event_type == 'goal'

        shooter_name = extract_player_info(play, 'shooter')
        goalie_name = extract_player_info(play, 'goalie')

        shot_type = get_shot_type(play)

        empty_net = is_empty_net(play)

        situation_code = play.get('situationCode')
        strength = determine_game_strength(play, situation_code)

        event_record = {
            'game_id': game_id,
            'period': period,
            'period_time': period_time,
            'team': team_abbrev if team_abbrev else team_id,
            'event_type': 'Goal' if is_goal else 'Shot',
            'x_coord': x,
            'y_coord': y,
            'standardized_x_coord': standardized_x,
            'standardized_y_coord': standardized_y,
            'shooter': shooter_name,
            'goalie': goalie_name,
            'shot_type': shot_type,
            'empty_net': empty_net,
            'strength': strength
        }

        events.append(event_record)

    df = pd.DataFrame(events)

    return df


def clean_season_data(season_data_obj) -> pd.DataFrame:
    """
    Clean all games from a SeasonData object.
    """
    all_events = []

    for game_id, game_data in season_data_obj.reg_season_data.items():
        try:
            game_df = clean_play_by_play_data(game_data)
            game_df['season'] = season_data_obj.year
            game_df['game_type'] = 'Regular'
            all_events.append(game_df)
        except Exception as e:
            print(f"Error processing regular season game {game_id}: {e}")

    for game_id, game_data in season_data_obj.playoffs_data.items():
        try:
            game_df = clean_play_by_play_data(game_data)
            game_df['season'] = season_data_obj.year
            game_df['game_type'] = 'Playoffs'
            all_events.append(game_df)
        except Exception as e:
            print(f"Error processing playoff game {game_id}: {e}")

    if all_events:
        combined_df = pd.concat(all_events, ignore_index=True)

        combined_df = combined_df.sort_values(['game_id', 'period', 'period_time'])

        return combined_df
    else:
        return pd.DataFrame(columns=[
            'game_id', 'period', 'period_time', 'team', 'event_type',
            'x_coord', 'y_coord', 'standardized_x_coord', 'standardized_y_coord',
            'shooter', 'goalie', 'shot_type',
            'empty_net', 'strength', 'season', 'game_type'
        ])


def get_additional_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # distance from net - goal is at x=89
    df['distance_from_net'] = np.sqrt((df['standardized_x_coord'] - 89)**2 + df['standardized_y_coord']**2)

    df['angle_from_net'] = np.degrees(np.arctan2(np.abs(df['standardized_y_coord']), 89 - df['standardized_x_coord']))

    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return 0
        try:
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0

    df['period_seconds'] = df['period_time'].apply(time_to_seconds)

    df['game_seconds'] = (df['period'] - 1) * 1200 + df['period_seconds']

    df['is_overtime'] = df['period'] > 3

    return df
