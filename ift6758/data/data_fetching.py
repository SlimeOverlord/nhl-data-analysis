import pandas as pd
import os
import json
import time
import requests
import random

class SeasonData:
    def __init__(self, year):
        self.year = year

        # A python dictionary which contains the JSON data for regular season games, indexed by game IDs
        self.reg_season_data = {}

        # A python dictionary which contains the JSON data for playoff games, indexed by game IDs
        self.playoffs_data = {}

    def get_data_from_api(self):
        # Creating the directory to hold all the game data, if it hasn't been already 
        dataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NHLData")
        os.makedirs(dataDir, exist_ok=True)

        # The files that will hold the regular season and playoff game data for the year
        filename_reg = os.path.join(dataDir, f"{self.year}_reg_season.json")
        filename_playoffs = os.path.join(dataDir, f"{self.year}_playoffs_season.json")

        max_games_reg_season = self.get_max_games(self.year)
        
        ## Regular season data
        # First, we check if the data has already been saved somewhere. If yes, we load it directly 
        if os.path.exists(filename_reg):
            with open(filename_reg, 'r') as file:
                for line in file:
                    game_data = json.loads(line)
                    self.reg_season_data[f"{game_data["id"]}"] = game_data
        
        # If no, we call the api to get the data for each game and save it into a file
        else:
            for game_no in range(1, max_games_reg_season + 1):
                four_digit_game_no = f"{game_no:04d}"
                url = f"https://api-web.nhle.com/v1/gamecenter/{self.year}02{four_digit_game_no}/play-by-play"
                
                api_response = requests.get(url)

                # If the api finds the game, save it in the dict and write it to the file
                if api_response.status_code == 200:
                    api_response_json = api_response.json()
                    self.reg_season_data[f"{api_response_json["id"]}"] = api_response_json

                    with open(filename_reg, 'a') as file:
                        file.write(json.dumps(api_response_json) + "\n")

                    # Delay the API calls for a little time in order to not tax the API too much (to not get potentially blacklisted)
                    time.sleep(random.uniform(0.5, 1))
        
        ## Playoffs data
        # First, we check if the data has already been saved somewhere. If yes, we load it directly
        if os.path.exists(filename_playoffs):
            with open(filename_playoffs, 'r') as file:
                for line in file:
                    game_data = json.loads(line)
                    self.playoffs_data[f"{game_data["id"]}"] = game_data

        # If no, we call the api to get the data for each game and save it into a file
        else:
            # We have 8 matchups in round 1, 4 matchups in round 2, 2 matchups in the semifinals and 1 matchup in the finals
            round_and_matchup = [(1, 8),(2, 4),(3, 2),(4, 1)]

            for round_no, max_matchup in round_and_matchup:
                for matchup_no in range(1, max_matchup + 1):
                    # There are 7 games maximum per matchup
                    for game_no in range(1, 8):
                        four_digit_game_no = f"0{round_no}{matchup_no}{game_no}"
                        url = f"https://api-web.nhle.com/v1/gamecenter/{self.year}03{four_digit_game_no}/play-by-play"
                        api_response = requests.get(url)

                        # If the api finds the game, save it in the dict and write it to the file
                        if api_response.status_code == 200:
                            api_response_json = api_response.json()
                            self.playoffs_data[f"{api_response_json["id"]}"] = api_response_json
                            with open(filename_playoffs, 'a') as file:
                                file.write(json.dumps(api_response_json) + "\n")

                            # Delay the API calls for a little time in order to not tax the API too much (to not get potentially blacklisted)
                            time.sleep(random.uniform(0.5, 1))

                        # If not, then the previous game was the last of the matchup, so we stop there    
                        else:
                            break

    def get_max_games(self, year):
        if year in [2022, 2023, 2024, 2025]:
            return 1353
        elif year in [2017, 2018, 2019, 2020]:
            return 1271
        else:
            return 1230        