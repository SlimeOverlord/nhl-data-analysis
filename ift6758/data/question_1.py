import asyncio
import aiohttp
import json
import os
from typing import Dict, Tuple, Optional, List

import pandas as pd


# ---------------------------
# Public: Scrape player stats
# ---------------------------

def get_player_stats(year: int, player_type: str) -> pd.DataFrame:
    """
    Scrape skater/goalie tables from hockey-reference for a given season start year.
    player_type: 'skaters' or 'goalies'
    """
    if player_type not in ["skaters", "goalies"]:
        raise RuntimeError("'player_type' must be either 'skaters' or 'goalies'")

    url = f'https://www.hockey-reference.com/leagues/NHL_{year}_{player_type}.html'
    print(f"Retrieving data from '{url}'...")
    df = pd.read_html(url, header=1)[0]

    # Keep 'TOT' rows for multi-team players
    players_multiple_teams = df[df['Tm'].eq('TOT')]

    # Drop per-team rows when a TOT row exists
    df = df[~df['Player'].isin(players_multiple_teams['Player'])]

    # Occasionally extra header rows show up inside the table; remove them
    df = df[df['Player'] != "Player"]

    # Concat final set
    df = pd.concat([df, players_multiple_teams], ignore_index=True)

    return df


# ---------------------------
# Async game fetchers (NHL)
# ---------------------------

def regular_season_game_url(year: int, game_no: int) -> str:
    four_digit = f"{game_no:04d}"
    return f"https://api-web.nhle.com/v1/gamecenter/{year}02{four_digit}/play-by-play"


def playoff_game_url(year: int, game_tuple: Tuple[int, int, int]) -> str:
    # (round_no, matchup_no, match_game_no) -> 0R M G
    r, m, g = game_tuple
    four_digit = f"0{r}{m}{g}"
    return f"https://api-web.nhle.com/v1/gamecenter/{year}03{four_digit}/play-by-play"


def default_max_games(year: int) -> int:
    # Conservative caps (update as needed)
    if year in [2022, 2023, 2024, 2025]:
        return 1353
    elif year in [2017, 2018, 2019, 2020]:
        return 1271
    return 1230


class SeasonData:
    """
    Async NHL season downloader with caching and polite concurrency.
    Writes JSON Lines files:
      - NHLData/{year}_reg_season.json
      - NHLData/{year}_playoffs_season.json
    """
    def __init__(self, year: int, base_dir: Optional[str] = None):
        self.year = year
        self.reg_season_data: Dict[str, dict] = {}
        self.playoffs_data: Dict[str, dict] = {}

        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_dir, "NHLData")
        os.makedirs(self.data_dir, exist_ok=True)

        self.filename_reg = os.path.join(self.data_dir, f"{self.year}_reg_season.json")
        self.filename_playoffs = os.path.join(self.data_dir, f"{self.year}_playoffs_season.json")

    # ---------- Public API ----------

    async def get_data_from_api(
        self,
        max_concurrency: int = 12,
        max_games_reg_season: Optional[int] = None,
        timeout_seconds: float = 10.0,
        retries: int = 5,
        backoff_base: float = 0.5,
    ):
        """
        Download regular season and playoffs for the given year.
        Respects existing cache files; only fetches when missing.
        """
        if max_games_reg_season is None:
            max_games_reg_season = default_max_games(self.year)

        # Load existing cache if available
        self._load_existing_cache()

        connector = aiohttp.TCPConnector(limit_per_host=max_concurrency, limit=0)  # let semaphore control concurrency
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=timeout_seconds, sock_read=timeout_seconds)
        headers = {
            "User-Agent": "nhl-data-grabber/1.0 (contact: you@example.com)",
            "Accept": "application/json",
        }

        sem = asyncio.Semaphore(max_concurrency)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            # --- Regular Season ---
            if not os.path.exists(self.filename_reg):
                print(f"Fetching regular season {self.year} ({max_games_reg_season} games) ...")
                reg_results = await self._fetch_regular_season(
                    session, sem, max_games_reg_season, retries, backoff_base
                )
                if reg_results:
                    self._write_jsonl(self.filename_reg, reg_results)

            # --- Playoffs ---
            if not os.path.exists(self.filename_playoffs):
                print(f"Fetching playoffs {self.year} ...")
                playoffs_results = await self._fetch_playoffs(session, sem, retries, backoff_base)
                if playoffs_results:
                    self._write_jsonl(self.filename_playoffs, playoffs_results)

        print("Done.")

    # ---------- Internal helpers ----------

    def _load_existing_cache(self):
        def load_file(path: str) -> Dict[str, dict]:
            out: Dict[str, dict] = {}
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        try:
                            g = json.loads(line)
                            out[str(g["id"])] = g
                        except Exception:
                            continue
            return out

        self.reg_season_data.update(load_file(self.filename_reg))
        self.playoffs_data.update(load_file(self.filename_playoffs))

    @staticmethod
    def _write_jsonl(path: str, items: List[dict]):
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")

    async def _bounded_fetch_json(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        url: str,
        retries: int,
        backoff_base: float,
    ) -> Optional[dict]:
        """
        GET with retries + exponential backoff + jitter.
        Returns parsed JSON (dict) or None.
        """
        async with sem:
            delay = backoff_base
            for attempt in range(retries):
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        # Retry on 429/5xx
                        if resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 8.0)  # cap backoff
                            continue
                        # Non-retryable statuses
                        return None
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8.0)
            return None

    async def _fetch_regular_season(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        max_games: int,
        retries: int,
        backoff_base: float,
    ) -> List[dict]:
        tasks = []
        for n in range(1, max_games + 1):
            url = regular_season_game_url(self.year, n)
            tasks.append(self._bounded_fetch_json(session, sem, url, retries, backoff_base))

        results: List[dict] = []
        for coro in asyncio.as_completed(tasks):
            game = await coro
            if game:
                gid = str(game.get("id", ""))
                if gid and gid not in self.reg_season_data:
                    self.reg_season_data[gid] = game
                    results.append(game)
        print(f"  Saved {len(results)} new regular-season games.")
        return results

    async def _fetch_playoffs(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        retries: int,
        backoff_base: float,
    ) -> List[dict]:
        # NHL playoff structure per season: Rounds: (round_no, number_of_matchups)
        round_and_matchup = [(1, 8), (2, 4), (3, 2), (4, 1)]

        tasks = []
        for r, max_m in round_and_matchup:
            for m in range(1, max_m + 1):
                for g in range(1, 8):  # up to 7 games per series
                    url = playoff_game_url(self.year, (r, m, g))
                    tasks.append(self._bounded_fetch_json(session, sem, url, retries, backoff_base))

        results: List[dict] = []
        for coro in asyncio.as_completed(tasks):
            game = await coro
            if game:
                gid = str(game.get("id", ""))
                if gid and gid not in self.playoffs_data:
                    self.playoffs_data[gid] = game
                    results.append(game)
        print(f"  Saved {len(results)} new playoff games.")
        return results