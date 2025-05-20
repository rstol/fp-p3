from pathlib import Path

import polars as pl
from polars import DataFrame

from backend.resources.game import Game
from backend.settings import VIDEO_DATA_DIR
from backend.video.event import Event


class DatasetManager:
    def __init__(self, data_dir: str | None) -> None:
        if not data_dir or not Path(data_dir).exists():
            msg = "Data directory given to DatasetManager does not exist"
            raise ValueError(msg)
        self.data_dir = data_dir

        self.teams = pl.read_ndjson(f"{self.data_dir}/teams.jsonl")
        self.games = pl.read_ndjson(f"{self.data_dir}/games.jsonl")
        self.plays = pl.scan_parquet(f"{self.data_dir}/plays/*")
        self.games_data: dict[str, Game] = {}

    def get_teams(self) -> list[dict[str, str | int | list[dict[str, str | int]]]]:
        return self.teams.to_dicts()

    def get_team_details(self, team_id: str) -> dict[str, str] | None:
        team_id = int(team_id)
        team_dicts = self.teams.filter(pl.col("teamid") == team_id).head(1).to_dicts()
        return team_dicts[0] if len(team_dicts) > 0 else None

    def get_games(self):
        return self.games.to_dicts()

    def get_games_for_team(
        self, team_id: str, as_dicts: bool = True
    ) -> list[dict[str, str]] | DataFrame:
        team_id = int(team_id)
        games = self.games.filter(
            (pl.col("home_team_id") == team_id) | (pl.col("visitor_team_id") == team_id)
        )
        return games.to_dicts() if as_dicts else games

    def get_game_details(self, game_id: str) -> dict[str, str] | None:
        game_dicts = self.games.filter(pl.col("game_id") == game_id).head(1).to_dicts()
        return game_dicts[0] if len(game_dicts) > 0 else None

    def get_plays_for_game(
        self, game_id: str, as_dicts: bool = True
    ) -> list[dict[str, str]] | DataFrame:
        game = self._load_game_plays(game_id)
        return game.to_dicts() if as_dicts else game

    def get_plays_for_games(self, game_ids: list[str]) -> DataFrame:
        return self.plays.filter(pl.col("game_id").is_in(game_ids))

    def get_play_raw_data(self, game_id: str, play_id: str) -> dict[str, str] | None:
        plays = self._load_game_plays(game_id).fill_null("").fill_nan(0)
        play_dicts = plays.filter(pl.col("event_id") == play_id).head(1).to_dicts()
        return play_dicts[0] if len(play_dicts) > 0 else None

    def get_play_details(self, game_id: str, play_id: str) -> dict[str, str] | None:
        # TODO use play Object abstraction instead of raw dict
        play = self.get_play_raw_data(game_id, play_id)
        if play:
            play["possession_team_id"] = int(play["possession_team_id"])
            del play["moments"]
            del play["primary_player_info"]
            del play["secondary_player_info"]
        return play

    def get_play_video(self, game_id: str, event_id: str) -> bytes | None:
        prerender_file = Path(VIDEO_DATA_DIR) / game_id / f"{event_id}.mp4"
        try:
            with prerender_file.open("rb") as f:
                return f.read()
        except FileNotFoundError:
            event_raw = self.get_play_raw_data(game_id, event_id)
            game = self.get_game_details(game_id)
            if not game:
                return None
            home = self.get_team_details(game["home_team_id"])
            visitor = self.get_team_details(game["visitor_team_id"])
            event = Event(event_raw, home, visitor)
            return event.generate_mp4()

    def _load_game_plays(self, game_id: str) -> DataFrame:
        return self.plays.filter(pl.col("game_id") == game_id).collect()
