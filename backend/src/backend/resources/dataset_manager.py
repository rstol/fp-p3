import polars as pl
from polars import DataFrame

from backend.resources.game import Game
from backend.resources.play import Play
from backend.resources.playid import PlayId


class DatasetManager:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.teams = pl.read_ndjson(f"{self.data_dir}/teams.jsonl")
        self.games = pl.read_ndjson(f"{self.data_dir}/games.jsonl")
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
        game = self._get_game(game_id)
        return game.data_df.to_dicts() if as_dicts else game.data_df

    def get_play_id(self, game_id: str, event_id: str) -> Play | None:
        play_id = PlayId(game_id, event_id)
        game = self._get_game(game_id)
        return game.get_play_by_id(play_id)

    def _get_game(self, game_id: str):
        return self.games_data.setdefault(game_id, Game(game_id))
