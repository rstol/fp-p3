import polars as pl


class DatasetManager:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.teams = pl.read_ndjson(f"{self.data_dir}/teams.jsonl")
        self.games = pl.read_ndjson(f"{self.data_dir}/games.jsonl")

    def get_teams(self) -> list[dict[str, str | int | list[dict[str, str | int]]]]:
        return self.teams.to_dicts()

    def get_team_details(self, team_id: str) -> dict[str, str] | None:
        team_id = int(team_id)
        team_dicts = self.teams.filter(pl.col("teamid") == team_id).head(1).to_dicts()
        return team_dicts[0] if len(team_dicts) > 0 else None

    def get_games(self):
        return self.games.to_dicts()

    def get_games_for_team(self, team_id: str, as_dicts: bool = True) -> list[dict[str, str]]:
        team_id = int(team_id)
        games = self.games.filter(
            (pl.col("home_team_id") == team_id) | (pl.col("visitor_team_id") == team_id)
        )
        return games.to_dicts() if as_dicts else games

    def get_game_details(self, game_id: str) -> dict[str, str] | None:
        game_dicts = self.games.filter(pl.col("game_id") == game_id).head(1).to_dicts()
        return game_dicts[0] if len(game_dicts) > 0 else None

    def get_plays_for_game(self, game_id: str, as_dicts: bool = True) -> list[dict[str, str]]:
        plays = self._load_game_plays(game_id)
        return plays.to_dicts() if as_dicts else plays

    def get_play_id(self, game_id: str, play_id: str) -> dict[str, str] | None:
        plays = self._load_game_plays(game_id)
        play_dicts = plays.filter(pl.col("event_id") == play_id).head(1).to_dicts()
        return play_dicts[0] if len(play_dicts) > 0 else None

    def _load_game_plays(self, game_id: str) -> list[dict[str, str]]:
        try:
            return pl.read_ndjson(f"{self.data_dir}/plays/{game_id}.jsonl")
        except FileNotFoundError:
            return []
