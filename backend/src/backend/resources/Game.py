import polars as pl

from backend.resources.Play import Play
from backend.resources.PlayId import PlayId
from backend.settings import TRACKING_DIR


class Game:
    def __init__(self, game_id: str):
        self._game_id = game_id
        self.data_df = self._load_game()
        self.plays: dict[PlayId, Play] = self._get_plays()

    def _get_plays(self):
        plays = {}
        # looping over data_df does not work I guess
        for play in self.data_df.iter_rows(named=True):
            play_id = PlayId(play["game_id"], play["event_id"])
            plays[play_id] = Play(play_id, play)
        return plays

    def get_play_by_id(self, play_id: PlayId):
        if play_id not in self.plays:
            return None
        return self.plays[play_id]

    def _load_game(self):
        try:
            return pl.read_parquet(f"{TRACKING_DIR}/plays/{self._game_id}.parquet")
        except FileNotFoundError:
            return pl.DataFrame()

    # TODO more helper functions
