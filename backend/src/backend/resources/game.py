from dataclasses import dataclass, field

import polars as pl

from backend.resources.play import Play
from backend.resources.playid import PlayId
from backend.settings import TRACKING_DIR


@dataclass(slots=True, frozen=True)
class Game:
    game_id: str
    data_df: pl.DataFrame = field(init=False)
    plays: dict[PlayId, Play] = field(init=False)

    def __post_init__(self):
        self.data_df = self._load_game()
        self.plays = self._get_plays()

    def _get_plays(self):
        plays = {}
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
            return pl.read_parquet(f"{TRACKING_DIR}/plays/{self.game_id}.parquet")
        except FileNotFoundError:
            return pl.DataFrame()

    # TODO more helper functions
