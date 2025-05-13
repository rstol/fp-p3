from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

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
        df = self._load_game()
        plays = self._build_plays(df)

        object.__setattr__(self, "data_df", df)
        object.__setattr__(self, "plays", MappingProxyType(plays))

    def get_play_by_id(self, play_id: PlayId):
        if play_id not in self.plays:
            return None
        return self.plays[play_id]

    def _load_game(self) -> pl.DataFrame:
        path = Path(TRACKING_DIR) / "plays" / f"{self.game_id}.parquet"
        try:
            return pl.read_parquet(path)
        except FileNotFoundError:
            return pl.DataFrame()

    @staticmethod
    def _build_plays(df: pl.DataFrame) -> dict[PlayId, Play]:
        return {
            (pid := PlayId(row["game_id"], row["event_id"])): Play(pid, row)
            for row in df.iter_rows(named=True)
        }
