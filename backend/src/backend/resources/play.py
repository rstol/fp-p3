from dataclasses import dataclass
from typing import Any

from backend.resources.playid import PlayId


@dataclass(slots=True)
class Play:
    play_id: PlayId
    game_date: str
    event_type: int
    event_score: str
    event_score_margin: str
    possession_team_id: str
    event_desc_home: dict[str, Any]
    event_desc_away: dict[str, Any]
    note: str | None = None
    is_tagged: bool = False
