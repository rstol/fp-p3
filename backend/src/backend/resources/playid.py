from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PlayId:
    game_id: str
    event_id: str
