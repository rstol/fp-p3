from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PlayId:
    game_id: str
    event_id: str

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PlayId):
            return NotImplemented
        return self.game_id == value.game_id and self.event_id == value.event_id
