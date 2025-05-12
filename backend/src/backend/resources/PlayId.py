class PlayId:
    def __init__(self, game_id: str, event_id: str):
        self._game_id = game_id
        self._event_id = event_id

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PlayId):
            return NotImplemented
        return self._game_id == value.game_id and self._event_id == value.event_id

    def __hash__(self):
        return hash((self._game_id, self._event_id))

    @property
    def game_id(self):
        return self._game_id

    @property
    def event_id(self):
        return self._event_id
