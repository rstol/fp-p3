from typing import Any

from backend.resources.PlayId import PlayId


class Play:
    def __init__(self, play_id: PlayId, play_data: dict[str, Any]):
        self.play_id = play_id
        self.play_data = play_data

    def get_play_details(self):
        # TODO return details as dict for frontend
        return self.play_data

    def get_ppp(self):
        # TODO calc point per possession
        pass

    def get_video(self):
        # TODO animate and return video of play
        pass
