from dataclasses import dataclass
from typing import Any

from backend.resources.playid import PlayId


@dataclass(slots=True, frozen=True)
class Play:
    play_id: PlayId
    play_data: dict[str, Any]

    def get_play_details(self) -> dict[str, Any]:
        # TODO return details as dict for frontend
        return self.play_data

    def get_ppp(self) -> float:
        # TODO calc point per possession
        raise NotImplementedError

    def get_video(self) -> bytes:
        # TODO animate and return video of play
        raise NotImplementedError
