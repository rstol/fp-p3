import os
import sys
from pathlib import Path

# this file lives in e.g. …/backend/src/backend/scripts/my_script.py
script_dir = os.path.dirname(os.path.abspath(__file__))
# go up two levels to hit …/backend/src
src_root = os.path.abspath(os.path.join(script_dir, "..", "src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from backend.resources.dataset_manager import DatasetManager
from backend.settings import TRACKING_DIR
from backend.video.Event import Event

if __name__ == "__main__":
    dataset_manager = DatasetManager(TRACKING_DIR)

    games = dataset_manager.get_games()[
        :2
    ]  # BE AWARE that the [:2] is only for debugging

    for game in games:
        home = dataset_manager.get_team_details(game["home_team_id"])
        visitor = dataset_manager.get_team_details(game["visitor_team_id"])
        plays = dataset_manager.get_plays_for_game(game["game_id"])[
            :3
        ]  # BE AWARE that the [:3] is only for debugging
        for play in plays:
            event = Event(play, home, visitor)
            event.prerender()
