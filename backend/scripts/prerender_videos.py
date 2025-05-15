import multiprocessing
from functools import partial

from backend.resources.dataset_manager import DatasetManager
from backend.settings import FRONTEND_PUBLIC_VIDEOS_DIR, TRACKING_DIR
from backend.video.Event import Event


def process_game(dataset_manager, game):
    plays = dataset_manager.get_plays_for_game(game["game_id"])
    home = dataset_manager.get_team_details(game["home_team_id"])
    visitor = dataset_manager.get_team_details(game["visitor_team_id"])
    for play in plays:
        event = Event(play, home, visitor)
        event.prerender(video_dir=FRONTEND_PUBLIC_VIDEOS_DIR)


if __name__ == "__main__":
    dataset_manager = DatasetManager(TRACKING_DIR)
    games = dataset_manager.get_games()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(partial(process_game, dataset_manager), games)
