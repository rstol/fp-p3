import argparse
import json
import logging
import math
import os
from pathlib import Path

from datasets import load_dataset
from server.settings import SAMPLING_RATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaNEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return super().default(obj)


def load_nba_dataset(split: str = "train", name: str = "full"):
    """
    Load the NBA tracking dataset.

    Args:
        split (str): Dataset split to load ('train' or 'test')
        name (str): Dataset size to load ('tiny', 'small', 'medium', or 'full')
    """
    return load_dataset(
        "dcayton/nba_tracking_data_15_16",
        trust_remote_code=True,
        name=name,
        split=split,
        num_proc=4,
    )


def downsample_moments(moments, rate):
    return moments[::rate]


def simplify_player_coords(player_coords):
    return [
        {"teamid": p["teamid"], "playerid": p["playerid"], "x": p["x"], "z": p["z"]}
        for p in player_coords
    ]


def process_dataset(dataset, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/plays", exist_ok=True)

    game_data = {}
    team_data = {}
    game_plays = {}

    for play in dataset:
        game_id = play["gameid"]

        # Game metadata
        if game_id not in game_data:
            game_data[game_id] = {
                "id": game_id,
                "date": play["gamedate"],
                "home_team_id": play["home"]["teamid"],
                "visitor_team_id": play["visitor"]["teamid"],
            }
            game_plays[game_id] = []

        # Team data
        for team in ["home", "visitor"]:
            team_info = play[team]
            team_id = team_info["teamid"]
            if team_id not in team_data:
                team_data[team_id] = {
                    "id": team_id,
                    "name": team_info["name"],
                    "abbreviation": team_info["abbreviation"],
                    "players": team_info["players"],
                }

        # Build play objects
        event_info = play["event_info"]
        game_play = {
            "gameid": game_id,
            "eventid": event_info["id"],
            "type": event_info["type"],
            "possession_team_id": event_info["possession_team_id"],
            "desc_home": event_info["desc_home"],
            "desc_away": event_info["desc_away"],
            "moments": [],
        }

        if "moments" not in play:
            logger.warning(f"No moments found in play {event_info['id']} of game {game_id}")
            continue

        moments = play["moments"]
        if not moments:
            logger.warning(f"Empty moments array in play {event_info['id']} of game {game_id}")
            continue

        for m in downsample_moments(moments, SAMPLING_RATE):
            game_play["moments"].append(
                {
                    "quarter": m["quarter"],
                    "game_clock": m["game_clock"],
                    "shot_clock": m["shot_clock"],
                    "ball_coordinates": m["ball_coordinates"],
                    "player_coordinates": simplify_player_coords(m["player_coordinates"]),
                }
            )

        game_plays[game_id].append(game_play)

    # Sort games by date
    sorted_games = dict(sorted(game_data.items(), key=lambda x: x[1]["date"]))

    # Sort teams by ID
    sorted_teams = dict(sorted(team_data.items(), key=lambda x: x[0]))

    # Write JSON files
    with open(f"{output_path}/games.json", "w") as f:
        json.dump(sorted_games, f, indent=4, cls=NaNEncoder)

    with open(f"{output_path}/teams.json", "w") as f:
        json.dump(sorted_teams, f, indent=4, cls=NaNEncoder)

    for game_id, plays in game_plays.items():
        with open(f"{output_path}/plays/{game_id}.json", "w") as f:
            json.dump(plays, f, indent=4, cls=NaNEncoder)

    print("Saved processed data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NBA tracking dataset")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Dataset split to load (train or test)",
    )
    parser.add_argument(
        "--name",
        type=str,
        choices=["tiny", "small", "medium", "full"],
        default="small",
        help="Dataset size to load (tiny: 5 games, small: 25 games, medium: 100 games, full: all games)",
    )
    target_path = Path(os.getenv("DATASET_DIR", "data/nba_tracking_data")).resolve()

    args = parser.parse_args()
    dataset = load_nba_dataset(split=args.split, name=args.name)
    process_dataset(dataset, target_path)
