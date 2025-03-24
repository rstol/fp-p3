import argparse
import json
import logging
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from server.settings import SAMPLING_RATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_PROCESSES = 2 * (os.cpu_count() or 2)


def load_nba_dataset(split: str | None = None, name: str = "full"):
    """
    Load the NBA tracking dataset.

    Args:
        split (str, Optional): Dataset split to load ('train', 'test', 'validation')
        name (str): Dataset size to load ('tiny', 'small', 'medium', or 'full')
    """
    return load_dataset(
        "dcayton/nba_tracking_data_15_16",
        trust_remote_code=True,
        name=name,
        split=split,
        num_proc=NUM_PROCESSES,
    )


def downsample_moments(moments, rate):
    return moments[::rate]


def simplify_player_coords(player_coords):
    return [
        {"teamid": p["teamid"], "playerid": p["playerid"], "x": p["x"], "z": p["z"]}
        for p in player_coords
    ]


def process_play(play):
    """Process a single play for the dataset map function."""
    if "moments" not in play or not play["moments"]:
        return None

    processed_moments = []
    for m in downsample_moments(play["moments"], SAMPLING_RATE):
        processed_moments.append(
            {
                "quarter": m["quarter"],
                "game_clock": m["game_clock"],
                "shot_clock": m["shot_clock"],
                "ball_coordinates": m["ball_coordinates"],
                "player_coordinates": simplify_player_coords(m["player_coordinates"]),
            }
        )
    return {
        "game_id": play["gameid"],
        "primary_player_info": play["primary_info"],
        "secondary_player_info": play["secondary_info"],
        "event_id": play["event_info"]["id"],
        "event_type": play["event_info"]["type"],
        "possession_team_id": play["event_info"]["possession_team_id"],
        "event_desc_home": play["event_info"]["desc_home"],
        "event_desc_away": play["event_info"]["desc_away"],
        "moments": processed_moments,
    }


def extract_game_and_team_info(play):
    return {
        "game_id": play["gameid"],
        "game_date": play["gamedate"],
        "home_team_id": play["home"]["teamid"],
        "visitor_team_id": play["visitor"]["teamid"],
        "home_team": play["home"],
        "visitor_team": play["visitor"],
    }


def process_dataset(dataset, output_path):
    """Process the dataset using datasets library features."""
    logger.info("Start processing data.")

    filtered_dataset = dataset.filter(
        lambda x: "moments" in x and bool(x["moments"]),
        num_proc=NUM_PROCESSES,
        desc="Filtering empty plays",
    )

    processed_plays = filtered_dataset.map(
        process_play,
        num_proc=NUM_PROCESSES,
        remove_columns=list(set(dataset.column_names) - {"moments"}),
        desc="Processing plays",
    )

    plays_dir = output_path / "plays"
    os.makedirs(plays_dir, exist_ok=True)

    # TODO parallelise this due to disk io
    seen_game_ids = set()
    for play in processed_plays:
        game_id = play["game_id"]
        seen_game_ids.add(game_id)
        # play.to_json(plays_dir / f"{game_id}.jsonl")
        with open(plays_dir / f"{game_id}.jsonl", "a") as f:
            f.write(f"{json.dumps(play)}\n")

    info_dataset = filtered_dataset.map(
        extract_game_and_team_info,
        num_proc=NUM_PROCESSES,
        remove_columns=filtered_dataset.column_names,
        desc="Extracting game and team metadata",
    )

    seen_games = {}
    for row in info_dataset:
        game_id = row["game_id"]
        if game_id not in seen_games:
            seen_games[game_id] = {
                "game_id": game_id,
                "game_date": row["game_date"],
                "home_team_id": row["home_team_id"],
                "visitor_team_id": row["visitor_team_id"],
            }
    games_dataset = Dataset.from_list(list(seen_games.values())).sort("game_date")
    games_dataset.to_json(output_path / "games.jsonl")

    seen_teams = {}
    for row in info_dataset:
        for prefix in ["home", "visitor"]:
            team_id = row[f"{prefix}_team_id"]
            if team_id not in seen_teams:
                seen_teams[team_id] = row[f"{prefix}_team"]
    teams_dataset = Dataset.from_list(list(seen_teams.values())).sort("teamid")
    teams_dataset.to_json(output_path / "teams.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NBA tracking dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split to load (train or test)",
    )
    parser.add_argument(
        "--name",
        type=str,
        choices=["tiny", "small", "medium", "full"],
        default="tiny",
        help="Dataset size to load (tiny: 5 games, small: 25 games, medium: 100 games, full: all games)",
    )
    target_path = Path(os.getenv("DATASET_DIR", "data/nba_tracking_data")).resolve()

    args = parser.parse_args()
    dataset = load_nba_dataset(split=args.split, name=args.name)
    process_dataset(dataset, target_path)
