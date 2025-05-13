# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "polars",
#     "py7zr",
# ]
# ///
import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from datasets import Dataset, load_dataset

from backend.settings import EMBEDDINGS_DIR, TEAM_IDS_SAMPLE, TRACKING_DIR

NUM_PROCESSES = os.cpu_count()


def load_nba_dataset(split: str | None = None, name: str = "full"):
    """
    Load the NBA tracking dataset.

    Args:
        split (str, Optional): Dataset split to load ('train', 'test', 'validation')
        name (str): Dataset size to load ('tiny', 'small', 'medium', or 'full')
    """
    return load_dataset(
        f"{os.getenv('SCRIPTS_DIR', 'backend/scripts')}/load_nba_tracking_data_15_16.py",
        trust_remote_code=True,
        name=name,
        split=split,
        num_proc=NUM_PROCESSES,
    )


def downsample_moments(moments: list[dict], rate: int) -> list[dict]:
    return moments[::rate]


def simplify_player_coords(player_coords: list[dict]) -> list[dict]:
    return [
        {"teamid": p["teamid"], "playerid": p["playerid"], "x": p["x"], "y": p["y"]}
        for p in player_coords
    ]


def process_play(play: dict[str, Any], sampling_rate: int) -> dict[str, Any]:
    """Process a single play for the dataset map function."""
    play["moments"] = downsample_moments(play["moments"], sampling_rate)
    for moment in play["moments"]:
        moment["player_coordinates"] = simplify_player_coords(moment["player_coordinates"])

    score_margin = play["event_info"]["score_margin"]
    if score_margin == "TIE":
        score_margin = "0"

    return {
        "game_id": play["gameid"],
        "primary_player_info": play["primary_info"],
        "secondary_player_info": play["secondary_info"],
        "event_id": play["event_info"]["id"],
        "event_type": play["event_info"]["type"],
        "event_score": play["event_info"]["score"],
        "event_score_margin": score_margin,
        "possession_team_id": play["event_info"]["possession_team_id"],
        "event_desc_home": play["event_info"]["desc_home"],
        "event_desc_away": play["event_info"]["desc_away"],
        "moments_processed": play["moments"],
    }


def extract_game_and_team_info(play: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": play["gameid"],
        "game_date": play["gamedate"],
        "home_team_id": play["home"]["teamid"],
        "visitor_team_id": play["visitor"]["teamid"],
        "home_team": play["home"],
        "visitor_team": play["visitor"],
    }


def process_dataset(dataset: Dataset, output_path: Path, sampling_rate: int) -> None:
    """Process the dataset using datasets library features."""
    # Use preprocessed plays for which embeddings are saved
    embedding_ids = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "embedding_sources.csv"), dtype="str")
    play_ids = list(embedding_ids["game_id"] + "_" + embedding_ids["event_id"])

    dataset = dataset.with_format("polars")

    def filter_pair_id(df):
        return (
            df.with_columns(
                [
                    (
                        pl.col("gameid").cast(str)
                        + "_"
                        + pl.col("event_info").struct.field("id").cast(str)
                    ).alias("pair_id")
                ]
            )
            .filter(
                pl.col("pair_id").is_in(play_ids)
                & pl.col("home").struct.field("teamid").cast(str).is_in(TEAM_IDS_SAMPLE)
            )
            .drop("pair_id")
        )

    dataset = dataset.map(
        filter_pair_id,
        batched=True,
        desc="Filtering dataset",
        num_proc=NUM_PROCESSES,
    )
    dataset = dataset.with_format(None)  # Back to default dict format

    dataset_plays = dataset.select_columns(
        [
            "gameid",
            "primary_info",
            "secondary_info",
            "event_info",
            "moments",
        ]
    )
    dataset_plays = dataset_plays.map(
        process_play,
        fn_kwargs={"sampling_rate": sampling_rate},
        remove_columns=list(set(dataset_plays.column_names)),
        num_proc=NUM_PROCESSES,
        desc="Processing plays",
    )
    dataset_plays = dataset_plays.rename_column("moments_processed", "moments")

    plays_dir = output_path / "plays"
    os.makedirs(plays_dir, exist_ok=True)

    def get_games_ranges(dataset: Dataset) -> dict:
        game_ids_all = dataset["gameid"]
        game_ranges = {}
        start_idx = 0
        current_game = game_ids_all[0]

        for idx, game_id in enumerate(game_ids_all):
            if game_id != current_game:
                game_ranges[current_game] = (start_idx, idx)
                current_game = game_id
                start_idx = idx
        # Add the final game
        game_ranges[current_game] = (start_idx, len(game_ids_all))
        return game_ranges

    game_ranges = get_games_ranges(dataset)
    play_datasets = {
        game_id: dataset_plays.select(range(start, end))
        for game_id, (start, end) in game_ranges.items()
    }

    for game_id, game_dataset in play_datasets.items():
        output_file = plays_dir / f"{game_id}.parquet"
        # Load just a single game into memory with to_polars
        print(f"Writing {output_file} for game {game_id}")
        game_dataset.to_polars(batched=False).write_parquet(output_file)

    # Extract game and team info
    dataset_info = dataset.select_columns(
        [
            "gameid",
            "gamedate",
            "home",
            "visitor",
        ]
    )
    dataset_info = dataset_info.map(
        extract_game_and_team_info,
        remove_columns=dataset_info.column_names,
    )

    dataset_game = dataset_info.select_columns(
        [
            "game_id",
            "game_date",
            "home_team_id",
            "visitor_team_id",
        ]
    )
    df_game = dataset_game.to_polars()
    df_game = df_game.unique(subset=["game_id"], keep="first")
    df_game = df_game.sort("game_date")
    df_game.write_ndjson(output_path / "games.jsonl")

    df_teams_home = dataset_info.select_columns(["home_team_id", "home_team"]).to_polars()
    df_teams_home = df_teams_home.rename({"home_team_id": "teamid", "home_team": "name"})
    df_teams_visitor = dataset_info.select_columns(["visitor_team_id", "visitor_team"]).to_polars()
    df_teams_visitor = df_teams_visitor.rename(
        {"visitor_team_id": "teamid", "visitor_team": "name"}
    )
    df_teams = pl.concat([df_teams_home, df_teams_visitor])
    df_teams = df_teams.unique(subset=["teamid"])
    df_teams = df_teams.sort("teamid")
    df_teams = df_teams.drop("teamid")
    df_teams = df_teams.unnest("name")
    df_teams.write_ndjson(output_path / "teams.jsonl")


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
        help="Dataset size (tiny: 5 games, small: 25 games, medium: 100 games, full: all games)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=3,
        help="Rate at which to downsample moments (e.g., 3 means keep every 3rd moment)",
    )
    target_path = Path(TRACKING_DIR).resolve()
    args = parser.parse_args()
    dataset = load_nba_dataset(split=args.split, name=args.name)
    process_dataset(dataset, target_path, args.sampling_rate)
