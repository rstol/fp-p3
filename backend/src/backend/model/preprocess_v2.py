import argparse
import os
import pickle
import typing as t
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import polars as pl
from datasets import Dataset

from backend.model.Possesion import Possession
from backend.model.preprocess import (
    find_direction_possession_team,
    get_games_ranges,
    load_nba_dataset,
    set_direction,
)
from backend.settings import COURT_LENGTH, COURT_WIDTH, EMBEDDINGS_DIR, GAMES_DIR, TRACKING_DIR

NUM_PROCESSES = os.cpu_count() or 2


def rotate_coordinates(event: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    After rotation the attacking team moves towards the left
    """
    direction = event["event_info"]["direction"]

    if direction == "unknown":
        direction = (
            "right" if event["moments"][0]["ball_coordinates"]["x"] > (COURT_LENGTH / 2) else "left"
        )

    rotated = event.copy()

    if direction == "right":
        for moment in rotated["moments"]:
            moment["ball_coordinates"]["x"] = COURT_LENGTH - moment["ball_coordinates"]["x"]
            moment["ball_coordinates"]["y"] = COURT_WIDTH - moment["ball_coordinates"]["y"]
            for player_coord in moment["player_coordinates"]:
                player_coord["x"] = COURT_LENGTH - player_coord["x"]
                player_coord["y"] = COURT_WIDTH - player_coord["y"]
    return rotated


def convert_game(game_id: str, game_dataset: Dataset, shots_df: pd.DataFrame):
    game_df = game_dataset.to_polars()
    first_poss_team_id, first_direction = find_direction_possession_team(game_id, game_df, shots_df)
    if first_poss_team_id is None:
        return game_df
    possesion_ids = []
    for event in game_df.iter_rows(named=True):
        set_direction(first_poss_team_id, first_direction, event)
        event = rotate_coordinates(event)
        possesion = Possession(event, game_id)
        p_id = (possesion.gameid, possesion.eventid, possesion.off_teamid)
        possesion_ids.append(p_id)
        with open(f"{GAMES_DIR}/{'_'.join(p_id)}.pkl", "wb") as f:
            pickle.dump(possesion, f)
    return possesion_ids


def convert_scene_trajectory(dataset: Dataset):
    game_ranges = get_games_ranges(dataset)
    game_datasets = {
        game_id: dataset.select(range(start, end)) for game_id, (start, end) in game_ranges.items()
    }

    Path(GAMES_DIR).mkdir(parents=True, exist_ok=True)
    shots_df = pd.read_csv(os.path.join(TRACKING_DIR, "shots_fixed.csv"))
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(
            convert_game,
            [(game_id, game_dataset, shots_df) for game_id, game_dataset in game_datasets.items()],
        )
    play_ids_flat = [play_id for play_ids in results for play_id in play_ids]
    print(f"Saved {len(play_ids_flat)} number of plays...")
    return play_ids_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NBA tracking dataset")
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

    args = parser.parse_args()
    dataset: Dataset = load_nba_dataset(split="train", name=args.name)

    dataset = dataset.with_format("polars")

    # The following filters out approx 50% of the plays
    def filter_plays(df):
        return df.filter(
            (pl.col("moments").list.len() >= 100)
            & (pl.col("event_info").struct.field("possession_team_id").is_not_null())
            & (~pl.col("event_info").struct.field("possession_team_id").is_nan())
            & (pl.col("moments").list.first().struct.field("player_coordinates").list.len() == 10)
            & pl.col("event_info").struct.field("type").is_in([1, 2])  # made or missed shots
        )

    dataset = dataset.map(
        filter_plays, batched=True, num_proc=NUM_PROCESSES, desc="Filtering dataset"
    )

    play_ids = convert_scene_trajectory(dataset)
    play_ids_df = pd.DataFrame(play_ids, columns=["game_id", "event_id", "offense_team_id"])
    play_ids_df.to_csv(f"{EMBEDDINGS_DIR}/embedding_sources.csv", index=False)
