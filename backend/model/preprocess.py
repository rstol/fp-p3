import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from datasets import load_dataset, Dataset
import numpy as np
from typing import List, Any, Dict
import torch

import polars as pl

# Event type to event name mapping
event_type_to_name = {
    1: "made",
    2: "miss",
    3: "free_throw",
    4: "rebound",
    5: "turnover",
    6: "foul",
    7: "violation",
    8: "substitution",
    9: "timeout",
    10: "jump_ball",
    11: "ejection",
    12: "start_period",
    13: "end_period",
}    

NUM_PROCESSES = os.cpu_count()


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


def get_game_time(game_clock_secs, period):
    period_secs = 720 if period <= 4 else 300
    period_time = period_secs - game_clock_secs
    if period <= 4:
        return (period - 1) * 720 + period_time
    else:
        return 4 * 720 + (period - 5) * 300 + period_time
    
def get_shot_times(dataset):
    game_shot_times = {}
    for event in dataset: # TODO use datasets map
        moment = event['moments'][0]
        game_clock = moment['game_clock']
        quarter = moment['quarter']
        game_time = get_game_time(game_clock, quarter)
        event_type = event['event_info']["type"]
        team_abbreviations = {event[side]["teamid"]: event[side]["abbreviation"] for side in ["home", "visitor"]}
        if (event_type == 1) or (event_type == 2): # miss or make
            game_shot_times[game_time] = team_abbreviations[event['primary_info']["team_id"]]
    return game_shot_times

def get_shot_times_p(df: pl.DataFrame):
    # period seconds
    period_secs_expr = pl.when(pl.col("moments").list.first().struct.field("quarter") <= 4)
    period_secs_expr = period_secs_expr.then(720).otherwise(300)

    # Game time
    period_time_expr = period_secs_expr - pl.col("moments").list.first().struct.field("game_clock")
    game_time_expr = (
        pl.when(pl.col("moments").list.first().struct.field("quarter") <= 4)
        .then((pl.col("moments").list.first().struct.field("quarter") - 1) * 720 + period_time_expr)
        .otherwise(4 * 720 + (pl.col("moments").list.first().struct.field("quarter") - 5) * 300 + period_time_expr)
    )


def left_basket(moment):
    """
    This function takes a moment in the game and returns if the ball is in the left basket.
    """
    return (3.5 <= moment['ball_coordinates']['x'] <= 6) and (24 <= moment['ball_coordinates']['y'] <= 26)

def right_basket(moment):
    """
    This function takes a moment in the game and returns if the ball is in the right basket.
    """
    return (88 <= moment['ball_coordinates']['x'] <= 90.5) and (24 <= moment['ball_coordinates']['y'] <= 26)

def rotate_coordinates(event: Dict[str, Any]) -> Dict[str, Any]:
    direction = event["event_info"]["direction"]
    
    if direction == "unknown":
        return event
        
    rotated = event.copy()
    
    if direction == "left":
        for moment in rotated["moments"]:
            ball_x = moment["ball_coordinates"]["y"]
            ball_y = 94 - moment["ball_coordinates"]["x"]
            moment["ball_coordinates"]["x"] = ball_x
            moment["ball_coordinates"]["y"] = ball_y
            
            for player_coord in moment["player_coordinates"]:
                x = player_coord["y"]
                y = 94 - player_coord["x"]
                player_coord["x"] = x
                player_coord["y"] = y
                del player_coord["z"]
                
    elif direction == "right":
        for moment in rotated["moments"]:
            ball_x = 50 - moment["ball_coordinates"]["y"]
            ball_y = moment["ball_coordinates"]["x"]
            moment["ball_coordinates"]["x"] = ball_x
            moment["ball_coordinates"]["y"] = ball_y
            
            for player_coord in moment["player_coordinates"]:
                x = 50 - player_coord["y"]
                y = player_coord["x"]
                player_coord["x"] = x
                player_coord["y"] = y
                del player_coord["z"]
                
    return rotated

def rotate_court(game_id: str, game_df: pl.DataFrame):
    """
    Modifies the events to be worked with in a uniform format by rotating the coordinates depending on the direction of play.
    There is a bit of hard-coding in the directionality section, which is necessary due to mistimed events in the raw data.
    """
    # First pass to determine directions
    events = game_df.to_dicts()
    
    # First pass to identify first possession team and directions
    first_poss_team_id = None
    first_direction = None
    second_direction = None
    for event in events:
        quarter = event['moments'][0]['quarter']

        if quarter == 1:
          for moment in event['moments']:
            if left_basket(moment):
              first_poss_team_id = event['event_info']['possession_team_id']
              first_direction = 'left'
              second_direction = 'right'
              if game_id in ["0021500292", "0021500648"]:
                first_direction = 'right'
                second_direction = 'left'
              break
            elif right_basket(moment):
              first_poss_team_id = event['event_info']['possession_team_id']
              first_direction = 'right'
              second_direction = 'left'
              if game_id == "0021500648":
                first_direction = 'left'
                second_direction = 'right'
              break
          if first_poss_team_id is not None:
              break
        
    if first_poss_team_id is not None:
        processed_events = []
        for event in events:
            quarter = event["moments"][0]["quarter"]
            direction = "unknown"
          
            if quarter == 1:
                direction = first_direction
            elif quarter == 2:
                if event["event_info"]["possession_team_id"] == first_poss_team_id:
                    direction = first_direction
                else:
                    direction = second_direction
            elif quarter >= 3:
                if event["event_info"]["possession_team_id"] == first_poss_team_id:
                    direction = second_direction
                else:
                    direction = first_direction
            event["event_info"]["direction"] = direction
              
            # Now rotate coordinates
            rotated_event = rotate_coordinates(event)
            processed_events.append(rotated_event)
            
        return pl.DataFrame(processed_events)
    else:
        return game_df
      

def preprocess_nba_tracking_from_moments(sequence_length=50, downsample_factor=4, normalize=True):
    dataset = load_dataset("dcayton/nba_tracking_data_15_16", split="train", name = "tiny")
    sequences = []

    
    # for example in dataset:
    #     moments = example['moments']
    #     player_data = []
    #     for moment in moments[::downsample_factor]:
    #         coords = moment['player_coordinates']
    #         if len(coords) < 10:
    #             continue

    #         players = sorted(coords, key=lambda p: (p['teamid'], p['playerid']))
    #         if len(players) < 10:
    #             continue

    #         frame = []
    #         # Offensive (first 5 players)
    #         for player in players[:5]:
    #             frame.extend([player['x'], player['y']])
    #         # Defensive (next 5 players)
    #         for player in players[5:10]:
    #             frame.extend([player['x'], player['y']])
    #         # Ball coordinates (include z if requested)
    #         ball = moment['ball_coordinates']
    #         frame.extend([ball['x'], ball['y'], ball['z']])

    #         if normalize:
    #             frame = np.array(frame)
    #             frame[0::2] /= 94  # Normalize x (court length)
    #             frame[1::2] /= 50  # Normalize y (court width)
    #             frame = frame.tolist()

    #         player_data.append(frame)

    #     player_data = np.array(player_data)
    #     # Downsample from 25 fps to 25/4 = 6.25 fps (take every 4th frame)
    #     player_data = player_data[::downsample_factor]
    #     if player_data.shape[0] >= sequence_length:
    #                 trimmed = player_data[:sequence_length]
    #     flattened = trimmed.reshape(sequence_length, -1)
    #         sequences.append(flattened)
    # sequences = np.stack(sequences)
    # return torch.tensor(sequences, dtype=torch.float32)


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

    target_path = Path(os.getenv("DATASET_DIR", "data/nba_tracking_data")).resolve()

    args = parser.parse_args()
    dataset = load_nba_dataset(split="train", name=args.name)

    df_plays = dataset.to_polars()
    df_filtered = df_plays.filter(
        (pl.col("moments").list.len() > 0) &(pl.col("event_info").struct.field("possession_team_id").is_not_null()) &
        (~pl.col("event_info").struct.field("possession_team_id").is_nan()) &
        (pl.col("moments").list.first().struct.field("player_coordinates").list.len() == 10) &
        pl.col("event_info").struct.field("type").is_in([1, 2, 5, 6])
    )

    result = df_filtered.group_by("gameid").map_groups(lambda df: rotate_court(df["gameid"][0], df))

    # create feature vectors and store them efficient file format for each game
    for game_id, game_df in result.iter_rows():
        # The features are 
        # 1. player coordinates (sorted, without z)
        # 2. ball coordinates 
        # 3. game time
        # 4. shot clock
        # 5. period
        # 6. player velocities
        # 7. ball velocity
        # 8. possesion team id
        # maybe add score?
        
        # Loop over moments and create feature vectors
        # TODO: add shot clock, period, player velocities, ball velocity

        # Moments can overlap temporally, so previously processed time points are
        # skipped along with clock stoppages.
        # if game_time <= cur_time: continue
        # Check if game is over
        # while game_time > event_stream[event_idx]["game_time"]:
        #         event_idx += 1
        #         if event_idx >= len(event_stream):
        #             game_over = True
        #             break
        #     if game_over:
        #         break

        game_df.write_parquet(target_path / f"{game_id}_X.parquet") 
        # TODO compute y soft-labels and store them in _y.parquet