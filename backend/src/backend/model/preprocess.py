import argparse
import os
import pickle
import typing as t
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from datasets import Dataset, load_dataset

from backend.model.utils import get_game_time
from backend.settings import COURT_LENGTH, COURT_WIDTH, GAMES_DIR, TRACKING_DIR

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

NUM_PROCESSES = os.cpu_count() or 2


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
        num_proc=NUM_PROCESSES // 2,
    )


def left_court_offense(moment, poss_team_id):
    """
    This function takes a moment in the game and returns if ball and possession players are
    in the left court side.
    """
    return moment["ball_coordinates"]["x"] < COURT_LENGTH / 2 and (
        all(
            player["x"] < COURT_LENGTH / 2
            for player in moment["player_coordinates"]
            if player["teamid"] == poss_team_id
        )
    )


def right_court_offense(moment, poss_team_id):
    """
    This function takes a moment in the game and returns if the ball is in the right basket.
    """
    return moment["ball_coordinates"]["x"] > COURT_LENGTH / 2 and (
        all(
            player["x"] > COURT_LENGTH / 2
            for player in moment["player_coordinates"]
            if player["teamid"] == poss_team_id
        )
    )


def rotate_coordinates(event: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    After rotation the attacking team moves towards increasing y-coordinates.
    """
    direction = event["event_info"]["direction"]

    if direction == "unknown":
        return event

    rotated = event.copy()

    if direction == "left":
        for moment in rotated["moments"]:
            ball_x = moment["ball_coordinates"]["y"]
            ball_y = COURT_LENGTH - moment["ball_coordinates"]["x"]
            moment["ball_coordinates"]["x"] = ball_x
            moment["ball_coordinates"]["y"] = ball_y

            for player_coord in moment["player_coordinates"]:
                x = player_coord["y"]
                y = COURT_LENGTH - player_coord["x"]
                player_coord["x"] = x
                player_coord["y"] = y
                del player_coord["z"]

    elif direction == "right":
        for moment in rotated["moments"]:
            ball_x = COURT_WIDTH - moment["ball_coordinates"]["y"]
            ball_y = moment["ball_coordinates"]["x"]
            moment["ball_coordinates"]["x"] = ball_x
            moment["ball_coordinates"]["y"] = ball_y

            for player_coord in moment["player_coordinates"]:
                x = COURT_WIDTH - player_coord["y"]
                y = player_coord["x"]
                player_coord["x"] = x
                player_coord["y"] = y
                del player_coord["z"]

    return rotated


def trim_to_halfcourt(event: dict[str, t.Any]) -> dict[str, t.Any]:
    """
    Trims the beginning of an event to start when the ball crosses half-court
    """
    trimmed = event.copy()
    half_court_y = COURT_LENGTH / 2

    # Find the first moment where ball crosses half-court or dribbling begins
    start_index = 0
    end_index = len(trimmed["moments"]) - 1
    for i, moment in enumerate(trimmed["moments"]):
        ball_y = moment["ball_coordinates"]["y"]

        if ball_y >= half_court_y and not start_index:
            start_index = i
        if ball_y < half_court_y and i > start_index:
            end_index = i - 1
            break

    trimmed["moments"] = trimmed["moments"][start_index:end_index]
    return trimmed


def detect_unreasonable_trajectories(event: dict[str, t.Any]) -> dict[str, list]:
    """
    ! Currently unused function !
    Analyzes an event to detect unreasonable trajectory sequences.

    Args:
        event: The event dictionary with moments

    Returns:
        Dictionary of detected issues with timestamps
    """
    issues = {
        "impossible_speed": [],
        "teleportation": [],
        "stuck_coordinates": [],
        "ball_physics": [],
    }

    moments = event["moments"]
    if len(moments) < 2:
        return issues

    MAX_SPEED_FT_PER_FRAME = 1.2  # ~30 ft/s at 25Hz
    TELEPORT_THRESHOLD = 10.0  # ft

    prev_ball_pos = None
    prev_player_pos = {}

    # Track multiple frames for jitter detection
    player_positions = {}  # player_id -> list of recent positions
    ball_heights = []

    for moment_idx, moment in enumerate(moments):
        ball_pos = (moment["ball_coordinates"]["x"], moment["ball_coordinates"]["y"])

        if "z" in moment["ball_coordinates"]:
            ball_height = moment["ball_coordinates"]["z"]
            ball_heights.append(ball_height)

            # Check for ball physics - maintain constant height
            if len(ball_heights) > 10:
                height_variance = sum(
                    (h - sum(ball_heights) / len(ball_heights)) ** 2 for h in ball_heights
                ) / len(ball_heights)
                if height_variance < 0.01 and ball_height > 1.0:  # Ball hovering
                    issues["ball_physics"].append(moment_idx)
                ball_heights.pop(0)

        # Check ball speed/teleportation
        if prev_ball_pos:
            distance = (
                (ball_pos[0] - prev_ball_pos[0]) ** 2 + (ball_pos[1] - prev_ball_pos[1]) ** 2
            ) ** 0.5

            if distance > TELEPORT_THRESHOLD:
                issues["teleportation"].append(moment_idx)

        prev_ball_pos = ball_pos

        for player in moment["player_coordinates"]:
            player_id = player["playerid"]
            player_pos = (player["x"], player["y"])

            # Initialize tracking for this player if new
            if player_id not in player_positions:
                player_positions[player_id] = []

            # Add current position to history
            player_positions[player_id].append(player_pos)
            if len(player_positions[player_id]) > 10:
                player_positions[player_id].pop(0)

            # Check player speed/teleportation
            if player_id in prev_player_pos:
                distance = (
                    (player_pos[0] - prev_player_pos[player_id][0]) ** 2
                    + (player_pos[1] - prev_player_pos[player_id][1]) ** 2
                ) ** 0.5

                if distance > MAX_SPEED_FT_PER_FRAME:
                    issues["impossible_speed"].append(moment_idx)
                elif distance > TELEPORT_THRESHOLD:
                    issues["teleportation"].append(moment_idx)
                elif distance < 0.01:  # Barely moving
                    # Check if stuck in place for multiple frames
                    if len(player_positions[player_id]) > 5:
                        avg_movement = sum(
                            (
                                (
                                    player_positions[player_id][i][0]
                                    - player_positions[player_id][i - 1][0]
                                )
                                ** 2
                                + (
                                    player_positions[player_id][i][1]
                                    - player_positions[player_id][i - 1][1]
                                )
                                ** 2
                            )
                            ** 0.5
                            for i in range(1, len(player_positions[player_id]))
                        ) / (len(player_positions[player_id]) - 1)

                        if avg_movement < 0.05:  # Very little movement over multiple frames
                            issues["stuck_coordinates"].append(moment_idx)

            prev_player_pos[player_id] = player_pos

    # Remove duplicates and sort
    for issue_type in issues:
        issues[issue_type] = sorted(list(set(issues[issue_type])))

    return issues


def filter_unreasonable_plays(event: dict[str, t.Any], threshold: int = 10) -> bool:
    """
    ! Currently unused function !
    Determines if a play should be filtered out due to too many trajectory issues.

    Args:
        event: The event dictionary
        threshold: Maximum number of issues allowed before filtering

    Returns:
        True if play should be kept, False if it should be filtered out
    """
    issues = detect_unreasonable_trajectories(event)

    total_issues = sum(len(timestamps) for timestamps in issues.values())
    return total_issues <= threshold


def find_direction_possession_team(game_id: str, game_df: pl.DataFrame, shots_df: pd.DataFrame):
    """
    Modifies the events to be worked with in a uniform format by rotating the coordinates depending
    on the direction of play.
    """
    first_poss_team_id = None
    first_direction = None
    for event in game_df.iter_rows(named=True):
        moments = event["moments"]
        poss_team_id = event["event_info"]["possession_team_id"]
        event_id = event["event_info"]["id"]

        quarter = moments[0]["quarter"]
        if quarter != 1:
            continue

        shot_data = shots_df[
            (shots_df["GAME_EVENT_ID"] == int(event_id)) & (shots_df["GAME_ID"] == int(game_id))
        ]
        if not shot_data.empty:
            shot_time = shot_data["SHOT_TIME"].iloc[0]
            for moment in moments:
                if shot_time is None or moment["game_clock"] is None:
                    continue
                if shot_time == moment["game_clock"]:
                    if left_court_offense(moment, poss_team_id):
                        first_poss_team_id = poss_team_id
                        first_direction = "left"
                    elif right_court_offense(moment, poss_team_id):
                        first_poss_team_id = poss_team_id
                        first_direction = "right"
                    break
        if first_poss_team_id is not None:
            break

        # Fall back to heuristic method if no shot data is found
        resets = []
        # Find all shot clock resets (where next shot clock > current)
        for i in range(len(moments) - 1):
            shot_clock = moments[i]["shot_clock"]
            next_shot_clock = moments[i + 1]["shot_clock"]
            if shot_clock and next_shot_clock and next_shot_clock > shot_clock + 3:
                resets.append(i)
        # Add the ending as a reset point
        resets.append(len(moments) - 1)
        directions = []
        # Analyze each possession segment separately
        start_idx = 0
        for reset_idx in resets:
            segment = moments[start_idx : reset_idx + 1]

            # Get position at the middle of this segment for stability
            mid_idx = len(segment) // 2
            mid_moment = segment[mid_idx]

            if left_court_offense(mid_moment, poss_team_id):
                directions.append("left")
            elif right_court_offense(mid_moment, poss_team_id):
                directions.append("right")

            start_idx = reset_idx + 1

        if len(directions) == 0:
            continue
        # Use most common direction
        directions, counts = np.unique(directions, return_counts=True)
        ind = np.argmax(counts)
        first_direction = directions[ind]
        first_poss_team_id = poss_team_id
        break
    return (first_poss_team_id, first_direction)


def process_events(game_id: str, game_df: pl.DataFrame, shots_df: pd.DataFrame):
    # First pass to identify first possession team and directions
    first_poss_team_id, first_direction = find_direction_possession_team(game_id, game_df, shots_df)
    if first_poss_team_id is None:
        return game_df

    processed_events = []
    for event in game_df.iter_rows(named=True):
        # if not filter_unreasonable_plays(event):
        #     print(f"Filtered out event {event['event_info']['id']} due to unreasonable trajectory")

        set_direction(first_poss_team_id, first_direction, event)
        if event["event_info"]["direction"] not in ["left", "right"]:
            # Skip events with unknown direction
            continue

        # Now rotate coordinates
        rotated_event = rotate_coordinates(event)
        # Trim to offensive plays
        trimmed_event = trim_to_halfcourt(rotated_event)
        if len(trimmed_event["moments"]) < 50:
            continue

        processed_events.append(trimmed_event)

    return pl.DataFrame(processed_events, schema=game_df.schema)


def set_direction(first_poss_team_id, first_direction, event):
    quarter = event["moments"][0]["quarter"]
    direction = "unknown"
    second_direction = "right" if first_direction == "left" else "left"
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


def build_player_index_map(dataset: Dataset) -> dict[int, int]:
    """
    Build a mapping from player_id to a consistent player_idx across the dataset.
    This ensures the same player always has the same position in feature vectors.

    Args:
        df: Dataset with the game data

    Returns:
        Dictionary mapping player_id to player_idx
    """

    def extract_with_expressions(df: pl.DataFrame) -> dict:
        return {
            "player_ids": (
                (
                    pl.concat(
                        [
                            df.select(pl.col("home").struct.field("players")),
                            df.select(pl.col("visitor").struct.field("players")),
                        ]
                    )
                    .explode("players")
                    .select(pl.col("players").struct.field("playerid"))
                )
                .to_series()
                .unique()
                .to_list()
            )
        }

    results = dataset.map(
        extract_with_expressions,
        batched=True,
        batch_size=4000,
        desc="Build player_index_map",
        num_proc=NUM_PROCESSES,
    )
    # Create a mapping from player_id to index

    results = results.sort("player_ids").unique("player_ids")
    return {player_id: idx for idx, player_id in enumerate(results)}


def create_feature_vectors(game_id: str, game_df: pl.DataFrame, playerid_to_idx: dict[int, int]):
    """
    Creates feature vector for a single game in the DataFrame.

    Each feature vector includes:
      0:11 in feature Vector:
        - Game time
        - Game clock
        - Shot clock
        - Quarter
        - Score difference
        - Ball coordinates (x, y, z)
        - Ball velocity (dx, dy, dz)
      11:21 in feature Vector:
        - Player indices for each player on the court
      21:31 in feature Vector:
        - Player coordinates x for each player on the court
      31:41 in feature Vector:
        - Player coordinates y for each player on the court
      41:51 in feature Vector:
        - Team IDs for each player on the court
      51:61 in feature Vector:
        - Player velocities dx for each player on the court
      61:71 in feature Vector:
        - Player velocities dy for each player on the court

    Args:
        df: Polars DataFrame with one game data schema
        playerid_to_idx: player ID to index mapping.

    Returns:
        List of feature vectors for consecutive moments
    """
    play_features = []
    play_identifiers = []
    prev_moment = None
    prev_time = -1
    current_play_id = None
    current_play = []

    # print(game_df.head())

    moments_df = (
        game_df.select(
            pl.col("moments"),
            pl.col("event_info").struct.field("id").alias("event_id"),
            pl.col("event_info").struct.field("score_margin").alias("score_margin"),
        )
        .explode("moments")
        .rename({"moments": "moment"})
    )

    for row in moments_df.iter_rows(named=True):
        current = row["moment"]
        quarter = current["quarter"]
        game_clock = current["game_clock"]
        shot_clock = current["shot_clock"]
        game_time = get_game_time(game_clock, quarter)

        # Skip if time isn't advancing (e.g., clock stoppage or duplicate)
        if game_time <= prev_time:
            continue

        player_x_velocities = np.zeros(len(playerid_to_idx))
        player_y_velocities = np.zeros(len(playerid_to_idx))

        ball_x_velocity, ball_y_velocity, ball_z_velocity = 0.0, 0.0, 0.0

        if prev_moment is not None:
            time_delta = prev_moment["game_clock"] - current["game_clock"]
            if time_delta <= 0:
                # Skip invalid time deltas
                prev_moment = current
                prev_time = game_time
                continue

            # Calculate ball velocity (3D)
            cb = current["ball_coordinates"]
            pb = prev_moment["ball_coordinates"]

            ball_x_velocity = (cb["x"] - pb["x"]) / time_delta
            ball_y_velocity = (cb["y"] - pb["y"]) / time_delta
            ball_z_velocity = (cb["z"] - pb["z"]) / time_delta

            # Map current players with previous players
            cur_players = {(p["teamid"], p["playerid"]): p for p in current["player_coordinates"]}
            prev_players = {
                (p["teamid"], p["playerid"]): p for p in prev_moment["player_coordinates"]
            }

            # Calculate velocities for all players present in both moments
            for key, cp in cur_players.items():
                if key in prev_players:
                    pp = prev_players[key]

                    if key[1] in playerid_to_idx:
                        idx = playerid_to_idx[key[1]]
                        player_x_velocities[idx] = (cp["x"] - pp["x"]) / time_delta
                        player_y_velocities[idx] = (cp["y"] - pp["y"]) / time_delta

        score_margin = row["score_margin"] if row["score_margin"] != "TIE" else 0
        ball = current["ball_coordinates"]
        feature_vector = [
            game_time,  # Continuous game time
            game_clock,
            shot_clock,
            quarter,
            score_margin,
            ball["x"],
            ball["y"],
            ball["z"],
            ball_x_velocity,
            ball_y_velocity,
            ball_z_velocity,
        ]

        player_ids = {
            player["playerid"]
            for player in current["player_coordinates"]
            if player["playerid"] in playerid_to_idx
        }
        if len(player_ids) != 10:
            continue

        player_indices = []
        player_xs = []
        player_ys = []
        team_ids = []
        for player in current["player_coordinates"]:
            player_id = player["playerid"]

            if player_id not in playerid_to_idx:
                continue

            idx = playerid_to_idx[player_id]
            player_indices.append(idx)
            player_xs.append(player["x"])
            player_ys.append(player["y"])
            team_ids.append(player["teamid"])

        order = np.argsort(player_indices)
        # Add player indices (who is on court)
        for i in order:
            feature_vector.append(player_indices[i])
        for i in order:
            feature_vector.append(player_xs[i])
        for i in order:
            feature_vector.append(player_ys[i])
        for i in order:
            feature_vector.append(team_ids[i])
        for i in order:
            feature_vector.append(player_x_velocities[player_indices[i]])
        for i in order:
            feature_vector.append(player_y_velocities[player_indices[i]])

        if len(feature_vector) != 71:
            raise ValueError(
                f"Feature vector length mismatch: {len(feature_vector)} != 71 for game {game_id}, event {row['event_id']}"
            )
        event_id = row["event_id"]
        if current_play_id is not None and event_id != current_play_id:
            if current_play:
                play_features.append(np.stack(current_play))
                play_identifiers.append((game_id, current_play_id))
            current_play = []

        current_play.append(np.array(feature_vector, dtype=np.float32))
        current_play_id = event_id
        prev_moment = current
        prev_time = game_time

    if current_play:
        play_features.append(np.stack(current_play))
        play_identifiers.append((game_id, current_play_id))

    return play_features, play_identifiers


def process_game(
    game_id: str, game: Dataset, playerid_to_idx: dict[int, int], shots_df: pd.DataFrame
) -> tuple[list[t.Any], list[t.Any]]:
    """
    Process a single game: rotate court and save feature arrays.
    """
    game_df = game.to_polars()
    game_df = process_events(game_id, game_df, shots_df)
    return create_feature_vectors(game_id, game_df, playerid_to_idx)


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


def normalize_play(
    play: np.ndarray, target_len: int = 50, downsample_factor: int = 4
) -> None | list[np.ndarray]:
    """
    Downsamples and normalizes a single play to fixed-length sequences.

    - Discards plays that are too short after downsampling (< 30 frames)
    - Pads shorter plays to target_len
    - Splits longer plays into non-overlapping chunks of target_len

    Args:
        play: np.ndarray of shape (T, 71)
        target_len: Desired fixed length (e.g., 50)
        downsample_factor: Factor to reduce frame rate (e.g., 5 for 25Hz → 5Hz)

    Returns:
        A list of plays, each with shape (target_len, 71), or None
    """
    play = play[::downsample_factor]
    T = play.shape[0]
    if T < 30:
        return None  # Too short after downsampling
    if target_len > T:
        # Pad with last frame
        pad_len = target_len - T
        padding = np.repeat(play[-1:, :], pad_len, axis=0)
        return [np.concatenate([play, padding], axis=0)]
    if target_len == T:
        return [play]
    # Too long → split into chunks of target_len
    overlap = target_len // 4  # 25% overlap
    if overlap == 0:
        overlap = 1
    return [play[i : i + target_len] for i in range(0, T - target_len + 1, overlap)]


def save_batched_features(
    dataset: Dataset,
    playerid_to_idx: dict[int, int],
    output_dir: Path | str,
    batch_size: int = NUM_PROCESSES * 10,
):
    os.makedirs(output_dir, exist_ok=True)

    game_ids = dataset.unique("gameid")
    total_plays = 0

    game_ranges = get_games_ranges(dataset)
    game_datasets = {
        game_id: dataset.select(range(start, end)) for game_id, (start, end) in game_ranges.items()
    }

    shots_df = pd.read_csv(os.path.join(TRACKING_DIR, "shots_fixed.csv"))

    for batch_idx, i in enumerate(range(0, len(game_ids), batch_size)):
        batch_game_ids = game_ids[i : i + batch_size]

        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.starmap(
                process_game,
                [
                    (game_id, game_datasets[game_id], playerid_to_idx, shots_df)
                    for game_id in batch_game_ids
                ],
            )

        # Flatten the list of results
        batch_plays = []
        batch_ids = []
        for game_plays, play_ids in results:
            if game_plays:
                batch_plays.extend(game_plays)  # plays: List[np.ndarray]
                batch_ids.extend(play_ids)

        if not batch_plays:
            print(f"Batch {batch_idx} has no valid plays.")
            continue

        normalized_plays = []
        normalized_ids = []
        for play, pid in zip(batch_plays, batch_ids, strict=False):
            normed_plays = normalize_play(play, target_len=50, downsample_factor=5)
            if normed_plays is None:
                continue
            normalized_plays.extend(normed_plays)
            normalized_ids.extend([pid] * len(normed_plays))

        # Optionally pad or trim all plays to shape (50, 71) here
        # For now, assume they are already shaped (50, 71)
        try:
            batch_array = np.stack(normalized_plays)
        except ValueError as e:
            print(f"Error stacking batch {batch_idx}: {e}")
            continue

        batch_ids_array = np.array(normalized_ids, dtype=object)

        batch_path = os.path.join(output_dir, f"plays_batch_{batch_idx}.npz")
        np.savez_compressed(batch_path, X=batch_array, ids=batch_ids_array)
        print(f"Saved batch {batch_idx} with {len(batch_plays)} fixed length play sequences")

        total_plays += len(batch_plays)

    print(f"Total fixed length play sequences processed and saved: {total_plays}")


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

    playerid_to_idx = build_player_index_map(dataset)

    # Save the player ID mapping
    Path(GAMES_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{GAMES_DIR}/playerid_to_idx.pydict", "wb") as f:
        pickle.dump(playerid_to_idx, f)
        print(f"Saved player ID mapping with {len(playerid_to_idx)} players")

    save_batched_features(dataset, playerid_to_idx, GAMES_DIR)
