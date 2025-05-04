import argparse
import os
import pickle
import typing as t
from pathlib import Path

import numpy as np
import polars as pl
from datasets import load_dataset

from backend.settings import COURT_LENGTH, COURT_WIDTH, EVENTS_DIR, GAMES_DIR

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
        "./scripts/load_nba_tracking_data_15_16.py",
        name=name,
        split=split,
        num_proc=NUM_PROCESSES,
    )


def get_game_time(game_clock: float, quarter: int) -> float:
    """
    Convert game clock and quarter to continuous game time in seconds.

    Args:
        game_clock: Seconds remaining in the quarter
        quarter: Quarter number (1-4 for regulation, 5+ for overtime)

    Returns:
        Total seconds elapsed since the start of the game
    """
    if quarter <= 4:
        return (quarter - 1) * 720 + (720 - game_clock)
    else:
        return 4 * 720 + (quarter - 5) * 300 + (300 - game_clock)


def left_basket(moment):
    """
    This function takes a moment in the game and returns if the ball is in the left basket.
    """
    return (3.5 <= moment["ball_coordinates"]["x"] <= 6) and (
        24 <= moment["ball_coordinates"]["y"] <= 26
    )


def right_basket(moment):
    """
    This function takes a moment in the game and returns if the ball is in the right basket.
    """
    return (88 <= moment["ball_coordinates"]["x"] <= 90.5) and (
        24 <= moment["ball_coordinates"]["y"] <= 26
    )


def rotate_coordinates(event: dict[str, t.Any]) -> dict[str, t.Any]:
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
        quarter = event["moments"][0]["quarter"]

        if quarter != 1:
            continue
        for moment in event["moments"]:
            moment = event["moments"][0]
            if left_basket(moment):
                first_poss_team_id = event["event_info"]["possession_team_id"]
                first_direction = "left"
                second_direction = "right"
                if game_id in ["0021500292", "0021500648"]:  # TODO check these special cases
                    first_direction = "right"
                    second_direction = "left"
                break
            elif right_basket(moment):
                first_poss_team_id = event["event_info"]["possession_team_id"]
                first_direction = "right"
                second_direction = "left"
                if game_id == "0021500648":
                    first_direction = "left"
                    second_direction = "right"
                break
        if first_poss_team_id is not None:
            break
    if first_poss_team_id is None:
        return game_df

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

    return pl.DataFrame(processed_events, schema=game_df.schema)


def build_player_index_map(df: pl.DataFrame) -> dict[int, int]:
    """
    Build a mapping from player_id to a consistent player_idx across the dataset.
    This ensures the same player always has the same position in feature vectors.

    Args:
        df: Polars DataFrame with the game data

    Returns:
        Dictionary mapping player_id to player_idx
    """
    player_ids = set()

    # Extract all player IDs from both home and visitor teams
    for game in df.iter_rows(named=True):
        # Get player IDs from home team
        for player in game["home"]["players"]:
            player_ids.add(player["playerid"])

        # Get player IDs from visitor team
        for player in game["visitor"]["players"]:
            player_ids.add(player["playerid"])

    # Create a mapping from player_id to index
    return {player_id: idx for idx, player_id in enumerate(sorted(player_ids))}


def add_score_fields(game_id: str, game_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds 'score' and 'score_diff' fields to the game DataFrame.

    Args:
        game_id: The ID of the game.
        game_df: Polars DataFrame containing the game events.

    Returns:
        Polars DataFrame with added 'score' and 'score_diff' fields.
    """
    events_df = pl.read_csv(
        f"{EVENTS_DIR}/{game_id}.csv", schema_overrides={"SCOREMARGIN": pl.Utf8}
    )

    # Ensure SCOREMARGIN is treated as a string for comparison
    events_df = events_df.with_columns(
        [
            pl.col("SCOREMARGIN")
            .fill_null(strategy="forward")
            .fill_null("0")  # beginning of game
            .alias("score_diff_raw"),
            pl.col("EVENTNUM").cast(pl.Utf8).alias("EVENTNUM"),
            pl.col("SCORE").fill_null(strategy="forward").fill_null("0 - 0").alias("score"),
        ]
    )

    # Replace "TIE" with "0", then cast to Int64
    events_df = events_df.with_columns(
        pl.when(pl.col("score_diff_raw") == "TIE")
        .then(pl.lit("0"))
        .otherwise(pl.col("score_diff_raw"))
        .cast(pl.Int64)
        .alias("score_diff")
    )

    game_df = game_df.join(
        events_df.select(["EVENTNUM", "score", "score_diff"]),
        left_on=pl.col("event_info").struct.field("id"),
        right_on="EVENTNUM",
        how="left",
    ).drop("EVENTNUM")

    return game_df


# def get_event_stream(gameid: str, game_df: pl.DataFrame):
#     df_events = pd.read_csv(f"{EVENTS_DIR}/{gameid}.csv")
#     df_events = df_events.fillna("")

#     # Posession times.
#     # EVENTMSGTYPE descriptions can be found at: https://github.com/rd11490/NBA_Tutorials/tree/master/analyze_play_by_play.
#     event = None
#     score = "0 - 0"
#     score_diff = 0
#     # pos_team is the team that had possession prior to the event.
#     (pos_team, pos_team_idx) = (df_events["home_team_id"], 0)
#     jump_ball_team_idx = None
#     # I think most of these are technical fouls.
#     skip_fouls = {10, 11, 16, 19}

#     events = set()
#     pos_stream = []
#     event_stream = []
#     for row_idx, row in game_df.iter_rows(named=True):
#         period = row["period"]
#         game_clock = row["PCTIMESTRING"].split(":")
#         game_clock_secs = 60 * int(game_clock[0]) + int(game_clock[1])
#         game_time = get_game_time(game_clock_secs, period)

#         description = row[description_col].lower().strip()

#         eventmsgtype = row[event_col]

#         # Don't know.
#         if eventmsgtype == 18:
#             continue

#         # Blank line.
#         elif eventmsgtype == 14:
#             continue

#         # End of a period.
#         elif eventmsgtype == 13:
#             if period == 4:
#                 jump_ball_team_idx = None

#             continue

#         # Start of a period.
#         elif eventmsgtype == 12:
#             if 2 <= period <= 4:
#                 if period == 4:
#                     pos_team_idx = jump_ball_team_idx
#                 else:
#                     pos_team_idx = (jump_ball_team_idx + 1) % 2

#             elif 6 <= period:
#                 pos_team_idx = (jump_ball_team_idx + (period - 5)) % 2

#             continue

#         # Ejection.
#         elif eventmsgtype == 11:
#             continue

#         # Jump ball.
#         elif eventmsgtype == 10:
#             pos_team = row["PLAYER3_TEAM_ABBREVIATION"]

#             if pos_team == teams[0]:
#                 pos_team_idx = 0
#             else:
#                 pos_team_idx = 1

#             if (period in {1, 5}) and (jump_ball_team_idx is None):
#                 jump_ball_team_idx = pos_team_idx

#             continue

#         # Timeout.
#         elif eventmsgtype == 9:
#             # TV timeout?
#             if description == "":
#                 continue

#             event = "timeout"

#         # Substitution.
#         elif eventmsgtype == 8:
#             continue

#         # Violation.
#         elif eventmsgtype == 7:
#             # With 35 seconds left in the fourth period, there was a kicked ball
#             # violation attributed to Wayne Ellington of the Brooklyn Nets, but the
#             # following event is a shot by the Nets, which means possession never changed.
#             if (gameid == "0021500414") and (row_idx == 427):
#                 continue

#             # Goaltending is considered a made shot, so the following event is always the
#             # made shot event.
#             if "goaltending" in description:
#                 score = row["SCORE"] if row["SCORE"] else score
#                 continue
#             # Jump ball violations have weird possession rules.
#             elif "jump ball" in description:
#                 if row["PLAYER1_TEAM_ABBREVIATION"] == teams[0]:
#                     pos_team_idx = 1
#                 else:
#                     pos_team_idx = 0

#                 pos_team = teams[pos_team_idx]
#                 if (period == 1) and (game_time == 0):
#                     jump_ball_team_idx = pos_team_idx

#                 continue

#             else:
#                 if row[player1_team_col] == teams[pos_team_idx]:
#                     event = "violation_offense"
#                     pos_team_idx = (pos_team_idx + 1) % 2
#                 else:
#                     event = "violation_defense"

#         # Foul.
#         elif eventmsgtype == 6:
#             # Skip weird fouls.
#             if row["EVENTMSGACTIONTYPE"] in skip_fouls:
#                 score = row["SCORE"] if row["SCORE"] else score
#                 continue
#             else:
#                 if row[player1_team_col] == teams[pos_team_idx]:
#                     event = "offensive_foul"
#                     pos_team_idx = (pos_team_idx + 1) % 2
#                 else:
#                     event = "defensive_foul"

#         # Turnover.
#         elif eventmsgtype == 5:
#             if "steal" in description:
#                 event = "steal"
#             elif "goaltending" in description:
#                 event = "goaltending_offense"
#             elif (
#                 ("violation" in description)
#                 or ("dribble" in description)
#                 or ("traveling" in description)
#             ):
#                 event = "violation_offense"
#             else:
#                 event = "turnover"

#             # Team turnover.
#             if row[player1_team_col] == "":
#                 team_id = int(row["PLAYER1_ID"])
#                 team_abb = TEAM_ID2PROPS[team_id]["abbreviation"]
#             else:
#                 team_abb = row[player1_team_col]

#             pos_team_idx = 1 if team_abb == teams[0] else 0

#         # Rebound.
#         elif eventmsgtype == 4:
#             # With 17 seconds left in the first period, Spencer Hawes missed a tip in,
#             # which was rebounded by DeAndre Jordan. The tip in is recorded as a rebound
#             # and a missed shot for Hawes. All three events have the same timestamp,
#             # which seems to have caused the order of the events to be slightly shuffled
#             # with the Jordan rebound occurring before the tip in.
#             if (gameid == "0021500550") and (row_idx == 97):
#                 continue

#             # Team rebound.
#             if row[player1_team_col] == "":
#                 team_id = int(row["PLAYER1_ID"])
#                 team_abb = row["PLAYER1_TEAM_ABBREVIATION"]
#                 if team_abb == teams[pos_team_idx]:
#                     event = "rebound_offense"
#                 else:
#                     event = "rebound_defense"
#                     pos_team_idx = (pos_team_idx + 1) % 2

#             elif row[player1_team_col] == teams[pos_team_idx]:
#                 event = "rebound_offense"
#             else:
#                 event = "rebound_defense"
#                 pos_team_idx = (pos_team_idx + 1) % 2

#         # Free throw.
#         elif eventmsgtype == 3:
#             # See rules for technical fouls: https://official.nba.com/rule-no-12-fouls-and-penalties/.
#             # Possession only changes for too mt.Any players, which is extremely rare.
#             if "technical" not in description:
#                 pos_team_idx = 0 if row[player1_team_col] == teams[0] else 1
#                 if (
#                     ("Clear Path" not in row[description_col])
#                     and ("Flagrant" not in row[description_col])
#                     and ("MISS" not in row[description_col])
#                     and (
#                         ("1 of 1" in description)
#                         or ("2 of 2" in description)
#                         or ("3 of 3" in description)
#                     )
#                 ):
#                     # Hack to handle foul shots for away from play fouls.
#                     if ((gameid == "0021500274") and (row_idx == 519)) or (
#                         (gameid == "0021500572") and (row_idx == 428)
#                     ):
#                         pass
#                     # This event is a made foul shot by Thaddeus Young of the Brooklyn
#                     # Nets following an and-one foul, so possession should have changed
#                     # to the Milwaukee Bucks. However, the next event is a made shot by
#                     # Brook Lopez (also of the Brooklyn Nets) with no event indicating a
#                     # change of possession occurring before it.
#                     elif (gameid == "0021500047") and (row_idx == 64):
#                         pass
#                     else:
#                         pos_team_idx = (pos_team_idx + 1) % 2

#                 pos_team = teams[pos_team_idx]

#             score = row["SCORE"] if row["SCORE"] else score

#             continue

#         # Missed shot.
#         elif eventmsgtype == 2:
#             if "dunk" in description:
#                 shot_type = "dunk"
#             elif "layup" in description:
#                 shot_type = "layup"
#             else:
#                 shot_type = "shot"

#             if "BLOCK" in row[description_col]:
#                 miss_type = "block"
#             else:
#                 miss_type = "miss"

#             event = f"{shot_type}_{miss_type}"

#             if row[player1_team_col] != teams[pos_team_idx]:
#                 print(pos_stream[-5:])
#                 raise ValueError(f"Incorrect possession team in row {str(row_idx)}.")

#         # Made shot.
#         elif eventmsgtype == 1:
#             if "dunk" in description:
#                 shot_type = "dunk"
#             elif "layup" in description:
#                 shot_type = "layup"
#             else:
#                 shot_type = "shot"

#             event = f"{shot_type}_made"

#             if row[player1_team_col] != teams[pos_team_idx]:
#                 print(pos_stream[-5:])
#                 raise ValueError(f"Incorrect possession team in row {str(row_idx)}.")

#             pos_team_idx = (pos_team_idx + 1) % 2

#         events.add(event)
#         pos_stream.append(pos_team_idx)

#         if row[player1_team_col] == "":
#             team_id = int(row["PLAYER1_ID"])
#             event_team = row["PLAYER1_TEAM_ABBREVIATION"]
#         else:
#             event_team = row[player1_team_col]

#         event_stream.append(
#             {
#                 "game_time": game_time - 1,
#                 "pos_team": pos_team,
#                 "event": event,
#                 "description": description,
#                 "event_team": event_team,
#                 "score": score,
#                 "score_diff": score_diff,
#             }
#         )

#         # With 17 seconds left in the first period, Spencer Hawes missed a tip in,
#         # which was rebounded by DeAndre Jordan. The tip in is recorded as a rebound
#         # and a missed shot for Hawes. All three events have the same timestamp,
#         # which seems to have caused the order of the events to be slightly shuffled
#         # with the Jordan rebound occurring before the tip in.
#         if (gameid == "0021500550") and (row_idx == 98):
#             event_stream.append(
#                 {
#                     "game_time": game_time - 1,
#                     "pos_team": pos_team,
#                     "event": "rebound_defense",
#                     "description": "jordan rebound (off:2 def:5)",
#                     "event_team": "LAC",
#                     "score": score,
#                     "score_diff": score_diff,
#                 }
#             )
#             pos_team_idx = (pos_team_idx + 1) % 2

#         # This event is a missed shot by Kawhi Leonard of the San Antonio Spurs. The next
#         # event is a missed shot by Bradley Beal of the Washington Wizards with no
#         # event indicating a change of possession occurring before it.
#         if (gameid == "0021500061") and (row_idx == 240):
#             pos_team_idx = (pos_team_idx + 1) % 2

#         pos_team = teams[pos_team_idx]
#         score = row["SCORE"] if row["SCORE"] else score
#         score_diff = row["SCOREMARGIN"] if row["SCOREMARGIN"] else score_diff

#     return event_stream


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
    features = []
    play_identifiers = []  # Store (game_id, event_id) pairs
    prev_moment = None
    prev_time = -1
    game_over = False

    game_df = add_score_fields(game_id, game_df)
    # print(game_df.head())

    moments_df = (
        game_df.select(
            pl.col("moments"),
            pl.col("event_info").struct.field("id").alias("event_id"),
            pl.col("score_diff"),
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

        # # Moments can overlap temporally, so previously processed time points are
        # # skipped along with clock stoppages.
        # while game_time > event_stream[event_idx]["game_time"]:
        #     event_idx += 1
        #     if event_idx >= len(event_stream):
        #         game_over = True
        #         break

        # if game_over:
        #     break

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

        ball = current["ball_coordinates"]
        feature_vector = [
            game_time,  # Continuous game time
            game_clock,
            shot_clock,
            quarter,
            row["score_diff"],
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
            feature_vector.extend(
                [
                    player_indices[i],
                    player_xs[i],
                    player_ys[i],
                    team_ids[i],
                    player_x_velocities[player_indices[i]],
                    player_y_velocities[player_indices[i]],
                ]
            )

        if len(feature_vector) != 71:
            raise ValueError(
                f"Feature vector length mismatch: {len(feature_vector)} != 71 for game {game_id}, event {row['event_id']}"
            )

        features.append(np.array(feature_vector))
        play_identifiers.append((game_id, row["event_id"]))

        prev_moment = current
        prev_time = game_time

    return features, play_identifiers


def save_feature_arrays(
    df: pl.DataFrame, output_dir: Path | str, playerid_to_idx: dict[int, int] = None
) -> None:
    """
    Process games and save feature vectors as numpy arrays (one file per game).

    Args:
        df: Polars DataFrame with the game data schema
        output_dir: Directory to save the output numpy arrays
        playerid_to_idx: Optional player ID to index mapping. If None, a new one will be created.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build player index mapping if not provided
    if playerid_to_idx is None:
        playerid_to_idx = build_player_index_map(df)

    for game_id in df["gameid"].unique():
        print(f"Processing game {game_id}")
        game_df = df.filter(pl.col("gameid") == game_id)

        features, identifiers = create_feature_vectors(game_id, game_df, playerid_to_idx)
        if features:
            features_array = np.stack(features)

            print(f"Feature vector shape: {features_array.shape}")
            # Save the features and identifiers
            # Save the features as a numpy array
            np.save(os.path.join(output_dir, f"{game_id}_X.npy"), features_array)
            print(f"Saved {len(features)} feature vectors for game {game_id}")
            identifiers_array = np.array(identifiers, dtype=object)
            print(f"Identifiers shape: {identifiers_array.shape}")
            np.save(os.path.join(output_dir, f"{game_id}_ids.npy"), identifiers_array)
        else:
            print(f"No features generated for game {game_id}")
        break

    # Save the player ID mapping
    pickle.dump(playerid_to_idx, open(f"{output_dir}/playerid_to_idx.pydict", "wb"))
    print(f"Saved player ID mapping with {len(playerid_to_idx)} players")


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
    dataset = load_nba_dataset(split="train", name=args.name)

    df_plays = dataset.to_polars()
    df_filtered = df_plays.filter(
        (pl.col("moments").list.len() > 0)
        & (pl.col("event_info").struct.field("possession_team_id").is_not_null())
        & (~pl.col("event_info").struct.field("possession_team_id").is_nan())
        & (pl.col("moments").list.first().struct.field("player_coordinates").list.len() == 10)
        & pl.col("event_info").struct.field("type").is_in([1, 2, 5, 6])
    )

    # TODO make the following more scalable (memory efficient)
    rotated_df = df_filtered.group_by("gameid").map_groups(
        lambda df: rotate_court(df["gameid"][0], df)
    )
    print(rotated_df.schema)

    # TODO make the following more scalable (memory efficient)
    save_feature_arrays(rotated_df, GAMES_DIR)

    # TODO compute y soft-labels and store them in _y.parquet
