import py7zr
from datasets import load_dataset, load_dataset_builder
import os
import argparse

NUM_PROC = os.cpu_count()


def load_ds(subset='tiny'):
    # Looks like there is only a train split (loading without split yields a dictionary with only "train": data not split already)
    dataset = load_dataset("dcayton/nba_tracking_data_15_16",
                           name = subset,
                           split="train",
                           trust_remote_code=True)
    
    return dataset

def downsample(example):
    example['moments'] = example['moments'][::10]
    return example

def prune_example(example):
    return {
        # top‐level ID
        "n_gameid": example["gameid"],

        # only the fields we care about, renaming `id` → `eventid`
        "n_event_info": {
            "id": example["event_info"]["id"],
            "type": example["event_info"]["type"],
            "possession_team_id": example["event_info"]["possession_team_id"],
        },

        # only player_id from primary and secondary
        "n_primary_info": {
            "player_id": example["primary_info"]["player_id"]
        },
        "n_secondary_info": {
            "player_id": example["secondary_info"]["player_id"]
        },

        # visitor: keep only each player’s position
        "n_visitor": {
            "players": [
                {"position": p["position"]}
                for p in example["visitor"]["players"]
            ]
        },

        # home: keep only playerid and position
        "n_home": {
            "players": [
                {"playerid": p["playerid"], "position": p["position"]}
                for p in example["home"]["players"]
            ]
        },

        # downsampled moments: only ball_coords and each player’s id+xyz
        "n_moments": [
            {
                "ball_coordinates": {
                    "x": m["ball_coordinates"]["x"],
                    "y": m["ball_coordinates"]["y"],
                    "z": m["ball_coordinates"]["z"],
                },
                "player_coordinates": [
                    {
                        "playerid": pc["playerid"],
                        "x": pc["x"],
                        "y": pc["y"],
                        "z": pc["z"],
                    }
                    for pc in m["player_coordinates"]
                ],
            }
            for m in example["moments"]
        ],
    }


def process_data(dataset):
    dataset_dsed = dataset.map(downsample, num_proc=NUM_PROC)

    datased_prcsd  = dataset_dsed.map(prune_example,
                                      num_proc=NUM_PROC,
                                      remove_columns=dataset_dsed.column_names)

    # hacky fix for map only dropping columns after comforming the new ones to the old data format
    datased_prcsd.rename_columns({'n_gameid':'gameid',
                              'n_event_info':'event_info',
                              'n_primary_info':'primary_info',
                              'n_secondary_info':'secondary_info',
                              'n_visitor':'visitor',
                              'n_home':'home',
                              'n_moments':'moments'})
    
    return datased_prcsd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        choices=["tiny", "small", "medium", "full"],
        default="tiny",
        help="Data Subset (tiny: 5 games, small: 25 games, medium: 100 games, full: all games)",
        )
    
    args = parser.parse_args()

    dataset = load_ds(args.subset)
    dataset = process_data(dataset)

    # save or process further


