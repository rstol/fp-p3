import typing as t

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.settings import GAMES_DIR, RAW_DATA_HZ


class Baller2PlayDataset(Dataset):
    def __init__(
        self,
        num_samples,
        mode: t.Literal["train", "test", "valid"],
        game_ids: list[str],
        n_player_ids: int,
        starts: list[int],
        hz: int = 5,
        secs: int = 6,
        max_player_move: float = 4.5,
    ):
        self.num_samples = num_samples
        self.mode = mode
        self.game_ids = game_ids
        self.downsample = int(RAW_DATA_HZ / hz)
        self.chunk_size = int(RAW_DATA_HZ * secs)
        self.seq_len = self.chunk_size // self.downsample
        self.n_player_ids = n_player_ids
        self.starts = starts
        self.n_players = 10

        self.max_player_move = max_player_move
        # define the start and chunk size in the train loop such that the samples overlap and have
        # a fixed length which is maybe X moments to get 6 seconds in one sequence

    def __len__(self):
        return self.num_samples

    def get_sample(self, X, start):
        # Downsample
        seq_data = X[start : start + self.chunk_size : self.downsample]
        keep_players = np.random.choice(np.arange(10), self.n_players, False)

        # Use preprocessed velocities for model input
        player_vxs = seq_data[:, 51:61][:, keep_players]
        player_vys = seq_data[:, 61:71][:, keep_players]
        # End sequence early if there is a position glitch. Often happens when there was
        # a break in the game, but glitches also happen for other reasons.
        try:
            glitch_x_break = np.where(np.abs(player_vxs) > 1.2 * self.max_player_move)[0].min()
        except ValueError:
            glitch_x_break = len(seq_data)
        try:
            glitch_y_break = np.where(np.abs(player_vys) > 1.2 * self.max_player_move)[0].min()
        except ValueError:
            glitch_y_break = len(seq_data)
        if glitch_x_break < len(seq_data) or glitch_y_break < len(seq_data):
            print(
                f"Glitch detected in game {self.game_ids[0]} at frame {start} for player {keep_players}"
            )
        seq_break = min(glitch_x_break, glitch_y_break)
        seq_data = seq_data[:seq_break]

        player_vxs = seq_data[:, 51:61][:, keep_players]
        player_vys = seq_data[:, 61:71][:, keep_players]
        player_xs = seq_data[:, 21:31][:, keep_players]
        player_ys = seq_data[:, 31:41][:, keep_players]
        player_idxs = seq_data[:, 11:21][:, keep_players].astype(int)

        return {
            "player_idxs": torch.LongTensor(player_idxs[: seq_break - 1]),
            "player_xs": torch.Tensor(player_xs[: seq_break - 1]),
            "player_ys": torch.Tensor(player_ys[: seq_break - 1]),
            "player_vxs": torch.Tensor(player_vxs[: seq_break - 1]),
            "player_vys": torch.Tensor(player_vys[: seq_break - 1]),
        }

    def __getitem__(self, idx):
        if self.mode == "train":
            game_id = np.random.choice(self.game_ids)
        elif self.mode in {"valid", "test"}:
            game_id = self.game_ids[idx]

        X = np.load(f"{GAMES_DIR}/{game_id}_X.npy")
        identifiers = np.load(
            f"{GAMES_DIR}/{game_id}_ids.npy", allow_pickle=True
        )  # Need allow_pickle for object arrays
        print(X.shape)
        print(identifiers.shape, identifiers[0])

        if self.mode == "train":
            start = np.random.randint(len(X) - self.chunk_size)
        elif self.mode in {"valid", "test"}:
            start = self.starts[idx]

        return self.get_sample(X, start)
