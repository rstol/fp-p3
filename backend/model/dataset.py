import typing as t

import numpy as np
from torch.utils.data import Dataset

# The raw data is recorded at 25 Hz.
RAW_DATA_HZ = 25


class Baller2PlayDataset(Dataset):
    def __init__(self, N, mode: t.Literal["train", "test", "valid"], game_ids):
        self.N = N
        self.mode = mode
        self.game_ids = game_ids

        # define the start and chunk size in the train loop such that the samples overlap and have
        # a fixed length which is maybe 100 moments to get 6 seconds in one sequence

    def __len__(self):
        return self.N

    def get_sample(self, X, start):
        pass
        # Downsample

        # End sequence early if there is a position glitch. Often happens when there was
        # a break in the game, but glitches also happen for other reasons. See
        # glitch_example.py for an example.

        # Move player trajectories computation to here from preprocessing?

    def __getitem__(self, idx):
        if self.mode == "train":
            game_id = np.random.choice(self.game_ids)

        elif self.mode in {"valid", "test"}:
            game_id = self.game_ids[idx]

        X = np.load(f"{GAMES_DIR}/{game_id}_X.npy")

        if self.mode == "train":
            start = np.random.randint(len(X) - self.chunk_size)

        elif self.mode in {"valid", "test"}:
            start = self.starts[idx]

        return self.get_sample(X, start)
