import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from backend.model.dataset import Baller2PlayDataset
from backend.model.model import Baller2Play
from backend.settings import EXPERIMENTS_DIR, GAMES_DIR

SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_printoptions(linewidth=160)


def worker_init_fn(worker_id):
    # Set the random seed for each worker
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def get_train_valid_test_gameids():
    with open("train_gameids.txt") as f:
        train_gameids = f.read().split()

    with open("valid_gameids.txt") as f:
        valid_gameids = f.read().split()

    with open("test_gameids.txt") as f:
        test_gameids = f.read().split()

    return (train_gameids, valid_gameids, test_gameids)


def init_basketball_datasets(opts):
    playerid_to_idx = pickle.load(open(f"{GAMES_DIR}/playerid_to_idx.pydict", "rb"))

    (train_gameids, valid_gameids, test_gameids) = get_train_valid_test_gameids()

    dataset_config = opts["dataset"]
    dataset_config["game_ids"] = train_gameids
    dataset_config["num_samples"] = opts["train"]["train_samples_per_epoch"]
    dataset_config["starts"] = []
    dataset_config["mode"] = "train"
    dataset_config["n_player_ids"] = len(playerid_to_idx)

    train_dataset = Baller2PlayDataset(**dataset_config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
        worker_init_fn=worker_init_fn,
    )

    num_samples = opts["train"]["valid_samples"]
    samps_per_gameid = int(np.ceil(num_samples / len(valid_gameids)))
    starts = []
    for game_id in valid_gameids:
        ids = np.load(f"{GAMES_DIR}/{game_id}_ids.npy", allow_pickle=True)
        max_start = len(ids) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["game_ids"] = np.repeat(valid_gameids, samps_per_gameid)
    dataset_config["num_samples"] = len(dataset_config["game_ids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "valid"
    valid_dataset = Baller2PlayDataset(**dataset_config)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    samps_per_gameid = int(np.ceil(num_samples / len(test_gameids)))
    starts = []
    for game_id in test_gameids:
        ids = np.load(f"{GAMES_DIR}/{game_id}_ids.npy", allow_pickle=True)
        max_start = len(ids) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["game_ids"] = np.repeat(test_gameids, samps_per_gameid)
    dataset_config["num_samples"] = len(dataset_config["game_ids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "test"
    test_dataset = Baller2PlayDataset(**dataset_config)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    return (
        train_loader,
        train_dataset,
        valid_loader,
        test_loader,
    )


def init_model(opts, train_dataset: Baller2PlayDataset):
    model_config = opts["model"]
    model_config["seq_len"] = train_dataset.seq_len - 1
    model_config["input_dim"] = train_dataset.n_players * 4 + 10
    # TODO: Initialize your model here based on the model_config and maybe on dataset_config?
    model = Baller2Play(**model_config)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    return model


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


def train_model(train_loader, valid_loader, model, device, opts):
    seq_len = model.seq_len
    # n_players = model.n_players
    print(f"seq_len: {seq_len}")

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = torch.optim.Adam(train_params, lr=opts["train"]["learning_rate"])

    best_valid_loss = float("inf")
    no_improvement = 0

    for epoch in range(opts["train"]["max_epochs"]):
        print(f"\nepoch: {epoch}", flush=True)

        model.train()
        total_train_loss = 0.0
        total_recon, total_kl = 0.0, 0.0
        n_train = 0

        start_time = time.time()
        # Train
        for train_idx, batch in enumerate(train_loader):
            if train_idx % 1000 == 0:
                print(train_idx, flush=True)
            # Skip bad sequences.
            if len(batch["player_idxs"]) < seq_len:
                print(f"Skipping bad sequence: {len(batch['player_idxs'])}")
                continue

            player_feats = torch.cat(
                [batch["player_xs"], batch["player_ys"], batch["player_vxs"], batch["player_vys"]],
                dim=-1,
            )  # [T, 40]

            x = torch.cat([batch["game_data"], player_feats], dim=-1).to(device)  # [T, 40 + 10]

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)

            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=opts["train"]["beta"])
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_recon += recon
            total_kl += kl
            n_train += 1

        print(
            f"Train loss: {total_train_loss / n_train:.4f}, Recon: {total_recon / n_train:.4f}, KL: {total_kl / n_train:.4f}"
        )

        # Validation
        model.eval()
        total_valid_loss = 0.0
        n_valid = 0
        with torch.no_grad():
            for batch in valid_loader:
                x = torch.cat([batch["player_xs"], batch["player_ys"]], dim=-1).to(device)
                x_hat, mu, logvar = model(x)
                loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta=opts["train"]["beta"])
                total_valid_loss += loss.item()
                n_valid += 1
        valid_loss = total_valid_loss / n_valid
        print(f"Validation loss: {valid_loss:.4f}")

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improvement = 0
            torch.save(model.state_dict(), f"{opts['save_path']}/model.pth")
        else:
            no_improvement += 1
            if no_improvement == opts["train"]["patience"]:
                print("Reducing LR due to no improvement.")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1
                no_improvement = 0

        # TODO gradually increase Beta-VAE during training
        epoch_time = time.time() - start_time
        print(f"epoch_time: {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    JOB = sys.argv[1]

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{EXPERIMENTS_DIR}/{JOB}/{JOB}.yaml"))

    (
        train_loader,
        train_dataset,
        valid_loader,
        test_loader,
    ) = init_basketball_datasets(opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = init_model(opts, train_dataset).to(device)

    train_model(train_loader, valid_loader, model, device, opts)
