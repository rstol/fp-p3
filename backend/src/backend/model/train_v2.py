import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from backend.model.dataset_v2 import PlayDataset, collate_batch
from backend.model.model_v2 import PlayTransformer
from backend.settings import COURT_WIDTH, EXPERIMENTS_DIR, GAMES_DIR


def main(cfg):
    pl.seed_everything(2025)

    dataset = PlayDataset(GAMES_DIR)
    train_size = int(len(dataset) * 0.8)  # 80% training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloader_cfg = cfg["DataLoader"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataloader_cfg["batch_size"],
        shuffle=True,
        num_workers=dataloader_cfg["num_workers"],
        collate_fn=collate_batch,
        drop_last=True,
    )  # collate_fn from collect.py, A MP; B MR;C MP+MR
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=dataloader_cfg["batch_size"],
        shuffle=False,
        num_workers=dataloader_cfg["num_workers"],
        collate_fn=collate_batch,
    )
    print(train_dataloader.batch_size)

    # train process
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device_ids"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlayTransformer(
        cfg["model"]["in_feature_dim"],
        cfg["model"]["time_steps"],
        cfg["model"]["feature_dim"],
        cfg["model"]["head_num"],
        cfg["model"]["k"],
        cfg["model"]["F"],
        COURT_WIDTH,
        float(cfg["model"]["lr"]),
    ).to(device)  # input_dim,timestep,out_dim,headnum,k,F,halfwith,lr
    tb_logger = TensorBoardLogger("logs/", name=cfg["check_point_name"])

    trainer = pl.Trainer(max_epochs=cfg["max_epochs"], logger=tb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint(cfg["save_path"])


if __name__ == "__main__":
    JOB = sys.argv[1]

    cfg = yaml.safe_load(open(f"{EXPERIMENTS_DIR}/{JOB}/{JOB}.yaml"))
    print(cfg)
    sys.exit(main(cfg))
