import glob
import os

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from backend.model.dataset_v2 import PlayDataset, collate_batch
from backend.model.model_v2 import PlayTransformer
from backend.model.utils import team_pooling
from backend.settings import COURT_WIDTH, DATA_DIR, EXPERIMENTS_DIR, GAMES_DIR

JOB = 20250509102003  # Manually update this
MAX_FEATURES_PER_FILE = 3000
OUTPUT_DIR = os.path.join(DATA_DIR, "embeddings")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_embeddings_to_file(embeddings, file_index):
    array = torch.cat(embeddings, dim=0).cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, f"embedding_{file_index}.npy"), array)


def main():
    cfg = yaml.safe_load(open(f"{EXPERIMENTS_DIR}/{JOB}/{JOB}.yaml"))
    # --- CONFIG ---
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device_ids"]

    # --- Load Model ---
    map_location = {f"cuda:{cfg['device_ids']}": DEVICE.type}
    model_opts = {
        "in_feat_dim": cfg["model"]["in_feature_dim"],
        "time_steps": cfg["model"]["time_steps"],
        "feature_dim": cfg["model"]["feature_dim"],
        "head_num": cfg["model"]["head_num"],
        "k": cfg["model"]["k"],
        "F": cfg["model"]["F"],
        "halfwidth": COURT_WIDTH,
        "lr": float(cfg["model"]["lr"]),
    }
    model = PlayTransformer.load_from_checkpoint(
        f"{EXPERIMENTS_DIR}/{JOB}/tmpu_rjniva", map_location=map_location, **model_opts
    )
    model.eval()
    model.freeze()
    # model.to(DEVICE)

    # --- Load Dataset ---
    dataset = PlayDataset(GAMES_DIR)
    dataloader_cfg = cfg["DataLoader"]
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    # Store embedding ids in same order
    paths = sorted(
        glob.glob(os.path.join(GAMES_DIR, "**/*.pkl"), recursive=True), key=os.path.getmtime
    )
    print(dataset.filepath_list == paths, paths[0:4], dataset.filepath_list[:4])
    df = pd.DataFrame({"Eventid": [fp.removeprefix(DATA_DIR) for fp in dataset.filepath_list]})
    df.to_csv(os.path.join(OUTPUT_DIR, "embedding_sources.csv"), index=False)

    # --- Embedding Extraction ---
    all_embeddings = []
    file_index = 0
    vector_count = 0

    for batch in tqdm.tqdm(dataloader):
        (
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_mask_batch,
            num_agents_accum,
            agent_ids_batch,
        ) = batch

        e = model.get_encoder_output(
            states_batch,
            agents_batch_mask,
            states_padding_mask_batch,
            states_hidden_mask_batch,
            agent_ids_batch,
        )  # Shape: [A*B, T, D]
        encoder_out: torch.Tensor = e["out"]
        AxB, T, D = encoder_out.shape
        A = model_opts["F"]
        B = AxB // A

        embedding_batch = encoder_out.reshape((B, A, T, D))
        embedding_batch = team_pooling(embedding_batch)  # [B, A, T, D] -> [B, D]
        # embedding = embedding.view(B, D)  # Flatten

        while embedding_batch.shape[0] > 0:
            chunk_size = min(embedding_batch.shape[0], MAX_FEATURES_PER_FILE - vector_count)
            chunk = embedding_batch[:chunk_size, :]
            all_embeddings.append(chunk)
            vector_count += chunk.shape[0]
            embedding_batch = embedding_batch[chunk_size:]

            if vector_count >= MAX_FEATURES_PER_FILE:
                save_embeddings_to_file(all_embeddings, file_index)
                file_index += 1
                vector_count = 0
                all_embeddings = []

    # Save remaining embeddings
    if all_embeddings:
        save_embeddings_to_file(all_embeddings, file_index)


if __name__ == "__main__":
    main()
