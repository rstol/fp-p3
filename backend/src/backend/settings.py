import os

import polars as pl

API = "/api/v1/"

SAMPLING_RATE = 3  # RAW_DATA_HZ / SAMPLING_RATE ≈ 8.33 Hz

(COURT_WIDTH, COURT_LENGTH) = (50, 94)
RAW_DATA_HZ = 25

TRACKING_DIR = os.getenv("TRACKING_DIR")
GAMES_DIR = os.getenv("GAMES_DIR")
DATA_DIR = os.getenv("DATA_DIR")
VIDEO_DATA_DIR = os.getenv("VIDEO_DATA_DIR", f"{DATA_DIR}/videos")
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", f"{DATA_DIR}/embeddings")

TEAM_IDS_SAMPLE = {1610612748, 1610612752, 1610612755}

UPDATE_PLAY_SCHEMA = {
    "game_id": pl.String,
    "event_id": pl.String,
    "cluster_id": pl.String,
    "cluster_label": pl.String,
    "note": pl.String,
}
