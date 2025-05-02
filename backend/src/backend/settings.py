import os

DATASET_DIR = os.getenv("DATASET_DIR", "data/nba_tracking_data")
SAMPLING_RATE = 3  # 25 Hz -> 25/3 Hz ≈ 8.33 Hz
(COURT_WIDTH, COURT_LENGTH) = (50, 94)
