import logging
import os

import numpy as np
import pandas as pd
import polars as pl
from flask import request
from flask_restful import Resource
from sklearn.cluster import KMeans

from backend.resources.dataset_manager import DatasetManager
from backend.settings import TRACKING_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetResource(Resource):
    """dataset resource."""

    data_root = os.path.join(".", "data")

    def get(self, name):
        path_name = os.path.join(self.data_root, f"dataset_{name}.csv")
        data = pd.read_csv(path_name)

        # process the data, e.g. find the clusters
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
        labels = kmeans.labels_.tolist()

        # Add cluster to data
        data["cluster"] = labels

        # Convert to dictionary
        return data.to_dict(orient="records")


class TeamPlaysScatterResource(Resource):
    """Resource for serving team play data for scatter plot visualization."""

    def __init__(self):
        self.dataset_manager = DatasetManager(TRACKING_DIR)

        self.play_centroids = {
            0: (20, 20),
            1: (80, 80),
            2: (40, 20),
            3: (60, 40),
            4: (60, 40),
            5: (80, 80),
            6: (20, 60),
            7: (40, 60),
            8: (40, 20),
            9: (20, 40),
        }

        self.cluster_std_dev = 25

    def get(self, team_id):
        """Get scatter plot data for a team's plays."""
        logger.info(f"Fetching scatter data for team_id: {team_id}")

        timeframe = request.args.get("timeframe", "last_3")
        try:
            num_games = int(timeframe.split("_")[1])
        except (ValueError, AttributeError):
            logger.error(f"Invalid timeframe format: {timeframe}")
            return {"error": "Invalid timeframe format"}, 400
        logger.info(f"Using timeframe: {timeframe} (limiting to {num_games} games)")

        try:
            team_id = int(team_id)
        except ValueError:
            logger.error(f"Invalid team_id format: {team_id}")
            return {"error": "Invalid team ID format"}, 400

        games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
        logger.info(f"Found {len(games)} games for team {team_id}")
        total_games = len(games)

        # Sort games by date (if available) and limit to the requested number
        if not games.is_empty() and "game_date" in games.columns:
            games = games.sort("game_date", descending=True)

        # Limit to the requested number of games based on timeframe
        games = games.limit(num_games)
        logger.info(f"Limited to {len(games)} games based on timeframe {timeframe}")

        if games.is_empty():
            logger.warning(f"No games found for team {team_id}")
            return {"error": "No games found for this team"}, 404

        plays = self._concat_plays_from_games(games)
        logger.info(f"Collected {len(plays)} plays for generating scatter points")
        if plays.is_empty():
            return {"error": "No plays found"}, 404

        scatter_data = self._generate_scatter_data(plays)

        scatter_data = self._apply_clustering(scatter_data)

        final_columns = [
            "x",
            "y",
            "cluster",
            "event_id",
            "game_id",
            "event_desc_home",  # Full home description
            "event_desc_away",  # Full away description
            "game_date",  # Game date
            "event_type",  # Original event_type (numeric)
        ]

        # TODO: check if this is needed?
        for col_name in final_columns:
            if col_name not in scatter_data.columns:
                if col_name in ["event_desc_home", "event_desc_away", "game_date"]:
                    scatter_data = scatter_data.with_columns(
                        pl.lit(None, dtype=pl.String).alias(col_name)
                    )
                else:
                    scatter_data = scatter_data.with_columns(pl.lit(None).alias(col_name))

        scatter_data = scatter_data.select(final_columns)

        return {"total_games": total_games, "points": scatter_data.to_dicts()}

    def _concat_plays_from_games(self, games: pl.DataFrame) -> pl.DataFrame:
        all_plays_list: list[pl.DataFrame] = []
        for game_row in games.rows(named=True):
            game_id = game_row["game_id"]
            current_game_date = game_row.get("game_date")
            logger.info(f"Processing game {game_id}")

            plays_df = self.dataset_manager.get_plays_for_game(game_id, as_dicts=False)
            if isinstance(plays_df, pl.DataFrame) and not plays_df.is_empty():
                if current_game_date:
                    plays_df = plays_df.with_columns(pl.lit(current_game_date).alias("game_date"))
                else:
                    plays_df = plays_df.with_columns(
                        pl.lit(None, dtype=pl.String).alias("game_date")
                    )

                all_plays_list.append(plays_df)

        if len(all_plays_list) == 0:
            return pl.DataFrame()

        # TODO: check this?
        plays = pl.concat(all_plays_list, how="diagonal_relaxed")

        return plays

    def _generate_scatter_data(self, plays):
        x_centroids = {k: v[0] for k, v in self.play_centroids.items()}
        y_centroids = {k: v[1] for k, v in self.play_centroids.items()}

        plays = plays.with_columns(
            x=pl.col("event_type").mod(10).replace(x_centroids),
            y=pl.col("event_type").mod(10).replace(y_centroids),
        )

        num_rows = plays.shape[0]
        return plays.with_columns(
            x=pl.col("x") + pl.Series(np.random.normal(0, self.cluster_std_dev, num_rows)),
            y=pl.col("y") + pl.Series(np.random.normal(0, self.cluster_std_dev, num_rows)),
        )

    def _apply_clustering(self, data_points):
        coords = data_points.select(["x", "y"]).to_numpy()

        n_clusters = 3 if len(data_points) >= 9 else min(len(data_points) // 3, 2)
        n_clusters = max(2, n_clusters)

        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(coords)
            data_points = data_points.with_columns(
                pl.Series(clusters, dtype=pl.Int32).alias("cluster")
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error during clustering: {e}")
            for i, point in enumerate(data_points):
                point["cluster"] = i % n_clusters

        return data_points
