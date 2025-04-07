import logging
import os

import numpy as np
import pandas as pd
import polars as pl
from flask_restful import Resource
from sklearn.cluster import KMeans

from backend.resources.dataset_manager import DatasetManager
from backend.settings import DATASET_DIR

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
        self.dataset_manager = DatasetManager(DATASET_DIR)

    def get(self, team_id):
        # Log the request
        logger.info(f"Fetching scatter data for team_id: {team_id}")

        # Convert team_id to proper format
        try:
            team_id = int(team_id)
        except ValueError:
            logger.error(f"Invalid team_id format: {team_id}")
            return {"error": "Invalid team ID format"}, 400

        # Get games for the specified team
        games = self.dataset_manager.get_games_for_team(team_id)
        logger.info(f"Found {len(games)} games for team {team_id}")

        if not games:
            logger.warning(f"No games found for team {team_id}")
            return {"error": "No games found for this team"}, 404

        scatter_data = []
        for game in games[:2]:  # Process first 2 games
            game_id = game["game_id"]
            logger.info(f"Processing plays for game {game_id}")

            # Get all plays for this game as a polars DataFrame
            plays_df = self.dataset_manager.get_plays_for_game(game_id, as_dicts=False)
            logger.info(f"Found {len(plays_df)} plays for game {game_id}")

            # Filter plays for the requested team
            team_plays = self._filter_plays_for_team(plays_df, team_id)

            # Extract coordinates from moments data for team plays
            points = self._extract_coordinates_from_df(team_plays, team_id)
            scatter_data.append(points)

        scatter_data = pl.concat(scatter_data)
        return self._apply_clustering(scatter_data).to_dicts()

    def _apply_clustering(self, data_points):
        """Apply KMeans clustering to the data points."""
        if len(data_points) < 3:
            logger.warning("Too few data points for clustering, defaulting all to cluster 0")
            return data_points.with_columns(pl.lit(0).cast(pl.Int64).alias("cluster"))

        # Extract coordinates for clustering
        coords = data_points.select(["x", "y"]).to_numpy()
        n_clusters = len(data_points["event_type"].unique())

        logger.info(f"Clustering {len(data_points)} points into {n_clusters} clusters")

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        clusters = kmeans.fit_predict(coords)
        return data_points.with_columns(pl.Series("cluster", clusters))

    def _filter_plays_for_team(self, plays_df, team_id):
        """Filter plays for a specific team from a polars DataFrame."""
        if plays_df is None or len(plays_df) == 0:
            logger.warning("No plays to filter")
            return plays_df

        # FIXME(mboss): why are these floats and structs???
        return plays_df.filter(
            (plays_df["possession_team_id"] == float(team_id))
            | (plays_df["primary_player_info"].struct[0] == float(team_id))
        ).unique()

    def _extract_coordinates_from_df(self, plays_df, team_id):
        """Extract coordinate data from a DataFrame of plays."""
        if plays_df is None or len(plays_df) == 0:
            logger.warning("No plays to extract coordinates from")
            return []

        points = plays_df.select(
            [
                "event_id",
                "event_type",
                "event_desc_home",
                "event_desc_away",
                "game_id",
            ]
        )
        return points.with_columns(
            [
                (
                    pl.col("event_type") * 100 + (pl.lit(np.random.rand(points.height)) * 40 - 20)
                ).alias("x"),
                (
                    pl.col("event_type") * 50 + (pl.lit(np.random.rand(points.height)) * 30 - 15)
                ).alias("y"),
            ]
        )
