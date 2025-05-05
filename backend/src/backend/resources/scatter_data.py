import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from flask import request
from flask_restful import Resource
from sklearn.cluster import KMeans

from backend.resources.dataset_manager import DatasetManager
from backend.settings import DATASET_DIR

from .clustering import cop_kmeans

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

        num_games = self._parse_timeframe(request.args.get("timeframe", "last_3"))

        try:
            team_id = int(team_id)
        except ValueError:
            logger.error(f"Invalid team_id format: {team_id}")
            return {"error": "Invalid team ID format"}, 400

        # Get and filter games
        games, total_games = self._get_filtered_games(team_id, num_games)
        if isinstance(games, tuple):
            return games  # This is an error response

        # Calculate recency values
        game_recency = self._calculate_game_recency(games)

        # Process plays
        plays = self._concat_plays_from_games(games, game_recency)
        logger.info(f"Collected {len(plays)} plays for generating scatter points")
        if plays.is_empty():
            return {"error": "No plays found"}, 404

        # Generate scatter data
        scatter_data = self._generate_scatter_data(plays)
        scatter_data = self._apply_clustering(scatter_data)
        scatter_data = scatter_data.select(
            [
                "x",
                "y",
                "cluster",
                "play_type",
                "description",
                "event_id",
                "game_id",
                "recency",
            ]
        )

        return {"total_games": total_games, "points": scatter_data.to_dicts()}

    def _parse_timeframe(self, timeframe):
        """Parse and validate the timeframe parameter."""
        try:
            num_games = int(timeframe.split("_")[1])
            logger.info(f"Using timeframe: {timeframe} (limiting to {num_games} games)")
            return num_games
        except (ValueError, AttributeError, IndexError) as e:
            logger.error(f"Invalid timeframe format: {timeframe}. Error: {e}")
            return {"error": "Invalid timeframe format"}, 400

    def _get_filtered_games(self, team_id, num_games):
        """Get and filter games for a team."""
        games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
        logger.info(f"Found {len(games)} games for team {team_id}")
        total_games = len(games)

        # Sort games by date (if available) and limit to the requested number
        if not games.is_empty() and "game_date" in games.columns:
            games = games.sort("game_date", descending=True)

        # Limit to the requested number of games based on timeframe
        games = games.limit(num_games)
        logger.info(f"Limited to {len(games)} games based on timeframe")

        if games.is_empty():
            logger.warning(f"No games found for team {team_id}")
            return {"error": "No games found for this team"}, 404

        return games, total_games

    def _calculate_game_recency(self, games):
        """Calculate recency values for games based on dates."""
        game_recency = {}
        if games.is_empty() or "game_date" not in games.columns:
            return game_recency

        game_dates = games.get_column("game_date")
        game_ids = games.get_column("game_id")
        dates_list = game_dates.to_list()
        ids_list = game_ids.to_list()
        try:
            datetime_dates = [datetime.strptime(str(date), "%Y-%m-%d") for date in dates_list]
            if len(set(datetime_dates)) > 1:
                max_date = max(datetime_dates)
                min_date = min(datetime_dates)
                date_range = (max_date - min_date).total_seconds()
                for game_id, game_date in zip(ids_list, datetime_dates, strict=True):
                    if date_range > 0:
                        normalized_recency = 1 - (
                            (max_date - game_date).total_seconds() / date_range
                        )
                    else:
                        normalized_recency = 1.0
                    game_recency[game_id] = normalized_recency
            else:
                for game_id in ids_list:
                    game_recency[game_id] = 1.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse dates: {e}. Using default recency values.")
            for i, game_id in enumerate(ids_list):
                game_recency[game_id] = 1.0 - (i / max(1, len(ids_list) - 1))
        return game_recency

    def _concat_plays_from_games(
        self, games: pl.DataFrame, game_recency: dict = None
    ) -> pl.DataFrame:
        all_plays: list[pl.DataFrame] = []
        schema = None
        if game_recency is None:
            game_recency = {}
            for game in games.rows(named=True):
                game_recency[game["game_id"]] = 1.0
        for game in games.rows(named=True):
            game_id = game["game_id"]
            logger.info(f"Processing game {game_id}")

            plays = self.dataset_manager.get_plays_for_game(game_id, as_dicts=False)
            if isinstance(plays, pl.DataFrame) and not plays.is_empty():
                recency = game_recency.get(game_id, 1.0)
                plays = plays.with_columns(pl.lit(recency).alias("recency"))
                if schema is None:
                    schema = plays.schema
                    all_plays.append(plays)
                else:
                    try:
                        all_plays.append(plays.cast(schema))
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Skipping game {game_id} due to schema issue: {e}")
        if not all_plays:
            return pl.DataFrame()

        plays = pl.concat(all_plays)
        return plays.with_columns(
            pl.when(pl.col("event_desc_home") == "nan")
            .then(pl.col("event_desc_away"))
            .otherwise(pl.col("event_desc_home"))
            .str.head(15)
            .alias("description"),
            pl.col("event_type").alias("play_type"),
        )

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
        n_clusters = max(2, n_clusters)  # At least 2 clusters

        try:
            clusters, centers = cop_kmeans(dataset=coords, k=n_clusters)
            data_points = data_points.with_columns(
                pl.Series(clusters, dtype=pl.Int32).alias("cluster")
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error during clustering: {e}")
            # Default to assigning alternating clusters
            clusters = []
            for i in range(len(data_points)):
                clusters.append(i % n_clusters)
            data_points = data_points.with_columns(
                pl.Series(clusters, dtype=pl.Int32).alias("cluster")
            )

        return data_points
