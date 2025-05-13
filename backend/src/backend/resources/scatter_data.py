import logging

import numpy as np
import polars as pl
from flask import request
from flask_restful import Resource
from sklearn.cluster import KMeans

from backend.resources.dataset_manager import DatasetManager
from backend.settings import TRACKING_DIR

from .play_clustering import PlayClustering

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def _prepare_scatter_data_for_response(self, team_id: str, timeframe_str: str):
        logger.info(f"Prep scatter data: team {team_id}, timeframe {timeframe_str}")

        try:
            # Ensure splitting timeframe_str and accessing index [1] is safe
            parts = timeframe_str.split("_")
            if len(parts) < 2 or not parts[1].isdigit():
                msg = "Timeframe format error"
                raise ValueError(msg)  # noqa: TRY301
            num_games = int(parts[1])
        except ValueError:
            logger.exception(f"Invalid timeframe format: {timeframe_str}")
            return {"error": "Invalid timeframe format"}, 400
        logger.info(f"Using timeframe: {timeframe_str} (limiting to {num_games} games)")

        try:
            team_id = int(team_id)
        except ValueError:
            logger.exception(f"Invalid team_id format: {team_id}")
            return {"error": "Invalid team ID format"}, 400

        games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
        logger.info(f"Found {len(games)} games for team {team_id}")
        total_games = len(games)

        if not games.is_empty() and "game_date" in games.columns:
            games = games.sort("game_date", descending=True)

        games = games.limit(num_games)
        logger.info(f"Limited to {len(games)} games for timeframe {timeframe_str}")

        if games.is_empty():
            logger.warning(
                f"No games for team {team_id} (timeframe: {timeframe_str}). Returning empty points."
            )
            return {"total_games": total_games, "points": []}, 200

        plays = self._concat_plays_from_games(games)
        logger.info(f"Collected {len(plays)} plays.")
        if plays.is_empty():
            return {"total_games": total_games, "points": []}, 200

        scatter_data_df = self._generate_scatter_data(plays, team_id)
        # scatter_data_df = self._apply_clustering(scatter_data_df)

        final_columns = [
            "x",
            "y",
            "cluster",
            "event_id",
            "game_id",
            "event_desc_home",
            "event_desc_away",
            "game_date",
            "event_type",
        ]

        # Ensure all final_columns exist, adding them with nulls if not present
        current_cols = scatter_data_df.columns
        for col_name in final_columns:
            if col_name not in current_cols:
                # Determine dtype for pl.lit carefully
                if col_name in ["x", "y", "cluster", "event_type"]:
                    # Numeric columns
                    scatter_data_df = scatter_data_df.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias(col_name)
                    )
                else:
                    # String columns
                    scatter_data_df = scatter_data_df.with_columns(
                        pl.lit(None, dtype=pl.String).alias(col_name)
                    )

        # Select only the final columns in the specified order
        scatter_data_df = scatter_data_df.select(final_columns)

        return {"total_games": total_games, "points": scatter_data_df.to_dicts()}, 200

    def get(self, team_id):
        """Get scatter plot data for a team's plays."""
        timeframe = request.args.get("timeframe", "last_3")
        data, status_code = self._prepare_scatter_data_for_response(team_id, timeframe)
        return data, status_code

    def post(self, team_id):
        logger.info(f"Received cluster update POST request for team_id: {team_id}")
        updated_plays_data = request.get_json()

        if not updated_plays_data:
            logger.warning(f"No data provided in cluster update for team_id: {team_id}")
            return {"error": "No data provided for update"}, 400

        logger.info(f"Cluster update data for team_id {team_id}: {updated_plays_data}")
        # TODO(mboss): Implement the actual logic to process these updates in the data source.

        # For now, return the original/current scatter data for this team.
        timeframe_for_refresh = request.args.get("timeframe", "last_3")
        data, status_code = self._prepare_scatter_data_for_response(team_id, timeframe_for_refresh)
        return data, status_code

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

        # TODO(mboss): check this?
        return pl.concat(all_plays_list, how="diagonal_relaxed")

    def _generate_scatter_data(self, plays: pl.DataFrame, team_id: int) -> pl.DataFrame:
        play_clustering = PlayClustering(team_id=team_id)
        initial_clusters = play_clustering.get_initial_clusters()

        x_centroids = {cluster.id: cluster.centroid[0] for cluster in initial_clusters}
        y_centroids = {cluster.id: cluster.centroid[1] for cluster in initial_clusters}

        plays = plays.with_columns(
            x=pl.col("event_type").mod(10).replace(x_centroids),
            y=pl.col("event_type").mod(10).replace(y_centroids),
        )

        num_rows = plays.shape[0]
        return plays.with_columns(
            x=pl.col("x") + pl.Series(np.random.normal(0, self.cluster_std_dev, num_rows)),
            y=pl.col("y") + pl.Series(np.random.normal(0, self.cluster_std_dev, num_rows)),
            cluster=pl.col("event_type").mod(10),
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
        except (AttributeError, TypeError, ValueError):
            logger.exception("Error during clustering")
            for i, point in enumerate(data_points):
                point["cluster"] = i % n_clusters

        return data_points
