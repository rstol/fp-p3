import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import umap
from flask import request
from flask_restful import Resource

from backend.resources.cluster import Cluster
from backend.resources.dataset_manager import DatasetManager
from backend.resources.play_clustering import PlayClustering
from backend.settings import DATA_DIR, TRACKING_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamPlaysScatterResource(Resource):
    """Resource for serving team play data for scatter plot visualization."""

    def __init__(self):
        self.dataset_manager = DatasetManager(TRACKING_DIR)

        if not Path(f"{DATA_DIR}/plays_by_team.pkl").exists():
            self.plays_by_team = {
                teamid: self.dataset_manager.get_games_for_team(teamid, as_dicts=False)
                for teamid in self.dataset_manager.teams["teamid"]
            }
            for teamid, games in self.plays_by_team.items():
                games = games.sort("game_date", descending=True)
                games = games.limit(3)
                plays = self._concat_plays_from_games(games)
                play_clustering = PlayClustering(
                    team_id=teamid, game_ids=plays["game_id"].unique().to_list()
                )
                initial_clusters, cluster_assignments, team_embedding_ids = (
                    play_clustering.get_initial_clusters()
                )
                plays = plays.with_columns(
                    pl.col("game_id").cast(pl.Int64), pl.col("event_id").cast(pl.Int64)
                ).join(
                    team_embedding_ids.drop("index").with_row_index("ids_index"),
                    on=["game_id", "event_id"],
                    how="inner",
                )
                plays_index = plays["ids_index"].to_numpy()
                plays = plays.with_columns(
                    team_embeddings=pl.lit(play_clustering.team_embeddings[plays_index]),
                    cluster=pl.lit(cluster_assignments[plays_index]).cast(pl.Int32),
                )
                self.plays_by_team[teamid] = {"initial_clusters": initial_clusters, "plays": plays}

            with Path(f"{DATA_DIR}/plays_by_team.pkl").open("wb") as f:
                pickle.dump(self.plays_by_team, f)
        else:
            with Path(f"{DATA_DIR}/plays_by_team.pkl").open("rb") as f:
                self.plays_by_team = pickle.load(f)
        self.umap_model = umap.UMAP(n_neighbors=5)

    def _prepare_scatter_data_for_response(self, team_id: str, timeframe_str: str):
        logger.info(f"Prep scatter data: team {team_id}, timeframe {timeframe_str}")

        try:
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

        plays_of_team = self.plays_by_team.get(team_id, pl.DataFrame())

        scatter_data_df = self._generate_scatter_data(
            plays_of_team["plays"], initial_clusters=plays_of_team["initial_clusters"]
        )

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
                if col_name in ["x", "y", "cluster", "event_type"]:
                    scatter_data_df = scatter_data_df.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias(col_name)
                    )
                else:
                    scatter_data_df = scatter_data_df.with_columns(
                        pl.lit(None, dtype=pl.String).alias(col_name)
                    )

        # Select only the final columns in the specified order
        scatter_data_df = scatter_data_df.select(final_columns)

        return {"points": scatter_data_df.to_dicts()}, 200

    def get(self, team_id):
        """Get scatter plot data for a team's plays."""
        timeframe = request.args.get("timeframe", "last_3")
        data, status_code = self._prepare_scatter_data_for_response(team_id, timeframe)
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

    def _generate_scatter_data(
        self, plays: pl.DataFrame, initial_clusters: list[Cluster]
    ) -> pl.DataFrame:
        y = np.full(len(plays["team_embeddings"]), -1)
        for cluster in initial_clusters:
            for play in cluster.plays:
                y[play.index] = cluster.id

        xys = self.umap_model.fit_transform(plays["team_embeddings"], y=y)

        return plays.with_columns(x=pl.lit(xys[:, 0]), y=pl.lit(xys[:, 1]))
