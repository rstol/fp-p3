import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import umap
from flask import request
from flask_restful import Resource

from backend.resources.dataset_manager import DatasetManager
from backend.resources.play_clustering import PlayClustering
from backend.settings import DATA_DIR, TRACKING_DIR, UPDATE_PLAY_SCHEMA

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
            initial_clusters_update = pl.DataFrame(schema=UPDATE_PLAY_SCHEMA)
            for teamid, games in self.plays_by_team.items():
                games = games.sort("game_date", descending=True)
                games = games.limit(3)
                plays = self._concat_plays_from_games(games)
                logger.info(f"Plays for team {teamid}: {plays.height}")
                play_clustering = PlayClustering(
                    team_id=teamid, game_ids=plays["game_id"].unique().to_list()
                )
                initial_clusters, cluster_assignments, team_embedding_ids = (
                    play_clustering.get_initial_clusters()
                )
                plays = plays.with_columns(pl.col("game_id"), pl.col("event_id")).join(
                    team_embedding_ids.drop("index").with_row_index("ids_index"),
                    on=["game_id", "event_id"],
                    how="inner",
                )
                plays_index = plays["ids_index"].to_numpy()
                plays = plays.with_columns(
                    team_embeddings=pl.lit(play_clustering.team_embeddings[plays_index]),
                    cluster_id=pl.lit(cluster_assignments[plays_index]).cast(pl.Int32),
                )
                self.plays_by_team[teamid] = plays.select(
                    "cluster_id",
                    "event_id",
                    "game_id",
                    "event_desc_home",
                    "event_desc_away",
                    "game_date",
                    "event_type",
                    "ids_index",
                    "team_embeddings",
                )

                initial_clusters_update_dicts = [
                    {
                        "game_id": cluster.plays[0].play_id.game_id,
                        "event_id": cluster.plays[0].play_id.event_id,
                        "cluster_id": cluster.id,
                        "cluster_name": cluster.label,
                        "note": "",
                    }
                    for cluster in initial_clusters
                ]
                initial_clusters_update = initial_clusters_update.update(
                    pl.from_dicts(initial_clusters_update_dicts, schema=UPDATE_PLAY_SCHEMA),
                    on=["game_id", "event_id"],
                    how="full",
                )

            if not Path(f"{DATA_DIR}/user_updates.parquet").exists():
                user_updates = pl.DataFrame(schema=UPDATE_PLAY_SCHEMA)
                user_updates.write_parquet(f"{DATA_DIR}/user_updates.parquet")

            initial_clusters_update.write_parquet(f"{DATA_DIR}/initial_clusters_update.parquet")

            with Path(f"{DATA_DIR}/plays_by_team.pkl").open("wb") as f:
                pickle.dump(self.plays_by_team, f)
        else:
            with Path(f"{DATA_DIR}/plays_by_team.pkl").open("rb") as f:
                self.plays_by_team = pickle.load(f)
        self.umap_model = umap.UMAP(n_neighbors=5, metric="cosine", verbose=True, low_memory=False)

    def _prepare_scatter_data_for_response(self, team_id: int, timeframe: int):
        logger.info(f"Prep scatter data: team {team_id}, timeframe {timeframe}")
        plays_of_team = self.plays_by_team.get(team_id, pl.DataFrame())

        def apply_user_updates(plays: pl.DataFrame) -> pl.DataFrame:
            user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")
            initial_clusters_update = pl.read_parquet(f"{DATA_DIR}/initial_clusters_update.parquet")
            user_updates = initial_clusters_update.update(
                user_updates, on=["game_id", "event_id"], how="full"
            )

            plays = plays.with_columns(pl.lit(value=False).alias("isTagged"))
            user_updates = user_updates.with_columns(pl.lit(value=True).alias("isTagged"))
            return plays.update(user_updates, on=["game_id", "event_id"], how="left")

        plays_of_team = apply_user_updates(plays_of_team)

        scatter_data_df = self._generate_scatter_data(plays_of_team)

        scatter_data_df = scatter_data_df.select(
            "x",
            "y",
            "cluster_id",
            "event_id",
            "game_id",
            "event_desc_home",
            "event_desc_away",
            "game_date",
            "event_type",
            "isTagged",
        )

        cluster_dicts = (
            scatter_data_df.with_columns(pl.struct(pl.exclude("cluster_id")).alias("points"))
            .select("cluster_id", "points")
            .group_by("cluster_id")
            .agg(pl.col("points"))
            .to_dicts()
        )

        return cluster_dicts, 200

    def get(self, team_id):
        """Get scatter plot data for a team's plays."""
        timeframe = request.args.get("timeframe", "last_3")
        try:
            parts = timeframe.split("_")
            if len(parts) < 2 or not parts[1].isdigit():
                msg = "Timeframe format error"
                raise ValueError(msg)  # noqa: TRY301
            last_games = int(parts[1])
        except ValueError:
            logger.exception(f"Invalid timeframe format: {timeframe}")
            return {"error": "Invalid timeframe format"}, 400
        logger.info(f"Using timeframe: {timeframe} (limiting to {last_games} games)")

        try:
            team_id = int(team_id)
        except ValueError:
            logger.exception(f"Invalid team_id format: {team_id}")
            return {"error": "Invalid team ID format"}, 400
        data, status_code = self._prepare_scatter_data_for_response(team_id, last_games)
        return data, status_code

    def _concat_plays_from_games(self, games: pl.DataFrame) -> pl.DataFrame:
        if games.height == 0:
            return pl.DataFrame()

        game_ids = games["game_id"].to_list()
        game_dates = games.select(["game_id", "game_date"])

        plays_df = self.dataset_manager.get_plays_for_games(game_ids).collect()

        return plays_df.join(game_dates, on="game_id", how="left")

    def _generate_scatter_data(self, plays: pl.DataFrame) -> pl.DataFrame:
        y = np.full(len(plays["team_embeddings"]), -1)
        y_plays = plays.filter(pl.col("isTagged") & pl.col("ids_index").is_not_null())
        y_play_ids = y_plays["ids_index"].to_numpy()
        y[y_play_ids] = y_plays["cluster_id"].to_numpy()

        xys = self.umap_model.fit_transform(plays["team_embeddings"], y=y)

        return plays.with_columns(x=pl.lit(xys[:, 0]), y=pl.lit(xys[:, 1]))


if __name__ == "__main__":
    resource = TeamPlaysScatterResource()
    resource._prepare_scatter_data_for_response(1610612755, 3)
