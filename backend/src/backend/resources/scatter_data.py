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
from backend.settings import DATA_DIR, TRACKING_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamPlaysScatterResource(Resource):
    """Resource for serving team play data for scatter plot visualization."""

    def __init__(self):
        self.dataset_manager = DatasetManager(TRACKING_DIR)

        self.user_updates_schema = {
            "game_id": pl.Int64,
            "event_id": pl.Int64,
            "cluster": pl.String,
            "cluster_name": pl.String,
            "note": pl.String,
        }

        if not Path(f"{DATA_DIR}/plays_by_team.pkl").exists():
            self.plays_by_team = {
                teamid: self.dataset_manager.get_games_for_team(teamid, as_dicts=False)
                for teamid in self.dataset_manager.teams["teamid"]
            }
            if not Path(f"{DATA_DIR}/user_updates.parquet").exists():
                user_updates = pl.DataFrame(schema=self.user_updates_schema)
            else:
                user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")

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

                user_update_dicts = [
                    {
                        "game_id": cluster.plays[0].play_id.game_id,
                        "event_id": cluster.plays[0].play_id.event_id,
                        "cluster": cluster.id,
                        "cluster_name": cluster.label,
                        "note": "",
                    }
                    for cluster in initial_clusters
                ]
                user_updates = user_updates.update(
                    pl.from_dicts(user_update_dicts, schema=self.user_updates_schema),
                    on=["game_id", "event_id"],
                    how="full",
                )

            user_updates.write_parquet(f"{DATA_DIR}/user_updates.parquet")

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

        def apply_user_updates(plays: pl.DataFrame, user_updates: pl.DataFrame) -> pl.DataFrame:
            plays = plays.with_columns(pl.lit(value=False).alias("isTagged"))
            user_updates = user_updates.with_columns(pl.lit(value=True).alias("isTagged"))
            return plays.update(user_updates, on=["game_id", "event_id"], how="left")

        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")
        plays_of_team["plays"] = apply_user_updates(
            plays=plays_of_team["plays"], user_updates=user_updates
        )

        scatter_data_df = self._generate_scatter_data(plays_of_team["plays"])

        scatter_data_df = scatter_data_df.select(
            "x",
            "y",
            "cluster",
            "event_id",
            "game_id",
            "event_desc_home",
            "event_desc_away",
            "game_date",
            "event_type",
            "isTagged",
        )

        cluster_dicts = (
            scatter_data_df.with_columns(pl.struct(pl.exclude("cluster")).alias("points"))
            .select("cluster", "points")
            .group_by("cluster")
            .agg(pl.col("points"))
            .to_dicts()
        )

        return cluster_dicts, 200

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

    def _generate_scatter_data(self, plays: pl.DataFrame) -> pl.DataFrame:
        y = np.full(len(plays["team_embeddings"]), -1)
        y_plays = plays.filter(pl.col("isTagged") & pl.col("ids_index").is_not_null())
        y_play_ids = y_plays["ids_index"].to_numpy()
        y[y_play_ids] = y_plays["cluster"].to_numpy()

        xys = self.umap_model.fit_transform(plays["team_embeddings"], y=y)

        return plays.with_columns(x=pl.lit(xys[:, 0]), y=pl.lit(xys[:, 1]))
