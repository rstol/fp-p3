import logging
import pickle
import uuid
from pathlib import Path

import numpy as np
import polars as pl
import umap
from flask import request
from flask_restful import Resource
from sklearn.preprocessing import LabelEncoder

from backend.resources.cluster import Cluster
from backend.resources.dataset_manager import DatasetManager
from backend.resources.play_clustering import PlayClustering
from backend.resources.playid import PlayId
from backend.settings import DATA_DIR, TEAM_IDS_SAMPLE, UPDATE_PLAY_SCHEMA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamPlaysScatterResource(Resource):
    """Resource for serving team play data for scatter plot visualization."""

    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.umap_model = umap.UMAP(n_neighbors=5, metric="cosine", verbose=True, low_memory=False)
        self.clusters: list[Cluster] | None = None

        if not Path(f"{DATA_DIR}/user_updates.parquet").exists():
            user_updates = pl.DataFrame(schema=UPDATE_PLAY_SCHEMA)
            user_updates.write_parquet(f"{DATA_DIR}/user_updates.parquet")

        for team_id in TEAM_IDS_SAMPLE:
            if Path(f"{DATA_DIR}/clusters/{team_id}.pkl").exists():
                continue
            games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
            games = games.sort("game_date", descending=True).limit(5)  # max timeframe?

            game_ids = games["game_id"].to_list()
            plays = self._concat_plays_from_games(game_ids, games)
            logger.info(f"Plays for team {team_id}: {plays.height}")
            play_clustering = PlayClustering(team_id=team_id, game_ids=game_ids)
            initial_clusters = play_clustering.get_initial_clusters(plays)

            fpath = Path(f"{DATA_DIR}/clusters/{team_id}.pkl")
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with fpath.open("wb") as f:
                pickle.dump(initial_clusters, f)

    def _load_clusters_for_team(self, team_id: int):
        if Path(f"{DATA_DIR}/clusters/{team_id}.pkl").exists():
            with Path(f"{DATA_DIR}/clusters/{team_id}.pkl").open("rb") as f:
                self.clusters = pickle.load(f)

    def apply_user_updates(self):
        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")

        if self.clusters is None:
            raise ValueError("Clusters attribute is not set")

        cluster_dict = {cluster.id: cluster for cluster in self.clusters}
        play_index = {
            (play.play_id): (cluster, play) for cluster in self.clusters for play in cluster.plays
        }
        new_cluster_ids = set()

        for row in user_updates.iter_rows(named=True):
            key = PlayId(row["game_id"], row["event_id"])
            updated_cluster_id = row["cluster_id"]
            updated_label = row["cluster_label"]

            if key not in play_index:
                continue  # Mabye log this

            current_cluster, cluster_play = play_index[key]

            cluster_play.play.note = row["note"]

            # Move to new cluster if cluster_id changed
            if current_cluster.id != updated_cluster_id:
                cluster_play.play.is_tagged = True
                current_cluster.plays.remove(cluster_play)

                # Add to new cluster, create if necessary
                if updated_cluster_id not in cluster_dict:
                    new_cluster_ids.add(updated_cluster_id)
                    print("SKIPPED PLAY UPDATE DUE TO NEW CLUSTER")
                    # TODO skip for now
                    continue
                    from copy import deepcopy

                    new_cluster = deepcopy(current_cluster)
                    new_cluster.id = updated_cluster_id
                    new_cluster.label = updated_label
                    new_cluster.plays = []
                    self.clusters.append(new_cluster)
                    cluster_dict[updated_cluster_id] = new_cluster

                cluster_dict[updated_cluster_id].plays.append(cluster_play)
                play_index[key] = (cluster_dict[updated_cluster_id], cluster_play)

            elif current_cluster.label != updated_label:
                current_cluster.label = updated_label
        print("New cluster ids", new_cluster_ids)

    def _clusters_to_dicts(self):
        return [
            {
                "cluster_id": cluster.id,
                "cluster_label": cluster.label,
                "points": [
                    {
                        "game_id": cluster_play.play_id.game_id,
                        "event_id": cluster_play.play_id.event_id,
                        "event_desc_home": cluster_play.play.event_desc_home,
                        "event_desc_away": cluster_play.play.event_desc_away,
                        "game_date": cluster_play.play.game_date,
                        "event_type": cluster_play.play.event_type,
                        "similarity_distance": float(cluster_play.distance),
                        "score": cluster_play.play.event_score,
                        "note": cluster_play.play.note,
                        "is_tagged": cluster_play.play.is_tagged,
                        "x": cluster_play.x,
                        "y": cluster_play.y,
                    }
                    for cluster_play in cluster.plays
                ],
            }
            for cluster in self.clusters
        ]

    def _prepare_scatter_data_for_response(self, team_id: int, timeframe: int):
        logger.info(f"Prep scatter data: team {team_id}, timeframe {timeframe}")
        # TODO timeframes
        self._load_clusters_for_team(team_id)

        self.apply_user_updates()

        self._generate_scatter_data()

    def _concat_plays_from_games(self, game_ids: list[str], games: pl.DataFrame) -> pl.DataFrame:
        if games.height == 0:
            return pl.DataFrame()

        game_dates = games.select(["game_id", "game_date"])

        plays_df = self.dataset_manager.get_plays_for_games(game_ids).drop("moments").collect()
        return plays_df.join(game_dates, on="game_id", how="left")

    def _generate_scatter_data(self):
        embeddings = []
        y = []
        for cluster in self.clusters:
            for cluster_play in cluster.plays:
                embeddings.append(cluster_play.embedding)
                if getattr(cluster_play.play, "is_tagged", False):
                    y.append(cluster.id)
                else:
                    y.append(-1)
        X = np.stack(embeddings)
        y = np.array(y)

        encoder = LabelEncoder()
        y_encoded = np.where(y != -1, encoder.fit_transform(y), -1)
        xys = self.umap_model.fit_transform(X, y=y_encoded)

        idx = 0
        for cluster in self.clusters:
            for play in cluster.plays:
                play.x = float(xys[idx, 0])
                play.y = float(xys[idx, 1])
                idx += 1

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
        self._prepare_scatter_data_for_response(team_id, last_games)
        data = self._clusters_to_dicts()

        return data, 200


def create_partial_dataframe(data: dict, full_schema: dict) -> pl.DataFrame:
    # Only use schema fields that exist in the incoming data
    partial_schema = {k: v for k, v in full_schema.items() if k in data}
    return pl.DataFrame(data, schema=partial_schema)


class ScatterPointResource(Resource):
    def __init__(self):
        super().__init__()

    def put(self, game_id: str, event_id: str):
        update_play_data = request.get_json()
        update_play_data["event_id"] = event_id
        update_play_data["game_id"] = game_id

        if not update_play_data["cluster_id"]:
            # Create new cluster id
            update_play_data["cluster_id"] = str(uuid.uuid4())

        print(update_play_data)
        try:
            df_update = create_partial_dataframe(update_play_data, UPDATE_PLAY_SCHEMA)
        except ValueError as err:
            logger.error(err)
            return {"error": "Invalid play ID format"}, 400

        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")
        user_updates = user_updates.update(df_update, on=["game_id", "event_id"], how="full")
        user_updates.write_parquet(f"{DATA_DIR}/user_updates.parquet")

        return {"message": "Play updated successfully"}, 200


class ClusterResource(Resource):
    def __init__(self):
        super().__init__()

    def post(self, cluster_id: str):
        update_cluster_data = request.get_json()
        update_cluster_data["cluster_id"] = cluster_id

        try:
            df_update = create_partial_dataframe(update_cluster_data, UPDATE_PLAY_SCHEMA)
        except ValueError as err:
            logger.error(err)
            return {"error": "Invalid play ID format"}, 400
        print(df_update)
        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates.parquet")
        user_updates = user_updates.update(df_update, on=["cluster_id"], how="full")
        user_updates.write_parquet(f"{DATA_DIR}/user_updates.parquet")

        return {"message": "Play updated successfully"}, 200


if __name__ == "__main__":
    resource = TeamPlaysScatterResource()
    resource._prepare_scatter_data_for_response(1610612755, 3)
