import logging
import pickle
import time
import uuid
from collections import defaultdict
from datetime import datetime
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
from backend.settings import DATA_DIR, UPDATE_PLAY_SCHEMA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamPlaysScatterResource(Resource):
    """Resource for serving team play data for scatter plot visualization."""

    INITIAL_CLUSTER_NUM = 8

    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.umap_model = umap.UMAP(n_neighbors=5, metric="cosine", verbose=True, low_memory=False)
        self.clusters: list[Cluster] | None = None

    def _init_clusters(self, team_id: int):
        fpath = Path(f"{DATA_DIR}/user_updates/{team_id}.parquet")
        if not fpath.exists():
            user_updates = pl.DataFrame(schema=UPDATE_PLAY_SCHEMA)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            user_updates.write_parquet(fpath)

        if Path(f"{DATA_DIR}/init_clusters/{team_id}.pkl").exists():
            return

        game_ids, plays = self._load_plays_for_team(team_id)
        logger.info(f"Plays for team {team_id}: {plays.height}")
        play_clustering = PlayClustering(
            team_id=team_id, game_ids=game_ids, initial_k=self.INITIAL_CLUSTER_NUM
        )
        initial_clusters = play_clustering.get_initial_clusters(plays)

        fpath = Path(f"{DATA_DIR}/init_clusters/{team_id}.pkl")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with fpath.open("wb") as f:
            pickle.dump(initial_clusters, f)

    def _load_plays_for_team(self, team_id: int, timeframe=5):
        games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
        games = games.sort("game_date", descending=True).limit(timeframe)

        game_ids = games["game_id"].to_list()
        return game_ids, self._concat_plays_from_games(game_ids, games)

    def _load_clusters_for_team(self, team_id: int):
        if Path(f"{DATA_DIR}/clusters/{team_id}.pkl").exists():
            with Path(f"{DATA_DIR}/clusters/{team_id}.pkl").open("rb") as f:
                self.clusters = pickle.load(f)
        if Path(f"{DATA_DIR}/init_clusters/{team_id}.pkl").exists():
            with Path(f"{DATA_DIR}/init_clusters/{team_id}.pkl").open("rb") as f:
                self.clusters = pickle.load(f)

    def apply_user_updates(self, team_id: int, timeframe: int):
        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates/{team_id}.parquet")

        if self.clusters is None:
            raise ValueError("Clusters attribute is not set")

        cluster_dict = {cluster.id: cluster for cluster in self.clusters}
        play_index = {
            (play.play_id): (cluster, play) for cluster in self.clusters for play in cluster.plays
        }
        new_cluster_ids = set()

        for row in user_updates.iter_rows(named=True):
            play_id = None
            updated_cluster_id = row.get("cluster_id")
            updated_label = row.get("cluster_label")
            if (game_id := row.get("game_id")) and (event_id := row.get("event_id")):
                play_id = PlayId(game_id, event_id)
            elif updated_cluster_id and updated_label:
                # No play just label update
                cluster_dict[updated_cluster_id].label = updated_label

            if not play_id or play_id not in play_index:
                continue  # Mabye log this

            current_cluster, cluster_play = play_index[play_id]

            cluster_play.play.note = row["note"]
            cluster_play.play.is_tagged = True

            # Move to new cluster if cluster_id changed
            if updated_cluster_id and current_cluster.id != updated_cluster_id:
                current_cluster.plays.remove(cluster_play)

                # Add to new cluster, create if necessary
                if updated_cluster_id not in cluster_dict:
                    new_cluster_ids.add(updated_cluster_id)

                    new_cluster = Cluster(
                        id=updated_cluster_id,
                        label=updated_label,
                        centroid=cluster_play.embedding,
                        plays=[],
                        confidence=None,
                        created=time.time(),
                        last_modified=time.time(),
                        created_by="user",
                    )

                    game_ids, plays = self._load_plays_for_team(team_id, timeframe)
                    play_clustering = PlayClustering(team_id=team_id, game_ids=game_ids)
                    embedding_ids, distances, embeddings = play_clustering.find_similar_plays(
                        new_cluster.centroid
                    )
                    cluster_assignments = np.full((len(embedding_ids),), 0)

                    cluster_play_list = play_clustering.merge_plays_embeddings(
                        plays, embedding_ids, embeddings, distances, cluster_assignments, 1
                    )[0]
                    target_play_ids = set(
                        play.play_id
                        for play in cluster_play_list
                        if play.play_id != cluster_play.play_id and not cluster_play.play.is_tagged
                    )
                    moved_plays = []
                    for cluster in cluster_dict.values():
                        remaining_plays = []
                        for cluster_play in cluster.plays:
                            if cluster_play.play_id in target_play_ids:
                                moved_plays.append(cluster_play)
                            else:
                                remaining_plays.append(cluster_play)
                        cluster.plays = remaining_plays

                    new_cluster.plays.extend(moved_plays)
                    self.clusters.append(new_cluster)
                    cluster_dict[updated_cluster_id] = new_cluster

                cluster_dict[updated_cluster_id].plays.append(cluster_play)
                cluster_dict[updated_cluster_id].last_modified = time.time()
                play_index[play_id] = (cluster_dict[updated_cluster_id], cluster_play)

        fpath = Path(f"{DATA_DIR}/clusters/{team_id}.pkl")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with fpath.open("wb") as f:
            pickle.dump(self.clusters, f)
        user_updates = pl.DataFrame(schema=UPDATE_PLAY_SCHEMA)
        user_updates.write_parquet(f"{DATA_DIR}/user_updates/{team_id}.parquet")

    def _clusters_to_dicts(self, cluster_plays):
        json = []
        for cluster in self.clusters:
            plays = cluster_plays[cluster.id]
            json.append(
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
                            "possession_team_id": cluster_play.play.possession_team_id,
                            "quarter": cluster_play.play.quarter,
                            "similarity_distance": float(cluster_play.distance),
                            "score": cluster_play.play.event_score,
                            "note": cluster_play.play.note,
                            "is_tagged": cluster_play.play.is_tagged,
                            "x": cluster_play.x,
                            "y": cluster_play.y,
                            "recency": cluster_play.recency,
                        }
                        for cluster_play in plays
                    ],
                }
            )
        return json

    def _prepare_scatter_data_for_response(self, team_id: int, timeframe: int):
        logger.info(f"Prep scatter data: team {team_id}, timeframe {timeframe}")
        self._init_clusters(team_id)

        self._load_clusters_for_team(team_id)

        self.apply_user_updates(team_id, timeframe)

        return self._generate_scatter_data(team_id, timeframe)

    def _concat_plays_from_games(self, game_ids: list[str], games: pl.DataFrame) -> pl.DataFrame:
        if games.height == 0:
            return pl.DataFrame()

        game_dates = games.select(["game_id", "game_date"])

        plays_df = self.dataset_manager.get_plays_for_games(game_ids)
        # add quarter from moments
        plays_df = (
            plays_df.with_columns(
                [pl.col("moments").list.get(0).struct.field("quarter").alias("quarter")]
            )
            .drop("moments")
            .collect()
        )
        return plays_df.join(game_dates, on="game_id", how="left")

    def _generate_scatter_data(self, team_id, timeframe):
        games = self.dataset_manager.get_games_for_team(team_id, as_dicts=False)
        games = games.sort("game_date", descending=True).limit(timeframe)
        allowed_game_ids = games.select("game_id").to_series().to_list()

        cluster_plays = defaultdict(list)
        for cluster in self.clusters:
            for cluster_play in cluster.plays:
                if cluster_play.play_id.game_id in allowed_game_ids:
                    cluster_plays[cluster.id].append(cluster_play)

        play_dates = []
        for plays in cluster_plays.values():
            for cluster_play in plays:
                try:
                    date = datetime.strptime(cluster_play.play.game_date, "%Y-%m-%d").date()
                    play_dates.append(date)
                except Exception:
                    pass
        min_date = min(play_dates)
        max_date = max(play_dates)
        date_range = (max_date - min_date).days or 1

        embeddings = []
        y = []
        for plays in cluster_plays.values():
            for cluster_play in plays:
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
        for plays in cluster_plays.values():
            for cluster_play in plays:
                cluster_play.x = float(xys[idx, 0])
                cluster_play.y = float(xys[idx, 1])

                play_date = datetime.strptime(cluster_play.play.game_date, "%Y-%m-%d").date()
                days_since_oldest = (play_date - min_date).days
                recency = days_since_oldest / date_range

                cluster_play.recency = recency
                idx += 1
        return cluster_plays

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
        cluster_plays = self._prepare_scatter_data_for_response(team_id, last_games)
        data = self._clusters_to_dicts(cluster_plays)
        return data, 200


def create_partial_dataframe(data: dict, full_schema: dict) -> pl.DataFrame:
    # Only use schema fields that exist in the incoming data
    partial_schema = {k: v for k, v in full_schema.items() if k in data}
    return pl.DataFrame(data, schema=partial_schema)


class ScatterPointResource(Resource):
    def __init__(self):
        super().__init__()

    def put(self, team_id: str, game_id: str, event_id: str):
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

        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates/{team_id}.parquet")
        user_updates = user_updates.update(df_update, on=["game_id", "event_id"], how="full")
        user_updates.write_parquet(f"{DATA_DIR}/user_updates/{team_id}.parquet")

        return {"message": "Play updated successfully"}, 200


class ClusterResource(Resource):
    def __init__(self):
        super().__init__()

    def post(self, team_id: str, cluster_id: str):
        update_cluster_data = request.get_json()
        update_cluster_data["cluster_id"] = cluster_id

        try:
            df_update = create_partial_dataframe(update_cluster_data, UPDATE_PLAY_SCHEMA)
        except ValueError as err:
            logger.error(err)
            return {"error": "Invalid play ID format"}, 400
        print(df_update)
        user_updates = pl.read_parquet(f"{DATA_DIR}/user_updates/{team_id}.parquet")
        user_updates = user_updates.update(df_update, on=["cluster_id"], how="full")
        print(user_updates)
        user_updates.write_parquet(f"{DATA_DIR}/user_updates/{team_id}.parquet")

        return {"message": "Play updated successfully"}, 200


if __name__ == "__main__":
    resource = TeamPlaysScatterResource()
    resource._prepare_scatter_data_for_response(1610612755, 3)
