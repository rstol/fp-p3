import time
import uuid
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch

from backend.resources.cluster import Cluster, ClusterPlay
from backend.resources.dataset_manager import DatasetManager
from backend.resources.play import Play
from backend.resources.playid import PlayId
from backend.settings import EMBEDDINGS_DIR, TEAM_IDS_SAMPLE


class PlayClustering:
    def __init__(self, team_id: int, game_ids: list[str], initial_k: int = 12) -> None:
        self.team_id: int = team_id
        self.game_ids: list[str] = game_ids
        self.index: faiss.Index | None = None
        self.kmeans: faiss.Kmeans | None = None
        self.clusters = []
        self.team_embeddings: np.ndarray | None = None
        self.embedding_ids = self._get_embedding_ids()
        self.d = 256
        self.initial_k = initial_k
        self._init()

    def _get_embedding_ids(self) -> pl.LazyFrame:
        embeddings_sources_df = pl.scan_csv(
            Path(EMBEDDINGS_DIR) / "embedding_sources.csv",
            schema={"game_id": pl.String, "event_id": pl.String, "offensive_team_id": pl.Int32},
        )
        if self.game_ids is not None:
            embeddings_sources_df = embeddings_sources_df.filter(
                pl.col("game_id").is_in(self.game_ids)
            )
        return embeddings_sources_df

    def _init(self) -> None:
        embeddings = self._load_all_embeddings()
        # self._build_index(embeddings)
        self._set_game_embeddings(embeddings)

    def _load_all_embeddings(self) -> np.ndarray:
        npy_files = list(Path(EMBEDDINGS_DIR).glob("*.npy"))
        embeddings_all = []
        for file_path in npy_files:
            embedding: list[torch.Tensor] = np.load(file_path, allow_pickle=True)
            embeddings_all.append([e.numpy() for e in embedding])
        all_embeddings = np.concatenate(embeddings_all, axis=0)
        faiss.normalize_L2(all_embeddings)
        return all_embeddings

    def _build_index(self, embeddings: np.ndarray) -> None:
        index = faiss.index_factory(self.d, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)
        assert index.is_trained
        self.index = index

    def _init_kmeans(self) -> None:
        clustering = faiss.Clustering(self.d, self.initial_k)
        clustering.niter = 25
        index = faiss.IndexFlatIP(self.d)
        clustering.train(self.game_embeddings, index)

        self.kmeans = clustering
        self.kmeans_index = index

    def get_games_embedding_ids(self) -> pl.LazyFrame:
        return self.embedding_ids.with_row_index().filter(
            pl.col("game_id").is_in(self.game_ids) & pl.col("offensive_team_id").eq(self.team_id)
        )

    def _set_game_embeddings(self, embeddings) -> None:
        self.game_embedding_ids = self.get_games_embedding_ids().collect()
        self.game_embedding_idxs = np.squeeze(self.game_embedding_ids.select(pl.col("index")))
        self.game_embeddings = embeddings[self.game_embedding_idxs]

    def get_initial_clusters(self, plays: pl.DataFrame) -> list[Cluster]:
        self._init_kmeans()
        distances, index = self.kmeans_index.search(self.game_embeddings, 1)
        distances = distances[..., 0]
        cluster_assignments = index[..., 0]
        centroids = faiss.vector_to_array(self.kmeans.centroids).reshape(self.initial_k, self.d)
        cluster_play_lists = self.merge_plays_embeddings(
            plays, self.game_embedding_ids, self.game_embeddings, distances, cluster_assignments
        )

        clusters = []
        for i in range(self.initial_k):
            timestamp = time.time()
            cluster = Cluster(
                id=str(uuid.uuid4()),
                label=f"Cluster {i + 1}",
                centroid=centroids[i],
                plays=cluster_play_lists[i],  # Each cluster gets its own list
                confidence=None,
                created=timestamp,
                last_modified=timestamp,
                created_by="system",
            )
            clusters.append(cluster)

        self.clusters = clusters

        return clusters

    def merge_plays_embeddings(
        self,
        plays: pl.DataFrame,
        embedding_ids: pl.DataFrame,
        embeddings,
        distances,
        cluster_assignments,
        num_clusters: int | None = None,
    ):
        plays = plays.join(
            embedding_ids.with_row_index("local_index"), on=["game_id", "event_id"], how="inner"
        ).sort("local_index")
        num_clusters = num_clusters or self.initial_k
        cluster_play_lists = [[] for _ in range(num_clusters)]
        for play, embedding, distance, cluster_assignment in zip(
            plays.iter_rows(named=True),
            embeddings[plays["local_index"]],
            distances[plays["local_index"]],
            cluster_assignments[plays["local_index"]],
            strict=True,
        ):
            play_id = PlayId(play["game_id"], play["event_id"])
            play = Play(
                play_id=play_id,
                event_type=play["event_type"],
                game_date=play["game_date"],
                quarter=play["quarter"],
                event_score=play["event_score"],
                event_score_margin=play["event_score_margin"],
                possession_team_id=0
                if np.isnan(play["possession_team_id"])
                else int(play["possession_team_id"]),
                event_desc_away=play["event_desc_away"],
                event_desc_home=play["event_desc_home"],
            )
            cluster_play = ClusterPlay(play_id, distance, embedding, play)
            cluster_play_lists[cluster_assignment].append(cluster_play)
        return cluster_play_lists

    def find_similar_plays(self, q_embedding: np.ndarray):
        self._build_index(self.game_embeddings)
        q = np.expand_dims(q_embedding, axis=0)
        distances, index = self.index.search(q, self.game_embeddings.shape[0])
        mask = (distances[0] > 0.95) & (distances[0] < 1)  # Radius: Cosine similarity threshold
        masked_indices = index[0][mask]
        masked_distances = distances[0][mask]

        # Handle limits
        min_limit, max_limit = 3, 10
        if len(masked_indices) < min_limit:
            final_indices = index[0][:min_limit]
            final_distances = distances[0][:min_limit]
        elif len(masked_indices) > max_limit:
            # Too many — keep top max_limit most similar in radius
            final_indices = masked_indices[:max_limit]
            final_distances = masked_distances[:max_limit]
        else:
            final_indices = masked_indices
            final_distances = masked_distances

        embedding_ids = self.game_embedding_ids[final_indices]
        embeddings = self.game_embeddings[final_indices]
        return embedding_ids, final_distances, embeddings


if __name__ == "__main__":
    """DEBUG"""
    team_id = TEAM_IDS_SAMPLE.pop()
    dataset_manager = DatasetManager()
    games = dataset_manager.get_games_for_team(str(team_id), as_dicts=False)
    games = games.sort("game_date", descending=True).limit(5)
    game_ids = games["game_id"].to_list()
    clustering = PlayClustering(team_id, game_ids)
    game_dates = games.select(["game_id", "game_date"])

    plays_df = dataset_manager.get_plays_for_games(game_ids)
    plays_df = (
        plays_df.with_columns(
            [pl.col("moments").list.get(0).struct.field("quarter").alias("quarter")]
        )
        .drop("moments")
        .collect()
    )
    plays_df = plays_df.join(game_dates, on="game_id", how="left")

    clusters = clustering.get_initial_clusters(plays_df)
    cluster = clusters[0]
    play = cluster.plays[0]
    clustering.find_similar_plays(play.embedding)
