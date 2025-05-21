import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch

from backend.resources.cluster import Cluster, ClusterPlay
from backend.resources.feedback_item import FeedbackItem
from backend.resources.playid import PlayId
from backend.settings import EMBEDDINGS_DIR, TEAM_IDS_SAMPLE


class PlayClustering:
    def __init__(
        self, team_id: int, game_ids: list[str] | None = None, initial_k: int = 12
    ) -> None:
        self.team_id = team_id
        self.game_ids = game_ids
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
        self._build_index(embeddings)
        self._init_kmeans(embeddings)
        self._set_team_embeddings(embeddings)

    def _load_all_embeddings(self) -> np.ndarray:
        npy_files = list(Path(EMBEDDINGS_DIR).glob("*.npy"))
        embeddings_all = []
        for file_path in npy_files:
            embedding: list[torch.Tensor] = np.load(file_path, allow_pickle=True)
            embeddings_all.append([e.numpy() for e in embedding])
        return np.concatenate(embeddings_all, axis=0)

    def _build_index(self, embeddings: np.ndarray) -> None:
        index = faiss.index_factory(self.d, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        assert index.is_trained
        self.index = index

    def _init_kmeans(self, embeddings: np.ndarray) -> None:
        kmeans = faiss.Kmeans(self.d, self.initial_k, niter=20)
        faiss.normalize_L2(embeddings)
        kmeans.train(embeddings)
        self.kmeans = kmeans

    def get_team_embedding_ids(self) -> pl.LazyFrame:
        return self.embedding_ids.with_row_index().filter(
            pl.col("offensive_team_id").eq(self.team_id)
        )

    def _set_team_embeddings(self, embeddings: np.ndarray) -> None:
        self.team_embedding_ids = self.get_team_embedding_ids().collect()
        self.team_embeddings_idx = (
            self.team_embedding_ids.select(pl.col("index")).to_series().to_numpy()
        )
        self.team_embeddings = embeddings[self.team_embeddings_idx]

    def get_initial_clusters(
        self, initial_k: int = 12, niter: int = 20
    ) -> tuple[list[Cluster], np.ndarray, pl.DataFrame]:
        distances, index = self.kmeans.index.search(self.team_embeddings, 1)
        distances = distances[..., 0]
        cluster_assignments = index[..., 0]
        centroids = self.kmeans.centroids

        distance_indices = distances.argsort()
        points_min_distance_indices = distance_indices[
            (cluster_assignments[distance_indices][None] == np.arange(initial_k)[:, None]).argmax(
                axis=1
            )
        ]

        plays = self.team_embedding_ids[points_min_distance_indices]
        min_distances = distances[points_min_distance_indices]

        cluster_plays = [
            ClusterPlay(PlayId(play["game_id"], play["event_id"]), distance, index)
            for play, distance, index in zip(
                plays.to_dicts(), min_distances, points_min_distance_indices, strict=True
            )
        ]
        clusters = []
        for i, cluster_play in enumerate(cluster_plays):
            timestamp = time.time()
            cluster = Cluster(
                id=str(i),
                label=f"Cluster {i + 1}",
                centroid=centroids[i],
                plays=[cluster_play],
                confidence=None,
                created=timestamp,
                last_modified=timestamp,
                created_by="system",
            )
            clusters.append(cluster)

        self.clusters = clusters

        return clusters, cluster_assignments, self.team_embedding_ids

    def find_similar_plays(self, k: int = 10) -> list[int]:
        q = np.expand_dims(embeddings_team[target_play_idx], axis=0)
        faiss.normalize_L2(q)
        distance, index = index.search(q, k)  # actual search
        print(distance, index)  # Distance index of top k neighbors to query
        # TODO

    def apply_scout_feedback(self, feedback: list[FeedbackItem]) -> list[Cluster]:
        """Apply feedback from scouts to refine clusters.

        Args:
            feedback: List of feedback items

        Returns:
            Updated list of clusters
        """
        updated_clusters = self.clusters.copy()
        new_clusters = []

        # Handle creation of new clusters
        new_cluster_feedback = [f for f in feedback if f.to_cluster_id is None]
        new_clusters_by_label = {}

        for fb in new_cluster_feedback:
            label = fb.new_cluster_label
            if label not in new_clusters_by_label:
                new_clusters_by_label[label] = []
            new_clusters_by_label[label].append(fb.play_id)

        # TODO

        # Create new clusters
        for label in new_clusters_by_label:
            play_embeddings = None
            faiss.normalize_L2(play_embeddings)

            centroid = np.mean(play_embeddings, axis=0)
            faiss.normalize_L2(centroid.reshape(1, -1))
            plays = []
            new_clusters.append(
                Cluster(
                    id=f"cluster-new-{int(time.time())}-{np.random.randint(1000, 9999)}",
                    label=label,
                    centroid=centroid,
                    plays=plays,
                    created=time.time(),
                    last_modified=time.time(),
                    created_by=feedback[0].scout_id,
                )
            )
        # Handle moves between existing clusters
        move_feedback = [f for f in feedback if f.to_cluster_id is not None]

        for fb in move_feedback:
            # Remove from source cluster if assigned
            if fb.from_cluster_id:
                for i, cluster in enumerate(updated_clusters):
                    if cluster.id == fb.from_cluster_id and fb.play_id in cluster.play_ids:
                        updated_play_ids = [pid for pid in cluster.play_ids if pid != fb.play_id]
                        updated_clusters[i] = Cluster(
                            id=cluster.id,
                            label=cluster.label,
                            centroid=cluster.centroid,  # Centroid will be recalculated later
                            plays=updated_play_ids,
                            created=cluster.created,
                            last_modified=time.time(),
                            created_by=cluster.created_by,
                        )

            # Add to target cluster
            for i, cluster in enumerate(updated_clusters):
                if cluster.id == fb.to_cluster_id:
                    updated_play_ids = [*cluster.play_ids, fb.play_id]
                    updated_clusters[i] = Cluster(
                        id=cluster.id,
                        label=cluster.label,
                        centroid=cluster.centroid,  # Will be recalculated
                        plays=updated_play_ids,
                        created=cluster.created,
                        last_modified=time.time(),
                        created_by=cluster.created_by,
                    )

        # Recalculate centroids and confidence for all modified clusters
        final_clusters = []

        for cluster in updated_clusters + new_clusters:
            if not cluster.play_ids:
                continue  # Skip empty clusters

            play_embeddings = np.array(
                [self.plays_by_id[pid].embedding for pid in cluster.play_ids], dtype=np.float32
            )
            faiss.normalize_L2(play_embeddings)

            # Recalculate centroid
            centroid = np.mean(play_embeddings, axis=0)
            faiss.normalize_L2(centroid.reshape(1, -1))

            final_clusters.append(
                Cluster(
                    id=cluster.id,
                    label=cluster.label,
                    centroid=centroid,
                    plays=cluster.plays,
                    created=cluster.created,
                    last_modified=cluster.last_modified,
                    created_by=cluster.created_by,
                )
            )

        self.clusters = final_clusters
        return final_clusters


if __name__ == "__main__":
    clustering = PlayClustering(TEAM_IDS_SAMPLE.pop())
    clustering.get_initial_clusters()
