import glob
import os
import time

import faiss
import numpy as np
import pandas as pd
import torch

from backend.resources.Cluster import Cluster, ClusterPlay
from backend.resources.FeedbackItem import FeedbackItem
from backend.resources.PlayId import PlayId

# import polars as pl
from backend.settings import EMBEDDINGS_DIR, TEAM_IDS_SAMPLE


class PlayClustering:
    def __init__(self, team_id: str, initial_k=12):
        self.team_id = team_id
        self.index = None  # FAISS index
        self.kmeans = None  # FAISS kMeans to init clusters
        self.clusters = []
        self.team_embeddings: np.ndarray = None
        self.embedding_ids = self._get_embedding_ids()
        self.d = 256
        self.initial_k = initial_k
        self._init()  # TODO lazy?

    def _get_embedding_ids(self):
        return pd.read_csv(os.path.join(EMBEDDINGS_DIR, "embedding_sources.csv"), dtype="str")

    def _init(self):
        embeddings = self._load_all_embeddings()
        self._build_index(embeddings)
        self._init_kmeans(embeddings)
        self.team_embeddings = self._set_team_embeddings(embeddings)

    def _load_all_embeddings(self):
        npy_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.npy"))
        embeddings_all = None
        for file_path in npy_files:
            embedding: list[torch.Tensor] = np.load(file_path, allow_pickle=True)
            if embeddings_all is None:
                embeddings_all = [e.numpy() for e in embedding]
            else:
                embeddings_all = np.concatenate(
                    (embeddings_all, [e.numpy() for e in embedding]), axis=0
                )
        return embeddings_all

    def _build_index(self, embeddings):
        index = faiss.index_factory(self.d, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)  # Normalise in place
        index.add(embeddings)
        assert index.is_trained
        self.index = index

    def _init_kmeans(self, embeddings):
        kmeans = faiss.Kmeans(self.d, self.initial_k, niter=20)
        faiss.normalize_L2(embeddings)
        kmeans.train(embeddings)
        self.kmeans = kmeans

    def get_team_embedding_ids(self):
        return self.embedding_ids[self.embedding_ids["offensive_team_id"].isin([self.team_id])]

    def _set_team_embeddings(self, embeddings):
        index = self.get_team_embedding_ids().index
        return embeddings[index]

    def get_initial_clusters(self, initial_k=12, niter=20):
        distances, index = self.kmeans.index.search(
            self.team_embeddings, 1
        )  # assign each embedding to cluster centroid
        distances = distances.reshape(-1)
        cluster_assignments = index.reshape(-1)
        centroids = self.kmeans.centroids
        clusters = []
        timestamp = time.time()
        play_ids = self.get_team_embedding_ids()
        play_ids_reset_idx = play_ids.reset_index(drop=True)

        for i in range(initial_k):
            cluster_idxs = [
                idx for idx, cluster_idx in enumerate(cluster_assignments) if cluster_idx == i
            ]
            cluster_play_ids = play_ids_reset_idx.iloc[cluster_idxs].reset_index(drop=True)

            # Add play closest to centroid
            subset_distance = distances[cluster_idxs]
            centroid_idx = np.argmin(subset_distance)
            play = cluster_play_ids.iloc[centroid_idx]
            cluster_plays = [
                ClusterPlay(
                    PlayId(play["game_id"], play["event_id"]), subset_distance[centroid_idx]
                )
            ]

            if not cluster_plays:
                continue

            # cluster_embeddings = self.team_embeddings[clusted_idxs]

            clusters.append(
                Cluster(
                    id=f"cluster-{i}",
                    label=f"Cluster {i + 1}",  # Default name
                    centroid=centroids[i],
                    plays=cluster_plays,
                    confidence=None,
                    created=timestamp,
                    last_modified=timestamp,
                    created_by="system",
                )
            )

        self.clusters = clusters

    def find_similar_plays(self, k=10):
        q = np.expand_dims(embeddings_team[target_play_idx], axis=0)
        faiss.normalize_L2(q)
        distance, index = index.search(q, k)  # actual search
        print(distance, index)  # Distance index of top k neighbors to query
        # TODO

    def apply_scout_feedback(self, feedback: list[FeedbackItem]):
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

        # Create new clusters
        for label, play_ids in new_clusters_by_label.items():
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
                    updated_play_ids = cluster.play_ids + [fb.play_id]
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
