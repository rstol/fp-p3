from dataclasses import dataclass

import numpy as np

from backend.resources.playid import PlayId


@dataclass(slots=True, frozen=True)
class ClusterPlay:
    play_id: PlayId
    distance: float
    index: int | None = None


@dataclass(slots=True, frozen=True)
class Cluster:
    """Representation of a play cluster."""

    id: str
    label: str
    centroid: np.ndarray
    plays: list[ClusterPlay]
    created: float
    last_modified: float
    created_by: str
    confidence: float = 1.0  # 0-1 value representing cluster cohesion

    # TODO helper methods
    def sort_plays_by_cluster_distance(self):
        pass

    def get_umap(self):
        pass

    def update_centroids(self):
        pass

    def update_label(self):
        pass

    def get_usage_percent(self):
        pass
