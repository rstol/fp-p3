from dataclasses import dataclass

import numpy as np

from backend.resources.play import Play
from backend.resources.playid import PlayId


@dataclass(slots=True)
class ClusterPlay:
    play_id: PlayId
    distance: float
    embedding: np.ndarray
    play: Play
    x: float | None = None
    y: float | None = None
    recency: float | None = None


@dataclass(slots=True)
class Cluster:
    """Representation of a play cluster."""

    id: str
    label: str
    centroid: np.ndarray | None
    plays: list[ClusterPlay]
    created: float
    last_modified: float
    created_by: str
    confidence: float | None = None  # 0-1 value representing cluster cohesion

    # TODO helper methods
    def sort_plays_by_cluster_distance(self):
        pass

    def update_centroids(self):
        pass

    def get_usage_percent(self):
        pass
