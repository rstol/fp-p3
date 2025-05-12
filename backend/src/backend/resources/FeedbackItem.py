from dataclasses import dataclass


@dataclass
class FeedbackItem:
    """Representation of scout feedback on play clustering."""

    play_id: str
    from_cluster_id: str | None  # None if unassigned
    to_cluster_id: str | None  # None if creating new cluster
    new_cluster_label: str | None  # Required if to_cluster_id is None
    timestamp: float
    scout_id: str
