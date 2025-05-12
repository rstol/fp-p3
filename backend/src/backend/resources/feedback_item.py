from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FeedbackItem:
    """Representation of scout feedback on play clustering."""

    play_id: str
    from_cluster_id: str | None
    to_cluster_id: str | None
    new_cluster_label: str | None
    timestamp: float
    scout_id: str
