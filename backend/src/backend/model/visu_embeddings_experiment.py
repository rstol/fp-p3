import glob
import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from backend.settings import EMBEDDINGS_DIR

# Assuming you have:
# - embeddings: numpy array of shape (~90000, 256)
# - play_ids: list of 90000 play identifiers


def create_umap_projection(embeddings, play_ids: list[str], y, n_neighbors=15, min_dist=0.1):
    """
    Create a 2D UMAP projection of high-dimensional embeddings

    Parameters:
    -----------
    embeddings : numpy.ndarray
        Array of shape (n_samples, n_features) with embedding vectors
    play_ids : list
        List of play identifiers corresponding to each embedding
    n_neighbors : int
        Number of neighbors to consider in local neighborhood (UMAP parameter)
    min_dist : float
        Minimum distance between points in low dimensional space (UMAP parameter)

    Returns:
    --------
    DataFrame with columns 'x', 'y', and 'play_id'
    """
    # Create and fit UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=0.5,
        verbose=True,
        low_memory=False,
        metric="cosine",  # Use cosine since you mentioned cosine similarity
    )

    # Get 2D embeddings
    embedding_2d = reducer.fit_transform(embeddings, y)

    # Create DataFrame with results
    result_df = pd.DataFrame(
        {
            "x": embedding_2d[:, 0],
            "y": embedding_2d[:, 1],
            "play_id": play_ids,
        }
    )

    return result_df, reducer


def cluster_plays(projection_df, method="dbscan", **kwargs):
    # Get the 2D points
    points = projection_df[["x", "y"]].values

    # Apply clustering
    if method.lower() == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 10)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)

    elif method.lower() == "kmeans":
        n_clusters = kwargs.get("n_clusters", 8)

        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clustering.fit_predict(points)

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Add cluster labels to the DataFrame
    result_df = projection_df.copy()
    result_df["cluster"] = labels

    return result_df


def load_embeddings_and_play_ids(team_ids: set[str]):
    embedding_detail = pd.read_csv(os.path.join(EMBEDDINGS_DIR, "embedding_sources.csv"))
    embedding_ids = (
        embedding_detail["Eventid"]
        .str.split("/")
        .str[-1]
        .str.removesuffix(".pkl")
        .str.split("_", expand=True)
    )
    embedding_ids_teams = (
        embedding_ids[embedding_ids[2].isin(team_ids)].iloc[:, :2].agg("_".join, axis=1)
    )
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
    embeddings_teams = embeddings_all[embedding_ids_teams.index]
    return embeddings_all, embeddings_teams, embedding_ids_teams.to_list()


def find_similar_plays(
    play_id,
    projection_df: pd.DataFrame,
    embeddings: list[np.ndarray],
    n=10,
    use_projected=False,
):
    """Find similar plays to a given play ID"""
    # Find the index of the target play
    idx = projection_df[projection_df["play_id"] == play_id].index[0]

    if use_projected:
        # Use 2D coordinates from projection
        target = projection_df.loc[idx, ["x", "y"]].values.reshape(1, -1)

        # Calculate Euclidean distances in 2D space
        all_points = projection_df[["x", "y"]].values
        distances = np.sqrt(np.sum((all_points - target) ** 2, axis=1))

        # Create pairs of (play_id, distance)
        result = list(zip(projection_df["play_id"].values, distances, strict=False))

        # Remove the target play
        result = [(pid, dist) for pid, dist in result if pid != play_id]

    else:
        # Use original high-dimensional embeddings
        target = embeddings[idx].reshape(1, -1)

        # Calculate cosine similarities - higher means more similar
        similarities = cosine_similarity(target, embeddings)[0]

        # Create pairs of (play_id, similarity)
        result = list(zip(projection_df["play_id"].values, similarities, strict=False))

        # Remove the target play and convert similarity to distance (1-sim)
        result = [(pid, 1 - sim) for pid, sim in result if pid != play_id]

    # Sort by distance (ascending)
    result.sort(key=lambda x: x[1])

    # Return the n most similar plays
    return [play_id for play_id, _ in result[:n]]


TEAM_IDS_SAMPLE = {
    "1610612741",
    # "1610612748",
    # "1610612752",
    # "1610612754",
    # "1610612755",
    # "1610612761",
    # "1610612766",
}


if __name__ == "__main__":
    try:
        embeddings_all, embeddings_team, play_ids_team = load_embeddings_and_play_ids(
            TEAM_IDS_SAMPLE
        )
        d = embeddings_all.shape[1]
        index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        print(index.ntotal)
        faiss.normalize_L2(embeddings_all)
        index.add(embeddings_all)
        print(index.is_trained)
        target_play_idx = np.random.choice(len(play_ids_team) - 1, 1)[0]
        k = 10
        print("Target index: ", target_play_idx)
        q = np.expand_dims(embeddings_team[target_play_idx], axis=0)
        faiss.normalize_L2(q)
        distance, index = index.search(q, k)  # actual search
        print(distance, index)  # Distance index of top k neighbors to query

        ncentroids = 12
        niter = 20
        verbose = True
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(embeddings_all)
        kdistances, k_I = kmeans.index.search(
            embeddings_team, 1
        )  # assign each vector to cluster centroid
        print(kmeans.centroids)
        k_I_flat = [i[0] for i in k_I]
        print(k_I_flat, kdistances)

        print("Creating UMAP projection...")
        projection_df, umap_model = create_umap_projection(
            embeddings_team, play_ids_team, k_I_flat, min_dist=0.05, n_neighbors=30
        )

        print("Clustering plays...")
        clustered_df = cluster_plays(projection_df, method="kmeans", eps=0.5, min_samples=10)

        print(f"Number of clusters found: {clustered_df['cluster'].nunique()}")

        target_play = play_ids_team[target_play_idx]
        # Example: Find similar plays to the first play
        # similar_plays = find_similar_plays(target_play, projection_df, embeddings, n=10)
        # print(f"Plays similar to {target_play}: {similar_plays}")

        points_to_draw = projection_df[
            projection_df["play_id"].isin(pd.Series(play_ids_team)[index[0]])
        ][["x", "y"]]

        # Plot the results
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            clustered_df["x"],
            clustered_df["y"],
            c=clustered_df["cluster"],
            cmap="tab20",
            s=5,
            alpha=0.8,
        )
        plt.scatter(
            points_to_draw["x"],
            points_to_draw["y"],
            c="red",
            s=10,
            label="Similar Plays",
            edgecolor="black",
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title("Clustered Basketball Plays")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
