# https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
import torch


def cop_kmeans(
    dataset: torch.Tensor,
    k: int,
    ml: list[tuple[int, int]] | None = None,
    cl: list[tuple[int, int]] | None = None,
    initialization: str = "kmpp",
    max_iter: int = 300,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if ml is None:
        ml = []
    if cl is None:
        cl = []
    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)

    centers = initialize_centers(dataset, k, initialization)

    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)
        indices_dataset = torch.cdist(dataset, centers).argsort(dim=-1)
        for i in range(len(dataset)):
            indices = indices_dataset[i]
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1
                if not found_cluster:
                    return None, None

        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info)
        shift = torch.cdist(centers, centers_, p=2).sum()
        if shift <= tol:
            break

        centers = centers_

    return clusters_, centers_


def tolerance(tol: float, dataset: torch.Tensor) -> float:
    return tol * dataset.var(dim=0).mean()


def initialize_centers(dataset: torch.Tensor, k: int, method: str) -> torch.Tensor:
    if method == "random":
        ids = torch.randperm(len(dataset))
        return dataset[ids[:k]]

    centers = torch.empty((k, dataset.shape[1]), dtype=dataset.dtype)
    first_idx = torch.randint(0, len(dataset), (1,))
    centers[0] = dataset[first_idx]

    for i in range(1, k):
        distances = torch.cdist(dataset, centers[:i]).amin(dim=1)
        probs = distances / distances.sum()
        next_idx = torch.multinomial(probs, 1)
        centers[i] = dataset[next_idx]

    return centers


def violate_constraints(
    data_index: int,
    cluster_index: int,
    clusters: list[int],
    ml: dict[int, set[int]],
    cl: dict[int, set[int]],
) -> bool:
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True
    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True
    return False


def compute_centers(
    clusters: list[int],
    dataset: torch.Tensor,
    k: int,
    ml_info: tuple[list[list[int]], list[float], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_ids = torch.unique(torch.as_tensor(clusters))
    k_new = len(cluster_ids)
    id_map = {int(cluster_ids[i]): i for i in range(k_new)}
    clusters_tensor = torch.as_tensor([id_map[int(x)] for x in clusters])
    centers = torch.zeros((k, dataset.shape[1]), dtype=dataset.dtype)

    for c in range(k_new):
        mask = clusters_tensor == c
        if mask.sum() > 0:
            centers[c] = dataset[mask].mean(dim=0)

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info

        current_scores = torch.tensor(
            [
                (
                    (
                        centers[clusters_tensor[torch.tensor(group)]] - dataset[torch.tensor(group)]
                    ).norm(dim=1)
                    ** 2
                ).sum()
                for group in ml_groups
            ]
        )

        ml_scores_tensor = torch.tensor(ml_scores)
        score_diff = current_scores - ml_scores_tensor
        group_ids = torch.argsort(score_diff, descending=True)

        for j in range(k - k_new):
            gid = int(group_ids[j])
            cid = k_new + j
            centers[cid] = torch.tensor(ml_centroids[gid], dtype=centers.dtype)
            for i in ml_groups[gid]:
                clusters_tensor[i] = cid

    return clusters_tensor, centers


def get_ml_info(
    ml: dict[int, set[int]], dataset: torch.Tensor
) -> tuple[list[list[int]], list[float], torch.Tensor]:
    flags = torch.ones(len(dataset), dtype=torch.bool)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]:
            continue
        group = torch.as_tensor(list(ml[i] | {i}))
        groups.append(group)
        flags[group] = False

    scores = torch.zeros(len(groups))
    centroids = torch.zeros((len(groups), dataset.shape[1]))

    for j, group in enumerate(groups):
        centroids[j] = dataset[group].mean(dim=0)

    for j, group in enumerate(groups):
        scores[j] = torch.cdist(dataset[group], centroids[j][None]).sum()

    return groups, scores, centroids


def transitive_closure(
    ml: list[tuple[int, int]], cl: list[tuple[int, int]], n: int
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    ml_graph = {i: set() for i in range(n)}
    cl_graph = {i: set() for i in range(n)}

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for i, j in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for i, j in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception("inconsistent constraints between %d and %d" % (i, j))

    return dict(ml_graph), dict(cl_graph)
