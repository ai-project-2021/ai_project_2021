import numpy as np


def dist(X, y):
    return np.sum(np.power((X - y), 2), axis=1) ** 0.5


def inertia(features, labels, centriod):
    return np.sum([np.sum(np.power((x - centriod[k]), 2)) for x, k in zip(features, labels)])


def elbow(models, range_k, datas):
    WCSS = [0 for _ in range_k]
    model_ = models
    for i, k in enumerate(range_k):
        model_.k = k
        WCSS[i] = model_.fit(datas).inertia_
    return WCSS


def silhouette_score(models, range_k, datas):
    score = [0 for _ in range_k]
    model_ = models
    for i, k in enumerate(range_k):
        model_.k = k
        score[i] = get_silhouette(datas, model_.fit_predict(datas))
    return score[i]


def get_silhouette(X, labels):
    n = labels.shape[0]

    A = np.zeros((n))  # Intra Cluster
    B = np.zeros((n))  # Other Cluster

    for i, (v_X, v_y) in enumerate(zip(X, labels)):
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[np.where(labels == v_y)[0]] = True
        mask[i] = False
        A[i] = np.mean(dist(X[mask], v_X))
        B[i] = np.min(
            [np.mean(dist(X[np.where(labels == v)[0]], v_X)) for v in set(labels) if not v == v_y]
        )

    return np.mean(np.nan_to_num((B - A) / np.maximum(A, B)))