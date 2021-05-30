import numpy as np
from scipy.spatial import distance_matrix
import time


def dist(X, y):
    """Eucildian Distance = L_2 Distance

    Args:
        X (np.ndarray): np.ndarray
        y (np.array): np.array

    Returns:
        [np.array]: Eucildian Distance Array
    """
    return np.sum(np.power((X - y), 2), axis=1) ** 0.5


def inertia(features, labels, centriod):
    return np.sum(np.power((features - centriod[labels]), 2))


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

    distance = distance_matrix(X, X)
    label2Idx = {l: np.where(labels == l)[0] for l in np.unique(labels)}
    distance[np.where(np.eye(n) == 1)] = 0
    labels_ = set(labels)

    for i, v in enumerate(labels):
        A[i] = np.sum(distance[i, label2Idx[v]]) / (label2Idx[v].shape[0] - 1)
        B[i] = np.min([np.mean(distance[i, label2Idx[v_]]) for v_ in labels_ if not v_ == v])

    return np.mean(np.nan_to_num((B - A) / np.maximum(A, B)))