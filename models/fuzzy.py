import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples, silhouette_score
import argparse
from tqdm import trange
import time

from utils import get_rfm_data
from utils.metrics import inertia
from matplotlib import pyplot as plt


def d_tool(X, y):
    return np.sum((X - y) ** 2) ** 0.5


class FuzzyKMeans:
    def __init__(self, k, m=2, max_iter=100, random_state=0, threshold=1e-4):
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.threshold = threshold

    def _e_step(self, X):
        D = (1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)) ** (
            1.0 / (self.m - 1)
        )

        # shape: n_samples x k
        self.fuzzy_labels_ = D / np.sum(D, axis=1)[:, np.newaxis]
        self.labels_ = self.fuzzy_labels_.argmax(axis=1)

    def _m_step(self, X):
        # shape: n_clusters x n_features
        weights = self.fuzzy_labels_ ** self.m
        self.cluster_centers_ = np.dot(X.T, weights).T / weights.sum(axis=0)[:, np.newaxis]

    def _average(self, X):
        return X.mean(axis=0)

    def fit(self, X, y=None):
        n_samples, _ = X.shape
        vdata = np.mean(np.var(X, 0))

        self.fuzzy_labels_ = check_random_state(self.random_state).rand(n_samples, self.k)
        self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1)[:, np.newaxis]
        self._m_step(X)

        for _ in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.threshold * vdata:
                break

        self.centroid = self.cluster_centers_
        self.labels_ = self.fuzzy_labels_.argmax(axis=1)
        self.inertia_ = inertia(X, self.labels_, self.cluster_centers_)

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Centriod Arguments")
    parser.add_argument("--models", "-m", type=str, action="append", help="K-Centriod Method Type")
    parser.add_argument("--n_clusters", "-k", type=int, action="append")
    args = parser.parse_args()

    dataset = get_rfm_data()[["R_Value", "F_Value", "M_Value"]]
    model = FuzzyKMeans(k=3, m=2, max_iter=100, random_state=0, threshold=1e-6)

    s = time.time()
    elbow = []
    n_cols = len(args.n_clusters)
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    for i, k in enumerate(args.n_clusters):
        e = time.time()
        model.k = k
        elbow.append(model.fit(dataset.values).inertia_)
        silhouette_avg = silhouette_score(dataset.values, model.labels_)
        print(
            "K : {}, Inertia : {}, Avg. Silhouette_Score : {}".format(k, elbow[-1], silhouette_avg)
        )

    plt.figure()
    plt.plot(args.n_clusters, elbow)
    plt.savefig(f"./fuzzy.png")