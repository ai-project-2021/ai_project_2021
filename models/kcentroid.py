import numpy as np
from utils import get_product_recommend, get_rfm_data
from utils.metrics import inertia
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


class KMeans:
    def __init__(self, k):
        self.k = k
        self.f = np.mean

    def _update(self):
        _clu = lambda c: np.array(np.sum((self.datas - c) ** 2, axis=1) ** 0.5)
        return np.eye(self.k)[np.array([_clu(c) for c in self.centriod_]).argmin(axis=0)]

    def fit(self, datas):
        self.datas = datas
        self.centriod_ = datas[np.random.choice(range(len(datas)), self.k)].astype("float32")

        prev_labels = np.zeros((len(datas)), dtype=int)

        while True:
            cluster = self._update()
            labels = cluster.argmax(axis=1)

            if (labels == prev_labels).all():
                break

            prev_labels = labels.copy()
            self.centriod_ = [
                self.f(datas[np.where(cluster[:, p] == 1)], axis=0) for p in range(self.k)
            ]

        self.labels_ = labels
        self.inertia_ = inertia(datas, labels, self.centriod_)
        return self

    def fit_predict(self, datas):
        return self.fit(datas).labels_


class KMedians(KMeans):
    def __init__(self, k):
        super().__init__(k=k)
        self.f = np.median

    def _update(self):
        _clu = lambda c: np.array(np.sum(abs(self.datas - c), axis=1))
        return np.eye(self.k)[np.array([_clu(c) for c in self.centriod_]).argmin(axis=0)]


class KMedoids:
    def __init__(self, k, max_iters=500):
        self.k = k
        self.max_iters = max_iters

    def _update(self, indices):
        return np.argmin(self.dist(self.datas[:, None, :], self.datas[None, indices, :]), axis=1)

    def dist(self, xa, xb):
        return np.sqrt(np.sum(np.square(xa - xb), axis=-1))

    def find_medoids(self, labels):
        medoids = np.full(self.k, -1, dtype=int)
        subset = np.random.choice(self.datas.shape[0], self.batch_size, replace=False)

        for i in range(self.k):
            indices = np.intersect1d(np.where(labels == i)[0], subset)
            distances = self.dist(self.datas[indices, None, :], self.datas[None, indices, :]).sum(
                axis=0
            )
            medoids[i] = indices[np.argmin(distances)]

        return medoids

    def fit(self, datas):
        self.datas = datas
        self.batch_size = datas.shape[0] // 10
        medoids = np.random.choice(self.datas.shape[0], self.k, replace=False)
        prev_labels = self._update(medoids)
        best_diffs = 1e10
        best_steps_ = -1

        for i in range(self.max_iters):
            medoids = self.find_medoids(prev_labels)
            labels = self._update(medoids)

            diffs = np.mean(labels != prev_labels)
            prev_labels = labels

            print("iteration {:2d}: {:.6%}p of points got reassigned." "".format(i, diffs))

            if diffs < 0.01:
                self.medoids_ = medoids
                self.labels_ = prev_labels
                break
            elif best_diffs > diffs:
                self.medoids_ = medoids
                self.labels_ = prev_labels
                best_steps_ = i
            elif i - best_steps_ > (self.max_iters // 10):
                break

        self.inertia_ = inertia(datas, self.labels_, self.medoids_)

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Centriod Arguments")
    parser.add_argument("--models", "-m", type=str, action="append", help="K-Centriod Method Type")
    parser.add_argument("--n_clusters", "-k", type=int, action="append")
    args = parser.parse_args()

    dataset = get_rfm_data(get_product_recommend())[["R_Value", "F_Value", "M_Value"]]
    model_dict = {"KMeans": KMeans, "KMedians": KMedians, "KMedoids": KMedoids}

    for model_ in args.models:
        s = time.time()
        elbow = []
        model = model_dict[model_](args.n_clusters[0])
        n_cols = len(args.n_clusters)
        fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

        for i, k in enumerate(args.n_clusters):
            e = time.time()
            model.k = k
            elbow.append(model.fit(dataset.values).inertia_)
            silhouette_avg = silhouette_score(dataset.values, model.labels_)
            print(
                "K : {}, Inertia : {}, Avg. Silhouette_Score : {}".format(
                    k, elbow[-1], silhouette_avg
                )
            )

        plt.figure()
        plt.plot(args.n_clusters, elbow)
        plt.savefig(f"./{model_}.png")
