import numpy as np
from utils import get_product_recommend, get_rfm_data
from utils.metrics import inertia


class KMeans:
    def __init__(self, k):
        self.k = k
        self.f = np.mean

    def _update(self):
        _clu = lambda r: np.array([np.sum((c - r) ** 2) ** 0.5 for c in self.centriod_]).argmin()
        return np.array([[1 if _clu(row) == j else 0 for j in range(self.k)] for row in self.datas])

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


class KMedoids:
    def __init__(self, k, max_iters):
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

        for i in range(self.max_iters):
            medoids = self.find_medoids(prev_labels)
            labels = self._update(medoids)

            diffs = np.mean(labels != prev_labels)
            prev_labels = labels

            print("iteration {:2d}: {:.6%}p of points got reassigned." "".format(i, diffs))

            if diffs < 0.01:
                break

        self.medoids_ = medoids
        self.labels_ = prev_labels
        self.inertia_ = inertia(datas, self.labels_, self.medoids_)

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


if __name__ == "__main__":
    dataset = get_rfm_data(get_product_recommend())[["R_Value", "F_Value", "M_Value"]]