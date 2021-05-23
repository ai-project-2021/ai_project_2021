import numpy as np
from utils import get_product_recommend, get_rfm_data


def d_tool(X, y):
    return np.sum((X - y) ** 2) ** 0.5


class KMeans:
    def __init__(self, k):
        self.k = k
        self.f = np.mean

    def _update(self, datas, centriod):
        _clu = lambda r: np.array([np.sum((c - r) ** 2) ** 0.5 for c in centriod]).argmin()
        return np.array([[1 if _clu(row) == j else 0 for j in range(self.k)] for row in datas])

    def fit(self, datas):
        centriod = datas[np.random.choice(range(len(datas)), self.k)].astype("float32")

        prev_labels = np.zeros((len(datas)), dtype=int)

        while True:
            cluster = self._update(datas, centriod)
            labels = cluster.argmax(axis=1)

            if (labels == prev_labels).all():
                break

            prev_labels = labels.copy()
            centriod = [self.f(datas[np.where(cluster[:, p] == 1)], axis=0) for p in range(self.k)]

        self.inertia_ = (
            sum(
                [
                    d_tool(
                        datas[np.where(cluster[:, p] == 1)],
                        self.f(datas[np.where(cluster[:, p] == 1)], axis=0),
                    )
                    for p in range(self.k)
                ]
            )
            / datas.shape[0]
        )

        return centriod, labels


class KMedians(KMeans):
    def __init__(self, k):
        super().__init__(k=k)
        self.f = np.median

