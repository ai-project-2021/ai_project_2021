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


class KMedoids:
    def __init__(self, k, max_iters):
        self.k = k
        self.max_iters = max_iters

    def assign_nearest(self, ids_of_mediods):
        return np.argmin(self.dist(self.x[:, None, :], self.x[None, ids_of_mediods, :]), axis=1)

    def dist(self, xa, xb):
        return np.sqrt(np.sum(np.square(xa - xb), axis=-1))

    def find_medoids(self, assignments):
        medoid_ids = np.full(self.k, -1, dtype=int)
        subset = np.random.choice(self.x.shape[0], self.batch_size, replace=False)

        for i in range(self.k):
            indices = np.intersect1d(np.where(assignments == i)[0], subset)
            distances = self.dist(self.x[indices, None, :], self.x[None, indices, :]).sum(axis=0)
            medoid_ids[i] = indices[np.argmin(distances)]

        return medoid_ids

    def fit(self, x):
        self.x = x
        self.batch_size = x.shape[0] // 10
        ids_of_medoids = np.random.choice(self.x.shape[0], self.k, replace=False)
        class_assignments = self.assign_nearest(ids_of_medoids)

        for i in range(self.max_iters):
            ids_of_medoids = self.find_medoids(class_assignments)
            new_class_assignments = self.assign_nearest(ids_of_medoids)

            diffs = np.mean(new_class_assignments != class_assignments)
            class_assignments = new_class_assignments

            print("iteration {:2d}: {:.6%}p of points got reassigned." "".format(i, diffs))

            if diffs < 0.01:
                break

        return class_assignments, ids_of_medoids


if __name__ == "__main__":
    # model = Kmeans(k=3)
    dataset = get_rfm_data(get_product_recommend())[["R_Value", "F_Value", "M_Value"]]
    kmedians, kmedians_assignments = KMedians(k=3).fit(dataset.values)
    model = KMedoids(k=3, max_iters=100)

    # print("\nFitting Kmedoids.")
    final_assignments, final_medoid_ids = model.fit(dataset.values)

    # kmeans, kmeans_assignments = KMeans(k=3).fit(dataset.values)

    # mismatch = np.zeros((3, 3))
    # for i, m in zip(kmedians_assignments, kmeans_assignments):
    #     mismatch[i, m] += 1
    # print(mismatch)
    # clu, c, labels = model.fit(dataset.values)
    # print(model.inertia_)

    # fig = px.scatter_3d(dataset, x="R_Value", y="F_Value", z="M_Value", color=labels)

    # fig.show()