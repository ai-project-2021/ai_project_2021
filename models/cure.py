# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm, trange
import pickle as pkl
from numba import njit

from utils import get_rfm_data
from utils.metrics import inertia
from sklearn.metrics import silhouette_score


DEBUG = True


@njit
def col_dist(mat_, vec_):
    return np.sum((mat_ - vec_) ** 2, axis=1) ** 0.5


@njit
def col_dist_max(mat_, vec_):
    return np.max(np.sum((mat_ - vec_) ** 2, axis=1) ** 0.5)


@njit
def mat_dist_max(mat_x, mat_y):
    return np.sqrt(np.sum(np.square(mat_x - mat_y), axis=-1)).max()


# This class describes the data structure and method of operation for CURE clustering.
class Cluster:
    def __init__(self, id__, center__):
        # center == 1-d array
        self.points = center__  # 2d Array(최종으로) == Cluster 내 Records, n_f
        self.repPoints = center__
        self.center = center__  # 중점.
        self.index = [id__]

    def __repr__(self):
        return "Cluster " + " Size: " + str(len(self.points))

    def _updateCentroid(self, clu):
        w1 = len(self.index)
        w2 = len(clu.index)
        self.center = (self.center * w1 + clu.center * w2) / (w1 + w2)  # 중점

    # Computes and stores representative points for this cluster
    def _updateRepPoints(self, numRepPoints, alpha, distance_):
        idx_ = np.zeros((numRepPoints), dtype=int)
        idx_[0] = distance_[0].argmax()
        idx_[1] = distance_[idx_[0]].argmax()

        for i in range(2, numRepPoints):
            idx_[i] = distance_[idx_[:i]].sum(axis=1).argmax()

        self.repPoints = self.points[idx_] + alpha * (self.center - self.points[idx_])

    # Computes and stores distance between this cluster and the other one.
    def distRep(self, clu):
        if type(clu.repPoints[0]) != list:
            return col_dist_max(self.repPoints, clu.repPoints)
        else:
            return mat_dist_max(clu.repPoints[:, None, :], self.repPoints[None, :, :])

    # Merges this cluster with the given cluster, recomputing the centroid and the representative points.
    def merge(self, clu, numRepPoints, alpha, dist_):
        self._updateCentroid(clu)
        self.points = np.vstack((self.points, clu.points))
        self.index = np.append(self.index, clu.index)
        self._updateRepPoints(numRepPoints, alpha, dist_)


class CURE:
    def __init__(self, numRepPoints, alpha, k):
        self.numRepPoints = numRepPoints
        self.alpha = alpha
        self.k = k

    def fit(self, data):
        print("START FIT")

        # # Initialization
        numPts = len(data)
        self.data = data

        Clusters = [Cluster(idPoint, data[idPoint, :]) for idPoint in range(len(data))]

        dist_mat = np.vstack([col_dist(data, data[i, :]) for i in range(len(data))])
        cluster_distance = np.array(np.triu(dist_mat, 1), dtype="float")
        cluster_distance[cluster_distance == 0] = float("inf")
        indexes = list(range(numPts))

        for _ in trange(numPts, self.k, -1, desc="Update Centroids"):
            # Find a pair of closet clusters

            @njit
            def find_idx():
                closet_ = np.where(cluster_distance == np.min(cluster_distance))
                return int(closet_[0][0]), int(closet_[1][0])

            clu_src, clu_dst = find_idx()
            src = indexes.index(clu_src)

            # Merge
            idx_ = np.append(Clusters[clu_src].index, Clusters[clu_dst].index)

            Clusters[clu_src].merge(
                Clusters[clu_dst], self.numRepPoints, self.alpha, dist_mat[idx_][:, idx_]
            )

            dist_f = col_dist_max if type(Clusters[clu_src].repPoints[0]) != list else mat_dist_max

            # Update the distCluster matrix
            for i in indexes[:src]:
                cluster_distance[clu_src, i] = dist_f(
                    Clusters[clu_src].repPoints, Clusters[i].repPoints
                )
            for i in indexes[src + 1 :]:  # range(src + 1, numCluster):
                cluster_distance[i, clu_src] = dist_f(
                    Clusters[clu_src].repPoints, Clusters[i].repPoints
                )

            # Delete the merged cluster and its disCluster vector.
            cluster_distance[clu_dst, :] = float("inf")
            cluster_distance[:, clu_dst] = float("inf")
            indexes.remove(clu_dst)

        self.cluster_distance = cluster_distance[indexes][:, indexes]
        self.clusters = [Clusters[i] for i in sorted(list(indexes))]
        self.cluster_centers_ = np.array([c.center for c in Clusters])
        self.indices = [c.index[0] for c in self.clusters]

        # Generate sample labels
        self.labels_ = np.zeros((numPts))
        for c in self.clusters:
            self.labels_[c.index] = c.index[0]

        label2Idx = {c.index[0]: i for i, c in enumerate(self.clusters)}
        self.inertia_ = inertia(
            self.data,
            np.array([label2Idx[v] for v in self.labels_]),
            self.cluster_centers_,
        )

        with open("./cure_5_0.1_20.pkl", "wb") as f:
            pkl.dump(self, f)
        return self

    def fit_predict(self, data):
        return self.fit(data).labels_

    def fit_restart(self, k):
        assert k < self.k
        self.k = k

        Clusters = self.clusters

        cluster_distance = self.cluster_distance

        dist_mat = np.vstack([col_dist(self.data, self.data[i, :]) for i in range(len(self.data))])

        for cur_k in trange(len(self.indices), self.k, -1):
            # Find a pair of closet clusters

            @njit
            def find_idx():
                closet_ = np.where(cluster_distance == np.min(cluster_distance))
                return int(closet_[0][0]), int(closet_[1][0])

            src, dst = find_idx()
            print(src, dst)

            # Merge
            idx_ = np.append(Clusters[src].index, Clusters[dst].index)
            Clusters[src].merge(
                Clusters[dst], self.numRepPoints, self.alpha, dist_mat[idx_][:, idx_]
            )

            dist_f = col_dist_max if type(Clusters[src].repPoints[0]) != list else mat_dist_max

            # Update the distCluster matrix
            for i in range(src):
                cluster_distance[src, i] = dist_f(Clusters[src].repPoints, Clusters[i].repPoints)
            for i in range(src + 1, dst):
                cluster_distance[i, src] = dist_f(Clusters[src].repPoints, Clusters[i].repPoints)

            # Delete the merged cluster and its disCluster vector.
            cluster_distance[dst, :] = float("inf")
            cluster_distance[:, dst] = float("inf")
            self.labels_[self.clusters[dst].index] = self.clusters[src].index[0]

            cluster_distance = np.delete(np.delete(cluster_distance, dst, 0), dst, 1)
            del Clusters[dst]

            self.cluster_distance = cluster_distance

            self.clusters = Clusters
            self.cluster_centers_ = np.array([c.center for c in Clusters])
            self.indices = [c.index[0] for c in self.clusters]
            if DEBUG:
                print(
                    self.cluster_distance.shape[0],
                    len(self.clusters),
                    len(self.cluster_centers_),
                    len(self.indices),
                    len(np.unique(self.labels_)),
                )

            with open(f"./cure_5_0.1_{cur_k-1}.pkl", "wb") as f:
                pkl.dump(self, f)

    def get_inertia(self):
        label2Idx = {c.index[0]: i for i, c in enumerate(self.clusters)}
        self.inertia_ = inertia(
            self.data,
            np.array([label2Idx[v] for v in self.labels_]),
            self.cluster_centers_,
        )

        return self.inertia_, silhouette_score(self.data, self.labels_)


if __name__ == "__main__":
    datas = get_rfm_data()
    CURE(5, 0.1, 20).fit(datas.values)
