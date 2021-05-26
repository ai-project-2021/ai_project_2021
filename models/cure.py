# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm, trange
import os
import sys
import pickle as pkl
from numba import njit

from utils import get_rfm_data


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

        Clusters = [
            Cluster(idPoint, data[idPoint, :])
            for idPoint in trange(len(data), desc="init clusters")
        ]

        dist_mat = np.vstack([col_dist(data, data[i, :]) for i in range(len(data))])
        cluster_distance = np.array(np.triu(dist_mat, 1), dtype="float")
        cluster_distance[cluster_distance == 0] = float("inf")
        indexes = list(range(numPts))

        for _ in trange(numPts, self.k, -1, desc="cluster count"):
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

        # Generate sample labels
        self.labels_ = np.zeros((numPts))
        for i, c in enumerate(Clusters, 1):
            self.labels_[c.index] = i

        with open("./cure.pkl", "w") as f:
            pkl.dump(self, f)
        return self

    def fit_predict(self, data):
        return self.fit(data).labels_


if __name__ == "__main__":
    datas = get_rfm_data()
    CURE(5, 0.1, 3).fit(datas.values)