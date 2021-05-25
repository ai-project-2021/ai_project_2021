# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm, trange
import os
import sys
from multiprocessing import pool
import time
import pickle as pkl
from numba import njit

from utils import get_product_recommend, get_rfm_data


@njit
def col_dist(mat_, vec_):
    return np.sum((mat_ - vec_) ** 2, axis=1) ** 0.5


@njit
def col_dist_max(mat_, vec_):
    return np.max(np.sum((mat_ - vec_) ** 2, axis=1) ** 0.5)


@njit
def mat_dist_max(mat_x, mat_y):
    return np.sqrt(np.sum(np.square(mat_x - mat_y), axis=-1)).max()


def parallel_call(params):  # a helper for calling 'remote' instances
    cls = getattr(sys.modules[__name__], params[0])  # get our class type
    instance = cls.__new__(cls)  # create a new instance without invoking __init__
    instance.__dict__ = params[1]  # apply the passed state to the new instance
    method = getattr(instance, params[2])  # get the requested method
    args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
    return method(*args)  # expand arguments, call our method and return the result


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

    def prepareDistRep(self, name, args):
        for arg in args:
            yield [self.__class__.__name__, self.__dict__, name, arg]

    def multiDistRep(self, clu):
        t = pool.ThreadPool(processes=os.cpu_count())
        rs = t.map(parallel_call, self.prepareDistRep("distRep", clu))
        t.close()
        return rs


class CURE:
    def __init__(self, numRepPoints, alpha, k):
        self.numRepPoints = numRepPoints
        self.alpha = alpha
        self.k = k

    def fit(self, data):
        print("START FIT")

        # # Initialization
        # numCluster = len(data)
        numPts = len(data)

        Clusters = [
            Cluster(idPoint, data[idPoint, :])
            for idPoint in trange(len(data), desc="init clusters")
        ]

        dist_mat = np.vstack([col_dist(data, data[i, :]) for i in range(len(data))])
        cluster_distance = np.triu(dist_mat, 1)
        cluster_distance[cluster_distance == 0] = float("inf")
        indexes = list(range(numPts))
        # remove_list = [0 for _ in range(100)]
        # remove_index = 0
        for numCluster in trange(numPts, self.k, -1, desc="cluster count"):
            # Find a pair of closet clusters
            closet_ = np.where(cluster_distance == np.min(cluster_distance))
            clu_src = int(closet_[0][0])
            clu_dst = int(closet_[1][0])
            src = indexes.index(clu_src)
            dst = indexes.index(clu_dst)
            s1 = time.time()

            # Merge
            idx_ = np.append(Clusters[src].index, Clusters[dst].index)
            # dist_mat.ravel()[np.ravel_multi_index((idx_, idx_), dist_mat.shape)].reshape(dist_mat)
            Clusters[src].merge(
                Clusters[dst], self.numRepPoints, self.alpha, dist_mat[idx_][:, idx_]
            )
            e1 = time.time()

            s2 = time.time()

            # Update the distCluster matrix
            if src != 0:
                cluster_distance[clu_src, indexes[:src]] = Clusters[src].multiDistRep(
                    Clusters[:src]
                )
            if src + 2 != len(indexes):
                cluster_distance[indexes[src + 1 :], clu_src] = Clusters[src].multiDistRep(
                    Clusters[src + 1 :]
                )
            e2 = time.time()

            dist_f = col_dist_max if type(Clusters[src].repPoints[0]) != list else mat_dist_max

            for i, ii in enumerate(indexes[:src]):
                cluster_distance[clu_src, ii] = dist_f(
                    Clusters[src].repPoints, Clusters[i].repPoints
                )
            for i, ii in enumerate(indexes[src + 1 :], src + 1):  # range(src + 1, numCluster):
                cluster_distance[ii, clu_src] = dist_f(
                    Clusters[src].repPoints, Clusters[i].repPoints
                )

            # for i, ii in enumerate(indexes[:src]):
            #     # cluster_distance[clu_src, ii] = Clusters[src].distRep(Clusters[i])
            #     cluster_distance[clu_src, ii] = Clusters[src].distRep(Clusters[i])
            # for i, ii in enumerate(indexes[src + 1 :], src + 1):  # range(src + 1, numCluster):
            #     # cluster_distance[ii, clu_src] = Clusters[src].distRep(Clusters[i])
            #     cluster_distance[ii, clu_src] = Clusters[src].distRep(Clusters[i])

            # Delete the merged cluster and its disCluster vector.
            s3 = time.time()
            cluster_distance[clu_dst, :] = float("inf")
            cluster_distance[:, clu_dst] = float("inf")
            del Clusters[dst]
            indexes.remove(clu_dst)
            e3 = time.time()

        # Generate sample labels
        labels = np.zeros((numPts))
        for i, c in enumerate(Clusters, 1):
            labels[c.index] = i
        with open("./cure.pkl", "w") as f:
            pkl.dump(self, f)
        return labels


if __name__ == "__main__":
    datas = get_rfm_data(get_product_recommend())
    CURE(5, 0.1, 3).fit(datas.values)