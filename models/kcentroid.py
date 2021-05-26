import numpy as np
from utils import get_rfm_data
from utils.metrics import inertia
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
)  # Clustering의 품질을 정량적으로 평가해주는 지표. [0-1] 1에 가까울 수록 우수한 품질


class KMeans:
    def __init__(self, k):
        """K-Means Clustering

        Args:
            k (int): 군집으로 형성 될 cluster의 갯수
        """
        self.k = k
        # f : 지정된 축을 따라 array 요소들의 산술 평균을 계산하는 numpy 함수이다.
        self.f = np.mean

    def _update(self):
        """군집을 갱신하는 Assignment과정을 수행하는 함수이다. 모든 데이터에 대하여, 가장 거리가 가까운 클러스터를 선택한다. 거리를 구하는데는 Euclidean distance 공식이 사용되었다. 클러스터의 중심과 각 클러스터에 속해 있는 데이터간의 거리를 계산하여, 그 중 가장 거리가 가까운 클러스터에 데이터를 할당한다.

        Returns:
            np.eye(k): 데이터와 각 클러스터 사이의 유클리드 거리를 계산하여, 가장 거리가 작은 값을 가진 클러스터에 데이터를 할당시킨다.
        """
        _clu = lambda c: np.array(np.sum((self.datas - c) ** 2, axis=1) ** 0.5)
        return np.eye(self.k)[np.array([_clu(c) for c in self.centriod_]).argmin(axis=0)]

    def fit(self, datas):
        """초기 클러스터의 중심을 랜덤하게 초기화 해주는 initialization 과정과, 새로운 군집 변화에 대하여 클러스터의 중심을 다시 계산하는 centroid Update 과정을 수행하는 함수이다.
        centroid Update 과정으로 클러스터의 중심이 갱신된 후, update함수가 갱신된 클러스터를 중심으로 군집을 갱신한다.


        Args:
            datas (list): 원본 data중 'order date (Date Orders)', 'Order Item Quantity','Order Item Total', 'Order Id'를 이용하여 'Customer Id'를 기준으로 분류되어 계산된 R_value, F_value, M_value 값을 나타내는 데이터이다.

        Returns:
            self.fit.labels_ : 갱신을 종료한 이후 클러스터들의 데이터 별 레이블 (각 데이터 별 할당된 군집)
        """
        self.datas = datas
        # 초기 cluster의 중심은 numpy random.choice 함수를 이용하여 data 크기 범위 중 k개로 랜덤하게 초기화된다.
        self.centriod_ = datas[np.random.choice(range(len(datas)), self.k)].astype("float32")
        # 갱신되기 전 군집의 데이터 별 레이블을 저장한다.
        prev_labels = np.zeros((len(datas)), dtype=int)

        while True:
            cluster = self._update()
            labels = cluster.argmax(axis=1)
            # 갱신되기 전, 후의 클러스터의 데이터 별 레이블을 비교한 후 모든 데이터에 대하여 레이블 값 차이가 없을 경우 갱신을 종료한다.
            if (labels == prev_labels).all():
                break
            # 레이블 값 차이가 있을 경우, 기존 레이블 값을 prev_labels에 저장하고 새로운 클러스터 중심(centroid) 계산
            prev_labels = labels.copy()
            self.centriod_ = [
                # K Means clustering에서 self.f == self.mean , 클러스터 내 모든 데이터 들의 평균
                self.f(datas[np.where(cluster[:, p] == 1)], axis=0)
                for p in range(self.k)
            ]

        self.labels_ = labels
        # 갱신이 종료되면 레이블 저장 후 최적의 k값을 계산하기 위한 inertia 값 계산
        self.inertia_ = inertia(datas, labels, self.centriod_)
        return self

    def fit_predict(self, datas):
        return self.fit(datas).labels_


class KMedians(KMeans):
    def __init__(self, k):
        """K-Medians Clustering
        군집의 중심인 centroid 를 계산할 때, 군집에 속한 데이터들 중 median값을 사용하는 clustering 기법이다.
        거리를 측정할 때 맨해튼 거리 공식을 이용한다.

        KMedians class는 KMeans class를 부모 클래스로 가지며 KMeans class의 Method, Attribute등을 상속받아 사용한다.
        따라서 __init__, _update, fit, fit_predict를 사용한다.

        Args:
            k (int): 군집으로 형성 될 cluster의 갯수
        """
        super().__init__(k=k)
        # K Means clustering에서 self.f == self.median , 클러스터 내 모든 데이터 중 중앙값
        self.f = np.median

    def _update(self):
        """군집을 갱신하는 Assignment과정을 수행하는 함수이다. 모든 데이터에 대하여, 가장 거리가 가까운 클러스터를 선택한다.
        K-medians clustering은 Manhattan distance 거리 공식을 사용하여 데이터간의 거리를 계산한다.
        클러스터의 중심과 각 클러스터에 속해 있는 데이터간의 맨해튼 거리를 계산하여, 그 중 가장 거리가 가까운 클러스터에 데이터를 할당한다.


        Returns:
            np.eye(k): 데이터와 각 클러스터 사이의 맨해튼 거리를 계산하여, 가장 거리가 작은 값(argmin)을 가진 클러스터에 데이터를 할당시킨다.
        """
        # 맨해튼 거리는 좌표간 차이의 절댓값이다.
        _clu = lambda c: np.array(np.sum(abs(self.datas - c), axis=1))
        return np.eye(self.k)[np.array([_clu(c) for c in self.centriod_]).argmin(axis=0)]


class KMedoids:
    def __init__(self, k, max_iters=500):
        """K-Medoid Clustering
        초기 중심 값에 민감한 반응을 보이고, 노이즈와 아웃라이어에 민감한 K-means clustering의 단점을 보완할 수 있는 클러스트링 방법이다.

        Args:
            k (int): 군집으로 형성 될 cluster의 갯수
            max_iter (int): Centroid 갱신 횟수를 나타내며, 최대 100회로 제한한다.
        """
        self.k = k
        self.max_iters = max_iters

    def _update(self, indices):
        """medoid을 갱신하는 Assignment과정을 수행하는 함수이다. 모든 데이터에 대하여, 가장 거리가 가까운 클러스터를 선택한다.
        거리를 구하는데는 Euclidean distance 공식이 사용되었다.
        medoid와 cluster data간 거리를 최소로 하는 data(object)를 medoid로 설정한다.

        Args:
            indices : 기존(갱신 전) medoid

        Returns:
            np.argmin(dist) : data간 사이의 유클리드 거리를 가장 작게 만드는 object들을 medoid로 반환한다.
        """
        return np.argmin(self.dist(self.datas[:, None, :], self.datas[None, indices, :]), axis=1)

    def dist(self, xa, xb):
        """두 데이터  xa, xb 사이의 Euclidean distance를 계산해주는 함수이다.

        Args:
            xa, xb (array): 데이터의 R_Value, F_Value, M_Value

        Returns:
            np.sqrt(sum(xa-xb 제곱합)): 두 데이터( cluster에 속한 object와 medoid) 사이의 거리를 유클리드 공식을 사용하여 반환한다.
        """
        return np.sqrt(np.sum(np.square(xa - xb), axis=-1))

    def find_medoids(self, labels):
        """각 클러스터를 대표하는 데이터인 medoids를 지정하는 함수이다.

        Args:
            labels (int): data당 속해있는 cluster의 레이블을 나타낸다.

        Returns:
            medoid를 반환한다.
        """
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
        """초기 medoid을 랜덤하게 초기화 해주는 initialization 과정과,
        새로운 군집 변화에 대하여 medoid를 다시 계산하는 medoid Update 과정을 수행하는 함수이다.

        Args:
            datas : 원본 data중 'order date (Date Orders)', 'Order Item Quantity','Order Item Total'
            , 'Order Id'를 이용하여 'Customer Id'를 기준으로 분류되어 계산된 R_value, F_value, M_value 값을 나타내는 데이터이다.

        Returns:
            self: 각 데이터와 데이터의 라벨값을 반환한다.
        """
        self.datas = datas
        # 임의로 정한 기존 데이터의 10% 크기의 batch size
        self.batch_size = datas.shape[0] // 10
        medoids = np.random.choice(self.datas.shape[0], self.k, replace=False)
        prev_labels = self._update(medoids)
        best_diffs = 1e10
        best_steps_ = -1

        for i in range(self.max_iters):
            medoids = self.find_medoids(prev_labels)
            labels = self._update(medoids)
            # diff : 갱신 전 후 레이블 값의 차이
            diffs = np.mean(labels != prev_labels)
            prev_labels = labels

            print("iteration {:2d}: {:.6%}p of points got reassigned." "".format(i, diffs))
            # 갱신된 라벨과 갱신되기 전 레이블값을 비교하여 갱신을 계속 할 지 결정한다.
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

    dataset = get_rfm_data()[["R_Value", "F_Value", "M_Value"]]
    model_dict = {"KMeans": KMeans, "KMedians": KMedians, "KMedoids": KMedoids}

    for model_ in args.models:
        s = time.time()
        inertia_list = []
        model = model_dict[model_](args.n_clusters[0])
        n_cols = len(args.n_clusters)
        fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

        for i, k in enumerate(args.n_clusters):
            e = time.time()
            model.k = k
            inertia_list.append(model.fit(dataset.values).inertia_)
            silhouette_avg = silhouette_score(dataset.values, model.labels_)
            print(
                "K : {}, Inertia : {}, Avg. Silhouette_Score : {}".format(
                    k, inertia_list[-1], silhouette_avg
                )
            )

        plt.figure()
        plt.plot(args.n_clusters, inertia_list)
        plt.savefig(f"./{model_}.png")
