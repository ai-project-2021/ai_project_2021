import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
)  # Clustering의 품질을 정량적으로 평가해주는 지표. [0-1] 1에 가까울 수록 우수한 품질
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
        """Fuzzy K-Means Clustering

        Args:
            k (int): 군집으로 형성 될 cluster의 갯수
            m (int): cluster의 fuzzy 정도를 제어하는 하이퍼 파라미터이다. Defaults 값은 2이다.
                    m값이 1에 가까울 수록 data당 클러스터가 하나에 종속되고, m값이 높을수록 여러 클러스터에 종속된다.
            max_iter (int): Centroid 갱신 횟수를 나타내며, 최대 100회로 제한한다.
            random_state (int): random_state가 0인 경우, numpy check_random_state를 계산하기 위한 개체이다. Defaults 겂은 0이다.
            threshold (float): cluster 갱신 반복을 종료해주 인자로 사용된다. Defaults 값은 1e-4.
        """
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.threshold = threshold

    def _e_step(self, X):
        """데이터와 클러스터 중심 사이의 거리를 이용하여, 데이터가 각 클러스터에 대하여 얼마나 속해있는지를 계산하는 함수이다.
            fuzzy clustering 에서 데이터 간 거리는 Euclidean 거리 공식을 이용한다.

        Args:
            X (datas): 원본 data중 'order date (Date Orders)', 'Order Item Quantity','Order Item Total'
                    , 'Order Id'를 이용하여 'Customer Id'를 기준으로 분류되어 계산된 R_value, F_value, M_value 값을 나타내는 데이터이다.
        """
        # D : fuzzy 멤버십, 즉 data가 클러스터에 얼마나 속해 있는지를 계산한 값
        D = (1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)) ** (
            2.0 / (self.m - 1)
        )

        # shape: n_samples x k
        self.fuzzy_labels_ = D / np.sum(D, axis=1)[:, np.newaxis]
        self.labels_ = self.fuzzy_labels_.argmax(axis=1)

    def _m_step(self, X):
        """fuzzy 지수인 m값을 바탕으로 각 cluster의 중심을 Update하는 함수이다.

        Args:
            X (datas): 원본 data중 'order date (Date Orders)', 'Order Item Quantity','Order Item Total'
                    , 'Order Id'를 이용하여 'Customer Id'를 기준으로 분류되어 계산된 R_value, F_value, M_value 값을 나타내는 데이터이다.
        """
        # shape: n_clusters x n_features
        # self.m : fuzzy 지수를 나타낸다. defaluts = 2
        weights = self.fuzzy_labels_ ** self.m
        self.cluster_centers_ = np.dot(X.T, weights).T / weights.sum(axis=0)[:, np.newaxis]

    def fit(self, X):  # y가 사용되지 않는데 지워도 되는지?
        """초기 레이블 값을 설정하고, 최대 갱신 횟수만큼 cluster 중심을 갱신하며 데이터 별 최적의 레이블 값을 선택하는 함수이다.
        Args:
            X (datas): 원본 data중 'order date (Date Orders)', 'Order Item Quantity','Order Item Total'
                    , 'Order Id'를 이용하여 'Customer Id'를 기준으로 분류되어 계산된 R_value, F_value, M_value 값을 나타내는 데이터이다.

        Returns:
            self : clustering 의 결과로 산출된 data의 레이블 값을 반환한다.
        """

        n_samples, _ = X.shape
        # np.mean : 평균 계산, np.var : 분산 계산
        vdata = np.mean(np.var(X, 0))

        # self.random_state ==0 인 경우, 무작위로 초기화된 객체가 반환되며, rand()를 통해 0~1 사이의 난수가 (sample number, k)만큼 생성된다.
        self.fuzzy_labels_ = check_random_state(self.random_state).rand(n_samples, self.k)
        self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1)[:, np.newaxis]
        # 초기 레이블 값 생성 이후 레이블 값 갱신 단계
        self._m_step(X)

        # 최대 반복 횟수만큼 cluster 갱신
        for _ in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            # data 간 차이가 설정한 threshold보다 작은 경우 cluster 갱신 종료
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