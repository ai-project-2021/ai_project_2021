# -*- coding: utf-8 -*-

from math import sqrt
import numpy as np
from collections import defaultdict
from utils.loader import get_order
from tqdm import tqdm


class Recommendation:
    def __init__(self, data, k=10, metric="pearson", n=5):
        self.k = k  # 사람 수
        self.n = n  # Product 개수
        if metric == "pearson":
            self.fn = self.pearson
        elif metric == "cosin":
            self.fn = self.cos_sim
        self.data = data

    def cos_sim(self, rating1, rating2):
        cols = list([key for key in rating1] + [key for key in rating2 if key not in rating1])
        a = np.array([rating1[c] if c in rating1 else 0 for c in cols])
        b = np.array([rating2[c] if c in rating2 else 0 for c in cols])
        return np.dot(a, b) / np.linalg.norm(a) * np.linalg.norm(b)

    def pearson(self, rating1, rating2):

        res_ = np.array(
            [
                [
                    rating1[key],
                    rating2[key],
                    rating1[key] ** 2,
                    rating2[key] ** 2,
                    rating1[key] * rating2[key],
                ]
                for key in rating1
                if key in rating2
            ]
        )

        n = res_.shape[0]
        res_ = res_.sum(axis=0)

        if n == 0:
            return 0

        denominator = sqrt(res_[2] - pow(res_[0], 2) / n) * sqrt(res_[3] - pow(res_[1], 2) / n)

        return 0 if denominator == 0 else (res_[4] - (res_[0] * res_[1]) / n) / denominator

    def computeNearestNeighbor(self, name):
        return sorted(
            [(row, self.fn(self.data[name], self.data[row])) for row in self.data if row != name],
            key=lambda item: -item[1],
        )

    def recommend(self, user):
        nearest = self.computeNearestNeighbor(user)
        nearest = nearest[: self.k]
        userRatings = self.data[user]
        totalDistance = sum([dist_ for _, dist_ in nearest])


        recommendations = defaultdict(int)
        for name, weight in nearest:
            neighborRatings = self.data[name]
            for artist in neighborRatings:
                if artist in userRatings:
                    recommendations[artist] += neighborRatings[artist] * (
                        (weight / totalDistance) if totalDistance != 0 else 0
                    )
                    # 추천 W = 자신의 W * (같은 상품을 산 피어슨 계수 / K 피어슨)

        return sorted(list(recommendations.items()), key=lambda row: -row[1])[: self.n]


if __name__ == "__main__":
    import random

    # datas = get_order("all")

    # for _type in ["quantity", "count", "all"]:
    datas = get_order("quantity")
    _id_list = [random.choice(list(datas.keys())) for _ in range(10)]
    _id_list = list(datas.keys())
    # _id_list = [1622]
    model = Recommendation(datas)

    print(_id_list)
    for _id in _id_list:
        print(_id)
        print(model.recommend(_id))
