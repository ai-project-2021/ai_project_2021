import numpy as np
import plotly.express as px
from utils import get_product_recommend, get_rfm_data


class Kmeans:
    def __init__(self, k):
        self.k = k

    def fit(self, datas):
        centriod = np.random.choice(range(len(datas)), self.k)
        centriod = datas[centriod].astype("float32")

        prev_labels = np.zeros((len(datas)))

        while True:
            cluster = np.zeros((len(datas), self.k))
            labels = np.zeros((len(datas)))

            for i in range(len(datas)):  # Customer ID
                temp = np.zeros((self.k))
                for j in range(self.k):
                    temp[j] = np.sum((centriod[j] - datas[i]) ** 2) ** 0.5
                cluster[i][temp.argmin()] = 1
                labels[i] = temp.argmin()

            if (labels == prev_labels).all():
                break
            else:
                prev_labels = labels.copy()

            for p in range(self.k):
                clu_vec = datas[np.where(cluster[:, p] == 1)]
                k_to_c = np.sum(clu_vec, axis=0) / len(clu_vec)
                centriod[p] = k_to_c

            print(centriod)

        return cluster, centriod, labels


if __name__ == "__main__":
    model = Kmeans(k=3)
    dataset = get_rfm_data(get_product_recommend())[["R_Value", "F_Value", "M_Value"]]

    clu, c, labels = model.fit(dataset.values)

    print("CLU")
    print(clu)
    print("C")
    print(c)
    print("labels")
    print(labels)

    fig = px.scatter_3d(dataset, x="R_Value", y="F_Value", z="M_Value", color=labels)

    fig.show()