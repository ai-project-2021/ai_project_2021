from numpy.lib.function_base import select

import numpy as np

from sklearn.metrics import cluster

from models import RFClassifier, KMeans
from models.recommend import Recommendation

from utils.loader import get_product_recommend
from utils.loader import get_rfm_data, get_order, order_filter

import random

if __name__ == "__main__":
    _id = 1302

    clustering_model = KMeans

    rfm = get_rfm_data()
    customer_id = rfm["Customer Id"].tolist()
    _id = random.choice(customer_id)
    print(_id)
    customer_idx_ = customer_id.index(_id)

    labels_ = clustering_model(k=3).fit(rfm[["R_Value", "F_Value", "M_Value"]].values).labels_

    cluster_labels_indices = np.where(labels_ == labels_[customer_idx_])

    select_customer_list = rfm["Customer Id"].to_numpy()[
        np.where(labels_ == labels_[customer_idx_])
    ]

    # print(recommend_data["Customer Id"].unique())
    orders_data = get_order("quantity")

    recommend_data = order_filter(orders_data, select_customer_list)

    _get_item = lambda item: {
        sub_key: sub_item["Order Item Quantity"].tolist()[0]
        for sub_key, sub_item in item.groupby("Product Name")
    }

    order_dict = {
        c_id: _get_item(recommend_data[recommend_data["Customer Id"] == c_id])
        for c_id in recommend_data["Customer Id"].unique()
    }

    model = Recommendation(order_dict)
    for idx, (name, weight) in enumerate(model.recommend(_id), 1):
        print("Best Item {} : {}".format(idx, name))
