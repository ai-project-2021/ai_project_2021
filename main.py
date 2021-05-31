import os
import argparse
import numpy as np

from models import RFClassifier, KMeans, KMedian, KMedoid, FuzzyKMeans
from models.recommend import Recommendation

from utils.loader import get_rfm_data, get_order, order_filter
from utils.product_filter import product_filter


def input_user_id():
    rfm = get_rfm_data()
    customer_id = rfm["Customer Id"].tolist()
    while True:
        _id = int(input("Input Your Id(type : int) : "))
        if _id in customer_id:
            return np.array(customer_id), _id, customer_id.index(_id)
        else:
            print("Please Input Collect Id")
            sample_user_id_list = ", ".join(
                np.random.choices(np.array(customer_id), 10, replace=False)
            )
            print(f"cf) {sample_user_id_list}")


def get_customer_cluster(clustering_model="KMeans", k=4):
    model_dict = {"KMeans": KMeans, "KMedian": KMedian, "KMedoid": KMedoid, "Fuzzy": FuzzyKMeans}
    if os.path.exists(f"./saved/{clustering_model}_{k}.pkl"):
        return model_dict[clustering_model](k=k).labels_
    else:
        print(f"Not Found {clustering_model} Models")
        print("Start Customer Clustering")
        rfm = get_rfm_data()
        return (
            model_dict[clustering_model](k=k)
            .fit(rfm[["R_Value", "F_Value", "M_Value"]].values)
            .labels_
        )


def get_order_dict(recommend_data):
    _get_item = lambda item: {
        sub_key: sub_item["Order Item Quantity"].tolist()[0]
        for sub_key, sub_item in item.groupby("Product Name")
    }

    return {
        c_id: _get_item(recommend_data[recommend_data["Customer Id"] == c_id])
        for c_id in recommend_data["Customer Id"].unique()
    }


def customer_filtering(args):
    customer_id_list, _id, customer_idx_ = input_user_id()
    labels_ = get_customer_cluster(clustering_model=args.clustering, k=args.k)
    select_customer_list = customer_id_list[np.where(labels_ == labels_[customer_idx_])]
    return _id, select_customer_list


def product_recommend(_id, select_customer_list, fraud, n):
    order_dict = get_order_dict(order_filter(get_order("quantity"), select_customer_list))

    fraud_product_ = product_filter(model=fraud)

    model = Recommendation(order_dict, n=n)
    idx_ = 1

    for name, _ in model.recommend(_id):
        if name not in fraud_product_:
            print("{}'s Best Item {} : {}".format(_id, idx_, name))
            idx_ += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Recommendation for You")
    parser.add_argument(
        "--clustering",
        "-c",
        type=str,
        help="Select Clustering Models [KMeans, Kmedian, Kmedoid, Fuzzy]",
        default="KMeans",
    )
    parser.add_argument(
        "--fraud",
        "-f",
        type=str,
        help="Select Clustering Models [RF, DT, DNN]",
        default="RF",
    )
    parser.add_argument("--k", "-k", type=int, help="Set n_clusters", default=4)
    parser.add_argument(
        "--max_best_k", "-n", type=int, help="Recommend Maximum Products", default=10
    )
    args = parser.parse_args()

    assert args.clustering in ["KMeans", "KMedian", "Kmedoid", "Fuzzy"]
    assert args.fraud in ["RF", "DT", "DNN"]

    _id, select_customer_list = customer_filtering(args)
    product_recommend(_id, select_customer_list, args.fraud, args.max_best_k)