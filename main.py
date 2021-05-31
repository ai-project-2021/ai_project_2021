import os
import argparse
import numpy as np
import dirr

from models import RFClassifier, KMeans, KMedian, KMedoid, FuzzyKMeans
from models.recommend import Recommendation

from utils.loader import get_rfm_data, get_order, order_filter
from utils.product_filter import product_filter


def input_user_id():
    """rfm 정보를 불러와 Customer Id를 넘겨 리스트를 형성한다.

    사용자가 Id(Input 정보) 를 입력한다, 이 아이디가 dataset에 존재하면, 아이디 정보가 반환된다.
    Returns:
        test data로 사용 될 user id list가 반환된다.
    """
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
    """
        각 clustering model에 따른 최적의 k값인 4를 기준으로 구헌한 고객 등급을 분류하는 모델이다.
    Args:
        clustering_model : 4가지 모델 들 중 가장 클러스트링이 균일하게 되었던 모델인 k means model을 사용하였다.
        k : 최적의 효율로 군집화되는 군집의 개수인 4를 k로 지정하였다.

    Returns: clustering model에 따른 r, f, m value의 clustering과 그 label을 저장한다.

    """
    model_dict = {"KMeans": KMeans, "KMedian": KMedian, "KMedoid": KMedoid, "Fuzzy": FuzzyKMeans}
    if os.path.exists(f"./saved/{clustering_model}_{k}.pkl"):
        with open(f"./saved/{clustering_model}_{k}.pkl","rb") as f:
            return dill.load(f).labels_
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
    # recommend data 중, customer id 가 cid 인 사옹자들을 리스트화 하여 아이디별로 정리한다.
    _get_item = lambda item: {
        sub_key: sub_item["Order Item Quantity"].tolist()[0]
        for sub_key, sub_item in item.groupby("Product Name")
    }

    return {
        c_id: _get_item(recommend_data[recommend_data["Customer Id"] == c_id])
        for c_id in recommend_data["Customer Id"].unique()
    }

# 추천을 원하는 사용자의 customer group에 따라 customer id 가 분류된다
def customer_filtering(args):
    customer_id_list, _id, customer_idx_ = input_user_id()
    labels_ = get_customer_cluster(clustering_model=args.clustering, k=args.k)
    select_customer_list = customer_id_list[np.where(labels_ == labels_[customer_idx_])]
    return _id, select_customer_list


def product_recommend(_id, select_customer_list, fraud, n):
    # quantity, count, quanty+count 값 중 quantity값을 이용하였다.
    order_dict = get_order_dict(order_filter(get_order("quantity"), select_customer_list))

    fraud_product_ = product_filter(model_=fraud)

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