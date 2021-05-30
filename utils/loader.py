from os import dup
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE

import datetime as dt
from functools import reduce


DEBUG = True


def rescaler(data, scaler=None):
    dic = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "normalize": Normalizer,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
    }
    assert scaler in dic.keys()
    return dic[scaler]().fit_transform(data)


def get_raw_data():
    """load CSV File to pd.DataFrame

    Returns:
        [pd.DataFrame]: DataCoSupplyChainDataset DataFrame
    """
    dataset = pd.read_csv(
        "dataset/DataCoSupplyChainDataset.csv",
        header=0,
        encoding="unicode_escape",
    )

    dataset["Customer Full Name"] = dataset["Customer Fname"].astype(str) + dataset[
        "Customer Lname"
    ].astype(str)

    # Fraud Detection / Customer Segmentation 모두 사용하지 않는 Column Drop
    data = dataset.drop(
        [
            "Customer Email",
            "Product Status",
            "Customer Password",
            "Customer Street",
            "Customer Fname",
            "Customer Lname",
            "Latitude",
            "Longitude",
            "Product Description",
            "Product Image",
            "Order Zipcode",
            "Customer Zipcode",
            "shipping date (DateOrders)",
        ],
        axis=1,  # By Column
    )

    customer_product_unique = (
        data[["Customer Id", "Product Name"]]
        .drop_duplicates()
        .groupby(["Customer Id"])
        .agg({"Product Name": lambda x: len(x)})
        .reset_index()
    )

    # 1가지 종류의 상품만을 구입한 사람은 Product Recomandation이 어렵기 때문에 Filtering
    customer_filter = customer_product_unique[customer_product_unique["Product Name"] > 3][
        "Customer Id"
    ]

    return data[data["Customer Id"].isin(customer_filter)]


def get_product_recommend():
    """Customer Order Dataset

    Returns:
        pd.DataFrame: Customer Order DataFrame
    """

    cols = [
        "Customer Id",
        "order date (DateOrders)",
        "Order Id",
        "Order Item Quantity",
        "Order Item Total",
        "Product Name",
    ]

    return get_raw_data()[cols]


def R_Score(a, b, c):
    if a <= c[b][0.25]:
        return 1
    elif a <= c[b][0.50]:
        return 2
    elif a <= c[b][0.75]:
        return 3
    else:
        return 4


def FM_Score(x, y, z):
    if x <= z[y][0.25]:
        return 4
    elif x <= z[y][0.50]:
        return 3
    elif x <= z[y][0.75]:
        return 2
    else:
        return 1


def get_rfm_score(rfm):
    """R/F/M Value 정보를 바탕으로 4분위수를 기준으로 Score로 환산한 뒤, RFM Score를 계산하여 반환

    Args:
        rfm (pd.DataFrame): [Customer_Id, R_Value, F_Value, M_Value] 정보를 가진 DataFrame

    Returns:
        [pd.DataFrame]: Customer_Id, R_Value, F_Value, M_Value, RFM_Score
    """
    quantiles = rfm.quantile(q=[0.25, 0.5, 0.75]).to_dict()  # Dividing RFM data into four quartiles

    rfm["R_Score"] = rfm["R_Value"].apply(R_Score, args=("R_Value", quantiles))
    rfm["F_Score"] = rfm["F_Value"].apply(FM_Score, args=("F_Value", quantiles))
    rfm["M_Score"] = rfm["M_Value"].apply(FM_Score, args=("M_Value", quantiles))
    rfm["RFM_Total_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    return rfm[["Customer Id", "R_Value", "F_Value", "M_Value", "RFM_Total_Score"]]


def get_rfm_data():
    """상품 주문 정보를 바탕으로 R/F/M Value를 계산하고, Customer Id과 RFM Value를 가지는 pd.DataFrame 반환

    Returns:
        [pd.DataFrame]: Customer_Id, R_Value, F_Value, M_Value 정보를 가진 pd.DataFrame
    """
    data = get_product_recommend()

    cur = dt.datetime(2018, 2, 1)  # Dataset에서 가장 최근인 날.
    data["R_Value"] = pd.to_datetime(data["order date (DateOrders)"]).apply(
        lambda t: (cur - t).days
    )
    data["M_Value"] = data["Order Item Quantity"] * data["Order Item Total"]

    r_value = (data.groupby("Customer Id")["R_Value"].max().reset_index())[
        ["Customer Id", "R_Value"]
    ]

    f_value = (
        data.groupby("Customer Id")["Order Id"]
        .count()
        .reset_index()
        .rename(columns={"Order Id": "F_Value"})
    )[["Customer Id", "F_Value"]]

    m_value = data.groupby("Customer Id")["M_Value"].sum().reset_index()[["Customer Id", "M_Value"]]

    # Create R/F/M Value DataFrame
    rfm = reduce(
        lambda l, r: pd.merge(l, r, on="Customer Id", how="left"), [r_value, f_value, m_value]
    )

    if DEBUG:
        print(rfm.describe())

    return rfm


def get_fraud(sampling=None, is_get_dummies=False):
    """Generate Fraud Detection Dataset

    Args:
        sampling ([str], optional): [Sampling Method]. Defaults to None.
        is_get_dummies (bool, optional): [One-Hot Encoding]. Defaults to False.

    Returns:
        [(np.ndarray, np.array)]: X, y ((records, n_features), (records, ))
    """
    data = get_raw_data()  # get Row Dataset

    # 주문 총액, 주문 사기 여부, 배송 지연 여부 Column 생성
    data["TotalPrice"] = data["Order Item Quantity"] * data["Order Item Total"]
    data["fraud"] = np.where(data["Order Status"] == "SUSPECTED_FRAUD", 1, 0)
    data["late_delivery"] = np.where(data["Delivery Status"] == "Late delivery", 1, 0)

    # Pre-Processing Columns Drop
    data.drop(
        [
            "Delivery Status",
            "Late_delivery_risk",
            "Order Status",
            "order date (DateOrders)",
        ],
        axis=1,
        inplace=True,
    )

    # Categorical Columns

    X, y = data.loc[:, data.columns != "fraud"], data["fraud"]

    categorical_columns = [c for c, dtype_ in zip(X.columns, X.dtypes) if dtype_ == "object"]

    if is_get_dummies == True:  # One-Hot Vector
        numeric_columns = [c for c, dtype_ in zip(X.columns, X.dtypes) if dtype_ != "object"]
        X = X[numeric_columns] + pd.get_dummies(X[categorical_columns])

    else:  # Object to Integer (is_get_dummies == False)
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col])

    if sampling == None:
        return X, y
    elif sampling == "over":
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    elif sampling == "smote":
        return SMOTE(random_state=42).fit_resample(X, y)


def get_order(key_, customer_id_list=None):
    """Product Recommdatation Dataset

    Args:
        key_ (str): Quantity, Count, All : Summarize For Order Item Quantity Method

    Returns:
        dict(key :dict()): id : dict(Product Name = Summarize For Order Item Quantity)
    """
    order_data = get_raw_data()[["Customer Id", "Product Name", "Order Item Quantity"]]

    if key_ == "quantity":
        f = lambda x: x.sum()
    elif key_ == "count":
        f = lambda x: len(x)
    elif key_ == "all":
        f = lambda x: x.sum() + len(x)

    order_data_summarize = (
        order_data.drop_duplicates()
        .groupby(["Customer Id", "Product Name"])
        .agg({"Order Item Quantity": lambda x: f(x)})
        .reset_index()
    )

    return order_data_summarize


def order_filter(order_data, customer_id_list):
    return order_data[order_data["Customer Id"].isin(customer_id_list)]


if __name__ == "__main__":
    get_rfm_data()
