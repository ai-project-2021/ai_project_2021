import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import datetime as dt


DEBUG = False


def get_raw_data():
    dataset = pd.read_csv(
        "dataset/DataCoSupplyChainDataset.csv",
        header=0,
        encoding="unicode_escape",
    )

    dataset["Customer Full Name"] = dataset["Customer Fname"].astype(str) + dataset[
        "Customer Lname"
    ].astype(str)

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
            "shipping date (DateOrders)",
        ],
        axis=1,
    )
    data["Customer Zipcode"] = data["Customer Zipcode"].fillna(0)  # Filling NaN columns with zero
    return data


def get_product_recommend():
    data = get_raw_data()
    data["TotalPrice"] = data["Order Item Quantity"] * data["Order Item Total"]

    cols = [
        "Customer Id",
        "order date (DateOrders)",
        "Order Id",
        "Order Item Quantity",
        "Order Item Total",
        "Product Name",
        "TotalPrice",
    ]

    dataset = data[cols]

    if DEBUG:
        print(data[cols].nunique())  # each column's unique value == unique().shape[0]
        print(data["Customer Id"].nunique())
        print("\n")

    return dataset


# R_Score should be minimum so 1st quantile is set as 1.
def R_Score(a, b, c):
    if a <= c[b][0.25]:
        return 1
    elif a <= c[b][0.50]:
        return 2
    elif a <= c[b][0.75]:
        return 3
    else:
        return 4


# The higher the F_Score,M_Score the better so 1st quantile is set as 4.
def FM_Score(x, y, z):
    if x <= z[y][0.25]:
        return 4
    elif x <= z[y][0.50]:
        return 3
    elif x <= z[y][0.75]:
        return 2
    else:
        return 1


def get_rfm_score(datas):
    present = dt.datetime(2018, 2, 1)
    datas["order date (DateOrders)"] = pd.to_datetime(datas["order date (DateOrders)"])
    Customer_seg = (
        datas.groupby("Customer Id")
        .agg(
            {
                "order date (DateOrders)": lambda x: (present - x.max()).days,
                "Order Id": lambda x: len(x),
                "TotalPrice": lambda x: x.sum(),
            }
        )
        .reset_index()
    )

    Customer_seg["order date (DateOrders)"] = Customer_seg["order date (DateOrders)"].astype(int)
    Customer_seg.rename(
        columns={
            "order date (DateOrders)": "R_Value",
            "Order Id": "F_Value",
            "TotalPrice": "M_Value",
        },
        inplace=True,
    )
    quantiles = Customer_seg.quantile(q=[0.25, 0.5, 0.75])  # Dividing RFM data into four quartiles
    quantiles = quantiles.to_dict()

    Customer_seg["R_Score"] = Customer_seg["R_Value"].apply(R_Score, args=("R_Value", quantiles))
    Customer_seg["F_Score"] = Customer_seg["F_Value"].apply(FM_Score, args=("F_Value", quantiles))
    Customer_seg["M_Score"] = Customer_seg["M_Value"].apply(FM_Score, args=("M_Value", quantiles))

    Customer_seg["RFM_Total_Score"] = Customer_seg[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    Customer_seg["RFM_Total_Score"].unique()

    rfm = Customer_seg[["Customer Id", "R_Value", "F_Value", "M_Value", "RFM_Total_Score"]]
    return rfm


def get_rfm_data(customer_data):
    customer_data["order date (DateOrders)"] = pd.to_datetime(
        customer_data["order date (DateOrders)"]
    )
    customer_data["M_Value"] = (
        customer_data["Order Item Quantity"] * customer_data["Order Item Total"]
    )

    tmp = (
        customer_data.groupby("Customer Id")["order date (DateOrders)"]
        .max()
        .reset_index()
        .rename(columns={"order date (DateOrders)": "R_Value"})
    )

    tmp2 = (
        customer_data.groupby("Customer Id")["Order Id"]
        .count()
        .reset_index()
        .rename(columns={"Order Id": "F_Value"})
    )

    tmp3 = customer_data.groupby("Customer Id")["M_Value"].sum().reset_index()

    customer_data.drop(
        [
            "Order Id",
            "order date (DateOrders)",
            "Order Item Quantity",
            "Order Item Total",
            "Product Name",
        ],
        axis=1,
        inplace=True,
    )

    rfm = pd.merge(tmp, tmp2, on="Customer Id", how="left")
    rfm = pd.merge(rfm, tmp3, on="Customer Id", how="left")

    if DEBUG:
        print("RFM\n", rfm)

    time = rfm["R_Value"] - pd.to_datetime("20141231")  # 가장 이른 날인 2015.01.01보다 이른 날짜를 기준으로 사용

    # Timedelta의 attribute인 total_seconds() : time을 초 단위로 바꾸어 줌
    rfm["R_Value"] = [int(x.total_seconds() / (10 ** 4)) for x in time]

    return rfm


def get_fraud():
    data = get_raw_data()
    data["TotalPrice"] = data["Order Item Quantity"] * data["Order Item Total"]
    data["fraud"] = np.where(data["Order Status"] == "SUSPECTED_FRAUD", 1, 0)
    data["late_delivery"] = np.where(data["Delivery Status"] == "Late delivery", 1, 0)

    data.drop(
        [
            "Delivery Status",
            "Late_delivery_risk",
            "Order Status",
            # "order_month_year",
            "order date (DateOrders)",
        ],
        axis=1,
        inplace=True,
    )

    # data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])

    le = LabelEncoder()

    data["Customer Country"] = le.fit_transform(data["Customer Country"])
    data["Market"] = le.fit_transform(data["Market"])
    data["Type"] = le.fit_transform(data["Type"])
    data["Product Name"] = le.fit_transform(data["Product Name"])
    data["Customer Segment"] = le.fit_transform(data["Customer Segment"])
    data["Customer State"] = le.fit_transform(data["Customer State"])
    data["Order Region"] = le.fit_transform(data["Order Region"])
    data["Order City"] = le.fit_transform(data["Order City"])
    data["Category Name"] = le.fit_transform(data["Category Name"])
    data["Customer City"] = le.fit_transform(data["Customer City"])
    data["Department Name"] = le.fit_transform(data["Department Name"])
    data["Order State"] = le.fit_transform(data["Order State"])
    data["Shipping Mode"] = le.fit_transform(data["Shipping Mode"])
    # data["order_week_day"] = le.fit_transform(data["order_week_day"])
    data["Order Country"] = le.fit_transform(data["Order Country"])
    data["Customer Full Name"] = le.fit_transform(data["Customer Full Name"])

    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(data.loc[:, data.columns != "fraud"], data["fraud"])
    # return data.loc[:, data.columns != "fraud"], data["fraud"]


if __name__ == "__main__":
    get_rfm_data(get_product_recommend())
