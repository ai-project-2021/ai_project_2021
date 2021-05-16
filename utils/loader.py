import pandas as pd
import numpy as np


def get_raw_data():
    dataset = pd.read_csv(
        "../dataset/DataCoSupplyChainDataset.csv",
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
    ]

    return data[cols]


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
            "order_month_year",
            "order date (DateOrders)",
        ],
        axis=1,
        inplace=True,
    )

    data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])

    return data[data.columns != "fraud"], data["fraud"]
