import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
            # "order_month_year",
            "order date (DateOrders)",
        ],
        axis=1,
        inplace=True,
    )

    # data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])

    le = LabelEncoder()
    # for col_types in data.dtypes:
    #     print(col_types)
    # print(data.dtypes)

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

    return data.loc[:, data.columns != "fraud"], data["fraud"]
