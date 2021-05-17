import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DEBUG = True

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
    
    dataset = data[cols]
    
    if DEBUG : 
        print(data[cols].nunique()) # each column's unique value == unique().shape[0]
        print(data["Customer Id"].nunique())
        print("\n")

    return dataset

def get_rfm_data(customer_data) : 
    customer_data["order date (DateOrders)"] = pd.to_datetime(customer_data["order date (DateOrders)"])
    customer_data["M_Value"] = customer_data["Order Item Quantity"] * customer_data["Order Item Total"]

    customer_data = customer_data.rename(columns = {"TotalPrice" : "M_Value"})

    tmp = customer_data.groupby("Customer Id")["order date (DateOrders)"].max().reset_index()
    tmp = tmp.rename(columns={"order date (DateOrders)":"R_Value"})

    tmp2 = customer_data.groupby("Customer Id")["Order Id"].count().reset_index()
    tmp2 = tmp2.rename(columns={"Order Id": "F_Value"})

    tmp3 = customer_data.groupby("Customer Id")["M_Value"].sum().reset_index()


    # index를 기준으로 data의 모든 column의 값을 보여줌.
    customer_data.drop(["Order Id", "order date (DateOrders)", "Order Item Quantity", "Order Item Total", "Product Name"], axis =1, inplace = True)

    rfm = pd.merge(tmp, tmp2, on = "Customer Id", how = "left")
    rfm = pd.merge(rfm, tmp3, on = "Customer Id", how = "left")
    
    if DEBUG : 
        print("RFM\n", rfm)

    time = rfm["R_Value"] - pd.to_datetime("20141231") # 가장 이른 날인 2015.01.01보다 이른 날짜를 기준으로 사용

    # Timedelta의 attribute인 total_seconds() : time을 초 단위로 바꾸어 줌
    rfm["R_Value"] = [int(x.total_seconds()/(10**4)) for x in time]

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

if __name__ == "__main__" : 
    get_rfm_data(get_product_recommend())
