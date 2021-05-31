import numpy as np
import pandas as pd

from utils.loader import get_fraud
from models.rf import RFClassifier
from models.dt import DTClassifier
from models.dnn import DNN_model


def product_filter(model, threshold=0.15):
    X, y, product_name = get_fraud()

    model_dict = {"RF": RFClassifier, "DT": DTClassifier, "DNN": DNN_model}

    model = model_dict[model](X=X, y=y, load=True)
    _pred = model.predict(X.values)

    df = pd.DataFrame({"Product Name": product_name, "Fraud": _pred})
    fraud_rate = (
        df.groupby(["Product Name"]).agg({"Fraud": lambda x: sum(x) / len(x)}).reset_index()
    )

    return fraud_rate[fraud_rate["Fraud"] > threshold]["Product Name"].tolist()