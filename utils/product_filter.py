import numpy as np
import pandas as pd

from utils.loader import get_fraud
from models.rf import RFClassifier, load_rf_model
from models.dt import DTClassifier, load_dt_model
from models.dnn import DNN_model, load_dnn_model


def product_filter(model, threshold=0.15):
    X, _, product_name = get_fraud()

    model_dict = {
        "RF": [load_rf_model, RFClassifier],
        "DT": [load_dt_model, DTClassifier],
        "DNN": [load_dnn_model, DNN_model],
    }

    model = model_dict[model][0]()
    # model = model_dict[model](X=X, y=y)
    _pred = model.predict(X.values)

    df = pd.DataFrame({"Product Name": product_name, "Fraud": _pred})
    fraud_rate = (
        df.groupby(["Product Name"]).agg({"Fraud": lambda x: sum(x) / len(x)}).reset_index()
    )

    return fraud_rate[fraud_rate["Fraud"] > threshold]["Product Name"].tolist()