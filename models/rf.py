import numpy as np
import pandas as pd
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, f1_score, make_scorer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from utils import get_fraud


class RFClassifier:
    def __init__(self, **kwargs):
        self.params = dict()

        X, y = kwargs.get("X", None), kwargs.get("y", None)
        if X is None or y is None:
            X, y = get_fraud()
        self.params["test_size"] = kwargs.get("test_size", 0.2)
        self.get_data(X, y)

        self.model_params = dict()
        self.model_params["n_estimators"] = 100
        self.model_params["oob_score"] = True
        self.model_params["max_depth"] = 10
        self.model_params["min_samples_leaf"] = 8
        self.model_params["min_samples_split"] = 8
        self.model_params["max_features"] = "sqrt"
        self.model_params["random_state"] = 123456
        self.model_params["criterion"] = "entropy"
        self.model_params["class_weight"] = {
            y_: y[y != y_].shape[0] / y[y == y_].shape[0] for y_ in np.unique(y.values)
        }

        self.clf = RandomForestClassifier(
            **self.model_params,
            n_jobs=-1,
        )

        self.k_fold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.model_params["random_state"]
        )

    def save_params(self, _path):
        with open(_path, "w") as f:
            json.dump(dict(self.params, **self.model_params), f)

    def get_data(self, X, y):
        self.features_names = X.columns

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.values, y.values, test_size=self.params["test_size"], stratify=y, random_state=123456
        )

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
        return self.clf.score(self.X_train, self.y_train)

    def test(self):
        return self.clf.score(self.X_test, self.y_test)

    def predict(self, X):
        return self.clf.predict(X)

    def kfold_train(self):
        self.clf.fit(self.X_train, self.y_train)
        return cross_val_score(self.clf, self.X_train, self.y_test, cv=self.k_fold).mean()

    def kfold_test(self):
        return cross_val_score(self.clf, self.X_test, self.y_test, cv=self.k_fold).mean()

    def kfold_predict(self):
        return cross_val_predict(self.clf, self.X_test, self.y_test, cv=self.k_fold)

    def gridSearch(self):
        f1 = make_scorer(f1_score, average="macro")
        self.param_grid = {
            "n_estimators": [10, 20, 30, 50],
            "max_depth": [2, 4, 6, 8, 10],
            "min_samples_leaf": [8, 12, 18],
            "min_samples_split": [8, 16, 20],
        }

        self.grid_clf = GridSearchCV(self.clf, self.param_grid, cv=3, scoring=f1)
        self.grid_clf.fit(self.X_train, self.y_train)
        return (
            self.grid_clf.best_params_,
            self.grid_clf.best_estimator_,
        )

    def result(self):
        _pred = self.predict(self.X_test)

        reversefactor = dict(zip(range(2), range(2)))
        y_pred = np.vectorize(reversefactor.get)(_pred)
        y_test = np.vectorize(reversefactor.get)(self.y_test)

        cm = pd.crosstab(y_test, y_pred)
        cm = pd.DataFrame(
            confusion_matrix(self.y_test, _pred),
        )

        print(classification_report(self.y_test, _pred))
        print(cm)
        return f1_score(self.y_test, _pred, average="macro")


if __name__ == "__main__":
    X, y = get_fraud()
    # X = X[["Type", "late_delivery"]]
    model = RFClassifier(X=X, y=y)
    print(model.train())
    print(model.test())
    # best_params, model.clf = model.gridSearch()
    # print(classification_report(model.y_train, model.predict(model.X_train)))
    # print(confusion_matrix(model.y_train, model.predict(model.X_train)))
    # print(classification_report(model.y_test, model.predict(model.X_test)))
    # print(confusion_matrix(model.y_test, model.predict(model.X_test)))

    # rfe = RFE(model.clf, 10)
    # fit = rfe.fit(X, y)
    # print(fit.n_features_, fit.support_, fit.ranking_)
    for n in range(10, 2, -1):
        rfe = RFE(model.clf, 10)
        fit = rfe.fit(X, y)
        model = RFClassifier(X=X.iloc[:, fit.support_], y=y)
        model.train()
        print(classification_report(model.y_test, model.predict(model.X_test)))
        print(confusion_matrix(model.y_test, model.predict(model.X_test)))

    #     print(fit.n_features_, fit.support_, fit.ranking_)
    # print(classification_report(model.y_test, model.predict(model.X_test)))
    # print(confusion_matrix(model.y_test, model.predict(model.X_test)))
    # import matplotlib.pyplot as plt

    # plt.figure()
    # idx_ = model.clf.feature_importances_.argsort()
    # plt.barh(model.features_names[idx_], model.clf.feature_importances_[idx_])
    # plt.savefig("./graph/rf2.png")
    # print(best_params)