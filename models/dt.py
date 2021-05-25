import numpy as np
import pandas as pd
import json
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, f1_score, make_scorer
from utils import get_fraud


class DTClassifier:
    def __init__(self, **kwargs):
        self.params = dict()
        X, y = kwargs.get("X", None), kwargs.get("y", None)
        self.get_data(X, y)
        # self.params.train_size = kwargs.get("train_size", None)
        # self.params.test_size = kwargs.get("test_size", None)

        # self.data = kwargs.get("data", None)

        self.model_params = dict()
        self.model_params["max_depth"] = 8
        self.model_params["min_samples_leaf"] = 1
        self.model_params["min_samples_split"] = 2
        self.model_params["max_features"] = "auto"
        self.model_params["random_state"] = 123456
        self.model_params["criterion"] = "entropy"
        self.model_params["class_weight"] = {
            label: (float(y[y != label].shape[0]) / float(y[y == label].shape[0])) ** 0.5
            for label in np.unique(y)
        }

        print(self.model_params["class_weight"])
        print(y[y == 0].shape[0], y[y == 1].shape[0])

        self.clf = DecisionTreeClassifier(
            **self.model_params,
        )

        self.k_fold = RepeatedStratifiedKFold(
            n_splits=5, random_state=self.model_params["random_state"]
        )

    def save_params(self, _path):
        with open(_path, "w") as f:
            json.dump(dict(self.params, **self.model_params), f)

    def get_data(self, X, y):
        self.features_names = X.columns

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123456
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
        f1 = make_scorer(f1_score, average="macro")
        return cross_val_score(
            self.clf, self.X_train, self.y_train, cv=self.k_fold, scoring=f1
        ).mean()

    def kfold_test(self):
        f1 = make_scorer(f1_score, average="macro")
        return cross_val_score(
            self.clf, self.X_test, self.y_test, cv=self.k_fold, scoring=f1
        ).mean()

    def kfold_predict(self, X):
        return cross_val_predict(self.clf, X, cv=self.k_fold)

    def gridSearch(self):
        f1 = make_scorer(f1_score, average="macro")
        self.param_grid = {
            "max_depth": [4, 6, 8, 10, 12, 15],
            "min_samples_leaf": [8, 12, 18, 24, 40],
            "min_samples_split": [8, 16, 20, 30, 40, 50],
        }

        self.grid_clf = GridSearchCV(self.clf, self.param_grid, cv=3, scoring=f1)
        self.grid_clf.fit(self.X_train, self.y_train)
        return (
            self.grid_clf.best_params_,
            self.grid_clf.cv_results_,
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
    model = DTClassifier(X=X, y=y)
    print(model.gridSearch())
