"""사용할 패키지 불러오기
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from utils.loader import get_fraud
from re import X
import pickle as pkl


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class DNN_model:
    def __init__(self, X, y):
        """Create Dataset
        
        model 생성에 이용될 input layer, hidden layer, 그리고 output layer의 형태를 결정하고,
        마지막 layer에 이용할 activation 함수와 dropout 확률의 비율을 설정합니다.
        모델 학습에 적용할 batch size와 epoch 수, 그리고 class weight 값을 설정합니다.
        callbacks에 EarlyStopping 정보를 저장하고, dnn_model에 create_model 함수를 적용하여 모델을 구현합니다.

        Args:
            X (int): fraud 여부를 판단하는데 사용되는 feature 정보
            y (int): fraud 인지 아닌지에 대한 label 정보(1: 사기, 0: 사기 아님)
        """
        self.k_fold = StratifiedKFold(n_splits=5)
        self.get_data(X, y)
        self.n_in = self.X_train.shape[1]
        self.n_hiddens = [20, 20]
        self.n_out = 1
        self.activation = "sigmoid"
        self.p_keep = 0.4

        self.batch_size = 512
        self.epochs = 2000
        self.class_weights = {
            y_: y[y != y_].shape[0] / y[y == y_].shape[0] for y_ in np.unique(y.values)
        }
        self.callbacks = EarlyStopping(monitor="f1", patience=10, verbose=1)

        self.dnn_model = self.create_model()

    def get_data(self, X, y):
        """data scaling and split

        데이터를 스케일링 하고, 전체 데이터 셋을 training data 80%, test data 20%로 split 합니다.
        training data를 다시 training data 80%, val_data 20%로 split 합니다.

        Args:
            X (int): fraud 여부를 판단하는데 사용되는 feature 정보
            y (int): fraud 인지 아닌지에 대한 label 정보(1: 사기, 0: 사기 아님)
        """
        sc = StandardScaler()
        X = sc.fit_transform(X)

        self.X_train_, self.X_test, self.y_train_, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_, self.y_train_, test_size=0.2, stratify=self.y_train_, random_state=1
        )

    def train(self):
        """dnn model 학습 및 그래프 도출

        학습 데이터를 dnn model에 넣어서 학습을 시키고, 
        loss, val_lossdp 대한 결과 그래프 하나,
        precision, accuracy, f1 score에 대한 결과 그래프 하나를 도출합니다.
        """
        self.hist = self.dnn_model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            class_weight=self.class_weights,
            validation_data=(self.X_val, self.y_val),
            verbose=1,
        )
        self.loss_graph()
        self.result_graph()

    def eval_test(self):
        return self.dnn_model.evaluate(self.X_test, self.y_test)

    def eval_val(self):
        return self.dnn_model.evaluate(self.X_val, self.y_val)

    def predict(self):
        return self.dnn_model.predict(self.X_test)

    def kfold_train(self):
        return cross_val_score(self.dnn_model, self.X_train, self.y_train, cv=self.k_fold).mean()

    def kfold_test(self):
        return cross_val_score(self.dnn_model, self.X_test, self.y_test, cv=self.k_fold).mean()

    def kfold_predict(self):
        return cross_val_predict(self.dnn_model, self.X_test, self.y_test, cv=self.k_fold)

    def create_model(self):
        """DNN model을 구현하는 함수입니다.

        Sequential 모델을 불러와서 배치 정규화를 진행한 후, hidden layer를 거쳐서 하나의 label 값을 도출합니다.
        여기서 모델을 학습하는데 activation 함수로 relu가 사용되고, 마지막 layer에서만 sigmoid를 사용합니다.
        모델 컴파일 시에 최적화를 위해 Adam optimizer와 손실함수 binary_crossentropy를 사용하고, 
        metrics에는 평가 지표로 사용되는 accuracy, f1, precision, 그리고 recall을 사용합니다.

        Returns:
            model: Deep Neural Network model을 반환합니다.
        """
        model = Sequential()
        model.add(BatchNormalization())

        for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
            model.add(
                Dense(
                    input_dim=input_dim,
                    units=self.n_hiddens[i],
                    kernel_initializer="random_uniform",
                )
            )
            model.add(Activation("relu"))
            model.add(Dropout(self.p_keep))

        model.add(Dense(units=self.n_out))
        model.add(Activation(self.activation))

        opt = keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False
        )
        model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=["accuracy", f1, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        return model

    def loss_graph(self):
        """loss, val_loss graph

        epoch 값의 증가에 따른 loss와 val_loss의 값의 변화를 그래프로 표현합니다.
        """
        fig, loss_ax = plt.subplots(figsize=(15, 5))
        loss_ax.plot(self.hist.history["loss"], "y", label="train_loss")
        loss_ax.plot(self.hist.history["val_loss"], "r", label="val loss")

        loss_ax.set_xlabel("epoch")
        loss_ax.set_ylabel("loss")

        loss_ax.legend(loc="upper left")

        plt.show()

    def result_graph(self):
        """precision, accuracy, f1 score graph

        epoch 값의 증가에 따른 precision, accuracy, 그리고 f1 score graph의 값의 변화를 그래프로 표현합니다.
        """
        fig, result_ax = plt.subplots(figsize=(15, 5))
        print(self.hist.history.keys())
        result_ax.plot(self.hist.history["precision"], "y", label="precision")
        result_ax.plot(self.hist.history["recall"], "r", label="recall")
        result_ax.plot(self.hist.history["f1"], "b", label="f1")

        result_ax.set_xlabel("epoch")
        result_ax.set_ylabel("y")

        result_ax.legend(loc="upper left")

        plt.show()

def load_result():
    """모델 학습 결과 load

    dnn model의 학습 결과를 load하고, load해서 가져온 정보를 comile하고, evaluate을 통해 모델 평가까지 이루어집니다.
    """
    model = load_model(s+".h5", custom_objects={"f1": f1})
    model_params = pkl.load(open("dnn_model_params.pkl", "rb"))

    opt = keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False
        )
    model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=["accuracy", f1, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

    X_test = model_params["X_test"]
    y_test = model_params["y_test"]
    X_val = model_params["X_val"]
    y_val = model_params["y_val"]
    n_hiddens = model_params["n_hiddens"]

    test_evaluate = model.evaluate(X_test, y_test)
    val_evaluate = model.evaluate(X_val, y_val)

    y_pred = model.predict(X_test).round()
    cm = confusion_matrix(y_test, y_pred)

    print("test evaluate : ", test_evaluate)
    print("test loss : ", test_evaluate[0])
    print("test accuracy : ", test_evaluate[1])
    print("test f1 : ", test_evaluate[2])
    print("val evaluate : ", val_evaluate)
    print("val loss : ", val_evaluate[0])
    print("val accuracy : ", val_evaluate[1])
    print("val f1 : ", val_evaluate[2])

    print(classification_report(y_test, y_pred))
    print(cm)
    print(f1_score(y_test, y_pred))
    print("hidden_layer 개수: ", n_hiddens)


if __name__ == "__main__":
    X, y = get_fraud(sampling="smote")
    n = DNN_model(X=X, y=y)

    n.train()

    test_evaluate = n.eval_test()
    val_evaluate = n.eval_val()

    print("accuracy for Test set is", test_evaluate)
    print("accuracy for Val set is", val_evaluate)

    s=time.ctime()
    n.dnn_model.save(s+".h5")
    with open("dnn_model_params.pkl", "wb") as f:
        pkl.dump(
            {
                "X_test": n.X_test,
                "y_test": n.y_test,
                "X_val": n.X_val,
                "y_val": n.y_val,
                "n_hiddens": n.n_hiddens,
            },
            f,
        )

    load_result()