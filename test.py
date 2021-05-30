"""사용할 패키지 불러오기
"""
from re import X
from imblearn.over_sampling._smote.base import SMOTE
from keras import optimizers
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn import datasets
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import SGD
from tensorflow.python.keras import activations
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from keras.models import load_model
from IPython.display import SVG
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

# macro_f1_score
from sklearn.metrics import f1_score

# loader.py 파일 import
from utils.loader import get_fraud

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class NN:

    def __init__(self):
        """Create Dataset
        원본 데이터를 불러옵니다.
        불러 온 데이터로부터 training set과 test set을 생성합니다.
        효과적으로 학습할 수 있도록 데이터셋을 scaling 합니다.

        model 생성에 이용될 input layer, hidden layer, 그리고 output layer의 형태를 결정하고, 
        마지막 layer에 이용할 activation 함수와 dropout 확률의 비율을 설정합니다. 

        """
        self.X, self.y = get_fraud(sampling="smote")

        sc = StandardScaler()
        self.X=sc.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y) # 학습 데이터: 80%, 테스트 데이터: 20%
        # random state: 재현 가능하도록 난수의 초기값을 설정해주는 것으로, 아무 숫자나 넣어주면 됨

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, stratify=self.y_train, random_state=1) # 학습 데이터, 검증 데이터
        self.k_fold = StratifiedKFold(n_splits=5)

        # for X_train_idx, y_train_idx in skf.split(self.X_train, self.y_train):

        # self.kf_X_train, self.kf_y_train

        # sc = StandardScaler()
        # self.X_train=sc.fit_transform(self.X_train)
        # self.X_test=sc.transform(self.X_test)

            # 입력 데이터의 크기: 41 columns
        self.n_in = 41
        # print('input 데이터의 크기: ', self.n_in)

        # self.n_hiddens = [2048, 2048, 2048]  # 각 은닉층의 뉴런 개수
        self.n_hiddens = [20, 20]
        # print('각 은닉층의 뉴런 개수: ', self.n_hiddens)

        self.n_out = 1   # 출력 데이터의 개수: 1개(0 또는 1)
        # print('output 데이터의 개수', self.n_out)

        self.activation = 'sigmoid'
        self.p_keep = 0.4    # dropout 확률의 비율
        # self.model = FraudDNN(d_in = self.n_in, d_out = self.n_out, hidden_layer=self.n_hiddens, p_keep = self.p_keep)

        self.dnn_model = self.create_model()

    def kfold_train(self):
        return cross_val_score(self.dnn_model, self.X_train, self.y_train, cv=self.k_fold).mean()

    def kfold_test(self):
        return cross_val_score(self.dnn_model, self.X_test, self.y_test, cv=self.k_fold).mean()

    def kfold_predict(self):
        return cross_val_predict(self.dnn_model, self.X_test, self.y_test, cv=self.k_fold)

    # DNN 수행
    def create_model(self):
        """DNN model을 구현하는 함수입니다. 은닉층을 3개, 


        Returns:
            model: 
        """
        model = Sequential()
        model.add(BatchNormalization()) # 배치 정규화

        # hidden layer만큼 Neural-Network 반복
        for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
            model.add(Dense(input_dim = input_dim, units = self.n_hiddens[i], kernel_initializer='random_uniform'))
            model.add(Activation('relu'))   # activation: relu, sigmoid, softmax 등
            model.add(Dropout(self.p_keep))

        model.add(Dense(units = self.n_out))
        model.add(Activation(self.activation))

        # opt = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', f1, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        return model

# class FraudDNN(Sequential) : 
#     """DNN model을 구현하는 함수입니다. 은닉층을 3개, 


#     Returns:
#         model: 
#     """
#     def __init__(self, d_in, d_out, hidden_layer, p_keep) : 
#         super(Sequential).__init__()
#         self.d_in = d_in
#         self.d_out = d_out
#         self.hidden_layers = hidden_layer
#         self.p_keep = p_keep
#         self.act = 'relu'
#         self.last_act = "sigmoid"
#         self.opt = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#         self._build()

#     def _build(self) : 
#         self.add(BatchNormalization()) # 배치 정규화
#         self.add(Dense(input_dim = self.d_in, units = self.hidden_layers[0], kernel_initializer='random_uniform', activation=self.act))
#         self.add(Dropout(self.p_keep))

#         # hidden layer만큼 Neural-Network 반복
#         for units_ in self.hidden_layers[:-1]:
#             self.add(Dense(units = units_, kernel_initializer='random_uniform', activation=self.act))
#             self.add(Dropout(self.p_keep))

#         self.add(Dense(units = self.d_out, kernel_initializer='random_uniform', activation=self.last_act))
        
#         self.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy', f1])
    

if __name__ == "__main__" :
    """[summary]
    """

    n = NN()

    model = n.create_model()
    # model = FraudDNN(Sequential, d_in=n.n_in, d_out=n.n_out, hidden_layer=n.n_hiddens, p_keep=n.p_keep)

    batch_size = 512
    epochs = 1000
    class_weights = {1: 0.9, 0: 0.1}
    callbacks = EarlyStopping(monitor='f1', patience=50, verbose=1)
    
    # sc = StandardScaler()
    # sc_X_train=sc.fit_transform(n.X_train)
    # sc_X_test=sc.transform(n.X_test)

    # # 훈련 단계
    hist = model.fit(n.X_train, n.y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weights, validation_data=(n.X_val, n.y_val), verbose=1)
    print(hist.history)
    # hist = model.fit(sc_X_train, n.y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weights, validation_data=(n.X_val, n.y_val), verbose=1)
    # model.fit(n.X_test, n.y_test, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weights, verbose=1)
    
    # 정확도 평가 단계
    # train_evaluate = model.evaluate(n.X_train, n.y_train)
    # test_evaluate = model.evaluate(n.X_test, n.y_test)
    # val_evaluate = model.evaluate(n.X_val, n.y_val)

    # print('accuracy for Train set is', train_evaluate)
    # print('accuracy for Test set is', test_evaluate)
    # print('accuracy for Val set is', val_evaluate)
    # print(model.predict(n.X_test))

    # y_pred = model.predict(n.X_test)
    # y_pred = np.argmax(y_pred1, axis = 1)
    # print(y_pred1)
    # cm = confusion_matrix(n.y_test, y_pred.round())

    # model.save("dnn_model.h5")
    # model = load_model('dnn_model.h5', custom_objects={"f1":f1})
    # print('test evaluate : ', test_evaluate)
    # print('test loss : ', test_evaluate[0])
    # print('test accuracy : ', test_evaluate[1])
    # print('test f1 : ', test_evaluate[2])
    # print('val evaluate : ', val_evaluate)
    # print('val loss : ', val_evaluate[0])
    # print('val accuracy : ', val_evaluate[1])
    # print('val f1 : ', val_evaluate[2])

    # print(classification_report(n.y_test, y_pred.round()))
    # print(cm)
    # print(f1_score(n.y_test, y_pred.round()))
    # print('hidden_layer 개수: ', n.n_hiddens)

    fig, result_ax = plt.subplots(figsize=(15, 5))
    result_ax.plot(hist.history['precision'], 'y', label='precision')
    result_ax.plot(hist.history['recall'], 'y', label='recall')
    # result_ax.plot(hist.history['f1'], 'y', label='accuracy')


    # fig, loss_ax = plt.subplots(figsize=(15, 5))
    # loss_ax.plot(hist.history['loss'], 'y', label = 'train_loss')   # 훈련 데이터의 loss
    # loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss') # 검증 데이터의 loss

    # loss_ax.set_xlabel('epoch')
    # loss_ax.set_ylabel('loss')

    # loss_ax.legend(loc='upper left')

    plt.show()