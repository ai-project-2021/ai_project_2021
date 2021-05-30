"""사용할 패키지 불러오기
"""
from re import X
import numpy as np
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


X, y = get_fraud()
# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                        #    n_redundant=2, n_repeated=0, n_classes=8,
                        #    n_clusters_per_class=1, random_state=0)
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy',
              min_features_to_select=min_features_to_select)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 학습 데이터: 80%, 테스트 데이터: 20%
        # random state: 재현 가능하도록 난수의 초기값을 설정해주는 것으로, 아무 숫자나 넣어주면 됨
        # print(len(self.X_train))    # 282331
        # print(len(self.y_train))    # 282331
        # print(len(self.X_test)) # 70583

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=1) # 학습 데이터, 검증 데이터
rfecv.fit(X_train, y_train)
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()

class NN:

    def __init__(self):
        """Create Dataset

        원본 데이터를 불러옵니다.
        불러 온 데이터로부터 training set과 test set을 생성합니다.
        효과적으로 학습할 수 있도록 데이터셋을 scaling 합니다.

        model 생성에 이용될 input layer, hidden layer, 그리고 output layer의 형태를 결정하고, 
        마지막 layer에 이용할 activation 함수와 dropout 확률의 비율을 설정합니다. 

        Arguments: 
            X: 
            y: 
            X_train: X의 training dataset(80%)
            X_test: X의 test dataset(20%)
            y_train: y의 training dataset(80%)
            y_test: y의 test dataset(20%)
            n_in: input data의 개수(column 크기)
            n_hiddens: hidden layer의 개수 및 각 layer의 뉴런 개수 
            n_out: output data의 개수
            activation: last layer의 activation function
            p_keep: dropout 확률의 비율

        Keyword Arguments: 

        Raises: 

        Returns: [none] -- [dataset 준비]

        """
        self.X, self.y = get_fraud()
        list(filter(func, X))
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y) # 학습 데이터: 80%, 테스트 데이터: 20%
        # random state: 재현 가능하도록 난수의 초기값을 설정해주는 것으로, 아무 숫자나 넣어주면 됨
        # print(len(self.X_train))    # 282331
        # print(len(self.y_train))    # 282331
        # print(len(self.X_test)) # 70583

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, stratify=self.y_train, random_state=1) # 학습 데이터, 검증 데이터

        sc = StandardScaler()
        self.X_train=sc.fit_transform(self.X_train)
        self.X_test=sc.transform(self.X_test)

        self.n_in = 41    # 입력 데이터의 크기: 41 columns
        # print('input 데이터의 크기: ', self.n_in)

        self.n_hiddens = [2048, 2048, 2048]  # 각 은닉층의 뉴런 개수
        # print('각 은닉층의 뉴런 개수: ', self.n_hiddens)

        self.n_out = 1   # 출력 데이터의 개수: 1개(0 또는 1)
        # print('output 데이터의 개수', self.n_out)

        self.activation = 'sigmoid'
        self.p_keep = 0.4    # dropout 확률의 비율

    # DNN 수행
    def create_model(self):
        """Create Dnn Model

        Arguments: 


        Keyword Arguments: 

        Raises: 

        Returns: [model] -- DNN model

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

        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', f1])
        # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        return model



