import numpy as np
import os

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

# macro_f1_score
from sklearn.metrics import f1_score, recall_score, precision_score

# loader.py 파일 import
from utils.loader import get_fraud, get_raw_data

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

# 예측 모델 개발 절차
# 1. 데이터 준비: __init__
#         iris = load_iris()

#         self.X = iris.data  # iris data input
#         self.y = iris.target    # iris target(label)
#         self.y_name = iris.target_names # iris target name

#         # 데이터를 학습 데이터와 검증 데이터로 분류
#         train_idx = np.array([i % 15 != 14 for i in range(self.y.shape[0])])
#         test_idx = ~train_idx

#         # 학습 데이터
#         self.X_train = self.X[train_idx]
#         self.Y_train = self.y[train_idx]

#         # 검증 데이터
#         self.X_test = self.X[test_idx]
#         self.Y_test = self.y[test_idx]
# 2. 모델 구축: create_model
        # model = Sequential()
        # model.add(BatchNormalization()) # 배치 정규화

        # # hidden layer만큼 Neural-Network 반복
        # for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
        #     model.add(Dense(input_dim = input_dim, units = self.n_hiddens[i]))
        #     model.add(Activation('relu'))   # activation: relu, sigmoid, softmax 등
        #     # model.add(kernel_initializer='random_normal')   # 가중치 초기화
        #     model.add(Dropout(self.p_keep))

        # model.add(Dense(units = self.n_out))
        # model.add(Activation(self.activation))
# 3. 모델 컴파일: create_model
#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 4. 모델 학습
#         model.fit(n.X_train, n.Y_train, epochs=epochs)
# 5. 모델 평가
#         test_loss, test_accuracy = model.evaluate(n.X_test, n.Y_test, verbose=2)
#         print('test loss : ', test_loss)
#         print('test accuracy : ', test_accuracy)
# 6. 예측



class NN: 

    def __init__(self):
        self.X, self.y = get_fraud()

        # 전체 데이터의 수: 5643개(4513+1130)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) # 학습 데이터: 80%, 검증 데이터: 20%

        sc = StandardScaler()
        self.X_train=sc.fit_transform(self.X_train)
        self.X_test=sc.fit_transform(self.X_test)

        self.n_in = 41    # 입력 데이터의 크기: 41 columns
        # print('input 데이터의 크기: ', self.n_in)

        self.n_hiddens = [250, 300]  # 각 은닉층의 뉴런 개수: 200, 200
        # print('각 은닉층의 뉴런 개수: ', self.n_hiddens)

        self.n_out = 1   # 출력 데이터의 개수: 1개(0 또는 1)
        # print('output 데이터의 개수', self.n_out)

        self.activation = 'relu'
        self.p_keep = 0.5    # dropout 확률의 비율

    # DNN 수행
    def create_model(self):
        model = Sequential()
        model.add(BatchNormalization()) # 배치 정규화

        # hidden layer만큼 Neural-Network 반복
        for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
            model.add(Dense(input_dim = input_dim, units = self.n_hiddens[i], kernel_initializer='random_uniform'))
            model.add(Activation('relu'))   # activation: relu, sigmoid, softmax 등
            # model.add()   # 가중치 초기화
            model.add(Dropout(self.p_keep))

        model.add(Dense(units = self.n_out))
        model.add(Activation(self.activation))
        # model = keras.Sequential([
        #     for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
        #         keras.layers.Dense(input_dim=input_dim, activation='relu'),
        #         keras.layers.Dense(self.n_in, activation='softmax')
        # ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1])
        # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        return model

#     # Simple-Neural-Network
#     def simpleDNN(self):
#         model=Sequential()
#         model.add(Dense(input_dim = self.n_in, units = self.n_hiddens[0]))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))

#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#         epochs = 50 # 최적값 찾기
#         batch_size = 200    # 최적값 찾기

#         model.fit(self.X_train, self.Y_train, epochs = epochs, batch_size = batch_size)
#         loss_and_metrics = model.evaluate(self.X_test, self.Y_test)

#         print('Loss :', loss_and_metrics[0])
#         print('Accuracy :', loss_and_metrics[1]*100)
        
#         return model

# model = NN().__init__()
# model = NN().simpleDNN()

#     # Deep-Neural-Network
#     def DNN(self):       
#         model = Sequential()
#         model.add(BatchNormalization()) # 배치 정규화

#         # hidden layer만큼 Neural-Network 반복
#         for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
#             model.add(Dense(input_dim = input_dim, units = self.n_hiddens[i]))
#             model.add(Activation('relu'))   # activation: relu, sigmoid 등
#             # model.add(kernel_initializer='random_normal')   # 가중치 초기화
#             model.add(Dropout(self.p_keep))

#         model.add(Dense(units = self.n_out))
#         model.add(Activation(self.activation))

#         # optimizer: SGD, adam, Nadam, NAG, RMSProp 등
#         model.compile(loss = 'binary_crossentropy', optimizer = SGD(lr=0.01), metrics = ['accuracy'])

#         epochs = 50 # 최적값 찾기
#         batch_size = 200    # 최적값 찾기

#         model.fit(self.X_train, self.Y_train, epochs = epochs, batch_size = batch_size)
#         loss_and_metrics = model.evaluate(self.X_test, self.Y_test)

#         print('Loss :', loss_and_metrics[0])
#         print('Accuracy :', loss_and_metrics[1]*100)

# model = NN().DNN()

if __name__ == "__main__" :
    
    n = NN()

    model = n.create_model()

    epochs = 1 

    # 훈련 단계
    model.fit(n.X_train, n.y_train, epochs=epochs)

    # 정확도 평가 단계
    test_evaluate = model.evaluate(n.X_test, n.y_test, verbose=2)

    print(model.predict(n.X_test))

    # print('test evaluate : ', test_evaluate)
    print('test loss : ', test_evaluate[0])
    print('test accuracy : ', test_evaluate[1])
    print('test f1 : ', test_evaluate[2])

