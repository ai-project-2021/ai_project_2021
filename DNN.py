import numpy as np
import tensorflow as tf
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
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# macro_f1_score
from sklearn.metrics import f1_score, recall_score, precision_score

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
        self.X, self.y = get_fraud()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y) # 학습 데이터: 80%, 검증 데이터: 20%
        # print(len(self.X_train))    # 282331
        # print(len(self.y_train))    # 282331
        # print(len(self.X_test)) # 70583

        sc = StandardScaler()
        self.X_train=sc.fit_transform(self.X_train)
        self.X_test=sc.fit_transform(self.X_test)

        self.n_in = 41    # 입력 데이터의 크기: 41 columns
        # print('input 데이터의 크기: ', self.n_in)

        self.n_hiddens = [2048, 2048, 2048]  # 각 은닉층의 뉴런 개수
        # print('각 은닉층의 뉴런 개수: ', self.n_hiddens)

        self.n_out = 1   # 출력 데이터의 개수: 1개(0 또는 1)
        # print('output 데이터의 개수', self.n_out)

        self.activation = 'sigmoid'
        self.p_keep = 0.4    # dropout 확률의 비율

        # self.class_weights = class_weight.compute_class_weight('balanced', np.unique(self.y_train), self.y_train)


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

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1])
        # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        return model

if __name__ == "__main__" :

    n = NN()

    model = n.create_model()

    batch_size = 512
    epochs = 1000
    class_weights = {1: 0.9, 0: 0.1}
    callbacks = EarlyStopping(monitor='f1', patience=50, verbose=1)

    # # 훈련 단계
    # model.fit(n.X_train, n.y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weights, verbose=1)

    model.save("my_model.h5")
    reconstructed_model = keras.models.load_model("my_model.h5")

    np.testing.assert_allclose(
        model.predict(n.X_test), reconstructed_model.predict(n.X_test)
    )

    # 정확도 평가 단계
    # train_evaluate = model.evaluate(n.X_train, n.y_train)
    test_evaluate = model.evaluate(n.X_test, n.y_test)

    # print('accuracy for Train set is', train_evaluate)
    print('accuracy for Test set is', test_evaluate)
    # print(model.predict(n.X_test))

    y_pred = model.predict(n.X_test)
    # y_pred = np.argmax(y_pred1, axis = 1)
    # print(y_pred1)
    cm = confusion_matrix(model.y_test, y_pred.round())

    # print('test evaluate : ', test_evaluate)
    print('test loss : ', test_evaluate[0])
    print('test accuracy : ', test_evaluate[1])
    print('test f1 : ', test_evaluate[2])
    print(classification_report(n.y_test, y_pred.round()))
    print(cm)
    print(f1_score(n.y_test, y_pred.round()))
    print('hidden_layer 개수: ', n.n_hiddens)
   
    # model.save('/Users/kimsujin/Desktop/2021_1학기수업/인공지능/ai_project_2021', format="h5")