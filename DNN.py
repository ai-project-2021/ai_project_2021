import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import SGD
from tensorflow.python.keras import activations

class NN: 

    # 학습 데이터
    X_train = train_data.loc[:,train.columns !='fraud']
    Y_train = train_data['fraud'] 

    # 검증 데이터
    X_test = 
    Y_test = 

    n_in = len(X[0])    # 입력 데이터의 크기
    n_hiddens = []  # 각 은닉층의 뉴런 개수
    n_out = len(Y[0])   # 출력 데이터의 개수(0, 1)
    activation = 'relu'
    p_keep = 0.5    # dropout 확률의 비율

    # # Simple-Neural-Network
    # model=Sequential()
    # model.add(Dense(input_dim=n_in, units=n_hiddens))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    def DNN(self):
        # Deep-Neural-Network
        model = Sequential()
        model.add(BatchNormalization()) #   배치 정규화

        # hidden layer만큼 Neural-Network 반복
        for i, input_dim in enumerate([n_in] + n_hiddens[:-1]):
            model.add(Dense(input_dim = n_in, units = n_hiddens))
            model.add(Activation('relu'))   # activation: relu, sigmoid 등
            model.add(kernel_initializer='random_normal')   # 가중치 초기화
            model.add(Dropout(p_keep))

        model.add(Dense(units = n_out))
        model.add(Activation(activation))

        # optimizer: SGD, adam, Nadam, NAG, RMSProp 등
        model.compile(loss = 'binary_crossentropy', optimizer = SGD(lr=0.01), metrics = ['accuracy'])

        epochs = 50 # 최적값 찾기
        batch_size = 200    # 최적값 찾기

        model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size)
        loss_and_metrics = model.evaluate(X_test, Y_test)