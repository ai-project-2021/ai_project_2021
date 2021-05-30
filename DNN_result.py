"""사용할 패키지 불러오기
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

class DNN_model:

    def __init__(self,X,y):
        """Create Dataset
        원본 데이터를 불러옵니다.
        불러 온 데이터로부터 training set과 test set을 생성합니다.
        효과적으로 학습할 수 있도록 데이터셋을 scaling 합니다.

        model 생성에 이용될 input layer, hidden layer, 그리고 output layer의 형태를 결정하고, 
        마지막 layer에 이용할 activation 함수와 dropout 확률의 비율을 설정합니다. 

        """
        self.k_fold = StratifiedKFold(n_splits=5)
        self.get_data(X, y)
        self.n_in = self.X_train.shape[1]
        self.n_hiddens = [20, 20]
        self.n_out = 1 
        self.activation = 'sigmoid'
        self.p_keep = 0.4
        self.dnn_model = self.create_model()

    def get_data(self, X, y):
        sc = StandardScaler()
        X=sc.fit_transform(X)

        self.X_train_, self.X_test, self.y_train_, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_, self.y_train_, test_size=0.2, stratify=self.y_train_, random_state=1) 
        
    def kfold_train(self):
        return cross_val_score(self.dnn_model, self.X_train, self.y_train, cv=self.k_fold).mean()

    def kfold_test(self):
        return cross_val_score(self.dnn_model, self.X_test, self.y_test, cv=self.k_fold).mean()

    def kfold_predict(self):
        return cross_val_predict(self.dnn_model, self.X_test, self.y_test, cv=self.k_fold)

    def create_model(self):
        """DNN model을 구현하는 함수입니다.



        Returns:
            model: Deep Neural Network model을 반환합니다.
        """
        model = Sequential()
        model.add(BatchNormalization())

        for i, input_dim in enumerate([self.n_in] + self.n_hiddens[:-1]):
            model.add(Dense(input_dim = input_dim, units = self.n_hiddens[i], kernel_initializer='random_uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(self.p_keep))

        model.add(Dense(units = self.n_out))
        model.add(Activation(self.activation))

        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', f1, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def loss_graph(self):
        fig, loss_ax = plt.subplots(figsize=(15, 5))
        loss_ax.plot(hist.history['loss'], 'y', label = 'train_loss')   
        loss_ax.plot(hist.history['val_loss'], 'r', label = 'val loss')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')

        loss_ax.legend(loc='upper left')

        plt.show()

    def result_graph(self):
        fig, result_ax = plt.subplots(figsize=(15, 5))
        result_ax.plot(hist.history['precision_1'], 'y', label='precision')
        result_ax.plot(hist.history['recall_1'], 'r', label='recall')
        result_ax.plot(hist.history['f1'], 'b', label='f1')

        result_ax.set_xlabel('epoch')
        result_ax.set_ylabel('y')

        result_ax.legend(loc='upper left')

        plt.show()

if __name__ == "__main__" :
    """[summary]
    """
    X, y = get_fraud(sampling="smote")
    n = DNN_model(X=X, y=y)
    model = n.dnn_model

    batch_size = 512
    epochs = 2000
    class_weights = {1: 0.9, 0: 0.1}
    callbacks = EarlyStopping(monitor='f1', patience=50, verbose=1)
    
    hist = model.fit(n.X_train, n.y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, class_weight=class_weights, validation_data=(n.X_val, n.y_val), verbose=1)
    
    # train_evaluate = model.evaluate(n.X_train, n.y_train)
    test_evaluate = model.evaluate(n.X_test, n.y_test)
    val_evaluate = model.evaluate(n.X_val, n.y_val)

    print('accuracy for Test set is', test_evaluate)
    print('accuracy for Val set is', val_evaluate)

    y_pred = model.predict(n.X_test)
    cm = confusion_matrix(n.y_test, y_pred.round())

    model.save("dnn_model.h5")
    model = load_model('dnn_model.h5', custom_objects={"f1":f1})
    print('test evaluate : ', test_evaluate)
    print('test loss : ', test_evaluate[0])
    print('test accuracy : ', test_evaluate[1])
    print('test f1 : ', test_evaluate[2])
    print('val evaluate : ', val_evaluate)
    print('val loss : ', val_evaluate[0])
    print('val accuracy : ', val_evaluate[1])
    print('val f1 : ', val_evaluate[2])

    print(classification_report(n.y_test, y_pred.round()))
    print(cm)
    print(f1_score(n.y_test, y_pred.round()))
    print('hidden_layer 개수: ', n.n_hiddens)

    