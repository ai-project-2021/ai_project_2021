import keras
import os


def save_model(_path, model):
    if os.path.isfile(_path):
        print("이미 존재하는 파일입니다.")
        return -1
    else:
        model.save(_path, format="h5")
        return 0


def load_model(_path):
    if os.path.isfile(_path):
        return keras.models.load_model(_path)
    else:
        print("존재하지 않는 파일입니다.")
        return -1
