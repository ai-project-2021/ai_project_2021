import keras
import pickle as pkl
import os
import plotly.express as px


def save_keras(_path, model, params):
    save(_path, model, "keras")


def load_keras(_path):
    return load(_path, "keras")


def save(_path, model, params, type=None):
    """Model Saver
    Args:
        _path (path): model path
        model (object): Pre-Train Model
        type (str): model type
    Raises:
        FileExistsError:
    """
    if os.path.isfile(_path):
        raise FileExistsError("이미 존재하는 파일입니다.")
    else:
        if type == "keras":
            model.save(_path, format="h5")
        else:
            pkl.dump(model, open(_path, "wb"))


def load(_path, type=None):
    """Model Loader
    Args:
        _path (str): model path
        type (str): model type
    Raises:
        FileExistsError: File {_path} does not exist
    Returns:
        object: model
    """
    if not os.path.isfile(_path):
        raise FileExistsError("존재하지 않는 파일입니다.")
    else:
        if type == "keras":
            return keras.models.load_model(_path)
        else:
            return pkl.load(open(_path, "rb"))