import gradio
from skimage import io
import base64
from tensorflow.keras.models import load_model
from src.moleimages import MoleImages
import numpy as np


def load():
    model = load_model("./models/mymodel-2.h5")
    return model


def predict(inp):
    inp = inp.split(';')[1]
    inp = inp.split(',')[1]
    if isinstance(inp, bytes):
        inp = inp.decode("utf-8")

    inp = base64.b64decode(inp)
    img = io.imread(inp, plugin='imageio')

    mimg = MoleImages()
    X = mimg.load_image(img)

    model = load()
    y_pred = model.predict(X)
    return np.asarray([y_pred, 1 - y_pred])


def no_pp(inp):
    return inp


INPUT = gradio.inputs.ImageUpload(preprocessing_fn=no_pp)
OUTPUT = gradio.outputs.Label(label_names=["benign", "malign"],
                            num_top_classes=2)

