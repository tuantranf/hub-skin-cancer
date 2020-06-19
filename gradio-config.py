from skimage import io
import base64
from tensorflow.keras.models import load_model
import numpy as np
import gradio
from src import moleimages

def load():
    model = load_model("./models/mymodel-2.h5")
    return model


def predict(inp, model):
    inp = inp.split(';')[1]
    inp = inp.split(',')[1]
    if isinstance(inp, bytes):
        inp = inp.decode("utf-8")

    inp = base64.b64decode(inp)
    img = io.imread(inp, plugin='imageio')

    mimg = moleimages.MoleImages()
    X = mimg.load_image(img)

    y_pred = model.predict(X)
    return {"benign": float(y_pred[0][0]), "cancerous": float(1-y_pred[0][0])}


INPUTS = gradio.inputs.ImageIn(cast_to="base64")
OUTPUTS = gradio.outputs.Label()
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS,
                             load_fn=load, capture_session=True)

INTERFACE.launch(inbrowser=True)
