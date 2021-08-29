from skimage import io
import base64
from tensorflow.keras.models import load_model
import numpy as np
import gradio
from src import moleimages

model = load_model("./models/mymodel-2.h5")

def predict(input):
    mimg = moleimages.MoleImages()
    X = mimg.load_image(input)
    y_pred = model.predict(X)
    return {"benign": float(y_pred[0][0]), "cancerous": float(1-y_pred[0][0])}


examples = [["benign.png"], ["cancerous.png"], ["demo1.jpg"], ["demo2.png"], ["demo3.jpg"]]

io = gradio.Interface(
    fn=predict, inputs='image', outputs='label', capture_session=True, examples=examples, server_port=3001, server_name="0.0.0.0",
    thumbnail="https://raw.githubusercontent.com/gradio-app/hub-skin-cancer/master/thumbnail.png", analytics_enabled=False,
    title="Identifying Skin Cancer", description="Predicts whether an image of skin is cancerous or not.")
io.launch(debug=False)
