import streamlit as st
from PIL import Image
import numpy as np
import keras

st.set_page_config(page_title="PCB Anomaly Detection", page_icon="🔍")

@st.cache_resource
def load_model():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    model = keras.layers.TFSMLayer("my_model", call_endpoint="serving_default")
    return model, labels

def predict(image, model, labels):
    img = image.resize((224, 224))
    img_array = np.asarray(img).astype(np.float32)
    norm_img = (img_array / 127.5) - 1
    data = np.expand_dims(norm_img, axis=0)
    result = model(data)
    prediction = list(result.values())[0].numpy()[0]
    idx = np.argmax(prediction)
    return labels[idx], prediction[idx]

st.title("🔍 PCB Anomaly Detector")

uploaded = st.file_uploader("Upload a PCB Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model, labels = load_model()
    label, confidence = predict(image, model, labels)

    st.subheader("Prediction")
    if label.lower() == "anomaly":
        st.error(f"Defective Detected - Confidence: {confidence:.2%}")
    else:
        st.success(f"Normal PCB - Confidence: {confidence:.2%}")
