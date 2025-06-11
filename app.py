import streamlit as st
from PIL import Image
import numpy as np
import keras
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="🔍 Anomaly-Detection-using-Teachable-Machine",
    page_icon="🧠",
    layout="wide"
)

# ---- CUSTOM CSS ----
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #00CCAA;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.3rem;
    }
    .stFileUploader {
        border: 2px dashed #00CCAA;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.title("🔍 Anomaly-Detection-using-Teachable-Machine")
st.markdown("""<h4 style='color:#AAAAAA;'>Upload a PCB image to classify it as Defective or Normal.</h4>""", unsafe_allow_html=True)

# ---- EXAMPLE IMAGES ----
st.markdown("### 🧾 Example Images")
col1, col2 = st.columns(2)

with col1:
    st.image("https://i.imgur.com/tMHD4uR.png", caption="✅ Normal PCB", use_column_width=True)

with col2:
    st.image("https://i.imgur.com/XbR8Myf.png", caption="❌ Defective PCB", use_column_width=True)

# ---- LOCAL EXAMPLE IMAGE FROM REPO ----
if os.path.exists("example_pcb.jpg"):
    st.markdown("### 🧪 Example Uploaded Image")
    st.image("example_pcb.jpg", caption="🧠 Repo Example: PCB Sample", use_column_width=True)

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("Upload a PCB Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Your Uploaded PCB", use_column_width=True)

    # ---- LOAD MODEL ----
    @st.cache_resource
    def load_model():
        with open("labels.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        model = keras.layers.TFSMLayer("model.savedmodel", call_endpoint="serving_default")
        return model, labels

    model, labels = load_model()

    # ---- PREDICT ----
    def predict(image, model, labels):
        img = image.resize((224, 224))
        img_array = np.asarray(img).astype(np.float32)
        norm_img = (img_array / 127.5) - 1
        data = np.expand_dims(norm_img, axis=0)
        result = model(data)
        prediction = list(result.values())[0].numpy()[0]
        idx = np.argmax(prediction)
        return labels[idx], prediction[idx]

    label, confidence = predict(image, model, labels)

    st.markdown("### 📊 Prediction Result")
    if label.lower() == "anomaly":
        st.error(f"❌ Defect Detected\n\nConfidence: `{confidence:.2%}`")
    else:
        st.success(f"✅ PCB is Normal\n\nConfidence: `{confidence:.2%}`")

else:
    st.info("⬆️ Upload a PCB image to start analysis. Or check the examples above.")
