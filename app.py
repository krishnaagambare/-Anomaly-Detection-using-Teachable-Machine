import streamlit as st
from PIL import Image
import numpy as np
import keras

# ---- PAGE CONFIG ----
st.set_page_config(page_title="🔍 Anomaly-Detection-using-Teachable-Machine", page_icon="🧠", layout="centered")


# ---- STYLING ----
st.markdown("""
    <style>
    .reportview-container {
        background-color: #0E1117;
        color: white;
    }
    .css-18e3th9 {
        padding: 2rem 1rem 10rem;
    }
    h1 {
        color: #00FFCC;
        font-size: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 style='text-align: center;'>🔍 Anomaly-Detection using Teachable Machine</h1>", unsafe_allow_html=True)
st.write("")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload a PCB Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

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

    # ---- RESULT DISPLAY ----
    st.markdown("### 🧪 Prediction Result")
    if label.lower() == "anomaly":
        st.error(f"❌ Defect Detected: **{label}**\n\nConfidence: `{confidence:.2%}`")
    else:
        st.success(f"✅ PCB is Normal\n\nConfidence: `{confidence:.2%}`")

else:
    st.markdown("⚠️ Please upload a PCB image to begin anomaly detection.")
