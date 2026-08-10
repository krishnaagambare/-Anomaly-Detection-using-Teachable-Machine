import os

import keras
import numpy as np
import streamlit as st
from PIL import Image


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PCB Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 PCB Anomaly Detection")

st.markdown(
    """
    <h4 style='color:#AAAAAA;'>
    Upload a PCB image to classify it as Normal or Defective.
    </h4>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

MODEL_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"


# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------

def load_labels():
    if not os.path.exists(LABELS_PATH):
        return ["Normal", "Anomaly"]

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory '{MODEL_PATH}' was not found."
        )

    model = keras.layers.TFSMLayer(
        MODEL_PATH,
        call_endpoint="serving_default"
    )

    labels = load_labels()

    return model, labels


# --------------------------------------------------
# EXAMPLE IMAGES
# --------------------------------------------------

st.markdown("### 🧾 Example Images")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "https://i.imgur.com/tMHD4uR.png",
        caption="✅ Normal PCB",
        width="stretch"
    )

with col2:
    st.image(
        "https://i.imgur.com/XbR8Myf.png",
        caption="❌ Defective PCB",
        width="stretch"
    )


# --------------------------------------------------
# LOCAL EXAMPLE
# --------------------------------------------------

if os.path.exists("example_pcb.jpg"):
    st.markdown("### 🧪 Example PCB")

    st.image(
        "example_pcb.jpg",
        caption="PCB Sample",
        width="stretch"
    )


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PCB Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="📷 Uploaded PCB",
        width="stretch"
    )

    try:

        model, labels = load_model()

        # Resize to Teachable Machine input size
        img = image.resize((224, 224))

        img_array = np.asarray(img).astype(np.float32)

        # Teachable Machine normalization
        normalized = (img_array / 127.5) - 1

        data = np.expand_dims(normalized, axis=0)

        # Run inference
        result = model(data)

        prediction = list(result.values())[0].numpy()[0]

        index = int(np.argmax(prediction))

        confidence = float(prediction[index])

        label = labels[index] if index < len(labels) else f"Class {index}"


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.markdown("### 📊 Prediction Result")

        if label.lower() in ["anomaly", "defective", "defect"]:

            st.error(
                f"❌ Defect Detected\n\n"
                f"**Class:** {label}\n\n"
                f"**Confidence:** {confidence:.2%}"
            )

        else:

            st.success(
                f"✅ PCB is Normal\n\n"
                f"**Class:** {label}\n\n"
                f"**Confidence:** {confidence:.2%}"
            )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)


else:

    st.info(
        "⬆️ Upload a PCB image to start analysis."
    )
