import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from streamlit_option_menu import option_menu
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="BoneScan AI",
    layout="centered"
)

# -----------------------------
# CONSTANTS
# -----------------------------
MODEL_PATH = "model/bone_fracture_efficientnetb4.keras"
IMAGE_SIZE = (380, 380)
CLASSES = ["Fractured", "Not Fractured"]
CLASS_INFO = {
    "Fractured": {
        "color": "#e74c3c",
        "bg": "#fdf0ef",
        "severity": "High",
        "desc": "A bone fracture has been detected in this X-ray image. Fractures may range from hairline cracks to complete breaks and require immediate medical attention. Please consult an orthopaedic specialist for proper diagnosis and treatment."
    },
    "Not Fractured": {
        "color": "#27ae60",
        "bg": "#eafaf1",
        "severity": "None",
        "desc": "No fracture was detected in this X-ray image. The bone structure appears to be intact. Always confirm results with a certified radiologist or orthopaedic physician before making any medical decisions."
    }
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def run_inference(model, image: Image.Image):
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0][0]  # single sigmoid output

    if pred > 0.5:
        label = "Not Fractured"
        confidence = float(pred)
    else:
        label = "Fractured"
        confidence = float(1 - pred)

    all_probs = [1 - float(pred), float(pred)]  # [Fractured, Not Fractured]

    return label, confidence, all_probs

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("BoneScan AI")
st.sidebar.markdown(
    """
**AI-Powered Bone Fracture Classification**

This system uses a deep learning model trained on
multi-region bone X-ray images to classify whether
a fracture is present or not.

**Tech Stack:** Python, TensorFlow, EfficientNetB4, Streamlit
"""
)
st.sidebar.markdown("### Capabilities")
st.sidebar.markdown(
    """
- X-ray image upload and analysis
- Binary fracture classification
- Per-class confidence breakdown
- Research-grade inference pipeline
"""
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
[GitHub](https://github.com/Fiazbhk) |
[LinkedIn](https://www.linkedin.com/in/fiazbhk/) |
[Dataset](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)
"""
)

# -----------------------------
# TITLE
# -----------------------------
st.title("BoneScan AI")

# -----------------------------
# TOP HORIZONTAL MENU
# -----------------------------
selected_tab = option_menu(
    menu_title=None,
    options=["Fracture Detection", "About the Model"],
    icons=["activity", "info-circle"],
    orientation="horizontal",
    default_index=0
)

# -----------------------------
# TAB 1: FRACTURE DETECTION
# -----------------------------
if selected_tab == "Fracture Detection":
    st.subheader("X-Ray Image Analysis")
    st.warning(
        "Medical Disclaimer: This tool is for educational and research purposes only. "
        "It is not a substitute for professional medical diagnosis. "
        "Always consult a qualified radiologist or orthopaedic physician."
    )

    uploaded_file = st.file_uploader(
        "Upload Bone X-Ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
            st.caption(f"File: {uploaded_file.name} · {uploaded_file.size // 1024} KB")

        with col2:
            st.markdown("#### Run Analysis")
            st.markdown(
                "The model will classify this X-ray image as either "
                "Fractured or Not Fractured using EfficientNetB4."
            )
            st.divider()

            if st.button("Generate Diagnostic Result"):
                with st.spinner("Analysing X-ray image..."):
                    try:
                        model = load_model()
                        label, confidence, all_probs = run_inference(model, image)
                        info = CLASS_INFO[label]

                        result_msg = (
                            f"### ASSESSMENT: {label.upper()}\n"
                            f"Confidence: {confidence * 100:.2f}% · Severity: {info['severity']}"
                        )

                        if label == "Not Fractured":
                            st.success(result_msg)
                        else:
                            st.error(result_msg)

                        st.markdown(
                            f"<div style='border-left: 4px solid {info['color']};"
                            f"padding: 0.6rem 0.9rem; background: {info['bg']};"
                            f"border-radius: 6px; font-size: 0.85rem; color: #374151; margin-top: 0.5rem'>"
                            f"{info['desc']}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("#### Confidence Breakdown")
                        for cls, prob in zip(CLASSES, all_probs):
                            pct = float(prob) * 100
                            is_top = cls == label
                            bar_color = CLASS_INFO[cls]["color"] if is_top else "#d1d5db"
                            fw = "700" if is_top else "400"
                            fc = "#1a1a2e" if is_top else "#6b7280"
                            st.markdown(
                                f"""<div style="margin-bottom: 0.6rem;">
                                    <div style="display: flex; justify-content: space-between;
                                                font-size: 0.82rem; font-weight: {fw}; color: {fc};">
                                        <span>{cls}</span>
                                        <span>{pct:.2f}%</span>
                                    </div>
                                    <div style="background: #f3f4f6; border-radius: 4px; height: 8px; margin-top: 3px;">
                                        <div style="width: {min(pct, 100):.1f}%; height: 8px; border-radius: 4px;
                                                    background: {bar_color};"></div>
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

            else:
                st.info("Click Generate Diagnostic Result to analyse the X-ray.")

# -----------------------------
# TAB 2: ABOUT
# -----------------------------
if selected_tab == "About the Model":
    st.subheader("Model Information")
    st.markdown(
        """
**Algorithm:** EfficientNetB4 with Transfer Learning (ImageNet weights)

**Training Strategy:** Two-phase training — frozen base then full fine-tuning

**Preprocessing:** EfficientNet-specific preprocess_input

**Input Size:** 380 x 380 x 3

**Dataset:** Fracture Multi Region X-Ray Data by B. Madushani Rodrigo (Kaggle)

**Target Classes:**
- 0 → Fractured
- 1 → Not Fractured

**Performance:**
- Validation Accuracy: 98.67%
- Test Accuracy: 98.02%
- Test Loss: 0.0506
- AUC Score: 0.9985
- Correct Predictions: 496 / 506

**Per Class Metrics:**

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Fractured | 0.97 | 0.98 | 0.98 |
| Not Fractured | 0.98 | 0.98 | 0.98 |

**Notes:**
- Predictions are probabilistic, not diagnostic
- Intended for educational and research use only
- Not a substitute for professional medical evaluation
- Model performs best on standard frontal X-ray images
"""
    )