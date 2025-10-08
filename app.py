import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(page_title="Face Skin Disease Detection", page_icon="🧠", layout="centered")

# --- Load Model with caching ---
@st.cache_resource
def load_skin_model():
    model = load_model("FaceSkinDisease_MobileNetV2_DL.h5")  # Path to your trained model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_skin_model()

# Replace with your 7 classes
CLASSES = ['Chickenpox', 'Monkeypox', 'Herpes', 'Lupus', 'Melanoma', 'Measles', 'Scabies']

# ---------------------- UI DESIGN ----------------------
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        h1 {color: #1e3a8a; text-align: center;}
        .result-box {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .confidence {font-size: 1.1rem; font-weight: 500;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Face Skin Disease Detection System")
st.markdown("### Upload a face image to identify possible skin conditions.")
st.markdown("---")

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📸 Upload Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    st.markdown("### 🔍 Analyzing the image... Please wait.")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # ------------------ Preprocess Image ------------------
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------ Prediction ------------------
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_label = CLASSES[pred_index]
    confidence = float(np.max(preds) * 100)

    # ------------------ Display Result ------------------
    st.markdown("---")
    st.markdown(
        f"<div class='result-box'><h2>🩺 Predicted Disease: <span style='color:#2563eb'>{pred_label}</span></h2></div>",
        unsafe_allow_html=True
    )

    # Confidence color
    color = "green" if confidence >= 90 else "orange" if confidence >= 70 else "red"
    st.markdown(f"<p class='confidence'>Confidence: <span style='color:{color}'>{confidence:.2f}%</span></p>", unsafe_allow_html=True)
    st.progress(int(confidence))

    # Class-wise probabilities
    st.markdown("#### 📊 Class-wise Probabilities:")
    for cls, prob in zip(CLASSES, preds[0]):
        st.write(f"**{cls}**: {prob*100:.2f}%")
        st.progress(float(prob))

    st.markdown("---")
    st.info("⚠️ This system is for educational and research purposes only — not a medical diagnostic tool.")

else:
    st.info("Please upload an image to start detection.")
