import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import json
from PIL import Image
from pathlib import Path
import random
import pandas as pd

# =====================================================
# 🧠 CONFIG
# =====================================================

st.set_page_config(page_title="Brain Tumor - TEST MODE", layout="centered")

st.title("🧠 Brain Tumor Classifier")
st.write("CNN vs SVM + Test Dataset Mode")

# =====================================================
# 📌 CLASES
# =====================================================

try:
    with open("models/class_names.json", "r") as f:
        CLASS_NAMES = json.load(f)
except:
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# =====================================================
# 📁 TEST DATASET
# =====================================================

TEST_PATH = Path("docs/data/datos/Testing")

def get_test_images():
    images = []
    for folder in TEST_PATH.iterdir():
        if folder.is_dir():
            for img in folder.glob("*"):
                images.append(img)
    return images

test_images = get_test_images()

# =====================================================
# 🤖 MODELOS
# =====================================================

@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("models/efficientnet.keras")

cnn_model = load_cnn()

@st.cache_resource
def load_svm():
    try:
        with open("models/modelo_svm.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

svm_model = load_svm()

# =====================================================
# 🧠 PREPROCESS CNN
# =====================================================

def preprocess_cnn(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))

    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# =====================================================
# 🧠 SVM FEATURE EXTRACTION (FIXED SIZE)
# =====================================================

def extract_features(image):
    # ⚠️ IMPORTANTE: debe coincidir con training
    image = image.convert("L")
    image = image.resize((90, 90))  # 🔥 AJUSTADO a 8100 features

    img = np.array(image)

    return img.flatten().reshape(1, -1)

# =====================================================
# 🎲 UI: TEST MODE
# =====================================================

st.sidebar.header("📂 Modo de prueba")

mode = st.sidebar.radio(
    "Selecciona fuente de imagen:",
    ["Upload", "Random Test Image", "Select Test Image"]
)

image = None

# =====================================================
# UPLOAD MODE
# =====================================================

if mode == "Upload":
    uploaded = st.file_uploader("Sube imagen MRI", type=["jpg","png","jpeg"])
    if uploaded:
        image = Image.open(uploaded)

# =====================================================
# RANDOM MODE
# =====================================================

elif mode == "Random Test Image":
    if st.sidebar.button("🎲 Generar imagen"):
        image_path = random.choice(test_images)
        image = Image.open(image_path)
        st.sidebar.write(image_path.name)

# =====================================================
# SELECT MODE
# =====================================================

elif mode == "Select Test Image":
    options = [str(p) for p in test_images[:200]]
    selected = st.sidebar.selectbox("Selecciona imagen", options)

    if selected:
        image = Image.open(selected)

# =====================================================
# DISPLAY IMAGE
# =====================================================

if image is not None:
    st.image(image, caption="Imagen seleccionada", use_column_width=True)

# =====================================================
# 🚀 PREDICCIÓN
# =====================================================

if image is not None and st.button("🔍 Comparar modelos"):

    results = {}

    # =================================================
    # CNN
    # =================================================
    try:
        x_cnn = preprocess_cnn(image)
        pred_cnn = cnn_model.predict(x_cnn)

        class_id = int(np.argmax(pred_cnn))

        results["CNN"] = {
            "class": CLASS_NAMES[class_id],
            "confidence": float(np.max(pred_cnn))
        }

    except Exception as e:
        results["CNN"] = {"error": str(e)}

    # =================================================
    # SVM
    # =================================================
    try:
        if svm_model is not None:
            x_svm = extract_features(image)

            pred_svm = svm_model.predict(x_svm)[0]

            results["SVM"] = {
                "class": CLASS_NAMES[int(pred_svm)] if isinstance(pred_svm, (int, np.integer)) else str(pred_svm)
            }

        else:
            results["SVM"] = {"error": "Modelo no cargado"}

    except Exception as e:
        results["SVM"] = {"error": str(e)}

    # =================================================
    # RESULTADOS
    # =================================================

    st.divider()
    st.subheader("📊 Comparación")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 CNN")
        st.write(results["CNN"])

    with col2:
        st.markdown("### 📊 SVM")
        st.write(results["SVM"])

    # =================================================
    # RESUMEN
    # =================================================

    st.divider()
    st.subheader("📌 Resumen")

    summary = pd.DataFrame([
        ["CNN",
         results["CNN"].get("class", "error"),
         results["CNN"].get("confidence", None)],
        ["SVM",
         results["SVM"].get("class", "error"),
         None]
    ], columns=["Modelo", "Clase", "Confianza"])

    st.dataframe(summary)

# =====================================================
# FOOTER
# =====================================================

st.divider()
st.caption("CNN vs SVM + Test Mode - Brain Tumor Classification")