from pathlib import Path
import pickle
import sys
import os

# =====================================================
# 🔥 PATH CONFIG
# =====================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(ROOT)

from brain_tumor_classifier.preprocessing.main import procesar_dataset
from brain_tumor_classifier.evaluation.metrics import evaluate_model

# =====================================================
# 📂 RUTAS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TEST_PATH = DATA_DIR / "Testing"

MODEL_PATH = ROOT_DIR / "models" / "modelo_svm.pkl"

# =====================================================
# 🧪 TEST
# =====================================================

if __name__ == "__main__":

    print("\n📥 Cargando dataset de test...")

    X_test, y_test = procesar_dataset(TEST_PATH)

    # =====================================================
    # 📦 CARGAR MODELO
    # =====================================================

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("🤖 Modelo cargado correctamente")

    # =====================================================
    # 📊 PREDICCIÓN + MÉTRICAS
    # =====================================================

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)

    print("\n📊 RESULTADOS EN TEST")
    print(f"Accuracy: {metrics['accuracy']}")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])