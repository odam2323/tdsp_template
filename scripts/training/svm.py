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
from brain_tumor_classifier.models.svm import train_svm

# =====================================================
# 📂 RUTAS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TRAIN_PATH = DATA_DIR / "Training"

MODEL_PATH = ROOT_DIR / "models" / "modelo_svm.pkl"

# =====================================================
# 🚀 TRAINING
# =====================================================

if __name__ == "__main__":

    print("\n🚀 Cargando dataset de entrenamiento...")

    X_train, y_train = procesar_dataset(TRAIN_PATH)

    print("🧠 Entrenando SVM...")

    model = train_svm(X_train, y_train)

    # =====================================================
    # 💾 GUARDAR MODELO
    # =====================================================

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\n🏁 Modelo guardado en: {MODEL_PATH}")