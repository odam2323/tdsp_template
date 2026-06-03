from pathlib import Path
import pickle
import sys
import os

# 🔥 CONFIGURAR PATH PRIMERO (ANTES DE IMPORTS)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(ROOT)

# 🔥 AHORA SÍ IMPORTAR
from brain_tumor_classifier.preprocessing.main import procesar_dataset
from brain_tumor_classifier.models.svm import train_svm
from brain_tumor_classifier.evaluation.metrics import evaluate_model

# 🔥 DEFINIR ROOT DEL PROYECTO
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TRAIN_PATH = DATA_DIR / "Training"
TEST_PATH = DATA_DIR / "Testing"

MODEL_PATH = ROOT_DIR / "models" / "modelo_svm.pkl"


if __name__ == "__main__":

    X_train, y_train = procesar_dataset(TRAIN_PATH)
    X_test, y_test = procesar_dataset(TEST_PATH)

    model = train_svm(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)

    print(metrics["accuracy"])
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)