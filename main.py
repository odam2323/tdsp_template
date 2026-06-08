from pathlib import Path
import pickle
import subprocess
import sys

from scripts.data_acquisition.main import pull_dataset
from brain_tumor_classifier.preprocessing.main import procesar_dataset
from scripts.eda.main import run_eda

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"
PROCESSED_DIR = ROOT_DIR / "docs" / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


def run_step(script_relative_path: str, label: str):
    script_path = ROOT_DIR / script_relative_path
    print(f"\n▶ {label}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR,
        check=True
    )


def main():
    print("=" * 50)
    print("DATA ACQUISITION")
    print("=" * 50)
    pull_dataset()

    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = procesar_dataset(DATA_DIR / "Training")
    X_test, y_test = procesar_dataset(DATA_DIR / "Testing")

    with open(PROCESSED_DIR / "dataset_train.pkl", "wb") as f:
        pickle.dump((X_train, y_train), f)

    with open(PROCESSED_DIR / "dataset_test.pkl", "wb") as f:
        pickle.dump((X_test, y_test), f)

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    print("\n" + "=" * 50)
    print("EDA")
    print("=" * 50)
    run_eda()

    print("\n" + "=" * 50)
    print("ENTRENAMIENTO")
    print("=" * 50)

    run_step("scripts/training/efficient.py", "EfficientNet")
    print("EfficientNet entrenada")

    run_step("scripts/training/svm.py", "SVM")
    print("SVM entrenada")

    print("\n" + "=" * 50)
    print("PRUEBA")
    print("=" * 50)

    run_step("scripts/evaluation/efficient.py", "Evaluación EfficientNet")
    print("EfficientNet testeada")

    run_step("scripts/evaluation/svm.py", "Evaluación SVM")
    print("SVM testeada")

    print("\nPipeline completado")


if __name__ == "__main__":
    main()