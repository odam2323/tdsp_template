from pathlib import Path
import sys
import pickle
import subprocess

from scripts.data_acquisition.main import pull_dataset
from src.brain_tumor_classifier.preprocessing.main import procesar_dataset
from scripts.eda.main import run_eda


def main():

    print("=" * 50)
    print("DATA ACQUISITION")
    print("=" * 50)

    pull_dataset()

    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"
    PROCESSED_DIR = ROOT_DIR / "docs" / "data" / "processed"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
    print("ENTRENAMIENTO (No incluido dado el .gitignore)")
    print("=" * 50)

    script_path_entrenamiento_efficient = Path("scripts/training/efficient.py")

    subprocess.run([sys.executable, str(script_path_entrenamiento_efficient)])

    print("EfficientNet entrenada")

    script_path_entrenamiento_svm = Path("scripts/training/svm.py")

    subprocess.run([sys.executable, str(script_path_entrenamiento_svm)])

    print("SVM entrenada")

    print("\n" + "=" * 50)
    print("PRUEBA")
    print("=" * 50)

    script_path_prueba_efficient = Path("scripts/evaluation/efficient.py")

    subprocess.run([sys.executable, str(script_path_prueba_efficient)])

    print("EfficientNet testeada")

    script_path_prueba_svm = Path("scripts/evaluation/svm.py")

    subprocess.run([sys.executable, str(script_path_prueba_svm)])

    print("SVM testeada")



    print("\nPipeline completado")


if __name__ == "__main__":
    main()