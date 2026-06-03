from pathlib import Path
import pickle
import sys

from brain_tumor_classifier.features.preprocessing import procesar_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / 'docs' / 'data' / 'datos'
PROCESSED_DIR = ROOT_DIR / 'docs' / 'data' / 'processed'


def guardar_dataset(X, y, ruta_archivo):
    with open(ruta_archivo, 'wb') as f:
        pickle.dump((X, y), f)


if __name__ == "__main__":

    if not DATA_DIR.exists():
        print(f"No existe: {DATA_DIR}")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # TRAIN
    X_train, y_train = procesar_dataset(DATA_DIR / "Training")
    guardar_dataset(X_train, y_train, PROCESSED_DIR / "dataset_train.pkl")

    # TEST
    X_test, y_test = procesar_dataset(DATA_DIR / "Testing")
    guardar_dataset(X_test, y_test, PROCESSED_DIR / "dataset_test.pkl")

    print("✅ Dataset procesado correctamente")