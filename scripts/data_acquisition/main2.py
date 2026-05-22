import cv2
import numpy as np
import pickle
import sys
import os
import shutil
from pathlib import Path
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# ==========================================
# EVITAR BLOQUEOS
# ==========================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==========================================
# RUTAS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / 'docs' / 'data' / 'datos'
PROCESSED_DIR = ROOT_DIR / 'docs' / 'data' / 'processed'

# ==========================================
# DATASET KAGGLE
# ==========================================

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

img_size = (128, 128)

# ==========================================
# CONFIG KAGGLE
# ==========================================

def configurar_kaggle():

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    origen = ROOT_DIR / "kaggle.json"
    destino = kaggle_dir / "kaggle.json"

    shutil.copy(origen, destino)

    try:
        os.chmod(destino, 0o600)
    except:
        pass

# ==========================================
# DESCARGAR DATASET
# ==========================================

def descargar_dataset():

    if (DATA_DIR / "Training").exists():
        print("\nDataset ya descargado.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("\nDescargando dataset desde Kaggle...")

    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DATA_DIR),
        unzip=True
    )

    print("\nDataset descargado correctamente.")

# ==========================================
# EXTRAER FEATURES
# ==========================================

def cargar_imagenes(ruta_dataset):

    X = []
    y = []

    for clase_folder in ruta_dataset.iterdir():

        if clase_folder.is_dir():

            print(f"\nProcesando clase: {clase_folder.name}")

            count = 0

            for img_path in clase_folder.glob('*'):

                if img_path.suffix.lower() in [
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.bmp'
                ]:

                    try:

                        img = cv2.imread(
                            str(img_path),
                            cv2.IMREAD_GRAYSCALE
                        )

                        if img is None:
                            continue

                        img = cv2.resize(img, img_size)

                        features = hog(
                            img,
                            orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            block_norm='L2-Hys'
                        )

                        X.append(features)
                        y.append(clase_folder.name)

                        count += 1

                        if count % 50 == 0:
                            print(f"Procesadas: {count}")

                    except Exception as e:
                        print(f"Error en {img_path.name}: {e}")

            print(f"Total: {count}")

    return np.array(X), np.array(y)

# ==========================================
# GUARDAR PKL
# ==========================================

def guardar_dataset(X, y, nombre):

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ruta = PROCESSED_DIR / nombre

    with open(ruta, 'wb') as f:
        pickle.dump((X, y), f)

    print(f"\nGuardado: {ruta}")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\n===== PIPELINE MRI =====")

    configurar_kaggle()

    descargar_dataset()

    # ==============================
    # TRAINING
    # ==============================

    train_dir = DATA_DIR / "Training"

    X_train_full, y_train_full = cargar_imagenes(train_dir)

    # ==============================
    # SPLIT TRAIN / VALIDATION
    # ==============================

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    guardar_dataset(
        X_train,
        y_train,
        'dataset_train.pkl'
    )

    guardar_dataset(
        X_val,
        y_val,
        'dataset_validation.pkl'
    )

    # ==============================
    # TEST
    # ==============================

    test_dir = DATA_DIR / "Testing"

    X_test, y_test = cargar_imagenes(test_dir)

    guardar_dataset(
        X_test,
        y_test,
        'dataset_test.pkl'
    )

    print("\n===== PROCESO FINALIZADO =====")