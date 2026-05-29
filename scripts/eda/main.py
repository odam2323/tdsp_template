import sys
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIGURACIÓN DE PATH (IMPORTANTE)
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.brain_tumor_classifier.visualization.plots import (
    plot_distribucion_clases,
    plot_train_test,
    plot_muestras,
    plot_intensidad,
    plot_pca,
    plot_hog
)

DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

IMG_SIZE = (128, 128)

VALID_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif"
]

# =========================================================
# CARGA METADATA
# =========================================================

def cargar_metadata():

    registros = []

    for split, split_dir in zip(
        ["Training", "Testing"],
        [TRAIN_DIR, TEST_DIR]
    ):

        if not split_dir.exists():
            print(f"\n❌ No existe: {split_dir}")
            continue

        print(f"\n📂 Leyendo: {split_dir}")

        for clase_dir in split_dir.iterdir():

            if not clase_dir.is_dir():
                continue

            clase = clase_dir.name
            contador = 0

            for img_path in clase_dir.rglob("*"):

                if img_path.suffix.lower() in VALID_EXTENSIONS:

                    registros.append({
                        "split": split,
                        "clase": clase,
                        "path": str(img_path)
                    })

                    contador += 1

            print(f"  {clase}: {contador} imágenes")

    return pd.DataFrame(registros)


# =========================================================
# HASH IMAGEN
# =========================================================

def calcular_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


# =========================================================
# EXTRACCIÓN HOG
# =========================================================

def extraer_hog(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return features


# =========================================================
# MAIN
# =========================================================

def main():

    df = cargar_metadata()

    if df.empty:
        print("\n❌ Dataset vacío.")
        return

    # =====================================================
    # INFORMACIÓN GENERAL
    # =====================================================

    print("\n==============================")
    print("INFORMACIÓN GENERAL")
    print("==============================\n")

    print(df.head())

    print("\nTotal imágenes:", len(df))

    print("\nDistribución por split:")
    print(df["split"].value_counts())

    print("\nDistribución por clase:")
    print(df["clase"].value_counts())

    # =====================================================
    # VISUALIZACIONES
    # =====================================================

    plot_distribucion_clases(df)
    plot_train_test(df)
    plot_muestras(df)

    # =====================================================
    # DETECCIÓN DE CORRUPTAS
    # =====================================================

    print("\n==============================")
    print("IMÁGENES CORRUPTAS")
    print("==============================\n")

    corruptas = []

    for path in df["path"]:
        img = cv2.imread(path)
        if img is None:
            corruptas.append(path)

    print(f"Cantidad corruptas: {len(corruptas)}")

    # =====================================================
    # DUPLICADOS
    # =====================================================

    print("\n==============================")
    print("DUPLICADOS")
    print("==============================\n")

    df["hash"] = df["path"].apply(calcular_hash)

    duplicados = df[df.duplicated("hash", keep=False)]

    print(f"Imágenes duplicadas: {len(duplicados)}")

    if not duplicados.empty:
        print(duplicados[["split", "clase", "path"]].head(10))

    # =====================================================
    # RESOLUCIONES
    # =====================================================

    print("\n==============================")
    print("RESOLUCIONES")
    print("==============================\n")

    alturas, anchos = [], []

    for path in df["path"]:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            alturas.append(h)
            anchos.append(w)

    print(f"Altura promedio: {np.mean(alturas):.2f}")
    print(f"Ancho promedio: {np.mean(anchos):.2f}")

    # =====================================================
    # INTENSIDAD
    # =====================================================

    print("\n==============================")
    print("INTENSIDAD")
    print("==============================\n")

    intensidades = []

    for path in df["path"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            intensidades.append(np.mean(img))

    plot_intensidad(intensidades)

    print(f"Intensidad promedio: {np.mean(intensidades):.2f}")

    # =====================================================
    # HOG + PCA
    # =====================================================

    print("\n==============================")
    print("HOG + PCA")
    print("==============================\n")

    hog_features = []
    hog_labels = []

    for _, row in df.sample(min(400, len(df)), random_state=42).iterrows():

        features = extraer_hog(row["path"])

        if features is not None:
            hog_features.append(features)
            hog_labels.append(row["clase"])

    hog_features = np.array(hog_features)

    print(f"HOG shape: {hog_features.shape}")

    scaler = StandardScaler()
    hog_scaled = scaler.fit_transform(hog_features)

    pca = PCA(n_components=2)
    hog_pca = pca.fit_transform(hog_scaled)

    plot_pca(hog_pca, hog_labels)

    # =====================================================
    # VISUALIZACIÓN HOG
    # =====================================================

    print("\n==============================")
    print("VISUALIZACIÓN HOG")
    print("==============================\n")

    ejemplo = df.iloc[0]["path"]

    img = cv2.imread(ejemplo, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)

    features, hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )

    plot_hog(img, hog_image)

    # =====================================================
    # RESUMEN
    # =====================================================

    print("\n==============================")
    print("RESUMEN FINAL")
    print("==============================\n")

    print(f"Total imágenes: {len(df)}")
    print(df["clase"].value_counts())

    print("\n✅ Dataset validado correctamente.")


# =========================================================

if __name__ == "__main__":
    main()