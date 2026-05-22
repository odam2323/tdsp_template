import os
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURACIÓN
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

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

            print(f"\nNo existe: {split_dir}")
            continue

        print(f"\nLeyendo: {split_dir}")

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

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return None

    img = cv2.resize(
        img,
        IMG_SIZE
    )

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return features


# =========================================================
# CARGAR DATASET
# =========================================================

df = cargar_metadata()

if df.empty:

    print("\nDataset vacío.")
    exit()


# =========================================================
# INFORMACIÓN GENERAL
# =========================================================

print("\n==============================")
print("INFORMACIÓN GENERAL")
print("==============================\n")

print(df.head())

print("\nTotal imágenes:")
print(len(df))

print("\nDistribución por split:")
print(df["split"].value_counts())

print("\nDistribución por clase:")
print(df["clase"].value_counts())


# =========================================================
# DISTRIBUCIÓN CLASES
# =========================================================

plt.figure(figsize=(10, 6))

df["clase"].value_counts().plot(
    kind="bar"
)

plt.title("Distribución de clases")
plt.xlabel("Clase")
plt.ylabel("Cantidad")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# TRAIN / TEST
# =========================================================

pivot = pd.crosstab(
    df["clase"],
    df["split"]
)

pivot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Distribución Train/Test")
plt.xlabel("Clase")
plt.ylabel("Cantidad")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# DETECCIÓN DE CORRUPTAS
# =========================================================

print("\n==============================")
print("IMÁGENES CORRUPTAS")
print("==============================\n")

corruptas = []

for path in df["path"]:

    img = cv2.imread(path)

    if img is None:

        corruptas.append(path)

print(f"Cantidad corruptas: {len(corruptas)}")


# =========================================================
# DETECCIÓN DUPLICADOS
# =========================================================

print("\n==============================")
print("DUPLICADOS")
print("==============================\n")

df["hash"] = df["path"].apply(
    calcular_hash
)

duplicados = df[
    df.duplicated("hash", keep=False)
]

print(f"Imágenes duplicadas: {len(duplicados)}")

if not duplicados.empty:

    print("\nPrimeros duplicados:")

    print(
        duplicados[
            ["split", "clase", "path"]
        ].head(10)
    )


# =========================================================
# RESOLUCIONES
# =========================================================

print("\n==============================")
print("RESOLUCIONES")
print("==============================\n")

anchos = []
alturas = []

for path in df["path"]:

    img = cv2.imread(path)

    if img is not None:

        h, w = img.shape[:2]

        alturas.append(h)
        anchos.append(w)

print(f"Altura promedio: {np.mean(alturas):.2f}")
print(f"Ancho promedio: {np.mean(anchos):.2f}")

print(f"Altura mínima: {np.min(alturas)}")
print(f"Altura máxima: {np.max(alturas)}")

print(f"Ancho mínimo: {np.min(anchos)}")
print(f"Ancho máximo: {np.max(anchos)}")


# =========================================================
# MUESTRAS VISUALES
# =========================================================

print("\n==============================")
print("MUESTRAS")
print("==============================\n")

clases = sorted(
    df["clase"].unique()
)

fig, axes = plt.subplots(
    len(clases),
    3,
    figsize=(12, 4 * len(clases))
)

if len(clases) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, clase in enumerate(clases):

    subset = df[
        df["clase"] == clase
    ]

    muestras = subset.sample(3)

    for j, (_, row) in enumerate(
        muestras.iterrows()
    ):

        img = cv2.imread(row["path"])

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        axes[i, j].imshow(img)

        axes[i, j].axis("off")

        if j == 0:

            axes[i, j].set_title(clase)

plt.tight_layout()
plt.show()


# =========================================================
# INTENSIDAD PROMEDIO
# =========================================================

print("\n==============================")
print("INTENSIDAD")
print("==============================\n")

intensidades = []

for path in df["path"]:

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is not None:

        intensidades.append(
            np.mean(img)
        )

plt.figure(figsize=(10, 5))

plt.hist(
    intensidades,
    bins=30
)

plt.title("Distribución intensidad")
plt.xlabel("Intensidad")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

print(
    f"Intensidad promedio: "
    f"{np.mean(intensidades):.2f}"
)


# =========================================================
# ANÁLISIS HOG
# =========================================================

print("\n==============================")
print("HOG")
print("==============================\n")

hog_features = []
hog_labels = []

for _, row in df.sample(
    min(400, len(df)),
    random_state=42
).iterrows():

    features = extraer_hog(
        row["path"]
    )

    if features is not None:

        hog_features.append(features)
        hog_labels.append(row["clase"])

hog_features = np.array(hog_features)

print(
    f"Shape HOG: {hog_features.shape}"
)

print(
    f"Cantidad features: "
    f"{hog_features.shape[1]}"
)


# =========================================================
# PCA HOG
# =========================================================

print("\n==============================")
print("PCA HOG")
print("==============================\n")

scaler = StandardScaler()

hog_scaled = scaler.fit_transform(
    hog_features
)

pca = PCA(n_components=2)

hog_pca = pca.fit_transform(
    hog_scaled
)

plt.figure(figsize=(10, 8))

for clase in np.unique(hog_labels):

    idx = np.array(hog_labels) == clase

    plt.scatter(
        hog_pca[idx, 0],
        hog_pca[idx, 1],
        label=clase,
        alpha=0.7
    )

plt.title("PCA Features HOG")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.legend()

plt.tight_layout()
plt.show()


# =========================================================
# HOG VISUAL
# =========================================================

print("\n==============================")
print("VISUALIZACIÓN HOG")
print("==============================\n")

ejemplo = df.iloc[0]["path"]

img = cv2.imread(
    ejemplo,
    cv2.IMREAD_GRAYSCALE
)

img = cv2.resize(
    img,
    IMG_SIZE
)

features, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    img,
    cmap="gray"
)

plt.title("MRI Original")

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(
    hog_image,
    cmap="gray"
)

plt.title("HOG")

plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# RESUMEN FINAL
# =========================================================

print("\n==============================")
print("RESUMEN")
print("==============================\n")

print(f"Total imágenes: {len(df)}")

print("\nDistribución final:")

print(df["clase"].value_counts())

print("\nDataset validado correctamente.")