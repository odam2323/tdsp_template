import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog


# =========================================================
# CONFIGURACIÓN
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "docs" / "data" / "datos"

TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

IMG_SIZE = (128, 128)

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def obtener_imagenes():

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

            imagenes_clase = 0

            for img_path in clase_dir.rglob("*"):

                if img_path.suffix.lower() in VALID_EXTENSIONS:

                    registros.append({
                        "split": split,
                        "clase": clase,
                        "path": str(img_path)
                    })

                    imagenes_clase += 1

            print(f"  {clase}: {imagenes_clase} imágenes")

    return pd.DataFrame(registros)


# =========================================================
# CARGA METADATA
# =========================================================

df = obtener_imagenes()

if df.empty:

    print("\nNo se encontraron imágenes.")
    exit()


# =========================================================
# INFORMACIÓN GENERAL
# =========================================================

print("\n==============================")
print("INFORMACIÓN GENERAL")
print("==============================\n")

print(df.head())

print("\nCantidad total de imágenes:")
print(len(df))

print("\nDistribución por split:")
print(df["split"].value_counts())

print("\nDistribución por clase:")
print(df["clase"].value_counts())


# =========================================================
# DISTRIBUCIÓN DE CLASES
# =========================================================

plt.figure(figsize=(10, 6))

df["clase"].value_counts().plot(
    kind="bar"
)

plt.title("Distribución de clases")
plt.xlabel("Clase")
plt.ylabel("Cantidad imágenes")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# DISTRIBUCIÓN TRAIN / TEST
# =========================================================

pivot = pd.crosstab(
    df["clase"],
    df["split"]
)

pivot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Distribución Train/Test por clase")
plt.xlabel("Clase")
plt.ylabel("Cantidad imágenes")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# VERIFICACIÓN DE IMÁGENES CORRUPTAS
# =========================================================

print("\n==============================")
print("VERIFICACIÓN DE IMÁGENES")
print("==============================\n")

imagenes_corruptas = []

for path in df["path"]:

    img = cv2.imread(path)

    if img is None:

        imagenes_corruptas.append(path)

print(f"Imágenes corruptas: {len(imagenes_corruptas)}")

if len(imagenes_corruptas) > 0:

    print("\nPrimeras imágenes corruptas:")

    for img in imagenes_corruptas[:10]:

        print(img)


# =========================================================
# RESOLUCIONES DE IMÁGENES
# =========================================================

print("\n==============================")
print("RESOLUCIONES")
print("==============================\n")

alturas = []
anchos = []

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
# HISTOGRAMA DE RESOLUCIONES
# =========================================================

plt.figure(figsize=(12, 5))

plt.hist(
    alturas,
    bins=30
)

plt.title("Distribución de alturas")
plt.xlabel("Altura")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

plt.hist(
    anchos,
    bins=30
)

plt.title("Distribución de anchos")
plt.xlabel("Ancho")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# =========================================================
# VISUALIZACIÓN DE MUESTRAS
# =========================================================

print("\n==============================")
print("MUESTRAS POR CLASE")
print("==============================\n")

clases = sorted(df["clase"].unique())

fig, axes = plt.subplots(
    len(clases),
    5,
    figsize=(15, 3 * len(clases))
)

if len(clases) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, clase in enumerate(clases):

    subset = df[df["clase"] == clase]

    n_samples = min(5, len(subset))

    muestras = subset.sample(n_samples)

    for j, (_, row) in enumerate(muestras.iterrows()):

        img = cv2.imread(row["path"])

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        axes[i, j].imshow(img)

        axes[i, j].axis("off")

        if j == 0:

            axes[i, j].set_title(clase)

    for j in range(n_samples, 5):

        axes[i, j].axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# ANÁLISIS DE INTENSIDAD
# =========================================================

print("\n==============================")
print("ANÁLISIS DE INTENSIDAD")
print("==============================\n")

mean_intensity = []

for path in df["path"]:

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is not None:

        mean_intensity.append(np.mean(img))

plt.figure(figsize=(10, 5))

plt.hist(
    mean_intensity,
    bins=30
)

plt.title("Distribución de intensidad promedio")
plt.xlabel("Intensidad")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()

print(f"Intensidad promedio global: {np.mean(mean_intensity):.2f}")


# =========================================================
# ANÁLISIS HOG
# =========================================================

print("\n==============================")
print("ANÁLISIS HOG")
print("==============================\n")

ejemplo_path = df.iloc[0]["path"]

img = cv2.imread(
    ejemplo_path,
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

print(f"Cantidad de features HOG: {len(features)}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    img,
    cmap="gray"
)

plt.title("Imagen original")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    hog_image,
    cmap="gray"
)

plt.title("Visualización HOG")

plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================================
# HISTOGRAMA DE PIXELES
# =========================================================

print("\n==============================")
print("HISTOGRAMA DE PIXELES")
print("==============================\n")

img = cv2.imread(
    ejemplo_path,
    cv2.IMREAD_GRAYSCALE
)

plt.figure(figsize=(10, 5))

plt.hist(
    img.ravel(),
    bins=256
)

plt.title("Histograma de intensidades")
plt.xlabel("Valor pixel")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()


# =========================================================
# ESTADÍSTICAS FINALES
# =========================================================

print("\n==============================")
print("RESUMEN FINAL")
print("==============================\n")

print(f"Total imágenes: {len(df)}")

print("\nCantidad por split:")

print(df["split"].value_counts())

print("\nDistribución final:")

print(df["clase"].value_counts())

print("\nDataset listo para preprocessing/training.")