import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


# =========================================================
# DISTRIBUCIÓN DE CLASES
# =========================================================

def plot_distribucion_clases(df):
    plt.figure(figsize=(10, 6))
    df["clase"].value_counts().plot(kind="bar")

    plt.title("Distribución de clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# =========================================================
# DISTRIBUCIÓN TRAIN / TEST
# =========================================================

def plot_train_test(df):
    pivot = pd.crosstab(df["clase"], df["split"])

    pivot.plot(kind="bar", figsize=(10, 6))

    plt.title("Distribución Train/Test")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# =========================================================
# MUESTRAS VISUALES
# =========================================================

def plot_muestras(df):
    clases = sorted(df["clase"].unique())

    fig, axes = plt.subplots(
        len(clases),
        3,
        figsize=(12, 4 * len(clases))
    )

    if len(clases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, clase in enumerate(clases):

        subset = df[df["clase"] == clase]
        muestras = subset.sample(3)

        for j, (_, row) in enumerate(muestras.iterrows()):

            img = cv2.imread(row["path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[i, j].imshow(img)
            axes[i, j].axis("off")

            if j == 0:
                axes[i, j].set_title(clase)

    plt.tight_layout()
    plt.show()


# =========================================================
# HISTOGRAMA DE INTENSIDAD
# =========================================================

def plot_intensidad(intensidades):
    plt.figure(figsize=(10, 5))

    plt.hist(intensidades, bins=30)

    plt.title("Distribución intensidad")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()


# =========================================================
# PCA HOG
# =========================================================

def plot_pca(hog_pca, labels):
    plt.figure(figsize=(10, 8))

    labels = np.array(labels)

    for clase in np.unique(labels):
        idx = labels == clase

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
# VISUALIZACIÓN HOG
# =========================================================

def plot_hog(img, hog_image):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG")
    plt.axis("off")

    plt.tight_layout()
    plt.show()