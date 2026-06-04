from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog

IMG_SIZE = (128, 128)


def procesar_dataset(ruta_dir: Path):
    X, y = [], []

    for clase_folder in ruta_dir.iterdir():
        if clase_folder.is_dir() and not clase_folder.name.startswith('.'):
            for img_path in clase_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    img = cv2.resize(img, IMG_SIZE)

                    features = hog(
                        img,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys'
                    )

                    X.append(features)
                    y.append(clase_folder.name)

    return np.array(X), np.array(y)