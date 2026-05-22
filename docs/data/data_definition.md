# Definición de los datos

## Origen de los datos

- [ ] En el presente proyecto vamos a fundamentarnos en el siguiente dataset de Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Especificación de los scripts para la carga de datos

- [ ] Para este tipo de proyecto se pretente utilizar un HOG, el cual nos va a permitir realizar un proceso de extracción de caracteristicas, el script para cargar las imagenes, realizar el HOG (partiendo de que las imagenes ya estan en escala de grises) y mantener los datos en arreglos de numpy divididos entre entrenamiento y prueba.

        import cv2
        import numpy as np
        import pickle
        import sys
        import os
        from pathlib import Path
        from skimage.feature import hog

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        ROOT_DIR = Path(__file__).resolve().parents[2]
        DATA_DIR = ROOT_DIR / 'docs' / 'data' / 'datos'
        PROCESSED_DIR = ROOT_DIR / 'docs' / 'data' / 'processed'

        img_size = (128, 128)

        def procesar_dataset(tipo_dataset):
            X, y = [], []
            ruta_dir = DATA_DIR / tipo_dataset

            if not ruta_dir.exists():
                print(f"Error: La carpeta {ruta_dir} no fue encontrada.", flush=True)
                return np.array([]), np.array([])

            print(f"\n--- Procesando conjunto: {tipo_dataset} ---", flush=True)

            for clase_folder in ruta_dir.iterdir():
                if clase_folder.is_dir() and not clase_folder.name.startswith('.'):
                    count = 0
                    print(f"  > Categoría: {clase_folder.name}", flush=True)

                    for img_path in clase_folder.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                            try:
                                # Carga y redimensionamiento
                                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    continue

                                img = cv2.resize(img, img_size)

                                # Extracción HOG
                                features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), block_norm='L2-Hys')

                                X.append(features)
                                y.append(clase_folder.name)

                                count += 1
                                if count % 10 == 0:
                                    print(f"    Procesadas: {count} imágenes...", end='\r', flush=True)

                            except Exception as e:
                                print(f"\nError al procesar {img_path.name}: {e}", flush=True)
                                continue
                    print(f"\n    Finalizada: {count} imágenes.", flush=True)

            return np.array(X), np.array(y)

        def guardar_dataset(X, y, nombre_archivo):
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            ruta_archivo = PROCESSED_DIR / nombre_archivo
            with open(ruta_archivo, 'wb') as f:
                pickle.dump((X, y), f)
            print(f"Dataset guardado en: {ruta_archivo}", flush=True)

        if __name__ == "__main__":
            if not DATA_DIR.exists():
                print(f"Error crítico: No se encuentra la carpeta de datos en {DATA_DIR}", flush=True)
                sys.exit(1)

            # Procesar y guardar Entrenamiento
            X_train, y_train = procesar_dataset('Training')
            if X_train.size > 0:
                guardar_dataset(X_train, y_train, 'dataset_train.pkl')

            # Procesar y guardar Pruebas
            X_test, y_test = procesar_dataset('Testing')
            if X_test.size > 0:
                guardar_dataset(X_test, y_test, 'dataset_test.pkl')

  Para ello hay que instalar lo siguiente: - pip install opencv-python numpy scikit-image

## Referencias a rutas o bases de datos origen y destino

- [ ] Origen de los datos: los datos estarán en la ruta raíz TDSP_TEMPLATE/docs/data/datos/ , carpeta que no quedará en Github. En donde:

        Training/: Contiene las imágenes utilizadas para el entrenamiento del modelo.

        Testing/: Contiene las imágenes utilizadas para la validación y prueba del modelo.

        Subdirectorios (Clases): glioma, meningioma, notumor, pituitary.

        Formato de archivos: Imágenes (ej. .jpg).

  Destino de los datos procesados: Tras el procesamiento de extracción de características (HOG), los datos se transforman de imágenes a vectores numéricos guardados en la carpeta data/processed, divididos entre datos de entrenamiento y prueba.

### Rutas de origen de datos

- [ ] Los datos fuente se encuentran alojados localmente dentro del repositorio del proyecto. La ruta de acceso raíz es: TDSP_TEMPLATE/docs/data/datos/. Esta ubicación contiene dos directorios principales que separan los datos para fines de entrenamiento y evaluación.
- [ ] los archivos estan alojados en carpetas diespuestas de la siguiente manera:

        ├── Testing/
        │   ├── glioma/
        │   ├── meningioma/
        │   ├── notumor/
        │   └── pituitary/
        └── Training/
            ├── glioma/
            ├── meningioma/
            ├── notumor/
            └── pituitary/

- [ ] En este caso, se realizará es un HOG, los datos ya se encuentran balanceados, en escala de grises y las imagenes son del mismo tamaño.

### Base de datos de destino

- [ ] En lugar de una base de datos relacional tradicional, se ha definido un almacén de datos estructurado en formato binario (archivos serializados). Este formato permite una persistencia eficiente de los descriptores HOG, facilitando su recuperación inmediata para las etapas de entrenamiento y validación del modelo sin necesidad de re-procesamiento. Los datos se almacenan en la ruta: TDSP_TEMPLATE/docs/data/processed/.
- [ ] La estructura de destino se organiza en dos conjuntos principales, almacenados como objetos de Python serializados (formato .pkl):

        dataset_train.pkl: Contiene la matriz de características $X_{train}$ (vectores HOG de las imágenes de entrenamiento) y el vector de etiquetas $y_{train}$.

        dataset_test.pkl: Contiene la matriz de características $X_{test}$ (vectores HOG de las imágenes de prueba) y el vector de etiquetas $y_{test}$.

        La estructura interna de cada archivo es un par (X, y) donde:X (Matriz NumPy): De dimensiones $(N \times M)$, siendo $N$ el número de imágenes y $M$ la longitud fija del vector HOG.y (Array NumPy): Un vector de longitud $N$ que contiene los nombres de las clases (etiquetas categóricas) correspondientes a cada vector de características.

- [ ] Las imágenes crudas se extraen, redimensionan y procesan mediante el descriptor HOG.
