import cv2
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from skimage.feature import hog

from kaggle_prepare import preparar_dataset_kaggle

# --- SOLUCIÓN PARA BLOQUEO EN MACOS ---
# Limita el uso de hilos para evitar conflictos con librerías del sistema
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Definir rutas relativas al archivo actual
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
    preparar_dataset_kaggle()
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