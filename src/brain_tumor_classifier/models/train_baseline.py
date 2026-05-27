import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

print(">>> SCRIPT INICIADO CORRECTAMENTE <<<")

def load_pkl_data(filepath):
    print(f"Intentando cargar: {filepath}")
    if not os.path.exists(filepath):
        print(f"[ERROR] El archivo NO existe en la ruta especificada.")
        return None, None
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Imprimir qué llaves tiene tu archivo .pkl para saber cómo extraer los datos
    if isinstance(data, dict):
        print(f"Llaves encontradas en el archivo pkl: {list(data.keys())}")
        # Intentamos obtener las características. Cambia 'features' si tus llaves se llaman distinto
        X = np.array(data.get('features', data.get('hog', None)))
        y = np.array(data.get('labels', data.get('targets', None)))
    else:
        # Si el pkl es una tupla directamente (X, y)
        print("El archivo pkl no es un diccionario, parece ser una tupla/lista directa.")
        X, y = data[0], data[1]
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
        
    return X, y

def train_and_evaluate():
    train_path = "/Users/andres/Desktop/tdsp_template/docs/data/processed/dataset_train.pkl"
    test_path = "/Users/andres/Desktop/tdsp_template/docs/data/processed/dataset_test.pkl"
    
    print("\n=== 1. CARGANDO DATASETS ===")
    X_train, y_train = load_pkl_data(train_path)
    X_test, y_test = load_pkl_data(test_path)
    
    if X_train is None or X_test is None:
        print("[ERROR] Deteniendo ejecución porque no se pudieron cargar los datos.")
        return

    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    print("\n=== 2. ENTRENANDO SVM (BASELINE) ===")
    clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    print("¡Entrenamiento completado!")
    
    print("\n=== 3. EVALUACIÓN ===")
    y_pred = clf.predict(X_test)
    
    print("\n[MATRIZ DE CONFUSIÓN]")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n[REPORTE DE CLASIFICACIÓN]")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Ocurrió un fallo durante la ejecución: {str(e)}")