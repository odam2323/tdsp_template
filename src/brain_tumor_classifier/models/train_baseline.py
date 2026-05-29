import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(">>> SCRIPT BASELINE SVM INICIADO <<<")


# =========================================================
# CONFIGURACIÓN DE RUTAS ✅ (PORTABLE)
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "docs" / "data" / "processed"

TRAIN_PATH = DATA_DIR / "dataset_train.pkl"
TEST_PATH = DATA_DIR / "dataset_test.pkl"

MODEL_PATH = DATA_DIR / "modelo_svm.pkl"
REPORT_PATH = DATA_DIR / "reporte_metricas.txt"


# =========================================================
# CARGA DE DATOS
# =========================================================

def load_pkl_data(filepath):

    print(f"\nIntentando cargar: {filepath}")

    if not filepath.exists():
        print(f"❌ Error: archivo no encontrado")
        return None, None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Caso: dict
    if isinstance(data, dict):
        print(f"Llaves encontradas: {list(data.keys())}")

        X = np.array(data.get('features', data.get('hog', None)))
        y = np.array(data.get('labels', data.get('targets', None)))

    # Caso: tupla
    else:
        print("Formato detectado: (X, y)")
        X, y = data[0], data[1]

    X = np.array(X)
    y = np.array(y)

    # Aplanar si es necesario
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    return X, y


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def train_and_evaluate():

    # ============================
    # 1. CARGA
    # ============================
    print("\n=== 1. CARGANDO DATASETS ===")

    X_train, y_train = load_pkl_data(TRAIN_PATH)
    X_test, y_test = load_pkl_data(TEST_PATH)

    if X_train is None or X_test is None:
        print("\n❌ Error: No se pudieron cargar los datos")
        return

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    print(f"\nClases detectadas: {np.unique(y_train)}")

    # ============================
    # 2. ENTRENAMIENTO
    # ============================
    print("\n=== 2. ENTRENANDO SVM ===")

    clf = SVC(
        kernel='rbf',
        C=1.0,
        class_weight='balanced',
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("✅ Modelo entrenado correctamente")

    # ============================
    # 3. PREDICCIÓN
    # ============================
    print("\n=== 3. PREDICIENDO ===")

    y_pred = clf.predict(X_test)

    # ============================
    # 4. MÉTRICAS
    # ============================
    print("\n=== 4. MÉTRICAS ===")

    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # ============================
    # 5. GUARDAR MODELO
    # ============================
    print("\n=== 5. GUARDANDO MODELO ===")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"✅ Modelo guardado en: {MODEL_PATH}")

    # ============================
    # 6. GUARDAR REPORTE
    # ============================
    print("\n=== 6. GUARDANDO REPORTE ===")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        f.write("=== REPORTE MODELO SVM (HOG) ===\n\n")

        f.write(f"Accuracy: {acc:.4f}\n\n")

        f.write("=== MATRIZ DE CONFUSIÓN ===\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")

        f.write("=== REPORTE DE CLASIFICACIÓN ===\n")
        f.write(classification_report(y_test, y_pred))

    print(f"✅ Reporte guardado en: {REPORT_PATH}")


# =========================================================
# EJECUCIÓN
# =========================================================

if __name__ == "__main__":
    try:
        print("ROOT_DIR:", ROOT_DIR)
        print("DATA_DIR:", DATA_DIR)

        train_and_evaluate()

        print("\n✅ PIPELINE COMPLETADO")

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {str(e)}")
