import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

print(">>> SCRIPT DE INSPECCIÓN INICIADO <<<")

# =========================================================
# CONFIGURACIÓN DE RUTAS (PORTABLE ✅)
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "docs" / "data" / "processed"

TRAIN_PATH = DATA_DIR / "dataset_train.pkl"
TEST_PATH = DATA_DIR / "dataset_test.pkl"

OUTPUT_PATH = DATA_DIR / "reporte_predicciones.txt"


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def inspect_predictions():

    print("\n1. Cargando archivos .pkl...")

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print("\n❌ Error: No se encontraron los datasets procesados.")
        return

    with open(TRAIN_PATH, 'rb') as f:
        X_train, y_train = pickle.load(f)

    with open(TEST_PATH, 'rb') as f:
        X_test, y_test = pickle.load(f)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # =====================================================
    # PREPROCESAMIENTO
    # =====================================================

    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)

    if len(X_test.shape) > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # =====================================================
    # ENTRENAMIENTO
    # =====================================================

    print("\n2. Entrenando modelo SVM...")

    clf = SVC(
        kernel='rbf',
        C=1.0,
        class_weight='balanced',
        random_state=42
    )

    clf.fit(X_train, y_train)

    # =====================================================
    # PREDICCIONES
    # =====================================================

    print("\n3. Generando predicciones...")

    y_pred = clf.predict(X_test)

    # =====================================================
    # REPORTE TEXTO
    # =====================================================

    print(f"\n4. Generando reporte en:\n{OUTPUT_PATH}")

    errores_count = 0

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:

        f.write("=== REPORTE DETALLADO DE PREDICCIONES (HOG + SVM) ===\n\n")
        f.write(f"Total imágenes evaluadas: {len(y_test)}\n\n")

        f.write(
            f"{'Indice':<10}{'Real':<20}{'Predicción':<20}{'Resultado':<10}\n"
        )
        f.write("-" * 60 + "\n")

        for i in range(len(y_test)):

            real_name = str(y_test[i]).strip().capitalize()
            pred_name = str(y_pred[i]).strip().capitalize()

            # Normalización opcional
            if real_name.lower() == 'notumor':
                real_name = 'No Tumor'

            if pred_name.lower() == 'notumor':
                pred_name = 'No Tumor'

            status = "OK" if real_name == pred_name else "ERROR"

            if status == "ERROR":
                errores_count += 1

            f.write(
                f"{i:<10}{real_name:<20}{pred_name:<20}{status:<10}\n"
            )

        f.write("\n" + "=" * 40 + "\n")
        f.write(f"Aciertos: {len(y_test) - errores_count}\n")
        f.write(f"Errores: {errores_count}\n")
        f.write(
            f"Accuracy: {(len(y_test) - errores_count) / len(y_test):.4f}\n"
        )

    print("\n✅ Proceso finalizado correctamente")
    print(f"Errores: {errores_count}")
    print(f"Accuracy: {(len(y_test) - errores_count) / len(y_test):.4f}")


# =========================================================

if __name__ == "__main__":
    try:
        inspect_predictions()
    except Exception as e:
        print(f"\n❌ Error en ejecución: {str(e)}")