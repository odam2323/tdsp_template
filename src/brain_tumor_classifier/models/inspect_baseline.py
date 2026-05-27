import pickle
import numpy as np
from sklearn.svm import SVC

print(">>> SCRIPT DE INSPECCIÓN INICIADO <<<")

def inspect_predictions():
    train_path = "/Users/andres/Desktop/tdsp_template/docs/data/processed/dataset_train.pkl"
    test_path = "/Users/andres/Desktop/tdsp_template/docs/data/processed/dataset_test.pkl"
    output_txt = "/Users/andres/Desktop/tdsp_template/docs/data/processed/reporte_predicciones.txt"
    
    print("1. Cargando archivos .pkl...")
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(test_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
        
    print(f"Dimensiones cargadas - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Asegurar formato matricial plano para SVM
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if len(X_test.shape) > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    print("2. Re-entrenando modelo SVC (espera un momento)...")
    clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    print("3. Generando predicciones...")
    y_pred = clf.predict(X_test)
    
    print(f"4. Escribiendo reporte en: {output_txt}")
    with open(output_txt, 'w') as f:
        f.write("=== REPORTE DETALLADO DE PREDICCIONES BASELINE (HOG + SVM) ===\n")
        f.write(f"Total imagenes evaluadas: {len(y_test)}\n\n")
        f.write(f"{'Indice Imagen':<15}{'Clase Real':<20}{'Prediccion Modelo':<20}{'Resultado':<10}\n")
        f.write("-" * 70 + "\n")
        
        errores_count = 0
        for i in range(len(y_test)):
            # Como ya son strings, los limpiamos y pasamos directamente a mayúscula inicial
            real_name = str(y_test[i]).strip().capitalize()
            pred_name = str(y_pred[i]).strip().capitalize()
            
            # Corrección rápida para mantener consistencia con el formato visual de la clase "No Tumor"
            if real_name.lower() == 'notumor': real_name = 'No Tumor'
            if pred_name.lower() == 'notumor': pred_name = 'No Tumor'
            
            status = "OK" if real_name == pred_name else "ERROR"
            
            if status == "ERROR":
                errores_count += 1
                
            f.write(f"{i:<15}{real_name:<20}{pred_name:<20}{status:<10}\n")
            
        f.write("\n" + "="*40 + "\n")
        f.write(f"Total de aciertos: {len(y_test) - errores_count}\n")
        f.write(f"Total de errores: {errores_count}\n")

    print(f"¡Proceso finalizado con éxito! Registrados {errores_count} errores.")

if __name__ == "__main__":
    try:
        inspect_predictions()
    except Exception as e:
        print(f"\n[ERROR EN EJECUCIÓN] Falló debido a: {str(e)}")