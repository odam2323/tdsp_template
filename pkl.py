import pandas as pd

# Cargar el archivo
data = pd.read_pickle('/Users/andres/Desktop/tdsp_template/docs/data/processed/dataset_train.pkl')

# Imprimir información detallada
print(f"La variable 'data' es de tipo: {type(data)}")
print(f"Contiene {len(data)} elementos.")

# Mostrar qué hay dentro de cada elemento
for i, item in enumerate(data):
    print(f"\n--- Elemento {i} ---")
    print(f"Tipo: {type(item)}")
    
    # Si es un DataFrame, mostramos las primeras filas
    if isinstance(item, pd.DataFrame):
        print("¡Es un DataFrame! Aquí están sus primeras filas:")
        print(item.head())
    else:
        # Si no es DataFrame, imprimimos los primeros caracteres/elementos para identificarlo
        print(f"Contenido (resumido): {str(item)[:200]}")