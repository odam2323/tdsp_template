# Diccionario de Datos

## Base de datos: Brain Tumor MRI Dataset

Dataset de imágenes de resonancia magnética (MRI) utilizado para clasificación multiclase de tumores cerebrales.

El conjunto de datos se encuentra organizado en carpetas correspondientes al tipo de diagnóstico y dividido en conjuntos de entrenamiento y prueba.

Estructura general:

```text
docs/data/datos/

├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

# Descripción general del dataset

| Característica | Valor |
|---|---|
| Tipo de problema | Clasificación multiclase |
| Dominio | Visión por computador / imágenes médicas |
| Tipo de imágenes | MRI cerebrales |
| Número total de imágenes | 7200 |
| Número de clases | 4 |
| Conjunto de entrenamiento | 5600 imágenes |
| Conjunto de prueba | 1600 imágenes |

---

# Diccionario de variables

| Variable | Descripción | Tipo de dato | Rango / Valores posibles | Fuente de datos |
|---|---|---|---|---|
| split | Subconjunto al que pertenece la imagen | Categórica | Training, Testing | Organización del dataset |
| clase | Diagnóstico asociado a la imagen MRI | Categórica | glioma, meningioma, pituitary, notumor | Dataset MRI |
| path | Ruta de almacenamiento de la imagen | Texto / Path | Ruta local del archivo | Sistema de archivos |
| imagen | Imagen MRI cerebral | Imagen digital | JPG, PNG, JPEG, BMP, TIF | Dataset MRI |
| height | Altura de la imagen en píxeles | Entero | 167 - 1446 | Calculado mediante OpenCV |
| width | Ancho de la imagen en píxeles | Entero | 150 - 1375 | Calculado mediante OpenCV |
| mean_intensity | Intensidad promedio de la imagen en escala de grises | Float | 0 - 255 | Calculado mediante NumPy/OpenCV |
| hog_features | Vector de características HOG extraídas de la imagen | Vector numérico | 8100 features | Scikit-image |
| corrupted | Indicador de integridad de lectura de imagen | Booleano | True / False | OpenCV |

---

# Descripción detallada de variables

## split

Indica si la imagen pertenece al conjunto de entrenamiento o prueba.

### Valores posibles

| Valor | Descripción |
|---|---|
| Training | Imágenes utilizadas para entrenamiento |
| Testing | Imágenes utilizadas para evaluación |

---

## clase

Corresponde al diagnóstico asociado a la imagen MRI.

### Valores posibles

| Clase | Descripción |
|---|---|
| glioma | Tumor cerebral tipo glioma |
| meningioma | Tumor cerebral tipo meningioma |
| pituitary | Tumor pituitario |
| notumor | Imagen sin presencia de tumor |

---

## path

Ruta absoluta o relativa donde se encuentra almacenada la imagen.

### Ejemplo

```text
docs/data/datos/Training/glioma/image_001.jpg
```

---

## imagen

Archivo de imagen MRI cerebral utilizado para clasificación.

### Formatos soportados

- .jpg
- .jpeg
- .png
- .bmp
- .tif

---

## height

Número de píxeles correspondientes a la altura de la imagen.

### Estadísticas observadas

| Métrica | Valor |
|---|---|
| Promedio | 456.90 px |
| Mínimo | 167 px |
| Máximo | 1446 px |

---

## width

Número de píxeles correspondientes al ancho de la imagen.

### Estadísticas observadas

| Métrica | Valor |
|---|---|
| Promedio | 453.30 px |
| Mínimo | 150 px |
| Máximo | 1375 px |

---

## mean_intensity

Promedio de intensidad de píxeles calculado en escala de grises.

### Estadística global

| Métrica | Valor |
|---|---|
| Intensidad promedio global | 47.56 |

### Interpretación

Valores bajos indican predominancia de regiones oscuras típicas de imágenes MRI.

---

## hog_features

Vector numérico generado mediante el descriptor HOG (Histogram of Oriented Gradients).

### Configuración utilizada

```python
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = "L2-Hys"
```

### Resultado

| Métrica | Valor |
|---|---|
| Número de features | 8100 |

### Uso

Las features HOG permiten representar:

- bordes,
- gradientes,
- texturas,
- estructuras anatómicas.

---

## corrupted

Variable utilizada para identificar imágenes dañadas o ilegibles.

### Valores posibles

| Valor | Significado |
|---|---|
| False | Imagen válida |
| True | Imagen corrupta |

### Resultado encontrado

```text
Imágenes corruptas: 0
```

---

# Balance de clases

| Clase | Cantidad |
|---|---|
| glioma | 1800 |
| meningioma | 1800 |
| pituitary | 1800 |
| notumor | 1800 |

---

# Distribución por split

| Split | Cantidad |
|---|---|
| Training | 5600 |
| Testing | 1600 |

---

# Observaciones generales

- El dataset se encuentra completamente balanceado.
- No existen imágenes corruptas.
- Las imágenes presentan alta variabilidad en resolución.
- El dataset es adecuado para tareas de clasificación multiclase utilizando Deep Learning.
- Las imágenes poseen características estructurales relevantes para modelos CNN.

---

# Uso esperado del dataset

Este dataset está diseñado para:

- clasificación de tumores cerebrales,
- entrenamiento de modelos CNN,
- transferencia de aprendizaje,
- análisis médico asistido por IA,
- investigación en visión por computador aplicada a salud.

---

# Estado del dataset

✅ Dataset cargado correctamente  
✅ Estructura validada  
✅ Imágenes verificadas  
✅ Clases balanceadas  
✅ Listo para preprocessing y modelado