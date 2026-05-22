# Reporte de Datos - Brain Tumor MRI Dataset

Este documento contiene los resultados del análisis exploratorio de datos realizado sobre el dataset de resonancias magnéticas cerebrales (MRI) para clasificación multiclase de tumores cerebrales.

---

# Resumen general de los datos

El dataset corresponde a imágenes MRI organizadas en dos subconjuntos principales:

- Training
- Testing

Las imágenes se encuentran clasificadas en cuatro categorías diagnósticas:

- glioma
- meningioma
- notumor
- pituitary

El dataset fue cargado desde la siguiente estructura de directorios:

```python
docs/data/datos/
│
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

# Cantidad total de imágenes

| Split | Cantidad |
|---|---|
| Training | 5600 |
| Testing | 1600 |
| Total | 7200 |

---

# Distribución por split

```text
Training    5600
Testing     1600
```

---

# Distribución por clase

| Clase | Cantidad |
|---|---|
| glioma | 1800 |
| meningioma | 1800 |
| notumor | 1800 |
| pituitary | 1800 |

---

# Observaciones generales

El dataset presenta un balance perfecto entre clases, lo que representa una ventaja importante para problemas de clasificación supervisada, ya que evita sesgos derivados de desbalance de datos.

Adicionalmente:

- Todas las clases poseen la misma cantidad de imágenes.
- El conjunto de entrenamiento contiene aproximadamente el 78% de los datos.
- El conjunto de testing contiene aproximadamente el 22% de los datos.

---

# Resumen de calidad de los datos

## Verificación de imágenes corruptas

Se realizó una validación utilizando OpenCV para asegurar que todas las imágenes pudieran ser leídas correctamente.

Código utilizado:

```python
img = cv2.imread(str(path))

if img is None:
    imagenes_corruptas.append(path)
```

---

## Resultado

```text
Imágenes corruptas: 0
```

---

# Conclusiones de calidad

- No existen imágenes dañadas.
- Todas las imágenes pueden cargarse correctamente.
- El dataset posee buena integridad estructural.
- No se identificaron problemas críticos de lectura.

---

# Valores faltantes

Debido a que el dataset corresponde a imágenes organizadas en carpetas, no existen valores nulos tradicionales como en datasets tabulares.

---

# Duplicados

No se identificaron duplicados durante esta etapa exploratoria.

---

# Variable objetivo

La variable objetivo corresponde al tipo de diagnóstico asociado a cada imagen MRI.

## Clases objetivo

| Clase | Descripción |
|---|---|
| glioma | Tumor tipo glioma |
| meningioma | Tumor tipo meningioma |
| pituitary | Tumor pituitario |
| notumor | Imagen sin presencia de tumor |

---

# Distribución de la variable objetivo

La variable objetivo presenta una distribución perfectamente balanceada:

| Clase | Porcentaje |
|---|---|
| glioma | 25% |
| meningioma | 25% |
| pituitary | 25% |
| notumor | 25% |

---

# Interpretación

Esta distribución balanceada es ideal para:

- Entrenamiento de modelos CNN.
- Evaluación justa de métricas.
- Evitar sobreajuste hacia una clase dominante.
- Reducir sesgos de clasificación.

---

# Variables individuales

## Resoluciones de imágenes

Se realizó un análisis de las dimensiones espaciales originales de todas las imágenes.

---

# Estadísticas de altura

| Métrica | Valor |
|---|---|
| Promedio | 456.90 px |
| Mínima | 167 px |
| Máxima | 1446 px |

---

# Estadísticas de ancho

| Métrica | Valor |
|---|---|
| Promedio | 453.30 px |
| Mínimo | 150 px |
| Máximo | 1375 px |

---

# Observaciones

Las imágenes poseen alta variabilidad en resolución espacial.

Esto implica que será necesario realizar:

- resizing,
- padding,
- normalización espacial,

antes del entrenamiento del modelo.

---

# Tamaño definido para procesamiento

Durante el análisis HOG se utilizó:

```python
IMG_SIZE = (128, 128)
```

Este tamaño puede utilizarse como referencia inicial para modelos CNN ligeros.

---

# Histograma de resoluciones

Se generaron histogramas para:

- alturas,
- anchos,

con el fin de visualizar la dispersión de tamaños presentes en el dataset.

---

# Interpretación de resoluciones

Los histogramas muestran:

- alta heterogeneidad de tamaños,
- presencia de imágenes pequeñas,
- presencia de estudios MRI de alta resolución,
- necesidad de estandarización previa.

---

# Visualización de muestras

Se visualizaron múltiples imágenes por clase para analizar:

- variabilidad visual,
- patrones anatómicos,
- diferencias entre clases,
- calidad de las imágenes.

---

# Observaciones visuales

## glioma

- Presencia de regiones tumorales irregulares.
- Alteraciones de intensidad localizadas.
- Diferencias estructurales pronunciadas.

## meningioma

- Lesiones más delimitadas.
- Bordes relativamente definidos.
- Cambios estructurales moderados.

## pituitary

- Tumores pequeños localizados cerca de la región pituitaria.
- Menor tamaño relativo del tumor.

## notumor

- Estructuras cerebrales homogéneas.
- Ausencia de masas tumorales visibles.

---

# Análisis de intensidad

Se calculó la intensidad promedio de cada imagen utilizando escala de grises.

Código utilizado:

```python
img = cv2.imread(
    str(path),
    cv2.IMREAD_GRAYSCALE
)

mean_intensity.append(np.mean(img))
```

---

# Resultado global

| Métrica | Valor |
|---|---|
| Intensidad promedio global | 47.56 |

---

# Interpretación

Las imágenes presentan predominancia de tonos oscuros, característica típica de resonancias magnéticas cerebrales.

El análisis de intensidad permite detectar:

- diferencias de contraste,
- regiones negras amplias,
- variabilidad de iluminación,
- diferencias entre adquisiciones MRI.

---

# Histogramas de intensidad

Se generaron histogramas para visualizar la distribución de intensidad promedio de las imágenes.

Estos histogramas permiten identificar:

- posibles outliers,
- diferencias de exposición,
- variabilidad entre estudios médicos.

---

# Análisis HOG (Histogram of Oriented Gradients)

Se realizó extracción de características HOG sobre una imagen de ejemplo.

---

# Configuración utilizada

```python
features, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)
```

---

# Parámetros utilizados

| Parámetro | Valor |
|---|---|
| orientations | 9 |
| pixels_per_cell | (8, 8) |
| cells_per_block | (2, 2) |
| block_norm | L2-Hys |

---

# Resultado HOG

| Métrica | Valor |
|---|---|
| Número de features HOG | 8100 |

---

# Interpretación HOG

Las características HOG permiten representar:

- bordes,
- gradientes,
- texturas,
- estructuras anatómicas,
- regiones tumorales.

Estas características son útiles para:

- modelos clásicos de Machine Learning,
- análisis estructural,
- clasificación basada en texturas.

---

# Visualización HOG

Se generó una visualización comparativa entre:

- imagen original,
- representación HOG.

La visualización HOG resalta:

- bordes anatómicos,
- cambios de intensidad,
- regiones estructurales importantes.

---

# Histograma de píxeles

Se realizó un análisis de distribución global de píxeles del dataset.

---

# Observaciones del histograma

El histograma muestra:

- predominancia de intensidades bajas,
- gran cantidad de píxeles oscuros,
- comportamiento típico de imágenes MRI.

---

# Ranking de variables

Debido a que el dataset corresponde a imágenes, no existen variables explícitas como en datasets tabulares.

Sin embargo, las principales fuentes de información relevantes para clasificación incluyen:

- intensidad de píxeles,
- patrones espaciales,
- gradientes,
- bordes,
- texturas,
- regiones anatómicas,
- estructuras tumorales.

---

# Técnicas futuras para ranking de importancia

En etapas posteriores podrían utilizarse:

- PCA,
- CNN feature extraction,
- Grad-CAM,
- SHAP,
- mapas de activación,
- embeddings visuales,
- attention maps.

---

# Relación entre variables explicativas y variable objetivo

En visión computacional, la relación entre variables explicativas y la variable objetivo se encuentra principalmente en:

- patrones espaciales,
- relaciones entre píxeles,
- distribución de texturas,
- bordes anatómicos,
- regiones de contraste.

---

# Posibles análisis futuros

## Visualización avanzada

- t-SNE
- UMAP
- PCA visual

## Interpretabilidad

- Grad-CAM
- SHAP
- mapas de activación CNN

## Evaluación de modelos

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de confusión

---

# Pipeline sugerido de preprocessing

## Procesamiento espacial

```python
Resize -> Normalización -> Augmentation
```

---

# Transformaciones sugeridas

## Normalización

```python
img = img / 255.0
```

## Data augmentation

- Rotaciones
- Flip horizontal
- Zoom
- Variaciones de brillo
- Shift espacial

---

# Arquitecturas sugeridas

## CNN clásicas

- CNN personalizada
- LeNet
- AlexNet

## Transfer Learning

- ResNet
- EfficientNet
- DenseNet
- MobileNet
- ConvNeXt

## Transformers

- Vision Transformer (ViT)
- Swin Transformer

---

# Conclusiones generales

## Hallazgos principales

- Dataset completamente balanceado.
- No existen imágenes corruptas.
- Alta calidad estructural.
- Alta variabilidad de resoluciones.
- Intensidades predominantemente bajas.
- Dataset adecuado para clasificación multiclase.
- Las características HOG muestran riqueza estructural significativa.

---

# Fortalezas del dataset

- Balance perfecto entre clases.
- Buena cantidad de imágenes.
- Diversidad anatómica.
- Buen tamaño para deep learning.

---

# Retos identificados

- Variabilidad de resoluciones.
- Diferencias de intensidad.
- Variabilidad entre adquisiciones MRI.
- Posibles diferencias entre pacientes.

---

# Estado final del dataset

✅ Dataset cargado correctamente  
✅ Dataset balanceado  
✅ Sin imágenes corruptas  
✅ Resoluciones analizadas  
✅ Intensidades analizadas  
✅ Features HOG extraídas  
✅ Visualizaciones generadas  
✅ Listo para preprocessing  
✅ Listo para entrenamiento de modelos CNN y Deep Learning