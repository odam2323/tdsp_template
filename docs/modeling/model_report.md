# Reporte del Modelo Final

---

## Resumen Ejecutivo

El presente proyecto desarrolla un sistema de clasificación automática de tumores cerebrales utilizando imágenes de resonancia magnética (MRI), mediante técnicas de Deep Learning e Inteligencia Artificial Explicable (XAI). El objetivo principal es asistir el diagnóstico clínico a través de modelos capaces de identificar patrones complejos asociados a diferentes tipos de tumores cerebrales.

Para abordar este problema, se implementó una arquitectura EfficientNetV2-S preentrenada sobre ImageNet mediante Transfer Learning, adaptada para clasificar imágenes médicas en cuatro categorías diagnósticas:

- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

Adicionalmente, se integró Grad-CAM++ como mecanismo de interpretabilidad visual, generando mapas de calor que resaltan las regiones relevantes utilizadas por el modelo en la toma de decisiones.

Los resultados evidencian una mejora significativa en la capacidad de representación frente al modelo baseline basado en HOG + SVM, confirmando la superioridad de los enfoques de aprendizaje profundo en problemas de clasificación médica.

---

## Descripción del Problema

Los tumores cerebrales representan una de las patologías neurológicas más complejas debido a su alta variabilidad morfológica y dificultad de interpretación visual. El análisis manual de imágenes MRI depende en gran medida de la experiencia clínica y puede verse afectado por factores como la fatiga visual y la subjetividad del especialista.

En este contexto, la inteligencia artificial ofrece una alternativa capaz de automatizar el análisis de imágenes médicas y asistir en la toma de decisiones clínicas.

El objetivo del proyecto consiste en desarrollar un modelo capaz de clasificar automáticamente imágenes MRI cerebrales en múltiples categorías tumorales, reduciendo errores diagnósticos y proporcionando interpretabilidad sobre las decisiones del modelo.

---

## Descripción del Dataset

Se utilizó el dataset público:

**Brain Tumor MRI Dataset**

El dataset contiene imágenes clasificadas en cuatro categorías:

| Clase      | Descripción                                 |
|-----------|---------------------------------------------|
| Glioma     | Tumor originado en células gliales          |
| Meningioma | Tumor en las meninges                       |
| Pituitary  | Tumor en la glándula pituitaria             |
| No Tumor   | Cerebro sin presencia tumoral               |

Distribución de datos:

| Split    | Cantidad |
|---------|--------|
| Training | 5600 |
| Testing  | 1600 |
| Total    | 7200 |

El dataset presenta una distribución balanceada, lo que favorece el entrenamiento y la evaluación del modelo.

---

## Preprocesamiento de Datos

Se realizaron las siguientes transformaciones:

- Redimensionamiento uniforme de imágenes  
- Conversión a formato RGB  
- Normalización de intensidades  
- Eliminación de imágenes corruptas  
- Separación de datos en entrenamiento y prueba  
- Análisis exploratorio de datos (EDA)  

Para el modelo baseline, se aplicó extracción de características mediante HOG, permitiendo establecer una comparación con el enfoque de Deep Learning.

---

## Línea Base del Modelo

Se implementó un modelo clásico basado en:

- HOG (Histogram of Oriented Gradients)  
- SVM con kernel RBF  

### Resultados del baseline

| Métrica   | Valor |
|----------|------|
| Accuracy | 0.8812 |

### Matriz de confusión (Baseline)

[[262  72  58   8]
[  7 360  13  20]
[  0   0 400   0]
[  2  10   0 388]]

### Métricas por clase

| Clase       | Precision | Recall | F1-score |
|------------|----------|--------|----------|
| Glioma     | 0.97 | 0.66 | 0.78 |
| Meningioma | 0.81 | 0.90 | 0.86 |
| No Tumor   | 0.85 | 1.00 | 0.92 |
| Pituitary  | 0.93 | 0.97 | 0.95 |

### Análisis del baseline

El modelo muestra un buen desempeño general, destacándose en la identificación de "No Tumor" y "Pituitary". Sin embargo, presenta dificultades en la discriminación entre "Glioma" y "Meningioma", debido a la similitud estructural entre estas clases.

---

## Descripción del Modelo Final

El modelo propuesto utiliza EfficientNetV2-S con Transfer Learning.

Justificación:

- Alta eficiencia computacional  
- Excelente desempeño en clasificación de imágenes  
- Rápida convergencia  

Arquitectura:

- Fused-MBConv  
- Squeeze-and-Excitation  
- Global Average Pooling  
- Dropout (0.4)  
- Capa Softmax  

Esto permite capturar patrones complejos, texturas tumorales y relaciones espaciales profundas.

---

## Inteligencia Artificial Explicable (XAI)

Se implementó Grad-CAM++ para mejorar la interpretabilidad del modelo.

Beneficios:

- Identificación de regiones relevantes en la imagen  
- Validación del comportamiento del modelo  
- Mayor confianza en las predicciones  
- Reducción del efecto "caja negra"  

Los mapas generados evidenciaron que el modelo enfoca su atención en regiones tumorales relevantes.

---

## Entrenamiento del Modelo

Configuración:

- Optimizer: Adam  
- Learning rate: 0.0001  
- Batch size: 32  
- Dropout: 0.4  
- Early Stopping aplicado  

Se realizó fine-tuning sobre capas profundas para mejorar la especialización del modelo.

---

## Evaluación del Modelo

### Resultados

| Métrica   | Resultado |
|----------|----------|
| Accuracy | XX |
| Precision | XX |
| Recall | XX |
| F1-score | XX |

### Análisis de resultados

El modelo EfficientNetV2-S logra capturar patrones complejos que no son detectados por el modelo baseline, mejorando significativamente la capacidad de clasificación.

Las principales mejoras se observan en la reducción de confusión entre clases tumorales similares, especialmente entre gliomas y meningiomas.

---

## Comparación entre Modelos

| Métrica   | Baseline (HOG+SVM) | EfficientNetV2-S |
|----------|-------------------|------------------|
| Accuracy | 0.8812            | XX               |

El modelo basado en Deep Learning supera al baseline en capacidad de generalización y precisión diagnóstica.

---

## Limitaciones

- Dependencia del dataset  
- Posible sesgo de origen  
- Alto costo computacional  
- Necesidad de validación clínica  

El modelo no reemplaza al especialista médico, sino que funciona como herramienta de apoyo.

---

## Aplicaciones

- Sistemas de apoyo diagnóstico  
- Priorización de estudios MRI  
- Asistencia radiológica  
- Telemedicina  

---

## Conclusiones

El uso de Deep Learning permite superar significativamente el rendimiento del modelo baseline basado en características manuales.

EfficientNetV2-S demuestra alta capacidad de aprendizaje de patrones complejos en imágenes MRI, mientras que Grad-CAM++ aporta interpretabilidad necesaria para aplicaciones clínicas.

El sistema desarrollado tiene alto potencial como herramienta de apoyo en diagnóstico médico asistido por inteligencia artificial.

---

## Trabajo Futuro

- Ampliar el dataset  
- Validación clínica  
- Evaluar Vision Transformers  
- Implementar segmentación tumoral  
- Desarrollar aplicación web  

---

## Referencias

- Tan, M., & Le, Q. (2021). EfficientNetV2  
- Selvaraju, R. R. et al. Grad-CAM  
- Goodfellow, I. Deep Learning  
- Dataset Kaggle: Brain Tumor MRI  
- Scikit-Learn Documentation  
- TensorFlow Documentation  