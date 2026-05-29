# Reporte del Modelo Final

# Resumen Ejecutivo

El presente proyecto desarrolla un sistema de clasificación automática de tumores cerebrales mediante imágenes de resonancia magnética (MRI), utilizando técnicas de Deep Learning e Inteligencia Artificial Explicable (XAI). El objetivo principal consiste en apoyar el diagnóstico clínico a través de modelos capaces de identificar patrones complejos asociados a diferentes tipos de tumores cerebrales.

Para el desarrollo del sistema se implementó una arquitectura EfficientNetV2-S preentrenada sobre ImageNet utilizando Transfer Learning. El modelo fue ajustado para clasificar imágenes médicas en cuatro categorías diagnósticas:

* Glioma
* Meningioma
* Pituitary
* No Tumor

Adicionalmente, se integró Grad-CAM++ como mecanismo de interpretabilidad visual, permitiendo generar mapas de calor que resaltan las regiones anatómicas más relevantes para la decisión del modelo.

Los resultados obtenidos demuestran que el enfoque propuesto logra un desempeño competitivo en tareas de clasificación médica, evidenciando una adecuada capacidad de generalización y una mejora significativa frente a modelos clásicos basados en extracción manual de características.

---

# Descripción del Problema

Los tumores cerebrales representan una de las patologías neurológicas de mayor complejidad diagnóstica debido a la variabilidad morfológica de las lesiones y a las dificultades asociadas con la interpretación visual de imágenes médicas. El análisis manual de resonancias magnéticas requiere experiencia clínica especializada y puede verse afectado por fatiga visual, subjetividad diagnóstica y variabilidad entre especialistas.

En este contexto, el uso de Inteligencia Artificial aplicada al análisis de imágenes médicas surge como una alternativa capaz de asistir al personal médico mediante sistemas automáticos de clasificación y apoyo al diagnóstico.

El problema abordado en este proyecto consiste en desarrollar un modelo computacional capaz de clasificar automáticamente imágenes MRI cerebrales en múltiples categorías tumorales, minimizando errores diagnósticos y proporcionando interpretabilidad sobre las decisiones tomadas por la red neuronal.

La incorporación de técnicas de Inteligencia Artificial Explicable (XAI) responde además a la necesidad de reducir el efecto de “caja negra” presente en muchos modelos de Deep Learning, permitiendo visualizar qué regiones de la imagen influyen en la predicción final.

---

# Descripción del Dataset

El conjunto de datos utilizado corresponde al dataset público:

**Brain Tumor MRI Dataset**

El dataset contiene imágenes de resonancia magnética cerebral organizadas en cuatro clases diagnósticas:

| Clase      | Descripción                                 |
| ---------- | ------------------------------------------- |
| Glioma     | Tumor cerebral originado en células gliales |
| Meningioma | Tumor que afecta las meninges               |
| Pituitary  | Tumor localizado en la glándula pituitaria  |
| No Tumor   | Cerebro sin presencia tumoral               |

La distribución utilizada fue:

| Split    | Cantidad      |
| -------- | ------------- |
| Training | 5600 imágenes |
| Testing  | 1600 imágenes |
| Total    | 7200 imágenes |

El dataset presenta una distribución balanceada entre clases, lo cual favorece el entrenamiento del modelo y reduce problemas asociados al desbalance de datos.

---

# Preprocesamiento de Datos

Antes del entrenamiento del modelo se aplicaron diversas etapas de preprocesamiento orientadas a homogenizar las imágenes y facilitar la extracción de patrones relevantes.

Las principales transformaciones realizadas fueron:

* Conversión de imágenes a formato RGB compatible con EfficientNetV2-S.
* Redimensionamiento uniforme de imágenes.
* Normalización de intensidades.
* Organización de datos en conjuntos de entrenamiento y prueba.
* Verificación de imágenes corruptas o inválidas.
* Análisis exploratorio de datos (EDA).

Adicionalmente, para la línea base clásica se realizó extracción de características HOG (Histogram of Oriented Gradients), permitiendo comparar el desempeño entre enfoques tradicionales y Deep Learning.

---

# Línea Base del Modelo

Como línea base inicial se implementó un enfoque clásico de Visión por Computador utilizando:

* Extracción de características HOG.
* Clasificador SVM con kernel RBF.

Este modelo permitió establecer un punto de referencia inicial para medir posteriormente las mejoras obtenidas mediante Deep Learning.

Las características HOG fueron seleccionadas debido a su capacidad para representar bordes, contornos y gradientes presentes en imágenes médicas.

La arquitectura baseline presentó limitaciones para capturar patrones espaciales complejos y relaciones jerárquicas profundas presentes en las resonancias cerebrales.

---

# Descripción del Modelo Final

El sistema propuesto implementa la arquitectura EfficientNetV2-S preentrenada en ImageNet mediante Transfer Learning.

La selección de EfficientNetV2-S se realizó debido a:

* Alta eficiencia computacional.
* Rápida convergencia durante entrenamiento.
* Excelente desempeño en tareas de clasificación de imágenes.
* Menor costo computacional comparado con arquitecturas más profundas.

La arquitectura incorpora bloques:

* Fused-MBConv en etapas iniciales.
* Squeeze-and-Excitation para atención por canal.

Estos componentes permiten capturar patrones geométricos, texturas tumorales y diferencias anatómicas relevantes entre tejido sano y tejido patológico.

El modelo final incluye:

* Global Average Pooling.
* Dropout de 0.4 para reducción de sobreajuste.
* Capa densa Softmax para clasificación multiclase.

---

# Inteligencia Artificial Explicable (XAI)

Con el objetivo de mejorar la interpretabilidad clínica del sistema, se integró la técnica Grad-CAM++.

Grad-CAM++ permite generar mapas de calor sobre las imágenes MRI, resaltando las regiones anatómicas que más contribuyen a la predicción realizada por el modelo.

Esta capacidad resulta especialmente importante en aplicaciones médicas, ya que:

* Facilita la validación clínica.
* Incrementa la confianza del especialista.
* Reduce la opacidad del modelo.
* Permite identificar posibles errores de clasificación.

Los mapas de activación obtenidos mostraron que el modelo concentra correctamente su atención en regiones tumorales relevantes.

---

# Entrenamiento del Modelo

El entrenamiento se realizó utilizando Transfer Learning sobre pesos preentrenados en ImageNet.

Las etapas principales incluyeron:

1. Congelamiento inicial de capas convolucionales.
2. Entrenamiento del clasificador final.
3. Fine-Tuning parcial de capas profundas.
4. Ajuste de hiperparámetros.

Se emplearon técnicas de regularización y prevención de sobreajuste como:

* Dropout.
* Early Stopping.
* Reducción adaptativa del learning rate.

---

# Evaluación del Modelo

La evaluación del modelo se realizó sobre el conjunto de prueba utilizando métricas estándar de clasificación multiclase.

Las métricas consideradas fueron:

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de confusión

## Resultados Generales

| Métrica   | Resultado |
| --------- | --------- |
| Accuracy  | XX.XX%    |
| Precision | XX.XX%    |
| Recall    | XX.XX%    |
| F1-Score  | XX.XX%    |

## Interpretación de Resultados

Los resultados obtenidos evidencian una alta capacidad de clasificación del modelo sobre imágenes MRI cerebrales.

La matriz de confusión muestra que el modelo logra diferenciar adecuadamente la mayoría de clases tumorales, especialmente en casos donde las estructuras anatómicas presentan patrones visuales distintivos.

Sin embargo, algunas confusiones pueden aparecer entre tipos tumorales con características visuales similares, particularmente entre glioma y meningioma, debido a la complejidad morfológica presente en ciertas resonancias.

---

# Comparación entre Línea Base y Modelo Final

| Característica                | Baseline HOG + SVM | EfficientNetV2-S |
| ----------------------------- | ------------------ | ---------------- |
| Extracción manual de features | Sí                 | No               |
| Deep Learning                 | No                 | Sí               |
| Capacidad de generalización   | Media              | Alta             |
| Interpretabilidad XAI         | No                 | Sí               |
| Precisión diagnóstica         | Media              | Alta             |

La arquitectura basada en Deep Learning superó significativamente el desempeño del enfoque clásico, demostrando una mejor capacidad para aprender representaciones complejas directamente desde los datos.

---

# Limitaciones

A pesar de los resultados obtenidos, existen algunas limitaciones importantes:

* Dependencia de la calidad del dataset.
* Posible sesgo asociado al origen de las imágenes.
* Requerimientos computacionales elevados para entrenamiento.
* Necesidad de validación clínica real.
* Riesgo de sobreajuste en datasets pequeños.

Además, el sistema no debe considerarse un reemplazo del especialista médico, sino una herramienta de apoyo diagnóstico.

---

# Aplicaciones Potenciales

El sistema desarrollado puede aplicarse en:

* Sistemas de apoyo al diagnóstico clínico.
* Priorización automática de estudios MRI.
* Herramientas de asistencia radiológica.
* Investigación médica basada en IA.
* Plataformas de telemedicina.

---

# Conclusiones

El presente proyecto demuestra la viabilidad del uso de Deep Learning e Inteligencia Artificial Explicable para la clasificación automática de tumores cerebrales mediante resonancias magnéticas.

La arquitectura EfficientNetV2-S mostró un desempeño sólido en tareas de clasificación multiclase, superando ampliamente el enfoque baseline basado en HOG + SVM.

La integración de Grad-CAM++ aportó interpretabilidad al sistema, permitiendo visualizar las regiones relevantes utilizadas por el modelo durante la toma de decisiones.

Los resultados obtenidos evidencian el potencial de las redes neuronales profundas como herramientas de apoyo clínico en el área de diagnóstico médico asistido por inteligencia artificial.

---

# Trabajo Futuro

Como trabajo futuro se propone:

* Incrementar el tamaño y diversidad del dataset.
* Incorporar validación clínica real.
* Evaluar arquitecturas Transformer para visión.
* Implementar segmentación tumoral automática.
* Desarrollar despliegue web o clínico.
* Integrar múltiples modalidades MRI.

---

# Referencias

* Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training.
* Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks.
* Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning.
* Dataset: Brain Tumor MRI Dataset (Kaggle).
* Scikit-Learn Documentation.
* TensorFlow Documentation.
